"""
###############################################################################
Part of the Molecular Dynamics project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)
################################################################################
Defines and sets up basic molecular dynamic simulation
Classes:
- Canvas
- Simulation
Functions:
- evolute
################################################################################
"""

from ctypes.wintypes import POINT
import math
from multiprocessing.pool import INIT
from multiprocessing.sharedctypes import Value
import os
import sys
from time import sleep
from tkinter.tix import MAX
from matplotlib.pyplot import axis
import numpy as np
import h5py as hdf
import scipy as sp
import scipy.constants as sp_const
import scipy.spatial.distance as sp_dist
import scipy.optimize as sp_optim
from itertools import combinations
from tqdm import tqdm

###### Main program:

class Canvas:
    """
    Defines a space in which simulation is taking place
    Parameters
    - dim::int = Number of dimensions of canvas (only 2D or 3D supported)
    - size::ndarray = Size of canvas in each dimension
    """
    def __init__(self, n_dim, size):
        self.type = str(n_dim) + "D-canvas"
        self.n_dim = n_dim
        self.size = size
        print("Created a {}D-canvas".format(self.n_dim))




def lennard_jones(distance, pot_args):
    """
    Calculate and return the value of the lennard jones potential for given 
    inter-particle distance, sigma and epsilon
    Args
    - distance = distance for which to evaluate the potential
    - sigma = Sigma parameter (Particle size) in LS potential
    - epsilon = Epsilon parameter (Temperature) in LS potential
    Return
    - pot_val::float = Value of the lennard jones potential for given sigma and epsilon
    """
    return 4 * ( (1/distance)**12 - (1/distance)**6 )

def upper_triang_mat(upper_entries, mat_size,  symmetric):
    """
    Returns a (anti-)symmetric upper-trinagular matrix for a given list of upper-trinagular entries with zero diagonal
                                / 0  a  b  c \ 
                               |  a  0  d  e  | 
    e.g. [a, b, c, d, e, f] -> |  b  d  0  f  |
                                \ c  e  f  0 /
    
    Args
    - upper_entries::list = list of strictly upper triangular entries
    - mat_size::Int = Target Matrix size n for an (n x x) output matrix.
    Return
    - sut_mat::ndarray = (anit-)Symmetric Upper-Trinagular (size x size) matrix with zero diagonal
    """
    if len(upper_entries) != int((mat_size**2 - mat_size)/2):
        raise ValueError("The provided list should match the number of upper triangular entries for a matrix of the given size")
    
    sut_mat = np.zeros((mat_size, mat_size))
    sut_mat[np.triu_indices(mat_size, k=1)] = upper_entries
    if symmetric: 
        sut_mat[np.tril_indices(mat_size, k=-1)] = sut_mat.T[np.tril_indices(mat_size, k=-1)]
    else:
        sut_mat[np.tril_indices(mat_size, k=-1)] = -sut_mat.T[np.tril_indices(mat_size, k=-1)]
    return sut_mat


def pdiff(coord_array, return_distance=False):
    """
        Pairwise (eucledian) difference
        Note: Similar to pdist in the scipy package.
        
        Args
        - coord_array::ndarray = 2D array (n_coordinates x n_dim)
        - return_distance::bool = Whether to return also the pairwise euclidean distance
        Return
        - 
    """
    if len(coord_array.shape) != 2:
        raise ValueError("The coordinate array should be 2 dimensional but isn't")
    
    pairs = list(combinations(range(coord_array.shape[0]), 2))
    pair_i, pair_j = zip(*pairs)
    differences = coord_array[pair_i, :] - coord_array[pair_j, :]
    if not return_distance:
        return differences
    else:
        distances = np.linalg.norm(differences, axis=1)
        return differences, distances
    

class Simulation:
    """
    Defines a molecular simulation with appropiate conditions
    Parameters
    - n_atoms::int = Number of atoms to be simulated
    - atom_mass::int = 
    - n_dim::int = Number of dimensions to simulate in
    - canvas_size::ndarray = Array of canvas side length in each dimension
    - time::ndarray = Ordered array of all time values to simulate over
    - delta_t::float = Timestep per iteration in simulation; Should equal used timestep in time array
    - pot_args::dict = Dictionary with arguments (e.g. constants) required for calculating the potential 
    - init_mode::string or callable = Specify method to initialize positions and velocities
    """
    def __init__(self, n_atoms, atom_mass, n_dim, canvas_size, pot_args, init_mode, data_path, data_filename):
        self.n_atoms = n_atoms
        self.atom_mass = atom_mass
        self.n_dim = n_dim
        self.canvas_size = canvas_size
        self.pot_args = pot_args
        self.data_path = data_path
        self.data_filename = data_filename
        
        #Check for correct (dimensional) format of size
        if type(canvas_size) != np.ndarray:
            raise ValueError('Provide size of canvas as numpy array.')
    
        elif canvas_size.shape != (n_dim,):
            print(canvas_size.shape)
            raise ValueError('Check number of dimensions of the canvas size array.')

        self.canvas = Canvas(n_dim, canvas_size)
        
        #Initialize and save state of simulation:
        if init_mode == "random":
            self.pos, self.veloc = self.initialize_atoms_random()
        elif callable(init_mode):
            self.pos, self.veloc = init_mode()
        else:
            raise ValueError("Unknown initialization mode; Give ")
        
        print("Succesfully initialized simulation!")

    def initialize_atoms_random(self):
        """
        Initializes position and velocity of N atoms randomly on canvas and returns them
        Args
        - canvas::Canvas = Space in which evolution should take place
        Return
        - pos::ndarray = Array of initial molecule positions
        - veloc::ndarray = Array of initial molecule velocities
        """
        
        self.pos = (np.random.randn(self.n_atoms, self.canvas.n_dim) * self.canvas.size) % np.gcd(*self.canvas.size)
        self.veloc = np.random.randn(self.n_atoms, self.canvas.n_dim) * (self.canvas.size/6) 
        print("Initial positions", self.pos)
        return self.pos, self.veloc

    def evolute(self, delta_t):
        """
        Evolutes the particles position and velocity arrays
        via Newtonian EOM with Lennard-Jones Potential and returns updated
        position and velocity arrays
        
        Args
        - canvas::Canvas = Space in which evolution should take place
        - c_pos::ndarray = Array of positions at current iteration
        - c_veloc::ndarray = Array of velocities at current iteration
        - c_force_array::ndarray = Summed force on each particle (n_atoms x n_dim)
        - atom_mass::float = Mass of each particle/atom (all assumed equal)
        - delta_t::float = Timestep per iteration in simulation
        
        Updates self.pos and self.veloc
        
        """
        
        # Calculate current forces
        c_force_array = self.forces()

        # Calculate and update new positions
        n_pos = self.pos + delta_t * self.veloc + (delta_t**2) * c_force_array
        for d in range(self.canvas.n_dim):
            mask = np.logical_or(n_pos[:, d] > self.canvas.size[d],  n_pos[:, d] < 0)
            (n_pos[:, d])[mask] = np.mod((n_pos[:, d])[mask], self.canvas.size[d])
        self.pos = n_pos
        
        # Calculate new forces
        n_force_array = self.forces()
        
        # Calculate and update new velocities
        n_veloc = self.veloc + delta_t * (n_force_array + c_force_array)
        for d in range(self.canvas.n_dim):
            mask = np.logical_or(n_pos[:, d] > self.canvas.size[d],  n_pos[:, d] < 0)
            (n_veloc[:, d])[mask] = (n_veloc[:, d])[mask]  #No loss of energy with periodic BCs + same direction
        self.veloc = n_veloc

    def distances(self, return_differences=True):
        """
        Calculate and return the reduced array of unique inter-particle distances
        
        Args
        - return_differences::bool = Whether, beside absolute distances, also differences x_i - x_j in each coordinate should be returned.
        Return
        - distances_arr::ndarray = Inter-particle distance between any unique pair of particles
        - differences_per_dim::list(ndarray) = Inter-particle distance (unique) per dimension component
        """
        
        #Calculate all pairwise distances between particles
        min_distance_squared = []
        differences_per_dim = []
        
        for d in range(self.canvas.n_dim):
            #Minimize pairwise per coordinate for closest instance of point.
            #pdist_d = sp_dist.pdist(self.pos[:, [d]], 'euclidean')
            pdiff_d = pdiff(self.pos[:, [d]], return_distance=False).reshape(-1)
            
            
            #Find minimum distance instance of interacting particle for dimension d.
            pdist_d_options = np.vstack((np.absolute(pdiff_d), np.absolute(np.absolute(pdiff_d)-self.canvas.size[d]))).T
            
            pdist_d_min = np.argmin(pdist_d_options, axis=1)
            
            min_distance_squared.append(pdist_d_options[np.arange(pdiff_d.shape[0]), pdist_d_min]**2)
            
            #Correct distance if minimum was outside of simulated canvas
            # -> determine whether +L or -L needs to be added if minimum was outside of simulated canvas
            pdiff_d_signs = np.ones(pdiff_d.shape)
            pdiff_d_signs[pdiff_d < 0] = -1
            
            differences_per_dim.append(pdiff_d - (pdist_d_min * pdiff_d_signs*self.canvas.size[d]))
        
        if return_differences:
            return np.sqrt(np.add.reduce(min_distance_squared)), differences_per_dim
        else:
            return np.sqrt(np.add.reduce(min_distance_squared))
        
    def energies(self):
            """
            Calculate and return the kinetic and potential energy on every particle

            Args
            - c_pos::ndarray = Current position of all particles
            - pot_args::dict = Arguments/constants required to evaluate the potential

            Return
            - kin_energy::ndarray = Kinetic energy of every particle
            - pot_energy::ndarray = Potential energy of every particle
            """
            kin_energy = 1/2*np.sum(self.veloc*self.veloc, axis=1)
            
            if np.any(kin_energy < 0.0):
                print("1")
            #Obtain current interparticle distances
            distance_arr = self.distances(return_differences=False) #minimal (efficient) list of all pairwise distances
            
            #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient) 
            # and sum potential  energy contributions for each particle
            pot_energy = np.sum(upper_triang_mat(list(lennard_jones(distance_arr, self.pot_args)), self.n_atoms, symmetric=True), axis=1)
            
            if np.any(kin_energy < 0.0):
                print("2")
            return kin_energy, pot_energy
        
    def forces(self):
        """
        Calculate and return the force on every particle, due to all other particles
        Args
        - c_pos::ndarray = Current position of all particles
        - pot_args::dict = Arguments/constants required to evaluate the potential
        Return
        - force::ndarray = Summed force on each particle by any other particle (n_atoms x n_dim)  
        """
        distance_list, differences_per_dim = self.distances() #minimal (efficient) list of all pairwise distances
        distance_list = list(distance_list)
        
        #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient)
        distance_mat = upper_triang_mat(distance_list, self.n_atoms,  symmetric=True)

        #Vectorize gradient function (low performance):
        vect_grad_func = np.vectorize(lambda d: sp_optim.approx_fprime(d, lennard_jones, np.sqrt(np.finfo(float).eps), self.pot_args))
        
        #For distances matrix calculate the gradient matrix: element i,j -> dU(r_ij)/dr
        lj_gradient_list = list(vect_grad_func(np.array(distance_list))) 
        lj_gradient_dist = upper_triang_mat(lj_gradient_list, self.n_atoms, symmetric=True)

        #Set diagonal of distances to infinity to prevent division by zero in next step
        distance_mat[np.diag_indices(self.n_atoms)] = np.inf
        
        #Calculate grad_r U(r_ij) = (dU/dr) / r_ij * \vec{ pos_i} = matrix \
        # and sum all gradients per particle at same time with dot (instead of *) operation.
        #To prevent 3D dimensional array multiplication complexity, split array in spatial dims. 
        lj_gradient_dist_divided_by_r = lj_gradient_dist/distance_mat
        
        #Apply the chain rule of the gradient x,y,z to distance r
        lj_gradient_pos = tuple([np.sum(np.multiply(lj_gradient_dist_divided_by_r, upper_triang_mat(list(-differences_per_dim[d]), self.n_atoms,symmetric=False)), axis=1) for d in range(self.n_dim)])
        #Note: minus sign before distance matrix required to match index i in F_i + sum_j grad_ij to axis=0 of gradient matrix with elements ij. (i.e. positive difference below diagonal)
        
        #Express the force (in particle-wise vectorform) and mind the extra sign flip from force formula.
        force_array = -np.column_stack(lj_gradient_pos)
        
        #Note c_pos.shape[0] = particle number, [1] = number of dimensions = shape of force array
        return force_array

    def __simulate__(self, n_iterations, delta_t):
        """
        Executes simulation on initialized simulation
        
        Args
        - n_iterations::int = Number of iterations/timesteps before end of simualtion
        - delta_t::float = Timestep / difference in time between iterations
        
        Return
        - --
        """

        with hdf.File(self.data_path + self.data_filename, "w") as file:
            for i in tqdm(range(1, n_iterations+1)):
                # Evolute 1 step
                self.evolute(delta_t)
                
                # Retrieve current kinetic and potential energies
                kin_energies, pot_energies = self.energies()
                
                
                # Store each iteration in a separate group of datasets
                datagroup = file.create_group(f"iter_{i}")
                datagroup.attrs['canvas_size'] = self.canvas.size
                datagroup.attrs['delta_t'] = delta_t
                datagroup.create_dataset(f"iter_{i}_pos", data=self.pos)
                datagroup.create_dataset(f"iter_{i}_veloc", data=self.veloc)
                datagroup.create_dataset(f"iter_{i}_kin_energy", data=kin_energies)
                datagroup.create_dataset(f"iter_{i}_pot_energy", data=pot_energies)
                

##### Main function tobe called at start

if __name__ == "__main__":
    print("Script running...")
    print(__name__)
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()
    
    global WORKDIR_PATH 
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    global DATA_PATH 
    WORKDIR_PATH = WORKDIR_PATH + "data/" 

    # Hardcoded inputs (Maybe replace with argv arguments)
    N_DIM = sys.argv[0] # Number of dimensions
    N_ATOMS = sys.argv[1] # Number of particles
    TEMPERATURE = sys.argv[2] # Kelvin
    ATOM_MASS = sys.argv[3] # Mass of atoms (kg); Argon = 39.948 u
    POT_ARGS = {'sigma': sys.argv[4], 'epsilon': sys.argv[5]} # sigma, epsilon for Argon in SI units (see slides Lec. 1)
    
    # Dimensionless constants

    MAX_LENGTH = sys.argv[6]
    CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH]) # Canvas size (must be ndarray!)
    END_OF_TIME = sys.argv[7] # Maximum time

    DELTA_T = sys.argv[8] # Timestep
    N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    
    INIT_MODE = sys.argv[9]
    
    DATA_FILENAME = sys.argv[10]
    
    #Main simulation procedure
    sim = Simulation(n_atoms=N_ATOMS, atom_mass=ATOM_MASS, n_dim=N_DIM, canvas_size=CANVAS_SIZE, pot_args=POT_ARGS, init_mode=INIT_MODE, data_path=DATA_PATH, data_filename=DATA_FILENAME)
    sim.__simulate__(n_iterations=N_ITERATIONS, delta_t=DELTA_T)