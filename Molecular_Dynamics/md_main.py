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
from multiprocessing.sharedctypes import Value
import os
import sys
from time import sleep
from tkinter.tix import MAX
import numpy as np
import h5py as hdf
import scipy as sp
import scipy.constants as sp_const
import scipy.spatial.distance as sp_dist
import scipy.optimize as sp_optim

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = WORKDIR_PATH + "/data/" 
DATA_FILENAME = "trajectories.hdf5"

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

def sym_upper_triang_mat(upper_entries, mat_size):
    """
    Returns a symmetric upper-trinagular matrix for a given list of upper-trinagular entries with zero diagonal
                                / 0  a  b  c \ 
                               |  a  0  d  e  | 
    e.g. [a, b, c, d, e, f] -> |  b  d  0  f  |
                                \ c  e  f  0 /
    
    Args
    - upper_entries::list = list of strictly upper triangular entries
    - mat_size::Int = Target Matrix size n for an (n x x) output matrix.
    Return
    - sut_mat::ndarray = Symmetric Upper-Trinagular (size x size) matrix with zero diagonal
    """
    if len(upper_entries) != int((mat_size**2 - mat_size)/2):
        raise ValueError("The provided list should match the number of upper triangular entries for a matrix of the given size")
    
    sut_mat = np.zeros((mat_size, mat_size))
    sut_mat[np.triu_indices(mat_size, k=1)] = upper_entries
    sut_mat[np.tril_indices(mat_size, k=-1)] = sut_mat.T[np.tril_indices(mat_size, k=-1)]
    return sut_mat

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
    """
    def __init__(self, n_atoms, atom_mass, n_dim, canvas_size, pot_args):
        self.n_atoms = n_atoms
        self.atom_mass = atom_mass
        self.n_dim = n_dim
        self.canvas_size = canvas_size
        self.pot_args = pot_args

        #Check for correct (dimensional) format of size
        if type(canvas_size) != np.ndarray:
            raise ValueError('Provide size of canvas as numpy array.')
    
        elif canvas_size.shape != (n_dim,):
            raise ValueError('Check number of dimensions of the canvas size array.')

        self.canvas = Canvas(n_dim, canvas_size)
        
        #Initialize and save state of simulation:
        self.pos, self.veloc = self.initialize_atoms_random()

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
        
        self.pos = np.random.randn(self.n_atoms, self.canvas.n_dim) * self.canvas.size
        self.veloc = np.random.randn(self.n_atoms, self.canvas.n_dim) * (self.canvas.size/4) #*0.000001
        return self.pos, self.veloc

    def evolute(self, force_array, delta_t):
        """
        Evolutes the particles position and velocity arrays
        via Newtonian EOM with Lennard-Jones Potential and returns updated
        position and velocity arrays
        
        Args
        - canvas::Canvas = Space in which evolution should take place
        - c_pos::ndarray = Array of positions at current iteration
        - c_veloc::ndarray = Array of velocities at current iteration
        - force_array::ndarray = Summed force on each particle (n_atoms x n_dim)
        - atom_mass::float = Mass of each particle/atom (all assumed equal)
        - delta_t::float = Timestep per iteration in simulation
        Return
        - n_pos::ndarray = Array of updated positions
        - n_veloc::ndarray = Array of updated velocities
        """

        n_pos = self.pos + (self.veloc * delta_t) 
        n_veloc = self.veloc - (self.pot_args['epsilon']/(self.atom_mass*self.pot_args['sigma']) * force_array * delta_t)
        
        #Apply boundary conditions to particles out of the simulated box
        for d in range(self.canvas.n_dim):
            mask = np.logical_or(n_pos[:, d] > self.canvas.size[d],  n_pos[:, d] < 0)
            (n_pos[:, d])[mask] = np.mod((n_pos[:, d])[mask], self.canvas.size[d])
            (n_veloc[:, d])[mask] = (n_veloc[:, d])[mask]  #No loss of energy with periodic BCs + same direction
        
        return n_pos, n_veloc

    def distances(self):
        """
        Calculate and return the reduced array of unique inter-particle distances
        
        Args
        - c_pos::ndarray = Current position of all particles
        - pot_args::dict = Arguments/constants required to evaluate the potential

        Return
        - distances_arr::ndarray = Inter-particle distance between any unique pair of particles
        """
        
        #Calculate all pairwise distances between particles
        min_distance_squared = []
        
        for d in range(self.canvas.n_dim):
            #Minimize pairwise per coordinate for closest instance of point.
            pdist_d = sp_dist.pdist(self.pos[:, [d]], 'euclidean')
            pdist_d_min = np.minimum(np.absolute(pdist_d-self.canvas.size[d]), pdist_d)
            min_distance_squared.append(pdist_d_min**2) 
        
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
            self.kin_energy = 1/2*np.sum(self.veloc*self.veloc, axis=1)
            
            #Obtain current interparticle distances
            distance_arr = self.distances() #minimal (efficient) list of all pairwise distances
            
            #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient) 
            # and sum potential  energy contributions for each particle
            self.pot_energy = np.sum(sym_upper_triang_mat(list(lennard_jones(distance_arr, self.pot_args)), self.n_atoms), axis=1)
            
            return self.kin_energy, self.pot_energy
        
    def forces(self):
        """
        Calculate and return the force on every particle, due to all other particles
        Args
        - c_pos::ndarray = Current position of all particles
        - pot_args::dict = Arguments/constants required to evaluate the potential
        Return
        - force::ndarray = Summed force on each particle by any other particle (n_atoms x n_dim)  
        """
        distance_list = list(self.distances()) #minimal (efficient) list of all pairwise distances
        
        #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient)
        distance_mat = sym_upper_triang_mat(distance_list, self.n_atoms)

        #Vectorize gradient function (low performance):
        vect_grad_func = np.vectorize(lambda d: sp_optim.approx_fprime(d, lennard_jones, np.sqrt(np.finfo(float).eps), self.pot_args))
        
        #For distances matrix calculate the gradient matrix: element i,j -> dU(r_ij)/dr
        lj_gradient_list = list(vect_grad_func(np.array(distance_list))) 
        lj_gradient_dist = sym_upper_triang_mat(lj_gradient_list, self.n_atoms)

        #Set diagonal of distances to infinity to prevent division by zero in next step
        distance_mat[np.diag_indices(self.n_atoms)] = np.inf
        
        #Calculate grad_r U(r_ij) = (dU/dr) / r_ij * \vec{ pos_i} = matrix \
        # and sum all gradients per particle at same time with dot (instead of *) operation.
        #To prevent 3D dimensional array multiplication complexity, split array in spatial dims. 
        lj_gradient_dist_divided_by_r = lj_gradient_dist/distance_mat
        
        lj_gradient_pos = tuple([np.dot(lj_gradient_dist_divided_by_r, self.pos[:, d]) for d in range(self.n_dim)])
        
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

        with hdf.File(DATA_PATH + DATA_FILENAME, "w") as file:
            for i in range(1, n_iterations+1):

                c_force_array = self.forces()
                print(c_force_array)
                n_pos, n_veloc = self.evolute(c_force_array, delta_t)
                
                #Retrieve current kinetic and potential energies
                kin_energies, pot_energies = self.energies()
                
                self.pos, self.veloc = n_pos, n_veloc

                #Store each iteration in a separate group of datasets
                datagroup = file.create_group(f"iter_{i}")
                datagroup.attrs['canvas_size'] = self.canvas.size 
                datagroup.create_dataset(f"iter_{i}_pos", data=self.pos)
                datagroup.create_dataset(f"iter_{i}_veloc", data=self.veloc)
                datagroup.create_dataset(f"iter_{i}_kin_energy", data=kin_energies)
                datagroup.create_dataset(f"iter_{i}_pot_energy", data=pot_energies)
                

##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    #required_args = 0
    #if len(sys.argv) != required_args:
    #    print_usage()

    # Hardcoded inputs (Maybe replace with argv arguments)
    N_DIM = 3 # Number of dimensions
    N_ATOMS = 2 # Number of particles
    TEMPERATURE = 100 # Kelvin
    ATOM_MASS = 6.6335e-26 # Mass of atoms (kg); Argon = 39.948 u
    POT_ARGS = {'sigma': 3.405e-10, 'epsilon': sp_const.k*119.8} # sigma, epsilon for Argon in SI units (see slides Lec. 1)
    
    # Dimensionless constants
    MAX_LENGTH = 5
    #CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH]) # 2D canvas size (must be ndarray!)
    CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH, MAX_LENGTH]) # 3D canvas size (must be ndarray!)
    END_OF_TIME = 0.5 # Maximum time
    DELTA_T = 0.01 # Timestep
    N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    
    #Main simulation procedure
    sim = Simulation(n_atoms=N_ATOMS, atom_mass=ATOM_MASS, n_dim=N_DIM, canvas_size=CANVAS_SIZE, pot_args=POT_ARGS)
    sim.__simulate__(n_iterations=N_ITERATIONS, delta_t=DELTA_T)