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



def evolute(canvas, c_pos, c_veloc, force_array, atom_mass, delta_t):
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

    n_pos = c_pos + (c_veloc * delta_t) 
    n_veloc = c_veloc + (1/atom_mass * force_array * delta_t)
    
    #Apply boundary conditions to particles out of the simulated box
    mask = np.logical_or(n_pos > math.gcd(*canvas.size),  n_pos < 0)
    n_pos[mask] = np.mod(n_pos[mask], math.gcd(*canvas.size))
    n_veloc[mask] = n_veloc[mask]  #No loss of energy with periodic BCs + same direction
    
    return n_pos, n_veloc



def initialize_atoms_random(canvas, n_atoms):
    """
    Initializes position and velocity of N atoms randomly on canvas and returns them

    Args
    - canvas::Canvas = Space in which evolution should take place

    Return
    - pos::ndarray = Array of initial molecule positions
    - veloc::ndarray = Array of initial molecule velocities
    """
    pos = np.random.randn(n_atoms, canvas.n_dim) * canvas.size
    veloc = np.random.randn(n_atoms, canvas.n_dim) * (canvas.size/2)
    return pos, veloc

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
    return 4*pot_args['epsilon']*((pot_args['sigma']/distance)**12 - (pot_args['sigma']/distance)**6)


def forces(c_pos, pot_args):
    """
    Calculate and return the force on every particle, due to all other particles

    Args
    - c_pos::ndarray = Current position of all particles
    - pot_args::dict = Arguments/constants required to evaluate the potential

    Return
    - force::ndarray = Summed force on each particle by any other particle (n_atoms x n_dim)  
    """

    #Calculate all pairwise distances between particles
    distance_list = list(sp_dist.pdist(c_pos, 'euclidean')) #small (efficient) list of all pairwise distances
    
    #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient)
    distances = np.zeros((c_pos.shape[0], c_pos.shape[0]))
    distances[np.triu_indices(c_pos.shape[0], k=1)] = distance_list
    distances[np.tril_indices(c_pos.shape[0], k=-1)] = distances.T[np.tril_indices(c_pos.shape[0], k=-1)]
    
    #Vectorize gradient function (low performance):
    vect_grad_func = np.vectorize(lambda d: sp_optim.approx_fprime(d, lennard_jones, np.sqrt(np.finfo(float).eps), pot_args))
    
    #For distances matrix calculate the gradient matrix: element i,j -> dU(r_ij)/dr
    lj_gradient_list = list(vect_grad_func(np.array(distance_list))) 
    lj_gradient_dist = np.zeros((c_pos.shape[0], c_pos.shape[0]))
    lj_gradient_dist[np.triu_indices(c_pos.shape[0], k=1)] = lj_gradient_list
    lj_gradient_dist[np.tril_indices(c_pos.shape[0], k=-1)] = lj_gradient_dist.T[np.tril_indices(c_pos.shape[0], k=-1)]
    
    #Set diagonal of distances to infinity to prevent division by zero in next step
    distances[np.diag_indices(c_pos.shape[0])] = np.inf
    
    #Calculate grad_r U(r_ij) = (dU/dr) / r_ij * \vec{ pos_i} = matrix \
    # and sum all gradients per particle at same time with dot (instead of *) operation.
    #To prevent 3D dimensional array multiplication complexity, split array in spatial dims. 
    lj_gradient_dist_divided_by_r = lj_gradient_dist/distances
    
    lj_gradient_pos = tuple([np.dot(lj_gradient_dist_divided_by_r, c_pos[:, d]) for d in range(c_pos.shape[1])])
    
    #Express the force (in particle-wise vectorform) and mind the extra sign flip from force formula.
    force_array = -np.column_stack(lj_gradient_pos)
    
    #Note c_pos.shape[0] = particle number, [1] = number of dimensions = shape of force array

    return force_array

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
        self.pos, self.veloc = initialize_atoms_random(self.canvas, n_atoms)

        print("Succesfully initialized simulation!")


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

                c_force_array = forces(self.pos, self.pot_args)

                n_pos, n_veloc = evolute(self.canvas, self.pos, self.veloc, c_force_array, self.atom_mass, delta_t)
                
                self.pos, self.veloc = n_pos, n_veloc

                #Store each iteration in a separate group of datasets
                datagroup = file.create_group("iter_{index}".format(index=i))
                datagroup.create_dataset("iter_{index}_pos".format(index=i), data=self.pos)
                datagroup.create_dataset("iter_{index}_veloc".format(index=i), data=self.veloc)
                



##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    #required_args = 0
    #if len(sys.argv) != required_args:
    #    print_usage()

    #Hardcoded inputs (Maybe replace with argv arguments)
    N_ATOMS = 3 #Number of particles
    ATOM_MASS = 6.6335e-26 #Mass of atoms (kg); Argon = 39.948 u
    N_DIM = 2 #Number of dimensions
    MAX_LENGTH = 5 #Canvas side length (m)
    CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH]) #Canvas size (must be ndarray!)
    
    POT_ARGS = {'sigma': 3.405e-10, 'epsilon': sp_const.k*119.8} #sigma, epsilon for Argon in SI units (see slides Lec. 1)

    #Main simulation procedure
    sim = Simulation(n_atoms=N_ATOMS, atom_mass=ATOM_MASS, n_dim=N_DIM, canvas_size=CANVAS_SIZE, pot_args=POT_ARGS)
    
    END_OF_TIME = 10.0 #Maximum time (s)
    DELTA_T = 0.1 #Timestep (s)
    N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    sim.__simulate__(n_iterations=N_ITERATIONS, delta_t = DELTA_T)
