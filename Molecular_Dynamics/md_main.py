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
import os
import sys
from tkinter.tix import MAX
import numpy as np
import h5py as hdf
import scipy as sp

WORKDIR_PATH = os.getcwd()
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

    Return
    - n_pos::ndarray = Array of updated positions
    - n_veloc::ndarray = Array of updated velocities
    """

    n_pos = c_pos + (delta_t * c_veloc)
    n_veloc = c_veloc + (1/atom_mass * force_array)

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
    pos = np.random.randn(canvas.n_dim, n_atoms) * canvas.size
    veloc = np.random.randn(canvas.n_dim, n_atoms) * (canvas.size/2)
    return pos, veloc

def lennard_jones(distance, sigma, epsilon):
    """
    Calculate and return the value of the lennard jones potential for given 
    inter-particle distance, sigma and epsilon

    Args
    - sigma = Sigma parameter (Particle size) in LS potential
    - epsilon = Epsilon parameter (Temperature) in LS potential

    Return
    - pot_val::float = Value of the lennard jones potential for given sigma and epsilon
    """
    return 4*epsilon*((sigma/distance)**12 - (sigma/distance)**6)


def forces(c_pos, pot_args):
    """
    Calculate and return the force on every particle, due to all other particles

    Args
    - c_pos = Current position of all particles
    - pot_args = Arguments/constants required to evaluate the potential

    Return
    - force::ndarray = Force on each particle by any other particle
    """
    #Vectorize the scalar lennard jones function
    lj_vect_func = np.vectorize(lennard_jones, excluded=['sigma', 'epsilon'])

    #Calculate all pairwise distances between particles
    distance_list = list(sp.spatial.distance.pdist(c_pos, 'euclidean')) #small (efficient) list of all pairwise distances
    
    #Get distances into symmetric array form with element i,j -> distance r_ij (less efficient)
    distances = np.zeros((c_pos.shape[0], c_pos.shape[0]))
    distances[np.triu_indices(c_pos.shape[0])] = distance_list
    distances[np.tril_indices(c_pos.shape[0])] = distances.T[np.tril_indices(c_pos.shape[0])]
    
    #For distances matrix calculate the gradient matrix: element i,j -> dU(r_ij)/dr
    lj_gradient_list = list(sp.optimize.approx_fprime(distances, lj_vect_func, *pot_args))
    lj_gradient_dist = np.zeros((c_pos.shape[0], c_pos.shape[0]))
    lj_gradient_dist[np.triu_indices(c_pos.shape[0])] = lj_gradient_list
    lj_gradient_dist[np.tril_indices(c_pos.shape[0])] = lj_gradient_dist.T[np.tril_indices(c_pos.shape[0])]
    
    #Calculate grad_r U(r_ij) = (dU/dr) / r_ij * \vec{pos_i} = matrix  
    #Need to add third dimension in order to have "vector-entries" x,y,z at each original 2D element i,j
    lj_gradient_pos = np.tile(lj_gradient_dist/distances, (1, 1, c_pos.shape[1])) * c_pos
    
    #Sum over either axis 0 or 1 (and reduce array to 2D with shape n_atoms x n_dim)
    force_array = -np.sum(lj_gradient_pos, axis=1)
    
    #Note c_pos.shape[0] = particle number, 1= number of dimensions
    
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
    - pot_args::list = List with arguments (e.g. constants) required for calculating the potential 
    """
    def __init__(self, n_atoms, atom_mass, n_dim, canvas_size, time, delta_t, pot_args):
        self.n_atoms = n_atoms
        self.atom_mass = atom_mass
        self.n_dim = n_dim
        self.canvas_size = canvas_size
        self.time = time
        self.delta_t = delta_t
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
                datagroup.create_dataset("iter_{index}_pos".format(index=i), self.pos)
                datagroup.create_dataset("iter_{index}_veloc".format(index=i), self.veloc)
                



##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()

    #Hardcoded inputs (Maybe replace with argv arguments)
    N_atoms = 3 #Number of particles
    ATOM_MASS = 6.6335e-26 #Mass of atoms (kg); Argon = 39.948 u
    N_DIM = 2 #Number of dimensions
    MAX_LENGTH = 5 #Canvas side length (m)
    CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH]) #Canvas size (must be ndarray!)
    
    POT_ARGS = [3.405e-10, sp.constants.k*119.8] #sigma, epsilon for Argon in SI units (see slides Lec. 1)

    #Main simulation procedure
    sim = Simulation(n_atoms=N_atoms, atom_mass=ATOM_MASS, n_dim=N_DIM, canvas_size=CANVAS_SIZE, pot_args=POT_ARGS)
    
    END_OF_TIME = 10.0 #Maximum time (s)
    DELTA_T = 0.1 #Timestep (s)
    TIME = np.arange(0, END_OF_TIME, DELTA_T) #Time array (s)
    sim.__simulate__(time=TIME, delta_t = DELTA_T)
