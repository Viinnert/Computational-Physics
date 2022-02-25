import sys
import os

#Workdir path = dir. of main scripts!
WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.insert(1, WORKDIR_PATH)

from md_main import *
from md_plot import *


def simple_init(size):
    i_pos = np.array([[0.3*size[0], 0.51 *size[1]],
                      [0.7*size[0], 0.49 *size[1]]])
    i_veloc = np.array([[0.6, 0],
                        [-0.6, 0]])

    return i_pos, i_veloc


if __name__ == "__main__":
    #def print_usage():
    #    print("Usage:\n", file=sys.stderr)
    #    print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    #required_args = 0
    #if len(sys.argv) != required_args:
    #    print_usage()
    
    # Hardcoded inputs (Maybe replace with argv arguments)
    N_DIM = 2 # Number of dimensions
    N_ATOMS = 2 # Number of particles
    TEMPERATURE = 100 # Kelvin
    ATOM_MASS = 6.6335e-26 # Mass of atoms (kg); Argon = 39.948 u
    POT_ARGS = {'sigma': 3.405e-10, 'epsilon': sp_const.k*119.8} # sigma, epsilon for Argon in SI units (see slides Lec. 1)
    POT_ARGS = {'sigma': 1, 'epsilon': 1} # sigma, epsilon for Argon in SI units (see slides Lec. 1)
    
    # Dimensionless constants

    MAX_LENGTH = 20
    CANVAS_SIZE = np.array([MAX_LENGTH, MAX_LENGTH]) # Canvas size (must be ndarray!)
    END_OF_TIME = 10 # Maximum time

    DELTA_T = 0.01 # Timestep
    N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    
    INIT_MODE = "random"
    INIT_MODE = lambda: simple_init(CANVAS_SIZE)
    
    DATA_PATH = WORKDIR_PATH + "data/" 
    DATA_FILENAME = "trajectories.hdf5"
    
    #Main simulation procedure
    sim = Simulation(n_atoms=N_ATOMS, atom_mass=ATOM_MASS, n_dim=N_DIM, canvas_size=CANVAS_SIZE, pot_args=POT_ARGS, init_mode=INIT_MODE, data_path=DATA_PATH, data_filename=DATA_FILENAME)
    sim.__simulate__(n_iterations=N_ITERATIONS, delta_t=DELTA_T)
    
    #Plot:
    with hdf.File(DATA_PATH + DATA_FILENAME,'r') as data_file:
        
        plot_energy(data_file)
        
        animate_trajectories2D(data_file)
        #plot_trajectories2D(data_file)
        
