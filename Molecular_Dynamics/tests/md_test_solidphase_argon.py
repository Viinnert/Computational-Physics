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
    i_veloc = np.array([[1, 0],
                        [-1, 0]])

    return i_pos, i_veloc


if __name__ == "__main__":
    #def print_usage():
    #    print("Usage:\n", file=sys.stderr)
    #    print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    #required_args = 0
    #if len(sys.argv) != required_args:
    #    print_usage()
    
    # Hardcoded inputs (Maybe replace with argv arguments)
    N_DIM = 3 # Number of dimensions
    N_UNIT_CELLS = (3,3,3) # Number of unit cells per dimension
    ATOM_MASS = 6.6335e-26 # Mass of atoms (kg); Argon = 39.948 u
    POT_ARGS = {'sigma': 3.405e-10, 'epsilon': 119.8} # sigma, epsilon for Argon in units of m and k_B respectively.

    # Dimensionless constants
    CANVAS_ASPECT_RATIO = (1,1,1) # Canvas size (must be ndarray!)
    END_OF_TIME = 2 # Maximum time
    DELTA_T = 0.01 # Timestep
    N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    
    TEMPERATURE = 0.5 # Kelvin
    DENSITY = 1.2 # Dimensionless: scaled by m/sigma**n_dim
    INIT_MODE = "fcc"

    DATA_PATH = WORKDIR_PATH + "data/" 
    TRAJ_DATA_FILENAME = "trajectories_fcc.hdf5"

    #Main simulation procedure
    sim = Simulation(n_atoms_or_unit_cells=N_UNIT_CELLS, 
                     atom_mass=ATOM_MASS, 
                     density=DENSITY, 
                     temperature=TEMPERATURE,
                     n_dim=N_DIM, 
                     canvas_aspect_ratio=CANVAS_ASPECT_RATIO, 
                     pot_args=POT_ARGS, init_mode=INIT_MODE, 
                     data_path=DATA_PATH, 
                     data_filename=TRAJ_DATA_FILENAME)
    
    sim.__simulate__(n_iterations=N_ITERATIONS, delta_t=DELTA_T)
    
    #Plot trajectories, energy and forces:
    with hdf.File(DATA_PATH + TRAJ_DATA_FILENAME,'r') as data_file:
        
        plot_energy(data_file)
        
        plot_forces(data_file)
        
        #animate_trajectories3D(data_file)

    
    #Pair correlation calculation and plotting:
    SIM_REPETITIONS = 4
    REDUCED_N_ITERATIONS = int(N_ITERATIONS / 1)
    
    histogram_list = []
    bin_edges = np.array([])
    
    for r in tqdm(range(SIM_REPETITIONS)):
        temp_sim = Simulation(n_atoms_or_unit_cells=N_UNIT_CELLS, 
                        atom_mass=ATOM_MASS, 
                        density=DENSITY, 
                        temperature=TEMPERATURE,
                        n_dim=N_DIM, 
                        canvas_aspect_ratio=CANVAS_ASPECT_RATIO, 
                        pot_args=POT_ARGS, 
                        init_mode=INIT_MODE, 
                        data_path=DATA_PATH, 
                        data_filename="temp.hdf5")
            
        temp_sim.__simulate__(n_iterations=REDUCED_N_ITERATIONS, delta_t=DELTA_T)
        
        histogram, bin_edges = temp_sim.paircor ,temp_sim.paircor_bin_edges
        histogram_list.append(histogram)
    
    plot_av_histogram(histogram_list, bin_edges)