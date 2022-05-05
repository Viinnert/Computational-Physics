import sys
import os
import numpy as np

np.random.seed(42) #Fix seed for reproducebility.

# Workdir path = dir. of main scripts!
EXPERIMENTS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
WORKDIR_PATH = EXPERIMENTS_PATH + "../"

sys.path.insert(1, WORKDIR_PATH)

from kh_main import *
from kh_plot import *

def init_kelvin_helmholtz(lattice_size, lattice_flow_vecs):
    median_density_flow = 1.0
    density_flow_ratio = 3.0 #higher/lower
    #r*l = h with (h+l)/2 = m -> 2m - l = h = r*l -> l = 2m/(r+1)
    lower_density_flow = 2*median_density_flow/(density_flow_ratio+1)
    
    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    init_map[:, :int(init_map.shape[1]/2),:] = lower_density_flow/lattice_flow_vecs.shape[1]
    init_map[:, int(init_map.shape[1]/2):,:] = (density_flow_ratio*lower_density_flow)/lattice_flow_vecs.shape[1]

    return init_map


if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (80,80) # Canvas size 
    END_OF_TIME = 50 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    RELAX_TIME = 4.0
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1.1, 0.0 ,1.1, 0.0, 1.1, 1/RELAX_TIME, 1/RELAX_TIME])
    #RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = 1.1
    ALPHA3 = 1.1
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              map_init=init_kelvin_helmholtz, 
                              relaxation_coeffs=RELAXATION_COEFFS, 
                              alpha3 = ALPHA3, 
                              gamma4 = GAMMA4) 
    
    lbm = LatticeBoltzmann(dfm)
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    plot_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
        


