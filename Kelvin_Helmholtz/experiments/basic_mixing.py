import sys
import os
import numpy as np

np.random.seed(42) #Fix seed for reproducebility.

# Workdir path = dir. of main scripts!
EXPERIMENTS_PATH = os.path.dirname(os.path.realpath(__file__))
WORKDIR_PATH = EXPERIMENTS_PATH + "/../"

sys.path.insert(1, WORKDIR_PATH)

from kh_main import *
from kh_plot import *

def init_helmholtz(lattice_size, lattice_flow_vecs):
    median_density_flow = 1.0
    density_flow_ratio = 3.0 #higher/lower
    #r*l = h with (h+l)/2 = m -> 2m - l = h = r*l -> l = 2m/(r+1)
    lower_density_flow = 2*median_density_flow/(density_flow_ratio+1)
    
    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    init_map[:, :int(init_map[1]/2),:] = lower_density_flow
    init_map[:, int(init_map[1]/2):,:] = density_flow_ratio*lower_density_flow
    return init_map


if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (100,100) # Canvas size 
    END_OF_TIME = 10 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    #RELAXATION_COEFFS = np.array([0.0, 0.5, 1.1, 0.0 ,1.1, 0.0, 1.1, 0.5, 0.5])
    RELAX_TIME = 1.0
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = 1.0
    ALPHA3 = 1.0
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              map_init=init_helmholtz, 
                              relaxation_coeffs=RELAXATION_COEFFS, 
                              alpha3 = ALPHA3, 
                              gamma4 = GAMMA4) 
    
    lbm = LatticeBoltzmann(dfm)
    
    results = lbm.__run__()
    
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    
    # Plot:
    with hdf.File(DATA_PATH + DATA_FILENAME,'r') as data_file:
        results = data_file        
        plot_world_map(results) 
        


        

        
