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

def init_periodic_pertub(lattice_size, lattice_flow_vecs):
    median_density_flow = 1.0

    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    
    #Uniform density
    init_map[:, :, 0] = median_density_flow
    
    #Periodic pertubation
    ampl = 0.8
    ang_freq = 2*np.pi*init_map.shape[1]/2
    offset = 0.2
    init_map[:, :, 2] = (ampl * median_density_flow * (np.sin(np.arange(0, init_map.shape[1])* ang_freq )+offset))[np.newaxis, :]
    
    return init_map



if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (50,50) # Canvas size 
    END_OF_TIME = 10 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    #RELAXATION_COEFFS = np.array([0.0, 0.5, 1.1, 0.0 ,1.1, 0.0, 1.1, 0.5, 0.5])
    RELAX_TIME = 10.0
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = 1.0
    ALPHA3 = 1.0
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              map_init=init_periodic_pertub, 
                              relaxation_coeffs=RELAXATION_COEFFS, 
                              alpha3 = ALPHA3, 
                              gamma4 = GAMMA4) 
    
    lbm = LatticeBoltzmann([dfm])
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    plot_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
        


        

        
