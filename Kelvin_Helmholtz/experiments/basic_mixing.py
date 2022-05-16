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

def init_static(lattice_size, lattice_flow_vecs):
    median_density_flow = 1.0
    density_flow_ratio = 3.0 #higher/lower
    #r*l = h with (h+l)/2 = m -> 2m - l = h = r*l -> l = 2m/(r+1)
    lower_density_flow = 2*median_density_flow/(density_flow_ratio+1)
    
    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    init_map[:, :int(init_map.shape[1]/2),0] = lower_density_flow
    init_map[:, int(init_map.shape[1]/2):,0] = density_flow_ratio*lower_density_flow

    #init_map[:, :int(init_map.shape[1]/2),:] = lower_density_flow/lattice_flow_vecs.shape[1]
    #init_map[:, int(init_map.shape[1]/2):,:] = (density_flow_ratio*lower_density_flow)/lattice_flow_vecs.shape[1]
    return init_map

def init_uniform(lattice_size, lattice_flow_vecs):
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
    LATTICE_SIZE = (426, 240) # Canvas size 
    END_OF_TIME = 100 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    #RELAXATION_COEFFS = np.array([0.0, 0.5, 1.1, 0.0 ,1.1, 0.0, 1.1, 0.5, 0.5])
    RELAX_TIME = 10
    #RELAXATION_COEFFS = np.array([1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME ,1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = -18
    ALPHA3 = 4
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              mass=1.0,
                              map_init=init_static, 
                              relaxation_coeffs=RELAXATION_COEFFS, 
                              alpha3 = ALPHA3, 
                              gamma4 = GAMMA4) 
    
    lbm = LatticeBoltzmann(density_flow_map=dfm,
                           advect_BCs=semi_periodic_BCs)
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    #plot_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
    animate_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
    
    #plot_D2Q9_pressure(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_velocity_profile(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_moments_vs_time(DATA_PATH+DATA_FILENAME)
        
     
        

        
