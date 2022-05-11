from mimetypes import init
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
    density_1 = 2.0
    x_veloc_1 = 0.5
    density_2 = 5.0
    x_veloc_2 = -0.5 
    assert density_1 <= density_2
    assert x_veloc_2 <= x_veloc_1
    
    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    x_coord = np.arange(init_map.shape[0])
    y_coord = np.arange(init_map.shape[1])
    
    density_trans_width = y_coord.shape[0]/20
    velocity_trans_width = y_coord.shape[0]/20
    boundary_y_idx = int(y_coord.shape[0]/2)
    
    #1 = negative x-direction flow, 3=positive x-direction-flow
    print(np.abs(np.tile((x_veloc_1 + x_veloc_2)/2 - ((x_veloc_1 - x_veloc_2)/2)*np.tanh(y_coord[boundary_y_idx:] / velocity_trans_width)[np.newaxis,:], reps=(x_coord.shape[0],1))))
    init_map[:, boundary_y_idx:, 1] = np.abs(np.tile((x_veloc_1 + x_veloc_2)/2 - ((x_veloc_1 - x_veloc_2)/2)*np.tanh(y_coord[boundary_y_idx:] / velocity_trans_width)[np.newaxis,:], reps=(x_coord.shape[0],1)))
    init_map[:, :boundary_y_idx, 3] = np.abs(np.tile((x_veloc_1 + x_veloc_2)/2 - ((x_veloc_1 - x_veloc_2)/2)*np.tanh(y_coord[:boundary_y_idx] / velocity_trans_width)[np.newaxis,:], reps=(x_coord.shape[0],1)))
    
    #Complete rest of uniform density
    print(np.tile(((density_1 + density_2)/2 - ((density_2 - density_1)/2)*np.tanh(y_coord/density_trans_width))[np.newaxis,:], reps=(x_coord.shape[0],1)).shape)
    init_target_density = np.tile(((density_1 + density_2)/2 - ((density_2 - density_1)/2)*np.tanh(y_coord/density_trans_width))[np.newaxis,:], reps=(x_coord.shape[0],1))
    init_map[:, :, 0] += (init_target_density - np.sum(init_map, axis=-1))
    
    
    #Periodic pertubation
    #ampl = 0.02
    #ang_freq = 1/(init_map.shape[1]/4)
    #offset = 0.0
    #init_map[:, :, 1] = ampl * (np.sin(2*np.pi*y_coord)* ang_freq +offset)[np.newaxis, :]
    return init_map



if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (100,60) # Canvas size 
    END_OF_TIME = 100 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    #RELAXATION_COEFFS = np.array([0.0, 0.5, 1.1, 0.0 ,1.1, 0.0, 1.1, 0.5, 0.5])
    RELAX_TIME = 8
    #RELAXATION_COEFFS = np.array([1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME ,1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = -18
    ALPHA3 = 4
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              mass=1.0,
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
    #plot_D2Q9_pressure(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_velocity_profile(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_moments_vs_time(DATA_PATH+DATA_FILENAME)
        
        

        
