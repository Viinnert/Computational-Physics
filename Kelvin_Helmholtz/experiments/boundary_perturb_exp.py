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




def init_boundary_perturb(lattice_size, lattice_flow_vecs, mom_space_transform):
    density_1 = 2.0
    x_veloc_1 = 0.5
    density_2 = 5.0
    x_veloc_2 = -0.5
    
    init_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    x_coord = np.arange(init_map.shape[0])
    y_coord = np.arange(init_map.shape[1])
    
    init_map[:, :, 0] = density_1

    y_boundary_idx = int(y_coord.shape[0]/2)
    density_trans_width = 1.0 #y_coord.shape[0]/1000
    velocity_trans_width = 1.0 #y_coord.shape[0]/1000
    
    init_target_density = np.tile(((density_1 + density_2)/2 - ((density_1 - density_2)/2)*np.tanh((y_coord-y_boundary_idx) / density_trans_width))[np.newaxis,:], reps=(x_coord.shape[0],1))
    init_map[:, :, 0] = init_target_density

    init_x_velocs = np.tile((x_veloc_1 + x_veloc_2)/2 - ((x_veloc_1 - x_veloc_2)/2)*np.tanh((y_coord-y_boundary_idx) / velocity_trans_width)[np.newaxis,:], reps=(x_coord.shape[0],1))
    (init_map[:, :, 1])[init_x_velocs > 0.0] = np.abs(init_x_velocs[init_x_velocs > 0.0])
    (init_map[:, :, 3])[init_x_velocs < 0.0] = np.abs(init_x_velocs[init_x_velocs < 0.0])
    
    #Periodic perturbation:
    ampl = 0.02
    freq = 8*1/(init_map.shape[0]*2/4)
    offset = 0.0
    perturb_decay_length = 8* (1/np.sqrt(2*np.pi*freq)) # y_coord.shape[0]/2
    periodic_perturb = ampl*(np.sin(2*np.pi*freq*x_coord + offset)[:, np.newaxis] * np.exp(-(y_coord - y_boundary_idx)**2 / (perturb_decay_length**2))[np.newaxis, :])
    (init_map[:, :, 2])[periodic_perturb > 0.0] = np.abs(periodic_perturb[periodic_perturb > 0.0])
    (init_map[:, :, 4])[periodic_perturb < 0.0] = np.abs(periodic_perturb[periodic_perturb < 0.0])
    
    init_target_density = np.tile(((density_1 + density_2)/2 - ((density_1 - density_2)/2)*np.tanh((y_coord-y_boundary_idx) / density_trans_width))[np.newaxis,:], reps=(x_coord.shape[0],1))
    init_map[:, :, 0] += (init_target_density - np.sum(init_map, axis=-1))
    #(np.sin(2*np.pi*freq*x_coord + offset)[:, np.newaxis] * (init_target_density)
    assert not any(init_map.flatten() < 0.0)
    
    return init_map


if __name__ == "__main__":

    # Dimensionless constants
    #LATTICE_SIZE = (1280, 720) # Canvas size 
    LATTICE_SIZE = (600, 600) # Canvas size  
    END_OF_TIME = 500 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    #RELAXATION_COEFFS = np.array([0.0, 0.5, 1.1, 0.0 ,1.1, 0.0, 1.1, 0.5, 0.5])
    RELAX_TIME = 8.0
    #RELAXATION_COEFFS = np.array([1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME ,1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    RELAXATION_COEFFS = np.array([0.0, 1/RELAX_TIME, 1/RELAX_TIME, 0.0 ,1/RELAX_TIME, 0.0, 1/RELAX_TIME, 1/RELAX_TIME, 1/RELAX_TIME])
    GAMMA4 = -18
    ALPHA3 = 4
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q9(lattice_size=LATTICE_SIZE, 
                              mass=1.0,
                              map_init=init_boundary_perturb, 
                              relaxation_coeffs=RELAXATION_COEFFS, 
                              alpha3 = ALPHA3, 
                              gamma4 = GAMMA4) 
    
    lbm = LatticeBoltzmann(density_flow_map = dfm,
                           advect_BCs=None)
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    plot_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
    #animate_D2Q9_density_flow(DATA_PATH+DATA_FILENAME)
    
    #plot_D2Q9_pressure(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_velocity_profile(DATA_PATH+DATA_FILENAME)
    #plot_D2Q9_moments_vs_time(DATA_PATH+DATA_FILENAME)
        
        

        
