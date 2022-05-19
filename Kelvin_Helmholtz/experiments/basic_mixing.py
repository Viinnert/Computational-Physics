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

def init_static(lattice_size, lattice_flow_vecs, mom_space_transform, inv_mom_space_transform):
    #Fix density, energy (/pressure) and velocities
    density_1 = 5.0
    x_veloc_1 = 0.2
    density_2 = 2.0
    x_veloc_2 = 0.2
    pressure_1 = 0.5
    pressure_2 = 0.2
    
    #Initialize empty moment space:
    init_mom_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    x_coord = np.arange(init_mom_map.shape[0])
    y_coord = np.arange(init_mom_map.shape[1])
    
    y_boundary_idx = int(y_coord.shape[0]/2)
    density_trans_width = 1.0 #y_coord.shape[0]/1000
    velocity_trans_width = 1.0 #y_coord.shape[0]/1000
    
    #Density
    init_target_density = np.tile(((density_1 + density_2)/2 - ((density_1 - density_2)/2)*np.tanh((y_coord-y_boundary_idx) / density_trans_width))[np.newaxis,:], reps=(x_coord.shape[0],1))
    init_mom_map[:, :, 0] = init_target_density

    #Velocity
    init_x_velocs = np.tile((x_veloc_1 + x_veloc_2)/2 - ((x_veloc_1 - x_veloc_2)/2)*np.tanh((y_coord-y_boundary_idx) / velocity_trans_width)[np.newaxis,:], reps=(x_coord.shape[0],1))
    init_mom_map[:, :, 1] = init_x_velocs
    
    init_y_velocs = np.zeros(init_mom_map[:, :, 2].shape)
    init_mom_map[:, :, 2] = init_y_velocs

    #Set pressure via energy:
    init_mom_map[:, y_boundary_idx:, 3] = pressure_1 / init_target_density[:, y_boundary_idx:]
    init_mom_map[:, :y_boundary_idx, 3] = pressure_2 / init_target_density[:, :y_boundary_idx]
    
    #Set all other moments from dependence and obtain regular density flow map:
    init_mom_map = D2Q16_set_nonconserv_moments(init_mom_map)
    init_map = np.einsum('ij,klj->kli', inv_mom_space_transform, init_mom_map)

    return init_map

if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (426, 240) # Canvas size 
    END_OF_TIME = 1000 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "mixing_temp_data.hdf5"
    
    #RELAX_TIME = 200.0
    #RELAXATION_COEFFS = np.array([1/RELAX_TIME for i in range(16)])
    RELAXATION_COEFFS = 0.2e-8 * np.array([1e5, 1e5, 1e5, 1e5, 6500, 1e5, 9e4, 9e4, 8e4, 1e5, 1e5, 1e5, 7e4, 8e3, 2.5e4, 1e5])
    
    # Main simulation procedure
    dfm = DensityFlowMap.D2Q16(lattice_size=LATTICE_SIZE, 
                              mass=1.0,
                              map_init=init_static, 
                              relaxation_coeffs=RELAXATION_COEFFS) 
    
    lbm = LatticeBoltzmann(density_flow_map=dfm,
                           advect_BCs=None)
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    plot_D2_density_flow(DATA_PATH+DATA_FILENAME)
    #animate_D2_density_flow(DATA_PATH+DATA_FILENAME)
    
    #plot_D2_pressure(DATA_PATH+DATA_FILENAME)
    #plot_D2_velocity_profile(DATA_PATH+DATA_FILENAME)
    #plot_D2_moments_vs_time(DATA_PATH+DATA_FILENAME)
        
     
        

        
