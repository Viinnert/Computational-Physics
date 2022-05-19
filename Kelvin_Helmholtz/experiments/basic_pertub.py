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

def init_periodic_pertub(lattice_size, lattice_flow_vecs, mom_space_transform, inv_mom_space_transform):
    #Fix density, energy (/pressure) and velocities
    density = 2.0
    x_veloc = 0.2
    y_veloc = 0.0
    pressure = 0.2

    #Initialize empty space:
   
    init_mom_map = np.zeros((*lattice_size, lattice_flow_vecs.shape[1]))
    x_coord = np.arange(init_mom_map.shape[0])
    y_coord = np.arange(init_mom_map.shape[1])

    init_mom_map[:, :, 0] = density

    #Periodic perturbation:
    ampl = x_veloc
    freq = 3* 1/(y_coord.shape[0])
    offset = 0.0
    periodic_y_perturb = ampl*(np.sin(2*np.pi*freq*y_coord + offset)[np.newaxis,:]) 
    init_mom_map[:, :, 1] = np.tile(periodic_y_perturb, reps=(x_coord.shape[0],1))

    init_mom_map[:, :, 2] = y_veloc

    #Set pressure via energy:
    pressure_perturb =  pressure *(np.sin(2*np.pi*freq*y_coord + offset)[np.newaxis,:]) 
    init_mom_map[:, :, 3] = np.tile(pressure / density, reps=(x_coord.shape[0],1))

    init_mom_map = D2Q16_set_nonconserv_moments(init_mom_map)
    init_map = np.einsum('ij,klj->kli', inv_mom_space_transform, init_mom_map)

    return init_map



if __name__ == "__main__":

    # Dimensionless constants
    LATTICE_SIZE = (300,200) # Canvas size 
    END_OF_TIME = 3000 # Maximum time

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "temp_data.hdf5"
    
    RELAXATION_COEFFS =  1e-8 * np.array([1e5, 1e5, 1e5, 1e5, 6500, 1e5, 9e4, 9e4, 8e4, 1e5, 1e5, 1e5, 7e4, 8e3, 2.5e4, 1e5])

    # Main simulation procedure
    dfm = DensityFlowMap.D2Q16(lattice_size=LATTICE_SIZE, 
                              mass=1.0,
                              map_init=init_periodic_pertub, 
                              relaxation_coeffs=RELAXATION_COEFFS) 
    
    lbm = LatticeBoltzmann(density_flow_map=dfm,
                           advect_BCs=None)
    
    output = lbm.__run__(end_of_time=END_OF_TIME)
    
    results = {'time_sweep_output': output}
    
    save_to_file(results, data_file_path=DATA_PATH+DATA_FILENAME)
    
    # Plot:
    #plot_D2_density_flow(DATA_PATH+DATA_FILENAME)
    #plot_D2_pressure(DATA_PATH+DATA_FILENAME)
    plot_D2_velocity_profile(DATA_PATH+DATA_FILENAME)
    #plot_D2_moments_vs_time(DATA_PATH+DATA_FILENAME)
        

        
