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

#Available plot routines:


def load_from_file(data_file_path):
    with hdf.File(data_file_path, "rb") as datafile:
        results = datafile['time_sweep_output']
        return results

if __name__ == "__main__":

    DATA_PATH = EXPERIMENTS_PATH + "data/" 
    DATA_FILENAME = "turb_kh_temp_data.hdf5"

    #results = load_from_file(DATA_PATH+DATA_FILENAME)
    
    # Plot:
    #plot_D2_density_flow(DATA_PATH+DATA_FILENAME)
    #animate_D2_density_flow(DATA_PATH+DATA_FILENAME)
    
    #plot_D2_pressure(DATA_PATH+DATA_FILENAME)
    #plot_D2_velocity_profile(DATA_PATH+DATA_FILENAME)
    plot_D2_moments_vs_time(DATA_PATH+DATA_FILENAME)
        
        