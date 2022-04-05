"""
###############################################################################

Part of the Monte-Carlo Simulation project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)

################################################################################

Defines the plotting functions and code used for the report

Classes:
- --

Functions:
- ---


################################################################################
"""

import numpy as np
import h5py as hdf
import os
import sys
import matplotlib.pyplot as plt

#Main Plotting Program
def plot_expectation_vs_time(data_file_path):
    """
    Insert docstring
    """
    output_kind = 'time_sweep_output'
    with hdf.File(data_file_path, "r") as datafile:
        data = datafile[output_kind]
        
        fig = plt.figure(figsize=(18, 10)); # plot the calculated values    

        print(np.array(data['time']), np.array(data['energy_per_time']))
        sp =  fig.add_subplot(2, 1, 1 );
        plt.plot(np.array(data['time']), np.array(data['energy_per_time']), 'o', color='RoyalBlue')
        average_window_size = int(np.array(data['time']).shape[0] / 12)
        running_mean_energy = np.convolve(np.array(data['energy_per_time']), np.ones(average_window_size)/average_window_size, mode='valid')
        plt.plot(np.array(data['time'])[average_window_size-1:], running_mean_energy, '.', color='red' , label="Mean")
        plt.xlabel(r"Time$", fontsize=20);
        plt.ylabel(r"Energy ($J$)", fontsize=20);         plt.axis('tight');

        sp =  fig.add_subplot(2, 1, 2 );
        plt.plot(np.array(data['time']), np.array(data['magnetization_per_time']), 'o', color='RoyalBlue')
        average_window_size = int(np.array(data['time']).shape[0] / 12)
        running_mean_mag = np.convolve(np.array(data['magnetization_per_time']), np.ones(average_window_size)/average_window_size, mode='valid')
        plt.plot(np.array(data['time'])[average_window_size-1:], running_mean_mag, '.', color='red' , label="Mean")
        plt.xlabel(r"Time", fontsize=20); 
        plt.ylabel(r"Magnetization $M$", fontsize=20);   plt.axis('tight');
        
        fig.suptitle("Expectation convergence for T={}".format(data['temperature']))
        plt.show()
    
def plot_expectation_vs_temp(data_file_path):
    """
    Insert docstring
    """
    output_kind = 'temp_sweep_output'
    with hdf.File(data_file_path, "r") as datafile:
        data = datafile[output_kind]
        
        fig = plt.figure(figsize=(18, 10)); # plot the calculated values    

        sp =  fig.add_subplot(2, 2, 1 );
        plt.plot(np.array(data['temperature']), np.array(data['energy_per_temp']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20);
        plt.ylabel(r"Energy ($J$)", fontsize=20);         plt.axis('tight');

        sp =  fig.add_subplot(2, 2, 2 );
        plt.plot(np.array(data['temperature']), abs(np.array(data['magnetization_per_temp'])), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20); 
        plt.ylabel(r"Magnetization $M$", fontsize=20);   plt.axis('tight');

        sp =  fig.add_subplot(2, 2, 3 );
        plt.plot(np.array(data['temperature']), np.array(data['specific_heat_per_temp']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20);  
        plt.ylabel(r"Specific Heat $C_v$", fontsize=20);   plt.axis('tight');   

        sp =  fig.add_subplot(2, 2, 4 );
        plt.plot(np.array(data['temperature']), np.array(data['susceptibility_per_temp']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20); 
        plt.ylabel(r"Susceptibility $\chi$", fontsize=20);   plt.axis('tight');

        plt.show()


def plot_correlation(data_file_path):

    with hdf.File(data_file_path, "r") as datafile:
        data = datafile['temp_sweep_output']['correlation_time_per_temp']
        print(data)

if __name__ == "__main__":
    def print_usage():
        print("You gave the wrong number of arguments for mc_plot \n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("python mc_plot.py [arg1] [arg2] ... [data filename] \n", file=sys.stderr)
        sys.exit()

    required_args = ["data filename"]
    #sys.argv = sys.argv[1:] #Remove default script path argument
    sys.argv = ["data.hdf5"]
    
    if len(sys.argv) != len(required_args):
        print_usage()
    
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = WORKDIR_PATH + "/data/" 
    DATA_FILENAME = sys.argv[-1]

    #plot_expectation_vs_time(DATA_PATH + DATA_FILENAME)
    #plot_expectation_vs_temp(DATA_PATH + DATA_FILENAME)
    plot_correlation(DATA_PATH + DATA_FILENAME)

    print("Success")
    


