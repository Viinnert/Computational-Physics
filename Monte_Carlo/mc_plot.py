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
import seaborn as sns

# Main Plotting Program
def plot_expectation_vs_time(data_file_path):
    '''
    Plots various physical properties over time

    Parameters
    ----------
    data_file_path: string
        the path name to the data file containing the data to plot

    Returns
    -------
    None
    '''
    with hdf.File(data_file_path, "r") as datafile:
        data = datafile['time_sweep_output']
        times = np.array(data['time'])
        lattice_size = np.array(data['lattice_size'])
        equilib_steps = np.array(data['equilib_steps'])
        corr_calc_steps = np.array(data['corr_calc_steps'])
        mc_steps = np.array(data['mc_steps'])
     
        energies = np.array(data['energy_per_time']) / (lattice_size**2)
        magnetizations = np.array(data['magnetization_per_time']) / (lattice_size**2)

        half_average_window_size = int(2 * times.shape[0] / 100)
        running_mean_energy = np.convolve(energies, np.ones(2*half_average_window_size)/(2*half_average_window_size), mode='valid')
        running_mean_mag = np.convolve(magnetizations, np.ones(2*half_average_window_size)/(2*half_average_window_size), mode='valid')

        fig = plt.figure(figsize=(18, 10))

        sp =  fig.add_subplot(2, 1, 1 )
        plt.plot(times, energies, 'o', color='RoyalBlue')
        plt.plot(times[half_average_window_size-1:-half_average_window_size], running_mean_energy, '.', color='red' , label="Mean")
        plt.xlabel(r"Time ", fontsize=20)
        plt.ylabel(r"Energy ($J$)", fontsize=20)
        plt.axis('tight')
        plt.vlines(equilib_steps, ymin=-2., ymax=2., linestyles = 'dashed', color='red')
        plt.vlines(equilib_steps + corr_calc_steps, ymin=-2.0, ymax=2.0, linestyles = 'dashed', color='red')

        sp =  fig.add_subplot(2, 1, 2 )
        plt.plot(times, magnetizations, 'o', color='RoyalBlue')
        plt.plot(times[half_average_window_size-1: -half_average_window_size], running_mean_mag, '.', color='red' , label="Mean")

        plt.xlabel(r"Time", fontsize=20)
        plt.ylabel(r"Magnetization $M$", fontsize=20)
        plt.axis('tight')
        
        fig.suptitle(f"Expectation convergence for T={data['temperature']}")
        plt.vlines(equilib_steps, ymin=-1.0, ymax=1.0, linestyles = 'dashed', color='red')
        plt.vlines(equilib_steps + corr_calc_steps, ymin=-1.0, ymax=1.0, linestyles = 'dashed', color='red')

        plt.show()
        
        converge_fig = plt.figure(figsize=(18, 10)); # plot the correlation function vs. time retard
        
        convergences = np.array(data['convergence_per_time'])
        half_average_window_size = int(2 * times.shape[0] / 100)
        running_mean_convergence = np.convolve(convergences *100, np.ones(2*half_average_window_size)/(2*half_average_window_size), mode='valid')

        plt.plot(times, convergences*100, 'o', color='RoyalBlue')
        plt.plot(times[half_average_window_size-1: -half_average_window_size], running_mean_convergence,'.', color='red' , label="Mean")
        plt.xlabel(r"Time", fontsize=20)
        plt.ylabel(r"Convergence (%)", fontsize=20)
        plt.axis('tight')
        plt.vlines(equilib_steps, ymin=0.0, ymax=100.0, linestyles = 'dashed', color='red')
        plt.vlines(equilib_steps + corr_calc_steps, ymin=0.0, ymax=100.0, linestyles = 'dashed', color='red')

        plt.show()
        
        corr_fig = plt.figure(figsize=(18, 10)); # plot the correlation function vs. time retard
        
        correlations = np.array(data['correlation_per_time'])
        time_differences = np.arange(0, correlations.shape[0])
        
        plt.plot(time_differences, correlations,marker='o', color='RoyalBlue')
        plt.xlabel(r"Time difference", fontsize=20)
        plt.ylabel(r"Correlation", fontsize=20)
        plt.axis('tight')
        plt.show()
        
    
def plot_expectation_vs_temp(data_file_path):
    '''
    Plots various physical properties over temperatute

    Parameters
    ----------
    data_file_path: string
        the path name to the data file containing the data to plot

    Returns
    -------
    None
    '''
    with hdf.File(data_file_path, "r") as datafile:
        
        data = datafile['temp_sweep_output']
        temperatures = np.array(data['temperature'])

        fig = plt.figure(figsize=(18, 10)); # plot the calculated values    

        sp =  fig.add_subplot(2, 2, 1 )
        plt.errorbar(temperatures, np.array(data['energy_per_temp_mean']), yerr= np.array(data['energy_per_temp_std']), 
                     fmt='o', color='RoyalBlue', ecolor='black', capsize=4)
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20)
        plt.ylabel(r"Energy ($J$)", fontsize=20)
        plt.axis('tight')

        sp =  fig.add_subplot(2, 2, 2 )
        plt.errorbar(temperatures,  np.array(data['magnetization_per_temp_mean']), yerr=  np.array(data['magnetization_per_temp_std']), 
                     fmt='o', color='RoyalBlue', ecolor='black', capsize=4)
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20)
        plt.ylabel(r"Magnetization $M$", fontsize=20)
        plt.axis('tight')

        sp =  fig.add_subplot(2, 2, 3 )
        plt.errorbar(temperatures, np.array(data['specific_heat_per_temp_mean']), yerr= np.array(data['specific_heat_per_temp_std']), 
                     fmt='o', color='RoyalBlue', ecolor='black', capsize=4)
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20)
        plt.ylabel(r"Specific Heat $C_v$", fontsize=20)
        plt.axis('tight')

        sp =  fig.add_subplot(2, 2, 4 )
        plt.errorbar(temperatures,  np.array(data['susceptibility_per_temp_mean']), yerr= np.array(data['susceptibility_per_temp_std']), 
                     fmt='o', color='RoyalBlue', ecolor='black', capsize=4)
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20)
        plt.ylabel(r"Susceptibility $\chi$", fontsize=20)
        plt.axis('tight')

        plt.show()
                
        mag_fig = plt.figure(figsize=(18, 10)); # plot the correlation time per temperature:
        
        mag_data = np.array(data["magnetization_per_time_per_temp"]) / np.array(data["lattice_size"])**2
        times = np.arange(0, mag_data.shape[1])
        
        equilib_steps = np.array(data["equilib_steps_per_temp"])[0] 
        corr_calc_steps = np.array(data["corr_calc_steps_per_temp"])[0] 
        mc_steps = np.array(data["mc_steps_per_temp"])[0]
        
        with sns.color_palette("coolwarm_r", n_colors=temperatures.shape[0], ):
            plt.plot(times, np.flip(mag_data[1:, :], axis=0).T,marker='o', alpha=0.4)
            plt.xlabel(r"Time", fontsize=20)
            plt.ylabel(r"Magnetization per spin", fontsize=20)
            plt.axis('tight')
            plt.legend(np.flip(np.round(temperatures, 1)))
            plt.vlines(equilib_steps, ymin=-1.0, ymax=1.0, linestyles = 'dashed', color='red')
            plt.vlines(equilib_steps + corr_calc_steps, ymin=-1.0, ymax=1.0, linestyles = 'dashed', color='red')
        plt.show()
        
        convergence_fig = plt.figure(figsize=(18, 10)); # plot the correlation time per temperature:
        
        converge_data = np.array(data["convergence_per_time_per_temp"])
        times = np.arange(0, converge_data.shape[1])
        with sns.color_palette("coolwarm", n_colors=temperatures.shape[0]):
            plt.plot(times, converge_data[1:, :].T,marker='o')
            plt.xlabel(r"Time", fontsize=20)
            plt.ylabel(r"Convergence (%)", fontsize=20)
            plt.axis('tight')
            plt.legend(np.round(temperatures, 1))
            plt.vlines(equilib_steps, ymin=0.0, ymax=100.0, linestyles = 'dashed', color='red')
            plt.vlines(equilib_steps+ corr_calc_steps, ymin=0.0, ymax=100.0, linestyles = 'dashed', color='red')

        plt.show()
         
        corrtime_fig = plt.figure(figsize=(18, 10)); # plot the correlation time per temperature:
        
        plt.plot(temperatures, np.array(data['correlation_time_per_temp']),marker='o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20)
        plt.ylabel(r"Correlation time $\tau$", fontsize=20)
        plt.axis('tight')

        plt.show()
        

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

    plot_expectation_vs_time(DATA_PATH + DATA_FILENAME)
    plot_expectation_vs_temp(DATA_PATH + DATA_FILENAME)

    print("Success")
    


