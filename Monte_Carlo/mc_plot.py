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

def plot_expectation_vs_temp(data_file_path):
    """
    Insert docstring
    """
    with hdf.File(data_file_path, "r") as datafile:
        fig = plt.figure(figsize=(18, 10)); # plot the calculated values    

        sp =  fig.add_subplot(2, 2, 1 );
        plt.plot(np.array(datafile['temperature']), np.array(datafile['energy']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20);
        plt.ylabel(r"Energy ($J$)", fontsize=20);         plt.axis('tight');

        sp =  fig.add_subplot(2, 2, 2 );
        plt.plot(np.array(datafile['temperature']), abs(np.array(datafile['magnetization'])), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20); 
        plt.ylabel(r"Magnetization $M$", fontsize=20);   plt.axis('tight');

        sp =  fig.add_subplot(2, 2, 3 );
        plt.plot(np.array(datafile['temperature']), np.array(datafile['specific_heat']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20);  
        plt.ylabel(r"Specific Heat $C_v$", fontsize=20);   plt.axis('tight');   

        sp =  fig.add_subplot(2, 2, 4 );
        plt.plot(np.array(datafile['temperature']), np.array(datafile['susceptibility']), 'o', color='RoyalBlue')
        plt.xlabel(r"Temperature $(J/k_B)$", fontsize=20); 
        plt.ylabel(r"Susceptibility $\chi$", fontsize=20);   plt.axis('tight');

        plt.show()

if __name__ == "__main__":
    def print_usage():
        print("You gave the wrong number of arguments for mc_plot \n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("python mc_plot.py [arg1] [arg2] ... [data filename] \n", file=sys.stderr)
        sys.exit()

    required_args = ["data filename"]
    sys.argv = sys.argv[1:] #Remove default script path argument
    if len(sys.argv) != len(required_args):
        print_usage()
    
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = WORKDIR_PATH + "/data/" 
    DATA_FILENAME = sys.argv[-1]

    plot_expectation_vs_temp(DATA_PATH + DATA_FILENAME)
    
    print("Success")
    


