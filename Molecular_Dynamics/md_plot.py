"""
###############################################################################

Part of the Molecular Dynamics project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)

################################################################################

Defines the plotting functions and code used for the report

Classes:
-

Functions:
- 


################################################################################
"""
import os
import sys
import numpy as np
import h5py as hdf
import matplotlib as mpl
import matplotlib.pyplot as plt


WORKDIR_PATH = os.getcwd()
DATA_PATH = WORKDIR_PATH + "/data/" 
DATA_FILENAME = "data.hdf5"

# Initalize plot parameters
params = {
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": True,
    "figure.figsize": (3.15, 3.15),
    "figure.subplot.left": 0.14,
    "figure.subplot.right": 0.99,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.99,
    "figure.subplot.wspace": 0.15,
    "figure.subplot.hspace": 0.12,
    "lines.markersize": 6,
    "lines.linewidth": 2.0,
    "text.latex.unicode": True,
}
mpl.rcParams.update(params)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})


def plot_trajectories(data_file):
    """
    Plots the trajectories of particles stored in a given data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the trajectory data
                                       Groups label iterations 'iter_{index}'
                                       ,groups should contain position and velocity datasets 
                                       named '{groupname}_pos' and '{groupname}_veloc' 

    Return
    - --
    """
    n_iterations = list(data_file.keys())

    #Extract number of dimensions and atoms from any array in file
    n_atoms, n_dim = data_file["iter_0"]["iter_pos"].shape
    
    #Plot trajectories iteration-wise
    for i in range(n_iterations+1):
        #Get arrays from the data file in shape (n_atoms x n_dim) 
        current_pos = data_file["iter_{index}".format(index=i)]["iter_{index}_pos".format(index=i)]
        current_veloc = data_file["iter_{index}".format(index=i)]["iter_{index}_veloc".format(index=i)]
        
        #plotFrame() or direct trajectory plotting


##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()
    
    with hdf.File(DATA_PATH + DATA_FILENAME,'r') as data_file:

        plot_trajectories(data_file)

        #Call other plotting functions from inside here

