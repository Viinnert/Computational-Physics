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
from matplotlib import cm

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = WORKDIR_PATH + "/data/" 
DATA_FILENAME = "trajectories.hdf5"

# Initalize plot parameters
params = {
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": (3.15, 3.15),
    "figure.subplot.left": 0.14,
    "figure.subplot.right": 0.99,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.99,
    "figure.subplot.wspace": 0.15,
    "figure.subplot.hspace": 0.12,
    "lines.markersize": 6,
    "lines.linewidth": 2.0,
}
mpl.rcParams.update(params)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})

def plot_energy(data_file):
    """
    Plots the kinetic, potential and total energy of particles stored in a given data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the energy data
                                       Groups label iterations 'iter_{index}'
                                       ,groups should contain position and velocity datasets 
                                       named '{groupname}_pos' and '{groupname}_veloc' 

    Return
    - --
    """
    n_iterations = len(list(data_file.keys()))
    
    delta_t = data_file[f"iter_0"].attrs["delta_t"]
    
    time = np.arange(0, n_iterations)*delta_t
    pot_energy = [data_file[f"iter_{i}"][f"iter_{i}_kin_energy"] for i in range(1, n_iterations)]
    kin_energy = [data_file[f"iter_{i}"][f"iter_{i}_pot_energy"] for i in range(1, n_iterations)]
    tot_energy = pot_energy + kin_energy
    
    fig = plt.figure(figsize=(10,7.5))
    
    plt.plot(time, kin_energy, color="red", label="Kinetic Energy")
    plt.plot(time, pot_energy, color="blue", label="Potential Energy")
    plt.plot(time, tot_energy, color="black", label="Total Energy")
    plt.xlabel(r"Time $\sqrt{\frac{\sigma^2  m}{epsilon}}$")
    plt.ylabel(r"Energy")
    plt.legend()
    plt.show()


def plot_trajectories3D(data_file):
    pass


def plot_trajectories2D(data_file):
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
    n_iterations = len(list(data_file.keys()))
    n_atoms, n_dim = data_file["iter_1"]["iter_1_pos"].shape
    
    fig = plt.figure(figsize=(10,7.5))
    

    #Plot trajectories iteration-wise
    for i in range(1,  n_iterations):
        plt.clf() # Clear figure / redraw
        
        #Get arrays from the data file in shape (n_atoms x n_dim) 
        current_pos = data_file[f"iter_{i}"][f"iter_{i}_pos"]
        current_veloc = data_file[f"iter_{i}"][f"iter_{i}_veloc"]
        
        cmap = cm.rainbow(np.linspace(0, 1, current_pos[:,1].shape[0])) 
        plt.scatter(current_pos[:,0], current_pos[:,1], c=cmap)
        
        canvas_size = data_file[f"iter_{i}"].attrs["canvas_size"]
        plt.ylim(0, canvas_size[0])
        plt.xlim(0, canvas_size[1])

        plt.pause(0.5)

    plt.show()
    

##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()
    
    with hdf.File(DATA_PATH + DATA_FILENAME,'r') as data_file:
        
        plot_trajectories2D(data_file)

        #Call other plotting functions from inside here

