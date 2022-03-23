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
from cmath import isnan
import os
import sys
import numpy as np
import h5py as hdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

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

def plot_forces(data_file):
    """
    Plots the forces on particles stored in a given data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the energy data
                                       Groups label iterations 'iter_{index}'
    """
    n_iterations = len(list(data_file.keys()))
    
    delta_t = data_file[f"iter_1"].attrs["delta_t"]
    
    time = np.arange(0, n_iterations-1)*delta_t
    
    forces_array = np.array([np.sum((np.array(data_file[f"iter_{i}"][f"iter_{i}_force"]))**2, axis=1) for i in range(1, n_iterations)])

    print(forces_array.shape)
    
    fig = plt.figure(figsize=(10,7.5))
    
    plt.plot(time, forces_array, label="Force")
    plt.xlabel(r"Time $\sqrt{\frac{\sigma^2  m}{epsilon}}$")
    plt.ylabel(r"Force")
    plt.show()


def plot_energy(data_file):
    """
    Plots the 3kinetic, potential and total energy of particles stored in a given data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the energy data
                                       Groups label iterations 'iter_{index}'
    """
    n_iterations = len(list(data_file.keys()))
    delta_t = data_file[f"iter_1"].attrs["delta_t"]
    time = np.arange(0, n_iterations-1)*delta_t
    
    
    pot_energy = np.array([np.sum(np.array(data_file[f"iter_{i}"][f"iter_{i}_pot_energy"])) for i in range(1, n_iterations)])
    kin_energy = np.array([np.sum(np.array(data_file[f"iter_{i}"][f"iter_{i}_kin_energy"])) for i in range(1, n_iterations)])
    tot_energy = pot_energy + kin_energy
    
    fig = plt.figure(figsize=(10,7.5))
    
    plt.plot(time, kin_energy, color="red", label="Kinetic Energy")
    plt.plot(time, pot_energy, color="blue", label="Potential Energy")
    plt.plot(time, tot_energy, color="black", label="Total Energy")
    
    try:
        kin_energy_target = data_file[f"iter_1"].attrs["kin_energy_target"]
        plt.hlines(kin_energy_target, xmin=0, xmax=time[-1], label='Kinetic energy target', linestyle='--')
    except:
        pass
    
    plt.xlabel(r"Time $\sqrt{\frac{\sigma^2  m}{epsilon}}$")
    plt.ylabel(r"Energy")
    plt.legend()
    plt.show()



def animate_trajectories2D(data_file):
    """
    animates the trajectories of particles stored in a given 2D data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the trajectory data
                                       Groups label iterations 'iter_{index}'
                                       ,groups should contain position and velocity datasets 
                                       named '{groupname}_pos' and '{groupname}_veloc' 
    """
    n_iterations = len(list(data_file.keys()))
    n_atoms, n_dim = data_file["iter_1"]["iter_1_pos"].shape
    
    fig = plt.figure(figsize=(10,7.5))

    #Plot trajectories iteration-wise
    for i in range(1, n_iterations):
        #plt.clf() # Clear figure / redraw
        
        #Get arrays from the data file in shape (n_atoms x n_dim) 
        current_pos = data_file[f"iter_{i}"][f"iter_{i}_pos"]
        
        cmap = cm.rainbow(np.linspace(0, 1, current_pos[:,1].shape[0])) 
        
        current_point_scatter = plt.scatter(current_pos[:,0], current_pos[:,1], c=cmap, marker="o", s=54)
        
        canvas_size = data_file[f"iter_{i}"].attrs["canvas_size"]
        plt.ylim(0, canvas_size[0])
        plt.xlim(0, canvas_size[1])

        plt.pause(0.02)
        
        past_point_scatter = plt.scatter(current_pos[:,0], current_pos[:,1], c=cmap, marker=".", s=6)
        current_point_scatter.remove()
        
    plt.show()
    


def animate_trajectories3D(data_file):
    """
    animates the trajectories of particles stored in a given 3D data file.
    
    Args
    - data_file::h5py._hl.files.File = File object from which to extract the trajectory data
                                       Groups label iterations 'iter_{index}'
                                       ,groups should contain position and velocity datasets 
                                       named '{groupname}_pos' and '{groupname}_veloc' 
    """
    n_iterations = len(list(data_file.keys()))
    n_atoms, n_dim = data_file["iter_1"]["iter_1_pos"].shape
    
    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(projection='3d')

    #Plot trajectories iteration-wise
    for i in range(1, n_iterations):
        #ax.cla() # Clear figure / redraw
        
        #Get arrays from the data file in shape (n_atoms x n_dim) 
        current_pos = data_file[f"iter_{i}"][f"iter_{i}_pos"]
        
        cmap = cm.rainbow(np.linspace(0, 1, current_pos[:,1].shape[0])) 
        current_point_scatter = ax.scatter(current_pos[:,0], current_pos[:,1], current_pos[:,2], c=cmap, marker="o", s=54)
        
        canvas_size = data_file[f"iter_{i}"].attrs["canvas_size"]
        ax.set_xlim(0, canvas_size[0])
        ax.set_ylim(0, canvas_size[1])
        ax.set_zlim(0, canvas_size[2])

        plt.pause(0.2)
        
        #Remove current position pointer and add trail to plot
        past_point_scatter = ax.scatter(current_pos[:,0], current_pos[:,1], current_pos[:,2], c=cmap, marker=".", s=6)
        current_point_scatter.remove()
    
    plt.show()


def plot_av_histogram(histogram_list, bin_edges):
    '''
    Plots (the average of a) given (array of) histogram(s) 

    Args
    - histogram_list::ndarray = array of histogram's counts
    - bin_edges::ndarray = array of the histogram's bin edges
    '''
    av_histogram = np.mean(histogram_list, axis=0)
    
    # Plot histogram
    plt.figure(figsize=(10,7.5))
    plt.hist(bin_edges[:-1], bin_edges, weights=av_histogram)
    plt.xlabel('Distance (m)', fontsize=15)
    plt.ylabel(r'g(r)', fontsize=15)
    plt.show()
    
##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()
    
    global WORKDIR_PATH 
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    global DATA_PATH 
    DATA_PATH = WORKDIR_PATH + "/data/" 
    global DATA_FILENAME 
    DATA_FILENAME = "trajectories.hdf5" 

    
    with hdf.File(DATA_PATH + DATA_FILENAME,'r') as data_file:
        
        animate_trajectories3D(data_file)
        #plot_trajectories2D(data_file)
