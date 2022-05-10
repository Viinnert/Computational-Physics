"""
###############################################################################
Part of the Kelvin Helmholtz Instabilities project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)
################################################################################
Defines the plotting functions and code used for the report
Classes:
-
Functions:
- 
################################################################################
"""
from cProfile import label
from cmath import isnan
import os
import sys
from time import sleep
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
    "figure.figsize": (10, 10),
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

def plot_D2Q9_density_flow(data_file_path):
    '''
    Plots the density of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted density

    Returns
    -------
    None.

    '''
    time, density_over_time, velocity_over_time = np.array([]), np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        density_over_time = np.array(data['density_per_time'])
        velocity_over_time = np.array(data['net_velocity_per_time'])
        print(density_over_time.shape, velocity_over_time.shape)

    delta_t = time[1] - time[0]
    
    fig = plt.figure()
    
    for (it, t) in enumerate(time):
        frame = density_over_time[it, :, :]
        veloc_vecs = velocity_over_time[it, :, :, :]
        x_coord = np.arange(density_over_time.shape[1])
        y_coord = np.arange(density_over_time.shape[2])
        x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
        plt.pcolormesh(x_mesh, y_mesh, frame.T, vmin=0, vmax=5, cmap='twilight')
        plt.draw()
        print(veloc_vecs[:,:,0])
        plt.quiver(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, frame.T, cmap='bone')
        #plt.streamplot(x_mesh, y_mesh, veloc_vecs[:,:,0], veloc_vecs[:,:,1], density=1, color=frame, cmap='bone')
        plt.title(f"Density at time {t}")
        plt.colorbar()
        plt.pause(0.1)
        fig.clear()
    sleep(100)

def plot_D2Q9_pressure(data_file_path):
    '''
    Plots the pressure of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted pressure

    Returns
    -------
    None.

    '''
    time, pressure_over_time = np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        pressure_over_time = np.array(data['pressure_per_time'])
        
    delta_t = time[1] - time[0]
    
    fig = plt.figure()
    
    for (it, t) in enumerate(time):
        pressure_frame = pressure_over_time[it, :, :]
        x_coord = np.arange(pressure_over_time.shape[1])
        y_coord = np.arange(pressure_over_time.shape[2])
        x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
        plt.pcolormesh(x_mesh, y_mesh, pressure_frame.T, vmin=0, vmax=5, cmap='twilight')
        plt.draw()
        plt.title(f"Pressure at time {t}")
        plt.colorbar()
        plt.pause(0.1)
        fig.clear()
    sleep(100)

def plot_D2Q9_velocity_profile(data_file_path):
    '''
    Plots the density of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted density

    Returns
    -------
    None.

    '''
    time, density_over_time, velocity_over_time = np.array([]), np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        velocity_over_time = np.array(data['net_velocity_per_time'])
        print(density_over_time.shape, velocity_over_time.shape)

    delta_t = time[1] - time[0]
    
    fig, ((ax_xx, ax_xy), (ax_yx, ax_yy)) = plt.subplots(nrows=2, ncols=2)
    
    x_coord = np.arange(velocity_over_time.shape[1])
    y_coord = np.arange(velocity_over_time.shape[2])

    for (it, t) in enumerate(time[::5]):
        veloc_vec_xx, veloc_vec_xy = np.mean(velocity_over_time[it, :, :, 0], axis=1), np.mean(velocity_over_time[it, :, :, 1], axis=1)
        veloc_vec_yx, veloc_vec_yy = np.mean(velocity_over_time[it, :, :, 0], axis=0), np.mean(velocity_over_time[it, :, :, 1], axis=0)

        veloc_vec_yx = velocity_over_time[it, 3, :, 0]
        ax_xx.plot(x_coord, veloc_vec_xx, label=f"{t} s")
        ax_xy.plot(x_coord, veloc_vec_xy, label=f"{t} s")
        ax_yx.plot(y_coord, veloc_vec_yx, label=f"{t} s")
        ax_yy.plot(y_coord, veloc_vec_yy, label=f"{t} s")
    plt.legend()
    plt.show()
    