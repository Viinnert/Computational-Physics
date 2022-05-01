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

def density_plot(data_file):
    '''
    Plots the density of a time-evoluted density flow map
    
    Parameters
    ----------
    data_file : hd5py
        File object

    Returns
    -------
    None.

    '''
    x_values = np.arange(self.Nx)
    y_values = np.arange(self.Ny)

    plt.pcolormesh(x_values, y_values, densities, vmin=0, vmax=5)
    plt.draw()
    plt.pause(0.01)
