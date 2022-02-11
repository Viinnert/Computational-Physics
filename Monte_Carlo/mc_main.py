"""
###############################################################################

Part of the Monte-Carlo Simulation project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)

################################################################################

Defines and sets up monte carlo simulation for Ising spins

Classes:
- --

Functions:
- ---


################################################################################
"""

import numpy as np
import h5py as hdf
import os


WORKDIR_PATH = os.getcwd()
DATA_PATH = WORKDIR_PATH + "/data/" 
DATA_FILENAME = "data.hdf5"


#Do something
