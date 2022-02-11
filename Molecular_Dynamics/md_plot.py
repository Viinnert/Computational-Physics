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
import numpy as np
import h5py as hdf
import os

WORKDIR_PATH = os.getcwd()
DATA_PATH = WORKDIR_PATH + "/data/" 
DATA_FILENAME = "data.hdf5"


#Define Plotting functions here

##### Main function to be called at start

if __name__ == "__main__":
    def print_usage():
        print("Usage:\n", file=sys.stderr)
        print("Check out usage in README, you gave the wrong number of arguments", file=sys.stderr)

    required_args = 0
    if len(sys.argv) != required_args:
        print_usage()

    #Call plotting functions only from here