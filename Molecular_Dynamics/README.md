# First Project COP: Molecular Dynamics (MD)

First Project for Computational Physics 2022: 
Molecular Dynamics (MD): Molecular Dynamics simulation of different phases of matter.

Including simulation of Argon molecules in Lennard-Jones potential using periodic boundary conditions
.....

## Authors
Vincent Krabbenborg (Studentnumber = s2029189)

Thomas Rothe (Studentnumber = s1930443)

## Usage

This folder consist of the following folders:

- _pycache_:
- data: The data generated from the simulation is saved to this folder in hdf5 format and used for analysis. 
- tests: This folder contains python scripts which result in pre-programmed experiment and produces figure which are used in the report. 

and the following files:

- md_main: This python script contains the primary simulation class and some extra essential functions for the simulation.
- md_plot: This python script contains functions to analyse and/or plot the data generated from the simumlation.

### Run simulation:

To run a simulation you must import "md_main".

To initialise the simulation you must call the "Simulation()" class with some chosen parameters. The initialisation mode "init_mode" can be set either "random", "fcc" or some self made function which returns 2 (#particles, #dimension) dimensional numpy arrays using the python lambda function

To run the simulation call "Simulation.__simulate__()" with some parameter values. The simulation will now run and save the data to the "data" folder. It will also print the pressure of the system.

### Analyse simulation

After succesfully running a simulation you can analyse it and make various plots using the "md_plot" python script. The following functions are available in "md_plot":

- plot_forces():
- plot_energy():
- animate_trajectories2D():
- animate_trajectories3D():

- To make a plot of the pair correlation the "get_pair_correlation()" function from the "md_main" script is required. Call this function using the data file and it returns the counts and bin edges of the histogram. If you want to run multiple simulations and average their pair correlation hisotgrams, use the "plot_av"histogram()" function from the "md_plot" script. This function requires a list of histogram lists from the "get_pair_correlation()" function.


```
</li>
<li> (Re)Initialize simulation environment and run:

```python

#Do (re)intialization of parameters (if required)
#Call main simulation function to run simulation

```
</li>
<li>
Plot results from data file at ./data/ directory using plotting script
```python

import os

include('./md_plot.py')
DATA_PATH = os.getcwd() + "/data/" + "DATA_FILENAME.hdf5"

#Call plotting function

```
</li>
</ul> 
