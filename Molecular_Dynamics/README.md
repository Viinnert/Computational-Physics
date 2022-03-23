# First Project COP: Molecular Dynamics (MD)

First Project for Computational Physics 2022: 
Molecular Dynamics (MD): Molecular Dynamics simulation of different phases of matter.

Including simulation of Argon molecules in Lennard-Jones potential using periodic boundary conditions
.....

## Authors
Vincent Krabbenborg (Studentnummer = s2029189)

Thomas Rothe (Studentnummer = s1930443)

## Usage

This folder consist of the following folders:

- _pycache_:
- data: The data generated from the simulation is saved to this folder in hdf5 format and used for analysis. 
- tests: This folder contains python scripts which result in pre-programmed experiment and produces figure which are used in the report. 

and the following files:

- md_main: This python script contains the primary simulation class and some extra essential functions for the simulation.
- md_plot: This python script contains functions to analyse and/or plot the data generated from the simumlation.

### Run simulation:

<ul>
<li> Include main file in new python script
```python

include('./md_main.py')

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
