# First Project COP: Molecular Dynamics (MD)

First Project for Computational Physics 2022: 
Molecular Dynamics (MD): Molecular Dynamics simulation of different phases of matter.

Including simulation of Argon molecules in Lennard-Jones potential using periodic boundary conditions
.....

## Authors
Vincent Krabbenborg (Studentnummer = XXXXXXX)
Thomas Rothe (Studentnummer = 1930443)

## Usage

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
