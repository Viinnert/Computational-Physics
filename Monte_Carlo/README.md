# Second Project COP: Monte-Carlo Simulation

Second Project for Computational Physics 2022: 
Monte-Carlo Simulation: Monte-Carlo simulation of the the Ising
model 

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