# Third Project COP: Kelvin-Helmholtz Instabilities

Third Project for Computational Physics 2022: 
Kelvin Helmholtz Instabilties: Fluid Dynamics simulation of fluid with a density boundary

## Authors
Vincent Krabbenborg (Studentnumber = s2029189)

Thomas Rothe (Studentnumber = s1930443)

## Usage
!! All scripts require python 3.7+ !!

This main folder contains the core scripts with routines for simulating and plotting the Lattice-Boltzmann model.

Running simulations is only possible via 'experiments'-scripts to be found in the experiments folder.

### Pre-Configured experiments
There are 4 pre-configured experiment scripts:
- basic_mixing.py: Simple situation set-up of two layers of the same fluid with different densities and pressures to show that the core evolution steps of collision and streaming indeed lead to mixing of the two layers over long time periods.
- basic_pertub.py: Single density fluid layer with a sinusiodal perturbation initially to show how the velocity oscillations decay through the collision / equilibriation.
- basic_kelvin_helmholtz.py: Main experiment file that tries to set-up and simulate kelvin-helmholtz instabilities for a fluid with two different densities along the y-axis
- boundary_perturb_exp.py: Outdated experiment which sets up a boundary and introduces a slight perturbation in the boundary layer.

and a last script: pure_data_plotting.py enables the plotting of previously runned and stored simulations.

All those scripts can just be run as in any other python environment with python version 3.7+.

### Creating experiments from scratch:
Make a python script inside the experiments folder and make sure to import all routines properly by including the header:

```python
from mimetypes import init
import sys
import os
import numpy as np

np.random.seed(42) #Fix seed for reproducebility.

EXPERIMENTS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
WORKDIR_PATH = EXPERIMENTS_PATH + "../"

sys.path.insert(1, WORKDIR_PATH)

from kh_main import *
from kh_plot import *
```

There needs to be defined a initialization function for creating the initial map, taking the 'lattice_size', 'lattice_flow_vecs', 'mom_space_transform', 'inv_mom_space_transform' as arguments and returning a map in the correct density flow map format.

Next choose a discrete velocty model to simulate the system in. Supported are currently only D2Q9 and D2Q16. Create a corresponding instance:
```python
DensityFlowMap.D2Q16
```

and a lattice boltzmann instance as well. 
Finally run a time sweep and extract the results via the "__run__()" method

```python
results = LatticeBoltzmann.__run__(...)
```
