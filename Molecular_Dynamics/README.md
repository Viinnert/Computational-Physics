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
- tests: This folder contains python scripts which result in pre-programmed experiment and produces figure which are used in the report. -> Just run those python files for figure reproduction of the report.

and the following files:

- md_main: This python script contains the primary simulation class and some extra essential functions for the simulation.
- md_plot: This python script contains functions to analyse and/or plot the data generated from the simumlation.

### Run simulation from scratch:

FOR FIGURE REPRODUCTION OF THE REPORT SEE "Pre-made experiments" FUTHER DOWN BELOW INSTEAD!

To run a simulation you must import "md_main".

To initialise the simulation you must call the "Simulation()" class with some chosen parameters. The initialisation mode "init_mode" can be set either "random", "fcc" or some self made function which returns 2 (#particles, #dimension) dimensional numpy arrays (one for the positions and one for the velocities) using the python lambda function

To run the simulation call "Simulation.__simulate__()" with some parameter values. The simulation will now run and save the data to the "data" folder. It will also print the pressure of the system.

An example:

```python
N_DIM = 3 
N_UNIT_CELLS = (3,3,3) 
TEMPERATURE = 3. 
DENSITY = 0.3  
ATOM_MASS = 6.6335e-26 
POT_ARGS = {'sigma': 3.405e-10, 'epsilon': 119.8}

CANVAS_ASPECT_RATIO = (1,1,1) 
END_OF_TIME = 2

DELTA_T = 0.01
N_ITERATIONS = int(END_OF_TIME / DELTA_T)
    
INIT_MODE = "fcc"

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.insert(1, WORKDIR_PATH)
DATA_PATH = WORKDIR_PATH + "data/" 
DATA_FILENAME = "trajectories_fcc.hdf5"
    
sim = Simulation(n_atoms_or_unit_cells=N_UNIT_CELLS, 
                 atom_mass=ATOM_MASS, 
                 density=DENSITY, 
                 temperature=TEMPERATURE,
                 n_dim=N_DIM, 
                 canvas_aspect_ratio=CANVAS_ASPECT_RATIO, 
                 pot_args=POT_ARGS, 
                 init_mode=INIT_MODE, 
                 data_path=DATA_PATH, 
                 data_filename=DATA_FILENAME)
    
sim.__simulate__(n_iterations=N_ITERATIONS, delta_t=DELTA_T)
```

### Analyse simulation

After succesfully running a simulation you can analyse it and make various plots using the "md_plot" python script. The following functions are available in "md_plot":

- plot_forces():
- plot_energy():
- animate_trajectories2D():
- animate_trajectories3D():

- To make a plot of the pair correlation the "get_pair_correlation()" function from the "md_main" script is required. Call this function using the data file and it returns the counts and bin edges of the histogram. If you want to run multiple simulations and average their pair correlation hisotgrams, use the "plot_av"histogram()" function from the "md_plot" script. This function requires a list of histogram lists from the "get_pair_correlation()" function.

An example (after having run the simulation code above):
```python
with hdf.File(DATA_PATH + TRAJ_DATA_FILENAME,'r') as data_file:
     animate_trajectories3D(data_file)
     plot_energy(data_file)
     plot_forces(data_file)
```

### Pre-made experiments in the "tests" folder

The test folder contains the following pre-made experiment. Nothing has to be set, the scripts can just be run and it will produce some figures automatically. e.g.
```bash
cd Molecular_Dynamics
python tests/XXXX.py
```

- md_test_gasphase_argon.py [report figure 5a and 5b]: This script simulates argon atoms in fcc positions. The temperature and density are set to the gas phase. After simulation it shows an animation of the movement of the particles. After this it calculates the average pair correlation histogram over 4 seperate simulations.

- md_test_liquidphase_argon.py [report figure 6a and 6b]: This script simulates argon atoms in fcc positions. The temperature and density are set to the liquid phase. After simulation it shows an animation of the movement of the particles. After this it calculates the average pair correlation histogram over 4 seperate simulations.

- md_test_solidphase_argon.py [report figure 7a and 7b]: This script simulates argon atoms in fcc positions. The temperature and density are set to the solid phase. After simulation it shows an animation of the movement of the particles. After this it calculates the average pair correlation histogram over 4 seperate simulations.

- md_test_paircor_random.py: This script simulates argon atoms in random initialised positions, the temperature and density are set to the gas phase, but you might play with other values for temperature and density. After simulation it shows an animation of the movement of the particles. After this it calculates the average pair correlation histogram over 6 seperate simulations.

- md_test_initialcondit_fcc.py: This script simulates argon atoms in random initialised positions, the temperature and density are set to the gas phase, but you might play with other values for temperature and density. After simulation it shows an animation of the movement of the particles. No pair correlation is calculated afterwards.

- md_test_dynamics.py [report figure 4a and 4b]: This script runs a simple 2D simulation of 2 particles moving towards each other to showcase to working of the implemented dynamics, i.e. a simple colllision.

- md_energy_convergence.py [report figure 1]: Demonstrates energy conservation and convergence to kinetic energy target from random initialization.

