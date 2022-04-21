# Second Project COP: Monte-Carlo Simulation

Second Project for Computational Physics 2022: 
Monte-Carlo Simulation: Monte-Carlo simulation of the the Ising
model 

## Authors
Vincent Krabbenborg (Studentnummer = 2029189)
Thomas Rothe (Studentnummer = 1930443)

## Usage
This folder contains the following files.

- data: This folder contains the .hdf5 data file to which the simulation results are saved.
- mc_main.py: This python script contains the main functions of Monte-Carlo simulation.
- mc_plot.py: This python script contains some functions to plot the results of the simulation saved in the data file.


## Reproducing results:

mc_main.py and mc_plot.py are set this way that just running these scripts would reproduce the results from the report.

Run the two scripts in the main folder by: (The provision of a filename is optional!)

python mc_main.py [data.hdf5]
python mc_plot.py [data.hdf5]

Each recreate the temperature sweep from 1.0 to 4.0 with steps of 0.2 automatically. Additionally data and plots for a single temperature are produced. This temperature can be adjusted by changing the "temp" variable at the end of the "mc_main.py" script.


## Running a custom simulation

One can initialise a model using the Ising2D_MD() class. The simulation is run using one of the two functions:

- \_\_simulate_mc\_\_(): This functions simulates the model for a given temperature and returns a dictionary containing all the physical properties.
- \_\_temp_sweep\_\_(): This function simulated the model for a multiple given temperatures and returns a dictionary containing all the physical properties for each temperatute.

The results must be saved to the correct data file using the save_results() function.

To make plots of the data you will need the two functions in mc_plot.py.

- plot_expectation_vs_time()
- plot_expectation_vs_temp()

By calling those functions with the data file the plots will pop up automatically.
