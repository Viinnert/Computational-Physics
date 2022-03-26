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
import sys
from sklearn import neighbors
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import timeit


def save_temp_sweep_results(sweep_output, data_file_path):
    with hdf.File(data_file_path, "w") as datafile:
        datafile['temperature'] = sweep_output['temperature']
        datafile['energy'] = sweep_output['energy_per_temp']
        datafile['magnetization'] = sweep_output['magnetization_per_temp']
        datafile['specific_heat'] = sweep_output['specific_heat_per_temp']
        datafile['susceptibility'] = sweep_output['susceptibility_per_temp']

class Ising2D_MC:
    '''
        Initializes the 2-D Ising model Monte Carlo simulation class

        Parameters
        ----------
        
        lattice_size : integer
            Number of spins along each lattice dimension
        equilib_steps : integer
            Number of MC sweeps for equilibration
        mc_steps : integer
            number of MC sweeps for actual calculation

        Returns
        -------
        None.

        '''
    
    def __init__(self, lattice_size=2**4):
        #System related paramters
        self.lattice_size = lattice_size
        self.config = np.zeros((lattice_size, lattice_size))

    def initialstate(self, lattice_size):   
        """ 
        generates a random spin configuration for initial condition
        """
        state = 2*np.random.randint(2, size=(lattice_size,lattice_size))-1
        return state

    def mcmove(self, config, beta):
        """
        Monte Carlo move using Metropolis algorithm 
        """
        
        #Keep sampling during one unit of time: 
        #give any spin in lattice chance to flip in one unit of time
        for i in range(self.lattice_size**2):
            
            #Draw random spin to try a flip:
            random_x = np.random.randint(0, self.lattice_size)
            random_y = np.random.randint(0, self.lattice_size)
            random_spin =  config[random_x, random_y]
            
            neighbor_spin_sum = config[(random_x+1)%self.lattice_size, random_y] + config[random_x,(random_y+1)%self.lattice_size] + config[(random_x-1)%self.lattice_size, random_y] + config[random_x,(random_y-1)%self.lattice_size]
            
            #Flipping cost only depends on local energy change:
            cost = 2*random_spin*neighbor_spin_sum #Factor 2: Each spin pair should occur twice in energy sum
            
            if cost < 0:
                random_spin *= -1 #Flip
            elif np.random.rand() < np.exp(-cost*beta):
                random_spin *= -1 #Flip anyway
            config[random_x, random_y] = random_spin
        return config


    def calc_energy(self, config):
        """
        Energy of a given configuration
        
        Return
            Energy in units J
        """
        
        energy = 0
        for i in range(len(config)):
            for j in range(len(config)):
                spin = config[i,j]
                neighbor_spin_sum = config[(i+1)%self.lattice_size, j] + config[i,(j+1)%self.lattice_size] + config[(i-1)%self.lattice_size, j] + config[i,(j-1)%self.lattice_size]
                energy += -neighbor_spin_sum*spin
        return energy/4.0 #Remove quadruple counting of spin-spin neighbor pair in 2D


    def calc_mag(self, config):
        """
        Returns the magnetization of a given configuration
        """    
        return np.sum(config)
    
    def __time_sweep__(self, temp, time_steps):
        """ 
        Insert docstring here
        #Temperature in units J/k_B
        """
        
        inv_temp = 1.0 / temp 

        #Calculate quantities after equilibriation
        output = {}
        output['cum_energy'] = output['cum_energy_squared'] = 0
        output['cum_mag'] = output['cum_mag_squared'] = 0
        
        for t in range(time_steps):
            self.config = self.mcmove(self.config, inv_temp) # Monte Carlo moves

            config_energy = self.calc_energy(self.config)     # calculate the energy in units J
            config_mag = self.calc_mag(self.config)        # calculate the magnetisation

            output['cum_energy'] += config_energy                 #Cumulative energy 
            output['cum_mag'] += config_mag
            output['cum_mag_squared'] += config_mag*config_mag
            output['cum_energy_squared'] += config_energy*config_energy
        
        return output
            
    def __simulate_mc__(self, temp, equilib_steps=2**10, mc_steps=2**10):   
        """ 
        Insert docstring here
        #Temperature in units J/k_B
        """
        #Reset observables:
        self.energy         = None  
        self.magnetization  = None
        self.specific_heat  = None
        self.susceptibility = None

        tic = timeit.default_timer() #Start to measure computation time
        
        self.config = self.initialstate(self.lattice_size)
        
        inv_temp = 1.0 / temp 
        inv_temp_squared = inv_temp*inv_temp

        #Minimize energy + Equilibrate:
        self.__time_sweep__(temp, equilib_steps)
        
        #Sample around non-zero prob. equilibrium config:
        output = self.__time_sweep__(temp, mc_steps)
        
        #Weights for MC averages:
        n1, n2  = 1.0/(mc_steps*self.lattice_size*1), 1.0/(mc_steps*mc_steps*self.lattice_size*1)  

        #Calculate observable expectation values:
        self.energy         = n1*output['cum_energy']   
        self.magnetization  = n1*output['cum_mag']
        self.specific_heat  = (n1*output['cum_energy_squared'] - n2*output['cum_energy']*output['cum_energy'])*inv_temp_squared
        self.susceptibility = (n1*output['cum_mag_squared'] - n2*output['cum_mag']*output['cum_mag'])*inv_temp_squared

        toc=timeit.default_timer()
        self.computation_time = toc-tic

    
    def __temp_sweep__(self, temp_min=1.5, temp_max=3.5,  n_temp_samples = 2**4, equilib_steps=2**10, mc_steps=2**10):
        '''
        Performs a temperature sweep over MC simulations for the defined Ising spin system. 

        Parameters
        ----------
        
        temp_min : float
            Minimum temperature (units J/k_B) to be considered
        temp_max : float
            Maximum temperature (units J/k_B) to be considered
        n_temp_samples : integer
            Number of temperature points to be considered

        Returns
        -------
        None.

        '''
        
        temp_array = np.linspace(temp_min, temp_max, n_temp_samples)
        
        #tm is the transition temperature
        sweep_output = {}
        sweep_output['temperature'] = temp_array
        sweep_output['energy_per_temp']       = np.zeros(n_temp_samples)
        sweep_output['magnetization_per_temp']  = np.zeros(n_temp_samples)
        sweep_output['specific_heat_per_temp'] = np.zeros(n_temp_samples)
        sweep_output['susceptibility_per_temp'] = np.zeros(n_temp_samples)

        #m is the temperature index
        for m in tqdm(range(len(temp_array))): 
            self.__simulate_mc__(temp_array[m], equilib_steps, mc_steps)
        
            sweep_output['energy_per_temp'][m]         = self.energy
            sweep_output['magnetization_per_temp'][m]  = self.magnetization
            sweep_output['specific_heat_per_temp'][m]  = self.specific_heat
            sweep_output['susceptibility_per_temp'][m] = self.susceptibility
        return sweep_output



if __name__ == "__main__":
    def print_usage():
        print("You gave the wrong number of arguments for mc_main \n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("python mc_main.py [arg1] [arg2] ... [data filename] \n", file=sys.stderr)
        sys.exit()

    required_args = ["data filename"]
    sys.argv = sys.argv[1:] #Remove default script path argument
    
    if len(sys.argv) != len(required_args):
        print_usage()
    
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = WORKDIR_PATH + "/data/" 
    DATA_FILENAME = sys.argv[-1]

    mc_Ising_model = Ising2D_MC()
    
    sweep_output = mc_Ising_model.__temp_sweep__()
    save_temp_sweep_results(sweep_output, DATA_PATH + DATA_FILENAME)
    
    print("Success")
    
