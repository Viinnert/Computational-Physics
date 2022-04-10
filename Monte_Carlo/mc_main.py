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

from cmath import e
from statistics import correlation
from time import time
import numpy as np
import h5py as hdf
import os
import sys
#from sklearn import neighbors
#from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import timeit
from numba import int64, float64
from numba.experimental import jitclass


def save_results(results, data_file_path):
    with hdf.File(data_file_path, "w") as datafile:
        for output_kind in results.keys():
            datagroup = datafile.create_group(output_kind)
            for key in results[output_kind].keys():
                data = (results[output_kind])[key]
                if isinstance(data, dict):
                    datagroup.create_dataset(key+'_mean', data=data['mean'])
                    datagroup.create_dataset(key+'_std', data=data['std'])
                else:
                    print(type(data), data, key, output_kind)
                    datagroup.create_dataset(key, data=data)

                    
                    
class_spec = [
    ('energy', float64),
    ('magnetization', float64),
    ('specific_heat', float64),
    ('susceptibility', float64), 
    ('computation_time', float64),
    ('config', float64[:, :]),
    ('lattice_size', int64)
]

#@jitclass(class_spec)
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

    def initialstate(self):   
        """ 
        generates a random spin configuration for initial condition
        """
        self.config = 2*np.random.randint(2, size=(self.lattice_size,self.lattice_size))-1
        #state = 2*np.random.randint(2, size=(self.lattice_size,self.lattice_size))-1

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

    def get_energy(self, config):
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

    def get_mag(self, config):
        """
        Returns the magnetization of a given configuration
        """    
        return np.sum(config)
    
    def get_correlations(self, mag_per_time):
        tmax = mag_per_time.shape[0]
        
        #Evaluate autocorrelation only in valid t range of length t_max-1
        #NOTE: -1 because no valid variance (yet) with single measurement point 
        t_array = np.arange(0, tmax-1) 
        
        #Create an array with each collumn = all valid values of t_prime for specific t
        t_prime_array =  np.flip(np.triu(np.tile(t_array, (t_array.shape[0], 1)).T, k=0), axis=1)
        
        norm_array = 1/(tmax - t_array)
        
        #Retarded time = add to each t_prime array (collumn) the corresponding t value
        retard_idx = (t_prime_array + t_array[np.newaxis, :])#.flatten()
        non_retard_idx = t_prime_array#.flatten()

        mag_retard = mag_per_time[retard_idx]
        mag_non_retard = mag_per_time[non_retard_idx]
        
        zero_indices = np.tril_indices(retard_idx.shape[0], k=-1) #Mask for resetting zero sum terms
        (mag_retard[:, ::-1])[zero_indices] = 0
        (mag_non_retard[:, ::-1])[zero_indices] = 0
        
        mag_squared = mag_non_retard * mag_retard
        
        #print(mag_retard)
        
        expect_of_product = norm_array * np.sum(mag_squared, axis=0)
        product_of_expect = (norm_array * np.sum(mag_retard, axis=0)) \
                            * (norm_array * np.sum(mag_non_retard, axis=0))
        #expect_of_product = norm_array * np.sum((mag_per_time[retard_idx] * mag_per_time[non_retard_idx]).reshape(t_prime_array.shape), axis=0)
        #product_of_expect = (norm_array * np.sum(mag_per_time[retard_idx].reshape(t_prime_array.shape), axis=0)) \
        #                    * (norm_array * np.sum(mag_per_time[non_retard_idx].reshape(t_prime_array.shape), axis=0))

        return expect_of_product - product_of_expect
        
        
    def get_correlation_time(self, correlations):
        """
        Returns the correlation time after system equilibriation
        """ 
        return np.sum(correlations)/correlations[0]

    def get_specific_heat(self, energies, temp, block_size):
        energies = np.array(energies)
        specific_heat = {}

        excessive_times = len(energies) % block_size
        energies = energies[excessive_times:]

        n_blocks = int( len(energies) / block_size )
        energies = np.reshape(energies, (n_blocks, block_size))

        specific_heats = np.mean(energies**2, axis=1) - np.mean(energies, axis=1)**2
        specific_heats /= (temp**2 * self.lattice_size**2)

        specific_heat['mean'] = np.mean(specific_heats)
        specific_heat['std'] = np.std(specific_heats)

        return specific_heat

    def get_susceptibility(self, magnetizations, temp, block_size):
        magnetizations = np.array(magnetizations)
        susceptibility = {}

        excessive_times = len(magnetizations) % block_size
        magnetizations = magnetizations[excessive_times:]

        n_blocks = int( len(magnetizations) / block_size )
        magnetizations = np.reshape(magnetizations, (n_blocks, block_size))

        susceptibilities = np.mean(magnetizations**2, axis=1) - np.mean(magnetizations, axis=1)**2
        susceptibilities /= (temp * self.lattice_size)

        susceptibility['mean'] = np.mean(susceptibilities)
        susceptibility['std'] = np.std(susceptibilities)

        return susceptibility 
    
    def thermal_average(self, input_per_time, temp):
        output_per_time = {}
        
        output_per_time['mean'] = np.mean(input_per_time) / self.lattice_size**2
        output_per_time['std'] = np.sqrt((2 * self.correlation_time / (len(input_per_time)-1) ) * np.std(input_per_time / self.lattice_size**2) )
        
        return output_per_time

    
    def __time_sweep__(self, temp, time_steps_or_stop_crit):
        """ 
        Insert docstring here
        #Temperature in units J/k_B
        """
        
        inv_temp = 1.0 / temp 

        #Calculate quantities after equilibriation
        output = {}
        output['energy_per_time'], output['energy_squared_per_time'] = np.array([]), np.array([])
        output['magnetization_per_time'], output['magnetization_squared_per_time'] = np.array([]), np.array([])
        output['correlation_per_time'] = np.array([np.nan])
        
        if isinstance(time_steps_or_stop_crit, int):
            time_steps = time_steps_or_stop_crit
            for t in range(time_steps):
                self.config = self.mcmove(self.config, inv_temp) # Monte Carlo moves

                config_energy = self.get_energy(self.config) # calculate the energy in units J
                config_mag = self.get_mag(self.config) # calculate the magnetisation

                output['energy_per_time'] = np.append(output['energy_per_time'], config_energy)   
                output['magnetization_per_time'] = np.append(output['magnetization_per_time'], config_mag)
                output['energy_squared_per_time'] = np.append(output['energy_squared_per_time'], config_energy**2)
                output['magnetization_squared_per_time'] = np.append(output['magnetization_squared_per_time'], config_mag**2)
            
            # Only calculate correlation function after last step:
            config_corr = self.get_correlations(output['magnetization_per_time'])
            output['correlation_per_time'] = config_corr #Update refined+extended correlation function
            return output
        
        elif callable(time_steps_or_stop_crit):
            stop_crit = time_steps_or_stop_crit
            corr_calc_steps = 0
            while(stop_crit(output) == False or corr_calc_steps<2*10):
                corr_calc_steps +=1
                
                self.config = self.mcmove(self.config, inv_temp) # Monte Carlo moves

                config_energy = self.get_energy(self.config) # calculate the energy in units J
                config_mag = self.get_mag(self.config) # calculate the magnetization

                output['energy_per_time'] = np.append(output['energy_per_time'], config_energy)   
                output['magnetization_per_time'] = np.append(output['magnetization_per_time'], config_mag)
                output['energy_squared_per_time'] = np.append(output['energy_squared_per_time'], config_energy**2)
                output['magnetization_squared_per_time'] = np.append(output['magnetization_squared_per_time'], config_mag**2)
                      
                if corr_calc_steps > 1: 
                    #Calculate correlation to check for integration bound chi(t) < 0
                    config_corr = self.get_correlations(output['magnetization_per_time'] / (self.lattice_size**2))
                    output['correlation_per_time'] = config_corr #Update refined+extended correlation function  
                    print(f"Correlation {config_corr[-1]} in computation step {corr_calc_steps}")
                    print(f"computed with magnetization {config_mag}")    
            return output, corr_calc_steps

        else:
            print(callable(time_steps_or_stop_crit), type(time_steps_or_stop_crit), time_steps_or_stop_crit)
            raise ValueError("time_steps_or_stop_crit should be integer or callable")
    
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
        
        self.initialstate()

        #Minimize energy + Equilibrate:
        equilib_output = self.__time_sweep__(temp, equilib_steps)
        
        #Measure correlation time:
        stop_crit = lambda output: bool(output['correlation_per_time'][-1] < -0.01)
        stop_crit = lambda output: True
        corr_time_output, corr_calc_steps = self.__time_sweep__(temp, stop_crit)
        self.correlation_time = self.get_correlation_time(corr_time_output['correlation_per_time'])
        
        #Sample around non-zero prob. equilibrium config / actual mc sampling:
        output = self.__time_sweep__(temp, mc_steps)

        #Calculate observable expectation values:
        self.correlation = corr_time_output['correlation_per_time']
        self.energy = self.thermal_average(output['energy_per_time'], temp) 
        self.magnetization = self.thermal_average(output['magnetization_per_time'], temp) 
        self.susceptibility = self.get_susceptibility(output['magnetization_per_time'], temp, int(16*self.correlation_time))
        self.specific_heat = self.get_specific_heat(output['energy_per_time'], temp, int(16*self.correlation_time)) 

        toc = timeit.default_timer()
        self.computation_time = toc-tic
        
        #Combine outputs into one:
        output_dic_list = [equilib_output, corr_time_output, output]
        total_output = {}
        for key in output.keys():
            th = np.hstack([d[key] for d in output_dic_list]).flatten()
            total_output[key] = th
            
        #Store and output meta_data
        total_output['time'] = np.arange(0, equilib_steps+corr_calc_steps+mc_steps)
        total_output['temperature'] = temp
        total_output['mc_steps'] = mc_steps
        total_output['lattice_size'] = self.lattice_size
        
        return total_output

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
        sweep_output['energy_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['magnetization_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['specific_heat_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['susceptibility_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['correlation_time_per_temp'] = np.zeros(n_temp_samples)

        #m is the temperature index
        for m in tqdm(range(len(temp_array))): 
            self.__simulate_mc__(temp_array[m], equilib_steps, mc_steps)
        
            (sweep_output['energy_per_temp']['mean'])[m] = self.energy['mean']
            (sweep_output['energy_per_temp']['std'])[m] = self.energy['std']
            (sweep_output['magnetization_per_temp']['mean'])[m] = self.magnetization['mean']
            (sweep_output['magnetization_per_temp']['std'])[m] = self.magnetization['std']
            (sweep_output['specific_heat_per_temp']['mean'])[m] = self.specific_heat['mean']
            (sweep_output['specific_heat_per_temp']['std'])[m] = self.specific_heat['std']
            (sweep_output['susceptibility_per_temp']['mean'])[m] = self.susceptibility['mean']
            (sweep_output['susceptibility_per_temp']['std'])[m] = self.susceptibility['std']
            (sweep_output['correlation_time_per_temp'])[m] = self.correlation_time
            
        return sweep_output



if __name__ == "__main__":
    def print_usage():
        print("You gave the wrong number of arguments for mc_main \n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("python mc_main.py [arg1] [arg2] ... [data filename] \n", file=sys.stderr)
        sys.exit()

    required_args = ["data filename"]
    #sys.argv = sys.argv[1:] #Remove default script path argument
    sys.argv = ["data.hdf5"]

    if len(sys.argv) != len(required_args):
        print_usage()
    
    WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = WORKDIR_PATH + "/data/" 
    DATA_FILENAME = sys.argv[-1]

    # set variables
    temp = 2.2
    equilib_steps = 2**10
    mc_steps = 2**10

    temp_min = 1.5
    temp_max = 3.5
    n_temp_samples = 2**0

    #Initialize and do single time sweep at fixed temperature
    mc_Ising_model = Ising2D_MC()
    time_sweep_output = mc_Ising_model.__simulate_mc__(temp=temp,
                                                       equilib_steps=equilib_steps, 
                                                       mc_steps=mc_steps)
    
    #Reset and do temperature sweep
    mc_Ising_model = Ising2D_MC()
    temp_sweep_output = mc_Ising_model.__temp_sweep__(temp_min=temp_min, 
                                                      temp_max=temp_max,
                                                      equilib_steps=equilib_steps, 
                                                      n_temp_samples=n_temp_samples,
                                                      mc_steps=mc_steps)
    
    results = {'time_sweep_output': time_sweep_output,
               'temp_sweep_output': temp_sweep_output}
    save_results(results, DATA_PATH + DATA_FILENAME)

    print("Success")
    
