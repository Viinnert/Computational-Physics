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
from time import sleep, time
import numpy as np
import h5py as hdf
import os
import sys
from sympy import true
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

class Ising2D_MC:
    '''
    Defines a 2-dimensional Ising model simulation.
    '''
    def __init__(self, lattice_size=2**5):
        '''
        Initializes the Ising model

        Parameters
        ----------
        lattice_size : Integer
            The amount of spins per dimension.

        Returns
        -------
        '''
        self.lattice_size = lattice_size
        self.config = np.zeros((lattice_size, lattice_size))

    def initialstate(self):   
        '''
        generates a random spin configuration for initial condition

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.config = 2*np.random.randint(2, size=(self.lattice_size,self.lattice_size))-1

    def mcmove(self, config, beta):
        '''
        Monte-Carlo move using Metropolis algorithm.

        Parameters
        ----------
        config: ndarray
            Current Ising model configuration
        beta: float
            1/K_b*T, the reciprocal temperature of the system
        Returns
        -------
        config: ndarray
            New Ising model configuration
        convergence_rate: float
            The convergence rate of the system
        '''
        a_cost = np.array([])
        b_cost = np.array([])
        n_rejections = 0

        # Keep sampling during one unit of time: 
        # give any spin in lattice chance to flip in one unit of time
        for i in range(self.lattice_size**2):
            
            # Draw random spin to try a flip:
            random_x = np.random.randint(0, self.lattice_size)
            random_y = np.random.randint(0, self.lattice_size)
            random_spin =  config[random_x, random_y]
            
            neighbor_spin_sum = config[(random_x+1)%self.lattice_size, random_y] + config[random_x,(random_y+1)%self.lattice_size] + config[(random_x-1)%self.lattice_size, random_y] + config[random_x,(random_y-1)%self.lattice_size]
            
            # Flipping cost only depends on local energy change:
            # Negative sign in energy expression and spin flip cancel
            cost = 2*random_spin*neighbor_spin_sum # Factor 2: Each spin pair should occur twice in energy sum
 
            if cost < 0:
                random_spin *= -1
            else:
                n_rejections += 1
                if np.random.rand() < np.exp(-cost*beta):
                    random_spin *= -1
                else:
                    pass # Change nothing / finally reject 
            
            # Update spin value
            config[random_x, random_y] = random_spin
        
        convergence_rate = n_rejections / (self.lattice_size**2)

        return config, convergence_rate
    
    def get_energy(self, config):
        '''
        Calculates energy of the given configuration

        Parameters
        ----------
        config: ndarray
            Current Ising model configuration

        Returns
        -------
        energy: float
            The energy of the given configuration
        '''
        energy = 0
        for i in range(len(config)):
            for j in range(len(config)):
                spin = config[i,j]
                neighbor_spin_sum = config[(i+1)%self.lattice_size, j] + config[i,(j+1)%self.lattice_size] + config[(i-1)%self.lattice_size, j] + config[i,(j-1)%self.lattice_size]
                energy += -neighbor_spin_sum*spin
        return energy/2.0 # Remove quadruple counting of spin-spin neighbor pair in 2D

    def get_mag(self, config):
        '''
        Calculates magnetization of the given configuration

        Parameters
        ----------
        config: ndarray
            Current Ising model configuration

        Returns
        -------
        magnetization: float
            magnetization of the given configuration
        '''   
        return np.sum(config)
    
     
    def get_correlations(self, mag_per_time):
        '''
        Calculates the correlation of the system's evolution through time

        Parameters
        ----------
        mag_per_time: ndarray
            magnetization of the system per timestep

        Returns
        -------
        correlations: ndarray
            the correlation values of the system per timestep
        '''
        tmax = mag_per_time.shape[0]
        
        # Evaluate autocorrelation only in valid t range of length t_max-1
        # NOTE: -1 because no valid variance (yet) with single measurement point 
        t_array = np.arange(0, tmax-1) 
        
        # Create an array with each collumn = all valid values of t_prime for specific t
        t_prime_array =  np.flip(np.triu(np.tile(t_array, (t_array.shape[0], 1)).T, k=0), axis=1)
        
        norm_array = 1/(tmax - t_array - 1)
        
        # Retarded time = add to each t_prime array (collumn) the corresponding t value
        retard_idx = (t_prime_array + t_array[np.newaxis, :])
        non_retard_idx = t_prime_array
        
        mag_retard = mag_per_time[retard_idx]
        mag_non_retard = mag_per_time[non_retard_idx]
        
        zero_indices = np.tril_indices(retard_idx.shape[0], k=-1) # Mask for resetting zero sum terms
        (mag_retard[:, ::-1])[zero_indices] = 0
        (mag_non_retard[:, ::-1])[zero_indices] = 0
          
        mag_squared = mag_non_retard * mag_retard
        
        expect_of_product = norm_array * np.sum(mag_squared, axis=0)
        product_of_expect = (norm_array * np.sum(mag_retard, axis=0)) \
                            * (norm_array * np.sum(mag_non_retard, axis=0))
        
        return expect_of_product - product_of_expect
        
        
    def get_correlation_time(self, correlations):
        '''
        calculates the correlation time of the given correlation values

        Parameters
        ----------
        correlations: ndarray
            correlation values of the system per timestep

        Returns
        -------
        corr_time: float
            the correlation time of the system
        '''
        first_neg_idx = np.where(correlations < 0.0)[0][0]
        corr_time = np.sum(correlations[:first_neg_idx])/correlations[0]
        corr_time_less = np.sum(correlations[:first_neg_idx-1])/correlations[0]
        return corr_time

    def get_specific_heat(self, energies, temp, block_size):
        '''
        Calculates the specific heat of the system

        Parameters
        ----------
        energies: ndarray
            energies of the system per timestep
        temp: float
            temperature of the system
        block_size: int
            the block averaging interval

        Returns
        -------
        specific_heat: dict
            the std and mean of the specific heat
        '''
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
        '''
        Calculates the magnetic susceptibilty of the system

        Parameters
        ----------
        magnetizations: ndarray
            magnetization of the system per timestep
        temp: float
            temperature of the system
        block_size: int
            the block averaging interval

        Returns
        -------
        susceptibility: dict
            the std and mean of the magnetic susceptibility
        '''
        magnetizations = np.array(magnetizations)
        susceptibility = {}

        excessive_times = len(magnetizations) % block_size
        magnetizations = magnetizations[excessive_times:]

        n_blocks = int( len(magnetizations) / block_size )
        magnetizations = np.reshape(magnetizations, (n_blocks, block_size))

        susceptibilities = np.mean(magnetizations**2, axis=1) - np.mean(magnetizations, axis=1)**2
        susceptibilities /= (temp * self.lattice_size**2)

        susceptibility['mean'] = np.mean(susceptibilities)
        susceptibility['std'] = np.std(susceptibilities)

        return susceptibility 
    
    def thermal_average(self, input_per_time, temp):
        output_per_time = {}
        
        output_per_time['mean'] = np.mean(input_per_time) / self.lattice_size**2
        output_per_time['std'] = np.sqrt((2 * self.correlation_time / (len(input_per_time)-1) ) * np.std(input_per_time / self.lattice_size**2) )
        
        return output_per_time

    
    def __time_sweep__(self, temp, time_steps_or_stop_crit):
        '''
        evolutes the system for a number of time steps 
        or until a certain stop criteria is met

        Parameters
        ----------
        temp: float
            the temperature of the system
        time_steps_or_stop_crit: int or callable
            The amount of time steps to evolute or the criteria needed to stop

        Returns
        -------
        output: dict
            a dictionary conaining the values of 
            various physical properties per timestep
        '''
        inv_temp = 1.0 / temp 

        # Calculate quantities after equilibriation
        output = {}
        output['energy_per_time'] = np.array([])
        output['magnetization_per_time'] = np.array([])
        output['correlation_per_time'] = np.array([])
        output['convergence_per_time'] = np.array([])
        
        if isinstance(time_steps_or_stop_crit, int):
            time_steps = time_steps_or_stop_crit
            
            for t in range(time_steps):
                self.config, convergence_rate = self.mcmove(self.config, inv_temp) # Monte Carlo moves

                config_energy = self.get_energy(self.config) # calculate the energy in units J
                config_mag = self.get_mag(self.config) # calculate the magnetisation

                output['energy_per_time'] = np.append(output['energy_per_time'], config_energy)   
                output['magnetization_per_time'] = np.append(output['magnetization_per_time'], config_mag)
                output['convergence_per_time'] = np.append(output['convergence_per_time'], convergence_rate)

            # Only calculate correlation function after last step:
            config_corr = self.get_correlations(output['magnetization_per_time'] / (self.lattice_size**2))
            output['correlation_per_time'] = config_corr # Update refined+extended correlation function
            return output
        
        elif callable(time_steps_or_stop_crit):
            stop_crit = time_steps_or_stop_crit
            
            min_corr_calc_steps = 2**8
            output['correlation_per_time'] = np.full((min_corr_calc_steps), fill_value=np.nan)
            
            corr_calc_steps = 0
            while(stop_crit(output) == False or corr_calc_steps<min_corr_calc_steps):
                corr_calc_steps +=1
                
                self.config, convergence_rate = self.mcmove(self.config, inv_temp) # Monte Carlo moves

                config_energy = self.get_energy(self.config) # calculate the energy in units J
                config_mag = self.get_mag(self.config) # calculate the magnetization

                output['energy_per_time'] = np.append(output['energy_per_time'], config_energy)   
                output['magnetization_per_time'] = np.append(output['magnetization_per_time'], config_mag)
                output['convergence_per_time'] = np.append(output['convergence_per_time'], convergence_rate)
     
                if corr_calc_steps > 1: 
                    # Calculate correlation to check for integration bound chi(t) < 0
                    config_corr = self.get_correlations(output['magnetization_per_time'] / (self.lattice_size**2))
                    output['correlation_per_time'] = config_corr # Update refined+extended correlation function  
                    print(f"Correlation {config_corr[-int(0.05*config_corr.shape[0])]} in computation step {corr_calc_steps}")
                    print(f"computed with magnetization {config_mag}")    
                
                if corr_calc_steps==min_corr_calc_steps:
                    break
            return output, corr_calc_steps

        else:
            print(callable(time_steps_or_stop_crit), type(time_steps_or_stop_crit), time_steps_or_stop_crit)
            raise ValueError("time_steps_or_stop_crit should be integer or callable")
    
    def __simulate_mc__(self, temp, equilib_steps=2**10, mc_steps=2**10):   
        '''
        Complete simulation of the system

        Parameters
        ----------
        temp: float
            the temperature of the system
        equilib_steps: int
            The amount of time steps to equilibriate the system
        mc_steps: int
            The amount of time steps to do measurements on after equilibriation

        Returns
        -------
        total_output: dict
            a dictionary conaining the values of 
            various physical properties per timestep
            for all phases of the simulation
        '''
        # Reset observables:
        self.energy         = None  
        self.magnetization  = None
        self.specific_heat  = None
        self.susceptibility = None

        tic = timeit.default_timer() # Start to measure computation time
        
        self.initialstate()

        # Minimize energy + Equilibrate:
        # equilib_stop_crit = lambda output: bool(output['convergence_per_time'][-1] > 0.99)
        # equilib_output, equilib_steps = self.__time_sweep__(temp, corr_stop_crit)
        equilib_output = self.__time_sweep__(temp, equilib_steps)
        
        # Measure correlation time:
        corr_stop_crit = lambda output: bool(output['correlation_per_time'][-int(0.05*output['correlation_per_time'].shape[0])] < 0.0)
        # corr_stop_crit = lambda output: True
        
        corr_time_output, corr_calc_steps = self.__time_sweep__(temp, corr_stop_crit)
        self.correlation_time = self.get_correlation_time(corr_time_output['correlation_per_time'])
        print("Corrtime = ", self.correlation_time)
        
        # Sample around non-zero prob. equilibrium config / actual mc sampling:
        output = self.__time_sweep__(temp, mc_steps)

        # Calculate observable expectation values:
        self.correlation = output['correlation_per_time']
        self.energy = self.thermal_average(output['energy_per_time'], temp) 
        self.magnetization = self.thermal_average(output['magnetization_per_time'], temp) 
        self.susceptibility = self.get_susceptibility(output['magnetization_per_time'], temp, int(16*self.correlation_time))
        self.specific_heat = self.get_specific_heat(output['energy_per_time'], temp, int(16*self.correlation_time)) 

        toc = timeit.default_timer()
        self.computation_time = toc-tic
        
        # Combine outputs into one:
        output_dic_list = [equilib_output, corr_time_output, output]
        total_output = {}
        for key in output.keys():
            th = np.hstack([d[key] for d in output_dic_list]).flatten()
            total_output[key] = th
        
        # Store and output meta_data
        total_output["correlation_per_time"] = output['correlation_per_time']
        total_output['time'] = np.arange(0, equilib_steps+corr_calc_steps+mc_steps)
        total_output['temperature'] = temp
        total_output['mc_steps'] = mc_steps
        total_output['corr_calc_steps'] = corr_calc_steps
        total_output['equilib_steps'] = equilib_steps
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
        equilib_steps: int
            The amount of time steps to equilibriate the system
        mc_steps: int
            The amount of time steps to do measurements on after equilibriation

        Returns
        -------
        sweep_output: dict
            a dictionary conaining the values of various physical properties per timestep
            for all phases of the simulation for all temperatures
        '''
        temp_array = np.linspace(temp_min, temp_max, n_temp_samples)
        
        # Collect & Store everything into a convienent collection for the whole temperature sweep
        sweep_output = {}        
        sweep_output['temperature'] = temp_array
        sweep_output['lattice_size'] = self.lattice_size
        sweep_output['equilib_steps_per_temp'] =  np.zeros(n_temp_samples)      
        sweep_output['corr_calc_steps_per_temp'] =  np.zeros(n_temp_samples)
        sweep_output['mc_steps_per_temp'] =  np.zeros(n_temp_samples)
        sweep_output['energy_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['magnetization_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['specific_heat_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['susceptibility_per_temp'] = {'mean': np.zeros(n_temp_samples), 'std': np.zeros(n_temp_samples)}
        sweep_output['correlation_time_per_temp'] = np.zeros(n_temp_samples)

        # m is the temperature index
        for m in tqdm(range(len(temp_array))): 
            single_output = self.__simulate_mc__(temp_array[m], equilib_steps, mc_steps)
        
            (sweep_output['equilib_steps_per_temp'])[m] = single_output['equilib_steps']      
            (sweep_output['corr_calc_steps_per_temp'])[m] = single_output['corr_calc_steps']
            (sweep_output['mc_steps_per_temp'])[m]  = single_output['mc_steps']
            (sweep_output['energy_per_temp']['mean'])[m] = self.energy['mean']
            (sweep_output['energy_per_temp']['std'])[m] = self.energy['std']
            (sweep_output['magnetization_per_temp']['mean'])[m] = self.magnetization['mean']
            (sweep_output['magnetization_per_temp']['std'])[m] = self.magnetization['std']
            (sweep_output['specific_heat_per_temp']['mean'])[m] = self.specific_heat['mean']
            (sweep_output['specific_heat_per_temp']['std'])[m] = self.specific_heat['std']
            (sweep_output['susceptibility_per_temp']['mean'])[m] = self.susceptibility['mean']
            (sweep_output['susceptibility_per_temp']['std'])[m] = self.susceptibility['std']
            (sweep_output['correlation_time_per_temp'])[m] = self.correlation_time
            if m == 0:
                sweep_output["convergence_per_time_per_temp"] = np.vstack((np.zeros(single_output['convergence_per_time'].shape), single_output['convergence_per_time']))
                sweep_output["magnetization_per_time_per_temp"] = np.vstack((np.zeros(single_output['magnetization_per_time'].shape), single_output['magnetization_per_time']))
            else:
                sweep_output["convergence_per_time_per_temp"] = np.vstack((sweep_output["convergence_per_time_per_temp"], single_output['convergence_per_time']))
                sweep_output["magnetization_per_time_per_temp"] = np.vstack((sweep_output["magnetization_per_time_per_temp"], single_output['magnetization_per_time']))
        
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

    # Derandomize the script:
    np.random.seed(43)

    # set variables
    temp = 3.8
    equilib_steps = 800
    mc_steps = 2**10

    temp_min = 1.0
    temp_max = 4.0
    n_temp_samples = int((temp_max - temp_min) / 0.2)

    # Initialize and do single time sweep at fixed temperature
    mc_Ising_model = Ising2D_MC()
    time_sweep_output = mc_Ising_model.__simulate_mc__(temp=temp,
                                                       equilib_steps=equilib_steps, 
                                                       mc_steps=mc_steps)
    
    # Reset and do temperature sweep
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
    
