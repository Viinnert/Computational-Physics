from LatticeBoltzmann import LatticeBoltzmann

# variables
Nx = 100
Ny = 100
delta_t = 0.5
N_timesteps = 10
init_method = 'test' # 'random' or 'test

# Simulation
simulation = LatticeBoltzmann(Nx, Ny, init_method)
simulation.simulate(N_timesteps, delta_t)