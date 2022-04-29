from LatticeBoltzmann import LatticeBoltzmann

# variables
Nx = 100
Ny = 100
delta_t = 0.5
N_timesteps = 20
init_method = 'random' # 'random' or ...

# Simulation
simulation = LatticeBoltzmann(Nx, Ny, init_method)
simulation.simulate(N_timesteps, delta_t)