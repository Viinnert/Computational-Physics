from LatticeBoltzmann import LatticeBoltzmann

# variables
Nx = 1000
Ny = 1000
delta_t = 0.5
N_timesteps = 100
initialisation_method = 'random' # 'random' or ...

# Simulation

simulation = LatticeBoltzmann(Nx, Ny, delta_t, N_timesteps)