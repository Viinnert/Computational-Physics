from LatticeBoltzmann import LatticeBoltzmann

# variables
Nx = 100
Ny = 100
delta_t = 0.5
N_timesteps = 5
init_method = 'random' # 'random' or ...

# Simulation
simulation = LatticeBoltzmann(Nx, Ny, init_method)
#field = simulation.initialise_field()
#densities, velocities, eq_field = simulation.get_properties(field)
#new_field = simulation.step(field, eq_field, delta_t)
#simulation.plot(densities, velocities)
simulation.simulate(N_timesteps, delta_t)