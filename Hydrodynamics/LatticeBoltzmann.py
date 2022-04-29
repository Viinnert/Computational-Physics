import numpy as np

class LatticeBoltzmann():

    def __init__(self, Nx, Ny, delta_t, N_timesteps, init_method):
        ''' Initialise simulation '''
        self.Nx = Nx
        self.Ny = Ny
        self.delta_t = delta_t
        self.N_timesteps = N_timesteps
        self.init_method = init_method

        self.unit_vectors = np.array([[-1,1], [0,1], [1,1],
                                      [-1,0], [0,0], [1,0],
                                      [-1,-1], [0,-1], [1,-1]])

    def step(self):
        pass

    def plot(self):
        pass

    def simulate(self):
        ''' Run simulation '''
        if self.init_method == 'random':
            field = np.random.random((self.Nx, self.Ny, 9, self.N_timesteps))

        elif self.init_method == '':
            pass

        density = np.sum(field, axis=2)
