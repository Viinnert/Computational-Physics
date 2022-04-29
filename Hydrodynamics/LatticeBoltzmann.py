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
                                      
        self.weights = np.array([1/36, 1/9, 1/36,
                                 1/9, 4/9, 1/9,
                                 1/36, 1/9, 1/36])

    def initialise_field(self):
        if self.init_method == 'random':
            field = np.random.random((self.Nx, self.Ny, 9))

        elif self.init_method == '':
            pass
        
        return field

    def step(self):
        pass

    def plot(self):
        pass

    def simulate(self):
        ''' Run simulation '''
        field = self.initialise_field()

        densities = np.sum(field, axis=2)

        velocities = np.zeros((self.Nx, self.Ny, 9))
        for i in range(self.Nx): # improve efficiency maybe?
            for j in range(self.Ny):
                velocity = field[i,j,:,None] * self.unit_vectors
                velocity = np.sum(velocity, axis=0)
                velocities[i,j] = velocity
