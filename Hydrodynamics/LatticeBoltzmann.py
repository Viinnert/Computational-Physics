import numpy as np

class LatticeBoltzmann():

    def __init__(self, Nx, Ny, init_method):
        ''' Initialise simulation '''
        self.Nx = Nx
        self.Ny = Ny
        self.init_method = init_method

        self.unit_vectors = np.array([[-1,1], [0,1], [1,1],
                                      [-1,0], [0,0], [1,0],
                                      [-1,-1], [0,-1], [1,-1]])
                                      
        self.weights = np.array([1/36, 1/9, 1/36,
                                 1/9, 4/9, 1/9,
                                 1/36, 1/9, 1/36])

        self.speeds_of_sound = 1/np.sqrt(3)

    def initialise_field(self):
        ''' Initialises field at first timestep '''
        if self.init_method == 'random':
            field = np.random.random((self.Nx, self.Ny, 9))

        elif self.init_method == '':
            pass
        
        return field

    def get_properties(self, field):
        ''' Calculated properties like density/velocity/eq_field for a given field '''
        densities = np.sum(field, axis=2)

        velocities = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx): # improve efficiency maybe?
            for j in range(self.Ny):
                velocity = field[i,j,:,None] * self.unit_vectors
                velocity = np.sum(velocity, axis=0)
                velocity /= densities[i,j]
                velocities[i,j] = velocity

        eq_field = None # TO DO: implement

        return densities, velocities, eq_field

    def step(self, field, densities, velocities, eq_field, delta_t):
        # Collision step
        # TO DO: implement
        
        # Streaming step
        # TO DO: implement
        pass #return new_field

    def plot(self, densities, velocities):
        pass

    def simulate(self, N_timesteps, delta_t):
        ''' Run simulation '''
        field = self.initialise_field()

        for t in range(N_timesteps):
            densities, velocities, eq_field = self.get_properties(field)
            field = self.step(field, densities, velocities, eq_field, delta_t)
            self.plot(densities, velocities)