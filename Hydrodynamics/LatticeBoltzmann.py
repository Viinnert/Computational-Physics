import numpy as np
import matplotlib.pyplot as plt

class LatticeBoltzmann():

    def __init__(self, Nx, Ny, init_method):
        ''' Initialise simulation '''
        self.Nx = Nx
        self.Ny = Ny
        self.init_method = init_method

        self.direction_vectors = np.array([[-1,1], [0,1], [1,1],
                                      [-1,0], [0,0], [1,0],
                                      [-1,-1], [0,-1], [1,-1]])
                                      
        self.weights = np.array([1/36, 1/9, 1/36,
                                 1/9, 4/9, 1/9,
                                 1/36, 1/9, 1/36])

        self.speed_of_sound = 1/np.sqrt(3)

    def initialise_field(self):
        ''' Initialises field for first timestep '''
        if self.init_method == 'random':
            field = np.random.random((self.Nx, self.Ny, 9))

        elif self.init_method == '':
            pass
        
        return field

    def get_properties(self, field):
        ''' Calculates properties like density/velocity/eq_field for a given field '''
        densities = np.sum(field, axis=2)

        velocities = np.zeros((self.Nx, self.Ny, 2))
        for i in range(self.Nx): # TO DO: improve efficiency
            for j in range(self.Ny):
                velocity = field[i,j,:,None] * self.direction_vectors
                velocity = np.sum(velocity, axis=0)
                velocity /= densities[i,j]
                velocities[i,j] = velocity

        eq_field = np.zeros((self.Nx, self.Ny, 9))
        for i in range(self.Nx): # TO DO: improve efficiency
            for j in range(self.Ny):
                term1 = np.dot(self.direction_vectors, velocities[i,j]) / self.speed_of_sound**2
                term2 = np.dot(self.direction_vectors, velocities[i,j])**2 / 2*self.speed_of_sound**4
                term3 = np.dot(velocities[i,j], velocities[i,j]) / 2*self.speed_of_sound**2
                eq_field[i,j] = densities[i,j] * self.weights * (1 + term1 + term2 - term3)

        return densities, velocities, eq_field

    def step(self, field, eq_field, delta_t):
        ''' Makes one step using the BGK method '''
        # Collision step
        field += (eq_field - field) / delta_t

        # Streaming step
        new_field = np.zeros((self.Nx, self.Ny, 9))
        for i in range(9):
            new_field[:,:,i] = np.roll(field[:,:,i], self.direction_vectors[i], axis=(0,1))

        return new_field

    def plot(self, densities, velocities):
        ''' Plots the densities and, optionaly, the velocities '''
        x_values = np.arange(self.Nx)
        y_values = np.arange(self.Ny)

        plt.pcolormesh(x_values, y_values, densities)
        plt.draw()
        plt.pause(0.2)

    def simulate(self, N_timesteps, delta_t):
        ''' Run simulation '''
        field = self.initialise_field()

        for t in range(N_timesteps):
            densities, velocities, eq_field = self.get_properties(field)
            field = self.step(field, eq_field, delta_t)
            self.plot(densities, velocities)