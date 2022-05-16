"""
###############################################################################
Part of the Kelvin Helmholtz Instabilities project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)
################################################################################
Defines the plotting functions and code used for the report
Classes:
-
Functions:
- 
################################################################################
"""
from cProfile import label
from cmath import isnan
import os
import sys
from time import sleep
from timeit import repeat
import numpy as np
import h5py as hdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors
from matplotlib.patches import FancyArrowPatch


# Initalize plot parameters
params = {
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": (10, 10),
    "figure.subplot.left": 0.14,
    "figure.subplot.right": 0.99,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.99,
    "figure.subplot.wspace": 0.15,
    "figure.subplot.hspace": 0.12,
    "lines.markersize": 6,
    "lines.linewidth": 2.0,
    'animation.html': 'html5',
}
mpl.rcParams.update(params)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})


def plot_D2Q9_init_map(init_map, x_coord, y_coord):
    fig,ax = plt.subplots(figsize=(15,8))
    
    #ax.set_xlim(150, 155)
    #ax.set_ylim(100, 150) 

    frame = np.sum(init_map, axis=-1)
    print(frame)
    print(init_map[:,:,5])
    
    delta_t = 1.0
    delta_s = (1.0, 1.0) #delta_x, delta_y
    lattice_flow_vecs = np.array([np.array([0,0]),
                            *[np.array([int(np.cos((i-1)*np.pi/2)) * delta_s[0]/delta_t,int(np.sin((i-1)*np.pi/2)) * delta_s[1]/delta_t]) for i in range(1,5)],
                            *[np.array([int(np.sign(np.cos((2*j-9)*np.pi/4))) * delta_s[0]/delta_t, int(np.sign(np.sin((2*j-9)*np.pi/4))) * delta_s[1]/delta_t]) for j in range(5,9)],
                            ]).T
    
    veloc_vecs =  np.einsum('ij,klj->kli', lattice_flow_vecs, init_map) / frame[:, :, np.newaxis]

    x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
    ax.pcolormesh(x_mesh, y_mesh, frame.T, vmin=0.0, vmax=8.0,cmap='twilight_shifted')
    plt.draw()
    #print(veloc_vecs[:,:,0])
    #ax.quiver(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, frame.T,  norm=colors.Normalize(vmin=np.min(frame.flatten()), vmax=np.max(frame.flatten())),cmap='twilight_shifted')
    ax.streamplot(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, density=2, color=frame.T, norm=colors.Normalize(vmin=np.min(frame.flatten()), vmax=np.max(frame.flatten())), cmap='twilight_shifted')
    plt.colorbar(cm.ScalarMappable(cmap='twilight_shifted'), ax=ax)

    plt.show()

def plot_D2Q9_density_flow(data_file_path):
    '''
    Plots the density of a time-evoluted density on the lattice over time
    
    Parameters
  
  ----------
    data_file_path : str
        File path to data file containing the time evoluted density
    Returns
    -------
    None.
    '''
    time, density_over_time, velocity_over_time = np.array([]), np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        density_over_time = np.array(data['density_per_time'])
        velocity_over_time = np.array(data['net_velocity_per_time'])
        print(density_over_time.shape, velocity_over_time.shape)

    delta_t = time[1] - time[0]
    
    fig,ax = plt.subplots(figsize=(16,9))
    
    v = (np.min(density_over_time.flatten()), np.max(density_over_time.flatten()))
    print("v: ", v)
    v = (0.0, 6.0)
    plt.colorbar(cm.ScalarMappable(cmap='plasma', norm=colors.Normalize(vmin=v[0], vmax=v[1])), ax=ax)

    #ax.set_xlim(150, 155)
    #ax.set_ylim(100, 150) 

    
    
    for (it, t) in enumerate(time):
        frame = density_over_time[it, :, :]
        veloc_vecs = velocity_over_time[it, :, :, :]
        x_coord = np.arange(density_over_time.shape[1])
        y_coord = np.arange(density_over_time.shape[2])
        x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
        ax.pcolormesh(x_mesh, y_mesh, frame.T, vmin=v[0], vmax=v[1],cmap='plasma')
        plt.draw()
        print(np.min(frame))
        #ax.quiver(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, frame.T,  norm=colors.Normalize(vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten())),cmap='twilight_shifted')
        ax.streamplot(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, density=2, color='blue') #color=frame.T, cmap='twilight_shifted')
        ax.set_title(f"Density at time {t}")
        plt.pause(0.2)
        ax.cla()
        #fig.clear()
    sleep(100)

def animate_D2Q9_density_flow(data_file_path):
    '''
    Animates the density of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted density

    Returns
    -------
    None.

    '''
    time, density_over_time, velocity_over_time = np.array([]), np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        density_over_time = np.array(data['density_per_time'])
        velocity_over_time = np.array(data['net_velocity_per_time'])
        print(density_over_time.shape, velocity_over_time.shape)

    delta_t = time[1] - time[0]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7,7))
    
    ax1.set_xlim((0, density_over_time.shape[1]))
    ax1.set_ylim((0, density_over_time.shape[2]))
    ax2.set_xlim((0, density_over_time.shape[1]))
    ax2.set_ylim((0, density_over_time.shape[2])) 
    
    x_coord = np.arange(density_over_time.shape[1])
    y_coord = np.arange(density_over_time.shape[2])
    x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
    
    density_plt = ax1.pcolormesh(x_mesh, y_mesh, np.zeros(x_mesh.shape), vmin=np.min(density_over_time[0,:,:].flatten()), vmax=np.max(density_over_time[0, :,:].flatten()), cmap='plasma')
    #velocity_plt = ax.quiver(x_mesh, y_mesh, np.zeros(x_mesh.shape), np.zeros(x_mesh.shape), np.zeros(x_mesh.shape), norm=colors.Normalize(vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten())),cmap='twilight_shifted')
    global velocity_plt
    velocity_plt = ax2.streamplot(x_mesh, y_mesh, np.zeros(x_mesh.shape), np.zeros(y_mesh.shape)) #color=[[]], density=1, norm=colors.Normalize(vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten())),cmap='twilight_shifted')
    
    def animate(it):
        t = time[it]
        frame = density_over_time[it, :, :]
        veloc_vecs = velocity_over_time[it, :, :, :]
        
        plt.suptitle(f"Density at time t = {t}")
        #density_plt.set_array(frame.T)
        ax1.cla()
        density_plt = ax1.pcolormesh(x_mesh, y_mesh, frame.T, vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten()), cmap='plasma')
    
        #velocity_plt.set_UVC(veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, frame.T)
        #velocity_plt.set_UVC(veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T)
        #return density_plt
    
    def animate_streamplot(it):
        ax2.cla()

        frame = density_over_time[it, :, :]
        veloc_vecs = velocity_over_time[it, :, :, :]
        
        #plt.suptitle(f"Density at time t = {t}")
        #density_plt.set_array(frame.T)
        #velocity_plt.set_UVC(veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T, frame.T)
        #velocity_plt.set_UVC(veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T)
        
        velocity_plt = ax2.streamplot(x_mesh, y_mesh, veloc_vecs[:,:,0].T, veloc_vecs[:,:,1].T) #color=[[]], density=1, norm=colors.Normalize(vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten())),cmap='twilight_shifted')
    
    def init_animation():
        return animate(0)
    
    def init_streamplot_animation():
        return animate_streamplot(0)
    
    #anim = animation.FuncAnimation(fig, animate, init_func=init_animation,frames=time.shape[0], interval=200, blit=True, repeat=False)
    anim = animation.FuncAnimation(fig, animate, init_func=init_animation,frames=time.shape[0], interval=80, blit=False, repeat=False)
    anim2 = animation.FuncAnimation(fig, animate_streamplot, init_func=init_streamplot_animation,frames=time.shape[0], interval=80, blit=False, repeat=False)
    
    plt.colorbar(cm.ScalarMappable(cmap='plasma',  norm=colors.Normalize(vmin=np.min(density_over_time.flatten()), vmax=np.max(density_over_time.flatten()))), ax=ax1)
    
    data_path = (data_file_path[::-1].split('/', maxsplit=1)[-1])[::-1] + '/'
    #anim.save(data_path+'test_animation.gif', writer='imagemagick', fps=2)
    #anim2.save(data_path+'test_animation2.gif', writer='imagemagick', fps=2)
    
    plt.show()

def plot_D2Q9_pressure(data_file_path):
    '''
    Plots the pressure of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted pressure

    Returns
    -------
    None.

    '''
    time, pressure_over_time = np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        pressure_over_time = np.array(data['pressure_per_time'])
        
    delta_t = time[1] - time[0]
    
    fig = plt.figure()
    
    for (it, t) in enumerate(time):
        pressure_frame = pressure_over_time[it, :, :]
        x_coord = np.arange(pressure_over_time.shape[1])
        y_coord = np.arange(pressure_over_time.shape[2])
        x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
        plt.pcolormesh(x_mesh, y_mesh, pressure_frame.T, vmin=np.min(pressure_over_time.flatten()), vmax=np.max(pressure_over_time.flatten()), cmap='plasma')
        plt.draw()
        plt.title(f"Pressure at time {t}")
        plt.colorbar()
        plt.pause(0.1)
        fig.clear()
    sleep(100)


def animate_D2Q9_pressure(data_file_path):
    '''
    Animates the pressure of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted pressure

    Returns
    -------
    None.

    '''
    time, pressure_over_time = np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        pressure_over_time = np.array(data['pressure_per_time'])
        
    delta_t = time[1] - time[0]
    
    x_coord = np.arange(pressure_over_time.shape[1])
    y_coord = np.arange(pressure_over_time.shape[2])
    x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
        
    pressure_plt = ax.pcolormesh(x_mesh, y_mesh, np.zeros(x_mesh.shape),vmin=np.min(pressure_over_time.flatten()), vmax=np.max(pressure_over_time.flatten()), cmap='plasma')
        
    fig, ax = plt.subplots()
    
    ax.set_xlim((0, pressure_over_time.shape[1]))
    ax.set_ylim((0, pressure_over_time.shape[2]))
    
    def animate(it):
        t = time[it]
        pressure_frame = pressure_over_time[it, :, :]
        plt.suptitle(f"Pressure at time {t}")
        pressure_plt.set_array(pressure_frame.T)
        return pressure_plt
        
    def init_animation():
        return animate(0)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init_animation,frames=time.shape[0], interval=200, blit=True, repeat=False)
    
    plt.colorbar(cm.ScalarMappable(cmap='plasma'), ax=ax)
    
    data_path = (data_file_path[::-1].split('/', maxsplit=1)[-1])[::-1] + '/'
    anim.save(data_path+'test_pressure_animation.gif', writer='imagemagick', fps=2)
    plt.show()


def plot_D2Q9_velocity_profile(data_file_path):
    '''
    Plots the density of a time-evoluted density on the lattice over time
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing the time evoluted density

    Returns
    -------
    None.

    '''
    time, density_over_time, velocity_over_time = np.array([]), np.array([]), np.array([])

    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        velocity_over_time = np.array(data['net_velocity_per_time'])
        print(density_over_time.shape, velocity_over_time.shape)

    delta_t = time[1] - time[0]
    
    fig, ((ax_xx, ax_xy), (ax_yx, ax_yy)) = plt.subplots(nrows=2, ncols=2)
    
    x_coord = np.arange(velocity_over_time.shape[1])
    y_coord = np.arange(velocity_over_time.shape[2])

    for (it, t) in enumerate(time[::5]):
        veloc_vec_xx, veloc_vec_xy = np.mean(velocity_over_time[it, :, :, 0], axis=1), np.mean(velocity_over_time[it, :, :, 1], axis=1)
        veloc_vec_yx, veloc_vec_yy = np.mean(velocity_over_time[it, :, :, 0], axis=0), np.mean(velocity_over_time[it, :, :, 1], axis=0)

        veloc_vec_yx = velocity_over_time[it, 3, :, 0]
        ax_xx.plot(x_coord, veloc_vec_xx, label=f"{t} s")
        ax_xy.plot(x_coord, veloc_vec_xy, label=f"{t} s")
        ax_yx.plot(y_coord, veloc_vec_yx, label=f"{t} s")
        ax_yy.plot(y_coord, veloc_vec_yy, label=f"{t} s")
    plt.legend()
    plt.show()


def plot_D2Q9_moments_vs_time(data_file_path):
    '''
    Plots the ....
    
    Parameters
    ----------
    data_file_path : str
        File path to data file containing ...

    Returns
    -------
    None.

    '''
    time,energy_over_time, mom_density_over_time, energy_flux_over_time, vis_stress_over_time = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    with hdf.File(data_file_path,'r') as data_file:
        data = data_file['time_sweep_output']
        time = np.array(data['time'])
        energy_over_time = np.array(data['energy_per_time']) 
        mom_density_over_time = np.array(data['mom_density_per_time'])
        energy_flux_over_time = np.array(data['energy_flux_per_time'])
        vis_stress_over_time = np.array(data['vis_stress_per_time'])
    
    fig, ((ax_11, ax_12), (ax_21, ax_22)) = plt.subplots(nrows=2, ncols=2)
    
    ax_11.plot(time, [np.sum(energy_over_time[it, :,:]) for (it,t) in enumerate(time) ])
    ax_12.plot(time, [np.sum(mom_density_over_time[it, :,:, 0]) for (it,t) in enumerate(time) ])
    ax_12.plot(time, [np.sum(mom_density_over_time[it, :,:, 1]) for (it,t) in enumerate(time) ])
    ax_21.plot(time, [np.sum(energy_flux_over_time[it, :,:, 0]) for (it,t) in enumerate(time) ])
    ax_21.plot(time, [np.sum(energy_flux_over_time[it, :,:, 1]) for (it,t) in enumerate(time) ])
    ax_22.plot(time, [np.sum(vis_stress_over_time[it, :,:, 0]) for (it,t) in enumerate(time) ])
    ax_22.plot(time, [np.sum(vis_stress_over_time[it, :,:, 1]) for (it,t) in enumerate(time) ])

    plt.show()
        