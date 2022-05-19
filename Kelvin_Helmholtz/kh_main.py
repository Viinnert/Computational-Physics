"""
###############################################################################
Part of the Kelvin Helmholtz Instabilities project for the Computational physics course 2022.
By Vincent Krabbenborg (XXXXXXX) & Thomas Rothe (1930443)
################################################################################
Defines and sets up 
Classes:
- 
Functions:
- 
################################################################################
"""

from dataclasses import dataclass, InitVar, field
from itertools import product
from multiprocessing.sharedctypes import Value
from typing import Dict, Tuple, List, Union, Callable
import os
import sys
import h5py as hdf
from time import sleep
import numpy as np
import scipy as sp
from scipy.fftpack import shift
from sympy import true
from tqdm import tqdm

def save_to_file(results, data_file_path):
    with hdf.File(data_file_path, "w") as datafile:
        for output_kind in results.keys():
            datagroup = datafile.create_group(output_kind)
            for key in results[output_kind].keys():
                data = (results[output_kind])[key]
                if isinstance(data, dict):
                    datagroup.create_dataset(key+'_mean', data=data['mean'])
                    datagroup.create_dataset(key+'_std', data=data['std'])
                else:
                    #print(type(data), data.shape, key, output_kind)
                    datagroup.create_dataset(key, data=data)


def semi_periodic_BCs(c_map):
    #Reverse at y=-1 towards +y-direction
    c_map[:,-1,2] += c_map[:,-1,4]
    c_map[:,-1,8] += np.roll(c_map[:,-1,5], shift=1) #/\
    c_map[:,-1,7] += np.roll(c_map[:,-1,6], shift=-1) #/\
    c_map[:,-1,[4, 5, 6]] = 0
    
    #Reverse at y=0 the 3 flows towards -y-direction
    c_map[:,0,4] += c_map[:,0,2]
    c_map[:,0,6] += np.roll(c_map[:,0,7], shift=-1) #/\
    c_map[:,0,5] += np.roll(c_map[:,0,8], shift=1) #/\
    c_map[:,0,[2,7,8]] = 0
    
    return c_map       


def D2Q16_set_nonconserv_moments(c_mom_map):
    density = c_mom_map[:,:, 0] 
    mom_density_x, mom_density_y = c_mom_map[:,:, 1], c_mom_map[:,:, 2]
    energy = c_mom_map[:,:, 3]

    #Set the remaining non-conserved moments:
    c_mom_map[:,:, 4] = (mom_density_x**2 - mom_density_y**2) / density
    c_mom_map[:,:, 5] = mom_density_x*mom_density_y / density
    c_mom_map[:,:, 6] = (energy + density*energy)*mom_density_x / density
    c_mom_map[:,:, 7] = (energy + density*energy)*mom_density_y / density
    c_mom_map[:,:, 8] = (mom_density_x**2 - 3*(mom_density_y**2))*mom_density_x / (density**2)
    c_mom_map[:,:, 9] = (3*(mom_density_x**2) - mom_density_y**2)*mom_density_y / (density**2)
    c_mom_map[:,:, 10] = 2*(energy**2) / density - (( mom_density_x**2 + mom_density_y**2)**2 / (4*density**3))
    c_mom_map[:,:, 11] = np.zeros(density.shape)
    c_mom_map[:,:, 12] = (6*density*energy - 2*(mom_density_x**2) - 2*(mom_density_y**2))*(mom_density_x**2 - mom_density_y**2)/(density**3)
    c_mom_map[:,:, 13] = (6*density*energy - 2*(mom_density_x**2) - 2*(mom_density_y**2))*(mom_density_x*mom_density_y)/(density**3)
    c_mom_map[:,:, 14] = np.zeros(density.shape)
    c_mom_map[:,:, 15] = np.zeros(density.shape)
    
    return c_mom_map


def D2Q9_collision_matrix(c_densities, c_velocities,  relaxation_coeffs, equilib_coeffs):
    coll_mat = np.zeros((*c_densities.shape, *relaxation_coeffs.shape, *relaxation_coeffs.shape)) #4D array
    coll_mat[:,:,1,0] = (relaxation_coeffs[2-1]*equilib_coeffs['alpha2'] / 4) * c_densities
    coll_mat[:,:,2,0] = (relaxation_coeffs[3-1]*equilib_coeffs['alpha3'] / 4) * c_densities
    
    coll_mat[:,:,1,1] = -relaxation_coeffs[2-1]
    
    coll_mat[:,:,2,2] = -relaxation_coeffs[3-1]
    
    coll_mat[:,:,1,3] = (relaxation_coeffs[2-1]*equilib_coeffs['gamma2']*c_velocities[:,:,0] / 3) / c_densities
    coll_mat[:,:,2,3] = (relaxation_coeffs[3-1]*equilib_coeffs['gamma4']*c_velocities[:,:,0] / 3) / c_densities
    coll_mat[:,:,4,3] = relaxation_coeffs[5-1]*equilib_coeffs['c1'] / 2
    coll_mat[:,:,7,3] = (3*relaxation_coeffs[8-1]*equilib_coeffs['gamma1']*c_velocities[:,:,0]) / c_densities
    coll_mat[:,:,8,3] = (3*relaxation_coeffs[9-1]*equilib_coeffs['gamma3']*c_velocities[:,:,1] / 2 ) / c_densities
    
    coll_mat[:,:,4,3] = -relaxation_coeffs[5-1]
    
    coll_mat[:,:,1,4] = (relaxation_coeffs[2-1]*equilib_coeffs['gamma2']*c_velocities[:,:,1] / 3) / c_densities
    coll_mat[:,:,2,4] = (relaxation_coeffs[3-1]*equilib_coeffs['gamma4']*c_velocities[:,:,1] / 3) / c_densities
    coll_mat[:,:,6,4] = relaxation_coeffs[7-1]*equilib_coeffs['c1'] / 2
    coll_mat[:,:,7,4] = (-3*relaxation_coeffs[8-1]*equilib_coeffs['gamma1']*c_velocities[:,:,1]) / c_densities
    coll_mat[:,:,8,4] = (3*relaxation_coeffs[9-1]*equilib_coeffs['gamma3']*c_velocities[:,:,0] / 2) / c_densities
    
    coll_mat[:,:,6,6] = -relaxation_coeffs[7-1]
    coll_mat[:,:,7,7] = -relaxation_coeffs[8-1]
    coll_mat[:,:,8,8] = -relaxation_coeffs[9-1]
    
    return coll_mat

def D2Q9_advection_matrix(lattice_flow_vecs, lattice_size):
    #Advection operator considering periodic BC's:
    
    #Initialize total matrix by first lattice flow vector [0, 0] -> identity
    advec_mat_x = np.identity(lattice_size[0])[:,:, np.newaxis]
    advec_mat_y = np.identity(lattice_size[1])[:,:, np.newaxis]
    
    for i in range(1, lattice_flow_vecs.shape[1]):
        lfvec = lattice_flow_vecs[:,i]
        loc_advec_x = np.roll(np.identity(lattice_size[0]), shift=int(lfvec[0]),axis=0)
        advec_mat_x = np.concatenate((advec_mat_x, loc_advec_x[:,:,np.newaxis]), axis=2)
        loc_advec_y = np.roll(np.identity(lattice_size[1]), shift=int(lfvec[1]), axis=1)
        advec_mat_y = np.concatenate((advec_mat_y, loc_advec_y[:,:,np.newaxis]), axis=2)
 
    assert advec_mat_x.shape == (lattice_size[0], lattice_size[0], lattice_flow_vecs.shape[1])
    assert advec_mat_y.shape == (lattice_size[1], lattice_size[1], lattice_flow_vecs.shape[1])
    
    return (advec_mat_x, advec_mat_y)

@dataclass(frozen=False)
class DensityFlowMap:
    '''
    Defines a model map of density flow on a lattice used for the lattice boltzmann model
    '''
    n_dim: int
    lattice_size: Tuple[int]
    lattice_type: str
    delta_t: float
    delta_s: float
    lattice_flow_vecs: np.ndarray
    mom_space_transform: np.ndarray
    relaxation_coeffs: np.ndarray
    equilib_coeffs: Dict
    mass: float
    
    map_init: InitVar[Union[Callable, str]]
    mom_coll_op_construct: InitVar[Callable]
    advec_op_construct: InitVar[Callable]
    
    _map: np.ndarray = field(init=False, repr=False)
    mom_coll_op: np.ndarray = field(init=False, repr=False)
    coll_op: np.ndarray = field(init=False, repr=False)
    advec_ops: Tuple[np.ndarray] = field(init=False, repr=False)
    inv_mom_space_transform: np.ndarray = field(init=False, repr=False)


    def __post_init__(self, map_init, mom_coll_op_construct, advec_op_construct):
        
        assert self.relaxation_coeffs.shape[0] == self.mom_space_transform.shape[0]
        self.inv_mom_space_transform = np.linalg.inv(self.mom_space_transform)
        
        if isinstance(map_init, Callable):
            self._map = map_init(self.lattice_size, self.lattice_flow_vecs, self.mom_space_transform, self.inv_mom_space_transform)
        elif map_init == "random":
            self._map = np.random.random(self.lattice_size + (self.lattice_flow_vecs.shape[1],))
        else:
            raise ValueError("Unknown map initialization")
        
        init_densities = self.densities()
        init_velocities = self.velocities(init_densities)
        
        if mom_coll_op_construct != None:
            self.mom_coll_op = mom_coll_op_construct(init_densities, init_velocities, self.relaxation_coeffs, self.equilib_coeffs)
            self.coll_op = np.einsum('ij,mnjk,kl->mnil' ,self.inv_mom_space_transform, self.mom_coll_op, self.inv_mom_space_transform)
        else:
            self.mom_coll_op = None
            self.coll_op = None
        
        if advec_op_construct != None:
            self.advec_ops = advec_op_construct(self.lattice_flow_vecs, self.lattice_size)
        else:
            self.advec_ops = None

    @property
    def map(self):
        return self._map
    
    @map.setter
    def map(self, new_map):
        if isinstance(new_map, np.ndarray):
            if new_map.shape == self._map.shape:
                self._map = new_map
            else:
                raise ValueError(f"Shape mismatch between current map ({self._map.shape}) and updated map ({new_map.shape}).")
        else:
            raise ValueError("The value of the provided (updated) map should be an numpy array")
    
    #def __sub__(self, other):
    #    return self._map - other.map
    
    @classmethod
    def D2Q9(dfm_class, lattice_size, mass, map_init, relaxation_coeffs, alpha3, gamma4):
        delta_t = 1.0
        delta_s = (1.0, 1.0) #delta_x, delta_y
        
        lattice_flow_vecs = np.array([np.array([0,0]),
                            *[np.array([int(np.cos((i-1)*np.pi/2)) * delta_s[0]/delta_t,int(np.sin((i-1)*np.pi/2)) * delta_s[1]/delta_t]) for i in range(1,5)],
                            *[np.array([int(np.sign(np.cos((2*j-9)*np.pi/4))) * delta_s[0]/delta_t, int(np.sign(np.sin((2*j-9)*np.pi/4))) * delta_s[1]/delta_t]) for j in range(5,9)],
                            ]).T
        #lattice_flow_vecs = (np.array([0,0]), np.array([-1,0]), np.array([0,-1]),
        #(wrong format!)     np.array([0,1]), np.array([1,0]), np.array([-1,-1]),
        #                    np.array([-1,1]), np.array([1,-1]), np.array([1,1]))
        
        mom_space_transform = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                                         [4, -2, -2, -2, -2, 1, 1, 1, 1],
                                         [0, 1, 0, -1, 0, 1, -1, -1, 1],
                                         [0, -2, 0, 2, 0, 1, -1, -1, 1],
                                         [0, 0, 1, 0, -1, 1, 1, -1,-1],
                                         [0, 0, -2, 0, 2, 1, 1, -1, -1],
                                         [0, 1, -1, 1, -1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, -1, 1, -1],
                                         ], dtype=float)
        
        #Relaxation params = const. or different:  
        if isinstance(relaxation_coeffs, int):
            relaxation_coeffs = np.full((lattice_flow_vecs.shape[1]), fill_value=relaxation_coeffs)
        elif isinstance(relaxation_coeffs, np.ndarray):
            if relaxation_coeffs.shape[0] == lattice_flow_vecs.shape[1]:
                #s2/s8/s9 = bulk/shear viscosity = most relevant
                #s3/s5/s7 = only higher order corrections = choose slightly larger 1
                #Must have s8 == s9! (mind python 0 indexing)
                assert relaxation_coeffs[8-1] == relaxation_coeffs[9-1] 
                assert relaxation_coeffs[1-1] == 0 and relaxation_coeffs[4-1] == 0 and relaxation_coeffs[6-1] == 0
            else:
                raise ValueError(f"Exactly {lattice_flow_vecs.shape[1]} have to be provided!")
        else:
            raise ValueError(f"Parameter relaxation_coeffs should be integer or array of length {lattice_flow_vecs.shape[1]}")
        
        #equilib_coeffs = {'c1': -2, 'alpha2': -8, 'alpha3': alpha3, 'gamma1': 2, 'gamma2': 18,'gamma3': 2, 'gamma4': gamma4}
        equilib_coeffs = {'c1': -2, 'alpha2': -8, 'alpha3': alpha3, 'gamma1': 2/3, 'gamma2': 18,'gamma3': 2/3, 'gamma4': gamma4}

        assert len(lattice_size) == 2
        
        inst = dfm_class(n_dim=2, 
                         lattice_size=lattice_size,
                         lattice_type='D2Q9',
                         mass=mass,
                         delta_t=delta_t,
                         delta_s=delta_s,
                         lattice_flow_vecs=lattice_flow_vecs, 
                         mom_space_transform=mom_space_transform,
                         relaxation_coeffs=relaxation_coeffs,
                         equilib_coeffs=equilib_coeffs,
                         map_init=map_init,
                         mom_coll_op_construct=D2Q9_collision_matrix,
                         advec_op_construct=D2Q9_advection_matrix)
        return inst

    @classmethod
    def D2Q16(dfm_class, lattice_size, mass, map_init, relaxation_coeffs):
        delta_t = 1.0
        delta_s = (1.0, 1.0) #delta_x, delta_y
        
        lattice_flow_vecs = np.array([*[np.array([1*delta_s[0]/delta_t, 0 * delta_s[1]/delta_t]), np.array([0*delta_s[0]/delta_t, 1 * delta_s[1]/delta_t]), np.array([-1*delta_s[0]/delta_t, 0 * delta_s[1]/delta_t]), np.array([0 * delta_s[0]/delta_t, -1 * delta_s[1]/delta_t])],
                            *[6*np.array([1*delta_s[0]/delta_t, 0 * delta_s[1]/delta_t]), 6*np.array([0*delta_s[0]/delta_t, 1 * delta_s[1]/delta_t]), 6*np.array([-1*delta_s[0]/delta_t, 0 * delta_s[1]/delta_t]), 6*np.array([0 * delta_s[0]/delta_t, -1 * delta_s[1]/delta_t])],
                            *[np.sqrt(2) *np.array([1 * delta_s[0]/delta_t, 1  * delta_s[1]/delta_t]), np.sqrt(2) *np.array([-1 * delta_s[0]/delta_t, 1  * delta_s[1]/delta_t]),np.sqrt(2) *np.array([-1 * delta_s[0]/delta_t, -1  * delta_s[1]/delta_t]),np.sqrt(2) *np.array([1 * delta_s[0]/delta_t, -1  * delta_s[1]/delta_t])],
                            *[(3/np.sqrt(2)) *np.array([1 * delta_s[0]/delta_t, 1  * delta_s[1]/delta_t]), (3/np.sqrt(2)) *np.array([-1 * delta_s[0]/delta_t, 1  * delta_s[1]/delta_t]),(3/np.sqrt(2)) *np.array([-1 * delta_s[0]/delta_t, -1  * delta_s[1]/delta_t]),(3/np.sqrt(2)) *np.array([1 * delta_s[0]/delta_t, -1  * delta_s[1]/delta_t])],
                            ]).T

        mom_space_transform = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 0, -1, 0, 6, 0, -6, 0, np.sqrt(2), -np.sqrt(2), -np.sqrt(2), np.sqrt(2), 3/np.sqrt(2), -3/np.sqrt(2), -3/np.sqrt(2), 3/np.sqrt(2)],
                                         [0, 1, 0, -1, 0, 6, 0, -6,  np.sqrt(2),  np.sqrt(2),  -np.sqrt(2), -np.sqrt(2), 3/np.sqrt(2), 3/np.sqrt(2), -3/np.sqrt(2), -3/np.sqrt(2)],
                                         [1/2, 1/2, 1/2, 1/2, 18, 18, 18, 18, 2,2,2,2, 9/2, 9/2, 9/2, 9/2],
                                         [1, -1, 1, -1, 36, -36, 36, -36, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 2, -2, 9/2, -9/2, 9/2, -9/2],
                                         [1/2, 0, -1/2, 0, 108, 0, -108, 0, 2*np.sqrt(2), -2*np.sqrt(2), -2*np.sqrt(2), 2*np.sqrt(2), 27/(2*np.sqrt(2)), -27/(2*np.sqrt(2)), -27/(2*np.sqrt(2)), 27/(2*np.sqrt(2))],
                                         [0, 1/2, 0, -1/2, 0, 108, 0, -108, 2*np.sqrt(2), 2*np.sqrt(2), -2*np.sqrt(2), -2*np.sqrt(2), 27/(2*np.sqrt(2)), 27/(2*np.sqrt(2)), -27/(2*np.sqrt(2)), -27/(2*np.sqrt(2))],
                                         [1, 0, -1, 0, 216, 0, -216, 0, -4*np.sqrt(2), 4*np.sqrt(2), 4*np.sqrt(2), -4*np.sqrt(2), -27/(2*np.sqrt(2)), 27/(2*np.sqrt(2)), 27/(2*np.sqrt(2)), -27/(2*np.sqrt(2))],
                                         [0, -1, 0, 1, 0, -216, 0, 216, 4*np.sqrt(2), 4*np.sqrt(2), -4*np.sqrt(2), -4*np.sqrt(2), 27/(2*np.sqrt(2)), 27/(2*np.sqrt(2)), -27/(2*np.sqrt(2)), -27/(2*np.sqrt(2))],
                                         [1/4, 1/4, 1/4, 1/4, 324, 324, 324, 324, 4, 4, 4, 4, 81/4, 81/4, 81/4, 81/4],
                                         [1, 1, 1, 1, 1296, 1296, 1296, 1296, -16, -16, -16, -16, -81, -81, -81, -81],
                                         [1, -1, 1, -1, 1296, -1296, 1296, -1296, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 8, -8, 8, -8, 81/2, -81/2, 81/2, -81/2],
                                         [1, 0, -1, 0, 7776, 0, -7776, 0, -16*np.sqrt(2), 16*np.sqrt(2), 16*np.sqrt(2), -16*np.sqrt(2), -243/np.sqrt(2), 243/np.sqrt(2), 243/np.sqrt(2), -243/np.sqrt(2)],
                                         [0, -1, 0, 1, 0, -7776, 0, 7776, 16*np.sqrt(2), 16*np.sqrt(2), -16*np.sqrt(2), -16*np.sqrt(2), 243/np.sqrt(2), 243/np.sqrt(2), -243/np.sqrt(2), -243/np.sqrt(2)]], dtype=float)
        
        #Relaxation params = const. or different:  
        if isinstance(relaxation_coeffs, int):
            relaxation_coeffs = np.full((lattice_flow_vecs.shape[1]), fill_value=relaxation_coeffs)
        elif isinstance(relaxation_coeffs, np.ndarray):
            if relaxation_coeffs.shape[0] == lattice_flow_vecs.shape[1]:
                #s2/s8/s9 = bulk/shear viscosity = most relevant
                #s3/s5/s7 = only higher order corrections = choose slightly larger 1
                #Must have s8 == s9! (mind python 0 indexing)
                #assert ....
                pass
            else:
                raise ValueError(f"Exactly {lattice_flow_vecs.shape[1]} have to be provided!")
        else:
            raise ValueError(f"Parameter relaxation_coeffs should be integer or array of length {lattice_flow_vecs.shape[1]}")
        
        equilib_coeffs = {'c1': -2}

        assert len(lattice_size) == 2
        
        inst = dfm_class(n_dim=2, 
                         lattice_size=lattice_size,
                         lattice_type='D2Q16',
                         mass=mass,
                         delta_t=delta_t,
                         delta_s=delta_s,
                         lattice_flow_vecs=lattice_flow_vecs, 
                         mom_space_transform=mom_space_transform,
                         relaxation_coeffs=relaxation_coeffs,
                         equilib_coeffs=equilib_coeffs,
                         map_init=map_init,
                         mom_coll_op_construct=None,
                         advec_op_construct=None)
        return inst
    
    def equilib_moments(self, c_mom_map, densities=None, velocities=None):
        if self.lattice_type == 'D2Q9':
            ''' more general equilibrium:
            density_0 = np.average(densities)
            eq_density = np.full(densities.shape, fill_value=density_0)
            jx = (densities * velocities[:,:,0])
            jy = (densities * velocities[:,:,1])
            eq_energy = self.equilib_coeffs['alpha2']*densities/4 + (self.equilib_coeffs['gamma2']*(jx**2 + jy**2)/6) / density_0
            eq_energy_sq = self.equilib_coeffs['alpha3']*densities + (self.equilib_coeffs['gamma4']*(jx**2 + jy**2)/6) / density_0
            eq_mom_density = [jx, jy]
            eq_energy_flux = [self.equilib_coeffs['c1'] * jx / 2, self.equilib_coeffs['c1'] * jy / 2]
            eq_vis_stress_xx = (self.equilib_coeffs['gamma1']*(jx**2 - jy**2)/2) * density_0
            eq_vis_stress_xy = (self.equilib_coeffs['gamma3']*(jx*jy)/2) / density_0
            '''
            density_0 = np.average(densities)
            eq_density = densities
            jx = densities * velocities[:,:,0]
            jy = densities * velocities[:,:,1]
            eq_energy = -2*densities + (3*(jx**2 + jy**2)/6) * (2*density_0 - densities) / (density_0)
            eq_energy_sq = densities - 3*(jx**2 + jy**2) * (2*density_0 - densities) / (density_0)
            eq_mom_density = [jx, jy]
            eq_energy_flux = [- jx,- jy]
            eq_vis_stress_xx = (jx**2 - jy**2) * (2*density_0 - densities) / (density_0)
            eq_vis_stress_xy = (jx*jy) * (2*density_0 - densities) / (density_0)
            
            return np.stack((eq_density, eq_energy, eq_energy_sq, eq_mom_density[0], eq_energy_flux[0], eq_mom_density[1], eq_energy_flux[1], eq_vis_stress_xx, eq_vis_stress_xy), axis=2)
        
        elif self.lattice_type == 'D2Q16':
            #'Conserved' moments: density, momentum density (x,y) and energy
            m1_eq = c_mom_map[:,:, 0] 
            m2_eq, m3_eq = c_mom_map[:,:, 1], c_mom_map[:,:, 2]
            m4_eq = c_mom_map[:,:, 3]
            print(np.max(m1_eq), np.max(m2_eq), np.max(m3_eq), np.max(m4_eq))

            #Remaining non-conserved moments:
            m5_eq = (m2_eq**2 - m3_eq**2) / m1_eq
            m6_eq = m2_eq*m3_eq / m1_eq
            m7_eq = (m4_eq + m1_eq*m4_eq)*m2_eq / m1_eq
            m8_eq = (m4_eq + m1_eq*m4_eq)*m3_eq / m1_eq
            m9_eq = (m2_eq**2 - 3*(m3_eq**2))*m2_eq / (m1_eq**2)
            m10_eq = (3*(m2_eq**2) - m3_eq**2)*m3_eq / (m1_eq**2)
            m11_eq = 2*(m4_eq**2) / m1_eq - ((m2_eq**2 + m3_eq**2)**2 / (4*m1_eq**3))
            m12_eq = np.zeros(m1_eq.shape)
            m13_eq = (6*m1_eq*m4_eq - 2*(m2_eq**2) - 2*(m3_eq**2))*(m2_eq**2 - m3_eq**2)/(m1_eq**3)
            m14_eq = (6*m1_eq*m4_eq - 2*(m2_eq**2) - 2*(m3_eq**2))*(m2_eq*m3_eq)/(m1_eq**3)
            m15_eq = np.zeros(m1_eq.shape)
            m16_eq = np.zeros(m1_eq.shape)
            
            return np.stack((m1_eq, m2_eq, m3_eq, m4_eq, m5_eq, m6_eq, m7_eq, m8_eq, m9_eq, m10_eq, m11_eq, m12_eq, m13_eq, m14_eq, m15_eq, m16_eq), axis=2)

    def densities(self):
        return np.sum(self._map, axis=-1)

    def velocities(self, densities=None):
        if not isinstance(densities, np.ndarray):
            densities = self.densities()
        
        return np.einsum('ij,klj->kli', self.lattice_flow_vecs, self._map) / densities[:, :, np.newaxis]
    
    def pressures(self):
        return np.einsum('ij,ij,klj->kl', self.lattice_flow_vecs, self.lattice_flow_vecs, self._map)
    
    def moment_basis(self):
        '''
        Returns the density flow map in the moment basis:
        (density / kin. energy / kin. energy squared / momentum density x / energy flux x
        / momentum space density y / energy flux y / viscous stress tensor diagonal / vis. stress tensor anti-diagonal 
        '''
        return np.einsum('ij,klj->kli', self.mom_space_transform, self._map)
    
    def moments(self): 
        c_mom_map = self.moment_basis()
        if self.lattice_type == 'D2Q9':
            moments = {'energy': c_mom_map[:,:,1], 'mom_density': c_mom_map[:,:,[3,4]], 'energy_flux': c_mom_map[:,:,[5,6]], 'vis_stress': c_mom_map[:,:,[7,8]]}
            return moments
        elif self.lattice_type == 'D2Q16':
            moments = {'density': c_mom_map[:,:,0], 'mom_density': c_mom_map[:,:,[1,2]], 'energy': c_mom_map[:,:,3], 'vis_stress': c_mom_map[:,:,[4,5]],'energy_flux': c_mom_map[:,:,[6, 7]], 'energy_sqaured': c_mom_map[:,:,10]}
            return moments
    

class LatticeBoltzmann:
    '''
    Defines a Lattice Boltzmann Model (LBM) from a given density flow map / lattice
    '''
    
    def __init__(self, density_flow_map, advect_BCs = None):
        '''
        ..
        '''
        self.dfm = density_flow_map
        self.speed_of_sound = 1/np.sqrt(3)
        self.advect_BCs = advect_BCs

    def collision(self):
        c_map = self.dfm.map
        delta_map = np.einsum('mnij,mnj->mni', self.dfm.coll_op, c_map)
        self.dfm.map = c_map + delta_map
    
    def direct_collision(self, velocities=None):

        print(self.dfm.mom_space_transform.shape)
        c_mom_map = np.einsum('kl,ijl->ijk', self.dfm.mom_space_transform,  self.dfm.map)
        if self.dfm.lattice_type=='D2Q9':
            densities = self.dfm.densities()
            if velocities == None:
                velocities = self.dfm.velocities()

            eq_map = self.dfm.equilib_moments(c_mom_map, densities, velocities)
            
            n_mom_map = c_mom_map - (self.dfm.relaxation_coeffs * (c_mom_map - eq_map) )
            self.dfm.map = np.einsum('kl,ijl->ijk', self.dfm.inv_mom_space_transform, n_mom_map)
        elif self.dfm.lattice_type=='D2Q16':
            eq_map = self.dfm.equilib_moments(c_mom_map)
            
            n_mom_map = c_mom_map - (self.dfm.relaxation_coeffs * (c_mom_map - eq_map) )
            self.dfm.map = np.einsum('kl,ijl->ijk', self.dfm.inv_mom_space_transform, n_mom_map)


    def classical_collision(self):
        if self.dfm.lattice_type=='D2Q9':
            c_map = self.dfm.map
            def get_equilib_map():
                equilib_map = np.zeros((*self.dfm.lattice_size, 9))
                weights = np.array([4/9, 1/9, 1/9,
                                1/9, 1/9, 1/36,
                                1/36, 1/36, 1/36])
                velocities = self.dfm.velocities()
                densities =  self.dfm.densities()

                term1 = np.einsum("ijk,kl->ijl", velocities, self.dfm.lattice_flow_vecs) / (self.speed_of_sound**2)
                term2 = np.einsum("ijk,kl->ijl", velocities, self.dfm.lattice_flow_vecs)**2 / (2*(self.speed_of_sound**4))
                term3 = np.einsum("ijk,ijk->ij", velocities, velocities) / (2*(self.speed_of_sound**2))
                equilib_map = np.einsum('ij,l,ijl->ijl', densities, weights, (1 + term1 + term2 - term3[:,:, np.newaxis]))
                return equilib_map
            
            equilib_map = get_equilib_map()
            
            self.dfm.map = c_map - (c_map - equilib_map) * self.dfm.relaxation_coeffs[1]
        else:
            raise ValueError("Linearized Collision currently only supports D2Q9 lattices")
        
    def apply_avect_BCs(self):
        if self.advect_BCs == None:
            pass #Leave periodic BC's
        elif callable(self.advect_BCs):
            self.dfm.map = self.advect_BCs(self.dfm.map)
        else:
            raise ValueError("Unknown type of advection boundary conditions. Only callables of maps or None supported.")
    
    def advection(self):
        c_map = self.dfm.map
        if self.dfm.lattice_type == 'D2Q9':
            self.dfm.map = np.einsum('ijn,jkn,kmn->imn', self.dfm.advec_ops[0], c_map, self.dfm.advec_ops[1])
        else:
            raise ValueError("Matrix-based Advection currently only supports D2Q9 lattices")
        
    def classical_advection(self):
        '''
        c_map = self.dfm.map
        new_map = np.zeros((*self.dfm.lattice_size, 9))
        for i in range(9):
            print(f"Advecting in direction {i}")
            speed_ratio = self.dfm.delta_s[0]/self.dfm.delta_t
            shift = (self.dfm.lattice_flow_vecs[:,i] / speed_ratio).astype(int)
            new_map[:,:,i] = speed_ratio*np.roll(c_map[:,:,i], shift=shift, axis=(0,1)) + (1 - speed_ratio)*c_map[:,:,i]
        self.dfm.map = new_map
        '''
        c_map = self.dfm.map
        new_map = np.zeros((*self.dfm.lattice_size, self.dfm.lattice_flow_vecs.shape[1]))
        for i in range(self.dfm.lattice_flow_vecs.shape[1]):
            print(f"Advecting in direction {i}")
            speed_ratio = self.dfm.delta_s[0]/self.dfm.delta_t
            raw_velocity_vec = self.dfm.lattice_flow_vecs[:,i] / speed_ratio
            shift = (raw_velocity_vec).astype(int)
            new_map[:,:,i] = self.dfm.delta_t*np.roll(c_map[:,:,i], shift=shift, axis=(0,1)) + (1 - self.dfm.delta_t)*c_map[:,:,i]
        self.dfm.map = new_map
    
    def evolve(self):

        self.direct_collision()
        #self.classical_collision()
        print("Completed collision")
        
        self.apply_avect_BCs()
        #self.advection()
        self.classical_advection()
        
        print("Completed advection")
    
    def __run__(self, end_of_time):
        time = np.arange(0, end_of_time, step=self.dfm.delta_t)
        
        time_samples = 40
        save_timestep = max([1, int(time.shape[0] / time_samples)]) #Save only 100 timesteps in total
        
        output = {'density_per_time': [], 'pressure_per_time': [], 'net_velocity_per_time': [], 'net_flow_vec_per_time': np.array([])}
        output['energy_per_time'], output['mom_density_per_time'], output['energy_flux_per_time'], output['vis_stress_per_time'] = [], [], [], []
        output['time'] = time[::save_timestep]
        
        for t in tqdm(time):
            if t in output['time']:
                #Store previous time-step before (next) iteration
                output['density_per_time'].append(self.dfm.densities())
                output['pressure_per_time'].append(self.dfm.pressures())
                
                output['net_velocity_per_time'].append(self.dfm.velocities())
                
                c_moments = self.dfm.moments()
                output['energy_per_time'].append(c_moments['energy'])
                output['mom_density_per_time'].append(c_moments['mom_density'])
                output['energy_flux_per_time'].append(c_moments['energy_flux'])
                output['vis_stress_per_time'].append(c_moments['vis_stress'])
                
            #Evolution of system
            self.evolve()

        output['density_per_time'] = np.array(output['density_per_time'])
        output['pressure_per_time'] = np.array(output['pressure_per_time'])
        output['net_velocity_per_time'] = np.array(output['net_velocity_per_time'])
        output['energy_per_time'] = np.array(output['energy_per_time'])
        output['mom_density_per_time'] = np.array(output['mom_density_per_time'])
        output['energy_flux_per_time'] = np.array(output['energy_flux_per_time'])
        output['vis_stress_per_time'] = np.array(output['vis_stress_per_time'])
            
        return output


class MultiComp_LatticeBoltzmann:
    '''
    Defines a Lattice Boltzmann Model (LBM) from a given density flow map / lattice for multiple (different) fluids
    '''
    
    def __init__(self, density_flow_maps, interact_strength):
        self.lb_models = [LatticeBoltzmann(dfm) for dfm in density_flow_maps]
        self.interact_strength = interact_strength  
    
    def average_velocities(self):
        norm = sum([lbm.dfm.mass * lbm.dfm.densities() for lbm in self.lb_models])
        return sum([lbm.dfm.mass * lbm.dfm.velocities() for lbm in self.lb_models]) / norm

    
    def lattice_interact_force(self, densities_per_lattice):
        forces_per_lattice = []
        
        for (idx, densities) in enumerate(densities_per_lattice):
            D2Q9_interact_kernel = self.interact_strength * np.array([np.linalg.norm(lfv) for lfv in self.lb_models[idx].dfm.lattice_flow_vecs])
            pot_per_prime_comp = []
            for (idx_prime, densities_prime) in enumerate(densities_per_lattice):
                if idx == idx_prime:
                    pass #latttices don't self-interact
                else:
                    densities_prime_stack = np.tile(densities_prime, (1,1,D2Q9_interact_kernel.shape[0]))
                    shifted_densities_prime = np.einsum('ijn,jkn,kmn->imn', self.lb_models[idx].dfm.advec_ops[0], densities_prime_stack, self.lb_models[idx].advec_ops[1])
                    pot_per_direc = np.einsum('k,ijk->ij', D2Q9_interact_kernel, shifted_densities_prime)
                    pot_per_prime_comp.append(pot_per_direc)
            forces_per_lattice.append(-densities * sum(pot_per_prime_comp))
        
        return forces_per_lattice
        
    def evolve(self):
        densities_per_lattice = []
        av_velocities = self.average_velocities()
        for lbm in self.lb_models:
            lbm.direct_collision(av_velocities)
            densities_per_lattice.append(lbm.dfm.densities())
        
        forces_per_lattice = self.lattice_interact_force(densities_per_lattice)
        
        for (idx, lbm) in enumerate(self.lb_models):
            lbm.dfm.map = lbm.dfm.map + (forces_per_lattice[idx])[:,:, np.newaxis]
            lbm.advection()
            
    
    def __run__(self, end_of_time):
        time = np.arange(0, end_of_time, step=self.dfm.delta_t)

        output = {}
        output['time'] = time
        
        for t in tqdm(time):
            #Store previous time-step before (next) iteration

            #Evolution of system
            self.evolve()
        
        return output
