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

                 

def D2Q9_collision_matrix(c_densities, c_velocities,  relaxation_coeffs, equilib_coeffs):
    coll_mat = np.zeros((*c_densities.shape, *relaxation_coeffs.shape, *relaxation_coeffs.shape)) #4D array
    coll_mat[:,:,1,0] = (relaxation_coeffs[2-1]*equilib_coeffs['alpha2'] / 4) * c_densities
    coll_mat[:,:,2,0] = (relaxation_coeffs[3-1]*equilib_coeffs['alpha3'] / 4) * c_densities
    
    coll_mat[:,:,1,1] = -relaxation_coeffs[2-1]
    
    coll_mat[:,:,2,2] = -relaxation_coeffs[3-1]
    
    coll_mat[:,:,1,3] = relaxation_coeffs[2-1]*equilib_coeffs['gamma2']*c_velocities[:,:,0] / 3
    coll_mat[:,:,2,3] = relaxation_coeffs[3-1]*equilib_coeffs['gamma4']*c_velocities[:,:,0] / 3
    coll_mat[:,:,4,3] = relaxation_coeffs[5-1]*equilib_coeffs['c1'] / 2
    coll_mat[:,:,7,3] = 3*relaxation_coeffs[8-1]*equilib_coeffs['gamma1']*c_velocities[:,:,0]
    coll_mat[:,:,8,3] = 3*relaxation_coeffs[9-1]*equilib_coeffs['gamma3']*c_velocities[:,:,1] / 2
    
    coll_mat[:,:,4,3] = -relaxation_coeffs[5-1]
    
    coll_mat[:,:,1,4] = relaxation_coeffs[2-1]*equilib_coeffs['gamma2']*c_velocities[:,:,1] / 3
    coll_mat[:,:,2,4] = relaxation_coeffs[3-1]*equilib_coeffs['gamma4']*c_velocities[:,:,1] / 3
    coll_mat[:,:,6,4] = relaxation_coeffs[7-1]*equilib_coeffs['c1'] / 2
    coll_mat[:,:,7,4] = -3*relaxation_coeffs[8-1]*equilib_coeffs['gamma1']*c_velocities[:,:,1]
    coll_mat[:,:,8,4] = 3*relaxation_coeffs[9-1]*equilib_coeffs['gamma3']*c_velocities[:,:,0] / 2
    
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
            self._map = map_init(self.lattice_size, self.lattice_flow_vecs)
        elif map_init == "random":
            self._map = np.random.random(self.lattice_size + (self.lattice_flow_vecs.shape[1],))
        else:
            raise ValueError("Unknown map initialization")
        
        init_densities = self.densities()
        init_velocities = self.velocities(init_densities)
        
        self.mom_coll_op = mom_coll_op_construct(init_densities, init_velocities, self.relaxation_coeffs, self.equilib_coeffs)
        self.coll_op = np.einsum('ij,mnjk,kl->mnil' ,self.inv_mom_space_transform, self.mom_coll_op, self.mom_space_transform)
        self.advec_ops = advec_op_construct(self.lattice_flow_vecs, self.lattice_size)
          
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
    def D2Q9(dfm_class, lattice_size, map_init, relaxation_coeffs, alpha3, gamma4):
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
            
        equilib_coeffs = {'c1': -2, 'alpha2': -8, 'alpha3': alpha3, 'gamma1': 2/3, 'gamma2': 18,'gamma3': 2/3, 'gamma4': gamma4}

        assert len(lattice_size) == 2
        
        inst = dfm_class(n_dim=2, 
                         lattice_size=lattice_size,
                         lattice_type='D2Q9',
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
    
    def densities(self):
        return np.sum(self._map, axis=-1)

    def velocities(self, densities=None):
        if not isinstance(densities, np.ndarray):
            densities = self.densities()
        
        return np.einsum('ij,klj->kli', self.lattice_flow_vecs, self._map) / densities[:, :, np.newaxis]
    
    def moment_basis(self):
        '''
        Returns the density flow map in the moment basis:
        (density / kin. energy / kin. energy squared / momentum density x / energy flux x
        / momentum space density y / energy flux y / viscous stress tensor diagonal / vis. stress tensor anti-diagonal 
        '''
        return np.matmul(self.mom_space_transform, self._map)
        
    

class LatticeBoltzmann:
    '''
    Defines a Lattice Boltzmann Model (LBM) from a given density flow map / lattice
    '''
    
    def __init__(self, density_flow_map):
        '''
        ..
        '''
        self.dfm = density_flow_map
    
    def collision(self):
        c_map = self.dfm.map
        delta_map = np.einsum('mnij,mnj->mni', self.dfm.coll_op, c_map)
        print(delta_map)
        self.dfm.map = c_map + delta_map
        
    def advection(self):
        c_map = self.dfm.map
        if self.dfm.lattice_type == 'D2Q9':
            self.dfm.map = np.einsum('ijn,jkn,kmn->imn', self.dfm.advec_ops[0], c_map, self.dfm.advec_ops[1])
        else:
            raise ValueError("Advection currently only supports D2Q9 lattices")
        
    def evolve(self):
        self.collision()
        
        self.advection()
        
    
    def __run__(self, end_of_time):
        time = np.arange(0, end_of_time, step=self.dfm.delta_t)
        
        output = {'density_per_time': [], 'net_flow_vec_per_time': np.array([])}
        output['time'] = time
        
        for t in tqdm(time):
            self.evolve()
            output['density_per_time'].append(self.dfm.densities())
        
        output['density_per_time'] = np.array(output['density_per_time'])
        return output