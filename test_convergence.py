# base imports
import numpy as np
from scipy.sparse.linalg import eigs

# classes and functions
from classes import World, Surface
from functions import get_initial_graph

# initialize world and surface
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

# build initial sparse transition matrix T
index_dict, vertices, T = surface.get_rw_sparse_matrix()

# find second largest eigenvalue in modulus
w, v = eigs(T, k=3, which='LM')
sle = min(np.abs(w))

# check for convergence
print(sle**1500)

'''
These results indicate that 1500 moves is not enough to acheive convergence in
a 200 by 200 by 200 world, with 1500 moves, even in the simplest of cases with 
no built structures. This means that we must be wary in using the mean field
approximation.
'''