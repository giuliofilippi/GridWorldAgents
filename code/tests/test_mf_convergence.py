# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
from scipy.sparse.linalg import eigs
from numpy.linalg import eig
import time

# classes and functions
from classes import World, Surface
from functions import get_initial_graph

# initialize world and surface
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

# build initial sparse transition matrix T
index_dict, vertices, T = surface.get_rw_sparse_matrix()


# start time
start_time = time.time()
# find second largest eigenvalue in modulus
w, v = eigs(T, k=3, which='LM')
sle = min(np.abs(w))
# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)
# check for convergence
print(sle**1500)


'''
# start time
start_time = time.time()
# find second largest eigenvalue in modulus
T_arr = T.toarray()
w, v = eig(T_arr)
# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)
'''

'''
These results indicate that 1500 moves is not enough to acheive convergence in
a 200 by 200 by 200 world, with 1500 moves, even in the simplest of cases with 
no built structures. This means that we must be wary in using the mean field
approximation.
'''