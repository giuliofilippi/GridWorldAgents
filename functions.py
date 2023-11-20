# imports
import numpy as np
from mayavi import mlab
from scipy.stats import skewnorm
import scipy.sparse as sp

# ------------ Useful Lists ----------------

# Lists and Dictionnaries
translation_dict = {-2:'agent', -1:'object', 0:'void', 1:'soil',2:'pellet'}
neighbour_directions_3d = np.array([
    [1, 0, 0],   # Right
    [-1, 0, 0],  # Left
    [0, 1, 0],   # Forward
    [0, -1, 0],  # Backward
    [0, 0, 1],   # Up
    [0, 0, -1],  # Down
    [1, 1, 0],   # Right-Forward
    [1, -1, 0],  # Right-Backward
    [-1, 1, 0],  # Left-Forward
    [-1, -1, 0], # Left-Backward
    [1, 0, 1],   # Right-Up
    [1, 0, -1],  # Right-Down
    [-1, 0, 1],  # Left-Up
    [-1, 0, -1], # Left-Down
    [0, 1, 1],   # Forward-Up
    [0, 1, -1],  # Forward-Down
    [0, -1, 1],  # Backward-Up
    [0, -1, -1], # Backward-Down
    [1, 1, 1],   # Right-Forward-Up
    [1, 1, -1],  # Right-Forward-Down
    [1, -1, 1],  # Right-Backward-Up
    [1, -1, -1], # Right-Backward-Down
    [-1, 1, 1],  # Left-Forward-Up
    [-1, 1, -1],  # Left-Forward-Down
    [-1, -1, 1],  # Left-Backward-Up
    [-1, -1, -1] # Left-Backward-Down
])
neighbour_directions = np.array([
    [0, 0, 1], # Up
    [0, 0, -1], # Down
    [0, 1, 0], # Forward
    [0, -1, 0], # Backwards
    [1, 0, 0], # Right
    [-1, 0, 0] # Left
])

# ------------ Helper functions ----------------

# get initial surface graph
def get_initial_graph(width, lenght, soil_height):
    # initialize
    graph = {}
    # inside cases
    for i in range(1,width-1):
        for j in range(1,lenght-1):
            graph[(i,j,soil_height)] = [(i-1,j,soil_height),(i,j-1,soil_height),(i+1,j,soil_height),(i,j+1,soil_height)]
    # sides
    for j in range(1,lenght-1):
        graph[(0,j,soil_height)] = [(0,j-1,soil_height),(1,j,soil_height),(0,j+1,soil_height)]
        graph[(width-1,j,soil_height)] = [(width-1,j-1,soil_height),(width-2,j,soil_height),(width-1,j+1,soil_height)]
    # sides
    for i in range(1,width-1):
        graph[(i,0,soil_height)] = [(i-1,0,soil_height),(i,1,soil_height),(i+1,0,soil_height)]
        graph[(i,lenght-1,soil_height)] = [(i-1,lenght-1,soil_height),(i,lenght-2,soil_height),(i+1,lenght-1,soil_height)]
    # corners
    graph[(0,0,soil_height)] = [(1,0,soil_height),(0,1,soil_height)]
    graph[(width-1,lenght-1,soil_height)] = [(width-2,lenght-1,soil_height),(width-1,lenght-2,soil_height)]
    graph[(width-1,0,soil_height)] = [(width-2,0,soil_height),(width-1,1,soil_height)]
    graph[(0,lenght-1,soil_height)] = [(0,lenght-2,soil_height),(1,lenght-1,soil_height)]
    # return
    return graph

# get neighbours of some voxel given current graph
def get_neighbours(pos, graph):
    neighbours = []
    # loop over potential neighbours
    for dir in neighbour_directions:
        new_pos = tuple(np.array(pos)+dir)
        # check if is in graph
        if new_pos in graph:
            # if yes, append
            neighbours.append(new_pos)
    return neighbours

# update graph
def update_surface_function(surface, type, pos, world):
    # position and tuple version
    x,y,z = pos
    vertex = (x,y,z)
    # case 1
    if type == 'pickup':
        new_vertex = (x,y,z-1)
        # add new vertex to graph and degrees
        surface.graph[new_vertex] = []
        surface.degrees[new_vertex] = 0
        # get neighbours
        nbr_list = get_neighbours(new_vertex,surface.graph)
        for nbr in nbr_list:
            # this adds edge to both and also updates degrees
            surface.add_edge((new_vertex, nbr))
    # case 2
    elif type == 'drop':
        # remove vertex at location of drop
        surface.remove_vertex(vertex)
        # initialize new vertices
        new_vertices = []
        # loop over 3D directions
        for dir in neighbour_directions_3d:
            # new vertex positions
            new_pos = np.array(pos) + dir
            new_vertex = tuple(new_pos)
            # check in bounds
            if 0<=new_vertex[0]<world.width and 0<=new_vertex[1]<world.length and 0<=new_vertex[2]<world.height:
                # check empty
                if world.grid[new_vertex[0],new_vertex[1],new_vertex[2]]==0:
                    # check it isnt already in graph
                    if new_vertex not in surface.graph:
                        # accept as new vertex in surface
                        new_vertices.append(new_vertex)
        # loop over newly added surface vertices
        for new_vertex in new_vertices:
            # add to graph and degrees dicts
            surface.graph[new_vertex] = []
            surface.degrees[new_vertex] = 0
            # get neighbours (with current graph)
            nbr_list = get_neighbours(new_vertex,surface.graph)
            # loop over neighbours
            for nbr in nbr_list:
                # add new edges (exactly once because of sequentiality)
                surface.add_edge((new_vertex, nbr))
    # case 3
    else:
        raise ValueError('type not in pickup, drop')

# random choices function
def random_choices(ls, size=1, p=None):
    # No prob dist given
    if p is None:
        chosen_indices = np.random.choice(range(len(ls)), size, replace=False)
        choices = [ls[i] for i in chosen_indices]
        return choices
    # A prob dist is given
    else:
        chosen_indices = np.random.choice(range(len(ls)), size, p=p, replace=False)
        choices = [ls[i] for i in chosen_indices]
        return choices
    
# dual random choices function
def dual_random_choices(ls, size0, size1, prob0, prob1=None):
    # index choices
    num_elements = len(ls)
    choices0 = []
    choices1 = []
    # convert to arrays
    prob0 = np.nan_to_num(prob0.toarray().flatten())
    prob0 = prob0/np.sum(prob0)
    if prob1.count_nonzero()==0:
        prob1 = None
    else:
        prob1 = np.nan_to_num(prob1.toarray().flatten())
        prob1 /= np.sum(prob1)
    # Generate choices from the first distribution
    chosen_indices0 = np.random.choice(range(num_elements), size0, p=prob0, replace=False)
    choices0 = [ls[i] for i in chosen_indices0]
    # Generate choices from the second distribution
    remaining_indices = np.setdiff1d(range(num_elements), chosen_indices0)  
    # Normalize prob1 over the remaining indices
    if prob1 is not None:
        summed = np.sum(prob1[remaining_indices])
        if summed > 0:
            prob1_normalized = prob1[remaining_indices] / np.sum(prob1[remaining_indices])
        else:
            prob1_normalized = None
    else:
        prob1_normalized = None
    chosen_indices1 = np.random.choice(remaining_indices, size1, p=prob1_normalized, replace=False)
    choices1 = [ls[i] for i in chosen_indices1]
    # return
    return choices0, choices1

# get local grid data from position in world
def local_grid_data(pos, world):
        x,y,z = pos
        if 0<x<world.width-2 and 0<y<world.length-2 and 0<z<world.height-2:
            # Not on any boundary -- No padding
            local_data = world.grid[x-1:x+2,y-1:y+2,z-1:z+2]
            return local_data
        else:
            # On some boundary -- Need padding
            x_min, x_max = max(0, x - 1), min(world.width, x + 2)
            y_min, y_max = max(0, y - 1), min(world.length, y + 2)
            z_min, z_max = max(0, z - 1), min(world.height, z + 2)
            # slicing start and end indices
            x_start, x_end = 1 if x_min == 0 else 0, 2 if x_max == world.width else 3
            y_start, y_end = 1 if y_min == 0 else 0, 2 if y_max == world.length else 3
            z_start, z_end = 1 if z_min == 0 else 0, 2 if z_max == world.height else 3
            # initialize with -1 then fill with available data
            local_data = np.full((3, 3, 3), -1)
            local_data[x_start:x_end, y_start:y_end, z_start:z_end] = world.grid[
                x_min:x_min + x_end - x_start, y_min:y_min + y_end - y_start, z_min:z_min + z_end - z_start]
            # return
            return local_data

# a height function that returns the number of empty cells below agent
def compute_height(pos, world):
    x,y,z = pos
    max_h = world.height
    h=0
    for _ in range(max_h):
        z-=1
        cell = world.grid[x,y,z]
        if cell > 0:
            return h
        h+=1
    return max_h

# ------------ Linear Algebra ----------------

# construct sparse tensors
def construct_rw_sparse_tensors(graph, pellet_pos_list, no_pellet_pos_list):
    # variables
    vertices = list(graph.keys())
    num_vertices = len(vertices)
    num_pellet = len(pellet_pos_list)
    num_no_pellet = len(no_pellet_pos_list)
    
    # build index
    index_dict = {vertex: i for i, vertex in enumerate(vertices)}
    
    # initialize coo_matrices
    T = sp.coo_matrix((num_vertices, num_vertices))
    v0 = sp.coo_matrix((1, num_vertices))
    v1 = sp.coo_matrix((1, num_vertices))

    # build T
    row_indices = []
    col_indices = []
    data = []
    for i, vertex in enumerate(vertices):
        nbrs = graph[vertex]
        num_nbrs = len(nbrs)
        map_nbrs = [index_dict[v] for v in nbrs]
        row_indices.extend([i] * num_nbrs)
        col_indices.extend(map_nbrs)
        data.extend([1 / num_nbrs] * num_nbrs)

    T = sp.coo_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices)).tocsr()

    # build v1
    if num_pellet == 0:
        v1_data = []
        v1_row_indices = []
        v1_col_indices = []
    else:
        v1_data = [1 / num_pellet] * num_pellet
        v1_row_indices = [0] * num_pellet
        v1_col_indices = [index_dict[vertex] for vertex in pellet_pos_list]
    v1 = sp.coo_matrix((v1_data, (v1_row_indices, v1_col_indices)), shape=(1, num_vertices)).tocsr()

    # build v0
    if num_no_pellet == 0:
        v0_data = []
        v0_row_indices = []
        v0_col_indices = []
    else:
        v0_data = [1 / num_no_pellet] * num_no_pellet
        v0_row_indices = [0] * num_no_pellet
        v0_col_indices = [index_dict[vertex] for vertex in no_pellet_pos_list]
    v0 = sp.coo_matrix((v0_data, (v0_row_indices, v0_col_indices)), shape=(1, num_vertices)).tocsr()

    # return
    return vertices, T, v0, v1

# a function that takes power of matrix through repeated squaring
def sparse_matrix_power(A, m):
    # Initialize with the identity matrix
    result = sp.eye(A.shape[0], format='csr')
    # Use repeated squaring to calculate A^m
    while m > 0:
        if m % 2 == 1:
            result = result @ A
        A = A @ A
        m //= 2
    # return
    return result

# apply the matrix to both vectors
def apply_matrix_to_vectors(T, m, v0, v1):
    Tm = sparse_matrix_power(T, m)
    v0_new = v0 @ Tm
    v1_new = v1 @ Tm
    return v0_new, v1_new

# ------------ Render functions ----------------

# render world and agents with mayavi
def render(world, show=True, save=False, name="image_1.png"):
    # figure
    fig = mlab.figure(size=(1024, 1024))
    # Define colors based on your values
    basic = {
        1: (1, 1, 1),    # Soil
        -1: (1, 0, 0)    # Objects
        }
    gradient ={
        2: 'Greys',  # Pellets
    }
    # plotting voxels
    for value, color in basic.items():
        x_vals, y_vals, z_vals = np.where(world.grid == value)
        mlab.points3d(x_vals, y_vals, z_vals,
                        mode="cube",
                        color=color,
                        scale_factor=1)
    
    # plotting gradients
    for value, colormap in gradient.items():
        x_vals, y_vals, z_vals = np.where(world.grid == value)
        scalar_values = world.pheromones[x_vals, y_vals, z_vals]
        pts = mlab.points3d(x_vals, y_vals, z_vals,
                      scalar_values,
                      mode="cube",
                      scale_mode='none',  # To keep the original cube size
                      colormap=colormap,
                      scale_factor=1,
                      vmin = -0.3,
                      vmax = 1.3)
    # save image
    if save:
        mlab.savefig(name)
        mlab.close()
    # show image
    if show:
        mlab.show()

# ------------ Khuong functions ----------------

# skew normal distribution cdf
mod_list = skewnorm.cdf(x=np.array(range(200))/2, a=8.582, loc=2.866, scale=3.727)

# pickup rate
def eta_p(N):
    # experiment params
    n_p1 = 0.029
    if N==0:
        return n_p1
    else:
        return n_p1/N

# dropping rate
def eta_d(N):
    # experiment params
    n_d0 = 0.025
    b_d = 0.11
    if N==0:
        return n_d0
    else:
        return n_d0 + b_d*N

# pickup prob function
def prob_pickup(N):
    # see paper for formula
    prob = 1 - np.e**(-eta_p(N))
    return prob

# drop prob function
def prob_drop(N, t_now, t_latest, decay_rate, h):
    # time delta
    tau = t_now-t_latest
    # see paper for formula
    prob = 1 - np.e**(-eta_d(N)*np.e**(-tau*decay_rate))
    if h>0:
        # add vertical modulation for height h>0 in mm
        prob = prob*mod_list[h]
    # return
    return prob