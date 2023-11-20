# imports
import numpy as np
from collections import OrderedDict
from functions import (update_surface_function,
                       construct_rw_sparse_matrix)

# World class
class World:
    # init
    def __init__(self, width, length, height, soil_height, objects=None):
        # attributes: width, lenght, height, soil_height, objects, grid, times, field, pheromones
        self.width = width
        self.length = length
        self.height = height
        self.soil_height = soil_height
        self.objects = objects
        self.grid = np.zeros((width, length, height), dtype=int)
        self.times = np.zeros((width, length, height), dtype=int)
        self.field = np.zeros((width, length, height), dtype=int)
        self.pheromones = np.zeros((width, length, height), dtype=int)
        # insert soil
        self.grid[:, :, :soil_height] = 1  # Soil
        # insert objects
        if self.objects is not None:
            for obj in self.objects:
                if obj.shape == self.grid.shape:
                    self.grid[obj == 1] = -1  # Object
    
    # can diffuse any tensor using zero gradient boundary condition
    def diffuse_tensor(self, tensor, diffusion_rate, num_iterations=1):
        # usually 1 iteration
        for _ in range(num_iterations):
            # copy to avoid overwrite
            new_tensor = tensor.copy()
            # loop, loop, loop
            for i in range(1, self.width - 1):
                for j in range(1, self.length - 1):
                    for k in range(1, self.height - 1):
                        concentration_change = 0
                        for nbr in [(i + 1, j, k), (i - 1, j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1), (i, j, k - 1)]:
                            # can only diffuse to empty neighbour cells
                            if self.grid[nbr[0], nbr[1], nbr[2]]==0:
                                concentration_change += diffusion_rate * (tensor[nbr[0], nbr[1], nbr[2]]-tensor[i,j,k])
                        new_tensor[i, j, k] += concentration_change
            # Apply zero-gradient boundaries
            new_tensor[0, :, :] = new_tensor[1, :, :]
            new_tensor[-1, :, :] = new_tensor[-2, :, :]
            new_tensor[:, 0, :] = new_tensor[:, 1, :]
            new_tensor[:, -1, :] = new_tensor[:, -2, :]
            new_tensor[:, :, 0] = new_tensor[:, :, 1]
            new_tensor[:, :, -1] = new_tensor[:, :, -2]
            # update
            tensor = new_tensor
        # return
        return tensor

# Surface class
class Surface:
    # init
    def __init__(self, graph):
        # attributes: graph, degrees, num_edges
        self.graph = OrderedDict(graph)
        self.degrees = OrderedDict()
        self.num_edges = 0
        for v in self.graph.keys():
            deg = len(self.graph[v])
            self.degrees[v] = deg
            self.num_edges += deg/2

    # number of vertices
    def get_num_vertices(self):
        return(len(self.graph.keys()))
    
    # number of edges
    def get_num_edges(self):
        return(np.sum([len(self.graph[v]) for v in self.graph.keys()])/2)
    
    # add an edge to graph
    def add_edge(self, pair):
        v1, v2 = pair
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)
        self.degrees[v1]+=1
        self.degrees[v2]+=1
        self.num_edges +=1

    # remove vertex from graph
    def remove_vertex(self, vertex):
        nbrs = self.graph[vertex]
        for nbr in nbrs:
            self.graph[nbr].remove(vertex)
            self.degrees[nbr]-=1
            self.num_edges -=1
        del self.graph[vertex]
        del self.degrees[vertex]
    
    # get sparse matrix T, sparse vectors v0, v1 and vertex_list (Random Walk)
    def get_rw_sparse_matrix(self, no_pellet_pos_list, pellet_pos_list):
        index_dict, vertices, T = construct_rw_sparse_matrix(self.graph, 
                                                          no_pellet_pos_list,
                                                          pellet_pos_list 
                                                       )
        return index_dict, vertices, T
    
    # get stationary distribution (Random Walk)
    def get_rw_stationary_distribution(self):
        p = np.array([self.degrees[v] for v in self.graph.keys()])
        p = p/(2*self.num_edges)
        return p
    
    # update surface
    def update_surface(self, type, pos, world):
        # see functions for details
        update_surface_function(self, type, pos, world)