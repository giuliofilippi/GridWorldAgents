# imports
import numpy as np
from collections import OrderedDict
from functions import (update_surface_function,
                       construct_rw_sparse_matrix)

# World class
class World:
    """
    Represents the environment in which agents and objects interact.

    Attributes:
    - width: Width of the space.
    - length: Length of the space.
    - height: Height of the space.
    - soil_height: Height of the soil.
    - objects: Array representing objects in the world.
    - grid: 3D array representing the state of the world.
    - times: 3D array representing time information for each voxel.
    - field: 3D array representing the field in the world.
    - pheromones: 3D array representing pheromone concentrations in the world.

    Methods:
    - __init__: Initializes a new instance of the World class.
    - diffuse_tensor: Diffuses a given tensor in the world.
    """
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
    
    # can diffuse any tensor in world using zero gradient boundary condition
    def diffuse(self, tensor, diffusion_rate, num_iterations=1):
        """
        Diffuses a given tensor in the world.

        Parameters:
        - tensor: The tensor to be diffused.
        - diffusion_rate: Rate of diffusion.
        - num_iterations: Number of diffusion iterations.

        Returns:
        - The diffused tensor.
        """
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
    
    # move method
    def move_agent(self, old_pos, new_pos):
        self.grid[old_pos[0],old_pos[1],old_pos[2]] = 0
        self.grid[new_pos[0],new_pos[1],new_pos[2]] = -2

    # pickup method
    def pickup(self, pos):
        self.grid[pos[0],pos[1],pos[2]-1] = 0

    # drop method
    def drop(self, pos):
        self.grid[pos[0],pos[1],pos[2]] = 2


# Surface class
class Surface:
    """
    Represents the surface graph of the environment.

    Attributes:
    - graph: Dictionary representing the graph structure.
    - degrees: Dictionary representing the degrees of vertices in the graph.
    - num_edges: Number of edges in the graph.

    Methods:
    - __init__: Initializes a new instance of the Surface class.
    - get_num_vertices: Returns the number of vertices in the graph.
    - get_num_edges: Returns the number of edges in the graph.
    - add_edge: Adds an edge to the graph.
    - remove_vertex: Removes a vertex from the graph.
    - get_rw_sparse_matrix: Gets the Random Walk sparse matrix.
    - get_rw_stationary_distribution: Gets the stationary distribution for Random Walk.
    - update_surface: Updates the surface based on a specific action type.
    """
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
        """
        Returns the number of vertices in the graph.

        Returns:
        - Number of vertices.
        """
        return(len(self.graph.keys()))
    
    # number of edges
    def get_num_edges(self):
        """
        Returns the number of edges in the graph.

        Returns:
        - Number of edges.
        """
        return(np.sum([len(self.graph[v]) for v in self.graph.keys()])/2)
    
    # add an edge to graph
    def add_edge(self, pair):
        """
        Adds an edge to the graph.

        Parameters:
        - pair: Tuple representing the edge.

        Returns:
        - None.
        """
        v1, v2 = pair
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)
        self.degrees[v1]+=1
        self.degrees[v2]+=1
        self.num_edges +=1

    # remove vertex from graph
    def remove_vertex(self, vertex):
        """
        Removes a vertex from the graph.

        Parameters:
        - vertex: Vertex to be removed.

        Returns:
        - None.
        """
        nbrs = self.graph[vertex]
        for nbr in nbrs:
            self.graph[nbr].remove(vertex)
            self.degrees[nbr]-=1
            self.num_edges -=1
        del self.graph[vertex]
        del self.degrees[vertex]
    
    # get sparse matrix T, sparse vectors v0, v1 and vertex_list (Random Walk)
    def get_rw_sparse_matrix(self):
        """
        Gets the Random Walk sparse matrix.

        Parameters:

        Returns:
        - Index dictionary, vertices list, and the sparse matrix T.
        """
        index_dict, vertices, T = construct_rw_sparse_matrix(self.graph)
        return index_dict, vertices, T
    
    # get stationary distribution (Random Walk)
    def get_rw_stationary_distribution(self):
        """
        Gets the stationary distribution for Random Walk.

        Returns:
        - Stationary distribution as a numpy array.
        """
        p = np.array([self.degrees[v] for v in self.graph.keys()])
        p = p/(2*self.num_edges)
        return p
    
    # update surface
    def update_surface(self, type, pos, world):
        """
        Updates the surface based on a specific action type.

        Parameters:
        - type: Type of action ('pickup' or 'drop').
        - pos: Position in the world.
        - world: The world object.

        Returns:
        - None
        """
        # see functions for details
        update_surface_function(self, type, pos, world)