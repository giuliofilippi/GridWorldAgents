# sys
import sys
sys.path.append('code')

# profiling
import cProfile

# base imports
import numpy as np
import pandas as pd
import time
from mayavi import mlab
from tqdm import tqdm

# classes and functions
from classes import World, Surface
from functions import (get_initial_graph,
                       random_choices,
                       conditional_random_choice,
                       construct_rw_sparse_matrix,
                       sparse_matrix_power,
                       render)

# khuong functions
from khuong.khuong_algorithms import pickup_algorithm, drop_algorithm_graph

def code_to_profile():
    # initialize world and surface
    world = World(200, 200, 200, 20) # 200, 200, 200, 20
    surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

    # khuong params
    num_steps = 100 # should be 345600 steps (for 96 hours)
    num_agents = 500 # number of agents
    m = 4 # num moves per agent
    lifetime = 1200 # pheromone lifetime in seconds
    decay_rate = 1/lifetime # decay rate nu_m

    # initialize agent lists
    p_agents = []
    np_agents = random_choices(list(surface.graph.keys()), num_agents)

    # extra params
    collect_data = False
    render_images = False
    final_render = True
    if render_images:
        mlab.options.offscreen = True

    # data storage
    pellet_proportion_list = []
    floor_proportion_list = []
    total_surface_area_list = []
    pellet_num = 0
    total_built_volume = 0
    total_built_volume_list = []

    # start time
    start_time = time.time()
    # loop over time steps
    for step in tqdm(range(num_steps)):
        # reset variables
        prop_on_floor = 0
        np_num, p_num = len(np_agents), len(p_agents)
        new_np_agents = []
        new_p_agents = []
        removed_indices = []
        random_values_0 = np.random.random(np_num)
        random_values_1 = np.random.random(p_num)
        # create transition matrix and take power
        index_dict, vertices, T = construct_rw_sparse_matrix(surface.graph)
        Tm = sparse_matrix_power(T, m)

        # pickup algorithm
        for i in range(np_num):
            # position
            prob_dist = Tm[index_dict[np_agents[i]]].toarray().flatten()
            random_pos = conditional_random_choice(vertices,
                                                    p = prob_dist, 
                                                    removed_indices=removed_indices)
            # remove position from index
            removed_indices.append(index_dict[random_pos])
            
            # on floor check for stats
            x,y,z = random_pos
            if world.grid[x,y,z-1] == 1:
                prop_on_floor += 1/num_agents

            # pickup algorithm
            material = pickup_algorithm(random_pos, world, x_rand=random_values_0[i])
            if material is not None:
                # update data and surface
                pellet_num += 1
                new_p_agents.append((x,y,z))
                surface.update_surface(type='pickup', 
                                        pos=random_pos, 
                                        world=world)
            else:
                new_np_agents.append((x,y,z))
    
        # drop algorithm
        for j in range(p_num):
            # position
            prob_dist = Tm[index_dict[p_agents[j]]].toarray().flatten()
            random_pos = conditional_random_choice(vertices,
                                                    p = prob_dist, 
                                                    removed_indices=removed_indices)
            # remove position from index
            removed_indices.append(index_dict[random_pos])

            # on floor check for stats
            x, y, z = random_pos
            if world.grid[x,y,z-1] == 1:
                prop_on_floor += 1/num_agents

            # drop algorithm
            new_pos = drop_algorithm_graph(random_pos, world, surface.graph, step, decay_rate, x_rand = random_values_1[j])
            if new_pos is not None:
                # update data
                pellet_num -= 1
                new_np_agents.append(new_pos)
                # also remove new position if it is in index dict
                if new_pos in index_dict:
                    removed_indices.append(index_dict[new_pos])
                # update surface
                surface.update_surface(type='drop', 
                                        pos=random_pos, 
                                        world=world)
            else:
                # append to p agents because didnt drop
                new_p_agents.append(random_pos)

        # reset variables for next loop
        np_agents = new_np_agents
        p_agents = new_p_agents

        # collect data
        if collect_data:
            pellet_proportion_list.append(pellet_num/num_agents)
            floor_proportion_list.append(prop_on_floor)
            total_surface_area_list.append(len(surface.graph.keys()))
            total_built_volume_list.append(total_built_volume)

        # if render images
        if render_images:
            # every 5 minutes
            if step % (5*60) == 0:
                # export image
                render(world, show=False, save=True, name="animation_folder/markov_{}.png".format(step+1))

    # end time
    end_time = time.time()
    print("total time taken for this loop: ", end_time - start_time)


cProfile.run("code_to_profile()")