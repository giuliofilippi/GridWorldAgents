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
                       render)

# algorithms
from algorithms import pickup_algorithm, drop_algorithm_graph

# initialize
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

# khuong params
num_steps = 100 # should be 345600 steps (96 hours)
num_agents = 500 # number of agents
no_pellet_num = 500 # number of agents with no pellet
lifetime = 1200
decay_rate = 1/lifetime

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
total_built_volume = 0
total_built_volume_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables and generate random values
    prop_on_floor = 0
    random_values = np.random.random(num_agents)
    # generate random positions synchronously
    vertex_list = list(surface.graph.keys())
    p = surface.get_rw_stationary_distribution()
    random_positions = random_choices(vertex_list, size=num_agents, p=p)
    # loop over agents
    for i in range(num_agents):
        # random position
        random_pos = random_positions[i]
        x,y,z = random_pos

        # on floor check for stats
        if world.grid[x,y,z-1] == 1:
            prop_on_floor += 1/num_agents
        
        # no pellet agents
        if i < no_pellet_num:
            # pickup algorithm
            pos = random_pos
            material = pickup_algorithm(pos, world, x_rand=random_values[i])
            if material is not None:
                # make data updates
                no_pellet_num -= 1
                if material == 2:
                    total_built_volume +=1
                surface.update_surface(type='pickup', 
                                            pos=random_pos, 
                                            world=world)

        # pellet agents
        else:
            pos = random_pos
            new_pos = drop_algorithm_graph(pos, world, surface.graph, step, decay_rate, x_rand = random_values[i])
            if new_pos is not None:
                # update data and surface
                total_built_volume += 1
                no_pellet_num += 1
                surface.update_surface(type='drop', 
                                            pos=random_pos, 
                                            world=world)

    # collect data
    if collect_data:
        pellet_proportion_list.append((num_agents-no_pellet_num)/num_agents)
        floor_proportion_list.append(prop_on_floor)
        total_surface_area_list.append(len(surface.graph.keys()))
        total_built_volume_list.append(total_built_volume)

    # render images
    if render_images:
        # every 5 minutes
        if step % 300 == 0:
            render(world, show=False, save=True, name="animation/image_{}.png".format(step+1))

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)

# export pandas
if collect_data:
    steps = np.array(range(num_steps))
    data_dict = {
        'steps':steps,
        'proportion pellet':pellet_proportion_list,
        'proportion floor':floor_proportion_list,
        'surface area':total_surface_area_list,
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./data_exports/mean_field_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world)