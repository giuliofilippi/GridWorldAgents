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
                       local_grid_data,
                       render)

# khuong functions
from functions import (prob_pickup,
                       prob_drop,
                       compute_height)

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
    new_no_pellet_num = no_pellet_num
    random_values = np.random.random((num_agents,2))
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
            if world.grid[x,y,z-1] > 0:
                v26 = local_grid_data(random_pos, world)
                N = np.sum(v26==2)
                prob = prob_pickup(N)
                x_temp = random_values[i][1]
                if x_temp < prob:
                    # check if is 2
                    if world.grid[x,y,z-1]==2:
                        total_built_volume -= 1
                    # do the pickup
                    world.grid[x,y,z-1]=0
                    # update pellet info
                    new_no_pellet_num -= 1
                    # update surface
                    surface.update_surface(type='pickup', 
                                            pos=random_pos, 
                                            world=world)
        # pellet agents
        else:
            # drop algorithm
            v26 = local_grid_data(random_pos, world)
            N = np.sum(v26==2)
            # slice lower bounds
            x_low_bound, y_low_bound, z_low_bound  = max(0, x-1), max(0, y-1), max(0, z-1)
            t_latest = np.max(world.times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
            t_now = step
            h = compute_height(random_pos, world)
            prob = prob_drop(N, t_now, t_latest, decay_rate, h)
            x_temp = random_values[i][1]
            if x_temp < prob:
                # update total built volume
                total_built_volume += 1
                # do the drop
                world.grid[x,y,z] = 2
                # update pellet info
                new_no_pellet_num += 1
                # update time tensor at pos
                world.times[x, y, z] = t_now
                # update surface
                surface.update_surface(type='drop', 
                                            pos=random_pos, 
                                            world=world)           
    # update variables
    no_pellet_num = new_no_pellet_num

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