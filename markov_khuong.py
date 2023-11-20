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
                       dual_random_choices,
                       local_grid_data,
                       get_neighbours,
                       apply_matrix_to_vectors,
                       render)

# khuong functions
from functions import (prob_pickup,
                       prob_drop,
                       compute_height)

# initialize world and surface
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

# khuong params
num_steps = 100 # should be 345600 steps (for 96 hours)
num_agents = 500 # number of agents
m = 6 # num moves per agent
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
    # compute numbers
    np_num, p_num = len(np_agents), len(p_agents)
    # generate random positions from Markov Synchronously
    vertices, T, v0, v1 = surface.get_rw_sparse_tensors(np_agents, p_agents)
    v0_new, v1_new = apply_matrix_to_vectors(T, m, v0, v1)
    new_np_agents, new_p_agents = dual_random_choices(vertices, np_num, p_num, v0_new, v1_new)
    # generate random values for later sampling
    random_values_0 = np.random.random(np_num)
    random_values_1 = np.random.random(p_num)
    # create copies
    new_np_agents_copy = new_np_agents.copy()
    new_p_agents_copy = new_p_agents.copy()

    # pickup algorithm
    for i in range(np_num):
        # position
        random_pos = new_np_agents[i]
        x,y,z = random_pos
        # on floor check for stats
        if world.grid[x,y,z-1] == 1:
            prop_on_floor += 1/num_agents
        # check for material
        if world.grid[x,y,z-1] > 0:
            # local data
            v26 = local_grid_data(random_pos, world)
            N = np.sum(v26==2)
            prob = prob_pickup(N)
            x_temp = random_values_0[i]
            # random sample
            if x_temp < prob:
                # check if is 2
                if world.grid[x,y,z-1]==2:
                    total_built_volume -= 1
                # do the pickup
                world.grid[x,y,z-1]=0
                # update agent lists
                new_np_agents_copy.remove((x,y,z))
                new_p_agents_copy.append((x,y,z))
                # update data
                pellet_num += 1
                # update surface
                surface.update_surface(type='pickup', 
                                            pos=random_pos, 
                                            world=world)
                
    # drop algorithm
    for j in range(p_num):
        # position
        random_pos = new_p_agents[j]
        x,y,z = random_pos
        # on floor check for stats
        if world.grid[x,y,z-1] == 1:
            prop_on_floor += 1/num_agents
        # first check for neighbours
        nbrs = get_neighbours((x,y,z), surface.graph)
        if len(nbrs)>0:
            # local data
            v26 = local_grid_data(random_pos, world)
            N = np.sum(v26==2)
            # slice the times tensor
            x_low_bound, y_low_bound, z_low_bound  = max(0, x-1), max(0, y-1), max(0, z-1)
            t_latest = np.max(world.times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
            t_now = step
            # get height
            h = compute_height(random_pos, world)
            prob = prob_drop(N, t_now, t_latest, decay_rate, h)
            x_temp = random_values_1[j]
            if x_temp < prob:
                total_built_volume+=1
                # chosen place to move
                chosen_nbr = random_choices(nbrs,1)[0]
                # do the drop
                world.grid[x,y,z] = 2
                # update time tensor at pos
                world.times[x, y, z] = t_now
                # update data
                pellet_num -= 1
                # update agent lists
                new_p_agents_copy.remove((x,y,z))
                new_np_agents_copy.append(chosen_nbr)
                # update surface
                surface.update_surface(type='drop', 
                                            pos=random_pos, 
                                            world=world)

    # reset variables for next loop
    np_agents = new_np_agents_copy
    p_agents = new_p_agents_copy

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        floor_proportion_list.append(prop_on_floor)
        total_surface_area_list.append(len(surface.graph.keys()))
        total_built_volume_list.append(total_built_volume)

    # if render images
    if render_images:
        # every 5 minutes
        if step % 300 == 0:
            # export image
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
    df.to_pickle('./data_exports/markov_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world)