# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# classes and functions
from classes import World, Surface
from functions import (get_initial_graph,
                       random_choices)

# algorithms
from khuong_algorithms import pickup_algorithm, drop_algorithm_graph

# initialize
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))

# khuong params
num_steps = 345600 # should be 345600 steps (96 hours)
num_agents = 500 # number of agents
pellet_num = 0 # number of agents with pellet in beginning
lifetime = 1000
decay_rate = 1/lifetime

# extra params
collect_data = True
render_images = True
final_render = False
if final_render:
    from render import render

# data storage
total_built_volume = 0
pellet_proportion_list = []
total_surface_area_list = []
total_built_volume_list = []
pickup_rate_list = []
drop_rate_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables and generate random values
    random_values = np.random.random(num_agents)
    # generate random positions synchronously
    vertex_list = list(surface.graph.keys())
    p = surface.get_rw_stationary_distribution()
    random_positions = random_choices(vertex_list, size=num_agents, p=p)
    # no pellet num for cycle
    no_pellet_num_cycle, pellet_num_cycle = num_agents-pellet_num, pellet_num
    # pickup and drop rates
    pickup_rate = 0
    drop_rate = 0
    # loop over permuted agents
    permutation = np.random.permutation(num_agents)
    for i in permutation:
        # random position
        random_pos = random_positions[i]
        x,y,z = random_pos
        
        # no pellet agents
        if i < no_pellet_num_cycle:
            # pickup algorithm
            pos = random_pos
            material = pickup_algorithm(pos, world, x_rand=random_values[i])
            if material is not None:
                # make data updates
                pellet_num += 1
                pickup_rate += 1/no_pellet_num_cycle
                if material == 2:
                    total_built_volume -=1
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
                pellet_num -= 1
                drop_rate += 1/pellet_num_cycle
                surface.update_surface(type='drop', 
                                            pos=random_pos, 
                                            world=world)

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        total_surface_area_list.append(len(surface.graph.keys()))
        total_built_volume_list.append(total_built_volume)
        pickup_rate_list.append(pickup_rate)
        drop_rate_list.append(drop_rate)

    # render images
    if render_images:
        if step % (60*60) == 0:
            np.save(file="./exports/tensors/meanfield_{}".format(step+1), arr=world.grid)

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)

# export pandas
if collect_data:
    steps = np.array(range(num_steps))
    params = ['num_steps={}'.format(num_steps),
              'num_agents={}'.format(num_agents),
              'lifetime={}'.format(lifetime),
              'runtime={}s'.format(int(end_time - start_time))]+['']*(num_steps-4)
    data_dict = {
        'params':params,
        'steps':steps,
        'proportion_pellet':pellet_proportion_list,
        'pickup_rate':pickup_rate_list,
        'drop_rate':drop_rate_list,
        'surface_area':total_surface_area_list,
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./exports/dataframes/meanfield_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world.grid)