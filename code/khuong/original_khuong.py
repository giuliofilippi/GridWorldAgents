# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time
from mayavi import mlab
from tqdm import tqdm

# classes and functions
from classes import World
from functions import (random_initial_config,
                       render)

# algorithms
from khuong_algorithms import (
    move_algorithm,
    move_algorithm_new,
    move_algorithm_ghost,
    pickup_algorithm,
    drop_algorithm,
    drop_algorithm_new)

# initialize world and agents
world = World(200, 200, 200, 20) # 200, 200, 200, 20
agent_dict = random_initial_config(world.width, world.length, world.soil_height, num_agents=500)
for agent,item in agent_dict.items():
    pos = item[0]
    world.grid[pos[0],pos[1],pos[2]] = -2

# khuong params
num_steps = 1000 # should be 345600 steps (96 hours)
num_agents = 500 # number of agents
m = 20 # should be 1500 num moves per agent
lifetime = 1200 # phermone lifetime
decay_rate = 1/lifetime # decay rate

# extra params
collect_data = True
render_images = False
final_render = False
if render_images:
    mlab.options.offscreen = True

# data storage
pellet_num = 0
total_built_volume = 0
pellet_proportion_list = []
total_built_volume_list = []
pickup_rate_list = []
drop_rate_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables and generate randoms
    x_random = np.random.rand(num_agents)
    # no pellet num for cycle
    no_pellet_num_cycle, pellet_num_cycle = num_agents-pellet_num, pellet_num
    # pickup and drop rates
    pickup_rate = 0
    drop_rate = 0
    # loop over agents
    for i in range(num_agents):
        # movement rule
        pos = agent_dict[i][0]
        final_pos = move_algorithm_ghost(pos, world, m)
        x,y,z = final_pos
        agent_dict[i][0] = (x, y, z)

        # pickup algorithm
        if agent_dict[i][1]==0:
            pos = agent_dict[i][0]
            material = pickup_algorithm(pos, world, x_rand=x_random[i])
            if material is not None:
                # make data updates
                pellet_num += 1
                pickup_rate += 1/no_pellet_num_cycle
                agent_dict[i][1] = 1
                if material == 2:
                    total_built_volume -=1

        # drop algorithm
        else:
            pos = agent_dict[i][0]
            new_pos = drop_algorithm_new(pos, world, step, decay_rate, x_rand = x_random[i])
            if new_pos is not None:
                # make data updates
                world.grid[new_pos[0],new_pos[1],new_pos[2]] = -2
                pellet_num -= 1
                drop_rate += 1/pellet_num_cycle
                agent_dict[i] = [new_pos, 0]
                total_built_volume += 1

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        total_built_volume_list.append(total_built_volume)
        pickup_rate_list.append(pickup_rate)
        drop_rate_list.append(drop_rate)

    # render images
    if render_images:
        # every minute
        if step % (5*60) == 0:
            render(world, show=False, save=True, name="./exports_image/original_{}.png".format(step+1))

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)

# export pandas
if collect_data:
    steps = np.array(range(num_steps))
    params = ['num_steps={}'.format(num_steps),
              'num_agents={}'.format(num_agents),
              'm={}'.format(m),
              'lifetime={}'.format(lifetime),
              'runtime={}s'.format(int(end_time - start_time))]+['']*(num_steps-5)
    data_dict = {
        'params':params,
        'steps':steps,
        'proportion_pellet':pellet_proportion_list,
        'pickup_rate':pickup_rate_list,
        'drop_rate':drop_rate_list,
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./exports_data/original_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world)