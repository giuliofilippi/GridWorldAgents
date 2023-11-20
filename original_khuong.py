# base imports
import numpy as np
import pandas as pd
import time
from mayavi import mlab
from tqdm import tqdm

# classes and functions
from classes import World
from functions import (random_choices,
                       random_initial_config,
                       local_grid_data,
                       valid_moves,
                       render)

# khuong functions
from functions import (prob_pickup,
                       prob_drop,
                       compute_height)

# initialize world and agents
world = World(200, 200, 200, 20) # 200, 200, 200, 20
agent_dict = random_initial_config(world.width, world.length, world.soil_height, num_agents=500)
for agent,item in agent_dict.items():
    pos = item[0]
    world.grid[pos[0],pos[1],pos[2]] = -2

# khuong params
num_steps = 100 # should be 345600 steps (96 hours)
num_agents = 500 # number of agents
m = 6 # should be 1500 num moves per agent
lifetime = 1200 # phermone lifetime
decay_rate = 1/lifetime # decay rate

# extra params
collect_data = False
render_images = False
final_render = True
if render_images:
    mlab.options.offscreen = True

# data storage
pellet_proportion_list = []
on_floor_proportion_list = []
pellet_num = 0
total_built_volume = 0
total_built_volume_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables and generate randoms
    x_s = np.random.rand(num_agents)
    prop_on_floor = 0
    # loop over agents
    for i in range(num_agents):
        # movement rule
        for j in range(m):
            pos = agent_dict[i][0]
            x,y,z = pos
            local_data = local_grid_data(pos, world)
            moves = valid_moves(local_data)
            if len(moves)>0:
                chosen_move = random_choices(moves)[0]
                new_pos = np.array(pos)+chosen_move[1]
                # do the step
                world.grid[x,y,z] = 0
                world.grid[new_pos[0],new_pos[1],new_pos[2]] = -2
                agent_dict[i][0] = (new_pos[0], new_pos[1], new_pos[2])

        # on floor check for stats
        x,y,z = agent_dict[i][0]
        if world.grid[x,y,z-1] == 1:
            prop_on_floor += 1/num_agents

        # pickup algorithm
        if agent_dict[i][1]==0:
            pos = agent_dict[i][0]
            x,y,z = pos
            # check for material
            if world.grid[x,y,z-1] > 0:
                v26 = local_grid_data(pos, world)
                N = np.sum(v26==2)
                prob = prob_pickup(N)
                x_temp = x_s[i]
                if x_temp < prob:
                    # check if is 2
                    if world.grid[x,y,z-1]==2:
                        total_built_volume -= 1
                    # do the pickup
                    world.grid[x,y,z-1]=0
                    pellet_num+=1
                    # update pellet info
                    agent_dict[i][1] = 1

        # drop algorithm
        else:
            pos = agent_dict[i][0]
            x,y,z = pos
            v26 = local_grid_data(pos, world)
            moves = valid_moves(local_data)
            # only act if there is an available move
            if len(moves)>0:
                N = np.sum(v26==2)
                # slice lower bounds
                x_low_bound, y_low_bound, z_low_bound  = max(0, x-1), max(0, y-1), max(0, z-1)
                t_latest = np.max(world.times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
                t_now = step
                h = compute_height(pos, world)
                prob = prob_drop(N, t_now, t_latest, decay_rate, h)
                x_temp = x_s[i]
                if x_temp < prob:
                    # chosen move
                    chosen_move = random_choices(moves)[0]
                    new_pos = np.array(pos)+chosen_move[1]
                    # do the step
                    world.grid[new_pos[0],new_pos[1],new_pos[2]] = -2
                    agent_dict[i][0] = (new_pos[0], new_pos[1], new_pos[2])
                    # update total built volume
                    total_built_volume += 1
                    # do the drop
                    world.grid[x,y,z] = 2
                    pellet_num-=1
                    # update pellet info
                    agent_dict[i][1] = 0
                    # update time tensor at pos
                    world.times[x, y, z] = t_now

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        on_floor_proportion_list.append(prop_on_floor)
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
        'proportion on floor':on_floor_proportion_list,
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./data_exports/original_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world)