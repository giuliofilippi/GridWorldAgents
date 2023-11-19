# imports
import numpy as np
from classes import World, Agent
from functions import (render, 
                          random_choice, 
                          valid_move_directions, 
                          compute_height,
                          prob_drop,
                          prob_pickup)
import time
from mayavi import mlab
#mlab.options.offscreen = True

# initialize
world = World(200, 200, 200, 20) # 200, 200, 200, 20
agents = [Agent(world) for i in range(500)] # 500

# parameters
num_steps = 100 # should be 345600 (96h)
m = 1500 # should be 1500 (see paper)
spontpick = 0.01 # spontaneous pickup prob
amplifpick = 1 # pickup amplification
spontdrop = 0.0001 # spontaneous drop prob
drop1 = 0.001 # initial drop prob for 1 voxel
amplifdrop = 0.036 # drop amplification
evap = 1.6 * 10**-5 # evaporation of pheromone
grid_shape = world.grid.shape # shape of grid world
times = np.zeros(world.grid.shape) # time tensor
num_agents = len(agents) # number of agents

# start time
start_time = time.time()
# loop over time steps
for step in range(num_steps):
    # print step and generate randoms
    print(step)
    x_s = np.random.rand(num_agents)
    # render every minute
    if step % 60 == 0:
        #render(world, show=False, save=True, name="animation/image_{}.png".format(step+1))
        pass
    # loop over agents
    for i,agent in enumerate(agents):
        # movement rule
        for j in range(m):
            moves = agent.get_valid_moves(world)
            if len(moves)>0:
                chosen_move = random_choice(moves)
                world.step(agent, chosen_move)

        # pickup rule
        if agent.has_pellet is False:
            x,y,z = agent.pos
            if world.grid[x,y,z-1] > 0:
                v26 = agent.get_local_grid_data(world)
                N = np.sum(v26==2)
                prob = prob_pickup(N, spontpick, amplifpick)
                x_temp = x_s[i]
                # pickup event
                if x_temp < prob:
                    chosen_move = ('pickup',np.array((0,0,-1)))
                    world.step(agent, chosen_move)
        
        # drop rule
        if agent.has_pellet is True:
            x, y ,z = agent.pos
            v26 = agent.get_local_grid_data(world)
            N = np.sum(v26==2)
            # slice lower bounds
            x_low_bound = max(0, x-1)
            y_low_bound = max(0, y-1)
            z_low_bound = max(0, z-1)
            t_latest = np.max(times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
            t = step
            h = compute_height(agent.pos, world)
            prob = prob_drop(N, t, t_latest, spontdrop, drop1, amplifdrop, h, evap)
            x_temp = x_s[i]
            # drop event
            if x_temp < prob:
                moves = valid_move_directions(v26)
                move = random_choice(moves)
                chosen_move = ('drop',move)
                times[x, y, z] = t
                world.step(agent, chosen_move)

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)
render(world)