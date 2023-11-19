# imports
import numpy as np
from classes import World
from functions import (render, 
                          random_choice,
                          local_grid_data,
                          compute_height,
                          prob_pickup,
                          prob_drop,
                          update_surface)
import time
from mayavi import mlab
#mlab.options.offscreen = True

# initialize
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = [(x,y,world.soil_height) for x in range(world.width) for y in range(world.length)]
print ('World and Surface Initialized')

# khuong params
num_steps = 1000 # should be 345600 steps (96 hours)
spontpick = 0.01 # spontaneous pickup prob
amplifpick = 1 # pickup amplification
spontdrop = 0.0001 # spontaneous drop prob
drop1 = 0.001 # initial drop prob for 1 voxel
amplifdrop = 0.036 # drop amplification
evap = 1.6 * 10**-5 # evaporation of pheromone
grid_shape = world.grid.shape # shape of grid world
times = np.zeros(world.grid.shape) # time tensor
num_agents = 500 # number of agents
no_pellet_num = 500 # number of agents with no pellet

# start time
start_time = time.time()
# loop over time steps
for step in range(num_steps):
    print(step)
    # generate random values
    random_values = np.random.random((num_agents,2))
    new_no_pellet_num = no_pellet_num
    # loop over agents
    for i in range(num_agents):
        # random position
        random_pos = np.array(random_choice(surface))
        # no pellet agents
        if i < no_pellet_num:
            # pickup algorithm
            x,y,z = random_pos
            if world.grid[x,y,z-1] > 0:
                v26 = local_grid_data(random_pos, world)
                N = np.sum(v26==2)
                prob = prob_pickup(N, spontpick, amplifpick)
                x_temp = random_values[i][1]
                if x_temp < prob:
                    # do the pickup
                    world.grid[x,y,z-1]=0
                    # update pellet info
                    new_no_pellet_num -= 1
                    # update surface
                    surface = update_surface(surface=surface, 
                                         type='pickup', 
                                         pos=random_pos, 
                                         world=world)
        # pellet agents
        else:
            # drop algorithm
            x,y,z = random_pos
            v26 = local_grid_data(random_pos, world)
            N = np.sum(v26==2)
            # slice lower bounds
            x_low_bound = max(0, x-1)
            y_low_bound = max(0, y-1)
            z_low_bound = max(0, z-1)
            t_latest = np.max(times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
            t = step
            h = compute_height(random_pos, world)
            prob = prob_drop(N, t, t_latest, spontdrop, drop1, amplifdrop, h, evap)
            x_temp = random_values[i][1]
            if x_temp < prob:
                # do the drop
                world.grid[x,y,z] = 2
                # update pellet info
                new_no_pellet_num += 1
                # update time tensor
                times[x, y, z] = t
                # update surface
                surface = update_surface(surface=surface, 
                                         type='drop', 
                                         pos=random_pos, 
                                         world=world)
                
    # update variable
    no_pellet_num = new_no_pellet_num

    # render every minute
    if step % 60 == 0:
        #render(world, show=False, save=True, name="animation/image_{}.png".format(step+1))
        pass

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)
render(world)