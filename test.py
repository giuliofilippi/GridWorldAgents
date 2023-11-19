# imports
from classes import (World, 
                     Agent, 
                     RandomPolicy)

from functions import render
import numpy as np
import time
from mayavi import mlab
# mlab.options.offscreen = True

# object
obj = np.zeros((200,200,200))
obj[95:105, 95:105, 20:100] = 1

# Initialize the environment, policy and agents
world = World(width=200, length=200, height=200, soil_height=20, objects=[obj])
agents = [Agent(world) for i in range(500)]
policy = RandomPolicy()

# start time
start_time = time.time()
# 1000 random steps
for step in range(1000):
    print (step)
    for agent in agents:
        action_space = agent.get_valid_actions(world)
        action = policy.choose_action(action_space)
        world.step(agent, action)
    
    # export render every minute
    '''
    if step % 60 == 0:
        render(world, show=False, save=True, name="animation/image_{}.png".format(step+1))
    '''

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)
render(world, show=True)