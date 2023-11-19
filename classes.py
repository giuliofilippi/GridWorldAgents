# imports
import numpy as np
from functions import (random_choice,
                          local_grid_data,
                          valid_actions,
                          valid_moves)

# World class
class World:
    # init
    def __init__(self, width, length, height, soil_height, objects=None):
        # attributes:
        # width, lenght, height, soil_height, objects, grid, drop_times
        self.width = width
        self.length = length
        self.height = height
        self.soil_height = soil_height
        self.objects = objects
        self.grid = np.zeros((width, length, height), dtype=int)
        self.drop_times = np.zeros((width, length, height), dtype=int)
        # insert soil
        self.grid[:, :, :soil_height] = 1  # Soil
        # insert objects
        if self.objects is not None:
            for obj in self.objects:
                if obj.shape == self.grid.shape:
                    self.grid[obj == 1] = -1  # Object

    # implement step
    def step(self, agent, action):
        # Update environment and agent based on action by some agent
        if action is None:
            return None
        elif action[0] == 'move':
            dir = action[1] # direction of action
            new_pos = agent.pos+dir # new agent position
            self.grid[agent.pos[0],agent.pos[1],agent.pos[2]] = 0 # remove agent
            agent.pos = new_pos # change agent pos
            agent.place_in_world(self) # place agent in world
            return None
        elif action[0] == 'pickup':
            agent.has_pellet = True # has pellet
            pos = agent.pos # agent position
            new_pos = np.array([pos[0],pos[1],pos[2]-1]) # new agent position
            self.grid[new_pos[0],new_pos[1],new_pos[2]]=0 # remove soil
            self.grid[pos[0],pos[1],pos[2]]=0 # remove agent
            agent.pos = new_pos # change agent pos
            agent.place_in_world(self) # move agent
            return None
        elif action[0] == 'drop':
            agent.has_pellet = False # no pellet
            pos = agent.pos # agent position
            self.grid[pos[0],pos[1],pos[2]] = 2 # place pellet
            agent.pos += action[1]
            agent.place_in_world(self) # move agent
            return None
        else:
            raise ValueError
        
    # reset
    def reset(self):
        self.grid = np.zeros((self.width, self.length, self.height), dtype=int)
        self.drop_times = np.zeros((self.width, self.length, self.height), dtype=int)
        # insert soil
        self.grid[:, :, :self.soil_height] = 1  # Soil
        # insert objects
        if self.objects is not None:
            for obj in self.objects:
                if obj.shape == self.grid.shape:
                    self.grid[obj == 1] = -1  # Object


# Agent class
class Agent:
    # init
    def __init__(self, world):
        # attributes: pos, has_pellet
        no_obj = np.where(world.grid[:,:,world.soil_height]==0)
        x = np.random.choice(no_obj[0])
        y = np.random.choice(no_obj[1])
        z = world.soil_height
        self.pos = np.array([x,y,z])
        self.has_pellet = False
        world.grid[x,y,z] = -2
    
    # places agent in world at their location
    def place_in_world(self, world):
        x,y,z = self.pos
        world.grid[x,y,z] = -2

    # get 3*3*3 local grid data
    def get_local_grid_data(self, world):
        # function in helper_tools
        return local_grid_data(self.pos, world)

    # get valid actions at loc
    def get_valid_actions(self, world):
        # inputs
        local_data = self.get_local_grid_data(world)
        has_pellet = self.has_pellet
        # function in helper_tools
        return valid_actions(local_data, has_pellet)
    
    # get only valid moves only at loc
    def get_valid_moves(self, world):
        # input
        local_data = self.get_local_grid_data(world)
        # function in helper_tools
        return valid_moves(local_data)
    
    # reset agent to random initial state
    def reset(self, world):
        # attributes: pos, has_pellet
        no_obj = np.where(world.grid[:,:,world.soil_height]==0)
        x = np.random.choice(no_obj[0])
        y = np.random.choice(no_obj[1])
        z = world.soil_height
        self.pos = np.array([x,y,z])
        self.has_pellet = False
        world.grid[x,y,z] = -2


# Policy class
class Policy:
    # init
    def __init__(self):
        # initialize chosen model based on observation data shape and action space.
        raise NotImplementedError

    def choose_action(self):
        # Implement a policy for action selection based on the observation data.
        raise NotImplementedError

    def learn(self):
        # Implement the learning algorithm here.
        raise NotImplementedError


# This policy picks a random valid action for each agent.
class RandomPolicy(Policy):
    # init
    def __init__(self):
        pass

    def choose_action(self, action_space):
        # Random move
        if len(action_space)>0:
            return random_choice(action_space)
        else:
            return None