# imports
import numpy as np

# classes and functions
from code.classes import World
from code.functions import (random_choices,
                       local_grid_data,
                       get_neighbours,
                       valid_moves,
                       compute_height)

# skew normal distribution cdf
from scipy.stats import skewnorm
mod_list = skewnorm.cdf(x=np.array(range(200))/2, a=8.582, loc=2.866, scale=3.727)

# ------------ Khuong functions ----------------
# ----------------------------------------------

# pickup rate
def eta_p(N):
    """
    Calculates the pickup rate.

    Parameters:
    - N: Number of particles.

    Returns:
    - Pickup rate.
    """
    # experiment params
    n_p1 = 0.029
    if N==0:
        return n_p1
    else:
        return n_p1/N

# dropping rate
def eta_d(N):
    """
    Calculates the dropping rate.

    Parameters:
    - N: Number of particles.

    Returns:
    - Dropping rate.
    """
    # experiment params
    n_d0 = 0.025
    b_d = 0.11
    if N==0:
        return n_d0
    else:
        return n_d0 + b_d*N

# pickup prob function
def prob_pickup(N):
    """
    Calculates the probability of pickup.

    Parameters:
    - N: Number of particles.

    Returns:
    - Pickup probability.
    """
    # see paper for formula
    prob = 1 - np.e**(-eta_p(N))
    return prob

# drop prob function
def prob_drop(N, t_now, t_latest, decay_rate, h):
    """
    Calculates the probability of dropping.

    Parameters:
    - N: Number of particles.
    - t_now: Current time step.
    - t_latest: Latest time step.
    - decay_rate: Rate of decay.
    - h: Height.

    Returns:
    - Drop probability.
    """
    # time delta
    tau = t_now-t_latest
    # see paper for formula
    prob = 1 - np.e**(-eta_d(N)*np.e**(-tau*decay_rate))
    if h>0:
        # add vertical modulation for height h>0 in mm
        prob = prob*mod_list[h]
    # return
    return prob

# ------------ Algorithms ----------------------
# ----------------------------------------------

# move algorithm
def move_algorithm(pos, world, m):
    """
    Executes the move algorithm.

    Parameters:
    - pos: Current position.
    - world: The World object.
    - m: Number of moves.

    Returns:
    - Final position after moving.
    """
    for j in range(m):
        x,y,z = pos
        local_data = local_grid_data(pos, world)
        moves = valid_moves(local_data)
        if len(moves)>0:
            chosen_move = random_choices(moves)[0]
            new_pos = np.array(pos)+chosen_move[1]
            # do the step
            world.grid[x,y,z] = 0
            world.grid[new_pos[0],new_pos[1],new_pos[2]] = -2
            pos = new_pos
    # return final position
    return (pos[0], pos[1], pos[2])

# pickup algorithm
def pickup_algorithm(pos, world, x_rand):
    """
    Executes the pickup algorithm.

    Parameters:
    - pos: Current position.
    - world: The World object.
    - x_rand: Random variable.

    Returns:
    - Material picked up or None.
    """
    x,y,z = pos
    if world.grid[x,y,z-1] > 0:
        v26 = local_grid_data(pos, world)
        N = np.sum(v26==2)
        prob = prob_pickup(N)
        if x_rand < prob:
            # do the pickup
            material = world.grid[x,y,z-1]
            world.grid[x,y,z-1]=0
            return material
    # if no pickup occured
    return None

# drop algorithm
def drop_algorithm(pos, world, step, decay_rate, x_rand):
    """
    Executes the drop algorithm.

    Parameters:
    - pos: Current position.
    - world: The World object.
    - step: Current step.
    - decay_rate: Rate of decay.
    - x_rand: Random variable.

    Returns:
    - New position after dropping or None.
    """
    x,y,z = pos
    v26 = local_grid_data(pos, world)
    moves = valid_moves(v26)
    # only act if there is an available move
    if len(moves)>0:
        N = np.sum(v26==2)
        # slice lower bounds
        x_low_bound, y_low_bound, z_low_bound  = max(0, x-1), max(0, y-1), max(0, z-1)
        t_latest = np.max(world.times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
        t_now = step
        h = compute_height(pos, world)
        prob = prob_drop(N, t_now, t_latest, decay_rate, h)
        if x_rand < prob:
            # chosen move
            chosen_move = random_choices(moves)[0]
            new_pos = np.array(pos)+chosen_move[1]
            # do the drop
            world.grid[x,y,z] = 2
            # update time tensor at pos
            world.times[x, y, z] = t_now
            # return new position
            return (new_pos[0], new_pos[1], new_pos[2])
    # if no drop occureed return None
    return None

# drop algorithm graph version
def drop_algorithm_graph(pos, world, graph, step, decay_rate, x_rand):
    """
    Executes the drop algorithm with a graph.

    Parameters:
    - pos: Current position.
    - world: The World object.
    - graph: Graph representing neighbors.
    - step: Current step.
    - decay_rate: Rate of decay.
    - x_rand: Random variable.

    Returns:
    - New position (neighbor) after dropping or None.
    """
    # neighbours of pos in iteration graph
    nbrs = graph[pos]
    # only act if there is an available place to move
    if len(nbrs)>0:
        # do the khuong stuff
        x,y,z = pos
        v26 = local_grid_data(pos, world)
        N = np.sum(v26==2)
        # slice lower bounds
        x_low_bound, y_low_bound, z_low_bound  = max(0, x-1), max(0, y-1), max(0, z-1)
        t_latest = np.max(world.times[x_low_bound:x+2,y_low_bound:y+2,z_low_bound:z+2])
        t_now = step
        h = compute_height(pos, world)
        prob = prob_drop(N, t_now, t_latest, decay_rate, h)
        # random probability
        if x_rand < prob:
            # chosen move
            chosen_nbr = random_choices(nbrs)[0]
            # do the drop
            world.grid[x,y,z] = 2
            # update time tensor at pos
            world.times[x, y, z] = t_now
            # return new position
            return chosen_nbr
    # if no drop occureed return None
    return None 