# imports
import numpy as np
from mayavi import mlab

# Lists and Dictionnaries
translation_dict = {-2:'agent', -1:'object', 0:'void', 1:'soil',2:'pellet'}
directions_3d = np.array([
    [1, 0, 0],   # Right
    [-1, 0, 0],  # Left
    [0, 1, 0],   # Forward
    [0, -1, 0],  # Backward
    [0, 0, 1],   # Up
    [0, 0, -1],  # Down
    [1, 1, 0],   # Right-Forward
    [1, -1, 0],  # Right-Backward
    [-1, 1, 0],  # Left-Forward
    [-1, -1, 0], # Left-Backward
    [1, 0, 1],   # Right-Up
    [1, 0, -1],  # Right-Down
    [-1, 0, 1],  # Left-Up
    [-1, 0, -1], # Left-Down
    [0, 1, 1],   # Forward-Up
    [0, 1, -1],  # Forward-Down
    [0, -1, 1],  # Backward-Up
    [0, -1, -1], # Backward-Down
    [1, 1, 1],   # Right-Forward-Up
    [1, 1, -1],  # Right-Forward-Down
    [1, -1, 1],  # Right-Backward-Up
    [1, -1, -1], # Right-Backward-Down
    [-1, 1, 1],  # Left-Forward-Up
    [-1, 1, -1],  # Left-Forward-Down
    [-1, -1, 1],  # Left-Backward-Up
    [-1, -1, -1] # Left-Backward-Down
])
neighbour_directions = np.array([
    [0, 0, 1], # Up
    [0, 0, -1], # Down
    [0, 1, 0], # Forward
    [0, -1, 0], # Backwards
    [1, 0, 0], # Right
    [-1, 0, 0] # Left
]) 
floor_directions = np.array([
    [1, 0, -1],  # Right-Down
    [-1, 0, -1], # Left-Down
    [0, 1, -1],  # Forward-Down
    [0, -1, -1], # Backward-Down
    [1, 1, -1],  # Right-Forward-Down
    [1, -1, -1], # Right-Backward-Down
    [-1, 1, -1],  # Left-Forward-Down
    [-1, -1, -1] # Left-Backward-Down
]) 
neighbour_filter = np.array([
       [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]],
       [[0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]],
       [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]])
center_loc = np.array([1,1,1])

# returns the valid move directions from local data
def valid_move_directions(local_data):
    dirs = []
    # loop over 6 neighbours
    for dir in neighbour_directions:
        new_pos = center_loc + dir
        if local_data[new_pos[0], new_pos[1], new_pos[2]]==0:
            # slice lower bounds
            x_low_bound = max(0, new_pos[0]-1)
            y_low_bound = max(0, new_pos[1]-1)
            z_low_bound = max(0, new_pos[2]-1)
            # sliced array
            new_local_data = local_data[x_low_bound:new_pos[0]+2, y_low_bound:new_pos[1]+2, z_low_bound:new_pos[2]+2]
            # check for any material
            if (new_local_data>0).any():
                dirs.append(dir)
    return dirs

# checks neighbours for some material
def voxel_shares_face_with_material(local_data):
    filtered = local_data*neighbour_filter
    shares_face = (filtered > 0).any()
    # return binary condition
    return shares_face

# random choice from list
def random_choice(ls, p=None):
    # No prob dist given
    if p is None:
        k = len(ls)
        ind = np.random.randint(0,k)
        return ls[ind]

    # Check if the probabilities sum to 1.0
    if not np.isclose(sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1.0")

    # Use np.random.choice to make a random selection based on probabilities
    chosen_index = np.random.choice(range(len(ls)), p=p)
    choice = ls[chosen_index]
    return choice

# get local grid data for position in world
def local_grid_data(pos, world):
        x,y,z = pos
        if 0<x<world.width-2 and 0<y<world.length-2 and 0<z<world.height-2:
            # Not on any boundary -- No padding
            local_data = world.grid[x-1:x+2,y-1:y+2,z-1:z+2]
            return local_data
        else:
            # On some boundary -- Need padding
            x_min, x_max = max(0, x - 1), min(world.width, x + 2)
            y_min, y_max = max(0, y - 1), min(world.length, y + 2)
            z_min, z_max = max(0, z - 1), min(world.height, z + 2)

            x_start, x_end = 1 if x_min == 0 else 0, 2 if x_max == world.width else 3
            y_start, y_end = 1 if y_min == 0 else 0, 2 if y_max == world.length else 3
            z_start, z_end = 1 if z_min == 0 else 0, 2 if z_max == world.height else 3

            local_data = np.full((3, 3, 3), -1)
            local_data[x_start:x_end, y_start:y_end, z_start:z_end] = world.grid[
                x_min:x_min + x_end - x_start, y_min:y_min + y_end - y_start, z_min:z_min + z_end - z_start]
            return local_data
        
# list of valid actions from local data tensor
def valid_actions(local_data, has_pellet):
        action_list = []
        move_directions = valid_move_directions(local_data)
        # move
        if len(move_directions)>0:
            for dir in move_directions:
                action_list.append(('move',dir)) # movements
            # drop
            if has_pellet is True:
                if voxel_shares_face_with_material(local_data):
                    ind = np.random.randint(0,len(move_directions))
                    dir = move_directions[ind]
                    action_list.append(('drop',dir)) # drop with random movement
        # pickup
        if has_pellet is False:
            under = np.array([0,0,-1])
            if local_data[1,1,0] in [1,2]:
                action_list.append(('pickup',under)) # pickup

        return action_list

# list of valid moves from local data tensor
def valid_moves(local_data):
        action_list = []
        move_directions = valid_move_directions(local_data)
        # moves in format
        if len(move_directions)>0:
            for dir in move_directions:
                action_list.append(('move',dir)) # movements
        return action_list

# a height function that returns the number of empty cells below
def compute_height(pos, world):
    x,y,z = pos
    max_h = world.height
    h=0
    for _ in range(max_h):
        z-=1
        cell = world.grid[x,y,z]
        if cell > 0:
            return h
        h+=1
    return max_h

# basic pickup prob function
def prob_pickup(N, spontpick, amplifpick):
    if N == 0:
        prob = spontpick
    else:
        prob = spontpick/amplifpick*N
    return prob

# fairly basic drop prob function
def prob_drop(N, t, t_latest, spontdrop, drop1, amplifdrop, h, evap):
    if N == 0:
        prob = spontdrop
    else:
        prob = (drop1+amplifdrop*(N-1))*np.exp(-evap*(t-t_latest))
    if h>0:
        prob *= (h**N)/(8**N + h**N) # multiply by cumulative density F(h) of height
    return prob
        
# update surface
def update_surface(surface, type, pos, world):
    new_surface = surface
    if type == 'pickup':
        x,y,z = pos
        new_surface.append((x,y,z-1))
    elif type == 'drop':
        x,y,z = pos
        new_surface.remove((x,y,z))
        for dir in neighbour_directions:
            new_pos = pos + dir
            x_new, y_new, z_new = new_pos
            if 0<=x_new<world.width and 0<=y_new<world.length and 0<=z_new<world.height:
                if world.grid[x_new,y_new,z_new]==0:
                    if (x_new, y_new, z_new) not in new_surface:
                        new_surface.append((x_new, y_new, z_new))
    else:
        raise ValueError
    return new_surface

# render world and agents with mayavi
def render(world, show=True, save=False, name="image_1.png"):
    # figure
    fig = mlab.figure(size=(1024, 1024))

    # Define colors based on your values
    colors = {
        1: (1, 1, 1),   # Soil
        -1: (1, 0, 0),  # Objects
        2: (0.1, 0.1, 0.1),  # Pellets
        -2: (0, 0, 1)   # Agents
        }

    # plotting
    for value, color in colors.items():
        x_vals, y_vals, z_vals = np.where(world.grid == value)
        mlab.points3d(x_vals, y_vals, z_vals,
                        mode="cube",
                        color=color,
                        scale_factor=1)

    # save image
    if save:
        mlab.savefig(name)
        mlab.close()
    
    # show image
    if show:
        mlab.show()