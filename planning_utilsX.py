from enum import Enum
from queue import PriorityQueue
import numpy as np


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)

from math import sqrt
SQRT2 = sqrt(2)
SQRT5 = sqrt(5)
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    NORTHEAST = (-1,  1, SQRT2)
    NORTHWEST = (-1, -1, SQRT2)
    SOUTHEAST = ( 1,  1, SQRT2)
    SOUTHWEST = ( 1, -1, SQRT2)
    WEST =  ( 0, -1, 1)
    EAST =  ( 0,  1, 1)
    NORTH = (-1,  0, 1)
    SOUTH = ( 1,  0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

class ActionK(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    NNE = (-2, 1, SQRT5)
    ENE = (-1, 2, SQRT5)
    ESE = (1, 2, SQRT5)
    SSE = (2, 1, SQRT5)
    SSW = (2, -1, SQRT5)
    WSW = (1, -2, SQRT5)
    WNW = (-1, -2, SQRT5)
    NNW = (-2, -1, SQRT5)
    NORTHEAST = (-1,  1, SQRT2)
    NORTHWEST = (-1, -1, SQRT2)
    SOUTHEAST = ( 1,  1, SQRT2)
    SOUTHWEST = ( 1, -1, SQRT2)
    WEST =  ( 0, -1, 1)
    EAST =  ( 0,  1, 1)
    NORTH = (-1,  0, 1)
    SOUTH = ( 1,  0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

def valid_actions(grid, current_node, km):
    """
    Returns a list of valid actions given a grid and current node.
    """
    if km:
        valid_actions = list(ActionK)
    else:
        valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if km:
        if x - 1 < 0 or grid[x - 1, y] == 1:
            valid_actions.remove(ActionK.NORTH)
            valid_actions.remove(ActionK.NORTHEAST) # cannot clip the obstacle
            valid_actions.remove(ActionK.NORTHWEST)
        if x + 1 > n or grid[x + 1, y] == 1:
            valid_actions.remove(ActionK.SOUTH)
            valid_actions.remove(ActionK.SOUTHEAST)
            valid_actions.remove(ActionK.SOUTHWEST)
        if y - 1 < 0 or grid[x, y - 1] == 1:
            valid_actions.remove(ActionK.WEST)
            if ActionK.NORTHWEST in valid_actions: valid_actions.remove(ActionK.NORTHWEST)
            if ActionK.SOUTHWEST in valid_actions: valid_actions.remove(ActionK.SOUTHWEST)
        if y + 1 > m or grid[x, y + 1] == 1:
            valid_actions.remove(ActionK.EAST)
            if ActionK.NORTHEAST in valid_actions: valid_actions.remove(ActionK.NORTHEAST)
            if ActionK.SOUTHEAST in valid_actions: valid_actions.remove(ActionK.SOUTHEAST)

        # reaching this point, the primary axes are clear
        if ActionK.NORTHEAST in valid_actions and (x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1):
            valid_actions.remove(ActionK.NORTHEAST)
        if ActionK.NORTHWEST in valid_actions and (x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1):
            valid_actions.remove(ActionK.NORTHWEST)
        if ActionK.SOUTHEAST in valid_actions and (x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1):
            valid_actions.remove(ActionK.SOUTHEAST)
        if ActionK.SOUTHWEST in valid_actions and (x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1):
            valid_actions.remove(ActionK.SOUTHWEST)
        if x - 1 < 0 or x - 2 < 0 or y + 1 > m or grid[x-2,y+1] == 1 or grid[x-1,y] == 1 or grid[x-1,y+1] == 1:
            valid_actions.remove(ActionK.NNE)
        if x - 1 < 0 or y + 1 > m or y + 2 > m or grid[x-1,y+2] == 1 or grid[x-1,y+1] == 1 or grid[x,y+1] == 1:
            valid_actions.remove(ActionK.ENE)
        if x - 1 < 0 or y - 1 < 0 or y - 2 < 0 or grid[x-1,y-2] == 1 or grid[x-1,y-1] == 1 or grid[x,y-1] == 1:
            valid_actions.remove(ActionK.WNW)
        if x - 1 < 0 or x - 2 < 0 or y - 1 < 0 or grid[x-2,y-1] == 1 or grid[x-1,y-1] == 1 or grid[x-1,y] == 1:
            valid_actions.remove(ActionK.NNW)
        if x + 1 > n or y + 1 > m or y + 2 > m or grid[x+1,y+2] == 1 or grid[x+1,y+1] == 1 or grid[x,y+1] == 1:
            valid_actions.remove(ActionK.ESE)
        if x + 1 > n or x + 2 > n or y + 1 > m or grid[x+2,y+1] == 1 or grid[x+1,y+1] == 1 or grid[x+1,y] == 1:
            valid_actions.remove(ActionK.SSE)
        if x + 1 > n or x + 2 > n or y - 1 < 0 or grid[x+2,y-1] == 1 or grid[x+1,y-1] == 1 or grid[x+1,y] == 1:
            valid_actions.remove(ActionK.SSW)
        if x + 1 > n or y - 1 < 0 or y - 2 < 0 or grid[x+1,y-2] == 1 or grid[x+1,y-1] == 1 or grid[x,y-1] == 1:
            valid_actions.remove(ActionK.WSW)
    else:
        if x - 1 < 0 or grid[x - 1, y] == 1:
            valid_actions.remove(Action.NORTH)
            valid_actions.remove(Action.NORTHEAST) # cannot clip the obstacle
            valid_actions.remove(Action.NORTHWEST)
        if x + 1 > n or grid[x + 1, y] == 1:
            valid_actions.remove(Action.SOUTH)
            valid_actions.remove(Action.SOUTHEAST)
            valid_actions.remove(Action.SOUTHWEST)
        if y - 1 < 0 or grid[x, y - 1] == 1:
            valid_actions.remove(Action.WEST)
            if Action.NORTHWEST in valid_actions: valid_actions.remove(Action.NORTHWEST)
            if Action.SOUTHWEST in valid_actions: valid_actions.remove(Action.SOUTHWEST)
        if y + 1 > m or grid[x, y + 1] == 1:
            valid_actions.remove(Action.EAST)
            if Action.NORTHEAST in valid_actions: valid_actions.remove(Action.NORTHEAST)
            if Action.SOUTHEAST in valid_actions: valid_actions.remove(Action.SOUTHEAST)

        # reaching this point, the primary axes are clear
        if Action.NORTHEAST in valid_actions and (x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1):
            valid_actions.remove(Action.NORTHEAST)
        if Action.NORTHWEST in valid_actions and (x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1):
            valid_actions.remove(Action.NORTHWEST)
        if Action.SOUTHEAST in valid_actions and (x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1):
            valid_actions.remove(Action.SOUTHEAST)
        if Action.SOUTHWEST in valid_actions and (x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1):
            valid_actions.remove(Action.SOUTHWEST)

    return valid_actions

def a_star(grid, h, start, goal, km):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    cycles = 0

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        cycles += 1

        if current_node == goal:
            print('Found a path on the grid. Cycles:', cycles)
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node, km):
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])
                new_cost = current_cost + a.cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def a_star2(grid, h, start, goal, km):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    cycles = 0
    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        cycles += 1
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path on the grid. Cycles:', cycles)
            found = True
            break
        else:
            for action in valid_actions(grid, current_node, km):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost
