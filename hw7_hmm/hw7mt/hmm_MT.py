'''
    Class: INF552 at USC
    HW7: HMM
    Minh Tran
    A python implementation of HMM from scratch
'''

"""
To run code
just hit run
"""
import math
import collections
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
"""
for each free pos, calculate distances to each tower [[[0.7d1, 1.3d1],[0.7d2, 1.3d2],[0.7d3, 1.3d3],[0.7d4, 1.3d4]],...]
"""
def distance_to_tower(free_position, tower_locations, lower_margin, upper_margin):
    free_pos_to_tower_dist = []
    for i, pos in enumerate(free_position):
        dist = []
        for j, tower in enumerate(tower_locations):
            # calculate euclidean distance between a pos and a tower
            temp_dist = math.sqrt(pow(pos[0] - tower[0], 2) + pow(pos[1] - tower[1], 2))

            # add noisy factor.
            dist.append([temp_dist * lower_margin, temp_dist * upper_margin])
        free_pos_to_tower_dist.append(dist)
    return free_pos_to_tower_dist

"""
find probable states of each time step: dict: {timestep0: [[i1,j1], ],...
by checking for each free pos, if the recorded distance of that pos to each tower is falling within 
the range of (0.7*d to 1.3*d)
"""
def find_probable_pos(free_position, recorded_distance, free_pos_to_tower_dist):
    probable_pos = []
    for i in range(0, len(free_position)):
        pos = free_position[i]
        state = True
        for j in range(0, len(recorded_distance)):
            if recorded_distance[j] > free_pos_to_tower_dist[i][j][1] or recorded_distance[j] < free_pos_to_tower_dist[i][j][0]:
                state = False

        if state == True:
            probable_pos.append(pos)
    return probable_pos

"""
get all neighbors of a point
"""
def find_neighbours(location, grid_size):
    x = location[0]
    y = location[1]
    neighbours = []

    # prevent cases where points are on the edge of grid
    if x + 1 < grid_size:
        neighbours.append((x + 1, y))
    if y + 1 < grid_size:
        neighbours.append((x, y + 1))
    if x - 1 > 0:
        neighbours.append((x - 1, y))
    if y - 1 > 0:
        neighbours.append((x, y - 1))
    return neighbours

"""
evaluate transitional probability from 1 pt to another:
{(4, 4): {(5, 4): 0.5, (4, 3): 0.5},...}
"""
def calc_trans_prob(free_pos_probable_timestep_dict, neighbours):
    # initialize
    trans_prob_neighbors = collections.defaultdict(dict) # prob counter
    total_trans_prob = collections.defaultdict(int) # total counter
    trans_prob = collections.defaultdict(dict) # required output

    # loop through each position in the dict
    for pos in free_pos_probable_timestep_dict:
        total_trans_prob[pos] = 0.0
        probable_timesteps = free_pos_probable_timestep_dict[pos] # get possible timestep for this pos
        neighboring_pos = neighbours[pos]  # get all neighbors for this pos

        for timestep in probable_timesteps:
            timestep += 1
            
            # check all neighboring poss
            for nei in neighboring_pos:
                # check if that neighbor can belong to a possible time step
                if nei in free_pos_probable_timestep_dict:
                    # check if the next time step is in that neighbor's possible timesteps
                    if timestep in free_pos_probable_timestep_dict[nei]:
                        # add the new neighbors to the trans prob dict if all checks are true
                        if nei not in trans_prob_neighbors[pos]:
                            trans_prob_neighbors[pos][nei] = 0.0

                        # update the value of prob
                        trans_prob_neighbors[pos][nei] += 1.0
                        total_trans_prob[pos] += 1.0

        # normalize to get the actual probability
        for nei in trans_prob_neighbors[pos]:
            trans_prob[pos][nei] = trans_prob_neighbors[pos][nei] / total_trans_prob[pos]
    return trans_prob

"""
do_viterbi_algo algorithm:
for each timestep: several possibilities of poss that robot can be in
--> find path that has highest probability from timestep 1 to 11 and the path itself
starting point is uniformly determined
"""
def do_viterbi_algo(numTimestep, timestep_probable_pos_dict, trans_prob):
    # initialize
    timestep = 0
    possible_pathways = collections.defaultdict(dict)
    possible_pathways[timestep] = collections.defaultdict(dict)

    # starting point
    for pos in timestep_probable_pos_dict[timestep]:
        pos = tuple(pos)
        possible_pathways[timestep][pos] = {}  # initialize
        possible_pathways[timestep][pos]['parent'] = '' # no parent
        possible_pathways[timestep][pos]['prob'] = 1.0 / len(timestep_probable_pos_dict[timestep]) # uniform probability of starting point

    # for each subsequent timestep
    for timestep in range(1, numTimestep):
        possible_pathways[timestep] = collections.defaultdict(dict)
        # looping through the previous timestep poss
        for pos in possible_pathways[timestep - 1]:
            if pos in trans_prob:  # check if there is any subsequent states corresponding to this state
                # looping through each neighbors in the transitional probability
                for nei in trans_prob[pos]:
                    # check if this neighbor is actually in correct timestep
                    if list(nei) in timestep_probable_pos_dict[timestep]:

                        # calculate probability to reach this pos = parent_prob * trans_prob (bayes rule)
                        current_prob = possible_pathways[timestep - 1][pos]['prob'] * trans_prob[pos][nei]

                        # if that node is new
                        if nei not in possible_pathways[timestep]:
                            possible_pathways[timestep][nei] = {} # initialize
                            possible_pathways[timestep][nei]['parent'] = pos # update parents for find_final_path

                            # update the probability to reach this node
                            possible_pathways[timestep][nei]['prob'] = current_prob

                        # if that node has been reached before, only keep the highest probability path
                        else:
                            if current_prob > possible_pathways[timestep][nei]['prob']:
                                # update new parents and probability
                                possible_pathways[timestep][nei]['parent'] = pos
                                possible_pathways[timestep][nei]['prob'] = current_prob
    return possible_pathways

"""
function to retrieve the hmm chain with highest probability
"""
def find_final_path(possible_pathways, numTimestep):
    max_prob = 0.0
    pos = None
    final_path = []

    # find end points with maximum probability
    for p in possible_pathways[numTimestep-1]:
        if max_prob < possible_pathways[numTimestep-1][p]['prob']:
            max_prob = possible_pathways[numTimestep-1][p]['prob']
            pos = p
    final_path.append(pos)

    # find_final_path
    for timestep in range(numTimestep-1, 0, -1):
        parent = possible_pathways[timestep][pos]['parent']
        final_path.append(parent)
        pos = parent
    return final_path


"""
get free pos location from input input_file
list of lists: [[i,j],...]

get tower location from input input_file:
list of lists: [[i,j],...]

get distance recorded at each time step: 
list of lists: [[d1,d2,d3,d4],...]
"""
def find_data(input_file, numInput, keyword):
    row = 0
    output = []
    keyword_located = False
    with open(input_file) as f:
        for line in f:
            # get key word
            if keyword in line:
                keyword_located = True
                continue

            line = line.strip()  # remove leading and trailing characters

            # skip blank line in input
            if (line == ''):
                continue

            if keyword_located:
                if keyword == "Grid-World":
                    line = line.split()  # split on space
                    for col, indicator in enumerate(line):
                        if indicator == '1':  # free pos indicator
                            output.append([int(row), int(col)])
                    row += 1
                elif keyword == "Tower Locations":
                    # split on : first, get second element then split second element based on space
                    loc = line.split(':')[1].split()
                    output.append([int(loc[0]), int(loc[1])])
                    row += 1
                elif keyword == "Noisy":
                    line = line.split()
                    dist = []
                    for pos in line:
                        dist.append(float(pos))
                    output.append(dist)
                    row += 1

                # stop after reaching last row
                if row == numInput:
                    break
    return output

def main():
    # input input_file name
    input_file = 'hmm-data.txt'
    numRow = 10
    numTower = 4
    numTimeStep = 11
    lower_margin = 0.7
    upper_margin = 1.3

    # get free pos locations: list of lists: [[i,j],...]
    free_positions = find_data(input_file, numRow, 'Grid-World')
    print("free poss: ",  free_positions)

    # get tower locations: list of lists: [[i,j],...]
    tower_locations = find_data(input_file, numTower,'Tower Locations')
    print("tower locations: ", tower_locations)

    # get distance recorded at each time step: list of lists: [[d1,d2,d3,d4],...]
    recorded_distance = find_data(input_file, numTimeStep, 'Noisy')
    # print("distance to each tower recorded in each timestep: ", recorded_distance)

    # for each free pos, calculate distances to each tower [[[0.7d1, 1.3d1],[0.7d2, 1.3d2],[0.7d3, 1.3d3],[0.7d4, 1.3d4]],...]
    free_pos_to_tower_dist = distance_to_tower(free_positions, tower_locations,lower_margin, upper_margin)
    # print("distance to tower: ", free_pos_to_tower_dist)

    # find probable states of each time step: dict: {timestep0: [[i1,j1], ],...
    timestep_probable_pos_dict = collections.defaultdict(list)

    # find probable timesteps each pos might belong: dict: {[i1,j1]: [timestep1, ...],...
    free_pos_probable_timestep_dict = collections.defaultdict(list)
    for i in range(0, len(recorded_distance)):
        timestep_probable_pos_dict[i] = find_probable_pos(free_positions, recorded_distance[i], free_pos_to_tower_dist)
        for pos in timestep_probable_pos_dict[i]:
            free_pos_probable_timestep_dict[tuple(pos)].append(i)
    # print("probable_state_dic: ", timestep_probable_pos_dict)
    # print("free_pos_probable_timestep_dict: ", free_pos_probable_timestep_dict)

    # find all neighbors of poss that might belong to one timestep
    # to construct the transitional probabilities matrix {(i,j): [(),(),(),()],...}
    neighbours = collections.defaultdict(list)
    for pos in free_pos_probable_timestep_dict:
        neighbours[pos] = find_neighbours(pos, 10)
    # print("neighbours: ", neighbours)

    # evaluate transitional probability
    trans_prob = calc_trans_prob(free_pos_probable_timestep_dict, neighbours)
    # print("transitional probabilities: ", trans_prob)

    possible_pathways = do_viterbi_algo(numTimeStep, timestep_probable_pos_dict, trans_prob)
    final_path = find_final_path(possible_pathways, numTimeStep)
    print("Final Path is:")
    print(final_path[::-1])

    cmap = colors.ListedColormap(['red', 'blue','black','yellow'])
    plt.figure(figsize=(6, 6))
    data = np.zeros((10,10))
    # plot obstacles
    for pos in free_positions:
        data[pos[0],pos[1]] = 1

    # plot towers:
    for pos in tower_locations:
        data[pos[0],pos[1]] = 2

    # plot final path
    for pos in final_path:
        data[pos[0],pos[1]] = 3
    plt.pcolor(data[::-1], cmap=cmap, edgecolors='k', linewidths=3)
    plt.show(block=False)
    plt.pause(1)
    plt.savefig('grid.png')
    plt.close()

if __name__ == "__main__":
    main()