import numpy as np
from sklearn.cluster import KMeans
from .test_parameter import *

def find_frontier(downsampled_belief, resolution=3): ###Review this function###
    y_len = downsampled_belief.shape[0]
    x_len = downsampled_belief.shape[1]
    mapping = downsampled_belief.copy()
    belief = downsampled_belief.copy()
    # 0-1 unknown area map
    mapping = (mapping == 127) * 1
    mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)    #Check 8 nearby unknown area checks, fro map value of each cells is the num of nearby unknown area
    fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
              2:] + \
                mapping[:y_len][:, :x_len]
    ind_free = np.where(belief.ravel(order='F') == 255)[0] # ‘F’ means to index the elements in column-major, Fortran-style order, with the first index changing fastest, and the last index changing slowest.
    ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0] #ind_free is the middle pixel
    ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
    ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
    ind_to = np.intersect1d(ind_free, ind_fron)

    map_x = x_len
    map_y = y_len
    x = np.linspace(0, map_x - 1, map_x)
    y = np.linspace(0, map_y - 1, map_y)
    t1, t2 = np.meshgrid(x, y)
    points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

    f = points[ind_to] 
    f = f.astype(int) #Matrices of indices of the frontier pixels

    f = f * resolution #Matrices of coordinates of the frontier pixels multiply by the resoultion value to upscale
    return f

def evaluate_exploration_rate(robot_belief, ground_truth):                 #Caculate percentage of explored area using free space
    rate = np.sum(robot_belief == 255) / np.sum(ground_truth == 255)
    return rate

def calculate_reward(ground_truth_size, dist, frontiers, old_frontiers):
    reward = 0
    dist = dist / np.max(np.array(ground_truth_size))
    reward -= dist * 10
    
    frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
    pre_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
    frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
    pre_frontiers_num = pre_frontiers_to_check.shape[0]
    delta_num = pre_frontiers_num - frontiers_num
    reward += delta_num / 50

    return reward

def check_done(node_utility):
    done = False
    if np.sum(node_utility ) <= 5:
    #if np.sum(ground_truth == 255) - np.sum(robot_belief == 255) == 0:
        done = True
    return done

#Function not used in env.py
def calculate_new_free_area(self):
    old_free_area = self.old_robot_belief == 255
    current_free_area = self.robot_belief == 255

    new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int)) * 255

    return new_free_area, np.sum(old_free_area)


def calculate_path_to_high_utility_node(self):
    highest_utility_node_index = np.argmax(self.node_utility)
    dist, path = self.prm.find_shortest_path(self.robot_position,
                                                                self.node_coords[highest_utility_node_index],
                                                                self.node_coords)
    return dist, path

def calculate_utility_along_path(self, path):
    observable_frontiers = []
    for index in path:
        observable_frontiers += self.nodes_list[index].observable_frontiers
    np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
    unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j)

    return unique_frontiers.shape[0]

def calculate_utility(self, node, frontiers):
    dist_list = np.linalg.norm(frontiers - node, axis=-1)
    frontiers_in_range = frontiers[dist_list < self.sensor_range - 5]
    num = 0
    if len(frontiers_in_range) > 0:
        for frontier in frontiers_in_range:
            collision = self.prm.check_collision(node, frontier, self.robot_belief)
            if not collision:
                num += 1

    return num

def calculate_node_gain_over_path(self, node_index, path): #Calculate the gain of the node over the path
    observable_frontiers = [] ####
    for index in path:
        observable_frontiers += self.nodes_list[index].observable_frontiers
    np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
    pre_unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j)
    observable_frontiers += self.nodes_list[node_index].observable_frontiers
    np_observable_frontiers = np.array(observable_frontiers).reshape(-1,2)
    unique_frontiers = np.unique(np_observable_frontiers[:, 0] + np_observable_frontiers[:, 1]*1j) 

    return unique_frontiers.shape[0] - pre_unique_frontiers.shape[0]

def calculate_num_observed_frontiers(old_frontiers, frontiers):
    frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
    pre_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
    frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
    pre_frontiers_num = pre_frontiers_to_check.shape[0]
    delta_num = pre_frontiers_num - frontiers_num

    return delta_num

def calculate_dist_path(self, path):
    dist = 0
    start = path[0]
    end = path[-1]
    for point in path:
        if point == end:
            break
        dist += np.linalg.norm(self.node_coords[start] - self.node_coords[point])
        start = point
    return dist

def get_frontier_centers(self, return_closest_candidate=True):
    if len(self.frontiers) == 0:
        return None
    num_centers = len(self.frontiers) // 30 + 1   ###wrong

    frontiers = np.array(self.frontiers)
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(frontiers)
    centers = kmeans.cluster_centers_
    if return_closest_candidate:
        closest_candidates = []
        for center in centers:
            dist = np.linalg.norm(center - self.node_coords, axis=-1)
            closest_candidate_index = np.argmin(dist)
            closest_candidates.append(self.node_coords[closest_candidate_index])
        closest_candidates = np.array(closest_candidates)
        return closest_candidates
    else:
        return centers   ###
    
def get_dist_to_nearest_frontier(self):
    if self.frontier_centers is not None:
        min_dist = 1e8
        for center in self.frontier_centers:
            dist, _ = self.prm.find_shortest_path(self.robot_position, center, self.node_coords)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    else:
        return 0   

def parameter_summary():    
    print("------------------Parameters Summary---------------------\n")
    print("                  Environment parameters\n")
    print("MAXIMUM AGENTS:", MAXIMUM_AGENTS)
    print("NUMBER OF HEADINGS:", NUM_HEADING)
    print("SPEED:", SPEED)
    print("SENSOR RANGE:", SENSOR_RANGE)
    print("SCALE:", SCALE)
    print("COVERAGE:", COVERAGE)
    
    print("                  Model parameters\n")
    print("REPLAY SIZE:", REPLAY_SIZE)
    print("MINIMUM BUFFER SIZE:", MINIMUM_BUFFER_SIZE)
    print("BATCH SIZE:", BATCH_SIZE)
    print("INPUT DIMENSION:", INPUT_DIM)
    print("LEARNING RATE:", LR)
    print("GAMMA:", GAMMA)
    print("DECAY STEP:", DECAY_STEP)
    print("MAXIMUM TIMESTEPS:", MAXIMUM_TIMESTEPS)

    print("\n                   Node parameters\n")
    print("ADAPTIVE GENERATION:", ADAPTIVE_GENERATION)
    print("EMBEDDING DIMENSION:", EMBEDDING_DIM)
    print("NODE PADDING SIZE:", NODE_PADDING_SIZE)
    print("K SIZE:", K_SIZE)
    print("RESOLUTION:", RESOLUTION)

    print("\n                   GPU parameters\n")
    print("USE GPU (DATA):", USE_GPU_DATA)
    print("USE GPU (TRAINING):", USE_GPU_TRAINING)
    print("NUMBER OF GPU:", NUM_GPU)
    print("NUMBER OF META AGENT:", NUM_META_AGENT)

    print("\n                   Data handling\n")
    print("TENSORBOARD UPDATE INTERVAL:", SUMMARY_WINDOW)
    print("FOLDER NAME:", FOLDER_NAME)
    print("MODEL PATH:", model_path)
    print("TRAIN PATH:", train_path)
    print("GIFS PATH:", gifs_path)
    print("LOAD MODEL:", LOAD_MODEL)
    print("SAVE MODEL INTERVAL:", SAVE_MODEL_INTERVAL)

    print("\n                   Image creation\n")
    print("SAVE IMAGE GAP:", SAVE_IMG_GAP)
    print("PLOT EDGE:", PLOT_EDGE)
    
    print("\n                   Episode summary\n")
    print("SUMMARY:", SUMMARY)
    print("SUMMARY INTERVAL:", SUMMARY_INTERVAL)
    print("TIMESTEP INTERVAL:", TIMESTEP_INTERVAL)
    print("-------------------------------------------------------\n")
    
    