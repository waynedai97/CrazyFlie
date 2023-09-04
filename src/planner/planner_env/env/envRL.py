from os import listdir
import numpy as np
from skimage import io
import copy
from skimage.measure import block_reduce
from matplotlib import pyplot as plt

from .MapRL import *
from .sensor import *
from .node import *
from .utility import *
from time import time

class EnvRL():
  def __init__(self, k_size=20, train=True, num_heading=4, plot=False, adaptive_generation=False, sensor_range=80, speed=1, resolution=4):
    
    # Prepare img path
    self.train = train
    if self.train:
      self.img_path = 'DungeonMaps/train'
    else:
      self.img_path = 'DungeonMaps/test'
    
    # Load environments
    #self.env_id = 'img_10906.png'
    #self.add = 0
    self.env_id = 'SAFMC_ACTUAL2.png'
    self.add = 1
    
    # Prepare environments
    self.ground_truth, self.start_location = self.prepareEnv(self.env_id)
    self.ground_truth_size = self.ground_truth.shape #size: (480, 640), might be 640 by 640
    self.start_location_heading = copy.deepcopy(self.start_location)
    self.start_location_heading = np.append(self.start_location_heading, 0)
    
    # Intialise robot belief
    self.robot_belief = np.ones(self.ground_truth_size) * 127
    self.downsampled_belief = None
    self.old_robot_belief = copy.deepcopy(self.robot_belief)
    
    # Initialise parameters
    self.sensor_range = sensor_range
    self.resolution = resolution #To reduce the size of the map
    self.explored_rate = 0
    self.num_heading = num_heading
    self.adaptive_generation = adaptive_generation
    self.speed = speed
    
    #Initialise Graph
    self.map = Map(k_size=k_size, ground_truth_size=self.ground_truth_size, sensor_range=self.sensor_range, num_heading=self.num_heading, plot=plot, adaptive_generation=self.adaptive_generation)
    self.map.route_node.append(self.start_location_heading)         # [208, 288, 0] list, only first robot position
    self.node_coordinates, self.graph, self.node_utility, self.node_full_utility, self.visited_node = None, None, None, None, None
    self.froniers = None
   
    self.begin()
    
    # Plotting functions
    self.plot=plot
    self.gif_frame_files = []
    self.heading_list = []
      
    
  def import_ground_truth(self, map_index):
    # occupied 1, free 255, unexplored 127
    ground_truth = (io.imread(map_index, 1) * 255).astype(int)
    robot_location = np.nonzero(ground_truth == 208)
    robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
    ground_truth = (ground_truth > 150)
    ground_truth = ground_truth * 254 + 1
    return ground_truth, robot_location   

  def prepareEnv(self, env, show=False):
    #127: occupied, 208: starting region, 194: free space
    ground_truth = (io.imread((self.img_path + '/' + env), as_gray=True)*255).astype(int) + self.add
    
    if show:
      io.imshow(ground_truth)
      io.show()
    starting_region = np.where(ground_truth == 208)
    robot_location = np.array([starting_region[1][0], starting_region[0][0]])
    ground_truth = (ground_truth>150)
    ground_truth = ground_truth * 254 + 1
    return ground_truth, robot_location
  
  def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coordinates - position, axis=1))
        return index
  
  def begin(self):
    # Assume the surrounding starting area is fully observable
    #self.robot_belief = sensor_work(self.start_location, self.sensor_range, self.robot_belief, self.ground_truth)     #Initialise robot belief, including starting location, self.start_location (2,)
    self.robot_belief = start_location_scan(self.start_location, self.robot_belief, self.ground_truth, self.sensor_range)
    
    # Downsample the belief map to reduce the size of the map for faster computations
    self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min) #Equivalent to min pooling
    self.old_robot_belief = copy.deepcopy(self.robot_belief)
    self.frontiers = find_frontier(self.downsampled_belief, self.resolution)      
    self.node_coordinates, self.graph, self.node_utility, self.node_full_utility, self.visited_nodes = self.map.generate_local_graph(self.start_location, self.robot_belief, self.frontiers)

  def step(self, robot_location):

    reward_list = []
    self.heading_list = []
    # dist list is based on arrival sequence
    for node in self.map.nodes_list:
        if (node.coords == robot_location).all():
            heading = node.index
            angle = node.current_heading
    self.heading_list.append(heading)
    robot_location_full = np.append(robot_location, heading)

    #Update the node
    self.map.route_node.append(robot_location_full)                                           # Append robot position to route
    next_node_index = self.find_index_from_coords(robot_location)
    # Check from here, the next node index is based on the current node and observations, havent updated the graph yet
    self.map.nodes_list[next_node_index].set_visited()                                        # Set the node as visited using node function
    
    # Update robot belief based on the current position and current heading
    self.robot_belief = sensor_work_heading(robot_location, self.sensor_range, self.robot_belief, self.ground_truth, angle)
    self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)

    # Update frontiers
    #frontiers = find_frontier(self.downsampled_belief, self.resolution)                       # Last calculations is based on the belief updated by the current position
    num_observed_frontiers = self.node_utility[next_node_index]                                # Based on current timsestep utility
    #individual_reward = num_observed_frontiers / 50 - dist / 64 
    #reward_list.append(individual_reward)

    # Calculate the frontiers and node data based on the selected current heading
    frontiers = find_frontier(self.downsampled_belief, self.resolution)                          # Find new frontiers based on the new belief map
    self.node_coordinates, self.graph, self.node_utility, self.node_full_utility, self.visited_nodes = self.map.update_local_graph(self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
    self.old_robot_belief = copy.deepcopy(self.robot_belief)  
    
    # Update old frontiers
    num_observed_frontiers = calculate_num_observed_frontiers(self.frontiers, frontiers)
    self.frontiers = frontiers 
    
    # Update the explored rate
    self.explored_rate = evaluate_exploration_rate(self.robot_belief, self.ground_truth)
    
    # Calculate reward based on teamwork
    team_reward = num_observed_frontiers / 50
    
    # Check if the task is done
    done = check_done(self.node_utility)
    
    if done:
        #reward += np.sum(self.robot_belief == 255) / travel_dist            #free area explored per unit distance
        team_reward += 20
        
    #for i in range(len(reward_list)):
    #  reward_list[i] += team_reward

    return team_reward, done

  def plot_live_env(self, robots_route, robots_name, next_position, heading, plot_robot_name = True, plot_edge=False):

    plt.ion()                                                                                # Turn on interactive mode, image will be shown
    plt.cla()                                                                                # clear axes
    plt.imshow(self.robot_belief, cmap='gray')                                               # cmap for colourmap
    plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))                   # (xmin,xmax,ymin,ymax)
    
    # Plot the edges
    if plot_edge:
      for i in range(len(self.map.x)):
        plt.plot(self.map.x[i], self.map.y[i], 'tan', zorder=1)                              # tan; is the colour, zorder is the layer, smaller means further away from the viewer
    
    # Plot nodes and frontiers
    plt.scatter(self.node_coordinates[:, 0], self.node_coordinates[:, 1], c=self.node_utility, zorder=5)
    plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)            # Frontiers
    
    # Plot robot route
    for i , route in enumerate(robots_route):
      xPoints = [route[0]]
      yPoints = [route[1]]
      #plt.plot(xPoints[-1], yPoints[-1], 'b', linewidth=2)                                   # Robot path
      #plt.plot(xPoints[0], yPoints[0], 'mo', markersize=8, zorder = 10)   
      plt.plot(xPoints[-1], yPoints[-1], 'mo', markersize=8, zorder = 10)                    # Current robot location
      plt.plot(xPoints[0], yPoints[0], 'co', markersize=8, zorder = 5)                       # Start location
     
      # Plot robot name and next position
      if robots_name != [] and plot_robot_name:
        plt.text(xPoints[-1]-20, yPoints[-1]+25, s=robots_name[i], fontsize=12, zorder=10, weight='bold')                # Robot name
      if next_position != [] and plot_robot_name and next_position[i] != []:
        plt.plot(next_position[i][0], next_position[i][1], 'bo', markersize=10, zorder = 5)                # Next position
        plt.text(next_position[i][0], next_position[i][1]+25, s=robots_name[i], fontsize=12, zorder=10, weight='bold')    # Robot name
        #plt.arrow(next_position[i][0], next_position[i][1], 35*np.cos(heading[i]), 35*np.sin(heading[i]), color='b', zorder=12)        # Robot heading
    
    # Image title
    plt.title('Explored ratio: {:.4g}'.format(self.explored_rate))
    plt.tight_layout()
    #plt.axis('off')
    plt.pause(1e-2)
   
    
  def plot_env(self, current_episode, path, step, travel_dist, robots_route, plot_edge=False, scale=32, show_heading=False):
    plt.switch_backend('agg')                                                                # Close all matplotlib and set the backend to agg for PNG image
    # plt.ion()                                                                              # Turn on interactive mode, image will be shown
    plt.cla()                                                                                # clear axes
    plt.imshow(self.robot_belief, cmap='gray')                                               # cmap for colourmap
    plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))                   # (xmin,xmax,ymin,ymax)
    
    # Plot the edges
    if plot_edge:
      for i in range(len(self.map.x)):
        plt.plot(self.map.x[i], self.map.y[i], 'tan', zorder=1)                              # tan; is the colour, zorder is the layer, smaller means further away from the viewer
    
    # Plot nodes and frontiers
    plt.scatter(self.node_coordinates[:, 0], self.node_coordinates[:, 1], c=self.node_utility, zorder=5)
    plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)            # Frontiers
    
    # Plot robot route
    for i , route in enumerate(robots_route):
      if SHOW_HEADINGS:
        if self.heading_list[i] == 0:
          marker = '>'
        elif self.heading_list[i] == 1:
          marker = '^'
        elif self.heading_list[i] == 2:
          marker = '<'
        else:
          marker = 'v'
      else:
        marker = 'o'
      xPoints = route[0]
      yPoints = route[1]
      plt.plot(xPoints, yPoints, 'b', linewidth=2)                                           # Robot path
      plt.plot(xPoints[-1], yPoints[-1], 'm'+marker, markersize=8, zorder = 10)                    # Current robot location
      plt.plot(xPoints[0], yPoints[0], 'c'+marker, markersize=8, zorder = 5)                       # Start location
    
    # Image title
    plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, travel_dist))
    plt.title('Time taken: {:.3g}s'.format(travel_dist/(self.speed*scale)))
    #plt.tight_layout()
    plt.savefig('{}/{}_{}_samples.png'.format(path, current_episode, step, dpi=150))
    # plt.show()
    frame = '{}/{}_{}_samples.png'.format(path, current_episode, step)
    self.gif_frame_files.append(frame)
  
if __name__ == '__main__':
    env = Env(1,True)
    print(env.ground_truth_size[1])



