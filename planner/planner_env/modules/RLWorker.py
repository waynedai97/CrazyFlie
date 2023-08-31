#! /home/jimmy/anaconda3/envs/coverage/bin/python3
import torch
from .envRL import EnvRL
from .robotRL import RobotRL
from .test_parameter import *
import copy
import numpy as np
import os
import imageio

class TestWorker:
    def __init__(self, agent_list, policy_net, map_data, device='cuda', greedy=True, save_image=False):
        self.device = device
        self.greedy = greedy
        #self.env_idx= env_idx                           # Img number
        self.node_padding_size = NODE_PADDING_SIZE      # Pad to 360 
        self.k_size = K_SIZE                            # Number of neigbors
        self.save_image = save_image
        self.num_heading = NUM_HEADING
        self.adaptive_generation = ADAPTIVE_GENERATION
        self.speed = SPEED
        self.sensor_range = SENSOR_RANGE
        self.resolution = RESOLUTION
        self.scale = SCALE
        self.agent_list = agent_list
        self.map_x = map_data[0]
        self.map_y = map_data[1]
        self.map_x_cells = map_data[2]
        self.map_y_cells = map_data[3]
        self.env = EnvRL(k_size=self.k_size, train=False, num_heading=self.num_heading, plot=save_image, adaptive_generation=self.adaptive_generation, sensor_range=self.sensor_range, speed=self.speed, resolution=self.resolution)
        self.local_policy_net = policy_net
        
        self.robot_list = []
        self.all_robot_positions = []
        for item in enumerate(self.agent_list.items()):        # item() returns a list of tuples
            agent_id = item[1][0]
            agent = item[1][1]
            position = (agent.actual_pos.pose.position.x, agent.actual_pos.pose.position.y)
            corrected_position = self.correct_position(position)
            #print('Robot:', position, 'Corrected:', corrected_position)
            index = np.argmin(np.linalg.norm((self.env.node_coordinates - corrected_position), axis=1))
            robot_position = self.env.node_coordinates[index]
            robot = RobotRL(robot_id=item[0], cf_id = agent_id, position = robot_position)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)
    
    def correct_position(self, real_position):
        return (-(real_position[1])/self.map_x*self.map_x_cells, (self.map_y - real_position[0])/self.map_y*self.map_y_cells)
    
            
    def get_observations(self, robot_position):
        #print('Robot position', robot_position)
        current_node_index = self.env.find_index_from_coords(robot_position)

        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        # get observations
        node_coordinates = copy.deepcopy(self.env.node_coordinates)
        graph = copy.deepcopy(self.env.graph) #Return edges of the graphs
        node_utility = copy.deepcopy(self.env.node_utility)
        visitation = copy.deepcopy(self.env.visited_nodes)
        
        n_nodes = node_coordinates.shape[0]         #Number of nodes
        
        if COVERAGE:
            node_full_utility = copy.deepcopy(self.env.node_full_utility)
            node_full_utility = node_full_utility / 50
            node_full_utility_inputs = node_utility.reshape((n_nodes, 1))

        # Normalize observations
        node_coordinates = node_coordinates / 640          #Image largest dimension
        node_utility = node_utility / 50                    #Max utility, oroiginal 50

        # Transfer to node inputs tensor
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        
        occupied_node = np.zeros((n_nodes, 1))
        

        for position in self.all_robot_positions:                                #Loop through all robot position
            #print('Decision based on:', self.all_robot_positions)
            index = self.env.find_index_from_coords(position)
            #print('Robot position:', position, 'Index:', index, 'current robot', current_index.item())
            if index == current_index.item():                 #If the robot is at the same position as the current node
                occupied_node[index] = -1
            else:
                if (visitation[index] != 1).any():                    #If the node has been visited
                    occupied_node[index] = 1
                     #Apply the padding, bottom rows of zeros

        # calculate a mask for padded node
        node_padding_mask = torch.zeros((1, 1, node_coordinates.shape[0]), dtype=torch.int64).to(self.device) #3D array, 1 row of 0
        node_padding = torch.ones((1, 1, self.node_padding_size-node_coordinates.shape[0]), dtype=torch.int64).to(self.device) #1 row of 1
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1) #-1 means concatenate along last dimension

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())  #Check this
        edge_inputs = []
        for node in graph:                
            node_edges = list(map(int, node))                         # Loop from node id 0 to n_nodes, and extract all the to nodes
            edge_inputs.append(node_edges)
            assert len(node_edges) != 0  
            
        bias_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(bias_matrix).float().unsqueeze(0).to(self.device) #Add 1 dimension to the first row
        
        # Pad the edge mask to the padding size
        assert len(edge_inputs) < self.node_padding_size
        #Pad right and bottom
        padding = torch.nn.ConstantPad2d((0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)    # Pad with 1
        edge_mask = padding(edge_mask)
        padding2 = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - len(edge_inputs)))

        for edges in edge_inputs:              #Pad zero until k-numbers size
            while len(edges) < self.k_size:
                edges.append(0)
        
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, k_size)
        edge_inputs = padding2(edge_inputs)
        edge_padding_mask = torch.zeros((1, len(edge_inputs), K_SIZE), dtype=torch.int64).to(self.device)

        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)   #Fills matrix with ones
        #print(one)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        if COVERAGE:
            node_inputs = np.concatenate((node_coordinates, node_full_utility_inputs, node_utility_inputs, visitation, occupied_node), axis=1)
        else:
            node_inputs = np.concatenate((node_coordinates, node_utility_inputs, visitation, occupied_node), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)
        
        assert node_coordinates.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coordinates.shape[0]))
        node_inputs = padding(node_inputs)
        
        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask,  edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)                                          # Return exponential of value, .long converts to integer, removes second dimension if its 1

        next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
        next_position = self.env.node_coordinates[next_node_index]                                                          # Converts index to heading coordinates

        return next_position, action_index

    def solve_conflict(self, arriving_sequence, next_position_list, dist_list):

        # arriving sequence is the robot id in the order of arrival based on the distance to the next position
        for j, [robot_id, next_position] in enumerate(zip(arriving_sequence, next_position_list)):
            moving_robot = self.robot_list[robot_id]

            if next_position[0] + next_position[1] * 1j in (next_position_list[:, 0] + next_position_list[:, 1] * 1j)[:j]:    #j to slice till the previous robot next position

                dist_to_next_position = np.argsort(np.linalg.norm(self.env.node_coordinates - next_position, axis=1))         # Only check for position without heading
                k = 0
                
                # dist_to_next_position is the index of the node in the order of distance to the next position
                while next_position[0] + next_position[1] * 1j in (next_position_list[:, 0] + next_position_list[:, 1] * 1j)[:j]:
                    k += 1
                    next_position = self.env.node_coordinates[dist_to_next_position[k]]                                         #Relocate robot to nearby node

            dist = np.linalg.norm(next_position - moving_robot.robot_position)
            next_position_list[j] = next_position #
            dist_list[j] = dist
            moving_robot.travel_dist += dist
            moving_robot.robot_position = next_position
            self.all_robot_positions[robot_id] = next_position

        return next_position_list, dist_list
       
    # Robot moves to the next node and get observations from that node then get the next node
    def run_episode(self, agent_id, agent_list):
        done = False
        robots_route = []
        self.all_robot_positions = []
        
        # Calculate all robot_positions and store it
        for item in enumerate(self.agent_list.items()):        # item() returns a list of tuples
            #agent_id = item[1][0]
            agent = item[1][1]
            position = (agent.actual_pos.pose.position.x, agent.actual_pos.pose.position.y)
            corrected_position = self.correct_position(position)
            index = np.argmin(np.linalg.norm((self.env.node_coordinates - corrected_position), axis=1))
            robot_position = self.env.node_coordinates[index]
            self.all_robot_positions.append(robot_position)
            
        # Get robot data
        current_agent = agent_list[agent_id]
        real_position = (current_agent.actual_pos.pose.position.x, current_agent.actual_pos.pose.position.y)
        current_position = self.correct_position(real_position)
            
        
        # Get robot observations
        robot_observations = self.get_observations(current_position)
        
        # Commpute location of the robot
        next_position, action_index = self.select_node(robot_observations)                       # Next_position in (x,y)
        
        # Identify robot heading
        for node in self.env.map.nodes_list:
                if (node.coords == next_position).all():
                    heading = node.current_heading
        
        # Update the environment
        team_reward, done = self.env.step(next_position)
        '''
        for item in self.agent_list.items():  # Iterate through list of available agents
            agent_id = item[0]
            agent = item[1]
            real_robot_position = (agent.actual_pos.pose.position.x, agent.actual_pos.pose.position.y)
            robot_position = (self.correct_position(real_robot_position))
            robots_route.append(robot_position)
        print(robots_route)
        self.env.plot_live_env(robots_route)
        '''
        '''
        next_position_list = []
        dist_list = []
        travel_dist_list = []
        dist_array = np.zeros((self.n_agent, 1)) 
        
        # Check the code for similiar location
        new_position_list = []
        for robot_id, deciding_robot in enumerate(self.robot_list):
            
            observations = deciding_robot.observations
            deciding_robot.save_observations(observations)
            
            next_position, action_index = self.select_node(observations)                       # Next_position in (x,y)
            deciding_robot.save_action(action_index)
            
            for node in self.env.map.nodes_list:
                if (node.coords == next_position).all():
                    heading = node.current_heading
            new_position_list.append([action_index.item(), heading])
            dist = np.linalg.norm(next_position - deciding_robot.robot_position)

            dist_array[robot_id] = dist
            dist_list.append(dist)
            travel_dist_list.append(deciding_robot.travel_dist)
            next_position_list.append(next_position)
        #assert new_position_list != old_position_list, ("Robot is stuck", print(new_position_list, old_position_list))
        
        old_position_list = new_position_list  
        arriving_sequence = np.argsort(dist_list)                                               # Sort the distance travelled by the robots and return the indices
        next_position_list = np.array(next_position_list)
        dist_list = np.array(dist_list)
        travel_dist_list = np.array(travel_dist_list)
        next_position_list = next_position_list[arriving_sequence]                              # Sort the next positions of the robots according to the indices
        dist_list = dist_list[arriving_sequence]                                                # Sort the distances travelled by the robots according to the indices
        travel_distance_list = travel_dist_list[arriving_sequence]                              # Sort the cumulative distances travelled by the robots according to the indices
        # distances increases when robot is not moving, check until here

        # Travel distance is based on the previous timestep, next position is based on the current timestep
        next_position_list, distance_list = self.solve_conflict(arriving_sequence, next_position_list, dist_list)   # Updates travel position, robot position, all robot positions
        reward_list, done = self.env.step(next_position_list, distance_list, travel_distance_list)                  # new_position_list is a list of (x,y) coordinates
        
        for reward, robot_id in zip(reward_list, arriving_sequence):
            robot = self.robot_list[robot_id]
            robot.observations = self.get_observations(robot.robot_position)
            robot.save_reward_done(reward, done)
            robot.save_next_observations(robot.observations)
        
        current_travel_dist_list = []  
        for robot_id, deciding_robot in enumerate(self.robot_list):
            current_travel_dist_list.append(deciding_robot.travel_dist)
        '''
        '''   
        # Display episode summary
        print("\n*****************************************")
        print("Summary {} ".format(i//TIMESTEP_INTERVAL))
        print("Current timestep: {}".format(i))
        print("Number of agents: {}".format(self.n_agent))
        print("Reward: {:.5g}".format(reward))
        print("Cumulative travel distance: {:.6g}".format(max(current_travel_dist_list)))
        print("Number of frontiers: {}".format(len(self.env.frontiers)))
        print("Exploration rate: {:.5g}".format(self.env.explored_rate))
        print("*****************************************")
        '''
        '''
        # save a frame
        if self.save_image:
            robots_route =[]
            for robot in self.robot_list:                          #Get all the robots' positions
                robots_route.append([robot.xPoints, robot.yPoints])
            if not os.path.exists(gifs_path):
                os.makedirs(gifs_path)
            self.env.plot_env(self.global_step, gifs_path, i, max(current_travel_dist_list), robots_route, plot_edge=PLOT_EDGE, scale=self.scale, show_heading=SHOW_HEADINGS)

        self.perf_metrics['travel_dist'] = max(current_travel_dist_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        
        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)
        '''
        return next_position, heading, done

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):       #Form matrix for all nodes
        size = len(edge_inputs)       
        bias_matrix = np.ones((size, size))           
        for i in range(size):                         #For each node
            for j in range(size):
                if j in edge_inputs[i]:               # Check index 0 all the edge node
                    bias_matrix[i][j] = 0.            #If edge_inputs got the index, set it as zero, which means neigbours node
        return bias_matrix
    
    def make_gif(self, path, current_episode):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, current_episode, self.env.explored_rate), mode='I', duration=0.3) as writer:
            for frame in self.env.gif_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('Generation of gif completed successfully!\n')

        # Remove files
        for filename in self.env.gif_frame_files[:-1]: #Keep last frame
            os.remove(filename)