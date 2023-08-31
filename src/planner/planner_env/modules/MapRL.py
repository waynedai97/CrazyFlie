from .Graph import Graph, a_star
import numpy as np
from .node import Node
from sklearn.neighbors import NearestNeighbors
import copy
from .test_parameter import *

class Map():
    def __init__(self, k_size, ground_truth_size, sensor_range, num_heading, plot=False, adaptive_generation=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coordinates = None
        self.plot = plot
        self.x = []
        self.y = []
        self.x_size = ground_truth_size[1] #640
        self.y_size = ground_truth_size[0] #480
        #assert self.x_size == 640 and self.y_size == 480, f"Ground truth size must be 640x480"
        self.sensor_range = sensor_range
        self.num_heading = num_heading
        self.heading_coordinates = np.empty([0,3], int) #([1,1,1],(1,3),ndarray)
        
        self.adaptive_generation = adaptive_generation
        
        #Generate the uniform points
        self.uniform_points = self.uniformPointsGenerator() #(900,2), ndarray
        
        #Node data
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.node_full_utility = None
        self.visited_nodes = None

    def uniformPointsGenerator(self):
        if self.adaptive_generation:
            Number_of_nodes_x = int(self.x_size / SPEED)
            Number_of_nodes_y = int(self.y_size / SPEED)
            node_number =max(Number_of_nodes_x, Number_of_nodes_y)
        else:
            node_number = 24
            
        x = np.linspace(0, self.x_size - 1, node_number).astype(int) #list start at index 0
        y = np.linspace(0, self.y_size - 1, node_number).astype(int) #cast and round to integer
        x_coordinates, y_coordinates = np.meshgrid(x, y) #give a matrix of x and y coordinates
        points = np.vstack([x_coordinates.T.ravel(), y_coordinates.T.ravel()]).T # ravel equivalent to reshape
        return points
    
    def clear_all_edge_nodes_data(self):
        self.graph = Graph()
        self.x = []
        self.y = []
    
    # Can use for x,y,h    
    def find_index_from_coords(self, node_coordinates, coordinates):
        return np.where(np.linalg.norm(node_coordinates - coordinates, axis=1) < 1e-5)[0][0] #Euclidean distance, L2 norm, take x axis and first value
        
    def clear_single_node_edges(self, coordinates):
        node_index = str(self.find_index_from_coords(self.node_coordinates, coordinates))
        self.graph.clear_edge(node_index) #Function from graph.py
        
    def free_area(self, robot_belief): #Robot belief must be numpy array
        unoccupied_pixels = np.where(robot_belief == 255) #or 194?
        free_area = np.asarray([unoccupied_pixels[1], unoccupied_pixels[0]]).T #(y,x), array of tuples (n,2), ndarray
        return free_area
    
    def unique_coords(self, coordinates):
        x = coordinates[:, 0] + coordinates[:, 1] * 1j #Slice the columns to get x and y coordinates
        indices = np.unique(x, return_index=True)[1] #[1] to return the indices [0] will return the unique coordinates in imaginary plane
        coordinates = np.array([coordinates[idx] for idx in sorted(indices)]) 
        return coordinates
 
    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision
    
    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords-node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)   #Return indices that will sort an array
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k < node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            distance = dist_list[k]
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief): #If collision free, the if statement will be true due to not
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, distance) 
                
                if self.plot:
                        self.x.append([start[0], end[0]])
                        self.y.append([start[1], end[1]])
            k += 1
        return neighbor_index_list
    
    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(node_coords)
        distances, indices = knn.kneighbors(node_coords)

        for i, p in enumerate(node_coords):
            for j, neighbour in enumerate(node_coords[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])
    
    def generate_local_graph(self, robot_location, robot_belief, frontiers): #check if node is in free space
        self.clear_all_edge_nodes_data()
        
        #  Find coordinates of all nodes in the free area
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True) # Return common points, indices in the free area, and unifrom point
        node_coords = self.uniform_points[candidate_indices]                                                       # Get the coordinates of the candidate nodes that is in the free space
        # Add robot coordinates to the node coordinates as the first node
        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords)) # Robot location added to the node coordinates
        self.node_coordinates = self.unique_coords(node_coords).reshape(-1, 2)  #-1 means that the dimension is inferred from the length of the array and remaining dimensions
        
        # Generate collision free map
        self.find_k_neighbor_all_nodes(self.node_coordinates, robot_belief)
        
        # Calculate the utlity of each node (observable frontiers) and save the value
        self.node_utility = []
        self.node_full_utility = []
        
        for coordinates in self.node_coordinates:
            node = Node(coordinates, frontiers, self.sensor_range, robot_belief, self.num_heading)
            self.nodes_list.append(node)
            utility = node.utility_heading
            full_utility = node.utility
            self.node_utility.append(utility)
            self.node_full_utility.append(full_utility)
        self.node_utility = np.array(self.node_utility)
        self.node_full_utility = np.array(self.node_full_utility)
        
        self.visited_nodes = np.zeros((self.node_coordinates.shape[0],self.num_heading))
        node_coordinates_list = self.node_coordinates[:,0] + self.node_coordinates[:,1]*1j
        for node in self.route_node:
            heading = node[2]
            index = np.argwhere(node_coordinates_list.reshape(-1) == node[0]+node[1]*1j)[0]
            self.visited_nodes[index,heading] = 1

        return self.node_coordinates, self.graph.edges, self.node_utility, self.node_full_utility, self.visited_nodes
    
    def update_local_graph(self, robot_belief, old_robot_belief, frontiers, old_frontiers): #Update whole graph
        # Add uniform points of the new free area to the map
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0)*255)    # Get new area of the uniform nodes
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        self.node_coordinates = np.concatenate((self.node_coordinates, new_node_coords))                            # Add new nodes to the node coordinates, node index from previous timestep is preserved
        
        old_node_to_update = []
        # Update the collision free map of newly added nodes
        for coordinates in new_node_coords:
            neigbour_indices = self.find_k_neighbor(coordinates, self.node_coordinates, robot_belief)
            old_node_to_update += neigbour_indices
        old_node_to_update = set(old_node_to_update)
        
        # Update the collision free map of the old nodes
        for index in old_node_to_update:
            coords = self.node_coordinates[index]
            self.clear_single_node_edges(coords)
            self.find_k_neighbor(coords, self.node_coordinates, robot_belief)                                      # Find the neigbors nodes
            
        # Update the observable frontiers by identifying changes in fronteirs
        old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        observed_frontiers_index = np.where(np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False) # Check whether old frontiers is in new frontiers, then extract all false to get all frontiers that are not in new frontiers
        new_frontiers_index = np.where(np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False) #Check for new frontiers that is not in old frontiers
        observed_frontiers = old_frontiers[observed_frontiers_index]                                                         # Frontiers that have been observed
        new_frontiers = frontiers[new_frontiers_index]
        
        # Update utility for old nodes
        for node in self.nodes_list:
            if node.zero_utility_node is True:
                pass
            else:
                node.update_observable_frontiers(observed_frontiers, new_frontiers, robot_belief)
        
        # Iterate through the nodes and append the new nodes
        for new_coordinates in new_node_coords:
            node = Node(new_coordinates, frontiers, self.sensor_range, robot_belief, self.num_heading)
            self.nodes_list.append(node)
            
        self.node_utility = []   #Reset node utility
        self.node_full_utility = []
        self.visited_nodes = []
        
        # Append the new utility of all nodes
        for i, coords in enumerate(self.node_coordinates):
            utility = self.nodes_list[i].utility_heading
            full_utility = self.nodes_list[i].utility
            self.node_utility.append(utility)
            self.node_full_utility.append(full_utility)
            visitations = self.nodes_list[i].visitations
            self.visited_nodes.append(visitations)
        
        self.node_utility = np.array(self.node_utility)
        self.node_full_utility = np.array(self.node_full_utility)
        self.visited_nodes = np.array(self.visited_nodes)

        return self.node_coordinates, self.graph.edges, self.node_utility, self.node_full_utility, self.visited_nodes  
    
    def find_shortest_path(self, current, destination, node_coords):
        self.startNode = str(self.find_index_from_coords(node_coords, current))
        self.endNode = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(self.startNode), int(self.endNode), self.node_coords, self.graph)
        if self.startNode != self.endNode:
            assert route != []
        route = list(map(str, route)) #Convert the route to string
        return dist, route
    
if __name__ == '__main__':
    env = Map(20, (480,640), 80, True)
    robot_belief = np.array([[23,1,1,1,1,1,11],[1,1,127,1,1,11,1],[1,194,1,1,1,1,1]]) 
    start = (2,6)
    end = (1,3)
    collision = env.check_collision(start, end, robot_belief)

    