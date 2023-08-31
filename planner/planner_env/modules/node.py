import numpy as np
from .sensor import bresenham
from sklearn.cluster import DBSCAN

class Node():
    def __init__(self, coords, frontiers, sensor_range, robot_belief, num_heading):
        self.coords = coords
        self.observable_frontiers = []
        self.sensor_range = sensor_range
        self.num_heading = num_heading
        self.heading = list(np.arange(0, num_heading, 1)) #[0,1,2,3], list
        self.observable_frontiers_heading = [] #{heading: [] for heading in self.heading} # {0:[], 1:[], 2:[], 3:[]}, dict
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility, self.utility_heading, self.current_heading = self.get_node_utility()   #full_node_utility, node_heading_utility, heading 
        self.visitations = [0]*self.num_heading
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    
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
    
    def initialize_observable_frontiers(self, frontiers, robot_belief):
        fronteirs_dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)         # Caculate distances between node and all frontier points
        frontiers_in_range = frontiers[fronteirs_dist_list < self.sensor_range - 10]   # Find frontiers in range of sensor with some buffer
        for point in frontiers_in_range:                                               # Frontiers in range of sensor per point
            collision = self.check_collision(self.coords, point, robot_belief)          
            if not collision:
                self.observable_frontiers.append(point)                                # Append obsrvable frontiers, full 360 degrees        
                
                
    def get_node_utility(self):
        full_node_utility = len(self.observable_frontiers)                              # Utility is number of observable frontiers points
        if full_node_utility > 0:
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(self.observable_frontiers)   #3
            
            # Identify unique labels
            unique_labels = set(clustering.labels_)
            if len(unique_labels)!= 0: 
                # Count the number of points in each cluster
                cluster_size = [list(clustering.labels_).count(label) for label in unique_labels]
                
                # Find the index of the largest cluster
                largest_cluster_index = np.argmax(cluster_size)
                
                largest_cluster_label = list(unique_labels)[largest_cluster_index]
                
                # get the coordinates of the points in the largest cluster
    
                largest_cluster_points = np.array([point for i, point in enumerate(self.observable_frontiers) if clustering.labels_[i] == largest_cluster_label])
                self.observable_frontiers_heading = largest_cluster_points
                #print("The largest cluster has {} points with index {}".format(cluster_size[largest_cluster_index], largest_cluster_label))
            
                # Find the heading of the largest cluster
                centroid = [sum(largest_cluster_points[:,0])/len(largest_cluster_points[:,0]), sum(largest_cluster_points[:,1])/len(largest_cluster_points[:,1])]
                dx = centroid[0] - self.coords[0]                                         # Difference in x coordinates
                dy = centroid[1] - self.coords[1]                                         # Difference in y coordinates
                theta = np.arctan2(dy, dx)
                theta = (theta + np.pi*2) % (np.pi*2)
                node_heading_utility = len(largest_cluster_points)
                heading = theta
            # If a very small cluster is available   
            else:
                node_heading_utility, heading = 2, np.random.choice(360,1)*np.pi/180
        else:
            node_heading_utility, heading = 0, np.random.choice(360,1)*np.pi/180
            
        if heading == 0 and heading < np.pi/2:
            self.index = 0
        elif heading >= np.pi/2 and heading < np.pi:
            self.index = 1
        elif heading >= np.pi and heading < 3*np.pi/2:
            self.index = 2
        else:
            self.index = 3 
        
        return full_node_utility, node_heading_utility, heading                          # Utility is number of observable frontiers points
    
    def update_observable_frontiers(self, observed_frontiers, new_frontiers, robot_belief):
        # Remove observed frontiers from observable frontiers
        if observed_frontiers != []:
            observed_index = []
            for i, point in enumerate(self.observable_frontiers):                       # i is index, point is coordinates
                if point[0] + point[1] * 1j in observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j:
                    observed_index.append(i)
            for index in reversed(observed_index):                                      # largest index is first element
                self.observable_frontiers.pop(index)                                    # Remove observed frontiers from observable frontiers
                    
        # Add new frontiers to observable frontiers
        if new_frontiers != []:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.sensor_range - 15]   # Only check frontiers in range of sensor
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.append(point)
                    
        self.utility, self.utility_heading, self.current_heading = self.get_node_utility() #Calculate node utility
        
        if self.utility == 0:
            self.zero_utility_node = True
            self.utility_heading = 0
        else:
            self.zero_utility_node = False
    
    # Set observable frontiers to empty list and utility to 0   
    # set_visited is called before update to local graph
    def set_visited(self):
        index_to_remove = []
        for frontier in self.observable_frontiers_heading:
            for index, point in enumerate(self.observable_frontiers):
                if (point == frontier).all():
                    if index not in index_to_remove:
                        index_to_remove.append(index)
        for index in reversed(index_to_remove):                                      # largest index is first element
                self.observable_frontiers.pop(index)  
        self.observable_frontiers_heading = []
        
     
        self.visitations[self.index] = 1
        self.utility, self.utility_heading, self.current_heading = self.get_node_utility()
        
        if self.utility == 0:
            self.zero_utility_node = True