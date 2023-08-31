import numpy as np
import copy

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = x0.round() #x0,y0: robot position, x1,y1: extent of sensor range
    y0 = y0.round() #round up to the nearest integer
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)   
    x, y = x0, y0
    error = dx - dy
    x_increment = 1 if x1 > x0 else -1 #According to orientation of sensor beam
    y_increment = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 10

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]: # Check if the sensor beam is within the map
        y = np.int64(y)
        k = ground_truth.item(y, x)                                          # Extract the pixel value
        if k == 1 and collision_flag < max_collision:                        # To build the thickness of the wall
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k !=1 and collision_flag > 0:
            break

        if x == x1 and y == y1:                                             # Reach end of sensor range
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_increment
            error -= dy
        else:
            y += y_increment
            error += dx

    return robot_belief

def bresenham(start, end):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point. Convert results to tuple.
    """
    (x0, y0) = start
    (x1, y1) = end
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx).astype(int)
    dy = abs(dy).astype(int)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def sensor_scan(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = x0.round() #x0,y0: robot position, x1,y1: extent of sensor range
    y0 = y0.round() #roound up to the nearest integer
    x1 = x1.round()
    y1 = y1.round()
    list_of_points = bresenham((x0, y0), (x1, y1)) #Generate a list of points on the sensor beam from origin to the end
    for (x,y) in list_of_points: 
        if 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:  #Check if the sensor beam is within the map
            pixel_value = ground_truth.item(y, x) #Extract the pixel value
            robot_belief.itemset((y, x), pixel_value) #Update the robot belief map
            if pixel_value == 1:                      #Terminate when the sensor beam hits an obstacle
                break
            if x == x1 and y == y1:                   #Reach end of sensor range
                break 
    return robot_belief

def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        #print(type(x0),type(y0),type(x1),type(y1)) ,numpyint63, numpyint64, numpyfloat64, numpyfloat64
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief

def start_location_scan(robot_position,robot_belief, ground_truth, sensor_range):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0] - 230
    y0 = robot_position[1]
    while x0 < 330:
        while sensor_angle < 2 * np.pi:
            if x0 < 320:
                y0 = robot_position[1] + 35
            else:
                y0 = robot_position[1]
            x1 = x0 + np.cos(sensor_angle) * sensor_range
            y1 = y0 + np.sin(sensor_angle) * sensor_range
            robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
            sensor_angle += sensor_angle_inc
        sensor_angle = 0
        x0 += 10
    return robot_belief

    # x0 = robot_position[0] - 50
    # y0 = robot_position[1] - 50
    # x_final = x0 + 600
    # y_final = y0 + 650
    # while y0 < y_final:
    #     robot_belief = collision_check(x0, y0, x_final, y0, ground_truth, robot_belief)
    #     y0 += 1
    #return robot_belief
        
def sensor_work_heading(robot_position, sensor_range, robot_belief, ground_truth, angle):
    sensor_angle_inc = 0.5 / 180 * np.pi
    x0 = robot_position[0]
    y0 = robot_position[1]
    starting_angle = angle - np.pi/4                           # Convert heading to angle
    ending_angle = angle + np.pi/4                         # Ending angle
    while starting_angle < ending_angle:
        x1 = x0 + np.cos(starting_angle) * sensor_range
        y1 = y0 + np.sin(starting_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        starting_angle += sensor_angle_inc
    return robot_belief
        

def simulate_sensor(robot_position, sensor_range, robot_belief, ground_truth):  #Calculate the collision free map
    sensor_angle_inc = 0.5 / 180 * np.pi   #0.5 degrees convert to radians
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:        #360 degrees
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = sensor_scan(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief

def unexplored_area_check(x0, y0, x1, y1, current_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_increment = 1 if x1 > x0 else -1
    y_increment = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < current_belief.shape[1] and 0 <= y < current_belief.shape[0]:
        k = current_belief.item(y, x)
        if x == x1 and y == y1:
            break

        if k == 1:
            break

        if k == 127:
            current_belief.itemset((y, x), 0)
            break

        if error > 0:
            x += x_increment
            error -= dy
        else:
            y += y_increment
            error += dx

    return current_belief

def compare_unexplored_area(x0, y0, x1, y1, current_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    list_of_points = bresenham((x0, y0), (x1, y1))
    for (x,y) in list_of_points: 
        if 0 <= x < current_belief.shape[1] and 0 <= y < current_belief.shape[0]:  #Check if the sensor beam is within the map
            pixel_value = current_belief.item(y, x) #Extract the pixel value
            if pixel_value == 127:
                current_belief.itemset((y, x), 0)     #Assign unexplored area to unallocated free space
            if pixel_value == 1:                      #Terminate when the sensor beam hits an obstacle
                break
            if x == x1 and y == y1:                   #Reach end of sensor range
                break 
    return current_belief
    
def calculate_utility(current_position, sensor_range, robot_belief):
    sensor_angle_increment = 5 / 180 * np.pi
    sensor_angle = 0
    x0 = current_position[0]
    y0 = current_position[1]
    current_belief = copy.deepcopy(robot_belief)
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        current_belief = compare_unexplored_area(x0, y0, x1, y1, current_belief)
        sensor_angle += sensor_angle_increment
    utility = np.sum(robot_belief == 127) - np.sum(current_belief == 127)    #Difference between the number of unexplored areas
    return utility