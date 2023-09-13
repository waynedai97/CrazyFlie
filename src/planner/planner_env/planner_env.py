import pickle
import numpy as np
from time import time  # https://realpython.com/python-time-module/
from functools import partial
from .env.task_env import TaskEnv
from std_srvs.srv import Trigger
import rclpy
import os
import copy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from crazyswarm_application.msg import AgentState, AgentsStateFeedback, UserCommand
from crazyflie_interfaces.srv import Land
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from crazyswarm_application.srv import Agents 


# import torch
# from .modules.test_parameter import *
# from .modules.model import PolicyNet
# from .modules.RLWorker import TestWorker

def get_euclid_dist(point_a, point_b):
    """Get the 3D euclidean distance between 2 geometry_msgs.msg.Point messages

    Parameters
    ----------
    point_a : geometry_msgs.msg.Point
        Origin Point
    point_b : geometry_msgs.msg.Point
        Goal Point

    Returns
    -------
    float
        Distance between point a and b
    """
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    dz = point_a.z - point_b.z

    return np.sqrt(dx * dx + dy * dy + dz * dz)



class planner_ROS(Node):

    def __init__(
            self, task_env=None, route_path=""
    ):
        super().__init__(
            "planner_env",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Set the policy network and planner
        # self.policy_net = policy_net
        self.RLPlanner = None
        self.episode_id = 8561
        self.robots_route = []
        self.robots_name = []
        self.next_position = []
        self.robots_heading = []

        # Set the agent list
        self.agent_list = {}
        self.num_agents = len(self.agent_list)
        self.inactive_agent_list = []
        self.check_time = time()
        self.mission_max_time = 60 * 20

        #####
        # Parameters
        #####

        # Agent parameters
        self.agent_timeout = self.get_parameter('agent_timeout').get_parameter_value().double_value
        
        self.pub_wp_timer_period = self.get_parameter('pub_wp_timer_period').get_parameter_value().double_value
        
        self.check_agent_timer_period = self.get_parameter(
            'check_agent_timer_period').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.height = self.get_parameter('agent_height').get_parameter_value().double_value
        self.working_height = 1.0
        self.route_path = self.get_parameter(
            'route_path').get_parameter_value().string_value  # "/home/ur10/swarming/crazyswarm2_ws/src/planner/planner_env/route_ros.yaml"#self.get_parameter('route_path').get_parameter_value().string_value
        self.env_path = self.get_parameter(
            'env_path').get_parameter_value().string_value  # "/home/ur10/swarming/crazyswarm2_ws/src/planner/planner_env/env_ros.pkl" #self.get_parameter('env_path').get_parameter_value().string_value
        self.arena_scale = self.get_parameter('arena_scale').get_parameter_value().double_value
        self.agent_arena_velocity = self.get_parameter('agent_arena_velocity').get_parameter_value().double_value
        self.agent_env_velocity = self.get_parameter('agent_env_velocity').get_parameter_value().double_value
        self.const_time_bias = self.get_parameter('const_time_bias').get_parameter_value().double_value
        self.goal_difference = self.get_parameter('goal_difference').get_parameter_value().double_value
        self.pos_bias = self.get_parameter('pos_bias').get_parameter_value().double_value
        self.working_time_bias = self.get_parameter('working_time_bias').get_parameter_value().double_value
        self.depot_loc = self.get_parameter('arena_depot').get_parameter_value().double_array_value
        print(f'The working time bias for this sim is - {self.working_time_bias}')
        print(f'The depot location in the rviz sim is {self.depot_loc}')
        self.land_on_node = True

        self.land_client = self.create_client(Land, '/all/land')

        # self.create_service(Agents, "/external/receive", self.external_callback)
        print("------- Check data -----------")
        # print('agent timeout', self.agent_timeout)
        print('Waypoint timer', self.pub_wp_timer_period)
        # print('check agent', self.check_agent_timer_period)
        # print('goal tolerance', self.goal_tolerance)
        # print('height', self.height)
        print(f'Loadin the env from the file present at {self.env_path}')
        self.task_env = pickle.load(open(self.env_path, 'rb'))
        # self.task_env = TaskEnv((5, 5), (10, 10), 1, 3, seed=0)
        self.agent_index = [0] * self.task_env.agents_num
        self.land_pose = []
        for i in range(self.task_env.agents_num):
            self.land_pose.append([0.0, 0.0, 0.0])
        self.first_call = [True for i in range(self.task_env.agents_num)]
        self.arrived = [False for i in range(self.task_env.agents_num)]
        self.debug = True
        self.agent_arrival_dict = dict()
        # Map parameters
        self.map_x = self.get_parameter('map_x').get_parameter_value().double_value
        self.map_y = self.get_parameter('map_y').get_parameter_value().double_value
        self.map_x_cells = self.get_parameter('map_x_cells').get_parameter_value().integer_value
        self.map_y_cells = self.get_parameter('map_y_cells').get_parameter_value().integer_value
        self.map_data = (self.map_x, self.map_y, self.map_x_cells, self.map_y_cells)
        self.finished_count = 0
        # self.pub_srv = self.create_service(Trigger, 'add_two_ints', self.publish_waypoints)
        # self.goto_cli = self.create_client(GoTo, 'add_two_ints')
        print('Check map data', self.map_data)
        self.marker_publishers = []
        for i in range(self.task_env.agents_num):
            self.marker_publishers.append(self.create_publisher(MarkerArray, 'cf' + str(i + 1) + '_marker', 10))

        if self.debug == True:
            for i in range(self.task_env.agents_num):
                t = (int(0), float(0.0), float(0.0))
                self.agent_arrival_dict[i] = [str(t)]
        #####
        # Create publishers
        #####
        self.usercommand_pub = self.create_publisher(UserCommand, "/user/external", 10)
        self.node_dic = dict()
        self.node_markers = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        for i in range(self.task_env.tasks_num):
            self.node_dic[i] = {'agents': [],
                                'requirement': self.task_env.task_dic[i]['requirements'][0],
                                'working_start_time': 0.0,
                                # 'agent_pose_bias': []#[0.5*x for x in range(self.task_env.task_dic[i]['requirements'][0])]
                                'travelling_agents': [],
                                }

        #####
        # Create subscribers
        #####
        '''
        Create subscriber for agent state feedback
        Agent state feedback will create a agents_list and create a subscriber for each agent's pose
        '''
        self.agent_state_sub = None
        self.create_service(Agents, "/external/receive", self.external_callback)
        # self.pub_wp_timer = self.create_timer(
        #     self.pub_wp_timer_period, self.publish_waypoints
        # )
        self.agent_pose_sub = []
        self.land_clients = []
        self.names = []
        # self.create_timer(0.2, self.publish_waypoints)

        # execute the task_env to load the agents
        self.load_execute_env(self.task_env, self.route_path)
        self.scale_arena_env()
        self.current_time = time()

    def publish_node_markers(self):

        marker_array = MarkerArray()
        marker_array.markers = []  # Clear previous markers
        for i in range(self.task_env.tasks_num):
            # Create a moving marker
            moving_marker = Marker()
            moving_marker.header = Header(frame_id='world')
            moving_marker.type = Marker.MESH_RESOURCE
            moving_marker.action = Marker.ADD
            moving_marker.pose.position.x = self.task_env.task_dic[i]['location'][
                0]  # Update the x position based on time
            moving_marker.pose.position.y = -self.task_env.task_dic[i]['location'][
                1]  # Update the y position based on time
            moving_marker.pose.position.z = 0.
            moving_marker.pose.orientation.w = 1.0
            moving_marker.scale.x = 1.0
            moving_marker.scale.y = 1.0
            moving_marker.scale.z = 1.0
            moving_marker.mesh_resource = 'package://planner_env/stl/cone.stl'  # Replace with the path to your STL file
            moving_marker.color.r = 1.0
            moving_marker.color.g = 1.0
            moving_marker.color.b = 0.0
            moving_marker.color.a = 1.0
            moving_marker.ns = str(i)

            marker_array.markers.append(moving_marker)

        # Publish the moving marker
        self.node_markers.publish(marker_array)

    def publish_drone_markers(self, agent_idx, pose):
        marker_array = MarkerArray()
        moving_marker = Marker()
        moving_marker.header = Header(frame_id='world')
        moving_marker.type = Marker.MESH_RESOURCE
        moving_marker.action = Marker.ADD
        moving_marker.pose.position.x = pose.pose.position.x
        moving_marker.pose.position.y = pose.pose.position.y
        moving_marker.pose.position.z = pose.pose.position.z
        moving_marker.scale.x = 0.005
        moving_marker.scale.y = 0.005
        moving_marker.scale.z = 0.005
        moving_marker.mesh_resource = 'package://planner_env/stl/cf2_model.stl'  # Replace with the path to your STL file
        moving_marker.color.r = 0.0
        moving_marker.color.g = 0.0
        moving_marker.color.b = 1.0
        moving_marker.color.a = 1.0

        marker_array.markers.append(moving_marker)

        # Publish the moving marker
        self.marker_publishers[agent_idx].publish(marker_array)

    def agent_callback(self, msg):
        print("Pose is ", msg.pose.position.x)

    def scale_arena_env(self):
        arena_vel = self.agent_arena_velocity
        env_vel = self.agent_env_velocity
        arena_scale = self.arena_scale
        working_time_bias = self.working_time_bias
        const_time_bias = self.const_time_bias

        print(f'The arena is being scaled by {arena_scale} times wrt to the env')
        print(f'The velocity of the agents in the arena is {arena_vel}')
        print(f'The velocity of the agents in the actual env was {env_vel}')


        for agent in self.task_env.agent_dic:
            for i in range(len(self.task_env.agent_dic[agent]['arrival_time'])):
                old_ar_t = self.task_env.agent_dic[agent]['arrival_time'][i]
                self.task_env.agent_dic[agent]['arrival_time'][i] *= (
                            (env_vel * arena_scale) / arena_vel)  # Magic numbers for scaling arrival time
                self.task_env.agent_dic[agent]['arrival_time'][i] += const_time_bias  # takeoff time
                new_arrival_time = self.task_env.agent_dic[agent]['arrival_time'][i]

        for node in self.task_env.task_dic:
            # print(self.task_env.task_dic[node]['requirement'])
            self.task_env.task_dic[node]['location'] *= arena_scale
            self.task_env.task_dic[node]['time_start'] *= ((env_vel * arena_scale) / arena_vel)
            self.task_env.task_dic[node]['time_start'] += const_time_bias
            self.task_env.task_dic[node]['time'] += working_time_bias
            new_working_time = self.task_env.task_dic[node]['time']
            print(f' The working time for task node {node} is changed to {new_working_time }')
            self.task_env.task_dic[node]['time_finish'] = self.task_env.task_dic[node]['time_start'] + \
                                                          self.task_env.task_dic[node]['time']

    def load_execute_env(self, task_env, route_path):

        # LOAD THE ROUTES
        print('here')
        yaml_result_file = self.route_path
        print(f'Loading the route path from the file {yaml_result_file}')
        # yaml_param_file = route_path + "/planner_param.yaml"
        routes = []
        if os.path.exists(yaml_result_file):
            with open(yaml_result_file, 'r') as stream:
                route_dict = yaml.safe_load(stream)

            for agent in route_dict:
                routes.append(route_dict[agent])

            if routes is None:
                return None
            for i in range(len(routes)):
                if not routes[i]:
                    continue
                else:
                    task_env.pre_set_route(copy.copy(routes[i]), i)

        task_env.execute_by_route()

    # def external_callback(splanner_env/env/task_env.py
    def external_callback(self, request, response):

        self.get_logger().info("start timer")

        # Start by checking adding any new agents to the agent list and start their pose subscriber
        if self.agent_list == {}:
            for name in request.names:
                self.names.append(name)

        self.agent_state_sub = self.create_subscription(
            AgentsStateFeedback, "/agents", self.agent_state_callback, 10
        )

        # self.pub_wp_timer = self.create_timer(
        #     self.pub_wp_timer_period, self.publish_waypoints
        # )


        # self.crash_agent_timer = self.create_timer(
        #     1, self.check_crash_agent
        # )
        return response

    def plot_agents(self):
        """
        Call back to plot the agent and map
        """
        if (self.robots_route != []) and (self.RLPlanner != None):
            self.RLPlanner.env.plot_live_env(self.robots_route, self.robots_name, self.next_position,
                                             self.robots_heading)
        else:
            return

    def change_height(self, goal_pos, agent_name):
        waypoint_cmd_old = self.create_usercommand(
            cmd="goto_velocity",
            uav_id=[agent_name],
            goal=Point(
                x=goal_pos[0],
                y=-goal_pos[1],
                z=self.working_height,
            ),
            yaw=0.0,  # float(heading_real),
            is_external=True)
        self.usercommand_pub.publish(waypoint_cmd_old)
        print(f'Landing the agent {agent_name}')

    def takeoff_all(self):
        """Helper method to command all crazyflies to take off
        """
        waypoint = self.create_usercommand(
            cmd="takeoff_all",
            uav_id=[],
        )

        self.usercommand_pub.publish(waypoint)

        self.get_logger().info("Commanded all crazyflies to take off")


    def agent_pose_callback(self, pose, agent_idx):
        """Subscriber callback to save the actual pose of the agent

        Parameters
        ----------
        pose : geometry_msgs.msg.PoseStamped
            PoseStamped message
        agent_id : String
            Id of Agent
        """
        # print(agent_id)
        agent_name = 'cf' + str(
            agent_idx + 1)  # change the var name to agent_name and agent_idx to agent_id to decrease redundancy
        # print(agent_id)
        current_pose = [pose.pose.position.x, pose.pose.position.y]
        self.publish_node_markers()

        if self.first_call[agent_idx]:
            self.first_call[agent_idx] = False
            self.land_pose[agent_idx] = [current_pose[0], current_pose[1], 0.0]

        agent_node_idx = self.agent_index[agent_idx]
        if (agent_node_idx < len(self.task_env.agent_dic[agent_idx]['arrival_time'])):
            next_task_node = self.task_env.agent_dic[agent_idx]['route'][self.agent_index[agent_idx]]
            if next_task_node == -1:
                goal_pos = self.land_pose[agent_idx]
                print(f'agent {agent_idx} going to goal pose {goal_pos} ')
                self.agent_index[agent_idx] += 1
                goal_abs_difference = abs(goal_pos[0] - current_pose[0]) + abs(goal_pos[1] - current_pose[1])
                # if goal_abs_difference < 1:
                #     if self.finished_count >= self.task_env.agents_num:
                #     self.finished_count += 1

                waypoint_cmd_old = self.create_usercommand(
                    cmd="goto_velocity",
                    uav_id=[agent_name],
                    goal=Point(
                        x=goal_pos[0],
                        y=-goal_pos[1],
                        z=self.height,
                    ),
                    yaw=0.0,  # float(heading_real),
                    is_external=True)
                self.usercommand_pub.publish(waypoint_cmd_old)
                self.publish_drone_markers(agent_idx, pose)
            else:
                waypoint_cmd_old = self.create_usercommand(
                    cmd="goto_velocity",
                    uav_id=['cf0'],
                    goal=Point(
                        x=0.0,
                        y=0.0,
                        z=self.height,
                    ),
                    yaw=0.0,  # float(heading_real),
                    is_external=True)
                self.usercommand_pub.publish(waypoint_cmd_old)
                self.publish_drone_markers(agent_idx, pose)
                if agent_idx not in self.node_dic[next_task_node]['travelling_agents']:
                    self.node_dic[next_task_node]['travelling_agents'].append(agent_idx)
                    pose_bias = self.pos_bias * self.node_dic[next_task_node]['travelling_agents'].index(agent_idx)
                else:
                    pose_bias = self.pos_bias * self.node_dic[next_task_node]['travelling_agents'].index(agent_idx)

                goal_pos = self.task_env.task_dic[next_task_node]['location'] - pose_bias
                current_time = time() - self.current_time

                tracker_idx = self.agent_index[agent_idx]
                arrival_time = self.task_env.agent_dic[agent_idx]['arrival_time'][tracker_idx]
                goal_abs_difference = abs(goal_pos[0] - current_pose[0]) + abs(-goal_pos[1] - current_pose[1])
                # check1 = (current_time > arrival_time)
                check1 = (goal_abs_difference < self.goal_difference)
                if (check1) or (agent_idx in self.node_dic[next_task_node]['agents']):
                    self.publish_drone_markers(agent_idx, pose)
                    # if abs(current_time - arrival_time) > 0.5:
                    #     print(f'agent {agent_idx} missed its arrival time at node {}')
                    if agent_idx not in self.node_dic[next_task_node]['agents']:
                        print(
                            f'agent  {agent_idx + 1} arrived at node {next_task_node} at time {current_time} while the arrival time was {arrival_time} ')
                        land_client = self.land_clients[agent_idx]
                        self.node_dic[next_task_node]['agents'].append(agent_idx)
                        if self.land_on_node:
                            self.change_height(goal_pos, agent_name)

                    if len(self.node_dic[next_task_node]['agents']) == self.node_dic[next_task_node]['requirement']:
                        if self.node_dic[next_task_node]['working_start_time'] == 0:
                            print(f'Starting work at node {next_task_node}')
                            self.node_dic[next_task_node]['working_start_time'] = time()

                        print(f'all agents have arrived to the node {next_task_node}')
                        working_time = time() - self.node_dic[next_task_node]['working_start_time']
                        print(f'The working time for the node {next_task_node} is {working_time}')
                        if (working_time > self.task_env.task_dic[next_task_node]['time']):  # Completed the task
                            for agent in (self.node_dic[next_task_node]['agents']):
                                self.agent_index[agent] += 1
                                agent_next_node = self.task_env.agent_dic[agent]['route'][self.agent_index[agent]]
                                time_taken = time() - self.node_dic[next_task_node]['working_start_time']
                                actual_time = self.node_dic[next_task_node]['working_start_time']
                                print(
                                    f'agent{agent + 1} going to the next node {agent_next_node} the time taken on the task was {time_taken} while the supposed time was{actual_time}')
                else:

                    # print(f'agent {agent_idx} going to goal pose {goal_pos} ')
                    waypoint_cmd_old = self.create_usercommand(
                        cmd="goto_velocity",
                        uav_id=[agent_name],
                        goal=Point(
                            x=goal_pos[0],
                            y=-goal_pos[1],
                            z=self.height,
                        ),
                        yaw=0.0,  # float(heading_real),
                        is_external=True)
                    self.usercommand_pub.publish(waypoint_cmd_old)
                    self.publish_drone_markers(agent_idx, pose)
            self.finished = False
        else:
            goal_pos = self.land_pose[agent_idx]
            # print(f'agent {agent_idx} going to goal pose {goal_pos} ')
            self.agent_index[agent_idx] += 1
            goal_abs_difference = abs(goal_pos[0] - current_pose[0]) + abs(goal_pos[1] - current_pose[1])

            if goal_abs_difference > self.goal_difference and not self.arrived[agent_idx]:

                waypoint_cmd_old = self.create_usercommand(
                cmd="goto_velocity",
                uav_id=[agent_name],
                goal=Point(
                    x=goal_pos[0],
                    y=goal_pos[1],
                    z=self.height,
                ),
                yaw=0.0,  # float(heading_real),
                is_external=True)
                self.usercommand_pub.publish(waypoint_cmd_old)


            else:
                self.arrived[agent_idx] = True
                print(f'Agent {agent_name} will not publish anymore')
            print(f'Agent {agent_name} has completed its routes')
            self.publish_drone_markers(agent_idx, pose)

    def agent_state_callback(self, agent_states):

        # # Start by checking adding any new agents to the agent list and start their pose subscriber
        if self.agent_list == {}:
            for agent in self.task_env.agent_dic:
                # print('hre')
                # print(agent)

                self.agent_list[agent] = self.task_env.agent_dic[agent]
                # print(agent.id)
                agent_name = "/cf" + str(agent + 1)
                print(agent_name)
                # self.current_time[agent] = time()
                self.agent_pose_sub.append(
                    self.create_subscription(
                        PoseStamped,
                        agent_name + "/pose",
                        partial(self.agent_pose_callback, agent_idx=agent),
                        10,
                    )
                )
                self.land_clients.append(self.create_client(Land, agent_name + "/land"))

        for agent in agent_states.agents:

            if agent.id in self.agent_list:
                self.agent_list[agent.id].flight_state = agent.flight_state
                self.agent_list[agent.id].connected = agent.connected
                self.agent_list[agent.id].completed = agent.completed
                self.agent_list[agent.id].mission_capable = agent.mission_capable
                # IF receiving pose of inactive agent, ignore
            else:
                # self.get_logger().error(
                #    f"Agent {agent.id} has been removed from agent list. Ignoring..."
                # )
                continue

    def create_usercommand(self, cmd, uav_id, goal=Point(), yaw=0.0, is_external=False):
        """Helper method to create a UserCommand message, used to send drone commands

        Parameters
        ----------
        cmd : String
            Command type
        uav_id : String
            Crazyflie agent ID
        goal : geometry_msgs.msg.Point()
            (x,y,z) goal
        yaw : float, optional
            yaw, by default 0.0
        is_external : bool, optional
            TODO, by default False

        Returns
        -------
        crazyswarm_application.msg.UserCommand()
            UserCommand message
        """
        usercommand = UserCommand()

        usercommand.cmd = cmd
        usercommand.uav_id = uav_id
        usercommand.goal = goal
        usercommand.yaw = yaw
        usercommand.is_external = is_external

        return usercommand


def main():
    rclpy.init(args=None)
    # route_path = "/home/ur10/swarming/crazyswarm2_ws/src/planner/planner_env/"
    # task_env = pickle.load(open(route_path+'env_ros.pkl', 'rb')) #WE NEED TO RESCALE THE ARRIVAL TIME AND THE START-FINISH TIME
    # task_env = TaskEnv((5, 5), tasks_range=(10, 10), traits_dim=1, max_coalition_size=2, seed=0)

    planner_ros = planner_ROS()

    rclpy.spin(planner_ros)  # Keep node alive

    planner_ros.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# point to address - agent distance covered, velocity, and the route file that is being given
#
