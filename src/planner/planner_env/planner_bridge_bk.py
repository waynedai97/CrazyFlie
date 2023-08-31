import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np


class planner_ROS(Node):
    def __init__(self, policy_net, k_size, device="cpu", greedy=False, save_image=False):
        super().__init__(
            "planner_ROS",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self._ros_parameters = self._param_to_dict(self._parameters)
        self.cfs = dict()
        robot_data = self._ros_parameters["robots"]

        # Start timer for mission
        self.t1 = time()

        names = []
        initial_states = []
        for cfname in robot_data:
            if robot_data[cfname]["enabled"]:
                type_cf = robot_data[cfname]["type"]
                # do not include virtual objects
                connection = self._ros_parameters["robot_types"][type_cf].get(
                    "connection", "crazyflie"
                )
                if connection == "crazyflie":
                    names.append(cfname)
                    pos = robot_data[cfname]["initial_position"]
                    initial_states.append(pos)

        for name, initial_state in zip(names, initial_states):
            self.cfs[name] = (name, initial_state, self.backend.time)

        self.waypoint = dict()

        self.create_subscription(
            AgentsStateFeedback, "/agents", self.getStatusCallback, 10
        )

        for name, _ in self.cfs.items():
            self.create_subscription(
                PoseStamped,
                name + "/pose",
                partial(self.getLocalCallback, name=name),
                10,
            )

        self.create_publisher(  # Declare publisher (type,topic name, queue size)
            UserCommand, "/user/external", 10  # .conmd
        )

        self.waypoint[name] = ()

        timer_period = 0.5  # Time period for callback (seconds)
        self.timer = self.create_timer(timer_period, self.waypoint_pub)
        self.is_shutdown = False

        self.n_agent = len(self.cfs.item())
        # Check correct env is pulled
        self.env = Env(
            env_index=self.global_step,
            k_size=self.k_size,
            train=False,
            num_heading=self.num_heading,
            plot=save_image,
            adaptive_generation=self.adaptive_generation,
            sensor_range=self.sensor_range,
            speed=self.speed,
            resolution=self.resolution,
        )
        self.local_policy_net = policy_net

        self.robot_list = []
        self.all_robot_positions = dict()
        self.perf_metrics = dict()

        # Pull correct robot positions
        # for i, name in enumerate(self.cfs.item()):
        #    robot_position_full = self.cfs[name][1];                                                                     # Pull robot position from ROS params
        #    robot_position = [(5-robot_position_full[0])*640/5, (5-robot_position_full[1])*480/5]
        #    robot = Robot(robot_id = i, cf_id = name, position=robot_position, plot=save_image)                          # Create robot object
        #    self.robot_list.append(robot)
        #    self.all_robot_positions[i] = robot_position

    def getLocalCallback(self, pos, name="all"):
        loc = np.array([0.0, 0.0, 0.0])
        loc[0] = pos.pose.pose.position.x
        loc[1] = pos.pose.pose.position.y
        loc[1] = pos.pose.pose.position.z
        self.real_pos[name] = loc

    def getStatusCallback(self, states):
        for agent in states.agents:
            if agent.flight_state != (
                AgentsState.MOVE
                or AgentsState.IDLE
                or AgentsState.TAKEOFF
                or AgentsState.HOVER
            ):
                try:
                    self.cfs.remove(agent.id)  # cf1
                    for robot in self.robot_list:
                        if robot.cf_id == agent.id:
                            self.robot_list.remove(robot)
                except:
                    pass

            # Check if mission has exceeded 4 minutes
            t2 = time()
            if (t2 - self.t1) > 60 * 4:
                try:
                    self.cfs.remove(agent.id)
                    for robot in self.robot_list:
                        if robot.cf_id == agent.id:
                            self.robot_list.remove(robot)
                except:
                    pass

            # Reassign robot IDs
            for i, robot in enumerate(self.robot_list):
                robot.robot_id = i

    def waypoint_pub(self):

        for name, _ in self.cfs.items():
            if (
                self.waypoint[name] is not None and self.real_pos[name] is not None
            ):  # only publish if waypoint is set and position is known

                if np.linalg.norm(self.waypoint[name] - self.real_pos[name]) <= 0.5:
                    waypoint = UserCommand()

                    waypoint.cmd = "goto_velocity"
                    waypoint.uav_id.append(name)  # cf1
                    waypoint.goal.x = (
                        (640 - self.waypoint[name][0][0]) * 5 / 640
                    )  # change the distance
                    waypoint.goal.y = (
                        (480 - self.waypoint[name][0][1]) * 5 / 480
                    )  # change the distance
                    waypoint.goal.z = 2
                    yaw = self.waypoint[name][1] * np.pi / 2 - np.pi / 4
                    waypoint.goal.yaw = yaw

                    self.publisher.publish(waypoint)

                    self.get_logger().info(
                        "Publishing waypoint: {},{},{} with yaw {} for crazyflie {}".format(
                            waypoint.goal.x,
                            waypoint.goal.y,
                            waypoint.goal.z,
                            waypoint.goal.yaw,
                            name,
                        )
                    )
                    self.waypoint[name] = ()
                    continue

        for name, _ in self.cfs.items():  # Need a flag to send next waypoint
            if self.waypoint[name] is None:
                self.get_logger().info("Calculating next waypoint")
                (
                    next_position_list,
                    done,
                ) = self.get_next_position()  # To replace with fake setpoints
                for i, position in enumerate(self.all_robot_positions):
                    name = self.robot_list[i].cf_id
                    for node in self.env.map.nodes_list:
                        if (node.coords == position).all():
                            heading = node.current_heading
                    self.waypoint[name] = [position, heading]

        if done:
            rclpy.shutdown()


def main():
    rclpy.init(args=None)
    policy_net = None
    K_SIZE = None
    device = None

    planner_ros = planner_ROS(
        policy_net=policy_net, k_size=K_SIZE, device=device, greedy=True
    )

    rclpy.spin(planner_ros)  # Keep node alive

    planner_ros.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
