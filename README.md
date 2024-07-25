# CrazyFlie

## Steps to launch ros_bridge
- sudo apt remove ros-humble-controller-manager-msgs
- sudo apt purge ros-humble-controller-manager-msgs
- Follow these instructions for [installation](https://docs.ros.org/en/humble/How-To-Guides/Using-ros1_bridge-Jammy-upstream.html#build-ros1-bridge)
- source ${ROS1_INSTALL_PATH}/setup.bash
- source ${ROS2_INSTALL_PATH}/setup.bash
- ros2 run ros1_bridge dynamic_bridge

 ## Setting configs for scaling the env
 - planner/config/config.yaml has the following parameters that can be tuned to sync ugv and uav performance. Typically increase the agent arena velocity and decrease the working time bias.
 - ![image](https://github.com/user-attachments/assets/3d0a8565-3ab6-4639-9aef-36aa461bf5ac)
