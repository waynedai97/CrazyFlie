import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    planner_env_cfg_filepath = os.path.join(
        get_package_share_directory('planner_env'),
        'config',
        'config.yaml')

    with open(planner_env_cfg_filepath, 'r') as ymlfile:
        
        planner_env_cfg = yaml.safe_load(ymlfile)
        print(planner_env_cfg)
    return LaunchDescription([
        Node(
            package='planner_env',
            executable='planner_env',
            name='planner_env',
            output='screen',
            emulate_tty=True,
            parameters=[planner_env_cfg],
        )
    ])
