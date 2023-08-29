import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from yaml.loader import SafeLoader

def generate_launch_description():

    # load ip and port from crazyflies.yaml
    crazyflies_yaml = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'crazyflies.yaml')
    
    # load apriltag
    cfg_yaml = os.path.join(
        get_package_share_directory('apriltag_ros'),
        'cfg', 'tags_36h11.yaml')

    # load blacklist
    blacklist_yaml = os.path.join(
        get_package_share_directory('apriltag_ros'),
        'cfg', 'blacklist.yaml')
    
    with open(cfg_yaml, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    with open(crazyflies_yaml, 'r') as ymlfile:
        cf = yaml.safe_load(ymlfile)

    with open(blacklist_yaml, 'r') as ymlfile:
        bll = yaml.safe_load(ymlfile)

    robots_list = cf['robots']
    blacklist = bll['blacklist_ids']

    ld = LaunchDescription()

    for x in robots_list.keys():
        # do not launch camera node for these in the rejected list
        if x in blacklist:
            continue
        # load calibration
        calibration_yaml = os.path.join(
            get_package_share_directory('apriltag_ros'),
            'calibration',
            x + '.yaml')
        
        with open(calibration_yaml, 'r') as ymlfile:
            calibration = yaml.safe_load(ymlfile)
        
        calib_params = [calibration] + [cf]
        cfg_params = [cfg]
    
        camera_node = Node(
                package="apriltag_ros",
                executable="cf_opencv_publisher.py",
                name='cf_streamer_' + x,
                output="screen",
                parameters=calib_params
            )
        tag_node = Node(
                package="apriltag_ros",
                executable="apriltag_node",
                name='apriltag_node' + x,
                output="screen",
                remappings=[
                    ('image_rect', x + '/image'),
                    ('camera_info', x + '/camera_info'),
                    ('detections', x + '/tag'),
                ],
                # remappings=[
                #     ('image_rect', '/image_raw'),
                #     ('camera_info', '/camera_info'),
                # ],
                parameters=cfg_params
            )
        ld.add_action(camera_node)
        ld.add_action(tag_node)

    return ld