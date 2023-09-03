# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import TransformStamped
# from std_msgs.msg import Header
# from tf2_ros import TransformBroadcaster

# class MarkerPublisherNode(Node):
#     def __init__(self):
#         super().__init__('marker_publisher')
#         self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
#         self.tf_broadcaster = TransformBroadcaster(self)
#         self.timer = self.create_timer(1.0, self.publish_marker_and_transform)

        

#     def publish_marker_and_transform(self):
#         # Create an STL marker
#         marker_array = MarkerArray()
#         for i in range(5):
#             stl_marker = Marker()
#             drone_id = 'cf' + str(i+1)
#             print(drone_id)
#             stl_marker.header = Header(frame_id=drone_id)
#             stl_marker.type = Marker.MESH_RESOURCE
#             stl_marker.action = Marker.ADD
#             stl_marker.pose.position.x = 0.0
#             stl_marker.pose.position.y = 0.0 + i
#             stl_marker.pose.position.z = 0.5
#             stl_marker.pose.orientation.w = 1.0
#             stl_marker.scale.x = 1.0
#             stl_marker.scale.y = 1.0
#             stl_marker.scale.z = 1.0
#             stl_marker.mesh_resource = 'package://rviz_markers/stl/StopSign.stl'  # Replace with the path to your STL file
#             stl_marker.color.r = 1.0
#             stl_marker.color.g = 1.0
#             stl_marker.color.b = 1.0
#             stl_marker.color.a = 1.0
#             stl_marker.ns = drone_id

#             # Create a transform for the base_link
#             transform_stamped = TransformStamped()
#             transform_stamped.header.stamp = self.get_clock().now().to_msg()
#             transform_stamped.header.frame_id = 'world'
#             transform_stamped.child_frame_id = 'base_link'
#             transform_stamped.transform.translation.x = 1.0
#             transform_stamped.transform.translation.y = 2.0
#             transform_stamped.transform.translation.z = 0.0  # Adjust as needed
#             transform_stamped.transform.rotation.w = 1.0
#             transform_stamped.transform.rotation.x = 0.0
#             transform_stamped.transform.rotation.y = 0.0
#             transform_stamped.transform.rotation.z = 0.0

#             # Publish the STL marker and transform
        
#             marker_array.markers.append(stl_marker)
#         self.marker_publisher.publish(marker_array)
#         # self.tf_broadcaster.sendTransform(transform_stamped)

# def main(args=None):
#     rclpy.init(args=args)
#     node = MarkerPublisherNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import math

class MovingMarkerNode(Node):
    def __init__(self):
        super().__init__('moving_marker')
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.update_marker_position)

        self.marker_id = 0
        self.marker_array = MarkerArray()

    def update_marker_position(self):
        self.marker_array.markers = []  # Clear previous markers

        # Create a moving marker
        moving_marker = Marker()
        moving_marker.header = Header(frame_id='cf1')
        moving_marker.type = Marker.MESH_RESOURCE
        moving_marker.action = Marker.ADD
        moving_marker.pose.position.x = ( 0.1)  # Update the x position based on time
        moving_marker.pose.position.y = ( 0.1)  # Update the y position based on time
        moving_marker.pose.position.z = 0.5
        moving_marker.pose.orientation.w = 1.0
        moving_marker.scale.x = 1.0
        moving_marker.scale.y = 1.0
        moving_marker.scale.z = 1.0
        moving_marker.mesh_resource = 'package://rviz_markers/stl/cone.stl'  # Replace with the path to your STL file
        moving_marker.color.r = 1.0
        moving_marker.color.g = 0.0
        moving_marker.color.b = 0.0
        moving_marker.color.a = 1.0

        self.marker_array.markers.append(moving_marker)

        # Publish the moving marker
        self.marker_publisher.publish(self.marker_array)

        # Create a transform for the base_link
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = 'world'
        transform_stamped.child_frame_id = 'cf1'
        transform_stamped.transform.translation.x = 1.0
        transform_stamped.transform.translation.y = 2.0
        transform_stamped.transform.translation.z = 0.0
        transform_stamped.transform.rotation.w = 1.0
        transform_stamped.transform.rotation.x = 0.0
        transform_stamped.transform.rotation.y = 0.0
        transform_stamped.transform.rotation.z = 0.0

        # Publish the transform
        # self.tf_broadcaster.sendTransform(transform_stamped)

        self.marker_id += 1

def main(args=None):
    rclpy.init(args=args)
    node = MovingMarkerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
