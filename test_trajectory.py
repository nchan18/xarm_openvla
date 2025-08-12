from oculus_reader.reader import OculusReader
import rclpy
from rclpy.node import Node
import geometry_msgs.msg
from std_msgs.msg import Header
from tf_transformations import quaternion_from_matrix  # Import missing function
class OculusPosePublisher(Node):
    def __init__(self):
        super().__init__('oculus_pose_publisher')
        self.oculus_reader = OculusReader()
        # Create a publisher for the /goal_pose topic
        self.pose_publisher = self.create_publisher(geometry_msgs.msg.PoseStamped, '/goal_pose', 10)
        # Create a timer to publish the pose at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_pose)
    def publish_pose(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'r' not in transformations:
            return
        from reader import OculusReader
import rclpy
from rclpy.node import Node
import geometry_msgs.msg
from std_msgs.msg import Header
from tf_transformations import quaternion_from_matrix  # Import missing function
class OculusPosePublisher(Node):
    def __init__(self):
        super().__init__('oculus_pose_publisher')
        self.oculus_reader = OculusReader()
        # Create a publisher for the /goal_pose topic
        self.pose_publisher = self.create_publisher(geometry_msgs.msg.PoseStamped, '/goal_pose', 10)
        # Create a timer to publish the pose at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_pose)
    def publish_pose(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'r' not in transformations:
            return
        right_controller_pose = transformations['r']
        self.publish_goal_pose(right_controller_pose)
        self.get_logger().info(f'Transformations: {transformations}')
        self.get_logger().info(f'Buttons: {buttons}')
    def publish_goal_pose(self, transform):
        translation = transform[:3, 3]
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'world'
        # Set the position
        pose_stamped.pose.position.x = translation[0]
        pose_stamped.pose.position.y = translation[1]
        pose_stamped.pose.position.z = translation[2]
        # Set the orientation using the quaternion from the transform
        quat = quaternion_from_matrix(transform)
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        # Publish the goal pose message
        self.pose_publisher.publish(pose_stamped)
        self.get_logger().info(f'Published Goal Pose: {pose_stamped}')
def main(args=None):
    rclpy.init(args=args)
    node = OculusPosePublisher()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()
    def publish_goal_pose(self, transform):
        translation = transform[:3, 3]
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'world'
        # Set the position
        pose_stamped.pose.position.x = translation[0]
        pose_stamped.pose.position.y = translation[1]
        pose_stamped.pose.position.z = translation[2]
        # Set the orientation using the quaternion from the transform
        quat = quaternion_from_matrix(transform)
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        # Publish the goal pose message
        self.pose_publisher.publish(pose_stamped)
        self.get_logger().info(f'Published Goal Pose: {pose_stamped}')
def main(args=None):
    rclpy.init(args=args)
    node = OculusPosePublisher()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()