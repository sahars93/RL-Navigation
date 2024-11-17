import rclpy
from geometry_msgs.msg import Twist
import time

def publish_cmd_vel():
    rclpy.init()

    node = rclpy.create_node('cmd_vel_publisher')

    cmd_vel_publisher = node.create_publisher(Twist, 'robot1/cmd_vel', 10)
    cmd_vel_publisher2 = node.create_publisher(Twist, 'robot2/cmd_vel', 10)

    twist = Twist()
    twist.linear.x = 0.24

    duration = 20  # seconds

    try:
        while rclpy.ok():
            node.get_logger().info('Publishing linear velocity: %f' % twist.linear.x)
            cmd_vel_publisher.publish(twist)
            cmd_vel_publisher2.publish(twist)
            time.sleep(duration)

            twist.linear.x = -twist.linear.x  # Reverse the linear velocity
            node.get_logger().info('Reversing linear velocity')

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    publish_cmd_vel()