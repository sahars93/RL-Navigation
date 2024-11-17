import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import onnxruntime as ort
import numpy as np
import math


class JetbotNode(Node):
    def __init__(self):
        super().__init__('jetbot_node')
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                    depth=1)


        self.target_pos = np.array([3.0, 6.0, 0.0])
        self.is_goal_received = True
        self.goal_subscriber = self.create_subscription(
            PointStamped, '/clicked_point', self.goal_callback, 10
        )


        self.pose_sub = self.create_subscription(
            Odometry, '/odom', self.position_callback, 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile=qos_policy)

        self.base_cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)


        self.ort_model = ort.InferenceSession("onnx_models/Jetbot_dynamic_new.onnx")
        self.base_control = True
        self.lidar_samples = 360
        self.hidden_state = np.zeros((1,1, 64), dtype=np.float32)  # Shape: [batch_size, num_units]
        self.cell_state = np.zeros((1,1, 64), dtype=np.float32)

        # self.target_pos = np.array([1.8, -1.17, 0.0])
        # self.target_pos = np.array([0.0, 0.0, 0.0])

        self.goal_distances = None
        self.headings = None
        self.ranges = None

        self.base_position = None
        self.base_yaw = None
        self.lin_vel = None
        self.ang_vel = None


        self.create_timer(1.0 / 10.0, self.send_control)
        self.create_timer(1.0 / 10.0, self.update_base_pose)


    def goal_callback(self, msg):
        # Update target_pos when a message is received 
        self.target_pos[0] = msg.point.x
        self.target_pos[1] = msg.point.y
        # self.target_pos[0] = 1.7
        # self.target_pos[1] = 1.0
        self.is_goal_received = True
        print("target's position is set to:",[self.target_pos[0],self.target_pos[1]])
    
    def euler_from_quaternion(self,orientation):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z # in radians


    
    def update_base_pose(self):
        if self.is_goal_received:
            goal_angles = np.arctan2(self.target_pos[1] - self.base_position.y, self.target_pos[0] - self.base_position.x)

            self.headings = goal_angles - self.base_yaw 
            self.headings = np.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
            self.headings = np.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)
            x1, y1, _ = self.target_pos
            x2, y2 = self.base_position.x, self.base_position.y
            self.goal_distances = abs(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            print(self.goal_distances)

            if self.goal_distances < 0.4:
                self.is_goal_received = False
                # self.base_control = False
                print("Goal reached!", "distance:", self.goal_distances)
        # self.publish_target()
    
    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        # orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        roll, pitch, yaw = self.euler_from_quaternion(self.base_orientation)
        self.base_yaw = yaw
        self.lin_vel = msg.twist.twist.linear
        self.ang_vel = msg.twist.twist.angular


    def scan_callback(self, msg1):
        self.ranges = msg1.ranges
        print(np.shape(self.ranges)) #Stop rel world turtlebot if it's running


    def send_control(self):

        if self.is_goal_received and self.goal_distances != None :
            range = np.array(self.ranges, dtype=np.float32)
            # range = np.roll(range, int(len(self.ranges)/2))
            range1 = np.roll(range, int(len(self.ranges)/2))
            # print(range1)
            reshaped_array = range1.reshape(120, 3).min(axis=1)
            reshaped_array[reshaped_array>2] = 2
            reshaped_array[reshaped_array<0.15] = 0.15
            # print(np.min(reshaped_array))
            dist = np.array([self.goal_distances])*0.5
            angle = np.array([self.headings])
            vel_x = np.array([self.lin_vel.x])
            vel_y = np.array([self.lin_vel.y])
            vel_ang = np.array([self.ang_vel.z])
            # print( dist)

            observation = np.concatenate((reshaped_array, angle, dist,vel_x,vel_ang)).astype(np.float32)
            # print(np.shape(observation))
            observation = observation.reshape((1,-1))
            observation[observation == np.inf] = 1.5
            # print(observation)
            outputs = self.ort_model.run(None, {"obs": observation})

            # outputs = self.ort_model.run(None, {"obs": observation,"out_state.1" : self.hidden_state, "hidden_state.1" : self.cell_state},)
            # print(outputs[0])
            # self.hidden_state = outputs[3]  # Shape: [batch_size, num_units]
            # self.cell_state = outputs[4]
            mu = outputs[0]
            mu[0] = mu[0]
            # print(mu)
            # sigma = np.exp(outputs[1].squeeze())
            # action = np.random.normal(mu, sigma)

            mu[0][0] = np.clip(mu[0][0], 0.1, 0.4)
            mu[0][1] = np.clip(mu[0][1], -0.5, 0.5)
            

            # mu[0][1]=mu[0][1]

            base_action = mu[0]
            # print(base_action)
    
        # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
            twist = Twist()
            twist.linear.x = abs(base_action[0])*0.8# check the speeds, 0.2 is safe
            twist.angular.z = (base_action[1] + 0.1)

            # if self.base_control:
            self.base_cmd_vel_pub.publish(twist)
        else:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.base_cmd_vel_pub.publish(twist)
            

def main(args=None):
    rclpy.init(args=args)
    jetbot_node = JetbotNode()
    rclpy.spin(jetbot_node)
    jetbot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

