import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import onnxruntime as ort
import numpy as np
from sensor_msgs.msg import LaserScan
import math
from numpy import inf
from geometry_msgs.msg import PoseStamped

class JetbotNode:
    def __init__(self):


        # self.pose_sub = rospy.Subscriber('vrpn_client_node/turtle06/pose', PoseStamped, self.position_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_scan)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.base_cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
        # self.target_pub = rospy.Publisher("/target", PointStamped, queue_size=20)

        self.ort_model = ort.InferenceSession("onnx_models/static.onnx")
        self.base_control = True
        self.lidar_samples = 20

        # self.target_pos = np.array([-0.5, -1.8, 0.0])
        self.target_pos = np.array([3.5, 6.0, 0.0])

        self.goal_distances = 100
        self.headings = None
        self.ranges = None
        self.lin_vel = None
        self.ang_vel = None
        self.lstm = True
        self.lstm_hidden_size = 64
        self.out_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)
        self.hidden_state = np.zeros((1, 1, self.lstm_hidden_size)).astype(np.float32)


        self.base_position_x = None
        self.base_position_y = None
        self.base_position_z = None

        self.base_yaw = None        


        rospy.Timer(rospy.Duration(1/5.0), self.send_control)
        rospy.Timer(rospy.Duration(1/5.0), self.update_base_pose)
  
    
    def update_base_pose(self,timer_event):

        goal_angles = np.arctan2(self.target_pos[1] - self.base_position_y, self.target_pos[0] - self.base_position_x)
        # print(goal_angles)
        self.headings = goal_angles - self.base_yaw 
        self.headings = np.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = np.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)
        x1, y1, _ = self.target_pos
        x2, y2 = self.base_position_x, self.base_position_y
        self.goal_distances = abs(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        # print(self.goal_distances)

        if self.goal_distances < 0.22:
            self.base_control = False
            print("Goal reached!", "distance:", self.goal_distances)
        # self.publish_target()
        


    # def position_callback(self, msg):
    #     self.base_position_x = msg.pose.position.x
    #     self.base_position_y = msg.pose.position.y
    #     self.base_orientation = msg.pose.orientation
    #     orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
    #     roll, pitch, yaw = euler_from_quaternion(orientation_list)
    #     self.base_yaw = yaw
    #     # print(self.base_position_x,self.base_position_y)
    #     # print(f"yaw: {yaw}")


    def lidar_scan(self, msg1):
        self.ranges = msg1.ranges
        

    def odom_callback(self, msg):
        if not hasattr(self, 'initial_position'):
            self.initial_position = msg.pose.pose.position
            self.initial_orientation = msg.pose.pose.orientation
            # Convert initial orientation to Euler angles to get initial yaw
            initial_orientation_list = [self.initial_orientation.x, self.initial_orientation.y, self.initial_orientation.z, self.initial_orientation.w]
            _, _, self.initial_yaw = euler_from_quaternion(initial_orientation_list)

        # Subtract initial position from current position
        self.base_position_x = msg.pose.pose.position.x 
        self.base_position_y = msg.pose.pose.position.y 
        self.base_position_z = msg.pose.pose.position.z 

        # Subtract initial orientation from current orientation
        current_orientation_list = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, current_yaw = euler_from_quaternion(current_orientation_list)
        self.base_yaw = current_yaw 

        # Update linear and angular velocities
        self.lin_vel = msg.twist.twist.linear
        self.ang_vel = msg.twist.twist.angular
        # print("----------------------------------------------------------")
        # print(self.base_position_x,self.base_position_y,self.base_yaw)



    def send_control(self, timer_event):

        range = np.array(self.ranges, dtype=np.float32)
        # range = np.roll(range, int(len(self.ranges)/2))
        range1 = np.roll(range, int(len(self.ranges)/2))
        
        reshaped_array = range1.reshape(120, 3).min(axis=1)
        # reshaped_array = np.flip(reshaped_array)

        reshaped_array[reshaped_array>2] = 2
        # reshaped_array[reshaped_array<0.25] = 0.15
        # print(reshaped_array)
        dist = np.array([self.goal_distances])
        angle = np.array([self.headings])
        vel_x = np.array([self.lin_vel.x])
        vel_y = np.array([self.lin_vel.y])
        vel_ang = np.array([self.ang_vel.z])
        print(dist,angle)

        observation = np.concatenate((reshaped_array, angle, dist,vel_x,vel_ang)).astype(np.float32)
        # print(np.shape(observation))
        observation = observation.reshape((1,-1))
        observation[observation == np.inf] = 2
        # print(observation)
        if self.lstm:
            outputs = self.ort_model.run(None, {"obs": observation, "out_state.1" : self.out_state, "hidden_state.1" : self.hidden_state})
            self.out_state = outputs[3]
            self.hidden_state = outputs[4]
        else:
            outputs = self.ort_model.run(None, {"obs": observation})
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
        if self.base_control:
            self.base_cmd_vel_pub.publish(twist)
        else:
            twist.linear.x = 0 
            twist.angular.z = 0
            self.base_cmd_vel_pub.publish(twist)

if __name__ == '__main__':
    rospy.init_node('rl_node', anonymous=True)
    JetbotNode()
    rospy.spin()