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

class JetbotNode:
    def __init__(self):


        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.position_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_scan)

        self.base_cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=20)
        self.target_pub = rospy.Publisher("/target", PointStamped, queue_size=20)

        self.ort_model = ort.InferenceSession("onnx_models/final360.onnx")
        self.base_control = True
        self.lidar_samples = 360

        # self.target_pos = np.array([-0.5, -1.8, 0.0])
        self.target_pos = np.array([0.5, 0.5, 0.0])

        self.goal_distances = None
        self.headings = None
        self.ranges = None


        self.base_position = None
        self.base_yaw = None 
        self.hidden_state = np.zeros((1,1, 64), dtype=np.float32)  # 64: num_units for lstm
        self.cell_state = np.zeros((1,1, 64), dtype=np.float32)       


        rospy.Timer(rospy.Duration(1/5.0), self.send_control)
        rospy.Timer(rospy.Duration(1/5.0), self.update_base_pose)
    
    
    def publish_target(self):
        target = PointStamped()
        target.header.frame_id = "odom"
        target.header.stamp = rospy.Time.now()
        target_point = self.target_pos 
        target.point.x = target_point[0]
        target.point.y = target_point[1]
        target.point.z = target_point[2]
        self.target_pub.publish(target)
    
    def update_base_pose(self,timer_event):

        goal_angles = np.arctan2(self.target_pos[1] - self.base_position.y, self.target_pos[0] - self.base_position.x)

        self.headings = goal_angles - self.base_yaw 
        self.headings = np.where(self.headings > math.pi, self.headings - 2 * math.pi, self.headings)
        self.headings = np.where(self.headings < -math.pi, self.headings + 2 * math.pi, self.headings)
        x1, y1, _ = self.target_pos
        x2, y2 = self.base_position.x, self.base_position.y
        self.goal_distances = abs(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

        if self.goal_distances < 0.2:
            self.base_control = False
            print("Goal reached!", "distance:", self.goal_distances)
        self.publish_target()
        


    def position_callback(self, msg):
        self.base_position = msg.pose.pose.position
        self.base_orientation = msg.pose.pose.orientation
        orientation_list = [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        self.base_yaw = yaw


    def lidar_scan(self, msg1):
        self.ranges = msg1.ranges[:self.lidar_samples]
        

    def send_control(self, timer_event):

        range = np.array(self.ranges, dtype=np.float32)
        range = np.roll(range, int(len(self.ranges)/2))
        dist = np.array([self.goal_distances])
        angle = np.array([self.headings])

        observation = np.concatenate((range, angle, dist)).astype(np.float32)
        observation = observation.reshape((1,-1))
        observation[observation == inf] = 1.5
        outputs = self.ort_model.run(None, {"obs": observation,"out_state.1" : self.hidden_state, "hidden_state.1" : self.cell_state},)
        self.hidden_state = outputs[3]  
        self.cell_state = outputs[4]
        mu = outputs[0].squeeze()
        action = np.clip(mu, -1.0, 1.0)
        base_action = action[:2]
    
        # publish base actions as twist message, base_action[0] is the linear velocity, base_action[1] is the angular velocity
        twist = Twist()
        twist.linear.x = abs(base_action[0])* 0.05 # check the speeds, 0.2 is safe
        twist.angular.z = base_action[1]* 0.15   # check the speeds, 0.1 is safe
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