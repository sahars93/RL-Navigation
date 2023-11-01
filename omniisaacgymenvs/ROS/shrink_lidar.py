#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan

class ScanShrinker:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('scan_shrinker')

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/scan')
        self.output_topic = rospy.get_param('~output_topic', '/scan_local')
        self.shrink_by = rospy.get_param('~shrink_by', 0.28)
        self.max_range = rospy.get_param('~max_range', 1.5)

        # Subscriber and Publisher
        self.subscriber = rospy.Subscriber(self.input_topic, LaserScan, self.callback)
        self.publisher = rospy.Publisher(self.output_topic, LaserScan, queue_size=10)

    def callback(self, data):
        # Shrink the scan ranges
        modified_scan = data
        modified_scan.ranges = [
            max(0.0, min(r - self.shrink_by, self.max_range)) if r >= self.shrink_by else r
            for r in data.ranges
        ]

        # Publish the modified scan
        self.publisher.publish(modified_scan)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        shrinker = ScanShrinker()
        shrinker.run()
    except rospy.ROSInterruptException:
        pass
