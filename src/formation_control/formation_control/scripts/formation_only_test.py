#!/usr/bin/env python3
"""
File: formation_only_test.py
Standalone formation control test — NO CBF, NO obstacles, NO consensus.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np

FORMATION_OFFSETS = {
    'tb3_1': np.array([-0.55,  0.32]),
    'tb3_2': np.array([-0.55, -0.32]),
}

class FormationOnly(Node):
    def __init__(self):
        super().__init__('formation_only')
        self.ns = self.get_namespace().strip('/')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/tb3_0/odom', self.leader_cb, 10)

        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.leader_pos = np.zeros(2)
        self.leader_heading = 0.0
        self.leader_hist = []

        self.k_form = 0.8
        self.v_max = 0.18
        self.w_max = 1.3
        self.ell = 0.12

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'Formation-only test started for [{self.ns}]')

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

    def leader_cb(self, msg):
        new = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if self.leader_hist:
            last = self.leader_hist[-1]
            if np.linalg.norm(new - last) > 0.03:
                raw = math.atan2(new[1] - last[1], new[0] - last[0])
                self.leader_heading = 0.85 * self.leader_heading + 0.15 * raw
        self.leader_pos = new
        self.leader_hist.append(new.copy())
        if len(self.leader_hist) > 5:
            self.leader_hist.pop(0)

    def control_loop(self):
        if np.all(self.leader_pos == 0):
            return

        offset = FORMATION_OFFSETS[self.ns]
        c = math.cos(self.leader_heading)
        s = math.sin(self.leader_heading)
        target = self.leader_pos + np.array([c*offset[0] - s*offset[1],
                                              s*offset[0] + c*offset[1]])

        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < 0.08:
            self._publish(0.0, 0.0)
            return

        u_des = self.k_form * diff / dist
        v = u_des[0] * math.cos(self.yaw) + u_des[1] * math.sin(self.yaw)
        w = (-u_des[0] * math.sin(self.yaw) + u_des[1] * math.cos(self.yaw)) / self.ell
        self._publish(v, w)

    def _publish(self, v, w):
        t = Twist()
        t.linear.x = float(np.clip(v, -self.v_max, self.v_max))
        t.angular.z = float(np.clip(w, -self.w_max, self.w_max))
        self.cmd_pub.publish(t)

def main():
    rclpy.init()
    rclpy.spin(FormationOnly())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
