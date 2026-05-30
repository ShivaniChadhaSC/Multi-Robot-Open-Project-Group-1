#!/usr/bin/env python3
"""
Standalone CBF test — one robot avoids obstacles, no formation.
Run with: ros2 run formation_control cbf_only_test.py --ros-args -r __ns:=/tb3_0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from scipy.optimize import minimize

class CBFOnlyTest(Node):
    def __init__(self):
        super().__init__('cbf_only_test')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)

        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.laser_ranges = []
        self.angle_min = 0.0
        self.angle_inc = 0.0

        self.d_safe = 0.4
        self.d_activate = 1.0
        self.alpha = 4.0
        self.v_max = 0.15
        self.w_max = 1.5
        self.ell = 0.12

        self.goal = np.array([3.0, 0.0])  # drive straight toward a wall
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

    def scan_cb(self, msg):
        self.laser_ranges = list(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_inc = msg.angle_increment

    def control_loop(self):
        if not self.laser_ranges:
            return

        # Desired: go straight toward goal (no formation, no consensus)
        dg = self.goal - self.pos
        if np.linalg.norm(dg) > 0.3:
            u_des = 0.5 * dg / np.linalg.norm(dg)
        else:
            u_des = np.zeros(2)

        # CBF constraints
        cons = []
        n = len(self.laser_ranges)
        step = max(1, n // 20)
        for i in range(0, n, step):
            r = self.laser_ranges[i]
            if r < 0.05 or r > self.d_activate or not math.isfinite(r):
                continue
            ang = self.angle_min + i * self.angle_inc
            obs = self.pos + r * np.array([math.cos(self.yaw+ang), math.sin(self.yaw+ang)])
            diff = self.pos - obs
            cons.append((-2*diff, self.alpha * (r**2 - self.d_safe**2)))

        # QP
        if cons:
            scipy_cons = [{'type': 'ineq', 'fun': lambda u, a=a, b=b: float(b - np.dot(a, u))}
                          for a, b in cons]
            res = minimize(lambda u: np.dot(u-u_des, u-u_des), u_des.copy(),
                           method='SLSQP', constraints=scipy_cons)
            u_safe = res.x if res.success else u_des
        else:
            u_safe = u_des

        v = u_safe[0]*math.cos(self.yaw) + u_safe[1]*math.sin(self.yaw)
        w = (-u_safe[0]*math.sin(self.yaw) + u_safe[1]*math.cos(self.yaw)) / self.ell
        v = np.clip(v, -self.v_max, self.v_max)
        w = np.clip(w, -self.w_max, self.w_max)

        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

def main():
    rclpy.init()
    rclpy.spin(CBFOnlyTest())
    rclpy.shutdown()

if __name__ == '__main__':
    main()