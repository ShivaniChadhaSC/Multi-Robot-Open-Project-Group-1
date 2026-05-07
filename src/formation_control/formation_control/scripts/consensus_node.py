#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np

ALL_ROBOTS = ['tb3_0', 'tb3_1', 'tb3_2']

class ConsensusNode(Node):
    def __init__(self):
        super().__init__('consensus_node')
        self.ns = self.get_namespace().strip('/')
        self.get_logger().info(f'Consensus Node started for [{self.ns}]')

        # Own position
        self.pos = np.zeros(2)
        self.create_subscription(
            Odometry, 'odom', self.odom_cb, 10)

        # Other robots positions
        self.others = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.others[r] = np.array([999.0, 999.0])
                self.create_subscription(
                    Odometry, f'/{r}/odom',
                    lambda msg, n=r: self.other_cb(msg, n), 10)

        # Publish consensus results
        self.pub_avg = self.create_publisher(
            Float32, f'/{self.ns}/consensus/avg_dist', 10)
        self.pub_min = self.create_publisher(
            Float32, f'/{self.ns}/consensus/min_dist', 10)
        self.pub_max = self.create_publisher(
            Float32, f'/{self.ns}/consensus/max_dist', 10)
        self.pub_all = self.create_publisher(
            Float32MultiArray, f'/{self.ns}/consensus/all', 10)

        # Dynamic average consensus state
        # xi = local estimate, zi = auxiliary variable
        self.xi = 0.0   # estimate of average distance
        self.zi = 0.0   # auxiliary variable

        # Subscribe to neighbors' consensus states
        self.neighbor_xi = {}
        self.neighbor_zi = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.neighbor_xi[r] = 0.0
                self.neighbor_zi[r] = 0.0
                self.create_subscription(
                    Float32MultiArray,
                    f'/{r}/consensus/state',
                    lambda msg, n=r: self.consensus_state_cb(msg, n), 10)

        # Publish own consensus state
        self.pub_state = self.create_publisher(
            Float32MultiArray, f'/{self.ns}/consensus/state', 10)

        # Consensus gain
        self.epsilon = 0.1

        self.create_timer(0.2, self.consensus_loop)
        self.create_timer(0.5, self.log_loop)

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x,
                             msg.pose.pose.position.y])

    def other_cb(self, msg, name):
        self.others[name] = np.array([msg.pose.pose.position.x,
                                      msg.pose.pose.position.y])

    def consensus_state_cb(self, msg, name):
        if len(msg.data) >= 2:
            self.neighbor_xi[name] = msg.data[0]
            self.neighbor_zi[name] = msg.data[1]

    def _compute_distances(self):
        dists = []
        for opos in self.others.values():
            d = np.linalg.norm(self.pos - opos)
            if d < 900:
                dists.append(d)
        return dists

    def consensus_loop(self):
        dists = self._compute_distances()
        if not dists:
            return

        # Local measurement = average of my distances to others
        yi = float(np.mean(dists))

        # Dynamic average consensus update
        # xi(t+1) = xi(t) + epsilon * sum(xj - xi) + (yi - yi_prev)
        sum_xi = sum(self.neighbor_xi.values())
        n_neighbors = len(self.neighbor_xi)

        if n_neighbors > 0:
            self.xi = self.xi + self.epsilon * (sum_xi - n_neighbors * self.xi)
            self.xi = 0.7 * self.xi + 0.3 * yi  # blend with local measurement

        # Min consensus: take minimum
        all_xi = list(self.neighbor_xi.values()) + [yi]
        min_dist = float(np.min(dists + [v for v in self.neighbor_xi.values()
                                          if v < 900]))
        max_dist = float(np.max(dists + [v for v in self.neighbor_xi.values()
                                          if v < 900]))

        # Publish state for neighbors
        state_msg = Float32MultiArray()
        state_msg.data = [float(self.xi), float(self.zi), yi]
        self.pub_state.publish(state_msg)

        # Publish results
        avg_msg = Float32()
        avg_msg.data = float(self.xi)
        self.pub_avg.publish(avg_msg)

        min_msg = Float32()
        min_msg.data = min_dist
        self.pub_min.publish(min_msg)

        max_msg = Float32()
        max_msg.data = max_dist
        self.pub_max.publish(max_msg)

        all_msg = Float32MultiArray()
        all_msg.data = [float(self.xi), min_dist, max_dist]
        self.pub_all.publish(all_msg)

    def log_loop(self):
        dists = self._compute_distances()
        if not dists:
            return
        self.get_logger().info(
            f'[{self.ns}] CONSENSUS → '
            f'avg={self.xi:.2f}m  '
            f'min={min(dists):.2f}m  '
            f'max={max(dists):.2f}m'
        )

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ConsensusNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
