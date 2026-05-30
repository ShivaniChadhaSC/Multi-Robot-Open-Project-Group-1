#!/usr/bin/env python3
"""
minmax_consensus_node.py  —  CORRECTED VERSION
------------------------------------------------
Implements the DYNAMIC min/max consensus algorithm so that the estimate
can track a TIME-VARYING signal (inter-robot distances change as robots move).

Static version  (WRONG for time-varying signals):
    mi(t+1) = min( yi(t),  min_{j∈Ni} mj(t) )
    Mi(t+1) = max( yi(t),  max_{j∈Ni} Mj(t) )

Dynamic version  (CORRECT — used here):
    mi(t+1) = min( yi(t),  min_{j∈Ni} { mj(t) + α } )
    Mi(t+1) = max( yi(t),  max_{j∈Ni} { Mj(t) − α } )

The relaxation term  +α / −α  allows the global estimate to rise/fall
when the true signal changes, instead of staying stuck at an old extreme.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np

ALL_ROBOTS = ['tb3_0', 'tb3_1', 'tb3_2']


class MinMaxConsensusNode(Node):
    def __init__(self):
        super().__init__('minmax_consensus_node')
        self.ns = self.get_namespace().strip('/')
        self.get_logger().info(
            f'Dynamic MinMax Consensus Node started for [{self.ns}]')

        # ── Helper ───────────────────────────────────────────────────────────
        def make_topic(*parts):
            non_empty = [p for p in parts if p]
            topic = '/'.join(non_empty)
            return topic if topic.startswith('/') else '/' + topic

        # ── Parameters ───────────────────────────────────────────────────────
        # alpha: relaxation gain.
        #   Too small → slow to track changes in the signal.
        #   Too large → estimate overshoots / oscillates.
        #   Typical value: slightly larger than the maximum rate of change
        #   of the measured signal per time step.
        self.alpha = 0.05

        # ── Own position ─────────────────────────────────────────────────────
        self.pos = np.zeros(2)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)

        # ── Other robots' positions ──────────────────────────────────────────
        self.others = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.others[r] = np.array([999.0, 999.0])
                self.create_subscription(
                    Odometry, make_topic(r, 'odom'),
                    lambda msg, n=r: self.other_cb(msg, n), 10)

        # ── Own estimates ────────────────────────────────────────────────────
        # Initialise min high and max low so the first real measurement wins
        self.min_est = 999.0
        self.max_est = 0.0

        # ── Neighbours' latest estimates ─────────────────────────────────────
        # State message carries  [min_est, max_est]
        self.neighbor_min = {}
        self.neighbor_max = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.neighbor_min[r] = 999.0
                self.neighbor_max[r] = 0.0
                self.create_subscription(
                    Float32MultiArray, make_topic(r, 'consensus', 'state'),
                    lambda msg, n=r: self.neighbor_cb(msg, n), 10)

        # ── Publishers ───────────────────────────────────────────────────────
        base = make_topic(self.ns, 'consensus') if self.ns else '/consensus'
        self.pub_min      = self.create_publisher(
            Float32, make_topic(base, 'min_dist'), 10)
        self.pub_max      = self.create_publisher(
            Float32, make_topic(base, 'max_dist'), 10)
        self.pub_state    = self.create_publisher(
            Float32MultiArray, make_topic(base, 'state'), 10)

        self.get_logger().info(f'Publishing state to {make_topic(base, "state")}')

        self.create_timer(0.2, self.consensus_loop)
        self.create_timer(0.5, self.log_loop)

    # ────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x,
                             msg.pose.pose.position.y])

    def other_cb(self, msg, name):
        self.others[name] = np.array([msg.pose.pose.position.x,
                                      msg.pose.pose.position.y])

    def neighbor_cb(self, msg, name):
        # State message carries  [min_est, max_est]
        if len(msg.data) >= 2:
            self.neighbor_min[name] = msg.data[0]
            self.neighbor_max[name] = msg.data[1]

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _local_measurement(self):
        """yi = mean distance to other robots (time-varying signal)."""
        dists = []
        for opos in self.others.values():
            d = float(np.linalg.norm(self.pos - opos))
            if d < 900.0:
                dists.append(d)
        return float(np.mean(dists)) if dists else None

    # ────────────────────────────────────────────────────────────────────────
    # Main consensus loop
    # ────────────────────────────────────────────────────────────────────────

    def consensus_loop(self):
        yi = self._local_measurement()
        if yi is None:
            return

        # ── Dynamic MIN consensus ─────────────────────────────────────────
        #
        #   mi(t+1) = min( yi(t),  min_{j∈Ni} { mj(t) + α } )
        #
        # Each neighbour's estimate is raised by α before taking the min.
        # This "leaks" the estimate upward over time, allowing it to follow
        # an increasing true minimum rather than staying stuck at an old low.

        neigh_min_vals = [v for v in self.neighbor_min.values() if v < 900.0]
        if neigh_min_vals:
            self.min_est = float(min(
                [yi] + [v + self.alpha for v in neigh_min_vals]
            ))
        else:
            # No neighbours heard yet — use own measurement
            self.min_est = yi

        # ── Dynamic MAX consensus ─────────────────────────────────────────
        #
        #   Mi(t+1) = max( yi(t),  max_{j∈Ni} { Mj(t) − α } )
        #
        # Each neighbour's estimate is lowered by α before taking the max.
        # This "leaks" the estimate downward over time, allowing it to follow
        # a decreasing true maximum rather than staying stuck at an old high.

        neigh_max_vals = [v for v in self.neighbor_max.values() if v > 0.0]
        if neigh_max_vals:
            self.max_est = float(max(
                [yi] + [v - self.alpha for v in neigh_max_vals]
            ))
        else:
            self.max_est = yi

        # ── Publish state  [min_est, max_est] ─────────────────────────────
        state_msg = Float32MultiArray()
        state_msg.data = [float(self.min_est), float(self.max_est)]
        self.pub_state.publish(state_msg)

        # ── Publish individual float results ──────────────────────────────
        min_msg = Float32()
        min_msg.data = float(self.min_est)
        self.pub_min.publish(min_msg)

        max_msg = Float32()
        max_msg.data = float(self.max_est)
        self.pub_max.publish(max_msg)

    # ────────────────────────────────────────────────────────────────────────

    def log_loop(self):
        yi = self._local_measurement()
        if yi is None:
            return
        self.get_logger().info(
            f'[{self.ns}]  '
            f'min_est={self.min_est:.3f}  '
            f'max_est={self.max_est:.3f}  '
            f'(local_yi={yi:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MinMaxConsensusNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()