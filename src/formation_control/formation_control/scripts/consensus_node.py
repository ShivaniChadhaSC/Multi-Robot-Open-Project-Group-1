#!/usr/bin/env python3
"""
consensus_node.py  —  CORRECTED VERSION
----------------------------------------
Average consensus  : Freeman et al. (2006) Proportional-Integral (PI) estimator
                     with two coupled state variables  xi  and  zi.
Min/Max consensus  : Dynamic version with relaxation term (+/- alpha) so the
                     estimate can track a TIME-VARYING signal.
"""

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

        # ── Own position ────────────────────────────────────────────────────
        self.pos = np.zeros(2)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)

        # ── Other robots' positions ─────────────────────────────────────────
        self.others = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.others[r] = np.array([999.0, 999.0])
                self.create_subscription(
                    Odometry, f'/{r}/odom',
                    lambda msg, n=r: self.other_cb(msg, n), 10)

        # ── Result publishers ───────────────────────────────────────────────
        self.pub_avg = self.create_publisher(
            Float32, f'/{self.ns}/consensus/avg_dist', 10)
        self.pub_min = self.create_publisher(
            Float32, f'/{self.ns}/consensus/min_dist', 10)
        self.pub_max = self.create_publisher(
            Float32, f'/{self.ns}/consensus/max_dist', 10)
        self.pub_all = self.create_publisher(
            Float32MultiArray, f'/{self.ns}/consensus/all', 10)

        # ── PI estimator state (Freeman et al. 2006) ────────────────────────
        # xi  : estimate of the global average distance
        # zi  : integral correction variable  (coupled with xi)
        # yi_prev : measurement at previous time step (needed for zi update)
        self.xi     = 0.0
        self.zi     = 0.0
        self.yi_prev = 0.0

        # Dynamic min / max estimates
        self.min_est = 999.0   # initialise high so first real measurement wins
        self.max_est = 0.0     # initialise low  so first real measurement wins

        # ── Consensus step size ─────────────────────────────────────────────
        # With 2 neighbours and a complete graph  epsilon < 1/n_neighbours = 0.5
        self.epsilon = 0.1

        # ── Relaxation gain for dynamic min/max ─────────────────────────────
        # alpha > 0  allows estimates to "relax" toward the current signal
        # value when the signal changes.  Typical range: 0.02 – 0.10.
        self.alpha_mm = 0.05

        # ── Neighbour state storage ─────────────────────────────────────────
        # state message carries  [xi, zi, min_est, max_est]
        self.neighbor_xi      = {}
        self.neighbor_zi      = {}
        self.neighbor_min_est = {}
        self.neighbor_max_est = {}
        for r in ALL_ROBOTS:
            if r != self.ns:
                self.neighbor_xi[r]      = 0.0
                self.neighbor_zi[r]      = 0.0
                self.neighbor_min_est[r] = 999.0
                self.neighbor_max_est[r] = 0.0
                self.create_subscription(
                    Float32MultiArray,
                    f'/{r}/consensus/state',
                    lambda msg, n=r: self.consensus_state_cb(msg, n), 10)

        # ── State publisher (for neighbours to read) ────────────────────────
        self.pub_state = self.create_publisher(
            Float32MultiArray, f'/{self.ns}/consensus/state', 10)

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

    def consensus_state_cb(self, msg, name):
        # Expect [xi, zi, min_est, max_est]
        if len(msg.data) >= 4:
            self.neighbor_xi[name]      = msg.data[0]
            self.neighbor_zi[name]      = msg.data[1]
            self.neighbor_min_est[name] = msg.data[2]
            self.neighbor_max_est[name] = msg.data[3]

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _compute_distances(self):
        dists = []
        for opos in self.others.values():
            d = float(np.linalg.norm(self.pos - opos))
            if d < 900.0:
                dists.append(d)
        return dists

    # ────────────────────────────────────────────────────────────────────────
    # Main consensus loop
    # ────────────────────────────────────────────────────────────────────────

    def consensus_loop(self):
        dists = self._compute_distances()
        if not dists:
            return

        # Local measurement  yi  = mean of distances to all other robots
        yi = float(np.mean(dists))

        n_neighbors = len(self.neighbor_xi)

        # ── 1. Correct PI estimator (Freeman et al. 2006) ───────────────────
        #
        # Two coupled discrete-time update equations:
        #
        #   xi(t+1) = xi(t)  +  ε · Σ_{j∈Ni} (xj(t) − xi(t))  +  zi(t)
        #
        #   zi(t+1) = zi(t)  +  ε · Σ_{j∈Ni} (zj(t) − zi(t))
        #                     +  (yi(t) − yi(t−1))
        #
        # xi  tracks the running average estimate.
        # zi  is the integral (PI) correction that injects the
        #     time-derivative of the local measurement so the estimate
        #     can follow a time-varying average.

        if n_neighbors > 0:
            sum_xi = sum(self.neighbor_xi.values())
            sum_zi = sum(self.neighbor_zi.values())

            # Update xi first (uses current zi before zi is updated)
            new_xi = (self.xi
                      + self.epsilon * (sum_xi - n_neighbors * self.xi)
                      + self.zi)

            # Update zi
            new_zi = (self.zi
                      + self.epsilon * (sum_zi - n_neighbors * self.zi)
                      + (yi - self.yi_prev))

            self.xi = new_xi
            self.zi = new_zi
        else:
            # No neighbours yet — initialise estimate from own measurement
            self.xi = yi

        # Save yi for next iteration (needed by zi update)
        self.yi_prev = yi

        # ── 2. Dynamic min consensus ─────────────────────────────────────────
        #
        #   mi(t+1) = min( yi(t),  min_{j∈Ni} { mj(t) + α } )
        #
        # The  +α  relaxation term allows the estimate to RISE when the
        # true minimum increases, so it can track a time-varying signal.
        # Without it (static version) the estimate can never increase once
        # it has converged to a low value.

        neigh_min_vals = [v for v in self.neighbor_min_est.values() if v < 900.0]
        if neigh_min_vals:
            self.min_est = float(min(
                [yi] + [v + self.alpha_mm for v in neigh_min_vals]
            ))
        else:
            self.min_est = yi

        # ── 3. Dynamic max consensus ─────────────────────────────────────────
        #
        #   Mi(t+1) = max( yi(t),  max_{j∈Ni} { Mj(t) − α } )
        #
        # The  −α  relaxation term allows the estimate to FALL when the
        # true maximum decreases, enabling time-varying tracking.

        neigh_max_vals = [v for v in self.neighbor_max_est.values() if v > 0.0]
        if neigh_max_vals:
            self.max_est = float(max(
                [yi] + [v - self.alpha_mm for v in neigh_max_vals]
            ))
        else:
            self.max_est = yi

        # ── Publish state for neighbours  [xi, zi, min_est, max_est] ────────
        state_msg = Float32MultiArray()
        state_msg.data = [float(self.xi), float(self.zi),
                          float(self.min_est), float(self.max_est)]
        self.pub_state.publish(state_msg)

        # ── Publish individual results ───────────────────────────────────────
        avg_msg = Float32()
        avg_msg.data = float(self.xi)
        self.pub_avg.publish(avg_msg)

        min_msg = Float32()
        min_msg.data = float(self.min_est)
        self.pub_min.publish(min_msg)

        max_msg = Float32()
        max_msg.data = float(self.max_est)
        self.pub_max.publish(max_msg)

        all_msg = Float32MultiArray()
        all_msg.data = [float(self.xi), float(self.min_est), float(self.max_est)]
        self.pub_all.publish(all_msg)

    # ────────────────────────────────────────────────────────────────────────

    def log_loop(self):
        dists = self._compute_distances()
        if not dists:
            return
        true_avg = float(np.mean(dists))
        true_min = float(np.min(dists))
        true_max = float(np.max(dists))
        self.get_logger().info(
            f'[{self.ns}] CONSENSUS → '
            f'avg_est={self.xi:.3f} (true={true_avg:.3f})  '
            f'min_est={self.min_est:.3f} (true={true_min:.3f})  '
            f'max_est={self.max_est:.3f} (true={true_max:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ConsensusNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()