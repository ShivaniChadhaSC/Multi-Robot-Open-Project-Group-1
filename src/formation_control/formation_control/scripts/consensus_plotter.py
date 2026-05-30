#!/usr/bin/env python3
"""
consensus_plotter.py
---------------------
Real-time live plot of all three consensus estimates (avg, min, max)
alongside the true values computed from odometry.

Run with:
  python3 consensus_plotter.py
(No ROS2 run needed — just run directly while your other nodes are running)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import math

WINDOW = 200        # number of time steps to show on screen
ALL_ROBOTS = ['tb3_0', 'tb3_1', 'tb3_2']
COLORS = ['#e74c3c', '#3498db', '#2ecc71']   # red, blue, green per robot


class ConsensusPlotter(Node):
    def __init__(self):
        super().__init__('consensus_plotter')

        # ── Storage: deques for rolling window ──────────────────────────────
        self.avg_est  = {r: deque([0.0]*WINDOW, maxlen=WINDOW) for r in ALL_ROBOTS}
        self.min_est  = {r: deque([0.0]*WINDOW, maxlen=WINDOW) for r in ALL_ROBOTS}
        self.max_est  = {r: deque([0.0]*WINDOW, maxlen=WINDOW) for r in ALL_ROBOTS}

        self.true_avg = deque([0.0]*WINDOW, maxlen=WINDOW)
        self.true_min = deque([0.0]*WINDOW, maxlen=WINDOW)
        self.true_max = deque([0.0]*WINDOW, maxlen=WINDOW)

        # Robot positions for computing true values
        self.positions = {r: np.array([0.0, 0.0]) for r in ALL_ROBOTS}

        # ── Subscribe to consensus estimates ─────────────────────────────────
        for r in ALL_ROBOTS:
            self.create_subscription(
                Float32, f'/{r}/consensus/avg_dist',
                lambda msg, n=r: self.avg_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{r}/consensus/min_dist',
                lambda msg, n=r: self.min_cb(msg, n), 10)
            self.create_subscription(
                Float32, f'/{r}/consensus/max_dist',
                lambda msg, n=r: self.max_cb(msg, n), 10)
            self.create_subscription(
                Odometry, f'/{r}/odom',
                lambda msg, n=r: self.odom_cb(msg, n), 10)

        self.create_timer(0.2, self.update_true_values)

    def avg_cb(self, msg, name):
        self.avg_est[name].append(msg.data)

    def min_cb(self, msg, name):
        self.min_est[name].append(msg.data)

    def max_cb(self, msg, name):
        self.max_est[name].append(msg.data)

    def odom_cb(self, msg, name):
        self.positions[name] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y])

    def update_true_values(self):
        """Compute true avg/min/max from actual odometry positions."""
        dists = []
        robots = list(ALL_ROBOTS)
        for i in range(len(robots)):
            for j in range(i+1, len(robots)):
                d = float(np.linalg.norm(
                    self.positions[robots[i]] - self.positions[robots[j]]))
                if d < 900:
                    dists.append(d)
        if dists:
            self.true_avg.append(float(np.mean(dists)))
            self.true_min.append(float(np.min(dists)))
            self.true_max.append(float(np.max(dists)))


def make_plot(node):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle('Distributed Consensus — Real-Time Convergence',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    titles   = ['Average Consensus', 'Minimum Consensus', 'Maximum Consensus']
    est_data = [node.avg_est, node.min_est, node.max_est]
    true_data= [node.true_avg, node.true_min, node.true_max]
    ylabels  = ['Distance (m)', 'Distance (m)', 'Distance (m)']

    lines = []   # [robot lines...] per subplot, plus true line
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_facecolor('#12122a')
        ax.set_title(title, color='white', fontsize=11, pad=6)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.grid(True, color='#1e1e3a', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(0, 3.0)

    subplot_lines = []
    for idx, (ax, title, edata, tdata) in enumerate(
            zip(axes, titles, est_data, true_data)):
        row = []
        for ri, r in enumerate(ALL_ROBOTS):
            ln, = ax.plot([], [], color=COLORS[ri], linewidth=1.8,
                          label=f'{r} estimate', alpha=0.9)
            row.append(ln)
        # True value — dashed white
        tln, = ax.plot([], [], color='white', linewidth=2.0,
                       linestyle='--', label='True value', alpha=0.85)
        row.append(tln)
        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#1a1a2e', edgecolor='#333355',
                  labelcolor='white')
        subplot_lines.append(row)

    axes[-1].set_xlabel('Time steps', color='#aaaaaa', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    x = list(range(WINDOW))

    def animate(_frame):
        for idx, (row, edata, tdata) in enumerate(
                zip(subplot_lines, est_data, true_data)):
            # Robot estimate lines
            for ri, r in enumerate(ALL_ROBOTS):
                row[ri].set_data(x, list(edata[r]))
            # True value line
            row[-1].set_data(x, list(tdata))

            # Auto-scale y axis
            all_vals = (
                [v for r in ALL_ROBOTS for v in edata[r]] +
                list(tdata)
            )
            valid = [v for v in all_vals if 0 < v < 900]
            if valid:
                mn = max(0, min(valid) - 0.2)
                mx = max(valid) + 0.2
                axes[idx].set_ylim(mn, mx)
        return [ln for row in subplot_lines for ln in row]

    ani = animation.FuncAnimation(
        fig, animate, interval=200, blit=True, cache_frame_data=False)
    plt.show()
    return ani


def main():
    rclpy.init()
    node = ConsensusPlotter()

    # Spin ROS2 in a background thread so matplotlib can use the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    ani = make_plot(node)   # blocks until window is closed

    rclpy.shutdown()


if __name__ == '__main__':
    main()