#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import json

ALL_ROBOTS = ['tb3_0', 'tb3_1', 'tb3_2']

class TopologyNode(Node):
    def __init__(self):
        super().__init__('topology_node')
        self.get_logger().info('Topology Control Node started')

        # Robot positions
        self.positions = {r: np.array([999.0, 999.0]) for r in ALL_ROBOTS}
        for r in ALL_ROBOTS:
            self.create_subscription(
                Odometry, f'/{r}/odom',
                lambda msg, n=r: self.odom_cb(msg, n), 10)

        # Communication range thresholds
        self.range_add    = 3.0   # add link if closer than this
        self.range_remove = 4.0   # remove link if farther than this

        # Current active links: set of tuples (robot_a, robot_b)
        self.active_links = set()

        # Publishers
        self.pub_topology = self.create_publisher(
            String, '/topology/links', 10)
        self.pub_matrix = self.create_publisher(
            Float32MultiArray, '/topology/adjacency_matrix', 10)

        self.create_timer(0.5, self.topology_loop)

    def odom_cb(self, msg, name):
        self.positions[name] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

    def _distance(self, r1, r2):
        return float(np.linalg.norm(
            self.positions[r1] - self.positions[r2]))

    def topology_loop(self):
        n = len(ALL_ROBOTS)
        adj = np.zeros((n, n))

        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                r1 = ALL_ROBOTS[i]
                r2 = ALL_ROBOTS[j]
                d  = self._distance(r1, r2)
                link = (r1, r2)

                if link in self.active_links:
                    # Remove link if too far
                    if d > self.range_remove:
                        self.active_links.discard(link)
                        self.get_logger().warn(
                            f'LINK REMOVED: {r1} ↔ {r2} '
                            f'(dist={d:.2f}m > {self.range_remove}m)')
                    else:
                        adj[i][j] = adj[j][i] = 1.0
                else:
                    # Add link if close enough
                    if d < self.range_add:
                        self.active_links.add(link)
                        self.get_logger().info(
                            f'LINK ADDED: {r1} ↔ {r2} '
                            f'(dist={d:.2f}m < {self.range_add}m)')
                        adj[i][j] = adj[j][i] = 1.0

        # Check connectivity — ensure graph is connected
        connected = self._is_connected(adj, n)
        if not connected:
            self.get_logger().warn(
                'CONNECTIVITY LOST — forcing minimum spanning links!')
            # Force add closest pair to restore connectivity
            self._restore_connectivity(adj)

        # Publish topology
        links_data = {
            'active_links': [list(l) for l in self.active_links],
            'connected': connected,
            'distances': {
                f'{ALL_ROBOTS[i]}-{ALL_ROBOTS[j]}':
                round(self._distance(ALL_ROBOTS[i], ALL_ROBOTS[j]), 2)
                for i in range(n) for j in range(i+1, n)
            }
        }
        msg = String()
        msg.data = json.dumps(links_data)
        self.pub_topology.publish(msg)

        # Publish adjacency matrix
        mat_msg = Float32MultiArray()
        mat_msg.data = adj.flatten().tolist()
        self.pub_matrix.publish(mat_msg)

        # Log current topology
        self.get_logger().info(
            f'TOPOLOGY: {len(self.active_links)} links | '
            f'connected={connected} | '
            f'links={[f"{a}↔{b}" for a,b in self.active_links]}')

    def _is_connected(self, adj, n):
        """BFS to check if graph is connected."""
        if n == 0: return True
        visited = set([0])
        queue   = [0]
        while queue:
            node = queue.pop(0)
            for j in range(n):
                if adj[node][j] > 0 and j not in visited:
                    visited.add(j)
                    queue.append(j)
        return len(visited) == n

    def _restore_connectivity(self, adj):
        """Add minimum links to restore connectivity."""
        n = len(ALL_ROBOTS)
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 0:
                    r1, r2 = ALL_ROBOTS[i], ALL_ROBOTS[j]
                    self.active_links.add((r1, r2))
                    adj[i][j] = adj[j][i] = 1.0
                    self.get_logger().warn(
                        f'FORCED LINK: {r1} ↔ {r2}')
                    if self._is_connected(adj, n):
                        return

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TopologyNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()