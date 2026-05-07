#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import json
import math

ALL_ROBOTS = ['tb3_0', 'tb3_1', 'tb3_2']
WORLD_X_MIN, WORLD_X_MAX = -2.8,  2.8
WORLD_Y_MIN, WORLD_Y_MAX = -2.3,  2.3
GRID_RES      = 0.2
SENSING_RADIUS = 2.5

class VoronoiCoverage(Node):
    def __init__(self):
        super().__init__('voronoi_coverage')
        self.get_logger().info('Voronoi Coverage Node started')

        self.positions = {r: np.zeros(2) for r in ALL_ROBOTS}
        for r in ALL_ROBOTS:
            self.create_subscription(
                Odometry, f'/{r}/odom',
                lambda msg, n=r: self.odom_cb(msg, n), 10)

        # Build grid
        xs = np.arange(WORLD_X_MIN, WORLD_X_MAX, GRID_RES)
        ys = np.arange(WORLD_Y_MIN, WORLD_Y_MAX, GRID_RES)
        gx, gy = np.meshgrid(xs, ys)
        self.grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

        # Precompute static part of density
        self.static_density = self._static_density(self.grid_pts)

        # Time counter for time-varying density
        self.t = 0.0

        self.pub_cells     = self.create_publisher(String, '/voronoi/cells', 10)
        self.pub_centroids = self.create_publisher(Float32MultiArray, '/voronoi/centroids', 10)
        self.pub_coverage  = self.create_publisher(Float32MultiArray, '/voronoi/coverage_score', 10)

        self.create_timer(1.0, self.voronoi_loop)

    def odom_cb(self, msg, name):
        self.positions[name] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y])

    def _static_density(self, pts):
        """Non-uniform static density: walls + center."""
        density = np.ones(len(pts))
        for i, (x, y) in enumerate(pts):
            wall = (abs(x-WORLD_X_MIN)<0.5 or abs(x-WORLD_X_MAX)<0.5 or
                    abs(y-WORLD_Y_MIN)<0.5 or abs(y-WORLD_Y_MAX)<0.5)
            center = math.exp(-0.3 * math.sqrt(x**2 + y**2))
            density[i] = 1.0 + 0.5*float(wall) + center
        return density

    def _time_varying_density(self, pts, t):
        """
        Time-varying density: a high-importance hotspot
        that moves around the world over time.
        Simulates a dynamic event (fire, signal source, etc.)
        """
        # Hotspot moves in a circle over time
        hx = 1.5 * math.cos(t * 0.1)
        hy = 1.5 * math.sin(t * 0.1)
        varying = np.zeros(len(pts))
        for i, (x, y) in enumerate(pts):
            d = math.sqrt((x-hx)**2 + (y-hy)**2)
            varying[i] = 2.0 * math.exp(-1.5 * d)
        return varying

    def _compute_density(self, pts, t):
        """Combined: static + time-varying."""
        return self.static_density + self._time_varying_density(pts, t)

    def _compute_voronoi_cells(self, density):
        robot_list = list(ALL_ROBOTS)
        positions  = np.array([self.positions[r] for r in robot_list])
        cells      = {r: [] for r in robot_list}

        for i, pt in enumerate(self.grid_pts):
            dists   = np.linalg.norm(positions - pt, axis=1)
            in_range = dists < SENSING_RADIUS
            if not any(in_range):
                continue
            masked  = np.where(in_range, dists, np.inf)
            nearest = robot_list[np.argmin(masked)]
            cells[nearest].append(i)

        return cells

    def _compute_centroid(self, cell_indices, density):
        if not cell_indices:
            return np.zeros(2)
        pts     = self.grid_pts[cell_indices]
        weights = density[cell_indices]
        total   = weights.sum()
        return (pts * weights[:,None]).sum(axis=0) / total if total > 0 \
               else pts.mean(axis=0)

    def voronoi_loop(self):
        self.t += 1.0

        # Compute time-varying density
        density = self._compute_density(self.grid_pts, self.t)

        # Voronoi cells
        cells      = self._compute_voronoi_cells(density)
        centroids  = {}
        cell_sizes = {}

        for r in ALL_ROBOTS:
            centroids[r]  = self._compute_centroid(cells[r], density)
            cell_sizes[r] = len(cells[r])

        coverage = sum(len(v) for v in cells.values()) / len(self.grid_pts)

        # Hotspot position for logging
        hx = round(1.5 * math.cos(self.t * 0.1), 2)
        hy = round(1.5 * math.sin(self.t * 0.1), 2)

        # Publish
        centroid_msg = Float32MultiArray()
        centroid_msg.data = [float(v)
                             for r in ALL_ROBOTS
                             for v in centroids[r].tolist()]
        self.pub_centroids.publish(centroid_msg)

        cov_msg = Float32MultiArray()
        cov_msg.data = [float(coverage)] + \
                       [float(cell_sizes[r]) for r in ALL_ROBOTS]
        self.pub_coverage.publish(cov_msg)

        cells_info = {
            r: {
                'cell_size':          cell_sizes[r],
                'centroid_x':         round(float(centroids[r][0]), 2),
                'centroid_y':         round(float(centroids[r][1]), 2),
                'coverage_fraction':  round(cell_sizes[r]/len(self.grid_pts), 3)
            }
            for r in ALL_ROBOTS
        }
        cells_msg = String()
        cells_msg.data = json.dumps({
            'coverage_score':  round(float(coverage), 3),
            'sensing_radius':  SENSING_RADIUS,
            'time':            self.t,
            'hotspot':         [hx, hy],
            'cells':           cells_info
        })
        self.pub_cells.publish(cells_msg)

        self.get_logger().info(
            f'VORONOI t={self.t:.0f}s '
            f'coverage={coverage*100:.1f}% '
            f'hotspot=({hx},{hy}) | '
            + ' | '.join([
                f'{r}:{cell_sizes[r]}cells '
                f'c=({centroids[r][0]:.1f},{centroids[r][1]:.1f})'
                for r in ALL_ROBOTS]))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VoronoiCoverage())
    rclpy.shutdown()

if __name__ == '__main__':
    main()