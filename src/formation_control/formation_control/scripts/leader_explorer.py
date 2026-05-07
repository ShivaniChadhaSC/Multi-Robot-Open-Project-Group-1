#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from scipy.optimize import minimize

WAYPOINTS = [
    np.array([2.0, 1.5]),
    np.array([2.0, -1.5]),
    np.array([-2.0, -1.5]),
    np.array([-2.0, 1.5]),
    np.array([0.0, 0.0]),
]

# Formation parameters
FORMATION_DISTANCE = 1.2
FORMATION_ANGLE = math.radians(50)   # ~60 degree triangle

class LeaderExplorer(Node):
    def __init__(self):
        super().__init__('leader_explorer')
       
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
       
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
       
        # Other robots
        self.others = {}
        for r in ['tb3_1', 'tb3_2']:
            self.others[r] = np.array([999.0, 999.0])
            self.create_subscription(
                Odometry, f'/{r}/odom',
                lambda msg, name=r: self.other_cb(msg, name), 10)

        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.ready = False
        self.laser_ranges = []
        self.laser_angle_min = 0.0
        self.laser_angle_inc = 0.0
        self.scan_buffer = []

        # CBF parameters
        self.d_safe_obs = 0.35
        self.d_safe_robot = 0.52
        self.d_activate = 1.2
        self.alpha = 6.0
        self.v_max = 0.17
        self.w_max = 1.2
        self.ell = 0.12
        self.k_goal = 0.8

        # Formation Control
        self.k_formation = 0.65     # How much leader cares about formation
        self.max_follower_lag = 1.5 # If follower is farther than this, slow down

        # Waypoint tracking
        self.wp_index = 0
        self.final_goal = WAYPOINTS[self.wp_index]
        self.wp_thresh = 0.40

        # Subgoal
        self.subgoal = None
        self.subgoal_dist = 1.0
        self.subgoal_thresh = 0.45
        self.min_clearance = 0.55

        # States
        self.state = 'scanning'
        self.scan_count = 0
        self.scan_ticks = 8

        # Tangent mode
        self.tangent_mode = False
        self.tangent_dir = 1.0
        self.tangent_count = 0
        self.tangent_max = 50

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'LEADER ready with FORMATION CONTROL | First waypoint: {self.final_goal}')

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        self.ready = True

    def other_cb(self, msg, name):
        self.others[name] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def scan_cb(self, msg):
        self.laser_ranges = list(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_inc = msg.angle_increment
        self.scan_buffer.append(list(msg.ranges))
        if len(self.scan_buffer) > 12:
            self.scan_buffer.pop(0)

    def _norm(self, a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def _next_waypoint(self):
        self.wp_index = (self.wp_index + 1) % len(WAYPOINTS)
        self.final_goal = WAYPOINTS[self.wp_index]
        self.tangent_mode = False
        self.subgoal = None
        self.get_logger().info(f'WP REACHED! Next: wp{self.wp_index} {self.final_goal}')

    # === All your previous helper functions remain unchanged ===
    def _clearance_in_direction(self, world_angle):
        if not self.laser_ranges: return 0.0
        cone = math.radians(15)
        min_r = 999.0
        n = len(self.laser_ranges)
        for i in range(n):
            sensor_ang = self.laser_angle_min + i * self.laser_angle_inc
            beam_world = self._norm(self.yaw + sensor_ang)
            if abs(self._norm(beam_world - world_angle)) < cone:
                r = self.laser_ranges[i]
                if math.isfinite(r) and r > 0.05:
                    min_r = min(min_r, r)
        return min_r if min_r < 999.0 else 3.5

    def _path_to_goal_blocked(self):
        dg = self.final_goal - self.pos
        dist_to_goal = np.linalg.norm(dg)
        if dist_to_goal < 0.3:
            return False, 999.0, 0.0
        goal_heading = math.atan2(dg[1], dg[0])
        clearance = self._clearance_in_direction(goal_heading)
        blocked = clearance < min(dist_to_goal * 0.8, self.d_safe_obs + 0.4)
        return blocked, clearance, goal_heading

    def _find_safe_direction(self):
        if not self.laser_ranges: return None, 0.0
        avg_ranges = np.mean(np.array(self.scan_buffer), axis=0) if self.scan_buffer else np.array(self.laser_ranges)
        avg_ranges = np.where(np.isfinite(avg_ranges) & (avg_ranges > 0.05), avg_ranges, 3.5)
        dg = self.final_goal - self.pos
        goal_heading = math.atan2(dg[1], dg[0])
        best_score = -999.0
        best_angle = None
        best_clear = 0.0
        for deg in range(0, 360, 5):
            world_angle = self._norm(math.radians(deg))
            clearance = self._clearance_in_direction(world_angle)
            if clearance < self.min_clearance * 0.7: continue
            angle_diff = abs(self._norm(world_angle - goal_heading))
            goal_score = math.cos(angle_diff)
            score = goal_score * 0.75 + (clearance / 4.0) * 0.25
            if score > best_score:
                best_score = score
                best_angle = world_angle
                best_clear = clearance
        return best_angle, best_clear

    def _place_subgoal(self):
        best_dir, clearance = self._find_safe_direction()
        if best_dir is None:
            self.get_logger().warn('No safe direction found - using goal direction')
            dg = self.final_goal - self.pos
            best_dir = math.atan2(dg[1], dg[0])
            clearance = 1.0
        safe_dist = min(self.subgoal_dist, max(0.6, clearance * 0.65))
        self.subgoal = self.pos + safe_dist * np.array([math.cos(best_dir), math.sin(best_dir)])
        self.get_logger().info(
            f'SUBGOAL placed at ({self.subgoal[0]:.2f}, {self.subgoal[1]:.2f}) '
            f'dir={math.degrees(best_dir):.0f}° clear={clearance:.2f}m')
        return True

    def _min_front_dist(self):
        if not self.laser_ranges: return 999.0
        front = []
        for i in range(len(self.laser_ranges)):
            ang = self.laser_angle_min + i * self.laser_angle_inc
            if abs(ang) < math.radians(45):
                r = self.laser_ranges[i]
                if math.isfinite(r) and r > 0.05:
                    front.append(r)
        return min(front) if front else 999.0

    def _build_constraints(self):
        cons = []
        n = len(self.laser_ranges)
        if n == 0: return cons
        step = max(1, n // 20)
        for i in range(0, n, step):
            r = self.laser_ranges[i]
            if r < 0.05 or r > self.d_activate or not math.isfinite(r): continue
            ang = self.laser_angle_min + i * self.laser_angle_inc
            obs = self.pos + r * np.array([math.cos(self.yaw + ang), math.sin(self.yaw + ang)])
            diff = self.pos - obs
            cons.append((-2 * diff, self.alpha * (r**2 - self.d_safe_obs**2)))

        for opos in self.others.values():
            d = np.linalg.norm(self.pos - opos)
            if 0.05 < d < 2.5:
                diff = self.pos - opos
                cons.append((-2 * diff, self.alpha * (d**2 - self.d_safe_robot**2)))
        return cons

    def _solve_qp(self, u_des, cons):
        if not cons: return u_des.copy()
        scipy_cons = [{'type': 'ineq', 'fun': lambda u, a=a, b=b: float(b - np.dot(a, u))}
                      for a, b in cons]
        res = minimize(
            lambda u: np.dot(u - u_des, u - u_des),
            u_des.copy(),
            jac=lambda u: 2 * (u - u_des),
            method='SLSQP',
            constraints=scipy_cons,
            options={'ftol': 1e-5, 'maxiter': 60}
        )
        return res.x if res.success else u_des.copy()

    def _publish(self, v, w):
        twist = Twist()
        twist.linear.x = float(np.clip(v, -self.v_max, self.v_max))
        twist.angular.z = float(np.clip(w, -self.w_max, self.w_max))
        self.cmd_pub.publish(twist)

    def _get_formation_adjustment(self):
        """Stronger formation feedback"""
        adjustment = np.zeros(2)
        total_error = 0.0
        count = 0

        for fpos in self.others.values():
            if np.any(fpos > 900): continue
            d = np.linalg.norm(self.pos - fpos)
            total_error += max(0, d - 1.4)   # penalize if farther than 1.4m
            count += 1

        if count > 0:
            avg_error = total_error / count
            if avg_error > 0.2:
                slowdown = 0.65 - 0.35 * avg_error
                slowdown = max(0.4, slowdown)
                adjustment = (slowdown - 1.0) * np.array([math.cos(self.yaw), math.sin(self.yaw)])

        return adjustment

    def control_loop(self):
        if not self.ready or not self.laser_ranges:
            return

        dist_to_wp = np.linalg.norm(self.pos - self.final_goal)
        if dist_to_wp < self.wp_thresh:
            self._next_waypoint()
            self.state = 'scanning'
            self.scan_count = 0
            self.scan_buffer.clear()
            self._publish(0, 0)
            return

        # Blocked check
        blocked, clearance_fwd, goal_heading = self._path_to_goal_blocked()
        if blocked and not self.tangent_mode:
            self.tangent_mode = True
            self.tangent_count = 0
            self.tangent_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            self.get_logger().warn(f'OBSTACLE blocking! Going {"LEFT" if self.tangent_dir > 0 else "RIGHT"}')

        if self.tangent_mode:
            # ... (your original tangent code unchanged)
            self.tangent_count += 1
            blocked_now, _, _ = self._path_to_goal_blocked()
            if not blocked_now or self.tangent_count > self.tangent_max:
                self.tangent_mode = False
                self.get_logger().info('Exiting tangent mode')
            else:
                perp_angle = self._norm(goal_heading + self.tangent_dir * math.radians(75))
                u_des = self.k_goal * np.array([math.cos(perp_angle), math.sin(perp_angle)])
                u_safe = self._solve_qp(u_des, self._build_constraints())
                v = u_safe[0] * math.cos(self.yaw) + u_safe[1] * math.sin(self.yaw)
                w = (-u_safe[0] * math.sin(self.yaw) + u_safe[1] * math.cos(self.yaw)) / self.ell
                self._publish(v, w)
                return

        # Scanning State
        if self.state == 'scanning':
            self.scan_count += 1
            self._publish(0.0, 0.4)
            if self.scan_count >= self.scan_ticks or self.scan_count > 35:
                self._place_subgoal()
                self.state = 'moving'
                self.scan_count = 0
            return

        # Moving State
        if self.subgoal is None:
            self.state = 'scanning'
            return

        dist_to_sub = np.linalg.norm(self.pos - self.subgoal)
        min_front = self._min_front_dist()

        if dist_to_sub < self.subgoal_thresh or min_front < 0.25:
            self.get_logger().info(f'Subgoal reached or blocked → rescanning')
            self.state = 'scanning'
            self.scan_count = 0
            self.scan_buffer.clear()
            self.subgoal = None
            self._publish(0, 0)
            return
            
        dg = self.subgoal - self.pos
        u_des = self.k_goal * dg / np.linalg.norm(dg)

        # Add formation adjustment
        form_adjust = self._get_formation_adjustment()
        u_des += self.k_formation * form_adjust

        cons = self._build_constraints()
        u_safe = self._solve_qp(u_des, cons)

        v = u_safe[0] * math.cos(self.yaw) + u_safe[1] * math.sin(self.yaw)
        w = (-u_safe[0] * math.sin(self.yaw) + u_safe[1] * math.cos(self.yaw)) / self.ell

        self._publish(v, w)

def main(args=None):
    rclpy.init(args=args)
    node = LeaderExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()