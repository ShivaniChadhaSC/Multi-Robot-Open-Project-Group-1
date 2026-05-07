#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from scipy.optimize import minimize

FORMATION_OFFSETS = {
    'tb3_1': np.array([-0.55,  0.32]),
    'tb3_2': np.array([-0.55, -0.32]),
}

class FollowerFormation(Node):
    def __init__(self):
        super().__init__('follower_formation')
        self.ns = self.get_namespace().strip('/')

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom',  self.odom_cb, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)

        self.pos   = np.zeros(2)
        self.yaw   = 0.0
        self.ready = False
        self.laser_ranges    = []
        self.laser_angle_min = 0.0
        self.laser_angle_inc = 0.0

        self.leader_pos     = np.zeros(2)
        self.leader_yaw_raw = 0.0

        self.leader_pos_history = []   # last N leader positions
        self.leader_heading     = 0.0  # smoothed heading
        self.leader_heading_set = False

        self.create_subscription(
            Odometry, '/tb3_0/odom', self.leader_cb, 10)

        other = 'tb3_2' if self.ns == 'tb3_1' else 'tb3_1'
        self.other_pos = np.array([999.0, 999.0])
        self.create_subscription(
            Odometry, f'/{other}/odom',
            lambda msg: self.other_cb(msg), 10)

        self.d_safe_obs   = 0.38
        self.d_safe_robot = 0.30
        self.d_activate   = 1.3
        self.alpha_obs    = 5.0
        self.alpha_robot  = 3.0
        self.v_max        = 0.20
        self.w_max        = 1.3
        self.ell          = 0.12
        self.k_form       = 1.2

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'FOLLOWER [{self.ns}] ready!')

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x,
                             msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y),
                              1-2*(q.y**2+q.z**2))
        self.ready = True

    def leader_cb(self, msg):
        new_pos = np.array([msg.pose.pose.position.x,
                            msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.leader_yaw_raw = math.atan2(2*(q.w*q.z+q.x*q.y),
                                          1-2*(q.y**2+q.z**2))

        if len(self.leader_pos_history) > 0:
            last = self.leader_pos_history[-1]
            moved = np.linalg.norm(new_pos - last)

            # Only update heading if leader moved at least 3cm
            # This filters out scanning rotations completely
            if moved > 0.03:
                raw_heading = math.atan2(
                    new_pos[1] - last[1],
                    new_pos[0] - last[0])

                if not self.leader_heading_set:
                    # First time — set directly
                    self.leader_heading     = raw_heading
                    self.leader_heading_set = True
                else:
                    # Smooth with exponential moving average
                    # alpha=0.15 → very smooth, slow to change
                    # This prevents formation jumping on small turns
                    alpha = 0.15
                    diff  = raw_heading - self.leader_heading
                    # Normalize angle difference
                    while diff >  math.pi: diff -= 2*math.pi
                    while diff < -math.pi: diff += 2*math.pi
                    self.leader_heading = self.leader_heading + alpha * diff

        self.leader_pos = new_pos

        # Keep only last 5 positions
        self.leader_pos_history.append(new_pos.copy())
        if len(self.leader_pos_history) > 5:
            self.leader_pos_history.pop(0)

    def other_cb(self, msg):
        self.other_pos = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y])

    def scan_cb(self, msg):
        self.laser_ranges    = list(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_inc = msg.angle_increment

    def _norm(self, a):
        while a >  math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def _formation_target(self):
      
        offset = FORMATION_OFFSETS[self.ns]

        # Use smoothed heading if available
        # Fall back to raw yaw only at startup
        heading = self.leader_heading if self.leader_heading_set \
                  else self.leader_yaw_raw

        c = math.cos(heading)
        s = math.sin(heading)
        rotated = np.array([c*offset[0] - s*offset[1],
                            s*offset[0] + c*offset[1]])
        return self.leader_pos + rotated

    def _build_constraints(self):
        cons = []
        n = len(self.laser_ranges)
        if n == 0: return cons

        step = max(1, n // 16)
        for i in range(0, n, step):
            r = self.laser_ranges[i]
            if r < 0.05 or r > self.d_activate or \
               not math.isfinite(r): continue
            ang = self.laser_angle_min + i*self.laser_angle_inc
            obs = self.pos + r * np.array([
                math.cos(self.yaw+ang),
                math.sin(self.yaw+ang)])
            diff = self.pos - obs
            cons.append((-2*diff,
                          self.alpha_obs*(r**2 - self.d_safe_obs**2)))

        # Other follower — soft constraint
        d_other = np.linalg.norm(self.pos - self.other_pos)
        if 0.05 < d_other < 2.0:
            diff = self.pos - self.other_pos
            cons.append((-2*diff,
                          self.alpha_robot*(d_other**2 - self.d_safe_robot**2)))

        # Leader — only if dangerously close
        d_leader = np.linalg.norm(self.pos - self.leader_pos)
        if 0.05 < d_leader < self.d_safe_robot + 0.05:
            diff = self.pos - self.leader_pos
            cons.append((-2*diff,
                          self.alpha_robot*(d_leader**2 - self.d_safe_robot**2)))
        return cons

    def _solve_qp(self, u_des, cons):
        if not cons: return u_des.copy()
        scipy_cons = [{
            'type': 'ineq',
            'fun':  lambda u, a=a, b=b: float(b - np.dot(a, u)),
            'jac':  lambda u, a=a: -a.astype(float)
        } for a, b in cons]
        res = minimize(
            lambda u: float(np.dot(u-u_des, u-u_des)),
            u_des.copy(),
            jac=lambda u: 2*(u-u_des),
            method='SLSQP',
            constraints=scipy_cons,
            options={'ftol':1e-5, 'maxiter':60}
        )
        return res.x if res.success else u_des.copy()

    def _best_gap_heading(self):
        if not self.laser_ranges: return self.yaw
        n = len(self.laser_ranges)
        best_w, best_a = 0, 0.0
        start = None
        for i in range(n):
            r = self.laser_ranges[i]
            ok = math.isfinite(r) and r > 0.55
            if ok:
                if start is None: start = i
            else:
                if start is not None:
                    w = i - start
                    if w > best_w:
                        best_w = w
                        ci = (start+i-1)//2
                        best_a = self.laser_angle_min + ci*self.laser_angle_inc
                    start = None
        return self._norm(self.yaw + best_a)

    def control_loop(self):
        if not self.ready or not self.laser_ranges: return

        target = self._formation_target()
        diff   = target - self.pos
        dist   = np.linalg.norm(diff)

        if dist < 0.08:
            self._publish(0.0, 0.0)
            return

        u_des  = self.k_form * diff / dist
        cons   = self._build_constraints()
        u_safe = self._solve_qp(u_des, cons)

        if np.linalg.norm(u_safe) < 0.05 and dist > 0.15:
            gap = self._best_gap_heading()
            err = self._norm(gap - self.yaw)
            self._publish(0.06, float(np.clip(
                2.5*err, -self.w_max, self.w_max)))
            self.get_logger().warn(
                f'[{self.ns}] gap redirect | '
                f'formation_dist={dist:.2f}m')
            return

        v = u_safe[0]*math.cos(self.yaw) + \
            u_safe[1]*math.sin(self.yaw)
        w = (-u_safe[0]*math.sin(self.yaw) + \
              u_safe[1]*math.cos(self.yaw)) / self.ell

        self._publish(v, w)
        self.get_logger().info(
            f'[{self.ns}] formation_dist={dist:.2f}m '
            f'heading={math.degrees(self.leader_heading):.0f}deg '
            f'v={v:.2f} w={w:.2f}')

    def _publish(self, v, w):
        twist = Twist()
        twist.linear.x  = float(np.clip(v, -self.v_max, self.v_max))
        twist.angular.z = float(np.clip(w, -self.w_max, self.w_max))
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FollowerFormation())
    rclpy.shutdown()

if __name__ == '__main__':
    main()