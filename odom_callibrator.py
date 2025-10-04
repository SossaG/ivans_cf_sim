#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from math import sqrt, pi

def quat_to_rot(qx,qy,qz,qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)]
    ], dtype=float)

class OdomLeashed(Node):
    def __init__(self):
        super().__init__('odom_leashed')

        # ---------- params ----------
        self.declare_parameter('input_topic', '/crazyflie_real/odom_raw')
        self.declare_parameter('output_topic','/crazyflie_real/odom')
        self.declare_parameter('calibration_seconds', 3.0)
        self.declare_parameter('min_samples', 40)
        self.declare_parameter('ang_units', 'deg')          # 'deg' or 'rad'
        self.declare_parameter('twist_in_odom_frame', False) # True if v already in odom
        self.declare_parameter('vel_lowpass_alpha', 0.85)    # 0..1 (closer to 1 = smoother)
        self.declare_parameter('zupt_lin_thresh', 0.03)      # m/s
        self.declare_parameter('zupt_ang_thresh', 0.03)      # rad/s (after unit convert)
        self.declare_parameter('position_leash_hz', 0.5)     # pull p toward 0 at ~0.5 Hz
        self.declare_parameter('max_dt', 0.03)               # s (clamp)
        self.declare_parameter('max_speed', 2.0)             # m/s clamp
        self.declare_parameter('log_every', 50)

        gp = self.get_parameter
        self.in_topic   = gp('input_topic').value
        self.out_topic  = gp('output_topic').value
        self.cal_secs   = float(gp('calibration_seconds').value)
        self.min_samples= int(gp('min_samples').value)
        self.ang_units  = gp('ang_units').value.lower()
        self.twist_is_odom = bool(gp('twist_in_odom_frame').value)
        self.alpha      = float(gp('vel_lowpass_alpha').value)
        self.zupt_lin   = float(gp('zupt_lin_thresh').value)
        self.zupt_ang   = float(gp('zupt_ang_thresh').value)
        self.leash_hz   = float(gp('position_leash_hz').value)
        self.max_dt     = float(gp('max_dt').value)
        self.max_speed  = float(gp('max_speed').value)
        self.log_every  = int(gp('log_every').value)

        # ---------- state ----------
        self.calibrated = False
        self.start_time = self.get_clock().now()
        self.last_time  = None
        self.n          = 0

        # bias (vx,vy,vz, wx,wy,wz)
        self.bias = np.zeros(6, float)
        self.buf_lin = []
        self.buf_ang = []

        self.p = np.zeros(3, float)
        self.vf = np.zeros(3, float)  # filtered linear vel (body or odom depending on path)

        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb, 200)
        self.pub = self.create_publisher(Odometry, self.out_topic, 20)

        self.get_logger().info(
            f"[Leashed] {self.in_topic} -> {self.out_topic}\n"
            f"Calibrating {self.cal_secs:.1f}s (keep still). ang_units={self.ang_units}, twist_in_odom_frame={self.twist_is_odom}"
        )

    def cb(self, msg: Odometry):
        now = self.get_clock().now()
        t_rel = (now - self.start_time).nanoseconds * 1e-9

        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        v = np.array([lv.x, lv.y, lv.z], float)
        w = np.array([av.x, av.y, av.z], float)

        # unit convert angular if needed
        if self.ang_units.startswith('deg'):
            w = w * (pi/180.0)

        if not self.calibrated:
            self.buf_lin.append(v)
            self.buf_ang.append(w)
            out = Odometry()
            out.header = msg.header
            out.child_frame_id = msg.child_frame_id
            out.pose = msg.pose
            out.pose.pose.position.x = 0.0
            out.pose.pose.position.y = 0.0
            out.pose.pose.position.z = 0.0
            out.twist = msg.twist  # pass through for now
            self.pub.publish(out)

            if (t_rel >= self.cal_secs) and (len(self.buf_lin) >= self.min_samples):
                self._finish_cal()
            return

        # timing
        if self.last_time is None:
            dt = 0.0
            self.last_time = now
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if dt < 0.0: dt = 0.0
            if dt > self.max_dt: dt = self.max_dt
            self.last_time = now

        # bias-correct
        v -= self.bias[:3]
        w -= self.bias[3:]

        # ZUPT
        if np.linalg.norm(v) < self.zupt_lin and np.linalg.norm(w) < self.zupt_ang:
            v[:] = 0.0
            w[:] = 0.0

        # low-pass velocity
        self.vf = self.alpha*self.vf + (1.0-self.alpha)*v

        # clamp speed
        spd = np.linalg.norm(self.vf)
        if spd > self.max_speed:
            self.vf *= (self.max_speed / spd)

        # choose frame for integration
        if self.twist_is_odom:
            v_odom = self.vf
        else:
            q = msg.pose.pose.orientation
            # normalize quaternion (defensive)
            nq = np.array([q.x,q.y,q.z,q.w], float)
            nq = nq / max(1e-9, np.linalg.norm(nq))
            R = quat_to_rot(nq[0],nq[1],nq[2],nq[3])
            v_odom = R @ self.vf

        # leashed integration: dp/dt = v_odom - lambda*p
        lam = 2.0 * np.pi * self.leash_hz  # rad/s → “strength”
        self.p += (v_odom - lam*self.p) * dt

        # publish
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.pose.pose.position.x = float(self.p[0])
        out.pose.pose.position.y = float(self.p[1])
        out.pose.pose.position.z = float(self.p[2])
        out.pose.covariance = msg.pose.covariance
        out.twist = msg.twist
        out.twist.twist.linear.x  = float(self.vf[0])
        out.twist.twist.linear.y  = float(self.vf[1])
        out.twist.twist.linear.z  = float(self.vf[2])
        out.twist.twist.angular.x = float(w[0])
        out.twist.twist.angular.y = float(w[1])
        out.twist.twist.angular.z = float(w[2])
        out.twist.covariance = msg.twist.covariance

        if (self.n % self.log_every) == 0:
            self.get_logger().info(f"[Leashed] dt={dt:.3f}s |v|={np.linalg.norm(self.vf):.2f} p=({self.p[0]:+.2f},{self.p[1]:+.2f},{self.p[2]:+.2f})")
        self.n += 1
        self.pub.publish(out)

    def _finish_cal(self):
        vlin = np.array(self.buf_lin, float)
        vang = np.array(self.buf_ang, float)
        self.bias[:3] = vlin.mean(axis=0) if len(vlin) else 0.0
        self.bias[3:] = vang.mean(axis=0) if len(vang) else 0.0
        self.calibrated = True
        self.last_time = None
        self.p[:] = 0.0
        self.vf[:] = 0.0
        self.get_logger().info(
            "Calibration complete.\n"
            f"  vel_bias lin = [{self.bias[0]:+.4f}, {self.bias[1]:+.4f}, {self.bias[2]:+.4f}] m/s\n"
            f"  vel_bias ang = [{self.bias[3]:+.4f}, {self.bias[4]:+.4f}, {self.bias[5]:+.4f}] rad/s\n"
            f"  twist_in_odom_frame={self.twist_is_odom}, leash_hz={self.leash_hz}"
        )

def main():
    rclpy.init()
    node = OdomLeashed()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
