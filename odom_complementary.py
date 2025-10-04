#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from math import pi

def quat_to_rot(qx, qy, qz, qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1 - 2*(xx+yy)]
    ], dtype=float)

class OdomComplementary(Node):
    """
    Hybrid odom:
      - Integrate bias-corrected twist to get p_int
      - Measure raw position relative to an anchor p_raw_rel
      - Leash p_int slightly toward p_raw_rel
      - Blend: p_fused = alpha * p_int + (1-alpha) * p_raw_rel,
               with alpha higher when moving
      - Learn velocity bias only when 'still'
    """

    def __init__(self):
        super().__init__('odom_complementary')

        # Topics & basic params
        self.declare_parameter('input_topic', '/crazyflie_real/odom_raw')
        self.declare_parameter('output_topic', '/crazyflie_real/odom')
        self.declare_parameter('twist_in_odom_frame', True)   # True = no rotation needed
        self.declare_parameter('ang_units', 'deg')            # 'deg' or 'rad'
        self.declare_parameter('vel_scale', 0.001)              # e.g. 0.001 if mm/s

        # Calibration (bias learn) params
        self.declare_parameter('calibration_seconds', 1.0)
        self.declare_parameter('calibration_max_seconds', 5.0)
        self.declare_parameter('min_samples', 40)
        self.declare_parameter('calibrate_only_when_still', True)
        self.declare_parameter('still_lin_thresh', 0.04)      # m/s
        self.declare_parameter('still_ang_thresh', 0.05)      # rad/s (after unit convert)

        # Integration & fusion params
        self.declare_parameter('max_dt', 0.02)
        self.declare_parameter('zupt_lin_thresh', 0.00)       # set >0 to clamp tiny motion
        self.declare_parameter('zupt_ang_thresh', 0.00)
        self.declare_parameter('alpha_move', 0.85)            # trust integration when moving
        self.declare_parameter('alpha_still', 0.30)           # trust raw when still
        self.declare_parameter('move_lin_thresh', 0.06)       # defines "moving"
        self.declare_parameter('leash_to_raw_hz', 0.15)       # weak pull toward raw pose

        # Read params
        gp = self.get_parameter
        self.in_topic   = gp('input_topic').value
        self.out_topic  = gp('output_topic').value
        self.twist_is_odom = bool(gp('twist_in_odom_frame').value)
        self.ang_units  = gp('ang_units').value.lower()
        self.vel_scale  = float(gp('vel_scale').value)

        self.cal_secs   = float(gp('calibration_seconds').value)
        self.cal_max    = float(gp('calibration_max_seconds').value)
        self.min_samples= int(gp('min_samples').value)
        self.cal_only   = bool(gp('calibrate_only_when_still').value)
        self.still_lin  = float(gp('still_lin_thresh').value)
        self.still_ang  = float(gp('still_ang_thresh').value)

        self.max_dt     = float(gp('max_dt').value)
        self.zupt_lin   = float(gp('zupt_lin_thresh').value)
        self.zupt_ang   = float(gp('zupt_ang_thresh').value)
        self.alpha_move = float(gp('alpha_move').value)
        self.alpha_still= float(gp('alpha_still').value)
        self.move_lin   = float(gp('move_lin_thresh').value)
        self.leash_hz   = float(gp('leash_to_raw_hz').value)

        # State
        self.start_time = self.get_clock().now()
        self.last_time  = None
        self.calibrated = False
        self.good_cal_samples = 0

        self.bias = np.zeros(6, float)  # [vx,vy,vz, wx,wy,wz]
        self.buf_lin, self.buf_ang = [], []

        self.p_int = np.zeros(3, float)   # integrated pose
        self.anchor = None                 # raw pose anchor (numpy[3])

        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb, 200)
        self.pub = self.create_publisher(Odometry, self.out_topic, 20)

        self.get_logger().info(
            f"[Comp] {self.in_topic} -> {self.out_topic} | twist_in_odom={self.twist_is_odom}, ang_units={self.ang_units}"
        )

    def cb(self, msg: Odometry):
        now = self.get_clock().now()
        t_rel = (now - self.start_time).nanoseconds * 1e-9

        # Read & scale twist
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        v = self.vel_scale * np.array([lv.x, lv.y, lv.z], float)
        w = np.array([av.x, av.y, av.z], float)
        if self.ang_units.startswith('deg'):
            w *= (pi/180.0)

        # Raw pose
        p_raw = np.array([msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          msg.pose.pose.position.z], float)
        if self.anchor is None:
            self.anchor = p_raw.copy()  # first sample defines anchor

        # --- Calibration phase (learn velocity bias only when still) ---
        if not self.calibrated:
            lin_norm = np.linalg.norm(v)
            ang_norm = np.linalg.norm(w)
            accept = (lin_norm < self.still_lin) and (ang_norm < self.still_ang) if self.cal_only else True
            if accept:
                self.buf_lin.append(v)
                self.buf_ang.append(w)
                self.good_cal_samples += 1

            # Publish frozen at origin so RViz stays centred
            out = Odometry()
            out.header = msg.header
            out.child_frame_id = msg.child_frame_id
            out.pose = msg.pose
            out.pose.pose.position.x = 0.0
            out.pose.pose.position.y = 0.0
            out.pose.pose.position.z = 0.0
            out.twist = msg.twist
            self.pub.publish(out)

            # Finish calibration if enough samples or time’s up
            if (t_rel >= self.cal_secs and self.good_cal_samples >= self.min_samples) or (t_rel >= self.cal_max):
                if self.good_cal_samples >= max(1, self.min_samples):
                    vlin = np.array(self.buf_lin, float)
                    vang = np.array(self.buf_ang, float)
                    self.bias[:3] = vlin.mean(axis=0)
                    self.bias[3:] = vang.mean(axis=0)
                    self.get_logger().info(f"[Comp] Bias lin={self.bias[:3]}, ang={self.bias[3:]}")
                else:
                    self.get_logger().warn("[Comp] No still window found, skipping bias calibration.")
                self.calibrated = True
                self.last_time = None
                self.p_int[:] = 0.0
                # reset anchor at end of calib for a clean zero
                self.anchor = p_raw.copy()
            return

        # --- After calibration ---
        # Timing
        if self.last_time is None:
            dt = 0.0
            self.last_time = now
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
            if dt < 0.0: dt = 0.0
            if dt > self.max_dt: dt = self.max_dt
            self.last_time = now

        # Bias-correct & ZUPT
        v -= self.bias[:3]
        w -= self.bias[3:]
        if np.linalg.norm(v) < self.zupt_lin and np.linalg.norm(w) < self.zupt_ang:
            v[:] = 0.0; w[:] = 0.0

        # Rotate velocity if needed
        if self.twist_is_odom:
            v_odom = v
        else:
            q = msg.pose.pose.orientation
            qn = np.array([q.x,q.y,q.z,q.w], float)
            nrm = np.linalg.norm(qn)
            if nrm > 1e-9:
                qn /= nrm
            R = quat_to_rot(qn[0], qn[1], qn[2], qn[3])
            v_odom = R @ v

        # Integrate
        self.p_int += v_odom * dt

        # Raw relative pose (detrended only by anchor)
        p_raw_rel = p_raw - self.anchor

        # Weak leash toward raw pose (prevents unbounded drift)
        lam = 2.0 * np.pi * self.leash_hz   # s^-1
        if lam > 0.0:
            self.p_int += (p_raw_rel - self.p_int) * lam * dt

        # Motion-aware blend
        alpha = self.alpha_move if np.linalg.norm(v) > self.move_lin else self.alpha_still
        p_fused = alpha * self.p_int + (1.0 - alpha) * p_raw_rel

        # Publish fused odom
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id

        out.pose = msg.pose
        out.pose.pose.position.x = float(p_fused[0])
        out.pose.pose.position.y = float(p_fused[1])
        out.pose.pose.position.z = float(p_fused[2])
        out.pose.covariance = msg.pose.covariance

        out.twist = msg.twist
        out.twist.twist.linear.x = v[0]
        out.twist.twist.linear.y = v[1]
        out.twist.twist.linear.z = v[2]
        out.twist.twist.angular.x = w[0]
        out.twist.twist.angular.y = w[1]
        out.twist.twist.angular.z = w[2]
        out.twist.covariance = msg.twist.covariance

        self.pub.publish(out)

def main():
    rclpy.init()
    node = OdomComplementary()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
