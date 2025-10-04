#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class OdomSubtractDrift(Node):
    def __init__(self):
        super().__init__('odom_subtract_drift')
        self.declare_parameter('input_topic', '/crazyflie_real/odom_raw')
        self.declare_parameter('output_topic', '/crazyflie_real/odom')
        self.declare_parameter('calibration_seconds', 2.0)
        self.declare_parameter('min_samples', 40)
        self.declare_parameter('freeze_output_during_calibration', True)

        self.in_topic  = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.cal_secs  = float(self.get_parameter('calibration_seconds').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.freeze = bool(self.get_parameter('freeze_output_during_calibration').value)

        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb, 200)
        self.pub = self.create_publisher(Odometry, self.out_topic, 20)

        # buffers
        self.t0 = None
        self.t_rel = []
        self.pos   = []   # Nx3 from pose
        self.vlin  = []   # Nx3 from twist
        self.vang  = []   # Nx3 from twist
        self.calibrated = False

        # learned params
        self.pos_slope = np.zeros(3)     # drift velocity (m/s)
        self.pos_intercept = np.zeros(3) # intercept
        self.pos_zero = np.zeros(3)      # shift so first post-calib sample = 0
        self.vel_bias = np.zeros(6)      # [vx,vy,vz, wx,wy,wz]

        self.get_logger().info(f"[DriftSub] {self.in_topic} -> {self.out_topic} | calibrating {self.cal_secs:.1f}s")

    def cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None: self.t0 = t
        tr = t - self.t0

        p = msg.pose.pose.position
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        raw_p = np.array([p.x, p.y, p.z], float)

        if not self.calibrated:
            self.t_rel.append(tr)
            self.pos.append(raw_p)
            self.vlin.append([lv.x, lv.y, lv.z])
            self.vang.append([av.x, av.y, av.z])

            # keep RViz calm during calibration
            out = Odometry()
            out.header = msg.header
            out.child_frame_id = msg.child_frame_id
            out.pose = msg.pose
            if self.freeze:
                out.pose.pose.position.x = 0.0
                out.pose.pose.position.y = 0.0
                out.pose.pose.position.z = 0.0
            out.twist = msg.twist
            self.pub.publish(out)

            if (tr >= self.cal_secs) and (len(self.t_rel) >= self.min_samples):
                self._finish_calibration(tr)
            return

        # apply drift subtraction to pose
        detrended = raw_p - (self.pos_slope * tr + self.pos_intercept)
        corrected_pos = detrended - self.pos_zero

        # apply twist bias subtraction
        vbx, vby, vbz, wbx, wby, wbz = self.vel_bias
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.pose.pose.position.x = float(corrected_pos[0])
        out.pose.pose.position.y = float(corrected_pos[1])
        out.pose.pose.position.z = float(corrected_pos[2])
        out.pose.covariance = msg.pose.covariance

        out.twist = msg.twist
        out.twist.twist.linear.x  = lv.x - vbx
        out.twist.twist.linear.y  = lv.y - vby
        out.twist.twist.linear.z  = lv.z - vbz
        out.twist.twist.angular.x = av.x - wbx
        out.twist.twist.angular.y = av.y - wby
        out.twist.twist.angular.z = av.z - wbz
        out.twist.covariance = msg.twist.covariance
        self.pub.publish(out)

    def _finish_calibration(self, tr_last: float):
        t = np.array(self.t_rel)
        pos = np.array(self.pos)          # N x 3
        A = np.vstack([t, np.ones_like(t)]).T
        slopes, intercepts = [], []
        for i in range(3):
            m, c = np.linalg.lstsq(A, pos[:, i], rcond=None)[0]
            slopes.append(m); intercepts.append(c)
        self.pos_slope = np.array(slopes)          # m/s drift in odom frame
        self.pos_intercept = np.array(intercepts)

        vlin = np.array(self.vlin); vang = np.array(self.vang)
        vel_bias_lin = vlin.mean(axis=0) if len(vlin) else np.zeros(3)
        vel_bias_ang = vang.mean(axis=0) if len(vang) else np.zeros(3)
        self.vel_bias = np.concatenate([vel_bias_lin, vel_bias_ang], axis=0)

        # zero so first post-calibration sample is (0,0,0)
        last_pos = pos[-1]
        last_detrended = last_pos - (self.pos_slope * tr_last + self.pos_intercept)
        self.pos_zero = last_detrended

        self.calibrated = True
        self.get_logger().info(
            f"[DriftSub] Done. drift_v={self.pos_slope} m/s, vel_bias={self.vel_bias}"
        )

def main():
    rclpy.init()
    node = OdomSubtractDrift()
    try: rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
