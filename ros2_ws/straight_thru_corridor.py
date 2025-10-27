#!/usr/bin/env python3
# minimal_explorer.py
# ROS2 Humble — Minimal corridor centring + front-triggered landing logic
#
# Publishes:
#   /auto_cmd_vel   (Twist)  — planar in navigate; z = -0.5 in landing
#   /auto_cmd_height(Float32)— maintained while navigating; stopped during landing

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


def ang_wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class Mode:
    NAVIGATE = 0
    DESCEND_OFFSET = 1
    FINAL_LAND = 2


class MinimalExplorer(Node):
    def __init__(self):
        super().__init__("minimal_explorer")

        # -------- Parameters --------
        self.declare_parameter("scan_topic", "/crazyflie/scan")
        self.declare_parameter("odom_topic", "/crazyflie/odom")
        self.declare_parameter("cmd_vel_topic", "/auto_cmd_vel")
        self.declare_parameter("cmd_height_topic", "/auto_cmd_height")

        self.declare_parameter("default_height", 0.5)     # m while navigating
        self.declare_parameter("corridor_v", 0.3)         # m/s forward speed
        self.declare_parameter("yaw_k", 1.5)              # P gain for centring
        self.declare_parameter("max_wz", 0.8)             # rad/s clamp
        self.declare_parameter("deadband", 0.02)          # m difference deadband
        self.declare_parameter("front_stop_thresh", 0.5)  # m -> start descent
        self.declare_parameter("descend_offset", 0.05)    # m per tick: target = z - offset
        self.declare_parameter("final_land_z", 0.2)       # m -> switch to z-velocity landing
        self.declare_parameter("final_land_vel_z", -0.5)  # m/s z command indefinitely
        self.declare_parameter("control_rate_hz", 20.0)   # Hz
        self.declare_parameter("side_limit", 2.0)         # m: side lidar "in-range" threshold

        # Fetch params
        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.cmd_height_topic = self.get_parameter("cmd_height_topic").value

        self.default_height = float(self.get_parameter("default_height").value)
        self.v_corridor = float(self.get_parameter("corridor_v").value)
        self.k_yaw = float(self.get_parameter("yaw_k").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.front_stop = float(self.get_parameter("front_stop_thresh").value)
        self.descend_offset = float(self.get_parameter("descend_offset").value)
        self.final_land_z = float(self.get_parameter("final_land_z").value)
        self.final_land_vel_z = float(self.get_parameter("final_land_vel_z").value)
        self.ctrl_hz = float(self.get_parameter("control_rate_hz").value)
        self.side_limit = float(self.get_parameter("side_limit").value)

        # -------- I/O --------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)

        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.height_pub = self.create_publisher(Float32, self.cmd_height_topic, 10)

        # -------- State --------
        self.mode = Mode.NAVIGATE
        self.last_scan = None
        self.have_map = False
        self.idx_front = self.idx_left = self.idx_back = self.idx_right = None

        self.have_odom = False
        self.odom_z = 0.0

        self.timer = self.create_timer(1.0 / self.ctrl_hz, self._tick)
        self.get_logger().info("[MODE] NAVIGATE — centring + forward")

    # ---------- Subscribers ----------
    def _on_scan(self, msg: LaserScan):
        self.last_scan = msg
        if not self.have_map:
            self._compute_scan_mapping(msg)

    def _on_odom(self, msg: Odometry):
        self.odom_z = float(msg.pose.pose.position.z)
        self.have_odom = True

    # ---------- Helpers ----------
    def _compute_scan_mapping(self, scan: LaserScan):
        n = len(scan.ranges)
        if n < 4 or scan.angle_increment == 0.0:
            return

        targets = {
            "front": 0.0,
            "left": math.pi / 2.0,
            "back": math.pi,
            "right": -math.pi / 2.0,
        }

        beams = []
        for i in range(n):
            ang = scan.angle_min + i * scan.angle_increment
            beams.append((i, ang_wrap(ang)))

        assigned, used = {}, set()
        for name, targ in targets.items():
            best_i, best_d = None, 1e9
            for i, ang in beams:
                if i in used:
                    continue
                d = abs(ang_wrap(ang - targ))
                if d < best_d:
                    best_d, best_i = d, i
            assigned[name] = best_i
            used.add(best_i)

        self.idx_front = assigned["front"]
        self.idx_left = assigned["left"]
        self.idx_back = assigned["back"]
        self.idx_right = assigned["right"]
        self.have_map = True

        self.get_logger().info(
            f"Scan mapping: front={self.idx_front}, left={self.idx_left}, back={self.idx_back}, right={self.idx_right}"
        )

    def _ranges(self):
        if self.last_scan is None or not self.have_map:
            return None
        r = self.last_scan.ranges
        try:
            return (r[self.idx_front], r[self.idx_left], r[self.idx_back], r[self.idx_right])
        except Exception as e:
            self.get_logger().warn(f"Scan indexing issue: {e}")
            return None

    def _send_vel(self, vx=0.0, vy=0.0, wz=0.0, vz=0.0):
        # guard against nan/inf
        vals = {"vx": vx, "vy": vy, "wz": wz, "vz": vz}
        for k, v in vals.items():
            if not math.isfinite(v):
                self.get_logger().warn(f"Invalid velocity ({k}={v}); publishing 0 instead.")
                vals[k] = 0.0

        t = Twist()
        t.linear.x = float(vals["vx"])
        t.linear.y = float(vals["vy"])
        t.linear.z = float(vals["vz"])
        t.angular.z = float(vals["wz"])
        self.vel_pub.publish(t)

    def _publish_height(self, h: float):
        msg = Float32()
        msg.data = float(h)
        self.height_pub.publish(msg)

    def _corridor_wz(self, left, right):
        """Centring policy:
           - If both sides <= side_limit: dual-sided P on (right-left)
           - If only one side <= side_limit: single-sided using the available side vs side_limit
           - If neither side <= side_limit: return 0.0 (go straight)
        """
        ls = self.last_scan
        rmax = (float(ls.range_max) if ls and math.isfinite(ls.range_max) else 3.5)

        # Replace non-finite with very large (no return)
        if not math.isfinite(left):
            left = rmax
        if not math.isfinite(right):
            right = rmax

        left_ok = left <= self.side_limit
        right_ok = right <= self.side_limit

        if left_ok and right_ok:
            # Dual-sided: standard difference
            diff = (right - left)
        elif left_ok and not right_ok:
            # Single-sided: pretend right = side_limit
            diff = (1.0 - left)
        elif right_ok and not left_ok:
            # Single-sided: pretend left = side_limit
            diff = (right - 1.0)
        else:
            # Neither side in range: no centring (straight)
            return 0.0

        if abs(diff) < self.deadband:
            diff = 0.0

        wz = -self.k_yaw * diff
        if wz > self.max_wz:
            wz = self.max_wz
        if wz < -self.max_wz:
            wz = -self.max_wz
        return float(wz)

    # ---------- Control loop ----------
    def _tick(self):
        rng = self._ranges()
        if rng is None or not self.have_odom:
            self._send_vel(0.0, 0.0, 0.0, 0.0)
            return

        front, left, _back, right = rng

        if self.mode == Mode.NAVIGATE:
            # Maintain default height while navigating
            self._publish_height(self.default_height)

            # Forward with centring (dual/single/none per side-limit logic)
            wz = self._corridor_wz(left, right)
            self._send_vel(vx=self.v_corridor, vy=0.0, wz=wz, vz=0.0)

            # Front-stop trigger → begin offset descent
            if math.isfinite(front) and (front < self.front_stop):
                self.mode = Mode.DESCEND_OFFSET
                self.get_logger().info("[MODE] DESCEND_OFFSET — hold still, lower height by offset")

        elif self.mode == Mode.DESCEND_OFFSET:
            # Planar vel zero; gently lower height by offset each tick
            self._send_vel(0.0, 0.0, 0.0, 0.0)
            target_h = max(0.0, self.odom_z - self.descend_offset)
            self._publish_height(target_h)

            if self.odom_z <= self.final_land_z:
                self.mode = Mode.FINAL_LAND
                self.get_logger().info("[MODE] FINAL_LAND — stop /auto_cmd_height, command z vel")

        else:  # Mode.FINAL_LAND
            # Stop publishing /auto_cmd_height; command negative z velocity indefinitely
            self._send_vel(0.0, 0.0, 0.0, vz=self.final_land_vel_z)


def main():
    rclpy.init()
    node = MinimalExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send_vel(0.0, 0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
