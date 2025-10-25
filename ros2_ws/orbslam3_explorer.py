#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Office corridor explorer for Crazyflie (ROS 2 Humble)
with detailed terminal logging (manual throttle helper for Humble).
"""

import math
from math import atan2, cos, sin, pi

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

# ===========================
# CONFIGURATION
# ===========================
FORWARD_SPEED = 0.30
BACKWARD_SPEED = -0.30
INCR_MOVE_SPEED = 0.10
TURN_SPEED = 0.35
STEP_DEG = 30.0
STEP_RAD = STEP_DEG * pi / 180.0
STEPS_PER_RIGHT_ANGLE = 3
STEP_FWD_DIST = 0.20
YAW_K = 0.50
MAX_YAW = 0.6
CORRIDOR_SIDE_MAX = 3.0
FRONT_BLOCK = 2.0
SIDE_OPEN = 3.0
SCAN_MIN = 0.01
SCAN_MAX = 3.49
INTERSECTION_COOLDOWN = 10.0
TARGET_INTERSECTIONS = 9
DEFAULT_HEIGHT = 0.5
LAND_OFFSET = 0.05
LAND_Z_THRESHOLD = 0.20
CONTROL_RATE = 30.0

STATE_CORRIDOR_FWD = "corridor_forward"
STATE_CORRIDOR_BACK = "corridor_backward"
STATE_TURNING_RIGHT = "turning_right"
STATE_TURNING_LEFT = "turning_left"
STATE_LANDING = "landing"
STATE_FINISHED = "finished"

SUB_TURN_YAW = "turn_yaw_step"
SUB_TURN_FWD = "turn_forward_step"
SUB_TURN_BACK = "turn_backward_step"


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def angle_diff(a, b):
    d = (a - b + pi) % (2.0 * pi) - pi
    return d


class Explorer(Node):
    def __init__(self):
        super().__init__("office_explorer_verbose")
        self.get_logger().info("=== Crazyflie Office Explorer Started ===")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscriptions
        self.create_subscription(Odometry, "/crazyflie/odom", self._odom_cb, 10)
        self.create_subscription(LaserScan, "/crazyflie/scan", self._scan_cb, qos)
        self.create_subscription(LaserScan, "/crazyflie/range_up_scan", self._up_cb, qos)
        self.create_subscription(LaserScan, "/crazyflie/range_down_scan", self._down_cb, qos)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/auto_cmd_vel", 10)
        self.height_pub = self.create_publisher(Float32, "/auto_cmd_height", 10)

        self.have_odom = False
        self.have_scan = False
        self.odom = Odometry()
        self.front = self.left = self.back = self.right = None
        self.up = self.down = None

        self.state = STATE_CORRIDOR_FWD
        self.forward_mode = True
        self.last_intersection_time = 0.0
        self.intersection_count = 0

        # turning state
        self.turn_target_yaw = None
        self.turn_step_index = 0
        self.turn_substate = SUB_TURN_YAW
        self.step_start_pose = None

        # manual throttle cache for logs
        self._last_log = {}

        self.dt = 1.0 / CONTROL_RATE
        self.create_timer(self.dt, self._tick)

        self.get_logger().info(f"Default height: {DEFAULT_HEIGHT:.2f} m")

    # ----------------- Logging helper (replaces info_throttle) -----------------
    def _log_throttle(self, key: str, period_sec: float, msg: str, level: str = "info"):
        now = self.get_clock().now().nanoseconds / 1e9
        last = self._last_log.get(key, 0.0)
        if (now - last) >= period_sec:
            self._last_log[key] = now
            # level: "info", "warn", "error", "debug"
            log = getattr(self.get_logger(), level if level in ("info", "warn", "error", "debug") else "info")
            log(msg)

    # ----------- Subscribers -----------
    def _odom_cb(self, msg):
        self.odom = msg
        self.have_odom = True

    def _scan_cb(self, msg):
        if not msg.ranges or len(msg.ranges) < 4:
            return
        vals = [self._clean_range(r) for r in msg.ranges[0:4]]
        self.front, self.left, self.back, self.right = vals
        self.have_scan = True

    def _up_cb(self, msg):
        if msg.ranges:
            self.up = self._clean_range(msg.ranges[0])

    def _down_cb(self, msg):
        if msg.ranges:
            self.down = self._clean_range(msg.ranges[0])

    def _clean_range(self, r):
        if r is None or math.isinf(r) or math.isnan(r):
            return SCAN_MAX
        return clamp(r, SCAN_MIN, SCAN_MAX)

    # ----------- Helpers -----------
    def _yaw_from_odom(self):
        q = self.odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    def _xy_from_odom(self):
        p = self.odom.pose.pose.position
        return p.x, p.y

    def _publish_cmd(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self.cmd_pub.publish(msg)

    def _publish_height(self, h):
        msg = Float32()
        msg.data = h
        self.height_pub.publish(msg)

    # ----------- Detection helpers -----------
    def _intersection_detected(self):
        dead = self.front < FRONT_BLOCK if self.front is not None else False
        open_side = ((self.left is not None and self.left > SIDE_OPEN) or
                     (self.right is not None and self.right > SIDE_OPEN))
        return dead or open_side

    def _in_corridor_mode(self):
        return (self.left is not None and self.right is not None and
                self.left < CORRIDOR_SIDE_MAX and self.right < CORRIDOR_SIDE_MAX)

    def _priorities(self):
        return ["forward", "right", "left", "back"] if self.forward_mode else ["back", "right", "left", "forward"]

    def _option_blocked(self, option):
        if option == "forward":
            return not (self.front is not None and self.front >= FRONT_BLOCK)
        if option == "back":
            return not (self.back is not None and self.back >= FRONT_BLOCK)
        if option == "right":
            return not (self.right is not None and self.right > SIDE_OPEN)
        if option == "left":
            return not (self.left is not None and self.left > SIDE_OPEN)
        return True

    # ----------- Main Tick Loop -----------
    def _tick(self):
        if not (self.have_odom and self.have_scan):
            self._publish_height(DEFAULT_HEIGHT)
            self._log_throttle("wait_sensors", 5.0, "Waiting for odom and scan data...")
            return

        # Regular status print
        self._log_throttle(
            "status", 1.0,
            f"[STATE={self.state}] LIDARS: F={self.front:.2f} L={self.left:.2f} "
            f"B={self.back:.2f} R={self.right:.2f} | Intersections={self.intersection_count}"
        )

        # Default height
        if self.state not in (STATE_LANDING, STATE_FINISHED):
            self._publish_height(DEFAULT_HEIGHT)

        # LANDING PHASE
        if self.state == STATE_LANDING:
            z = self.odom.pose.pose.position.z
            target_h = max(0.0, z - LAND_OFFSET)
            self._publish_height(target_h)
            self._publish_cmd(0.0, 0.0, 0.0)
            self._log_throttle("landing", 1.0, f"Landing... current z={z:.2f} target={target_h:.2f}")
            if z < LAND_Z_THRESHOLD:
                self.state = STATE_FINISHED
                self.get_logger().info("=== MAPPING FINISHED ===")
            return

        if self.state == STATE_FINISHED:
            self._publish_cmd(0.0, 0.0, 0.0)
            return

        # INTERSECTION HANDLING
        if self._intersection_detected():
            self.get_logger().info("Intersection detected.")
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - self.last_intersection_time) > INTERSECTION_COOLDOWN and self._in_corridor_mode():
                self.intersection_count += 1
                self.last_intersection_time = now
                self.get_logger().info(f"Intersection #{self.intersection_count} registered.")

                if self.intersection_count >= TARGET_INTERSECTIONS:
                    self.get_logger().info("Target intersections reached -> initiating landing.")
                    self.state = STATE_LANDING
                    return

                for opt in self._priorities():
                    if not self._option_blocked(opt):
                        self.get_logger().info(f"Chosen direction: {opt}")
                        if opt == "forward":
                            self.state = STATE_CORRIDOR_FWD
                            self.forward_mode = True
                        elif opt == "back":
                            self.state = STATE_CORRIDOR_BACK
                            self.forward_mode = False
                        elif opt == "right":
                            self._begin_turn("right")
                        elif opt == "left":
                            self._begin_turn("left")
                        return

        # STATE ACTIONS
        if self.state == STATE_CORRIDOR_FWD:
            err = (self.right - self.left)
            wz = clamp(YAW_K * err, -MAX_YAW, MAX_YAW)
            self._publish_cmd(FORWARD_SPEED, 0.0, wz)
            self._log_throttle("corridor_fwd", 2.0, f"Forward corridor | yaw corr={wz:+.2f}")

        elif self.state == STATE_CORRIDOR_BACK:
            err = (self.right - self.left)
            wz = clamp(YAW_K * err, -MAX_YAW, MAX_YAW)
            self._publish_cmd(BACKWARD_SPEED, 0.0, wz)
            self._log_throttle("corridor_back", 2.0, f"Backward corridor | yaw corr={wz:+.2f}")

        elif self.state in (STATE_TURNING_LEFT, STATE_TURNING_RIGHT):
            self._do_turn_tick(self.state == STATE_TURNING_RIGHT)

    # ----------- Turning Helpers -----------
    def _begin_turn(self, direction):
        self.turn_step_index = 0
        self.turn_substate = SUB_TURN_YAW
        yaw = self._yaw_from_odom()
        delta = STEP_RAD if direction == "right" else -STEP_RAD
        self.turn_target_yaw = yaw + delta
        self.state = STATE_TURNING_RIGHT if direction == "right" else STATE_TURNING_LEFT
        self.get_logger().info(f"Begin incremental {direction} turn sequence.")

    def _do_turn_tick(self, turning_right):
        sign = 1.0 if turning_right else -1.0

        if self.turn_substate == SUB_TURN_YAW:
            yaw = self._yaw_from_odom()
            err = angle_diff(self.turn_target_yaw, yaw)
            if abs(err) > 0.02:
                self._publish_cmd(0.0, 0.0, sign * TURN_SPEED)
                self._log_throttle("turn_yaw", 0.5, f"Yaw step {self.turn_step_index+1}/3 | err={err:.3f}")
                return
            self.turn_substate = SUB_TURN_FWD
            self.step_start_pose = (*self._xy_from_odom(), yaw)
            self.get_logger().info("Yaw step done → moving forward 0.2 m.")

        elif self.turn_substate == SUB_TURN_FWD:
            if self._advance_along_body_x(+STEP_FWD_DIST, INCR_MOVE_SPEED):
                self.turn_substate = SUB_TURN_BACK
                self.step_start_pose = (*self._xy_from_odom(), self._yaw_from_odom())
                self.get_logger().info("Forward step done → moving backward.")

        elif self.turn_substate == SUB_TURN_BACK:
            if self._advance_along_body_x(-STEP_FWD_DIST, -INCR_MOVE_SPEED):
                self.turn_step_index += 1
                if self.turn_step_index >= STEPS_PER_RIGHT_ANGLE:
                    self.get_logger().info("Completed full 90° incremental turn.")
                    self.state = STATE_CORRIDOR_FWD
                    self.forward_mode = True
                else:
                    yaw_now = self._yaw_from_odom()
                    delta = STEP_RAD if turning_right else -STEP_RAD
                    self.turn_target_yaw = yaw_now + delta
                    self.turn_substate = SUB_TURN_YAW
                    self.step_start_pose = None
                    self.get_logger().info(f"Next yaw target set (step {self.turn_step_index+1}/3).")

    def _advance_along_body_x(self, target_dist, vx):
        if self.step_start_pose is None:
            self.step_start_pose = (*self._xy_from_odom(), self._yaw_from_odom())

        x0, y0, yaw0 = self.step_start_pose
        x, y = self._xy_from_odom()
        dx, dy = x - x0, y - y0
        traveled = dx * cos(yaw0) + dy * sin(yaw0)

        if abs(target_dist - traveled) > 0.02:
            self._publish_cmd(vx, 0.0, 0.0)
            return False

        self._publish_cmd(0.0, 0.0, 0.0)
        return True


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down explorer...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
