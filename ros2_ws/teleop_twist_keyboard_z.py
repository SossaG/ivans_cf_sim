#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import termios
import tty
import select
from math import copysign

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from rcl_interfaces.msg import SetParametersResult

HELP_MSG = """
Reading from the keyboard and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

t : up (+z)   [uses current linear speed]
b : down (-z) [uses current linear speed]

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key or 's' : stop (all zeros)
CTRL-C to quit
---------------------------
Extra (added, independent of Twist) — publishes Int32 on /height_direc:
r : 2  (UP)
f : 1  (NEUTRAL / NONE)
v : 0  (DOWN)
"""

# (x, y, z, yaw) — stock bindings (keep exact behaviour)
moveBindings = {
    'i': (1, 0, 0, 0),
    'o': (1, -1, 0, 1),
    'j': (0, 1, 0, 0),
    'l': (0, -1, 0, 0),
    'u': (1, 1, 0, -1),
    ',': (-1, 0, 0, 0),
    '.': (-1, -1, 0, -1),
    'm': (-1, 1, 0, 1),
    'k': (0, 0, 0, 0),   # legacy stop
    't': (0, 0, 1, 0),   # +z uses current linear speed
    'b': (0, 0, -1, 0),  # -z uses current linear speed
}

speedBindings = {
    'q': (1.1, 1.1),
    'z': (0.9, 0.9),
    'w': (1.1, 1.0),
    'x': (0.9, 1.0),
    'e': (1.0, 1.1),
    'c': (1.0, 0.9),
}

def get_key(timeout=0.1):
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return ''

def save_terminal_settings():
    return termios.tcgetattr(sys.stdin)

def set_cbreak():
    tty.setraw(sys.stdin.fileno())

def restore_terminal_settings(settings):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

class TeleopTwistKeyboardHeightMode(Node):
    def __init__(self):
        super().__init__('teleop_twist_keyboard_heightmode')

        # Parameters (match stock defaults; topics configurable)
        self.declare_parameter('topic', '/manual_cmd_vel')
        self.declare_parameter('repeat_rate_hz', 10.0)
        self.declare_parameter('max_linear', 0.5)
        self.declare_parameter('max_angular', 1.0)
        self.declare_parameter('height_direc_topic', '/height_direc')
        self.declare_parameter('initial_height_state', 1)  # 1 = NEUTRAL on startup

        cmd_topic = self.get_parameter('topic').get_parameter_value().string_value
        height_topic = self.get_parameter('height_direc_topic').get_parameter_value().string_value
        initial_state = int(self.get_parameter('initial_height_state').value)

        self.pub_twist = self.create_publisher(Twist, cmd_topic, 10)
        self.pub_height = self.create_publisher(Int32, height_topic, 10)

        self.repeat_rate = float(self.get_parameter('repeat_rate_hz').value)
        self.speed = float(self.get_parameter('max_linear').value)
        self.turn = float(self.get_parameter('max_angular').value)

        # Targets (continuously re-published like the original)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_th = 0.0

        # Optional mild smoothing
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_th = 0.0
        self.accel = 0.5
        self.turn_accel = 1.0

        self.timer = self.create_timer(1.0 / self.repeat_rate, self._on_timer)
        self.add_on_set_parameters_callback(self._on_reconfigure)

        # Publish initial neutral state once (helps downstream nodes latch)
        self.pub_height.publish(Int32(data=initial_state))
        self.get_logger().info(f"Initial /height_direc = {initial_state}")

        self.get_logger().info(HELP_MSG)
        self._print_speeds()

    def _on_reconfigure(self, params):
        for p in params:
            if p.name == 'topic' and p.value:
                self.get_logger().warn("Restart to change 'topic'.")
            elif p.name == 'height_direc_topic' and p.value:
                self.get_logger().warn("Restart to change 'height_direc_topic'.")
            elif p.name == 'repeat_rate_hz':
                self.repeat_rate = float(p.value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.repeat_rate, self._on_timer)
            elif p.name == 'max_linear':
                self.speed = float(p.value)
                self._print_speeds()
                if self.target_z != 0.0:
                    self.target_z = copysign(self.speed, self.target_z)
            elif p.name == 'max_angular':
                self.turn = float(p.value)
                self._print_speeds()
        return SetParametersResult(successful=True)

    @staticmethod
    def _ramp(curr, target, step):
        if abs(target - curr) < step:
            return target
        return curr + copysign(step, target - curr)

    def _on_timer(self):
        # Smoothly move towards targets and publish
        self.curr_x = self._ramp(self.curr_x, self.target_x, self.accel / self.repeat_rate)
        self.curr_y = self._ramp(self.curr_y, self.target_y, self.accel / self.repeat_rate)
        self.curr_z = self._ramp(self.curr_z, self.target_z, self.accel / self.repeat_rate)
        self.curr_th = self._ramp(self.curr_th, self.target_th, self.turn_accel / self.repeat_rate)

        msg = Twist()
        msg.linear.x = self.curr_x
        msg.linear.y = self.curr_y
        msg.linear.z = self.curr_z
        msg.angular.z = self.curr_th
        self.pub_twist.publish(msg)

    def _publish_height_state(self, value: int):
        self.pub_height.publish(Int32(data=value))
        label = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}.get(value, str(value))
        self.get_logger().info(f"/height_direc: {label} ({value})")

    def handle_key(self, key: str):
        # Speed/turn keys
        if key in speedBindings:
            lin_scale, ang_scale = speedBindings[key]
            self.speed *= lin_scale
            self.turn *= ang_scale
            self._print_speeds()
            if self.target_z != 0.0:
                self.target_z = copysign(self.speed, self.target_z)
            return

        # Stop keys
        if key in (' ', 's', 'k'):
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_z = 0.0
            self.target_th = 0.0
            self._print_speeds(prefix="Stopped. ")
            return

        # Added: 3-state height direction
        if key == 'r':  # UP
            self._publish_height_state(2)
            return
        if key == 'f':  # NEUTRAL / NONE
            self._publish_height_state(1)
            return
        if key == 'v':  # DOWN
            self._publish_height_state(0)
            return

        # Stock movement keys (including t/b)
        if key in moveBindings:
            vx, vy, vz, vth = moveBindings[key]
            self.target_x = vx * self.speed
            self.target_y = vy * self.speed
            self.target_z = vz * self.speed
            self.target_th = vth * self.turn
            return

        # Unknown key: ignore (keeps last Twist)

    def _print_speeds(self, prefix: str = ""):
        self.get_logger().info(
            f"{prefix}current max linear: {self.speed:.3f} m/s, angular: {self.turn:.3f} rad/s"
        )

def main():
    settings = save_terminal_settings()
    rclpy.init()
    node = TeleopTwistKeyboardHeightMode()
    try:
        set_cbreak()
        while rclpy.ok():
            key = get_key(timeout=0.05)
            if key == '\x03':  # CTRL-C
                break
            if key:
                node.handle_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        restore_terminal_settings(settings)
        node.get_logger().info("Exiting teleop...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
