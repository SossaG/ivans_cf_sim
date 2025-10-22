#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Odometry


class GCVelMux(Node):
    def __init__(self):
        super().__init__('gc_vel_mux')

        # ---------- Params ----------
        self.declare_parameter('manual_topic', '/manual_cmd_vel')
        self.declare_parameter('auto_topic', '/auto_cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('manual_timeout', 3.0)        # seconds without manual before fallback
        self.declare_parameter('publish_rate', 30.0)         # Hz

        self.declare_parameter('odom_topic', '/crazyflie_real/odom')

        # Height topics: make the mux the ONLY publisher on cmd_height_out
        self.declare_parameter('auto_cmd_height_topic', '/auto_cmd_height')  # explorer should publish here (remap)
        self.declare_parameter('cmd_height_out', '/cmd_height')              # final height topic (published by mux only)

        # Height control
        self.declare_parameter('height_direc_topic', '/height_direc')  # Int32: 2=UP,1=NEUTRAL,0=DOWN
        self.declare_parameter('height_offset', 0.02)                  # meters magnitude
        self.declare_parameter('min_height', 0.0)
        self.declare_parameter('max_height', 0.0)

        manual_topic = self.get_parameter('manual_topic').value
        auto_topic = self.get_parameter('auto_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.manual_timeout = float(self.get_parameter('manual_timeout').value)
        publish_rate = float(self.get_parameter('publish_rate').value)

        odom_topic = self.get_parameter('odom_topic').value
        auto_h_in = self.get_parameter('auto_cmd_height_topic').value
        cmd_h_out = self.get_parameter('cmd_height_out').value
        height_direc_topic = self.get_parameter('height_direc_topic').value

        self.height_offset = float(self.get_parameter('height_offset').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)

        # ---------- State ----------
        self.last_manual: Twist = Twist()
        self.last_auto: Twist = Twist()
        self.last_manual_time: Optional[Time] = None
        self.last_auto_time: Optional[Time] = None

        self.last_odom_z: Optional[float] = None
        self.height_state: int = 1  # 2=UP, 1=NEUTRAL, 0=DOWN (default neutral)
        self.last_auto_height: Optional[float] = None  # latest auto height from explorer

        # ---------- IO ----------
        self.sub_manual = self.create_subscription(Twist, manual_topic, self._on_manual, 10)
        self.sub_auto   = self.create_subscription(Twist, auto_topic,   self._on_auto,   10)
        self.sub_odom   = self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.sub_hmode  = self.create_subscription(Int32, height_direc_topic, self._on_height_direc, 10)
        self.sub_auto_h = self.create_subscription(Float32, auto_h_in, self._on_auto_height, 10)

        self.pub_cmd    = self.create_publisher(Twist, output_topic, 10)
        self.pub_h_out  = self.create_publisher(Float32, cmd_h_out,   10)

        # ---------- Timer ----------
        period = 1.0 / max(1e-3, publish_rate)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "gc_vel_mux started:\n"
            f"  manual='{manual_topic}', auto='{auto_topic}', out='{output_topic}'\n"
            f"  odom='{odom_topic}', height_in(auto)='{auto_h_in}', height_out='{cmd_h_out}'\n"
            f"  height_direc='{height_direc_topic}', offset={self.height_offset} m\n"
            f"  manual_timeout={self.manual_timeout}s, rate={publish_rate}Hz\n"
            f"  clamp: min={self.min_height}, max={self.max_height} (disabled if max<=min)"
        )

    # ---------- Callbacks ----------
    def _on_manual(self, msg: Twist):
        self.last_manual = msg
        self.last_manual_time = self.get_clock().now()

    def _on_auto(self, msg: Twist):
        self.last_auto = msg
        self.last_auto_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        try:
            self.last_odom_z = float(msg.pose.pose.position.z)
        except Exception:
            self.last_odom_z = None

    def _on_height_direc(self, msg: Int32):
        val = int(msg.data)
        if val not in (0, 1, 2):
            self.get_logger().warn(f"Invalid /height_direc={val}; expected 0/1/2. Ignoring.")
            return
        if val != self.height_state:
            self.get_logger().info({0: "height: DOWN", 1: "height: NEUTRAL", 2: "height: UP"}[val])
        self.height_state = val

    def _on_auto_height(self, msg: Float32):
        self.last_auto_height = float(msg.data)

    # ---------- Helpers ----------
    def _manual_active(self) -> bool:
        if self.last_manual_time is None:
            return False
        elapsed = (self.get_clock().now() - self.last_manual_time).nanoseconds * 1e-9
        return elapsed <= self.manual_timeout

    def _clamp_height(self, h: float) -> float:
        if self.max_height > self.min_height:
            return min(max(h, self.min_height), self.max_height)
        return h

    # ---------- Main loop ----------
    def _on_timer(self):
        out_cmd = Twist()

        if self._manual_active():
            # Forward manual twist
            out_cmd = self.last_manual

            # Height publishing (manual only)
            if self.height_state in (0, 2):
                if self.last_odom_z is not None and not math.isnan(self.last_odom_z):
                    desired_h = self.last_odom_z + (self.height_offset if self.height_state == 2 else -self.height_offset)
                    desired_h = self._clamp_height(desired_h)
                    self.pub_h_out.publish(Float32(data=desired_h))
                else:
                    self.get_logger().warn("No odom z yet; cannot publish height (manual).", throttle_duration_sec=2.0)
            # If NEUTRAL (1): publish nothing to height (as requested)
        else:
            # Auto: forward auto twist if we have one
            if self.last_auto_time is not None:
                out_cmd = self.last_auto
            # Auto height: MUX republishes the explorer's value to the output topic
            if self.last_auto_height is not None:
                self.pub_h_out.publish(Float32(data=self._clamp_height(self.last_auto_height)))

        self.pub_cmd.publish(out_cmd)


def main():
    rclpy.init()
    node = GCVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
