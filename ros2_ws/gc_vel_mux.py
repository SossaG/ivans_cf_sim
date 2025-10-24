#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

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

        # Kept for launch compatibility, but no longer used (timeout logic removed).
        self.declare_parameter('manual_timeout', 3.0)

        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('odom_topic', '/crazyflie_real/odom')

        # Height topics
        self.declare_parameter('auto_cmd_height_topic', '/auto_cmd_height')
        self.declare_parameter('cmd_height_out', '/cmd_height')

        self.declare_parameter('height_direc_topic', '/height_direc')  # Int32: 2=UP,1=HOLD,0=DOWN
        self.declare_parameter('height_offset', 0.02)                   # meters
        self.declare_parameter('min_height', 0.0)
        self.declare_parameter('max_height', 0.0)

        manual_topic = self.get_parameter('manual_topic').value
        auto_topic = self.get_parameter('auto_topic').value
        output_topic = self.get_parameter('output_topic').value
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

        self.last_odom_z: Optional[float] = None
        self.height_state: int = 1  # 2=UP, 1=HOLD, 0=DOWN
        self.last_auto_height: Optional[float] = None

        # Per-tick activity flags (set by callbacks, cleared each timer tick)
        self._manual_seen_this_tick: bool = False
        self._heightdir_seen_this_tick: bool = False

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
            f"  TIMEOUTS DISABLED: selection is per-tick activity based.\n"
            f"  clamp: min={self.min_height}, max={self.max_height} (disabled if max<=min)"
        )

    # ---------- Callbacks ----------
    def _on_manual(self, msg: Twist):
        self.last_manual = msg
        self._manual_seen_this_tick = True  # indicates manual is actively publishing this cycle

    def _on_auto(self, msg: Twist):
        self.last_auto = msg  # always keep the freshest auto cmd

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
            self.get_logger().info({0: "height: DOWN", 1: "height: HOLD", 2: "height: UP"}[val])
        self.height_state = val
        self._heightdir_seen_this_tick = True  # indicates operator is actively commanding height direction

    def _on_auto_height(self, msg: Float32):
        self.last_auto_height = float(msg.data)

    # ---------- Helpers ----------
    def _clamp_height(self, h: float) -> float:
        if self.max_height > self.min_height:
            return min(max(h, self.min_height), self.max_height)
        return h

    # ---------- Main loop ----------
    def _on_timer(self):
        # Latch and immediately clear the per-tick flags so the decision is based on *this* cycle only.
        manual_active = self._manual_seen_this_tick
        height_active = self._heightdir_seen_this_tick
        self._manual_seen_this_tick = False
        self._heightdir_seen_this_tick = False

        out_cmd = Twist()

        # Rule:
        # - Use AUTO only if NEITHER manual_cmd_vel NOR height_direc published this tick.
        # - Otherwise, use MANUAL:
        #   * /cmd_vel: latest manual this tick if present, else zeros (no stale manual).
        #   * /cmd_height: "offset from odom.z" logic as before.
        if not manual_active and not height_active:
            # ----- AUTO passthrough -----
            out_cmd = self.last_auto  # safe even if default constructed
            if self.last_auto_height is not None:
                self.pub_h_out.publish(Float32(data=self._clamp_height(self.last_auto_height)))
        else:
            # ----- MANUAL selected -----
            # /cmd_vel
            if manual_active:
                out_cmd = self.last_manual
            else:
                # No manual twist this tick, publish zero to avoid stale motion while still letting manual height work.
                out_cmd = Twist()

            # /cmd_height (manual logic)
            if self.last_odom_z is not None and not math.isnan(self.last_odom_z):
                if self.height_state == 2:       # UP
                    desired_h = self.last_odom_z + self.height_offset
                    self.pub_h_out.publish(Float32(data=self._clamp_height(desired_h)))
                elif self.height_state == 0:     # DOWN
                    desired_h = self.last_odom_z - self.height_offset
                    self.pub_h_out.publish(Float32(data=self._clamp_height(desired_h)))
                else:                            # HOLD current height
                    self.pub_h_out.publish(Float32(data=self._clamp_height(self.last_odom_z)))
            else:
                # Only warn occasionally to avoid spam.
                self.get_logger().warn("No odom z yet; cannot publish /cmd_height (manual).", throttle_duration_sec=2.0)

        # Publish the chosen velocity command
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
