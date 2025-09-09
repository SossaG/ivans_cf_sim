#!/usr/bin/env python3
"""
MultiRanger + Odometry -> rolling 3D point cloud (PointCloud2)

- Subscribes to a 4-beam LaserScan (front/right/back/left) published by the MultiRanger
  simulator on /crazyflie/scan (angles typically [-pi, -pi/2, 0, +pi/2]).
- Optionally fuses separate upward/downward range topics (std_msgs/Float32 or sensor_msgs/Range)
  for vertical rays.
- Integrates over time using /crazyflie/odom pose to place each beam endpoint in world frame.
- Publishes a rolling, voxel-downsampled PointCloud2 in the desired frame (default: odom).

Notes & limitations:
- With 1D lidars, you only get sparse rays. You must move/rotate the drone to paint in 3D.
- Odometry drift will smear points; if you have a /tf map->odom->base chain, switch to tf usage.
- The LaserScan in your echo showed 4 ranges; up/down may be separate. This node can listen to
  optional /range_up and /range_down topics if you wire them in Gazebo.

Tested with ROS 2 Humble APIs.
"""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_msgs.msg import Header



@dataclass
class Pose:
    xyz: np.ndarray  # shape (3,)
    quat: np.ndarray  # (x, y, z, w)


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R


class MultiRangerPointCloudNode(Node):
    def __init__(self):
        super().__init__('multiranger_pointcloud_node')

        # --- Parameters ---
        self.declare_parameter('scan_topic', '/crazyflie/scan')
        self.declare_parameter('odom_topic', '/crazyflie/odom')
        self.declare_parameter('range_up_topic', '/crazyflie/range_up_scan')
        self.declare_parameter('range_down_topic', '/crazyflie/range_down_scan')
        self.declare_parameter('output_cloud_topic', '/crazyflie/pointcloud')
        self.declare_parameter('world_frame_id', 'crazyflie/odom')  # publish points in this frame
        self.declare_parameter('body_frame_id', 'crazyflie/base_footprint')
        self.declare_parameter('voxel_size', 0.05)  # 5 cm
        self.declare_parameter('decay_seconds', 20.0)  # rolling window; set <=0 to accumulate forever
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('max_range_clip', 3.5)  # clip/ignore > this
        self.declare_parameter('min_range_clip', 0.01)
        self.declare_parameter('max_blocks', 20000)  # increase history 
        self.declare_parameter('up_offset',   [0.0, 0.0, 0.05]) # the gz sensor is 5cm up from the actual model sensor
        self.declare_parameter('down_offset', [0.0, 0.0, -0.25]) # "" is 25 cm down, i.e 15cm below the base link of drone



        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.range_up_topic = self.get_parameter('range_up_topic').get_parameter_value().string_value
        self.range_down_topic = self.get_parameter('range_down_topic').get_parameter_value().string_value
        self.output_cloud_topic = self.get_parameter('output_cloud_topic').get_parameter_value().string_value
        self.world_frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        self.body_frame_id = self.get_parameter('body_frame_id').get_parameter_value().string_value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.decay_seconds = float(self.get_parameter('decay_seconds').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.max_range_clip = float(self.get_parameter('max_range_clip').value)
        self.min_range_clip = float(self.get_parameter('min_range_clip').value)
        self.up_offset   = np.array(self.get_parameter('up_offset').value,   dtype=np.float64)
        self.down_offset = np.array(self.get_parameter('down_offset').value, dtype=np.float64)



        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, qos)
        self.sub_up = self.create_subscription(LaserScan, self.range_up_topic, self.on_up_laserscan, qos)
        self.sub_down = self.create_subscription(LaserScan, self.range_down_topic, self.on_down_laserscan, qos)

        self.pub_cloud = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)

        self.last_pose: Optional[Pose] = None
        self.last_up: Optional[float] = None
        self.last_down: Optional[float] = None

        # Rolling store of (stamp_sec_float, Nx3 np.array) blocks
        self.blocks: Deque[Tuple[float, np.ndarray]] = deque(
            maxlen=int(self.get_parameter('max_blocks').value)
        )

        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_cloud)
        self.get_logger().info('MultiRangerPointCloudNode started.')

    # --- Odometry ---
    def on_odom(self, msg: Odometry):
        p = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z], dtype=np.float64)
        q = np.array([msg.pose.pose.orientation.x,
                      msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z,
                      msg.pose.pose.orientation.w], dtype=np.float64)
        self.last_pose = Pose(p, q)

    # --- LaserScan -> points in world frame ---
    def on_scan(self, scan: LaserScan):
        if self.last_pose is None:
            return

        # Build rays in body frame from scan angles (z=0 plane)
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=np.float64)

        # Filter valid ranges
        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]
        if ranges.size == 0:
            return

        ranges = np.clip(ranges, self.min_range_clip, self.max_range_clip)

        # Unit vectors in body frame (x-forward, y-left per REP‑103; adjust if your body frame differs)
        # Here we assume scan frame is co-located/oriented with body frame; if not, extend with a
        # static transform R_scan_to_body.
        dirs_body = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        pts_body = (dirs_body.T * ranges).T  # shape (N,3)


        # Transform to world frame via odom pose
        R = quat_to_rotm(self.last_pose.quat)
        t = self.last_pose.xyz
        pts_world = (R @ pts_body.T).T + t  # (N,3)

        # Append to rolling buffer
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        self.blocks.append((stamp, pts_world))

        # Cull by time window
        if self.decay_seconds > 0:
            cutoff = stamp - self.decay_seconds
            while self.blocks and self.blocks[0][0] < cutoff:
                self.blocks.popleft()

    # add below class header, alongside other methods
    def _add_vertical_point(self, rng: float, dir_body_z: float, stamp_sec: float, offset_body: np.ndarray):
        if self.last_pose is None or not math.isfinite(rng):
            return
        r = float(max(self.min_range_clip, min(self.max_range_clip, rng)))
        # point in the BODY frame = sensor_origin_offset + dir * range
        dir_body = np.array([0.0, 0.0, dir_body_z], dtype=np.float64)
        pt_body = offset_body + dir_body * r

        R = quat_to_rotm(self.last_pose.quat)
        t = self.last_pose.xyz
        pt_world = (R @ pt_body) + t
        self.blocks.append((stamp_sec, pt_world.reshape(1, 3)))

        if self.decay_seconds > 0:
            cutoff = stamp_sec - self.decay_seconds
            while self.blocks and self.blocks[0][0] < cutoff:
                self.blocks.popleft()



    def _first_valid(self, ranges):
        for r in ranges:
            if math.isfinite(r):
                return float(r)
        return None

    def on_up_laserscan(self, scan: LaserScan):
        val = self._first_valid(scan.ranges)
        if val is None:
            return
        self.last_up = max(self.min_range_clip, min(self.max_range_clip, float(val)))
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        self._add_vertical_point(self.last_up, +1.0, stamp, self.up_offset)

    def on_down_laserscan(self, scan: LaserScan):
        val = self._first_valid(scan.ranges)
        if val is None:
            return
        self.last_down = max(self.min_range_clip, min(self.max_range_clip, float(val)))
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        self._add_vertical_point(self.last_down, -1.0, stamp, self.down_offset)




    # --- Publish downsampled cloud ---
    def publish_cloud(self):
        if not self.blocks:
            return
        # Concatenate points
        pts = np.vstack([blk for (_, blk) in self.blocks])  # (M,3)
        if pts.size == 0:
            return

        # Voxel downsample (fast grid hashing)
        if self.voxel_size > 0:
            keys = np.floor(pts / self.voxel_size).astype(np.int64)
            # unique rows & take first occurrence (approx centroid by first)
            # For nicer centroids, you could average per key, but this is faster.
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts = pts[np.sort(idx)]

        # Create PointCloud2 (must use std_msgs/Header)
        now = self.get_clock().now().to_msg()
        hdr = Header()
        hdr.stamp = now
        hdr.frame_id = self.world_frame_id
        cloud_msg = point_cloud2.create_cloud_xyz32(hdr, pts.tolist())
        self.pub_cloud.publish(cloud_msg)



def main(args=None):
    rclpy.init(args=args)
    node = MultiRangerPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
