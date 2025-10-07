#!/usr/bin/env python3
"""
MultiRanger (front/right/back/left) + Odometry -> rolling 3D point cloud (PointCloud2)

- Subscribes to a 4-beam LaserScan on /crazyflie_real/scan (angles typically [-pi, -pi/2, 0, +pi/2]).
- Integrates over time using /crazyflie_real/odom to place each beam endpoint in world frame.
- Publishes a rolling, voxel-downsampled PointCloud2.

Notes:
- With 1D lidars, you only get sparse rays. Move/rotate the drone to "paint" the space.
- The LaserScan is assumed co-located with the body frame (x-forward, y-left, z-up per REP-103).
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
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

        # --- Parameters (tune as needed) ---
        self.declare_parameter('scan_topic', '/crazyflie_real/scan')
        self.declare_parameter('odom_topic', '/crazyflie_real/odom')
        self.declare_parameter('output_cloud_topic', '/crazyflie_real/pointcloud')
        self.declare_parameter('world_frame_id', 'world')  # publish points in this frame
        self.declare_parameter('body_frame_id', 'crazyflie_real')
        self.declare_parameter('voxel_size', 0.02)         # metres; 0.02 = 2 cm
        self.declare_parameter('decay_seconds', 10.0)      # rolling window; <=0 to accumulate forever
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('max_range_clip', 3.5)      # metres
        self.declare_parameter('min_range_clip', 0.01)
        self.declare_parameter('max_blocks', 20000)        # history size

        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.output_cloud_topic = self.get_parameter('output_cloud_topic').get_parameter_value().string_value
        self.world_frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        self.body_frame_id = self.get_parameter('body_frame_id').get_parameter_value().string_value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.decay_seconds = float(self.get_parameter('decay_seconds').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.max_range_clip = float(self.get_parameter('max_range_clip').value)
        self.min_range_clip = float(self.get_parameter('min_range_clip').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, qos)
        self.pub_cloud = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)

        self.last_pose: Optional[Pose] = None

        # Rolling store of (stamp_sec_float, Nx3 np.array) blocks
        self.blocks: Deque[Tuple[float, np.ndarray]] = deque(
            maxlen=int(self.get_parameter('max_blocks').value)
        )

        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_cloud)
        self.get_logger().info(f'MultiRangerPointCloudNode started. scan={self.scan_topic}, odom={self.odom_topic}')

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
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=np.float64)

        # Filter valid ranges
        valid = np.isfinite(ranges)
        if not np.any(valid):
            return
        ranges = ranges[valid]
        angles = angles[valid]

        # Clip ranges
        ranges = np.clip(ranges, self.min_range_clip, self.max_range_clip)

        # Unit vectors in body frame (x-forward, y-left, z-up)
        dirs_body = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        pts_body = (dirs_body.T * ranges).T  # (N,3)

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

    # --- Publish downsampled cloud ---
    def publish_cloud(self):
        if not self.blocks:
            return

        # Concatenate points
        pts = np.vstack([blk for (_, blk) in self.blocks])  # (M,3)
        if pts.size == 0:
            return

        # Voxel downsample (grid hashing)
        if self.voxel_size > 0:
            keys = np.floor(pts / self.voxel_size).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts = pts[np.sort(idx)]

        # Create PointCloud2
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
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
