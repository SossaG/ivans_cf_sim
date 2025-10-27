#!/usr/bin/env python3
import math, random, numpy as np
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Range  # kept import in case you flip back later
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 (keeps tf2 buffer happy)


def bresenham3d(p0i, p1i):
    (x0, y0, z0) = map(int, p0i)
    (x1, y1, z1) = map(int, p1i)
    dx = abs(x1 - x0); dy = abs(y1 - y0); dz = abs(z1 - z0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    sz = 1 if z1 >= z0 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x0 != x1:
            x0 += sx
            if p1 >= 0: y0 += sy; p1 -= 2 * dx
            if p2 >= 0: z0 += sz; p2 -= 2 * dx
            p1 += 2 * dy; p2 += 2 * dz
            yield (x0, y0, z0)
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y0 != y1:
            y0 += sy
            if p1 >= 0: x0 += sx; p1 -= 2 * dy
            if p2 >= 0: z0 += sz; p2 -= 2 * dy
            p1 += 2 * dx; p2 += 2 * dz
            yield (x0, y0, z0)
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z0 != z1:
            z0 += sz
            if p1 >= 0: y0 += sy; p1 -= 2 * dz
            if p2 >= 0: x0 += sx; p2 -= 2 * dz
            p1 += 2 * dy; p2 += 2 * dx
            yield (x0, y0, z0)


class CF3DExplorer(Node):
    """
    Minimal 3D explorer with height setpoint output:
      - Rebuilds a local voxel occupancy from /occupied_cells_vis_array
      - Chooses short-range random 3D goals that pass a straight-line free check
      - Applies simple repulsive 'nudge' from nearby occupied voxels
      - Publishes /cmd_vel (x,y only) and /cmd_height (Float32) for z control

    Updates per request:
      - Uses /vertscan (Float32MultiArray): data[0]=up, data[1]=down
      - No global Z restriction; Z is free (local sampling), safety handled via vertical repulsive nudges
      - Bias sampling/scoring to prioritise XY coverage
      - Never publishes non-zero z velocity
    """

    def __init__(self):
        super().__init__('cf3d_explorer3d')

        # --- Frames ---
        self.declare_parameter('world_frame_id', 'map')
        self.declare_parameter('body_frame_id', 'crazyflie/odom')

        # --- IO topics ---
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/crazyflie/odom')
        self.declare_parameter('markers_topic', '/occupied_cells_vis_array')
        self.declare_parameter('height_cmd_topic', '/auto_cmd_height')

        # New combined vertical scan (up,down)
        self.declare_parameter('vertscan_topic', '/vertscan')

        # For marker & TF alignment: default both to world (or map for sim) so no TF is needed
        self.declare_parameter('frame_id', 'map')   # marker frame
        self.declare_parameter('odom_frame', 'map')

        # --- Behaviour params ---
        self.declare_parameter('resolution', 0.05)      # match octomap res
        self.declare_parameter('inflate_radius', 0.3)   # match voxelizer inflate
        self.declare_parameter('goal_radius', 0.15)     # arrival threshold
        self.declare_parameter('cruise_speed', 0.1)     # xy speed cap (m/s)
        self.declare_parameter('kp', 0.35)              # proportional gain to goal
        self.declare_parameter('sample_radius_xy', 0.8)
        self.declare_parameter('sample_radius_z', 0.3)
        self.declare_parameter('repulse_radius', 0.2)
        self.declare_parameter('repulse_gain', 1.25)
        self.declare_parameter('loop_rate_hz', 30.0)

        # XY-coverage bias (higher = more lateral spread, less vertical bouncing)
        self.declare_parameter('xy_coverage_bias_xy', 1.0)   # weight on XY distance
        self.declare_parameter('xy_coverage_bias_z_penalty', 0.35)  # penalty * |dz|

        # Vertical clearances near floor/ceiling (repulsive nudges)
        self.declare_parameter('clearance_floor_m', 0.20)
        self.declare_parameter('clearance_ceiling_m', 0.20)

        # Range freshness
        self.declare_parameter('range_fresh_timeout_s', 0.6)

        # --- Read params ---
        self.world_frame = self.get_parameter('world_frame_id').value
        self.body_frame  = self.get_parameter('body_frame_id').value

        self.res = float(self.get_parameter('resolution').value)
        self.inflate = float(self.get_parameter('inflate_radius').value)
        self.goal_r = float(self.get_parameter('goal_radius').value)
        self.vmax = float(self.get_parameter('cruise_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.sr_xy = float(self.get_parameter('sample_radius_xy').value)
        self.sr_z = float(self.get_parameter('sample_radius_z').value)
        self.rep_r = float(self.get_parameter('repulse_radius').value)
        self.rep_k = float(self.get_parameter('repulse_gain').value)

        self.xy_w = float(self.get_parameter('xy_coverage_bias_xy').value)
        self.z_pen = float(self.get_parameter('xy_coverage_bias_z_penalty').value)

        self.frame_id = self.get_parameter('frame_id').value or self.world_frame
        self.odom_frame = self.get_parameter('odom_frame').value or self.world_frame

        markers_topic = self.get_parameter('markers_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        height_cmd_topic = self.get_parameter('height_cmd_topic').value
        vertscan_topic = self.get_parameter('vertscan_topic').value

        self.clear_floor = float(self.get_parameter('clearance_floor_m').value)
        self.clear_ceil = float(self.get_parameter('clearance_ceiling_m').value)
        self.range_timeout = float(self.get_parameter('range_fresh_timeout_s').value)

        # --- IO ---
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.height_pub = self.create_publisher(Float32, height_cmd_topic, 10)
        self.sub_markers = self.create_subscription(MarkerArray, markers_topic, self.on_markers, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        # NEW: combined vertical ranges subscriber
        self.sub_vertscan = self.create_subscription(Float32MultiArray, vertscan_topic, self.on_vertscan, 10)

        # --- TF buffer (optional map->odom) ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Grid state ---
        self.occ = None
        self.origin = None
        self.size = None
        self.last_grid_stamp = self.get_clock().now()

        # --- Robot state / goal ---
        self.p_map = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        self.has_odom = False
        self.goal: Optional[np.ndarray] = None

        # Height state (from /vertscan)
        self.last_down = None
        self.last_up = None
        self.last_down_stamp = None
        self.last_up_stamp = None
        self.h_sp = float(self.p_map[2])  # commanded height setpoint

        # Main control loop
        hz = float(self.get_parameter('loop_rate_hz').value)
        self.timer = self.create_timer(1.0 / hz, self.control_step)

        self.get_logger().info(
            f'CF3D Explorer initialised with /cmd_height output. world_frame={self.world_frame}, '
            f'body_frame={self.body_frame}, marker_frame={self.frame_id}, odom_frame={self.odom_frame}'
        )

        # --- Stagnation / exploration helpers ---
        self.hop_every = 5.0  # seconds
        self.last_hop = self.get_clock().now()
        self.last_p = self.p_map.copy()
        self.last_progress_check = self.get_clock().now()
        self.stuck_for = 0.0

    # --------- Subscriptions ----------
    def on_markers(self, marr: MarkerArray):
        pts = []
        res_from_markers = None
        for m in marr.markers:
            if m.type == Marker.CUBE_LIST and m.points:
                res_from_markers = m.scale.x if m.scale.x > 1e-6 else None
                for p in m.points:
                    pts.append((p.x, p.y, p.z))

        if not pts:
            return

        P = np.array(pts, dtype=np.float32)
        res = res_from_markers if res_from_markers else self.res

        inflate = max(self.inflate, 0.0)
        min_xyz = P.min(axis=0) - inflate - res
        max_xyz = P.max(axis=0) + inflate + res
        size_xyz = max_xyz - min_xyz
        nx, ny, nz = np.ceil(size_xyz / res).astype(int) + 1

        occ = np.zeros((nx, ny, nz), dtype=np.uint8)
        origin = min_xyz

        def w2g(xyz):
            return np.floor((xyz - origin) / res).astype(int)

        base_ids = np.unique(w2g(P), axis=0)

        r = max(0, int(math.ceil(inflate / res)))
        if r > 0:
            off = np.array([(i, j, k)
                            for i in range(-r, r + 1)
                            for j in range(-r, r + 1)
                            for k in range(-r, r + 1)
                            if (i*i + j*j + k*k) <= (r*r + 1)], dtype=int)
            for i, j, k in base_ids:
                ii = i + off[:, 0]; jj = j + off[:, 1]; kk = k + off[:, 2]
                mask = (ii >= 0) & (jj >= 0) & (kk >= 0) & (ii < nx) & (jj < ny) & (kk < nz)
                occ[ii[mask], jj[mask], kk[mask]] = 1
        else:
            occ[base_ids[:, 0], base_ids[:, 1], base_ids[:, 2]] = 1

        self.occ = occ
        self.origin = origin
        self.size = (nx, ny, nz)
        self.res = res
        self.last_grid_stamp = self.get_clock().now()

    def on_odom(self, od: Odometry):
        px = od.pose.pose.position.x
        py = od.pose.pose.position.y
        pz = od.pose.pose.position.z

        q = od.pose.pose.orientation
        _ = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.odom_frame != self.world_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, self.odom_frame, rclpy.time.Time())
                tx = tf.transform.translation
                px = px + tx.x
                py = py + tx.y
                pz = pz + tx.z
            except Exception:
                pass

        self.p_map = np.array([px, py, pz], dtype=np.float32)
        self.has_odom = True

    def on_vertscan(self, msg: Float32MultiArray):
        # Expect: data[0] = up range, data[1] = down range
        try:
            if len(msg.data) >= 2:
                self.last_up = float(msg.data[0])   # distance to ceiling
                self.last_down = float(msg.data[1]) # distance to floor
                now = self.get_clock().now()
                self.last_up_stamp = now
                self.last_down_stamp = now
        except Exception as e:
            self.get_logger().warn(f'/vertscan parse error: {e}')

    def occ_density_score(self, g, radius=0.6):
        if self.occ is None: return 0.0
        gi = self.world_to_idx(g)
        if gi is None: return -1e9
        r = max(1, int(radius / self.res))
        nx, ny, nz = self.size
        ix, iy, iz = gi
        i0 = max(0, ix - r); i1 = min(nx - 1, ix + r)
        j0 = max(0, iy - r); j1 = min(ny - 1, iy + r)
        k0 = max(0, iz - r); k1 = min(nz - 1, iz + r)
        occ_count = int(self.occ[i0:i1+1, j0:j1+1, k0:k1+1].sum())
        return -occ_count

    # --------- Helpers ----------
    def world_to_idx(self, xyz: np.ndarray) -> Optional[Tuple[int, int, int]]:
        if self.occ is None:
            return None
        ijk = np.floor((xyz - self.origin) / self.res).astype(int)
        nx, ny, nz = self.size
        if (ijk[0] < 0 or ijk[1] < 0 or ijk[2] < 0 or
            ijk[0] >= nx or ijk[1] >= ny or ijk[2] >= nz):
            return None
        return int(ijk[0]), int(ijk[1]), int(ijk[2])

    def is_free_line(self, a_w: np.ndarray, b_w: np.ndarray) -> bool:
        ai = self.world_to_idx(a_w); bi = self.world_to_idx(b_w)
        if ai is None or bi is None:
            return False
        for i, j, k in bresenham3d(ai, bi):
            if self.occ[i, j, k] == 1:
                return False
        return True

    def sample_goal_near(self) -> Optional[np.ndarray]:
        # With or without a map, sample relative to current pose within (sr_xy, sr_z)
        candidates = []
        for _ in range(40):
            dx = random.uniform(-self.sr_xy, self.sr_xy)
            dy = random.uniform(-self.sr_xy, self.sr_xy)
            dz = random.uniform(-self.sr_z, self.sr_z)
            g = self.p_map + np.array([dx, dy, dz], dtype=np.float32)

            if self.occ is not None:
                # Ensure the goal is inside the known grid and line-free
                if self.world_to_idx(g) is None:  # outside grid; skip
                    continue
                if not self.is_free_line(self.p_map, g):
                    continue
            # Else (no map yet), accept any local sample
            candidates.append(g)

        if not candidates:
            return None

        # Score for XY coverage: reward XY distance, lightly penalise |dz|, plus sparsity pref
        scores = []
        for g in candidates:
            d = g - self.p_map
            far_xy = math.hypot(d[0], d[1])
            dz = abs(float(d[2]))
            sparse = self.occ_density_score(g, radius=0.8) if self.occ is not None else 0.0
            score = self.xy_w * far_xy - self.z_pen * dz + 0.5 * sparse
            scores.append(score)
        return candidates[int(np.argmax(scores))]

    def repulsion(self) -> np.ndarray:
        if self.occ is None:
            return np.zeros(3, dtype=np.float32)
        idx = self.world_to_idx(self.p_map)
        if idx is None:
            return np.zeros(3, dtype=np.float32)

        r_vox = max(1, int(math.ceil(self.rep_r / self.res)))
        nx, ny, nz = self.size
        ix, iy, iz = idx
        acc = np.zeros(3, dtype=np.float32)

        i0 = max(0, ix - r_vox); i1 = min(nx - 1, ix + r_vox)
        j0 = max(0, iy - r_vox); j1 = min(ny - 1, iy + r_vox)
        k0 = max(0, iz - r_vox); k1 = min(nz - 1, iz + r_vox)

        for i in range(i0, i1 + 1):
            x = self.origin[0] + (i + 0.5) * self.res
            for j in range(j0, j1 + 1):
                y = self.origin[1] + (j + 0.5) * self.res
                for k in range(k0, k1 + 1):
                    if self.occ[i, j, k] == 0:
                        continue
                    z = self.origin[2] + (k + 0.5) * self.res
                    d = np.array([self.p_map[0] - x, self.p_map[1] - y, self.p_map[2] - z], dtype=np.float32)
                    dist = np.linalg.norm(d) + 1e-6
                    if dist <= self.rep_r:
                        acc += (d / (dist * dist))
        return acc

    def _range_is_fresh(self, stamp):
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        return age <= self.range_timeout

    # --------- Control loop ----------
    def control_step(self):

        now = self.get_clock().now()
        dt = (now - self.last_progress_check).nanoseconds / 1e9
        if dt > 2.0:
            moved = float(np.linalg.norm(self.p_map - self.last_p))
            if moved < 0.20:
                self.stuck_for += dt
                self.sr_xy = min(self.sr_xy * 1.35, 6.0)
                self.sr_z  = min(self.sr_z  * 1.35, 3.0)
            else:
                self.sr_xy = max(self.sr_xy * 0.9, 1.5)
                self.sr_z  = max(self.sr_z  * 0.9, 0.8)
                self.stuck_for = 0.0
            self.last_p = self.p_map.copy()
            self.last_progress_check = now

        # Hover if no odom yet
        if not self.has_odom:
            self.cmd_pub.publish(Twist())
            self.height_pub.publish(Float32(data=self.h_sp))
            return

        # Refresh / resample goal if needed
        if self.goal is None:
            self.goal = self.sample_goal_near()

        targ = self.goal if self.goal is not None else (self.p_map + np.array([0.0, 0.0, 0.3], dtype=np.float32))

        # Recheck straight line; if blocked, resample
        if self.occ is not None and not self.is_free_line(self.p_map, targ):
            self.goal = self.sample_goal_near()
            if self.goal is not None:
                targ = self.goal

        # Goal reached?
        if self.goal is not None and np.linalg.norm(self.goal - self.p_map) < self.goal_r:
            self.goal = self.sample_goal_near()

        # Attractive velocity to goal (x,y only)
        if self.goal is not None:
            v_des = (self.goal - self.p_map) * self.kp
        else:
            v_des = np.zeros(3, dtype=np.float32)

        v_rep = self.repulsion() * self.rep_k
        v = v_des + v_rep

        # --- Height logic (repulsive nudges, no global Z clamps) ---
        have_down = self._range_is_fresh(self.last_down_stamp)
        have_up = self._range_is_fresh(self.last_up_stamp)

        # Default desire: follow goal's z
        desired_h = float(targ[2])
        out_of_bounds = False

        if have_down and have_up:
            # Enforce ≥ clearances relative to floor and ceiling (repulsive nudge)
            if self.last_down is not None and self.last_down < (self.clear_floor - 0.02):
                # too close to floor → climb
                self.h_sp = float(self.p_map[2] + (self.clear_floor - self.last_down) + 0.05)
                out_of_bounds = True
            elif self.last_up is not None and self.last_up < (self.clear_ceil - 0.02):
                # too close to ceiling → descend
                self.h_sp = float(self.p_map[2] - (self.clear_ceil - self.last_up) - 0.05)
                out_of_bounds = True
            else:
                # within safe clearances: track desired goal z
                self.h_sp = desired_h
        else:
            # No reliable ranges → just track desired goal z (no clamps)
            self.h_sp = desired_h

        # If we are out of bounds vertically, pause x,y exploration to prioritise height recovery
        if out_of_bounds:
            v[0] = 0.0
            v[1] = 0.0

        # Clip x,y speed only (z is handled by height controller)
        v[2] = 0.0
        xy_speed = math.hypot(v[0], v[1])
        if xy_speed > 1e-3:
            scale = min(self.vmax, xy_speed) / xy_speed
            v[0] *= scale
            v[1] *= scale

        cmd = Twist()
        cmd.linear.x = float(v[0])
        cmd.linear.y = float(v[1])
        cmd.linear.z = 0.0  # z control moved to /cmd_height

        self.cmd_pub.publish(cmd)
        self.height_pub.publish(Float32(data=self.h_sp))


def main():
    rclpy.init()
    n = CF3DExplorer()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
