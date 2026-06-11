#!/usr/bin/env python3
"""
速度规划节点(10Hz)
订阅 LocalPlanningPath、PlanningObstacleArray、adjusted_initialpose、Odometry/Imu
发布 PlanningSpeedProfile、STGraph（stamps_display / stamps_qp / stamps_dp）、nav_msgs/Path（stop_line、dp_st_curve）
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from typing import DefaultDict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.parameter import Parameter

from carla_msgs.msg import CarlaEgoVehicleStatus
from tf_transformations import quaternion_matrix

try:
    from planning.msg import (
        LocalPlanningPath,
        PlanningObstacle,
        PlanningObstacleArray,
        PlanningSpeedProfile,
        PlanningSpeedPoint,
        STObstacleStamp,
        STGraph,
    )
except ImportError:
    LocalPlanningPath = None
    PlanningObstacle = None
    PlanningObstacleArray = None
    PlanningSpeedProfile = None
    PlanningSpeedPoint = None
    STObstacleStamp = None
    STGraph = None

from modified_EM_planner import math_utils
from modified_EM_planner.planning_params_loader import planning_section
from modified_EM_planner.st_dp_planner import run_dp_speed_plan
from modified_EM_planner.st_qp_planner import solve_longitudinal_speed_qp

K_ZERO_VAL = 1e-6
ST_STOP_LINE_S_EPS = 1e-3


def _world_linear_accel_to_body_xyz(
    ax_w: float,
    ay_w: float,
    az_w: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Tuple[float, float, float]:
    """世界系线加速度 → 车体系：a_body = R_wb^T @ a_world（R_wb 由 Odometry.pose.orientation，child→parent）。"""
    R_wb = quaternion_matrix([qx, qy, qz, qw])[:3, :3]
    aw = np.array([ax_w, ay_w, az_w], dtype=np.float64)
    ab = R_wb.T @ aw
    return float(ab[0]), float(ab[1]), float(ab[2])


def _intersect_seg_vertical(
    p1: Tuple[float, float], p2: Tuple[float, float], s_fix: float
) -> Tuple[float, float]:
    s1, l1 = p1
    s2, l2 = p2
    if abs(s2 - s1) < K_ZERO_VAL:
        return (s_fix, l1)
    t = (s_fix - s1) / (s2 - s1)
    t = max(0.0, min(1.0, t))
    return (s_fix, l1 + t * (l2 - l1))


def _intersect_seg_horizontal(
    p1: Tuple[float, float], p2: Tuple[float, float], l_fix: float
) -> Tuple[float, float]:
    s1, l1 = p1
    s2, l2 = p2
    if abs(l2 - l1) < K_ZERO_VAL:
        return (s1, l_fix)
    t = (l_fix - l1) / (l2 - l1)
    t = max(0.0, min(1.0, t))
    return (s1 + t * (s2 - s1), l_fix)


def _clip_polygon_convex_axis_aligned(
    poly_s: np.ndarray, poly_l: np.ndarray, s0: float, s1: float, l0: float, l1: float
) -> List[Tuple[float, float]]:
    """多边形裁剪，获取obs在ego当前规划轨迹内的部分"""
    n = len(poly_s)
    if n < 3:
        return []

    def clip_half(
        inp: List[Tuple[float, float]],
        inside,
        intersect,
    ) -> List[Tuple[float, float]]:
        if not inp:
            return []
        out: List[Tuple[float, float]] = []
        prev = inp[-1]
        for cur in inp:
            cin, pin = inside(cur), inside(prev)
            if cin:
                if not pin:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif pin:
                out.append(intersect(prev, cur))
            prev = cur
        return out

    poly = [(float(poly_s[i]), float(poly_l[i])) for i in range(n)]
    poly = clip_half(poly, lambda p: p[0] >= s0 - K_ZERO_VAL, lambda a, b: _intersect_seg_vertical(a, b, s0))
    poly = clip_half(poly, lambda p: p[0] <= s1 + K_ZERO_VAL, lambda a, b: _intersect_seg_vertical(a, b, s1))
    poly = clip_half(poly, lambda p: p[1] >= l0 - K_ZERO_VAL, lambda a, b: _intersect_seg_horizontal(a, b, l0))
    poly = clip_half(poly, lambda p: p[1] <= l1 + K_ZERO_VAL, lambda a, b: _intersect_seg_horizontal(a, b, l1))
    return poly


def _s_intervals_from_sl_polygon(
    boundary_s: Sequence[float], boundary_l: Sequence[float], s0: float, s1: float, l0: float, l1: float
) -> List[Tuple[float, float]]:
    """与走廊矩形求交后，按 s 连通分量划分区间（凸四边形通常一段）。"""
    if len(boundary_s) != len(boundary_l) or len(boundary_s) < 3:
        return []
    ps = np.array(boundary_s, dtype=float)
    pl = np.array(boundary_l, dtype=float)
    clipped = _clip_polygon_convex_axis_aligned(ps, pl, s0, s1, l0, l1)
    if len(clipped) < 3:
        return []
    ss = [p[0] for p in clipped]
    return [(min(ss), max(ss))]


def _xy_to_sl_global(
    xy: np.ndarray, ref_xy: np.ndarray, acc_s: np.ndarray
) -> Optional[Tuple[float, float]]:
    return math_utils.project_xy_to_sl_polyline(
        np.asarray(xy, dtype=float),
        np.asarray(ref_xy, dtype=float),
        np.asarray(acc_s, dtype=float),
        None,
        None,
        eps=K_ZERO_VAL,
    )


def _sl_boundary_for_box_corners(
    corners_xy: np.ndarray, ref_xy: np.ndarray, acc_s: np.ndarray, use_edge_midpoints: bool
) -> Optional[dict]:
    if len(corners_xy) != 4:
        return None
    ego_xy = ref_xy[0]
    obs = corners_xy.copy()
    fi = 0
    md = float("inf")
    for i in range(4):
        d = float(np.linalg.norm(obs[i] - ego_xy))
        if d < md:
            md = d
            fi = i
    if fi != 0:
        obs = np.roll(obs, -fi, axis=0)
    sl_corners: List[Tuple[float, float]] = []
    r0 = _xy_to_sl_global(obs[0], ref_xy, acc_s)
    if r0 is None:
        return None
    sl_corners.append(r0)
    prev_s = r0[0]
    for i in range(1, 4):
        p = obs[i]
        prev_p = obs[i - 1]
        dist = float(np.linalg.norm(p - prev_p))
        s_start = prev_s - 2.0 * dist
        s_end = prev_s + 2.0 * dist
        r = _xy_to_sl_window(p, ref_xy, acc_s, s_start, s_end)
        if r is None:
            return None
        sl_corners.append(r)
        prev_s = r[0]
    bs: List[float] = []
    bl: List[float] = []
    if not use_edge_midpoints:
        for s, l in sl_corners:
            bs.append(s)
            bl.append(l)
    else:
        for i in range(4):
            i0 = i
            i1 = (i + 1) % 4
            s0, l0 = sl_corners[i0]
            s1, l1 = sl_corners[i1]
            p0 = obs[i0]
            p1 = obs[i1]
            pm = (p0 + p1) * 0.5
            dist = float(np.linalg.norm(p0 - pm))
            ws = s0 - 2.0 * dist
            we = s0 + 2.0 * dist
            mid_sl = _xy_to_sl_window(pm, ref_xy, acc_s, ws, we)
            if mid_sl is None:
                continue
            v0 = np.array([s1 - s0, l1 - l0])
            v1 = np.array([mid_sl[0] - s0, mid_sl[1] - l0])
            cross = v0[0] * v1[1] - v0[1] * v1[0]
            bs.append(s0)
            bl.append(l0)
            if cross < K_ZERO_VAL:
                bs.append(mid_sl[0])
                bl.append(mid_sl[1])
    if len(bs) < 3:
        return None
    return {
        "boundary_s": bs,
        "boundary_l": bl,
        "start_s": min(bs),
        "end_s": max(bs),
        "start_l": min(bl),
        "end_l": max(bl),
    }


def _xy_to_sl_window(
    xy: np.ndarray, ref_xy: np.ndarray, acc_s: np.ndarray, s_start: float, s_end: float
) -> Optional[Tuple[float, float]]:
    acc = np.asarray(acc_s, dtype=float)
    if len(acc) < 2:
        return None
    if s_start > s_end:
        s_start, s_end = s_end, s_start
    return math_utils.project_xy_to_sl_polyline(
        np.asarray(xy, dtype=float),
        np.asarray(ref_xy, dtype=float),
        acc,
        float(s_start),
        float(s_end),
        eps=K_ZERO_VAL,
    )


def _box_corners(center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    hl = length / 2.0
    hw = width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ local.T).T + center


def _interpolate_xy_tangent_at_s(
    ref_xy: np.ndarray, acc_s: np.ndarray, s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """折线上弧长 s 处的平面位置与单位切向（前向）。"""
    acc = np.asarray(acc_s, dtype=float)
    xy = np.asarray(ref_xy, dtype=float)
    if len(acc) < 2 or xy.shape[0] < 2:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
    s = float(np.clip(s, acc[0], acc[-1]))
    hi = int(np.searchsorted(acc, s, side="right"))
    i0 = max(0, min(hi - 1, len(acc) - 2))
    i1 = i0 + 1
    s0, s1 = float(acc[i0]), float(acc[i1])
    if abs(s1 - s0) < K_ZERO_VAL:
        t = 0.0
    else:
        t = (s - s0) / (s1 - s0)
    p = (1.0 - t) * xy[i0] + t * xy[i1]
    seg = xy[i1] - xy[i0]
    tn = float(np.linalg.norm(seg))
    if tn < 1e-9:
        tang = np.array([1.0, 0.0])
    else:
        tang = seg / tn
    return p, tang


def _stop_line_corridor_path_from_center_tangent(
    p0: np.ndarray,
    tang: np.ndarray,
    half_corridor: float,
    stamp,
    frame_id: str,
) -> Path:
    msg = Path()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    nx, ny = -float(tang[1]), float(tang[0])
    d = float(half_corridor)
    a = p0 + np.array([-nx * d, -ny * d], dtype=float)
    b = p0 + np.array([nx * d, ny * d], dtype=float)

    def _pose(xy: np.ndarray) -> PoseStamped:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(xy[0])
        ps.pose.position.y = float(xy[1])
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        return ps

    msg.poses.append(_pose(a))
    msg.poses.append(_pose(b))
    return msg


def _build_stop_line_path_msg(
    ref_xy: np.ndarray,
    acc_s: np.ndarray,
    s_line: float,
    half_corridor: float,
    stamp,
    frame_id: str,
) -> Path:
    p0, tang = _interpolate_xy_tangent_at_s(ref_xy, acc_s, s_line)
    return _stop_line_corridor_path_from_center_tangent(p0, tang, half_corridor, stamp, frame_id)


def _interpolate_s_interval_stamps(
    coarse: List[Tuple[float, float, float]],
    dt_fine: float,
    t_horizon: float,
) -> List[Tuple[float, float, float]]:
    if not coarse:
        return []
    coarse = sorted(coarse, key=lambda x: x[0])
    t0_first = coarse[0][0]
    t0_last = coarse[-1][0]
    out: List[Tuple[float, float, float]] = []
    n_f = int(round(t_horizon / dt_fine)) + 1
    for i in range(n_f):
        t_f = min(i * dt_fine, t_horizon)
        if t_f <= t0_first + K_ZERO_VAL:
            out.append((t_f, coarse[0][1], coarse[0][2]))
            continue
        if t_f >= t0_last - K_ZERO_VAL:
            out.append((t_f, coarse[-1][1], coarse[-1][2]))
            continue
        ts = [c[0] for c in coarse]
        i = bisect.bisect_right(ts, t_f) - 1
        i = max(0, min(i, len(coarse) - 2))
        ta, sla, sha = coarse[i]
        tb, slb, shb = coarse[i + 1]
        if abs(tb - ta) < K_ZERO_VAL:
            out.append((t_f, sla, sha))
        else:
            a = (t_f - ta) / (tb - ta)
            out.append((t_f, (1.0 - a) * sla + a * slb, (1.0 - a) * sha + a * shb))
    return out


def _append_st_stamps_constant_interval(
    stamps_display: List[object],
    stamps_qp: List[object],
    stamps_dp: List[object],
    stamp_cls: type,
    obstacle_id: str,
    is_static: bool,
    t_horizon: float,
    qp_dt: float,
    dp_dt: float,
    *,
    s_low: Optional[float] = None,
    s_high: Optional[float] = None,
    obs_dt: Optional[float] = None,
    coarse: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    """
    写入 display / qp / dp 三套 ST 戳。
    - coarse 为 None：按 obs_dt 在 [0, t_horizon] 上均匀采样恒定 [s_low, s_high]。
    - coarse 已给定：用于动态障碍等时变 s 区间（通常已在 obs_dt 上采样）。
    """
    if coarse is None:
        if s_low is None or s_high is None or obs_dt is None:
            raise ValueError(
                "_append_st_stamps_constant_interval: coarse=None 时需 s_low、s_high、obs_dt"
            )
        coarse_list: List[Tuple[float, float, float]] = []
        ti_max = int(round(t_horizon / obs_dt)) + 1
        for ti in range(ti_max):
            t = min(ti * obs_dt, t_horizon)
            coarse_list.append((t, float(s_low), float(s_high)))
        coarse = coarse_list
    else:
        coarse = list(coarse)

    coarse.sort(key=lambda x: x[0])
    for t_c, slo, shi in coarse:
        st = stamp_cls()
        st.obstacle_id = obstacle_id
        st.is_static = is_static
        st.s_low = float(slo)
        st.s_high = float(shi)
        st.t = float(t_c)
        stamps_display.append(st)
    for t_f, slo, shi in _interpolate_s_interval_stamps(coarse, qp_dt, t_horizon):
        stq = stamp_cls()
        stq.obstacle_id = obstacle_id
        stq.is_static = is_static
        stq.s_low = float(slo)
        stq.s_high = float(shi)
        stq.t = float(t_f)
        stamps_qp.append(stq)
    for t_f, slo, shi in _interpolate_s_interval_stamps(coarse, dp_dt, t_horizon):
        stdp = stamp_cls()
        stdp.obstacle_id = obstacle_id
        stdp.is_static = is_static
        stdp.s_low = float(slo)
        stdp.s_high = float(shi)
        stdp.t = float(t_f)
        stamps_dp.append(stdp)


class SpeedPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__(
            'speed_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )
        if LocalPlanningPath is None or PlanningObstacleArray is None:
            self.get_logger().error("planning messages not built; speed_planner exiting")
            raise RuntimeError("msgs missing")

        sp = planning_section('speed_planner')
        self._dp_cfg = planning_section('st_dp')
        self._qp_cfg = planning_section('st_qp')
        self.default_v = float(sp['default_v'])
        self.ego_speed_dt = float(sp['ego_speed_dt'])
        self.ego_speed_t = float(sp['ego_speed_t'])
        self.obs_sample_dt = float(sp['obs_sample_dt'])
        self.dp_st_graph_dt = float(sp['dp_st_graph_dt'])
        self.st_t_horizon = float(sp['st_t_horizon'])
        self.stop_line_offset_back_m = float(sp['stop_line_offset_back_m'])
        self.stop_line_hold_empty_streak = int(sp['stop_line_hold_empty_streak'])
        self.ego_half_width = float(sp['ego_half_width'])
        self.ego_lat_buffer = float(sp['ego_lat_buffer'])
        self.obs_static_speed_thresh_mps = float(sp['obs_static_speed_thresh_mps'])
        self.sensor_threshold_dv = float(sp['sensor_threshold_dv'])
        self.sensor_threshold_da = float(sp['sensor_threshold_da'])

        self._local_path: Optional[LocalPlanningPath] = None
        self._planning_obstacles: Optional[PlanningObstacleArray] = None

        self._pob_callback_since_timer = False
        self._no_new_pob_streak = 0
        self._empty_obstacles_streak = 0
        # map 系下停车线中心（LP 上 s_line 处），obs 空窗期间用投影更新沿新 LP 的 s
        # shape (2, 2)：第 0 行为停车线中心 p0，第 1 行为切向 tang（与 _anchor_p0_from_s_stop 一致）
        self._hold_stop_center_xy: Optional[np.ndarray] = None
        # 仅在 /adjusted_initialpose 后置 True；冷启动兜底成功发线后置 False
        self._cold_start_after_adjust: bool = False
        # 上一帧成功规划并发布到 PlanningSpeedProfile 后，在 t=self.ego_speed_dt 处的 v、a，供本帧 run_dp 的 init_v/init_a
        self._next_dp_init_v: Optional[float] = None
        self._next_dp_init_a: Optional[float] = None
        # 上一周期成功 DP 的 (s_end-s_start)/(t_end-t_start)，供本帧 DP 速度代价；冷启动或不可算时为 None，run_dp 内退化为 v_cruise
        self._dp_v_avg_ref: Optional[float] = None
        # 传感器反馈（与链式 init 比对，超差则保底）
        self._ego_speed_mps: Optional[float] = None
        self._ego_accel_mps2: Optional[float] = None
        self._ego_accel_mps3: Optional[float] = None
        # 与 /carla/ego_vehicle/odometry pose.orientation 同步（车体→map）
        self._ego_odom_q_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

        self.create_subscription(PoseStamped, "/adjusted_initialpose", self._on_adjusted_initialpose, 10)
        self.create_subscription(LocalPlanningPath, "/planning/local_planning_path", self._on_local_path, 10)
        self.create_subscription(
            PlanningObstacleArray, "/planning/planning_obstacles", self._on_planning_obstacles, 10
        )

        self._pub_speed = self.create_publisher(PlanningSpeedProfile, "/planning/speed_profile", 10)
        self._pub_st = self.create_publisher(STGraph, "/planning/st_graph", 10)
        self._pub_stop_line = self.create_publisher(Path, "/planning/stop_line", 10)
        self._pub_dp_st = self.create_publisher(Path, "/planning/dp_st_curve", 10)

        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self._on_odometry, 10)
        self.create_subscription(
            CarlaEgoVehicleStatus,
            "/carla/ego_vehicle/vehicle_status",
            self._on_ego_vehicle_status,
            10,
        )

        self.create_timer(0.1, self._on_timer)
        self.get_logger().info("speed_planner started (10Hz ST + default speed profile)")

    def _on_local_path(self, msg: LocalPlanningPath) -> None:
        self._local_path = msg

    def _on_planning_obstacles(self, msg: PlanningObstacleArray) -> None:
        self._planning_obstacles = msg
        self._pob_callback_since_timer = True

    def _on_odometry(self, msg: Odometry) -> None:
        """map 系线速度；姿态四元数用于将 vehicle_status 世界系加速度旋到车体。"""
        try:
            q = msg.pose.pose.orientation
            self._ego_odom_q_xyzw = (
                float(q.x),
                float(q.y),
                float(q.z),
                float(q.w),
            )
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            vx = float(msg.twist.twist.linear.x)
            # vy = float(msg.twist.twist.linear.y)
        except (AttributeError, TypeError, ValueError):
            return
        self._ego_speed_mps = vx

    def _on_ego_vehicle_status(self, msg: CarlaEgoVehicleStatus) -> None:
        """vehicle_status 线加速度视为世界系，用最近一帧 odom 四元数旋到车体纵/横向。"""
        try:
            ax_w = float(msg.acceleration.linear.x)
            ay_w = float(msg.acceleration.linear.y)
            az_w = float(msg.acceleration.linear.z)
        except (AttributeError, TypeError, ValueError):
            return
        qx, qy, qz, qw = self._ego_odom_q_xyzw
        ax_b, ay_b, _az_b = _world_linear_accel_to_body_xyz(ax_w, ay_w, az_w, qx, qy, qz, qw)
        self._ego_accel_mps2 = ax_b
        self._ego_accel_mps3 = ay_b

    def _dp_init(self) -> Tuple[float, float]:
        """
        默认用上一帧规划在 self.ego_speed_dt 处的 v、a；若与当前传感器差超过阈值则改用传感器（无传感器则仅用链式值）。
        """
        v_plan = 0.0 if self._next_dp_init_v is None else max(0.0, float(self._next_dp_init_v))
        a_plan = 0.0 if self._next_dp_init_a is None else float(self._next_dp_init_a)
        return v_plan, a_plan

    def _on_adjusted_initialpose(self, _msg: PoseStamped) -> None:
        self._local_path = None
        self._planning_obstacles = None
        self._pob_callback_since_timer = False
        self._no_new_pob_streak = 0
        self._empty_obstacles_streak = 0
        self._hold_stop_center_xy = None
        self._cold_start_after_adjust = True
        self._next_dp_init_v = None
        self._next_dp_init_a = None
        self._dp_v_avg_ref = None
        self._ego_speed_mps = None
        self._ego_accel_mps2 = None
        self._ego_accel_mps3 = None
        self._ego_odom_q_xyzw = (0.0, 0.0, 0.0, 1.0)
        empty = Path()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = "map"
        self._pub_stop_line.publish(empty)
        self.get_logger().info(
            "speed_planner: 已清空 LocalPlanningPath / PlanningObstacleArray 缓存（adjusted_initialpose），冷启动标志已置位"
        )

    def _nudge_obstacle_encroaches_corridor(
        self,
        obj,
        ref_xy: np.ndarray,
        acc_s: np.ndarray,
        s0: float,
        s1: float,
        half_w: float,
        lat_buffer: float,
    ) -> bool:
        """
        静态 nudge 侵占：中心线 ±(half_w+lat_buffer) 走廊与障碍 box 是否相交。

        混合策略（LocalPlanningPath 起点切向为行驶方向）：
        - 四角相对 ego（ref_xy[0]）的点乘均 > 0：整框 SL 多边形 + 轴对齐裁剪（可检出边横穿、角在带外）。
        - 否则：仅对「点乘 > 0」的角点做 _xy_to_sl_global，若某角点 |l| 落在走廊带宽内则侵占
          （减轻全车已在后方时前向-only 参考线把 box 挤到 s≈0 的误报）。
        """
        try:
            cx = float(obj.pose.position.x)
            cy = float(obj.pose.position.y)
            q = obj.pose.orientation
            yaw = self._quat_yaw(q.x, q.y, q.z, q.w)
            dims = obj.shape.dimensions
            if len(dims) < 2:
                return False
            length, width = float(dims[0]), float(dims[1])
        except (AttributeError, TypeError, ValueError):
            return False

        corners = _box_corners(np.array([cx, cy], dtype=float), yaw, length, width)
        ego_xy = ref_xy[0]
        tangent = ref_xy[1] - ego_xy
        tn = float(np.linalg.norm(tangent))
        if tn < 1e-9:
            return False
        forward = tangent / tn

        dots = [float(np.dot(forward, corners[i] - ego_xy)) for i in range(4)]
        all_ahead = all(d > K_ZERO_VAL for d in dots)
        l_lo = -(half_w + lat_buffer)
        l_hi = half_w + lat_buffer

        if all_ahead:
            bd = _sl_boundary_for_box_corners(corners, ref_xy, acc_s, use_edge_midpoints=True)
            if bd is None or len(bd["boundary_s"]) < 3:
                return False
            ps = np.array(bd["boundary_s"], dtype=float)
            pl = np.array(bd["boundary_l"], dtype=float)
            clipped = _clip_polygon_convex_axis_aligned(ps, pl, s0, s1, l_lo, l_hi)
            return len(clipped) >= 3

        for i in range(4):
            if dots[i] <= K_ZERO_VAL:
                continue
            sl = _xy_to_sl_global(corners[i], ref_xy, acc_s)
            if sl is None:
                continue
            _s, l_val = sl
            if l_lo - K_ZERO_VAL < l_val < l_hi + K_ZERO_VAL:
                return True
        return False

    def _cold_start_fallback_stop_s_candidates(
        self,
        pob: PlanningObstacleArray,
        ref_xy: np.ndarray,
        acc_s: np.ndarray,
        s0: float,
        s1: float,
        lw: float,
        lat_buf: float,
    ) -> List[float]:
        """
        冷启动且常规静态分支未产生任何 stop 候选时：为每个静态障碍补一个沿 LP 的 s。
        仅当已 encroach 且 bd 有效但 s_line<=0：动态压缩后移量 off=min(OFFSET,max(0,s_lead-eps))。
        """
        out: List[float] = []
        eps = 2*K_ZERO_VAL
        s_end_clip = float(acc_s[-1]) - 1e-3
        if s_end_clip < eps:
            return out
        for po in pob.obstacles:
            if not po.is_static:
                continue
            obj = po.object
            try:
                cx = float(obj.pose.position.x)
                cy = float(obj.pose.position.y)
                q = obj.pose.orientation
                yaw = self._quat_yaw(q.x, q.y, q.z, q.w)
                dims = obj.shape.dimensions
                if len(dims) < 2:
                    continue
                length, width = float(dims[0]), float(dims[1])
            except (AttributeError, TypeError, ValueError):
                continue
            corners = _box_corners(np.array([cx, cy], dtype=float), yaw, length, width)
            if not self._nudge_obstacle_encroaches_corridor(
                obj, ref_xy, acc_s, s0, s1, lw, lat_buf
            ):
                continue
            bd = _sl_boundary_for_box_corners(corners, ref_xy, acc_s, use_edge_midpoints=True)
            if bd is None:
                continue
            s_lead = float(bd["start_s"])
            s_line = s_lead - self.stop_line_offset_back_m
            if s_line > K_ZERO_VAL:
                continue
            off = min(self.stop_line_offset_back_m, max(0.0, s_lead - eps))
            s_fb = max(eps, min(s_lead - off, s_end_clip))
            if s_fb > K_ZERO_VAL:
                out.append(float(s_fb))
        return out

    def _speed_profile_from_dp_resampled(
        self, t_dp: List[float], s_dp: List[float], init_v: float, init_a: float
    ) -> PlanningSpeedProfile:
        """将 DP (t,s) 重采样到 self.ego_speed_dt / self.ego_speed_t；首点 v、a 与传入的 init_v/init_a 对齐。"""
        msg = PlanningSpeedProfile()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        if len(t_dp) < 2:
            return self._default_speed_profile()
        t_arr = np.asarray(t_dp, dtype=float)
        s_arr = np.asarray(s_dp, dtype=float)
        n = int(round(self.ego_speed_t / self.ego_speed_dt)) + 1
        for i in range(n):
            t = min(i * self.ego_speed_dt, self.ego_speed_t)
            s_i = float(np.interp(t, t_arr, s_arr))
            p = PlanningSpeedPoint()
            p.t = float(t)
            p.s = s_i
            if i == 0:
                p.v = float(max(0.0, init_v))
                p.a = float(init_a)
            else:
                t_prev = (i - 1) * self.ego_speed_dt
                s_prev = float(np.interp(t_prev, t_arr, s_arr))
                dtu = max(t - t_prev, 1e-6)
                p.v = (s_i - s_prev) / dtu
                p_prev = msg.points[-1]
                p.a = (p.v - p_prev.v) / dtu
            p.da = 0.0
            msg.points.append(p)
        return msg

    def _speed_profile_from_qp_nodes(
        self,
        t_qp: Sequence[float],
        s_qp: Sequence[float],
        v_qp: Sequence[float],
        a_qp: Sequence[float],
    ) -> PlanningSpeedProfile:
        """由 QP 结点 (s,v,a) 直接构建剖面，与 self.ego_speed_dt / self.ego_speed_t 网格一致。"""
        msg = PlanningSpeedProfile()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        n_out = int(round(self.ego_speed_t / self.ego_speed_dt)) + 1
        n = min(len(t_qp), len(s_qp), len(v_qp), len(a_qp), n_out)
        for i in range(n):
            p = PlanningSpeedPoint()
            p.t = float(min(t_qp[i], self.ego_speed_t))
            p.s = float(s_qp[i])
            p.v = float(max(0.0, v_qp[i]))
            p.a = float(a_qp[i])
            p.da = 0.0
            msg.points.append(p)
        return msg

    def _default_speed_profile(self) -> PlanningSpeedProfile:
        msg = PlanningSpeedProfile()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        n = int(round(self.ego_speed_t / self.ego_speed_dt)) + 1
        for i in range(n):
            t = i * self.ego_speed_dt
            p = PlanningSpeedPoint()
            p.t = float(t)
            p.s = self.default_v * t
            p.v = self.default_v
            p.a = 0.0
            p.da = 0.0
            msg.points.append(p)
        return msg

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()
        st = STGraph()
        st.header.stamp = stamp
        st.header.frame_id = "map"
        st.t_horizon = self.st_t_horizon

        if self._pob_callback_since_timer:
            self._no_new_pob_streak = 0
            self._pob_callback_since_timer = False
        else:
            self._no_new_pob_streak += 1

        def _empty_stop_path() -> Path:
            e = Path()
            e.header.stamp = stamp
            e.header.frame_id = "map"
            return e

        lp = self._local_path
        if lp is None or len(lp.points) < 2:
            self._pub_stop_line.publish(_empty_stop_path())
            self._pub_speed.publish(self._default_speed_profile())
            self._next_dp_init_v = None
            self._next_dp_init_a = None
            empty_dp = Path()
            empty_dp.header.stamp = stamp
            empty_dp.header.frame_id = "map"
            self._pub_dp_st.publish(empty_dp)
            self._pub_st.publish(st)
            return

        ref_xy = np.array([[p.x, p.y] for p in lp.points], dtype=float)
        acc_s = np.array([float(p.s) for p in lp.points], dtype=float)
        kappa_abs_lp = np.abs(
            np.asarray(
                math_utils.compute_path_curvatures_triangle(
                    [(float(p.x), float(p.y)) for p in lp.points]
                ),
                dtype=float,
            )
        )
        lw = self.ego_half_width
        lat_buf = self.ego_lat_buffer
        half_corridor = lw + lat_buf
        corridor_L = float(lp.corridor_length_s) + self.stop_line_offset_back_m
        s0, s1 = 0.0, max(corridor_L, acc_s[-1] + 1e-3)
        l0, l1 = -lw, lw
        frame_id = lp.header.frame_id if lp.header.frame_id else "map"

        stamps_display: List[object] = []
        stamps_qp: List[object] = []
        stamps_dp: List[object] = []
        dynamic_coarse: DefaultDict[str, List[Tuple[float, float, float]]] = defaultdict(list)

        pob = self._planning_obstacles
        if pob is not None and len(pob.obstacles) == 0:
            self._empty_obstacles_streak += 1
        elif pob is not None:
            self._empty_obstacles_streak = 0

        force_clear_hold = (
            self._no_new_pob_streak >= self.stop_line_hold_empty_streak
            or self._empty_obstacles_streak >= self.stop_line_hold_empty_streak
        )
        if force_clear_hold:
            self._hold_stop_center_xy = None

        use_static_for_stop_line = (
            not force_clear_hold
            and pob is not None
            and len(pob.obstacles) > 0
        )

        stop_s_candidates: List[float] = []
        if pob is not None:
            for po in pob.obstacles:
                oid = po.obstacle_id
                obj = po.object

                if po.is_static:
                    if not use_static_for_stop_line:
                        continue
                    if not self._nudge_obstacle_encroaches_corridor(
                        obj, ref_xy, acc_s, s0, s1, lw, lat_buf
                    ):
                        continue
                    try:
                        cx = float(obj.pose.position.x)
                        cy = float(obj.pose.position.y)
                        q = obj.pose.orientation
                        yaw = self._quat_yaw(q.x, q.y, q.z, q.w)
                        dims = obj.shape.dimensions
                        if len(dims) < 2:
                            continue
                        length, width = float(dims[0]), float(dims[1])
                    except (AttributeError, TypeError, ValueError):
                        continue
                    corners = _box_corners(np.array([cx, cy], dtype=float), yaw, length, width)
                    bd = _sl_boundary_for_box_corners(corners, ref_xy, acc_s, use_edge_midpoints=True)
                    if bd is None:
                        continue
                    s_lead = float(bd["start_s"])
                    s_line = s_lead - self.stop_line_offset_back_m
                    if s_line > K_ZERO_VAL:
                        stop_s_candidates.append(s_line)
                        if self._cold_start_after_adjust:
                            self._cold_start_after_adjust = False
                    else:
                        self._hold_stop_center_xy = None
                    continue

                # 动态：匀速直线预测（决策暂不修改轨迹）；低速已在 perception 标为静态
                try:
                    vx = float(obj.twist.linear.x)
                    vy = float(obj.twist.linear.y)
                except (AttributeError, TypeError, ValueError):
                    vx, vy = 0.0, 0.0
                spd = math.hypot(vx, vy)
                if spd < self.obs_static_speed_thresh_mps:
                    continue
                try:
                    cx = float(obj.pose.position.x)
                    cy = float(obj.pose.position.y)
                    q = obj.pose.orientation
                    yaw = self._quat_yaw(q.x, q.y, q.z, q.w)
                    dims = obj.shape.dimensions
                    if len(dims) < 2:
                        continue
                    length, width = float(dims[0]), float(dims[1])
                except (AttributeError, TypeError, ValueError):
                    continue

                for ti in range(int(round(self.st_t_horizon / self.obs_sample_dt)) + 1):
                    t = ti * self.obs_sample_dt
                    ox = cx + math.cos(yaw) * spd * t
                    oy = cy + math.sin(yaw) * spd * t
                    center = np.array([ox, oy], dtype=float)
                    corners = _box_corners(center, yaw, length, width)
                    bd = _sl_boundary_for_box_corners(corners, ref_xy, acc_s, use_edge_midpoints=True)
                    if bd is None:
                        continue
                    intervals = _s_intervals_from_sl_polygon(
                        bd["boundary_s"], bd["boundary_l"], s0, s1, l0, l1
                    )
                    for slo, shi in intervals:
                        if shi - slo < K_ZERO_VAL:
                            continue
                        dynamic_coarse[str(oid)].append((float(t), float(slo), float(shi)))

        # 三套dt：一套用于显示动态obs不可行区域，一套用于DP，一套用于QP
        if STObstacleStamp is not None:
            for oid_key, coarse in sorted(dynamic_coarse.items(), key=lambda kv: kv[0]):
                _append_st_stamps_constant_interval(
                    stamps_display,
                    stamps_qp,
                    stamps_dp,
                    STObstacleStamp,
                    oid_key,
                    False,
                    self.st_t_horizon,
                    self.ego_speed_dt,
                    self.dp_st_graph_dt,
                    coarse=coarse,
                )

        cold_fallback_used = False
        if (
            len(stop_s_candidates) == 0
            and self._cold_start_after_adjust
            and use_static_for_stop_line
            and pob is not None
        ):
            fb_list = self._cold_start_fallback_stop_s_candidates(
                pob, ref_xy, acc_s, s0, s1, lw, lat_buf
            )
            if fb_list:
                stop_s_candidates.extend(fb_list)
                cold_fallback_used = True

        if len(stop_s_candidates) > 0:
            s_stop = min(stop_s_candidates)
        else:
            s_stop = None

        def _append_stop_line_st(s_stop_val: float) -> None:
            if STObstacleStamp is None:
                return
            slo = float(s_stop_val - ST_STOP_LINE_S_EPS)
            shi = float(s_stop_val + ST_STOP_LINE_S_EPS)
            _append_st_stamps_constant_interval(
                stamps_display,
                stamps_qp,
                stamps_dp,
                STObstacleStamp,
                "stop_line",
                True,
                self.st_t_horizon,
                self.ego_speed_dt,
                self.dp_st_graph_dt,
                s_low=slo,
                s_high=shi,
                obs_dt=self.obs_sample_dt,
            )

        def _publish_stop_line_at_s(s_raw: float) -> None:
            s_use = float(np.clip(s_raw, 0.0, s1))
            self._pub_stop_line.publish(
                _build_stop_line_path_msg(
                    ref_xy, acc_s, s_use, half_corridor, stamp, frame_id
                )
            )
            _append_stop_line_st(s_use)

        def _anchor_p0_from_s_stop(s_raw: float) -> np.ndarray:
            """与 Path 绘制一致：s 先 clip 到 [0,s1]，再限制在折线 acc_s 范围内取中心点。"""
            s_use = float(np.clip(s_raw, 0.0, s1))
            s_geom = float(np.clip(s_use, acc_s[0], acc_s[-1]))
            p0, tang = _interpolate_xy_tangent_at_s(ref_xy, acc_s, s_geom)
            return np.asarray([p0, tang], dtype=np.float64).copy()

        if s_stop is not None and s_stop <= s1 + K_ZERO_VAL:
            self._hold_stop_center_xy = _anchor_p0_from_s_stop(float(s_stop))
            _publish_stop_line_at_s(float(s_stop))
            if cold_fallback_used:
                self._cold_start_after_adjust = False
        elif self._hold_stop_center_xy is not None and not force_clear_hold:
            p0_hold = self._hold_stop_center_xy[0]
            vec1 = p0_hold - ref_xy[0]
            vec2 = ref_xy[1] - ref_xy[0]
            dot_of_vec = float(np.dot(vec1, vec2))
            if dot_of_vec > K_ZERO_VAL:
                p0, tang = self._hold_stop_center_xy[0], self._hold_stop_center_xy[1]
                self._pub_stop_line.publish(
                    _stop_line_corridor_path_from_center_tangent(
                        p0, tang, half_corridor, stamp, frame_id
                    )
                )
            else:
                self._pub_stop_line.publish(_empty_stop_path())
            
        else:
            self._pub_stop_line.publish(_empty_stop_path())

        st.stamps_display = stamps_display
        st.stamps_qp = stamps_qp
        st.stamps_dp = stamps_dp
        self._pub_st.publish(st)

        s_max_dp = float(min(s1, float(acc_s[-1])))
        s_cap_dp = max(s_max_dp, 0.5)
        init_v, init_a = self._dp_init()
        dp = self._dp_cfg
        dp_result = run_dp_speed_plan(
            stamps_dp,
            t_horizon=self.st_t_horizon,
            dt=self.dp_st_graph_dt,
            s_max=s_cap_dp,
            ds=None,
            v_cruise=self.default_v,
            init_v=init_v,
            init_a=init_a,
            v_avg_ref=self._dp_v_avg_ref,
            w_exceed=float(dp['w_exceed']),
            w_lower_speed=float(dp['w_lower_speed']),
            w_ref=float(dp['w_ref']),
            w_acc=float(dp['w_acc']),
            w_dacc=float(dp['w_dacc']),
            w_jerk=float(dp['w_jerk']),
            w_soft_obs=float(dp['w_soft_obs']),
            d0_soft_m=float(dp['d0_soft_m']),
            max_accel=float(dp['max_accel']),
            a_cost_min=float(dp['a_cost_min']),
            a_cost_max=float(dp['a_cost_max']),
            v_max_prune=float(dp['v_max_prune']),
            w_progress=float(dp['w_progress']),
            dense_s_points=int(dp['dense_s_points']),
            dense_ds=float(dp['dense_ds']),
            sparse_ds=float(dp['sparse_ds']),
            w_avg=float(dp['w_avg']),
        )
        dp_path = Path()
        dp_path.header.stamp = stamp
        dp_path.header.frame_id = frame_id or "map"
        if dp_result is not None:
            if dp_result.path_hits_obstacle:
                kk = dp_result.path_hit_segment_end_index
                oid = dp_result.path_hit_obstacle_id
                t0 = dp_result.t[kk - 1] if kk >= 1 else float("nan")
                s0 = dp_result.s[kk - 1] if kk >= 1 else float("nan")
                t1 = dp_result.t[kk] if kk >= 0 and kk < len(dp_result.t) else float("nan")
                s1 = dp_result.s[kk] if kk >= 0 and kk < len(dp_result.s) else float("nan")
                self.get_logger().error(
                    "DP path post-check: ST convex quad hit obstacle_id=%r segment_idx %d->%d "
                    "(t0,s0)=(%.6f,%.6f) (t1,s1)=(%.6f,%.6f)"
                    % (oid, kk - 1, kk, t0, s0, t1, s1)
                )
            for ti, si in zip(dp_result.t, dp_result.s):
                ps = PoseStamped()
                ps.header = dp_path.header
                ps.pose.position.x = float(ti)
                ps.pose.position.y = float(si)
                ps.pose.position.z = 0.0
                ps.pose.orientation.w = 1.0
                dp_path.poses.append(ps)
            tdp = dp_result.t
            sdp = dp_result.s
            if len(tdp) >= 2 and len(sdp) >= 2 and len(tdp) == len(sdp):
                dtm = float(tdp[-1]) - float(tdp[0])
                if dtm > 1e-9:
                    v_av = (float(sdp[-1]) - float(sdp[0])) / dtm
                    self._dp_v_avg_ref = float(v_av) if math.isfinite(v_av) else None
                else:
                    self._dp_v_avg_ref = None
            else:
                self._dp_v_avg_ref = None
            qp = self._qp_cfg
            qp_sol = solve_longitudinal_speed_qp(
                stamps_qp,
                np.asarray(dp_result.t, dtype=float),
                np.asarray(dp_result.s, dtype=float),
                acc_s,
                kappa_abs_lp,
                self.ego_speed_dt,
                self.st_t_horizon,
                self.default_v,
                s_cap_dp,
                0.0,
                init_v,
                init_a,
                w_s=float(qp['w_s']),
                w_v=float(qp['w_v']),
                w_a=float(qp['w_a']),
                w_j=float(qp['w_j']),
                init_tol_s=float(qp['init_tol_s']),
                init_tol_v=float(qp['init_tol_v']),
                init_tol_a=float(qp['init_tol_a']),
            )
            if qp_sol is not None:
                t_qp, s_qp, v_qp, a_qp = qp_sol
                sp = self._speed_profile_from_qp_nodes(t_qp, s_qp, v_qp, a_qp)
            else:
                self.get_logger().error("❌❌❌ QP 求解失败，使用 DP 结果 ❌❌❌")
                sp = self._speed_profile_from_dp_resampled(dp_result.t, dp_result.s, init_v, init_a)
            if len(sp.points) >= 2:
                self._next_dp_init_v = float(max(0.0, sp.points[1].v))
                self._next_dp_init_a = float(sp.points[1].a)
                # self.get_logger().info(f"✅ kappa: {kappa_abs_lp[0]:.2f}")
            else:
                self._next_dp_init_v = None
                self._next_dp_init_a = None
            self._pub_speed.publish(sp)
        else:
            self._pub_speed.publish(self._default_speed_profile())
            self._next_dp_init_v = None
            self._next_dp_init_a = None
        self._pub_dp_st.publish(dp_path)

    @staticmethod
    def _quat_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SpeedPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
