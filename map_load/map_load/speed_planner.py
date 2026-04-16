#!/usr/bin/env python3
"""
速度规划节点(10Hz)
订阅 LocalPlanningPath、PlanningObstacleArray、adjusted_initialpose
发布 PlanningSpeedProfile、STGraph、nav_msgs/Path（stop_line，走廊宽度可视化）
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.parameter import Parameter

try:
    from map_load.msg import (
        LocalPlanningPath,
        PlanningObstacle,
        PlanningObstacleArray,
        PlanningSpeedProfile,
        PlanningSpeedPoint,
        STObstacleRegion,
        STGraph,
    )
except ImportError:
    LocalPlanningPath = None
    PlanningObstacle = None
    PlanningObstacleArray = None
    PlanningSpeedProfile = None
    PlanningSpeedPoint = None
    STObstacleRegion = None
    STGraph = None

from map_load import math_utils

K_ZERO_VAL = 1e-6
OBS_STATIC_SPEED_THRESH_MPS = 0.1
DEFAULT_V = 1.5
EGO_SPEED_DT = 0.1
EGO_SPEED_T = 5.0
OBS_SAMPLE_DT = 0.25
ST_T_HORIZON = 5.0
# 静态障碍侵占走廊时，停车线相对障碍沿 LP 前沿 start_s 的后移距离 (m)
STOP_LINE_OFFSET_BACK_M = 5.0
ST_STOP_LINE_S_EPS = 1e-3  # ST 图中将停车线画成极窄 s 区间（t 全时段）
# 连续满足以下任一条件达此帧数后清空停车线缓存并发布空 Path：无新 PlanningObstacleArray；或 obstacles 个数为 0
STOP_LINE_HOLD_EMPTY_STREAK = 10


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
            if cross < 0.0:
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


class SpeedPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__(
            'speed_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )
        if LocalPlanningPath is None or PlanningObstacleArray is None:
            self.get_logger().error("map_load planning messages not built; speed_planner exiting")
            raise RuntimeError("msgs missing")

        self.declare_parameter("ego_half_width", 0.6)
        # 与 vehicle_perception EGO_LAT_BUFFER 一致：nudge 侵占判定走廊 = ±(ego_half_width + buffer)
        self.declare_parameter("ego_lat_buffer", 0.3)

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

        self.create_subscription(PoseStamped, "/adjusted_initialpose", self._on_adjusted_initialpose, 10)
        self.create_subscription(LocalPlanningPath, "/planning/local_planning_path", self._on_local_path, 10)
        self.create_subscription(
            PlanningObstacleArray, "/planning/planning_obstacles", self._on_planning_obstacles, 10
        )

        self._pub_speed = self.create_publisher(PlanningSpeedProfile, "/planning/speed_profile", 10)
        self._pub_st = self.create_publisher(STGraph, "/planning/st_graph", 10)
        self._pub_stop_line = self.create_publisher(Path, "/planning/stop_line", 10)

        self.create_timer(0.1, self._on_timer)
        self.get_logger().info("speed_planner started (10Hz ST + default speed profile)")

    def _on_local_path(self, msg: LocalPlanningPath) -> None:
        self._local_path = msg

    def _on_planning_obstacles(self, msg: PlanningObstacleArray) -> None:
        self._planning_obstacles = msg
        self._pob_callback_since_timer = True

    def _on_adjusted_initialpose(self, _msg: PoseStamped) -> None:
        self._local_path = None
        self._planning_obstacles = None
        self._pob_callback_since_timer = False
        self._no_new_pob_streak = 0
        self._empty_obstacles_streak = 0
        self._hold_stop_center_xy = None
        self._cold_start_after_adjust = True
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
            s_line = s_lead - STOP_LINE_OFFSET_BACK_M
            if s_line > K_ZERO_VAL:
                continue
            off = min(STOP_LINE_OFFSET_BACK_M, max(0.0, s_lead - eps))
            s_fb = max(eps, min(s_lead - off, s_end_clip))
            if s_fb > K_ZERO_VAL:
                out.append(float(s_fb))
        return out

    def _default_speed_profile(self) -> PlanningSpeedProfile:
        msg = PlanningSpeedProfile()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        n = int(round(EGO_SPEED_T / EGO_SPEED_DT)) + 1
        for i in range(n):
            t = i * EGO_SPEED_DT
            p = PlanningSpeedPoint()
            p.t = float(t)
            p.s = DEFAULT_V * t
            p.v = DEFAULT_V
            p.a = 0.0
            p.da = 0.0
            msg.points.append(p)
        return msg

    def _on_timer(self) -> None:
        self._pub_speed.publish(self._default_speed_profile())
        stamp = self.get_clock().now().to_msg()
        st = STGraph()
        st.header.stamp = stamp
        st.header.frame_id = "map"
        st.t_horizon = ST_T_HORIZON

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
            self._pub_st.publish(st)
            return

        ref_xy = np.array([[p.x, p.y] for p in lp.points], dtype=float)
        acc_s = np.array([float(p.s) for p in lp.points], dtype=float)
        lw = float(self.get_parameter("ego_half_width").value)
        lat_buf = float(self.get_parameter("ego_lat_buffer").value)
        half_corridor = lw + lat_buf
        corridor_L = float(lp.corridor_length_s) + STOP_LINE_OFFSET_BACK_M
        s0, s1 = 0.0, max(corridor_L, acc_s[-1] + 1e-3)
        l0, l1 = -lw, lw
        frame_id = lp.header.frame_id if lp.header.frame_id else "map"

        pob = self._planning_obstacles
        if pob is not None and len(pob.obstacles) == 0:
            self._empty_obstacles_streak += 1
        elif pob is not None:
            self._empty_obstacles_streak = 0

        force_clear_hold = (
            self._no_new_pob_streak >= STOP_LINE_HOLD_EMPTY_STREAK
            or self._empty_obstacles_streak >= STOP_LINE_HOLD_EMPTY_STREAK
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
                    s_line = s_lead - STOP_LINE_OFFSET_BACK_M
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
                if spd < OBS_STATIC_SPEED_THRESH_MPS:
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

                for ti in range(int(round(ST_T_HORIZON / OBS_SAMPLE_DT)) + 1):
                    t = ti * OBS_SAMPLE_DT
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
                        r = STObstacleRegion()
                        r.obstacle_id = f"{oid}_t{t:.2f}"
                        r.is_static = False
                        r.s_low = slo
                        r.s_high = shi
                        r.t_min = t
                        r.t_max = min(t + OBS_SAMPLE_DT, ST_T_HORIZON)
                        st.regions.append(r)

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
            rs = STObstacleRegion()
            rs.obstacle_id = "stop_line"
            rs.is_static = True
            rs.s_low = s_stop_val - ST_STOP_LINE_S_EPS
            rs.s_high = s_stop_val + ST_STOP_LINE_S_EPS
            rs.t_min = 0.0
            rs.t_max = ST_T_HORIZON
            st.regions.append(rs)

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

        self._pub_st.publish(st)

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
