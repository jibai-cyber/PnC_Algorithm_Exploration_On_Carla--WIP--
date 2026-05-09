from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]

STOP_LINE_OBSTACLE_ID = "stop_line"

_CROSS_EPS = 1e-9
_SEG_EPS = 1e-9
_EPS = 1e-6


def _edge_cost_velocity_dp(
    v: float,
    v_ref: float,
    v_max_prune: float,
    w_exceed: float,
    w_lower_speed: float,
    w_ref: float,
    inf: float,
) -> float:
    if v < 0.0:
        return inf
    denom = max(float(v_max_prune), 1e-9)
    v_det = (v - float(v_max_prune)) / denom
    ref_line = float(w_ref) * abs(v - float(v_ref))
    if v_det > 0.0:
        return float(w_exceed) * v_det + ref_line
    return -float(w_lower_speed) * v_det + ref_line


def _edge_cost_acceleration_dp(
    a: float,
    a_min: float,
    a_max: float,
    w_acc: float,
    w_dacc: float,
    inf: float,
) -> float:
    if a < float(a_min) or a > float(a_max):
        return inf
    z_lo = min(50.0, max(-50.0, a - float(a_min)))
    z_hi = min(50.0, max(-50.0, -a + float(a_max)))
    denom_lo = 1.0 + math.exp(z_lo)
    denom_hi = 1.0 + math.exp(z_hi)
    wa = float(w_acc)
    wd = float(w_dacc)
    boundary_lo = (wd * wd) * (a * a) / denom_lo
    boundary_hi = (wa * wa) * (a * a) / denom_hi
    soft_pair = boundary_lo + boundary_hi
    if a >= 0.0:
        return wa * (a * a) + soft_pair
    return wd * (a * a) + soft_pair


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _on_segment(a: Vec2, b: Vec2, p: Vec2) -> bool:
    return (
        min(a[0], b[0]) - _SEG_EPS <= p[0] <= max(a[0], b[0]) + _SEG_EPS
        and min(a[1], b[1]) - _SEG_EPS <= p[1] <= max(a[1], b[1]) + _SEG_EPS
    )


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


def _seg_intersect_seg(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    o1 = _orient(x1, y1, x2, y2, x3, y3)
    o2 = _orient(x1, y1, x2, y2, x4, y4)
    o3 = _orient(x3, y3, x4, y4, x1, y1)
    o4 = _orient(x3, y3, x4, y4, x2, y2)

    def on_seg(a: Vec2, b: Vec2, p: Vec2) -> bool:
        return _on_segment(a, b, p)

    if abs(o1) < _CROSS_EPS and on_seg(p1, p2, p3):
        return True
    if abs(o2) < _CROSS_EPS and on_seg(p1, p2, p4):
        return True
    if abs(o3) < _CROSS_EPS and on_seg(p3, p4, p1):
        return True
    if abs(o4) < _CROSS_EPS and on_seg(p3, p4, p2):
        return True

    return (o1 > _CROSS_EPS) != (o2 > _CROSS_EPS) and (o3 > _CROSS_EPS) != (o4 > _CROSS_EPS)


def _point_in_convex_poly(p: Vec2, poly: Sequence[Vec2]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    sign = None
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cr = _cross2(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1])
        if abs(cr) <= _CROSS_EPS:
            if not _on_segment(a, b, p):
                return False
            continue
        if sign is None:
            sign = cr > 0
        elif (cr > 0) != sign:
            return False
    return True


def segment_hits_convex_quad(p0: Vec2, p1: Vec2, quad_ccw: Sequence[Vec2]) -> bool:
    q = list(quad_ccw)
    if len(q) != 4:
        return False
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        if _seg_intersect_seg(p0, p1, a, b):
            return True
    if _point_in_convex_poly(p0, q) or _point_in_convex_poly(p1, q):
        return True
    return False


def _interval_at_t(
    series: List[Tuple[float, float, float]], t_query: float
) -> Optional[Tuple[float, float]]:
    if not series:
        return None
    ts = [r[0] for r in series]
    if t_query <= ts[0] + _SEG_EPS:
        return series[0][1], series[0][2]
    if t_query >= ts[-1] - _SEG_EPS:
        return series[-1][1], series[-1][2]
    i = bisect.bisect_right(ts, t_query) - 1
    i = max(0, min(i, len(series) - 2))
    ta, la, ha = series[i]
    tb, lb, hb = series[i + 1]
    if abs(tb - ta) < _SEG_EPS:
        return la, ha
    a = (t_query - ta) / (tb - ta)
    return (1.0 - a) * la + a * lb, (1.0 - a) * ha + a * hb


def _build_tagged_quads_between(
    by_id: Dict[str, List[Tuple[float, float, float]]],
    t0: float,
    t1: float,
) -> List[Tuple[str, List[Vec2]]]:
    out: List[Tuple[str, List[Vec2]]] = []
    for oid, series in by_id.items():
        if not series:
            continue
        a = _interval_at_t(series, t0)
        b = _interval_at_t(series, t1)
        if a is None or b is None:
            continue
        slo0, shi0 = a
        slo1, shi1 = b
        if slo0 > shi0:
            slo0, shi0 = shi0, slo0
        if slo1 > shi1:
            slo1, shi1 = shi1, slo1
        out.append((oid, [(t0, slo0), (t1, slo1), (t1, shi1), (t0, shi0)]))
    return out


def _build_forbidden_quads_between(
    by_id: Dict[str, List[Tuple[float, float, float]]],
    t0: float,
    t1: float,
) -> List[List[Vec2]]:
    return [quad for _oid, quad in _build_tagged_quads_between(by_id, t0, t1)]


def _point_in_any_hard_interval(
    by_id: Dict[str, List[Tuple[float, float, float]]], t_query: float, s: float
) -> bool:
    for series in by_id.values():
        intr = _interval_at_t(series, t_query)
        if intr is None:
            continue
        slo, shi = intr
        if slo > shi:
            slo, shi = shi, slo
        if slo - _SEG_EPS <= s <= shi + _SEG_EPS:
            return True
    return False


def _soft_pen_longitudinal(s: float, slo: float, shi: float, d0: float) -> float:
    if slo > shi:
        slo, shi = shi, slo
    pen = 0.0
    if s < slo:
        lo_ext = slo - d0
        if s > lo_ext + _SEG_EPS:
            dist = s - lo_ext
            pen = max(pen, dist * dist)
    elif s > shi:
        hi_ext = shi + d0
        if s < hi_ext - _SEG_EPS:
            dist = hi_ext - s
            pen = max(pen, dist * dist)
    return pen


def _node_soft_cost_max(
    by_id: Dict[str, List[Tuple[float, float, float]]],
    t_query: float,
    s: float,
    d0: float,
) -> float:
    best = 0.0
    for oid, series in by_id.items():
        if oid == STOP_LINE_OBSTACLE_ID:
            continue
        intr = _interval_at_t(series, t_query)
        if intr is None:
            continue
        slo, shi = intr
        if slo > shi:
            slo, shi = shi, slo
        if slo - _SEG_EPS <= s <= shi + _SEG_EPS:
            continue
        p = _soft_pen_longitudinal(s, slo, shi, d0)
        if p > best:
            best = p
    return best


def _parse_stamps_dp(
    stamps: Sequence[object],
) -> Tuple[List[float], Dict[str, List[Tuple[float, float, float]]]]:
    by_id: Dict[str, List[Tuple[float, float, float]]] = {}
    times_set: set = set()
    for st in stamps:
        oid = str(getattr(st, "obstacle_id", ""))
        t = float(getattr(st, "t", 0.0))
        slo = float(getattr(st, "s_low", 0.0))
        shi = float(getattr(st, "s_high", 0.0))
        if oid not in by_id:
            by_id[oid] = []
        by_id[oid].append((t, slo, shi))
        times_set.add(t)
    for oid in by_id:
        by_id[oid].sort(key=lambda x: x[0])
    times = sorted(times_set)
    uniq_t: List[float] = []
    for x in times:
        if not uniq_t or abs(x - uniq_t[-1]) > 1e-7:
            uniq_t.append(x)
    return uniq_t, by_id


def build_s_vals_dense_sparse(
    s_max: float,
    dense_n: int = 100,
    dense_ds: float = 0.1,
    sparse_ds: float = 1.0,
) -> np.ndarray:
    if s_max <= 0 or dense_n < 1 or dense_ds <= 0 or sparse_ds <= 0:
        return np.array([0.0], dtype=float)
    out: List[float] = []
    for j in range(dense_n):
        sj = float(j * dense_ds)
        if sj > s_max + 1e-9:
            break
        out.append(sj)
    if not out:
        return np.array([0.0], dtype=float)
    last_d = out[-1]
    s = last_d + sparse_ds
    while s <= s_max + 1e-9:
        out.append(float(s))
        s += sparse_ds
    if out[-1] < s_max - 1e-9:
        out.append(float(s_max))
    arr = np.unique(np.array(out, dtype=float))
    arr.sort()
    return arr


@dataclass
class DPSpeedPlanResult:
    t: List[float]
    s: List[float]
    v: List[float]
    a: List[float]
    path_hits_obstacle: bool = False
    path_hit_segment_end_index: int = -1
    path_hit_obstacle_id: str = ""

def run_dp_speed_plan(
    stamps_dp: Sequence[object],
    t_horizon: float,
    dt: float,
    s_max: float,
    ds: Optional[float] = None,
    v_cruise: float = 5.0,
    init_v: float = 0.0,
    init_a: float = 0.0,
    w_exceed: float = 20.0,
    w_lower_speed: float = 0.0,
    w_ref: float = 0.7,
    w_acc: float = 0.85,
    w_dacc: float = 0.25,
    w_jerk: float = 0.5,
    w_soft_obs: float = 1.0,
    d0_soft_m: float = 1.5,
    max_accel: float = 2.0,
    a_cost_min: float = -4.0,
    a_cost_max: float = 2.0,
    v_max_prune: float = 10.0,
    w_progress: float = 0.05,
    dense_s_points: int = 100,
    dense_ds: float = 0.1,
    sparse_ds: float = 1.0,
) -> Optional[DPSpeedPlanResult]:
    if dt <= 0 or s_max <= 0:
        return None
    if ds is not None and ds <= 0:
        return None

    times, by_id = _parse_stamps_dp(stamps_dp)
    if not times:
        times = [0.0]
        while times[-1] + dt <= t_horizon + 1e-9:
            times.append(times[-1] + dt)
    ts = list(times)
    if not ts:
        return None
    if ts[0] > 1e-9:
        ts = [0.0] + ts
    ts_dedup: List[float] = []
    for x in ts:
        if not ts_dedup or abs(x - ts_dedup[-1]) > 1e-7:
            ts_dedup.append(float(x))
    ts = ts_dedup
    nt = len(ts)
    if nt < 2:
        return None

    if ds is not None:
        s_vals = np.arange(0.0, s_max + 0.5 * ds, ds, dtype=float)
    else:
        s_vals = build_s_vals_dense_sparse(
            s_max, dense_n=dense_s_points, dense_ds=dense_ds, sparse_ds=sparse_ds
        )
    if len(s_vals) == 0:
        return None
    ns = len(s_vals)

    inf = 1e30
    # cost表示从起点到当前点的总代价，prev_j表示从起点到当前点的最优前驱节点索引。
    cost = np.full((nt, ns), inf, dtype=float)
    prev_j = np.full((nt, ns), -1, dtype=np.int32)

    cost[0, 0] = 0.0
    prev_j[0, 0] = 0

    for k in range(1, nt):
        t0, t1 = float(ts[k - 1]), float(ts[k])
        dtk = max(t1 - t0, 1e-9)
        quads = _build_forbidden_quads_between(by_id, t0, t1)
        max_ds = v_max_prune * dtk + 0.5 * max_accel * dtk * dtk
        for j in range(ns):
            sj = float(s_vals[j])
            if _point_in_any_hard_interval(by_id, t1, sj):
                continue
            soft_arrival = w_soft_obs * _node_soft_cost_max(by_id, t1, sj, d0_soft_m)
            best_c = inf
            best_i = -1
            # NOTE: 这里的实现是真正意义上的DP，剪枝是根据上一列的点进行的。也可以从当前列对下一列进行剪枝，
            # 随后只处理并更新下一列剪枝后的点，这样更像是BFS
            # i_hi 到 sj 就行，因为当前不允许倒车
            i_lo = int(np.searchsorted(s_vals, sj - max_ds, side="left"))
            i_hi = int(np.searchsorted(s_vals, sj, side="right"))
            i_lo = max(0, i_lo)
            i_hi = min(ns, i_hi)
            for i in range(i_lo, i_hi):
                if cost[k - 1, i] + _EPS >= inf:
                    continue
                si = float(s_vals[i])
                if _point_in_any_hard_interval(by_id, t0, si):
                    continue
                p0: Vec2 = (t0, si)
                p1: Vec2 = (t1, sj)
                hit = False
                for quad in quads:
                    if segment_hits_convex_quad(p0, p1, quad):
                        hit = True
                        break
                if hit:
                    continue
                pi = -1
                dt0 = 0.0
                if k == 1:
                    v_prev = float(init_v)
                else:
                    pi = int(prev_j[k - 1, i])
                    if pi < 0:
                        continue
                    dt0 = max(float(ts[k - 1]) - float(ts[k - 2]), 1e-9)
                    v_prev = (si - float(s_vals[pi])) / dt0
                v_edge = (sj - si) / dtk
                a_edge = (v_edge - v_prev) / dtk
                c_v = _edge_cost_velocity_dp(
                    v_edge,
                    v_cruise,
                    v_max_prune,
                    w_exceed,
                    w_lower_speed,
                    w_ref,
                    inf,
                )
                c_a = _edge_cost_acceleration_dp(
                    a_edge,
                    a_cost_min,
                    a_cost_max,
                    w_acc,
                    w_dacc,
                    inf,
                )
                if c_v >= 0.99 * inf or c_a >= 0.99 * inf:
                    continue
                c_edge = c_v + c_a
                if k == 1:
                    a_prev = float(init_a)
                else:
                    if k == 2:
                        v_pp = float(init_v)
                    else:
                        pi2 = int(prev_j[k - 2, pi])
                        if pi2 < 0:
                            continue
                        dtm = max(float(ts[k - 2]) - float(ts[k - 3]), 1e-9)
                        v_pp = (float(s_vals[pi]) - float(s_vals[pi2])) / dtm
                    a_prev = (v_prev - v_pp) / dt0
                jerk = (a_edge - a_prev) / dtk
                jerk_cost = w_jerk * jerk * jerk
                c = (
                    cost[k - 1, i]
                    + c_edge
                    + jerk_cost
                    # - w_progress * (sj - si)
                    + w_progress * (s_max - sj)/s_max
                    + soft_arrival
                )
                if c < best_c:
                    best_c = c
                    best_i = i
            if best_i >= 0:
                cost[k, j] = best_c
                prev_j[k, j] = best_i

    last = nt - 1
    feasible_last = cost[last, :] < 1e20
    cand = np.flatnonzero(feasible_last)
    if cand.size == 0:
        return None
    j_end = int(cand[np.argmin(cost[last, cand])])
    if cost[last, j_end] + _EPS >= inf:
        return None

    path_j_rev: List[int] = []
    j = j_end
    for k in range(last, -1, -1):
        path_j_rev.append(j)
        if k == 0:
            break
        j = int(prev_j[k, j])
        if j < 0:
            return None
    path_j = list(reversed(path_j_rev))

    t_out = [float(ts[k]) for k in range(len(path_j))]
    s_out = [float(s_vals[jj]) for jj in path_j]
    n = len(t_out)
    v_out = [0.0] * n
    a_out = [0.0] * n
    if n >= 2:
        v_out[0] = float(max(0.0, init_v))
        for kk in range(1, n):
            v_out[kk] = (s_out[kk] - s_out[kk - 1]) / max(t_out[kk] - t_out[kk - 1], 1e-9)
        for kk in range(1, n):
            a_out[kk] = (v_out[kk] - v_out[kk - 1]) / max(t_out[kk] - t_out[kk - 1], 1e-9)

    path_hits = False
    hit_seg = -1
    hit_oid = ""
    if n >= 2:
        for kk in range(1, n):
            dts = t_out[kk] - t_out[kk - 1]
            if abs(dts) < 1e-9:
                continue
            t0, t1 = t_out[kk - 1], t_out[kk]
            p0: Vec2 = (t0, s_out[kk - 1])
            p1: Vec2 = (t1, s_out[kk])
            for oid, quad in _build_tagged_quads_between(by_id, t0, t1):
                if segment_hits_convex_quad(p0, p1, quad):
                    path_hits = True
                    hit_seg = kk
                    hit_oid = oid
                    break
            if path_hits:
                break

    return DPSpeedPlanResult(
        t=t_out,
        s=s_out,
        v=v_out,
        a=a_out,
        path_hits_obstacle=path_hits,
        path_hit_segment_end_index=hit_seg,
        path_hit_obstacle_id=hit_oid,
    )
