"""
纵向速度 QP：在离散时间上优化 (s, v, a)，跟踪 DP 弧长参考并满足 ST 凸走廊（stamps_qp）；
速度上界仅保留绝对限速，曲率通过代价中线性项惩罚高速。
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

import osqp


V_ABS_MAX = 10.0

# NOTE:不可与 frenet_qp_planner 起点松约束同量级，否则震荡较大，不应牺牲这里的求解精度
DEFAULT_INIT_TOL_S = 1e-6
DEFAULT_INIT_TOL_V = 1e-6
DEFAULT_INIT_TOL_A = 1e-6

# 纵向动力学边界（与 DP 侧常用量级一致）
A_MIN = -4.0
A_MAX = 2.0
J_MIN = -4.0
J_MAX = 4.0

OSQP_LARGE = 1e8


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    iv = sorted((float(min(a, b)), float(max(a, b))) for a, b in intervals)
    out: List[Tuple[float, float]] = []
    lo, hi = iv[0]
    for a, b in iv[1:]:
        if a <= hi + 1e-9:
            hi = max(hi, b)
        else:
            out.append((lo, hi))
            lo, hi = a, b
    out.append((lo, hi))
    return out


def _feasible_intervals_on_segment(
    forbidden_merged: List[Tuple[float, float]], s_lo: float, s_hi: float
) -> List[Tuple[float, float]]:
    sl = float(min(s_lo, s_hi))
    sh = float(max(s_lo, s_hi))
    if sh <= sl + 1e-12:
        return []
    cur = sl
    feas: List[Tuple[float, float]] = []
    for flo, fhi in forbidden_merged:
        if fhi <= sl - 1e-9 or flo >= sh + 1e-9:
            continue
        flo = max(flo, sl)
        fhi = min(fhi, sh)
        if cur < flo - 1e-12:
            feas.append((cur, flo))
        cur = max(cur, fhi)
    if cur < sh - 1e-12:
        feas.append((cur, sh))
    return feas


def _pick_corridor_for_s_ref(
    corridors: List[Tuple[float, float]], s_ref: float
) -> Optional[Tuple[float, float]]:
    if not corridors:
        return None
    for lo, hi in corridors:
        if lo - 1e-6 <= s_ref <= hi + 1e-6:
            return (lo, hi)
    best = corridors[0]
    best_d = float("inf")
    for lo, hi in corridors:
        if s_ref < lo:
            d = lo - s_ref
        elif s_ref > hi:
            d = s_ref - hi
        else:
            d = 0.0
        if d < best_d:
            best_d = d
            best = (lo, hi)
    return best


def _bucket_stamps_by_time_index(
    stamps: Sequence[object], dt: float, n: int
) -> DefaultDict[int, List[Tuple[float, float]]]:
    buckets: DefaultDict[int, List[Tuple[float, float]]] = defaultdict(list)
    for st in stamps:
        try:
            t = float(getattr(st, "t", 0.0))
            slo = float(getattr(st, "s_low", 0.0))
            shi = float(getattr(st, "s_high", 0.0))
        except (TypeError, ValueError):
            continue
        lo, hi = (min(slo, shi), max(slo, shi))
        k = int(round(t / max(dt, 1e-12)))
        k = max(0, min(n - 1, k))
        buckets[k].append((lo, hi))
    return buckets


def _build_corridor_bounds(
    stamps_qp: Sequence[object],
    dt: float,
    t_horizon: float,
    s_cap: float,
    dp_t: np.ndarray,
    dp_s: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    根据DP粗解找到QP求解的凸空间
    """
    n = int(round(t_horizon / max(dt, 1e-12))) + 1
    if n < 2 or s_cap <= 0:
        return None
    buckets = _bucket_stamps_by_time_index(stamps_qp, dt, n)
    s_lo_arr = np.zeros(n, dtype=float)
    s_hi_arr = np.zeros(n, dtype=float)
    for k in range(n):
        t_k = min(k * dt, t_horizon)
        merged = _merge_intervals(buckets.get(k, []))
        feas = _feasible_intervals_on_segment(merged, 0.0, s_cap)
        if not feas:
            return None
        s_ref = float(np.interp(t_k, dp_t, dp_s, left=float(dp_s[0]), right=float(dp_s[-1])))
        picked = _pick_corridor_for_s_ref(feas, s_ref)
        if picked is None:
            return None
        s_lo_arr[k], s_hi_arr[k] = picked[0], picked[1]
        if s_hi_arr[k] <= s_lo_arr[k] + 1e-9:
            return None
    return s_lo_arr, s_hi_arr


def _kappa_abs_at_s(s_query: float, acc_s: np.ndarray, kappa_abs: np.ndarray) -> float:
    if acc_s.size == 0 or kappa_abs.size == 0:
        return 0.0
    sq = float(np.clip(s_query, float(acc_s[0]), float(acc_s[-1])))
    return float(np.interp(sq, acc_s, kappa_abs))


def solve_longitudinal_speed_qp(
    stamps_qp: Sequence[object],
    dp_t: np.ndarray,
    dp_s: np.ndarray,
    acc_s: np.ndarray,
    kappa_abs: np.ndarray,
    dt: float,
    t_horizon: float,
    v_cruise: float,
    s_cap: float,
    init_s: float,
    init_v: float,
    init_a: float,
    w_s: float = 0.5,
    w_v: float = 0.5,
    w_a: float = 0.4,
    w_j: float = 0.32,
    init_tol_s: float = DEFAULT_INIT_TOL_S,
    init_tol_v: float = DEFAULT_INIT_TOL_V,
    init_tol_a: float = DEFAULT_INIT_TOL_A,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    dp_t = np.asarray(dp_t, dtype=float).reshape(-1)
    dp_s = np.asarray(dp_s, dtype=float).reshape(-1)
    if dp_t.size < 2 or dp_s.size != dp_t.size:
        return None
    acc_s = np.asarray(acc_s, dtype=float).reshape(-1)
    kappa_abs = np.asarray(kappa_abs, dtype=float).reshape(-1)
    if acc_s.size != kappa_abs.size:
        return None

    n = int(round(t_horizon / max(dt, 1e-12))) + 1
    if n < 2:
        return None

    cor = _build_corridor_bounds(stamps_qp, dt, t_horizon, s_cap, dp_t, dp_s)
    if cor is None:
        return None
    s_min_b, s_max_b = cor

    t_nodes = np.array([min(i * dt, t_horizon) for i in range(n)], dtype=float)
    s_ref = np.array(
        [float(np.interp(t_nodes[i], dp_t, dp_s, left=float(dp_s[0]), right=float(dp_s[-1]))) for i in range(n)],
        dtype=float,
    )
    p_lin = np.zeros(n, dtype=float)
    for i in range(n):
        ka = _kappa_abs_at_s(float(s_ref[i]), acc_s, kappa_abs)
        p_lin[i] = float(max(ka, 0.0))

    nv = 3 * n
    P = sp.lil_matrix((nv, nv), dtype=float)
    q = np.zeros(nv, dtype=float)

    for i in range(n):
        bi = 3 * i
        if w_s > 0.0:
            P[bi, bi] += 2.0 * w_s
            q[bi] += -2.0 * w_s * float(s_ref[i])
        if w_v > 0.0:
            P[bi + 1, bi + 1] += 2.0 * w_v
            q[bi + 1] += -2.0 * w_v * float(v_cruise)
        # TODO: 曲率惩罚项效果不佳，把曲率打印出来再观察一下
        q[bi + 1] += (np.power(float(p_lin[i]) - 1, 3) + 1.0)*10.0
        if w_a > 0.0:
            P[bi + 2, bi + 2] += 2.0 * w_a
    inv_dt2 = 1.0 / (dt * dt)
    for i in range(n - 1):
        bi = 3 * i + 2
        bj = 3 * (i + 1) + 2
        if w_j > 0.0:
            P[bi, bi] += 2.0 * w_j * inv_dt2
            P[bj, bj] += 2.0 * w_j * inv_dt2
            P[bi, bj] += -2.0 * w_j * inv_dt2
            P[bj, bi] += -2.0 * w_j * inv_dt2

    reg = 1e-8
    for i in range(nv):
        P[i, i] += reg
    P = P.tocsc()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    l_list: list[float] = []
    u_list: list[float] = []

    def add_row(coeffs: List[Tuple[int, float]], lo: float, hi: float) -> None:
        r = len(l_list)
        for c, val in coeffs:
            rows.append(r)
            cols.append(c)
            data.append(val)
        l_list.append(lo)
        u_list.append(hi)

    for i in range(n - 1):
        bi = 3 * i
        d = dt
        d2 = d * d

        add_row(
            [
                (bi + 3, 1.0),
                (bi, -1.0),
                (bi + 1, -d),
                (bi + 2, -d2 / 3.0),
                (bi + 5, -d2 / 6.0),
            ],
            0.0,
            0.0,
        )
        add_row(
            [
                (bi + 4, 1.0),
                (bi + 1, -1.0),
                (bi + 2, -0.5 * d),
                (bi + 5, -0.5 * d),
            ],
            0.0,
            0.0,
        )

    for i in range(n):
        bi = 3 * i
        add_row([(bi, 1.0)], float(s_min_b[i]), float(s_max_b[i]))
        add_row([(bi + 1, 1.0)], 0.0, float(V_ABS_MAX))
        add_row([(bi + 2, 1.0)], float(A_MIN), float(A_MAX))

    for i in range(n - 1):
        bi = 3 * i + 2
        bj = 3 * (i + 1) + 2
        add_row([(bj, 1.0), (bi, -1.0)], float(J_MIN * dt), float(J_MAX * dt))

    b0 = 0
    add_row([(b0, 1.0)], float(init_s) - float(init_tol_s), float(init_s) + float(init_tol_s))
    add_row([(b0 + 1, 1.0)], float(init_v) - float(init_tol_v), float(init_v) + float(init_tol_v))
    add_row([(b0 + 2, 1.0)], float(init_a) - float(init_tol_a), float(init_a) + float(init_tol_a))

    n_con = len(l_list)
    A = sp.csc_matrix((data, (rows, cols)), shape=(n_con, nv))
    l_vec = np.array(l_list, dtype=float)
    u_vec = np.array(u_list, dtype=float)

    l_vec = np.where(np.isnan(l_vec), -OSQP_LARGE, l_vec)
    u_vec = np.where(np.isnan(u_vec), OSQP_LARGE, u_vec)
    l_vec = np.where(l_vec == -np.inf, -OSQP_LARGE, l_vec)
    u_vec = np.where(u_vec == np.inf, OSQP_LARGE, u_vec)
    if np.any(l_vec > u_vec):
        return None

    prob = osqp.OSQP()
    try:
        prob.setup(P=P, q=q, A=A, l=l_vec, u=u_vec, verbose=False, polish=True)
    except Exception:
        return None
    res = prob.solve()
    if res.info.status_val not in (1,):
        return None
    x = res.x
    s_nodes = np.array([x[3 * i] for i in range(n)], dtype=float)
    v_nodes = np.array([x[3 * i + 1] for i in range(n)], dtype=float)
    a_nodes = np.array([x[3 * i + 2] for i in range(n)], dtype=float)
    return t_nodes, s_nodes, v_nodes, a_nodes
