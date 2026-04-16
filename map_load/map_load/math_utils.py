#!/usr/bin/env python3
"""
数学工具模块 - 包含路径曲率计算和坐标转换功能

功能：
1. 计算离散路径点的曲率
2. Cartesian 坐标系到 Frenet 坐标系转换
"""

import numpy as np
from typing import Tuple, List, Optional


def compute_path_profile(xy_points: np.ndarray, point_spacing: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算路径轮廓：航向角、累积弧长、曲率、曲率变化率

    参数：
    xy_points: (n, 2) 路径点数组，每行为 [x, y]
    point_spacing: 路径点间距（米），用于优化累积弧长计算

    返回：
    headings: (n,) 航向角数组（弧度）
    accumulated_s: (n,) 累积弧长数组（米）
    kappas: (n,) 曲率数组（1/米）
    dkappas: (n,) 曲率变化率数组（1/米²）
    """
    n = len(xy_points)
    seg_len_flag = False
    if n < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # 计算 dx 和 dy（使用有限差分）
    dxs = np.zeros(n)
    dys = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            dxs[i] = xy_points[i + 1, 0] - xy_points[i, 0]
            dys[i] = xy_points[i + 1, 1] - xy_points[i, 1]
        elif i == n - 1:
            dxs[i] = xy_points[i, 0] - xy_points[i - 1, 0]
            dys[i] = xy_points[i, 1] - xy_points[i - 1, 1]
        else:
            dxs[i] = 0.5 * (xy_points[i + 1, 0] - xy_points[i - 1, 0])
            dys[i] = 0.5 * (xy_points[i + 1, 1] - xy_points[i - 1, 1])
    
    # 计算航向角
    headings = np.arctan2(dys, dxs)
    
    # 累积弧长：优先使用真实欧氏距离，退化为 point_spacing（当点距过小/退化时）
    accumulated_s = np.zeros(n)
    accumulated_s[0] = 0.0
    for i in range(1, n):
        seg_len = np.sqrt(
            (xy_points[i, 0] - xy_points[i-1, 0])**2 + (xy_points[i, 1] - xy_points[i-1, 1])**2
        )
        if seg_len < 1e-4:
            seg_len_flag = True
        accumulated_s[i] = accumulated_s[i-1] + (seg_len if seg_len > 1e-4 else point_spacing)
    
    # 计算 x 和 y 对 s 的一阶导数
    x_over_s_first_derivatives = np.zeros(n)
    y_over_s_first_derivatives = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            ds = accumulated_s[i + 1] - accumulated_s[i]
            if ds > 1e-6:
                x_over_s_first_derivatives[i] = (xy_points[i + 1, 0] - xy_points[i, 0]) / ds
                y_over_s_first_derivatives[i] = (xy_points[i + 1, 1] - xy_points[i, 1]) / ds
        elif i == n - 1:
            ds = accumulated_s[i] - accumulated_s[i - 1]
            if ds > 1e-6:
                x_over_s_first_derivatives[i] = (xy_points[i, 0] - xy_points[i - 1, 0]) / ds
                y_over_s_first_derivatives[i] = (xy_points[i, 1] - xy_points[i - 1, 1]) / ds
        else:
            ds = accumulated_s[i + 1] - accumulated_s[i - 1]
            if ds > 1e-6:
                x_over_s_first_derivatives[i] = (xy_points[i + 1, 0] - xy_points[i - 1, 0]) / ds
                y_over_s_first_derivatives[i] = (xy_points[i + 1, 1] - xy_points[i - 1, 1]) / ds
    
    # 计算 x 和 y 对 s 的二阶导数
    x_over_s_second_derivatives = np.zeros(n)
    y_over_s_second_derivatives = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            ds = accumulated_s[i + 1] - accumulated_s[i]
            if ds > 1e-6:
                x_over_s_second_derivatives[i] = (x_over_s_first_derivatives[i + 1] - x_over_s_first_derivatives[i]) / ds
                y_over_s_second_derivatives[i] = (y_over_s_first_derivatives[i + 1] - y_over_s_first_derivatives[i]) / ds
        elif i == n - 1:
            ds = accumulated_s[i] - accumulated_s[i - 1]
            if ds > 1e-6:
                x_over_s_second_derivatives[i] = (x_over_s_first_derivatives[i] - x_over_s_first_derivatives[i - 1]) / ds
                y_over_s_second_derivatives[i] = (y_over_s_first_derivatives[i] - y_over_s_first_derivatives[i - 1]) / ds
        else:
            ds = accumulated_s[i + 1] - accumulated_s[i - 1]
            if ds > 1e-6:
                x_over_s_second_derivatives[i] = (x_over_s_first_derivatives[i + 1] - x_over_s_first_derivatives[i - 1]) / ds
                y_over_s_second_derivatives[i] = (y_over_s_first_derivatives[i + 1] - y_over_s_first_derivatives[i - 1]) / ds
    
    # 计算曲率
    kappas = np.zeros(n)
    for i in range(n):
        xds = x_over_s_first_derivatives[i]
        yds = y_over_s_first_derivatives[i]
        xdds = x_over_s_second_derivatives[i]
        ydds = y_over_s_second_derivatives[i]
        
        speed_sq = xds * xds + yds * yds
        denominator = max(np.sqrt(speed_sq) * max(speed_sq, 1e-8) + 1e-8, 1e-6)
        kappas[i] = np.clip((xds * ydds - yds * xdds) / denominator, -10.0, 10.0)
    
    # 计算曲率变化率
    dkappas = np.zeros(n)
    for i in range(n):
        if i == 0:
            ds = accumulated_s[i + 1] - accumulated_s[i]
            if ds > 1e-6:
                dkappas[i] = (kappas[i + 1] - kappas[i]) / ds
        elif i == n - 1:
            ds = accumulated_s[i] - accumulated_s[i - 1]
            if ds > 1e-6:
                dkappas[i] = (kappas[i] - kappas[i - 1]) / ds
        else:
            ds = accumulated_s[i + 1] - accumulated_s[i - 1]
            if ds > 1e-6:
                dkappas[i] = (kappas[i + 1] - kappas[i - 1]) / ds
    
    return headings, accumulated_s, kappas, dkappas, seg_len_flag


def cartesian_to_frenet_simple(
    rs: float,
    rx: float,
    ry: float,
    rtheta: float,
    x: float,
    y: float
) -> Tuple[float, float]:
    """
    Cartesian 到 Frenet 坐标转换（简化版本，只计算 s 和 d）
    
    参数：
    rs: 参考点的累积弧长 s
    rx, ry: 参考点的 Cartesian 坐标
    rtheta: 参考点的航向角（弧度）
    x, y: 待转换点的 Cartesian 坐标
    
    返回：
    s: 纵向距离（沿参考线的累积弧长）
    d: 横向距离（垂直于参考线的距离，左正右负）
    """
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = np.cos(rtheta)
    sin_theta_r = np.sin(rtheta)
    
    # 计算横向距离 d（带符号）
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = np.copysign(np.sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    # 纵向距离 s 等于参考点的 s
    s = rs
    
    return s, d


def cartesian_to_frenet_full(
    rs: float,
    rx: float,
    ry: float,
    rtheta: float,
    rkappa: float,
    rdkappa: float,
    x: float,
    y: float,
    v: float = 0.0,
    a: float = 0.0,
    theta: float = 0.0,
    kappa: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cartesian 到 Frenet 坐标转换（完整版本，计算 s, d 及其导数）
    
    参数：
    rs: 参考点的累积弧长 s
    rx, ry: 参考点的 Cartesian 坐标
    rtheta: 参考点的航向角（弧度）
    rkappa: 参考点的曲率
    rdkappa: 参考点的曲率变化率
    x, y: 待转换点的 Cartesian 坐标
    v: 待转换点的速度（m/s），默认 0
    a: 待转换点的加速度（m/s²），默认 0
    theta: 待转换点的航向角（弧度），默认 0
    kappa: 待转换点的曲率，默认 0
    
    返回：
    s_condition: [s, s_dot, s_ddot] 纵向距离及其导数
    d_condition: [d, d_dot, d_ddot] 横向距离及其导数
    """
    dx = x - rx
    dy = y - ry
    
    cos_theta_r = np.cos(rtheta)
    sin_theta_r = np.sin(rtheta)
    
    # 计算横向距离 d
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = np.copysign(np.sqrt(dx * dx + dy * dy), cross_rd_nd)
    
    # 计算 d 的一阶导数
    delta_theta = theta - rtheta
    tan_delta_theta = np.tan(delta_theta)
    cos_delta_theta = np.cos(delta_theta)
    
    one_minus_kappa_r_d = 1 - rkappa * d
    d_dot = one_minus_kappa_r_d * tan_delta_theta
    
    # 计算 d 的二阶导数
    kappa_r_d_prime = rdkappa * d + rkappa * d_dot
    d_ddot = (-kappa_r_d_prime * tan_delta_theta +
              one_minus_kappa_r_d / (cos_delta_theta * cos_delta_theta) *
              (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
    
    # 计算 s 及其导数
    s = rs
    s_dot = v * cos_delta_theta / one_minus_kappa_r_d if one_minus_kappa_r_d > 1e-6 else 0.0
    
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa if cos_delta_theta > 1e-6 else 0.0
    s_ddot = ((a * cos_delta_theta -
               s_dot * s_dot * (d_dot * delta_theta_prime - kappa_r_d_prime)) /
              one_minus_kappa_r_d) if one_minus_kappa_r_d > 1e-6 else 0.0
    
    s_condition = np.array([s, s_dot, s_ddot])
    d_condition = np.array([d, d_dot, d_ddot])
    
    return s_condition, d_condition


def convert_path_to_frenet(
    reference_path: np.ndarray,
    headings: np.ndarray,
    accumulated_s: np.ndarray,
    kappas: np.ndarray,
    dkappas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将路径从 Cartesian 坐标系转换为 Frenet 坐标系
    
    注意：对于参考路径上的点（即参考线本身），d = 0，s = 累积弧长
    这个函数主要用于建立 Frenet 坐标系，方便后续局部路径规划
    
    参数：
    reference_path: (n, 2) 参考路径点数组（平滑后的路径段）
    headings: (n,) 参考路径点的航向角数组
    accumulated_s: (n,) 参考路径点的累积弧长数组
    kappas: (n,) 参考路径点的曲率数组
    dkappas: (n,) 参考路径点的曲率变化率数组
    
    返回：
    s_array: (n,) 纵向距离数组（累积弧长）
    d_array: (n,) 横向距离数组（参考线上点 d = 0）
    s_dot_array: (n,) 纵向速度数组（当前设为 0，后续可根据需要计算）
    d_dot_array: (n,) 横向速度数组（参考线上点 d_dot = 0）
    s_ddot_array: (n,) 纵向加速度数组（当前设为 0，后续可根据需要计算）
    d_ddot_array: (n,) 横向加速度数组（参考线上点 d_ddot = 0）
    """
    n = len(reference_path)
    if n == 0:
        return (np.array([]), np.array([]), np.array([]), 
                np.array([]), np.array([]), np.array([]))
    
    # 对于参考路径上的点：
    # - s 就是累积弧长
    # - d = 0（点在参考线上）
    s_array = accumulated_s.copy()
    d_array = np.zeros(n)
    
    # 导数信息：对于参考线上的点，d 及其导数都为 0
    # s 的导数可以根据路径几何信息计算，但当前先设为 0
    # 后续如果需要，可以根据路径点的速度信息计算
    s_dot_array = np.zeros(n)
    d_dot_array = np.zeros(n)
    s_ddot_array = np.zeros(n)
    d_ddot_array = np.zeros(n)
    
    return s_array, d_array, s_dot_array, d_dot_array, s_ddot_array, d_ddot_array


def project_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    计算点到线段的投影点和投影长度
    
    参数：
    point: (2,) 待投影的点 [x, y]
    seg_start: (2,) 线段起点 [x, y]
    seg_end: (2,) 线段终点 [x, y]
    
    返回：
    projected_point: (2,) 投影点 [x, y]
    projection_length: 投影长度（从起点到投影点的距离）
    """
    # 计算线段向量
    seg_vec = seg_end - seg_start
    seg_length = np.linalg.norm(seg_vec)
    
    if seg_length < 1e-6:
        # 线段退化为点
        return seg_start.copy(), 0.0
    
    # 计算从起点到待投影点的向量
    point_vec = point - seg_start
    
    # 计算投影长度（点乘）
    projection_length = np.dot(point_vec, seg_vec) / seg_length
    
    # 限制投影长度在线段范围内
    projection_length = np.clip(projection_length, 0.0, seg_length)
    
    # 计算投影点
    projection_ratio = projection_length / seg_length
    projected_point = seg_start + projection_ratio * seg_vec
    
    return projected_point, projection_length


def find_projection_point_on_path(
    vehicle_point: np.ndarray,
    reference_path: np.ndarray,
    nearest_idx: int,
    accumulated_s: np.ndarray
) -> Tuple[int, np.ndarray, float]:
    """
    在参考路径上找到车辆点的投影点
    
    参数：
    vehicle_point: (2,) 车辆位置 [x, y]
    reference_path: (n, 2) 参考路径点数组
    nearest_idx: 最近路径点索引（在reference_path中的索引）
    accumulated_s: (n,) 累积弧长数组（从参考路径起点开始）
    
    返回：
    best_seg_idx: 投影点所在线段的起始点索引
    projected_point: (2,) 投影点 [x, y]
    projected_s: 投影点的累积弧长 s
    """
    n = len(reference_path)
    if n < 2:
        return 0, reference_path[0] if n > 0 else vehicle_point, 0.0
    
    # 确保 nearest_idx 在有效范围内
    nearest_idx = max(0, min(nearest_idx, n - 1))
    
    best_proj_point = None
    best_proj_s = 0.0
    best_seg_idx = nearest_idx
    min_distance = float('inf')
    
    # 检查 nearest_idx 到 nearest_idx+1 的线段
    if nearest_idx < n - 1:
        seg_start = reference_path[nearest_idx]
        seg_end = reference_path[nearest_idx + 1]
        proj_point, proj_length = project_point_to_segment(vehicle_point, seg_start, seg_end)
        distance = np.linalg.norm(vehicle_point - proj_point)
        
        if distance < min_distance:
            min_distance = distance
            best_proj_point = proj_point
            best_seg_idx = nearest_idx
            # 计算投影点的 s 值
            if nearest_idx == 0:
                best_proj_s = proj_length
            else:
                best_proj_s = accumulated_s[nearest_idx] + proj_length
    
    # 检查 nearest_idx-1 到 nearest_idx 的线段
    if nearest_idx > 0:
        seg_start = reference_path[nearest_idx - 1]
        seg_end = reference_path[nearest_idx]
        proj_point, proj_length = project_point_to_segment(vehicle_point, seg_start, seg_end)
        distance = np.linalg.norm(vehicle_point - proj_point)
        
        if distance < min_distance:
            min_distance = distance
            best_proj_point = proj_point
            best_seg_idx = nearest_idx - 1
            # 计算投影点的 s 值
            if nearest_idx - 1 == 0:
                best_proj_s = proj_length
            else:
                best_proj_s = accumulated_s[nearest_idx - 1] + proj_length
    
    if best_proj_point is None:
        # 如果都没有找到，使用 nearest_idx 点
        best_proj_point = reference_path[nearest_idx]
        best_proj_s = accumulated_s[nearest_idx] if nearest_idx < len(accumulated_s) else 0.0
    
    return best_seg_idx, best_proj_point, best_proj_s


def project_xy_to_sl_polyline(
    xy: np.ndarray,
    ref_xy: np.ndarray,
    acc_s: np.ndarray,
    s_window_lo: Optional[float] = None,
    s_window_hi: Optional[float] = None,
    eps: float = 1e-6,
    s_soft_buffer: float = 50.0,
) -> Optional[Tuple[float, float]]:
    """
    将平面点投影到折线的 (s, l)。

    - s_window_lo / s_window_hi 均为 None：全局模式，整条折线参与搜索，首/末外延用路径物理首尾边。
    - 否则为窗口模式：用 s 在 acc_s 上二分定位段索引 k_first、k_last，先首边外延、再末边外延、再在
      [k_first, k_last] 段上找 |l| 最小垂足；不做全路径回退。投影得到 (s,l) 后做软约束：
      s ∈ [s_lo - s_soft_buffer, s_hi + s_soft_buffer]（s_lo/s_hi 为窗口与路径域求交后的边界）。
    """
    n = len(ref_xy)
    if n < 2:
        return None
    acc = np.asarray(acc_s, dtype=float)
    if len(acc) != n:
        return None

    path_min = float(acc[0])
    path_max = float(acc[-1])
    use_window = s_window_lo is not None and s_window_hi is not None

    if not use_window:
        s_lo, s_hi = path_min, path_max
    else:
        sw0, sw1 = float(s_window_lo), float(s_window_hi)
        if sw0 > sw1:
            sw0, sw1 = sw1, sw0
        if sw0 > path_max + eps or sw1 < path_min - eps:    # 窗口与路径域无交集，直接全局搜索
            use_window = False
            s_lo, s_hi = path_min, path_max
        else:
            s_lo = max(path_min, sw0)
            s_hi = min(path_max, sw1)
    if s_hi - s_lo < eps:
        return None

    xy = np.asarray(xy, dtype=float).reshape(2)

    def _soft_ok(s_val: float) -> bool:
        if not use_window:
            return True
        return s_val >= s_lo - s_soft_buffer - eps and s_val <= s_hi + s_soft_buffer + eps

    if not use_window:
        # --- 全局：首段前外延（物理首边）---
        p0_head = ref_xy[0]
        p1_head = ref_xy[1]
        seg_head = p1_head - p0_head
        slen_head = float(np.linalg.norm(seg_head))
        if slen_head >= eps:
            u_head = seg_head / slen_head
            rel_head = xy - p0_head
            t_head = float(np.dot(rel_head, u_head))
            if t_head < -eps:
                s_ext = float(acc[0] + t_head)
                l_ext = float(u_head[0] * rel_head[1] - u_head[1] * rel_head[0])
                return (s_ext, l_ext)

        # --- 全局：末段后外延 ---
        p0_tail = ref_xy[n - 2]
        p1_tail = ref_xy[n - 1]
        seg_tail = p1_tail - p0_tail
        slen_tail = float(np.linalg.norm(seg_tail))
        if slen_tail >= eps:
            u_tail = seg_tail / slen_tail
            rel_from_p0_tail = xy - p0_tail
            t_tail = float(np.dot(rel_from_p0_tail, u_tail))
            if t_tail > slen_tail + eps:
                rel2 = xy - p1_tail
                s_ext = float(acc[n - 1] + np.dot(rel2, u_tail))
                l_ext = float(u_tail[0] * rel2[1] - u_tail[1] * rel2[0])
                return (s_ext, l_ext)

        # --- 全局：段内，且 s 落在 [path_min, path_max]（与旧行为一致）---
        best_interior: Optional[Tuple[float, float]] = None
        best_abs_l = float("inf")
        for i in range(n - 1):
            p0 = ref_xy[i]
            p1 = ref_xy[i + 1]
            seg = p1 - p0
            slen = float(np.linalg.norm(seg))
            if slen < eps:
                continue
            u = seg / slen
            rel = xy - p0
            t = float(np.dot(rel, u))
            if t < -eps or t > slen + eps:
                continue
            s_val = float(acc[i] + t)
            if s_val < s_lo - eps or s_val > s_hi + eps:
                continue
            l_signed = float(u[0] * rel[1] - u[1] * rel[0])
            al = abs(l_signed)
            if best_interior is None or al < best_abs_l - eps:
                best_interior = (s_val, l_signed)
                best_abs_l = al

        if best_interior is not None:
            return (best_interior[0], best_interior[1])
        return None

    # --- 窗口模式：二分定位 k_first、k_last ---
    j_lo = int(np.searchsorted(acc, s_lo, side="right"))
    k_first = max(0, min(n - 2, j_lo - 1))
    j_hi = int(np.searchsorted(acc, s_hi, side="right"))
    k_last = max(0, min(n - 2, j_hi - 1))

    # 1) 窗口左边界所在边的前向外延
    p0_h = ref_xy[k_first]
    p1_h = ref_xy[k_first + 1]
    seg_h = p1_h - p0_h
    slen_h = float(np.linalg.norm(seg_h))
    if slen_h >= eps:
        u_h = seg_h / slen_h
        rel_h = xy - p0_h
        t_h = float(np.dot(rel_h, u_h))
        if t_h < -eps:
            s_ext = float(acc[k_first] + t_h)
            l_ext = float(u_h[0] * rel_h[1] - u_h[1] * rel_h[0])
            if _soft_ok(s_ext):
                return (s_ext, l_ext)

    # 2) 窗口右边界所在边的后向外延
    p0_t = ref_xy[k_last]
    p1_t = ref_xy[k_last + 1]
    seg_t = p1_t - p0_t
    slen_t = float(np.linalg.norm(seg_t))
    if slen_t >= eps:
        u_t = seg_t / slen_t
        rel_t = xy - p0_t
        t_t = float(np.dot(rel_t, u_t))
        if t_t > slen_t + eps:
            rel2 = xy - p1_t
            s_ext = float(acc[k_last + 1] + np.dot(rel2, u_t))
            l_ext = float(u_t[0] * rel2[1] - u_t[1] * rel2[0])
            if _soft_ok(s_ext):
                return (s_ext, l_ext)

    # 3) 窗口内段上垂足，不按 s_lo/s_hi 硬滤
    best_interior_w: Optional[Tuple[float, float]] = None
    best_abs_l_w = float("inf")
    for i in range(k_first, k_last + 1):
        p0 = ref_xy[i]
        p1 = ref_xy[i + 1]
        seg = p1 - p0
        slen = float(np.linalg.norm(seg))
        if slen < eps:
            continue
        u = seg / slen
        rel = xy - p0
        t = float(np.dot(rel, u))
        if t < -eps or t > slen + eps:
            continue
        s_val = float(acc[i] + t)
        l_signed = float(u[0] * rel[1] - u[1] * rel[0])
        al = abs(l_signed)
        if best_interior_w is None or al < best_abs_l_w - eps:
            best_interior_w = (s_val, l_signed)
            best_abs_l_w = al

    if best_interior_w is not None and _soft_ok(best_interior_w[0]):
        return (best_interior_w[0], best_interior_w[1])
    return None
