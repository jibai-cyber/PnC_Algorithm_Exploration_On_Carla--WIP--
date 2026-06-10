#!/usr/bin/env python3
"""质心参考自行车运动学 LQR 轨迹跟踪控制器"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from map_load.math_utils import compute_path_curvatures_triangle, interp_angle_1d

try:
    from scipy.linalg import solve_discrete_are
except ImportError:
    solve_discrete_are = None


def _wrap_angle(angle: float) -> float:
    a = float(angle)
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _beta_from_delta(delta: float, lr: float, wheelbase: float) -> float:
    return math.atan(lr / wheelbase * math.tan(delta))


def _beta_derivative(delta: float, lr: float, wheelbase: float) -> float:
    tan_d = math.tan(delta)
    cos_d = math.cos(delta)
    if abs(cos_d) < 1e-9:
        return 0.0
    sec2_d = 1.0 / (cos_d * cos_d)
    denom = 1.0 + (lr / wheelbase * tan_d) ** 2
    return lr / wheelbase * sec2_d / denom


def _g_delta(delta: float, lr: float, wheelbase: float) -> float:
    beta = _beta_from_delta(delta, lr, wheelbase)
    beta_p = _beta_derivative(delta, lr, wheelbase)
    tan_d = math.tan(delta)
    cos_d = math.cos(delta)
    if abs(cos_d) < 1e-9:
        return 0.0
    sec2_d = 1.0 / (cos_d * cos_d)
    return -math.sin(beta) * beta_p * tan_d + math.cos(beta) * sec2_d


def delta_r_from_kappa(kappa: float, wheelbase: float, lr: float) -> float:
    """由参考曲率反求前馈转向角（质心模型，固定点迭代）。"""
    if abs(kappa) < 1e-8:
        return 0.0
    delta = math.atan(wheelbase * kappa)
    for _ in range(3):
        beta = _beta_from_delta(delta, lr, wheelbase)
        cos_b = math.cos(beta)
        if abs(cos_b) < 1e-6:
            break
        delta = math.atan(wheelbase * kappa / cos_b)
    return float(delta)


def geometric_tracking_errors(
    current_x: float,
    current_y: float,
    current_yaw: float,
    ref_x: float,
    ref_y: float,
    ref_theta: float,
) -> tuple[float, float]:
    """与 Stanley 一致的几何横向/航向误差（左正）。"""
    dx = ref_x - current_x
    dy = ref_y - current_y
    cross_track = -math.sin(ref_theta) * dx + math.cos(ref_theta) * dy
    heading = math.atan2(
        math.sin(ref_theta - current_yaw),
        math.cos(ref_theta - current_yaw),
    )
    return float(cross_track), float(heading)


class CgLqrController:
    """质心参考点 CG-LQR：输出 v_cmd = v_r + Δv 与 δ_cmd。"""

    def __init__(
        self,
        wheelbase: float = 2.7,
        dt: float = 0.05,
        lr_ratio: float = 0.5,
        max_steer: float = 1.22,
        max_speed: float = 6.0,
        max_acceleration: float = 2.0,
        q_x: float = 100.0,
        q_y: float = 100.0,
        q_theta: float = 10.0,
        r_v: float = 1.0,
        r_delta: float = 50.0,
        v_r_floor: float = 0.15,
        curvature_dt: float = 0.05,
    ):
        self.wheelbase = float(wheelbase)
        self.lr = float(lr_ratio) * self.wheelbase
        self.dt = float(dt)
        self.max_steer = float(max_steer)
        self.max_speed = float(max_speed)
        self.max_dv = float(max_acceleration) * self.dt
        self.v_r_floor = float(v_r_floor)
        self.curvature_dt = float(curvature_dt)
        self.Q = np.diag([q_x, q_y, q_theta])
        self.R = np.diag([r_v, r_delta])

    def update_vehicle_params(
        self,
        wheelbase: float | None = None,
        max_steer: float | None = None,
    ) -> None:
        if wheelbase is not None:
            ratio = self.lr / self.wheelbase if self.wheelbase > 1e-6 else 0.5
            self.wheelbase = float(wheelbase)
            self.lr = ratio * self.wheelbase
        if max_steer is not None:
            self.max_steer = float(max_steer)

    def _interpolate_reference(
        self,
        tau: float,
        traj_t: Sequence[float],
        traj_x: Sequence[float],
        traj_y: Sequence[float],
        traj_theta: Sequence[float],
        traj_v: Sequence[float],
    ) -> tuple[float, float, float, float, float]:
        t_arr = np.asarray(traj_t, dtype=np.float64)
        x_arr = np.asarray(traj_x, dtype=np.float64)
        y_arr = np.asarray(traj_y, dtype=np.float64)
        th_arr = np.asarray(traj_theta, dtype=np.float64)
        v_arr = np.asarray(traj_v, dtype=np.float64)

        xr = float(np.interp(tau, t_arr, x_arr))
        yr = float(np.interp(tau, t_arr, y_arr))
        thr = interp_angle_1d(tau, t_arr, th_arr)
        vr = float(np.interp(tau, t_arr, v_arr))
        return xr, yr, thr, vr, float(t_arr[-1])

    def _curvature_at_tau(
        self,
        tau: float,
        traj_t: Sequence[float],
        traj_x: Sequence[float],
        traj_y: Sequence[float],
    ) -> float:
        """由轨迹离散点 (x,y) 三角形法求曲率，再按时间 tau 线性插值。"""
        n = len(traj_t)
        if n < 3 or len(traj_x) != n or len(traj_y) != n:
            return 0.0
        path = [(float(traj_x[i]), float(traj_y[i])) for i in range(n)]
        kappas = compute_path_curvatures_triangle(path, closed=False)
        t_arr = np.asarray(traj_t, dtype=np.float64)
        return float(np.interp(float(tau), t_arr, np.asarray(kappas, dtype=np.float64)))

    def _build_AB(
        self,
        v_r: float,
        theta_r: float,
        delta_r: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        L = self.wheelbase
        lr = self.lr
        T = self.dt
        v_eff = max(abs(v_r), self.v_r_floor)

        beta_r = _beta_from_delta(delta_r, lr, L)
        psi_r = theta_r + beta_r
        beta_p = _beta_derivative(delta_r, lr, L)
        g_val = _g_delta(delta_r, lr, L)
        tan_d = math.tan(delta_r)

        a = np.eye(3, dtype=np.float64)
        a[0, 2] = -T * v_eff * math.sin(psi_r)
        a[1, 2] = T * v_eff * math.cos(psi_r)

        b = np.zeros((3, 2), dtype=np.float64)
        b[0, 0] = T * math.cos(psi_r)
        b[0, 1] = -T * v_eff * math.sin(psi_r) * beta_p
        b[1, 0] = T * math.sin(psi_r)
        b[1, 1] = T * v_eff * math.cos(psi_r) * beta_p
        b[2, 0] = T * math.cos(beta_r) * tan_d / L
        b[2, 1] = T * v_eff * g_val / L
        return a, b

    def _solve_lqr_gain(self, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        if solve_discrete_are is None:
            return None
        try:
            p = solve_discrete_are(a, b, self.Q, self.R)
            k = np.linalg.solve(self.R + b.T @ p @ b, b.T @ p @ a)
            return k
        except (np.linalg.LinAlgError, ValueError):
            return None

    def compute_control(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        traj_t: Sequence[float],
        traj_x: Sequence[float],
        traj_y: Sequence[float],
        traj_theta: Sequence[float],
        traj_v: Sequence[float],
        tau: float,
    ) -> dict:
        """
        按规划相对时间 tau 插值参考，求解 LQR 并返回控制量与误差。
        """
        n = len(traj_t)
        if n < 2 or len(traj_x) != n or len(traj_y) != n or len(traj_theta) != n:
            return self._fallback(current_x, current_y)

        if len(traj_v) != n:
            traj_v = list(traj_v)
            while len(traj_v) < n:
                traj_v.append(traj_v[-1] if traj_v else 0.0)
            traj_v = traj_v[:n]

        xr, yr, thr, v_r, t_max = self._interpolate_reference(
            tau, traj_t, traj_x, traj_y, traj_theta, traj_v
        )
        kappa = self._curvature_at_tau(tau, traj_t, traj_x, traj_y)
        delta_r = delta_r_from_kappa(kappa, self.wheelbase, self.lr)
        delta_r = float(np.clip(delta_r, -self.max_steer, self.max_steer))

        ex = current_x - xr
        ey = current_y - yr
        eth = _wrap_angle(current_yaw - thr)
        state_err = np.array([ex, ey, eth], dtype=np.float64)

        cross_track, heading_err = geometric_tracking_errors(
            current_x, current_y, current_yaw, xr, yr, thr
        )

        a_mat, b_mat = self._build_AB(v_r, thr, delta_r)
        k_gain = self._solve_lqr_gain(a_mat, b_mat)
        lqr_ok = k_gain is not None

        if lqr_ok:
            u_err = -k_gain @ state_err
            delta_v = float(u_err[0])
            delta_delta = float(u_err[1])
        else:
            delta_v = 0.0
            delta_delta = 0.0

        delta_v = float(np.clip(delta_v, -self.max_dv, self.max_dv))
        v_cmd = float(np.clip(v_r + delta_v, 0.0, self.max_speed))
        steering = float(
            np.clip(delta_r + delta_delta, -self.max_steer, self.max_steer)
        )

        return {
            "steering_angle": steering,
            "v_cmd": v_cmd,
            "v_r": v_r,
            "delta_v": delta_v,
            "delta_r": delta_r,
            "cross_track_error": cross_track,
            "heading_error": heading_err,
            "ref_point": (xr, yr),
            "ref_theta": thr,
            "lqr_ok": lqr_ok,
        }

    @staticmethod
    def _fallback(current_x: float, current_y: float) -> dict:
        return {
            "steering_angle": 0.0,
            "v_cmd": 0.0,
            "v_r": 0.0,
            "delta_v": 0.0,
            "delta_r": 0.0,
            "cross_track_error": 0.0,
            "heading_error": 0.0,
            "ref_point": (current_x, current_y),
            "ref_theta": 0.0,
            "lqr_ok": False,
        }
