#!/usr/bin/env python3
"""基于质心参考的自行车运动学 EKF"""

import numpy as np


class BicycleModelEKF:
    def __init__(self, x0, P0, Q, R, l=2.8, dt=0.1, control_dt=0.05):
        self.x_hat = np.array(x0, dtype=float)  # 状态估计 [x, y, phi]
        self.P = np.array(P0, dtype=float)      # 估计协方差
        self.x_pred = None                      # 预测状态
        self.P_pred = None                      # 预测协方差

        # 噪声协方差矩阵
        self.Q = np.array(Q, dtype=float)       # 过程噪声协方差
        self.R = np.array(R, dtype=float)       # 观测噪声协方差

        # 车辆参数
        self.l = float(l)                       # 轴距
        self.dt = float(dt)                     # 采样时间
        self.is_updating = False

    def normalize_angle(self, angle):
        normalized = angle
        while normalized > np.pi:
            normalized -= 2 * np.pi
        while normalized < -np.pi:
            normalized += 2 * np.pi
        return normalized

    def motion_model(self, x, u):
        v, beta, delta = u
        phi = x[2]
        phi_vel = phi + beta

        x_new = x[0] + v * np.cos(phi_vel) * self.dt
        y_new = x[1] + v * np.sin(phi_vel) * self.dt
        phi_new = phi + (v * np.tan(delta) * np.cos(beta) / self.l) * self.dt

        phi_new = self.normalize_angle(phi_new)

        return np.array([x_new, y_new, phi_new])

    def compute_jacobian_F(self, x, u):
        v, beta, _delta = u
        phi = x[2]
        phi_vel = phi + beta

        F = np.eye(3, dtype=float)

        F[0, 2] = -v * np.sin(phi_vel) * self.dt
        F[1, 2] = v * np.cos(phi_vel) * self.dt

        return F

    def predict(self, u):
        # 1. 状态预测
        self.x_pred = self.motion_model(self.x_hat, u)

        # 2. 计算雅可比矩阵 F
        F = self.compute_jacobian_F(self.x_hat, u)

        # 3. 协方差预测: P_pred = F * P * F^T + Q
        self.P_pred = F @ self.P @ F.T + self.Q

        return self.x_pred, self.P_pred

    def update(self, z):

        H = np.eye(3, dtype=float)

        S = H @ self.P_pred @ H.T + self.R

        K = self.P_pred @ H.T @ np.linalg.inv(S)

        y = z - self.x_pred

        y[2] = self.normalize_angle(y[2])

        self.x_hat = self.x_pred + K @ y

        self.x_hat[2] = self.normalize_angle(self.x_hat[2])

        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P_pred @ (I - K @ H).T + K @ self.R @ K.T

        return self.x_hat, self.P

    def step(self, u, z):
        self.predict(u)
        return self.update(z)[0]
