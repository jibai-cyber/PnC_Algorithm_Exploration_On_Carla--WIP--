#!/usr/bin/env python3
"""
测试卡尔曼滤波器实现
"""
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """简化的卡尔曼滤波器用于测试"""
    def __init__(self):
        # 状态向量: [x, y, yaw, vx, vy, vyaw]
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1.0

        # 状态转移矩阵
        self.A = np.eye(6)
        self.dt = 0.02

        # 测量矩阵 (只测量x, y, yaw)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # yaw

        # 过程噪声协方差
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

        # 测量噪声协方差
        self.R = np.diag([0.01, 0.01, 0.01])

    def predict(self):
        """预测步骤"""
        self.A[0, 3] = self.dt
        self.A[1, 4] = self.dt
        self.A[2, 5] = self.dt

        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        """更新步骤"""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation

        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def init_state(self, x, y, yaw):
        """初始化状态"""
        self.state[0] = x
        self.state[1] = y
        self.state[2] = yaw
        self.state[3] = 0.0
        self.state[4] = 0.0
        self.state[5] = 0.0
        self.P = np.eye(6) * 1.0

def test_kalman():
    """测试卡尔曼滤波器"""
    kf = KalmanFilter()

    # 模拟真实轨迹（圆形轨迹）
    t = np.linspace(0, 4*np.pi, 200)
    true_x = 10 * np.cos(t)
    true_y = 10 * np.sin(t)
    true_yaw = t + np.pi/2  # 朝向圆心切线方向

    # 添加噪声
    noise_std = 0.1
    measured_x = true_x + np.random.normal(0, noise_std, len(t))
    measured_y = true_y + np.random.normal(0, noise_std, len(t))
    measured_yaw = true_yaw + np.random.normal(0, noise_std, len(t))

    # 初始化卡尔曼滤波器
    kf.init_state(measured_x[0], measured_y[0], measured_yaw[0])

    # 存储滤波结果
    filtered_x = [kf.state[0]]
    filtered_y = [kf.state[1]]
    filtered_yaw = [kf.state[2]]

    for i in range(1, len(t)):
        # 预测
        kf.predict()

        # 更新
        measurement = np.array([measured_x[i], measured_y[i], measured_yaw[i]])
        kf.update(measurement)

        # 存储结果
        filtered_x.append(kf.state[0])
        filtered_y.append(kf.state[1])
        filtered_yaw.append(kf.state[2])

    # 绘制结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(true_x, true_y, 'g-', label='True', linewidth=2)
    plt.plot(measured_x, measured_y, 'r.', label='Measured', alpha=0.5)
    plt.plot(filtered_x, filtered_y, 'b-', label='Filtered', linewidth=1.5)
    plt.legend()
    plt.title('Position')
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    plt.plot(t, true_x, 'g-', label='True X', linewidth=2)
    plt.plot(t, measured_x, 'r.', label='Measured X', alpha=0.5)
    plt.plot(t, filtered_x, 'b-', label='Filtered X', linewidth=1.5)
    plt.legend()
    plt.title('X Position')

    plt.subplot(1, 3, 3)
    plt.plot(t, true_yaw, 'g-', label='True Yaw', linewidth=2)
    plt.plot(t, measured_yaw, 'r.', label='Measured Yaw', alpha=0.5)
    plt.plot(t, filtered_yaw, 'b-', label='Filtered Yaw', linewidth=1.5)
    plt.legend()
    plt.title('Yaw Angle')

    plt.tight_layout()
    plt.savefig('/home/lmy/ws_lmy/ws_ads/kalman_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("卡尔曼滤波器测试完成，结果已保存到 kalman_test.png")

if __name__ == "__main__":
    test_kalman()