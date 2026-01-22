"""
基于扩展卡尔曼滤波(EKF)的自行车模型车辆状态估计
功能：使用EKF融合运动模型和位置/航向角观测，估计车辆位置(x, y)和航向角(phi)
状态向量: [x, y, phi]^T
控制输入: [v, delta]^T (速度, 前轮转角)
观测值: [x, y, phi]^T (直接位置和航向角测量)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class BicycleModelEKF:
    """
    基于自行车模型的扩展卡尔曼滤波器
    用于车辆位置和航向角跟踪
    """
    
    def __init__(self, x0, P0, Q, R, l=2.8, dt=0.1):
        # 状态和协方差初始化
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
        
        # 历史记录
        self.history = {
            'state': [self.x_hat.copy()],
            'covariance': [self.P.copy()],
            'predicted_state': [],
            'measurement': []
        }
        
    def normalize_angle(self, angle):
        normalized = angle
        while normalized > np.pi:
            normalized -= 2 * np.pi
        while normalized < -np.pi:
            normalized += 2 * np.pi
        return normalized
    
    def motion_model(self, x, u):
        v, delta = u
        phi = x[2]
        
        # 离散运动方程
        x_new = x[0] + v * np.cos(phi) * self.dt
        y_new = x[1] + v * np.sin(phi) * self.dt
        phi_new = phi + (v * np.tan(delta) / self.l) * self.dt
        
        # 角度归一化
        phi_new = self.normalize_angle(phi_new)
        
        return np.array([x_new, y_new, phi_new])
    
    def compute_jacobian_F(self, x, u):
        v = u[0]
        phi = x[2]
        
        # 初始化单位矩阵
        F = np.eye(3, dtype=float)
        
        # 设置非零偏导数
        F[0, 2] = -v * np.sin(phi) * self.dt
        F[1, 2] = v * np.cos(phi) * self.dt
        # F[2, 2] = 1 已经在单位矩阵中
        
        return F
    
    def predict(self, u):
        # 1. 状态预测
        self.x_pred = self.motion_model(self.x_hat, u)
        
        # 2. 计算雅可比矩阵 F
        F = self.compute_jacobian_F(self.x_hat, u)
        
        # 3. 协方差预测: P_pred = F * P * F^T + Q
        self.P_pred = F @ self.P @ F.T + self.Q
        
        # 保存预测历史
        self.history['predicted_state'].append(self.x_pred.copy())
        
        return self.x_pred, self.P_pred
    
    def update(self, z):
        # 观测矩阵 (单位矩阵，因为观测直接对应状态)
        H = np.eye(3, dtype=float)
        
        # 1. 计算卡尔曼增益
        # S = H * P_pred * H^T + R
        S = H @ self.P_pred @ H.T + self.R
        
        # K = P_pred * H^T * S^-1
        K = self.P_pred @ H.T @ np.linalg.inv(S)
        
        # 2. 计算测量残差 (新息)
        y = z - self.x_pred
        
        # 角度归一化
        y[2] = self.normalize_angle(y[2])
        
        # 3. 状态更新
        self.x_hat = self.x_pred + K @ y
        
        # 角度归一化
        self.x_hat[2] = self.normalize_angle(self.x_hat[2])
        
        # 4. 协方差更新 (Joseph形式，数值更稳定)
        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P_pred @ (I - K @ H).T + K @ self.R @ K.T
        
        # 保存历史
        self.history['state'].append(self.x_hat.copy())
        self.history['covariance'].append(self.P.copy())
        self.history['measurement'].append(z.copy())
        
        return self.x_hat, self.P
    
    def step(self, u, z):
        self.predict(u)
        return self.update(z)[0]  # 返回更新后的状态
    
    def get_estimation_results(self):
        return (
            np.array(self.history['state']),
            np.array(self.history['covariance']),
            np.array(self.history['predicted_state']),
            np.array(self.history['measurement'])
        )


def simulate_vehicle_motion(total_time=10.0, dt=0.1):
    n_steps = int(total_time / dt)
    time = np.arange(0, total_time, dt)
    
    # 初始化状态
    true_state = np.zeros((n_steps, 3))
    measurements = np.zeros((n_steps, 3))
    control_inputs = np.zeros((n_steps, 2))
    
    # 初始状态
    true_state[0] = [0, 0, 0]
    
    # 生成控制输入 (速度 + 转向角)
    for i in range(n_steps):
        # 速度: 随时间增加
        v = 5.0 + 2.0 * np.sin(0.5 * time[i])
        
        # 转向角: 正弦变化模拟转弯
        delta = 0.1 * np.sin(0.8 * time[i])
        
        control_inputs[i] = [v, delta]
    
    # 生成真实轨迹
    for i in range(1, n_steps):
        # 使用自行车模型生成真实轨迹
        phi = true_state[i-1, 2]
        v, delta = control_inputs[i-1]
        
        true_state[i, 0] = true_state[i-1, 0] + v * np.cos(phi) * dt
        true_state[i, 1] = true_state[i-1, 1] + v * np.sin(phi) * dt
        true_state[i, 2] = phi + (v * np.tan(delta) / 2.8) * dt
        
        # 角度归一化
        true_state[i, 2] = np.arctan2(np.sin(true_state[i, 2]), np.cos(true_state[i, 2]))
    
    # 生成带噪声的观测
    measurement_noise_std = np.array([0.5, 0.5, 0.05])  # 观测噪声标准差
    for i in range(n_steps):
        measurements[i] = true_state[i] + np.random.randn(3) * measurement_noise_std
        # 角度归一化
        measurements[i, 2] = np.arctan2(np.sin(measurements[i, 2]), np.cos(measurements[i, 2]))
    
    return time, true_state, control_inputs, measurements


def plot_results(time, true_state, measurements, estimated_state, covariances):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # 1. Position trajectory (x-y plane)
    ax = axes[0, 0]
    ax.plot(true_state[:, 0], true_state[:, 1], 'g-', linewidth=2, label='True Trajectory')
    ax.plot(measurements[:, 0], measurements[:, 1], 'r.', markersize=4, alpha=0.5, label='Measurements')
    ax.plot(estimated_state[:, 0], estimated_state[:, 1], 'b-', linewidth=1.5, label='EKF Estimate')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Vehicle Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # 2. X position estimation
    ax = axes[0, 1]
    ax.plot(time, true_state[:, 0], 'g-', linewidth=2, label='True Value')
    ax.plot(time, measurements[:, 0], 'r.', markersize=4, alpha=0.5, label='Measurements')
    ax.plot(time, estimated_state[:, 0], 'b-', linewidth=1.5, label='EKF Estimate')
    
    # Plot uncertainty interval (±2σ)
    std_x = np.sqrt(covariances[:, 0, 0])
    ax.fill_between(time, 
                    estimated_state[:, 0] - 2*std_x,
                    estimated_state[:, 0] + 2*std_x,
                    alpha=0.2, color='blue', label='±2σ Interval')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (m)')
    ax.set_title('X Position Estimation')
    ax.legend()
    ax.grid(True)
    
    # 3. Y position estimation
    ax = axes[1, 0]
    ax.plot(time, true_state[:, 1], 'g-', linewidth=2, label='True Value')
    ax.plot(time, measurements[:, 1], 'r.', markersize=4, alpha=0.5, label='Measurements')
    ax.plot(time, estimated_state[:, 1], 'b-', linewidth=1.5, label='EKF Estimate')
    
    # Plot uncertainty interval (±2σ)
    std_y = np.sqrt(covariances[:, 1, 1])
    ax.fill_between(time, 
                    estimated_state[:, 1] - 2*std_y,
                    estimated_state[:, 1] + 2*std_y,
                    alpha=0.2, color='blue')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Y Position Estimation')
    ax.legend()
    ax.grid(True)
    
    # 4. Heading angle estimation
    ax = axes[1, 1]
    ax.plot(time, np.degrees(true_state[:, 2]), 'g-', linewidth=2, label='True Value')
    ax.plot(time, np.degrees(measurements[:, 2]), 'r.', markersize=4, alpha=0.5, label='Measurements')
    ax.plot(time, np.degrees(estimated_state[:, 2]), 'b-', linewidth=1.5, label='EKF Estimate')
    
    # Plot uncertainty interval (±2σ)
    std_phi = np.sqrt(covariances[:, 2, 2])
    ax.fill_between(time, 
                    np.degrees(estimated_state[:, 2] - 2*std_phi),
                    np.degrees(estimated_state[:, 2] + 2*std_phi),
                    alpha=0.2, color='blue')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Angle (°)')
    ax.set_title('Heading Angle Estimation')
    ax.legend()
    ax.grid(True)
    
    # 5. Position error
    ax = axes[2, 0]
    pos_error = np.sqrt((estimated_state[:, 0] - true_state[:, 0])**2 + 
                       (estimated_state[:, 1] - true_state[:, 1])**2)
    ax.plot(time, pos_error, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('EKF Position Estimation Error')
    ax.grid(True)
    
    # 6. Heading angle error
    ax = axes[2, 1]
    # Calculate angle error (considering periodicity)
    phi_error = np.abs(np.arctan2(np.sin(estimated_state[:, 2] - true_state[:, 2]),
                                 np.cos(estimated_state[:, 2] - true_state[:, 2])))
    ax.plot(time, np.degrees(phi_error), 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Angle Error (°)')
    ax.set_title('EKF Heading Angle Estimation Error')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics
    print("="*50)
    print("EKF Performance Statistics:")
    print("="*50)
    print(f"Average Position Error: {np.mean(pos_error):.4f} m")
    print(f"Position Error Std: {np.std(pos_error):.4f} m")
    print(f"Maximum Position Error: {np.max(pos_error):.4f} m")
    print(f"Average Heading Error: {np.degrees(np.mean(phi_error)):.2f} °")
    print(f"Heading Error Std: {np.degrees(np.std(phi_error)):.2f} °")


def main():
    print("自行车模型EKF车辆状态估计演示")
    print("="*50)
    
    # 1. 仿真参数设置
    total_time = 20.0  # 总仿真时间
    dt = 0.1          # 采样时间
    
    # 2. 生成仿真数据
    print("生成车辆运动仿真数据...")
    time, true_state, control_inputs, measurements = simulate_vehicle_motion(total_time, dt)
    
    # 3. EKF初始化
    print("初始化扩展卡尔曼滤波器...")
    
    # 初始状态估计 (可以有一定误差)
    x0 = measurements[0] + np.array([0.5, -0.5, 0.1])  # 初始估计有误差
    
    # 初始协方差 (表示初始估计的不确定性)
    P0 = np.diag([1.0, 1.0, 0.5])
    
    # 过程噪声协方差 (模型不确定性)
    Q = np.diag([0.1, 0.1, 0.01])
    
    # 观测噪声协方差 (传感器误差)
    R = np.diag([0.5, 0.5, 0.05])
    
    # 创建EKF实例
    ekf = BicycleModelEKF(x0=x0, P0=P0, Q=Q, R=R, l=2.8, dt=dt)
    
    # 4. 运行EKF
    print("运行扩展卡尔曼滤波器...")
    estimated_states = []
    
    for i in range(len(time)):
        if i == 0:
            # 第一步只有初始状态
            estimated_states.append(ekf.x_hat.copy())
            continue
            
        # 获取当前控制输入和观测
        u = control_inputs[i-1]  # 使用上一时刻的控制
        z = measurements[i]      # 当前时刻的观测
        
        # 执行EKF步骤
        x_est = ekf.step(u, z)
        estimated_states.append(x_est.copy())
        
        # 打印进度
        if i % 20 == 0:
            print(f"  处理第 {i}/{len(time)} 步...")
    
    estimated_states = np.array(estimated_states)
    
    # 5. 获取历史数据用于绘图
    states_hist, covariances_hist, _, _ = ekf.get_estimation_results()
    
    # 6. 绘制结果
    print("绘制估计结果...")
    plot_results(time, true_state, measurements, states_hist, covariances_hist)
    
    print("仿真完成!")
    
    return ekf, time, true_state, measurements, states_hist


if __name__ == "__main__":
    # 运行主程序
    ekf, time, true_state, measurements, states_hist = main()
    
    # 示例：如何访问EKF的当前状态和参数
    print("\n当前EKF状态:")
    print(f"  估计位置: [{ekf.x_hat[0]:.2f}, {ekf.x_hat[1]:.2f}] m")
    print(f"  估计航向: {np.degrees(ekf.x_hat[2]):.1f}°")
    print(f"  位置协方差: diag([{ekf.P[0,0]:.3f}, {ekf.P[1,1]:.3f}])")
    print(f"  航向协方差: {ekf.P[2,2]:.3f}")
