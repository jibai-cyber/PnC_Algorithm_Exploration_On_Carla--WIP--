#!/usr/bin/env python3
"""
车辆控制绘图节点 - 独立的绘图可视化
订阅控制节点的绘图数据话题，进行实时可视化

数据格式：Float64MultiArray
数据顺序：[timestamp, cross_track_error, heading_error, normalized_steer,
          vehicle_speed, vehicle_accel, throttle, brake, speed_error, accel_error]
"""

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from std_msgs.msg import Float64MultiArray
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，支持交互式显示
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading
import time

# 颜色定义
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"


class VehiclePlotter(Node):
    """车辆控制绘图节点"""

    def __init__(self):
        super().__init__('vehicle_plotter')

        # 绘图配置
        self.enable_plotting = True
        self.plot_update_rate = 10.0  # 绘图更新频率 (Hz)
        self.error_history_size = 500  # 保存最近500个数据点

        # 数据存储
        self.error_time_history = deque(maxlen=self.error_history_size)
        self.cross_track_error_history = deque(maxlen=self.error_history_size)
        self.heading_error_history = deque(maxlen=self.error_history_size)
        self.normalized_steer_history = deque(maxlen=self.error_history_size)
        self.vehicle_speed_history = deque(maxlen=self.error_history_size)
        self.vehicle_accel_history = deque(maxlen=self.error_history_size)
        self.throttle_history = deque(maxlen=self.error_history_size)
        self.brake_history = deque(maxlen=self.error_history_size)
        self.speed_error_history = deque(maxlen=self.error_history_size)
        self.accel_error_history = deque(maxlen=self.error_history_size)

        # 数据锁（用于线程安全）
        self.data_lock = threading.Lock()

        # 绘图变量
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.ax4_accel = None
        self.ax5 = None
        self.ax6 = None
        self.line_cte = None
        self.line_he = None
        self.line_normalized_steer = None
        self.line_speed = None
        self.line_accel = None
        self.line_throttle = None
        self.line_brake = None
        self.line_speed_error = None
        self.line_accel_error = None

        # 订阅绘图数据话题
        self.plot_data_sub = self.create_subscription(
            Float64MultiArray,
            '/vehicle_control/plot_data',
            self.plot_data_callback,
            10
        )

        # 初始化绘图
        if self.enable_plotting:
            self._init_plot()

        self.get_logger().info(f"{GREEN}✓ 车辆控制绘图节点已启动{RESET}")

    def plot_data_callback(self, msg):
        """绘图数据回调函数"""
        if not self.enable_plotting:
            return

        try:
            if len(msg.data) < 10:
                self.get_logger().warn("绘图数据格式错误")
                return

            # 解析数据
            timestamp = msg.data[0]
            cross_track_error = msg.data[1]
            heading_error = msg.data[2]
            normalized_steer = msg.data[3]
            vehicle_speed = msg.data[4]
            vehicle_accel = msg.data[5]
            throttle = msg.data[6]
            brake = msg.data[7]
            speed_error = msg.data[8]
            accel_error = msg.data[9]

            # 线程安全地添加数据
            with self.data_lock:
                self.error_time_history.append(timestamp)
                self.cross_track_error_history.append(cross_track_error)
                self.heading_error_history.append(heading_error)
                self.normalized_steer_history.append(normalized_steer)
                self.vehicle_speed_history.append(vehicle_speed)
                self.vehicle_accel_history.append(vehicle_accel)
                self.throttle_history.append(throttle)
                self.brake_history.append(brake)
                self.speed_error_history.append(speed_error)
                self.accel_error_history.append(accel_error)

        except Exception as e:
            self.get_logger().warn(f"处理绘图数据失败: {e}")

    def _init_plot(self):
        """初始化误差实时绘图"""
        try:
            # 设置matplotlib使用非阻塞后端
            plt.ion()  # 开启交互模式

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
            plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

            # 创建图形和子图（6个子图，2列3行布局：横向误差、航向误差、转向角、速度、油门、误差）
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(14, 10))
            self.fig.suptitle('车辆控制误差实时监控', fontsize=14, fontweight='bold')

            # 初始化线条
            self.line_cte, = self.ax1.plot([], [], 'b-', label='横向误差 (Cross Track Error)', linewidth=1.5)
            self.line_he, = self.ax2.plot([], [], 'g-', label='航向误差 (Heading Error)', linewidth=1.5)
            self.line_normalized_steer, = self.ax3.plot([], [], 'r-', label='归一化转向角 (Normalized Steer)', linewidth=1.5)
            self.line_speed, = self.ax4.plot([], [], 'm-', label='车辆速度 (Vehicle Speed)', linewidth=1.5)
            self.line_throttle, = self.ax5.plot([], [], 'c-', label='油门 (Throttle)', linewidth=1.5)
            self.line_brake, = self.ax5.plot([], [], 'brown', label='刹车 (Brake)', linewidth=1.5)
            self.line_speed_error, = self.ax6.plot([], [], 'orange', label='速度误差 (Speed Error)', linewidth=1.5)
            self.line_accel_error, = self.ax6.plot([], [], 'purple', label='加速度误差 (Accel Error)', linewidth=1.5)

            # 设置子图1：横向误差
            self.ax1.set_xlabel('时间 (s)', fontsize=10)
            self.ax1.set_ylabel('横向误差 (m)', fontsize=10)
            self.ax1.set_title('横向误差 (Cross Track Error)', fontsize=12)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend(loc='upper right')
            self.ax1.set_ylim([-2.0, 2.0])  # 初始范围，会自动调整

            # 设置子图2：航向误差
            self.ax2.set_xlabel('时间 (s)', fontsize=10)
            self.ax2.set_ylabel('航向误差 (rad)', fontsize=10)
            self.ax2.set_title('航向误差 (Heading Error)', fontsize=12)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend(loc='upper right')
            self.ax2.set_ylim([-1.0, 1.0])  # 初始范围，会自动调整

            # 设置子图3：归一化转向角
            self.ax3.set_xlabel('时间 (s)', fontsize=10)
            self.ax3.set_ylabel('归一化转向角', fontsize=10)
            self.ax3.set_title('归一化转向角 (Normalized Steer)', fontsize=12)
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend(loc='upper right')
            self.ax3.set_ylim([-1.0, 1.0])  # 初始范围，会自动调整（CARLA转向范围是[-1, 1]）

            # 设置子图4：车辆速度和加速度（双Y轴）
            self.ax4.set_xlabel('时间 (s)', fontsize=10)
            self.ax4.set_ylabel('速度 (m/s)', fontsize=10, color='m')
            self.ax4.set_title('车辆速度和加速度 (Vehicle Speed & Acceleration)', fontsize=12)
            self.ax4.grid(True, alpha=0.3)
            self.ax4.tick_params(axis='y', labelcolor='m')
            self.ax4.set_ylim([0.0, 10.0])  # 初始范围，会自动调整

            # 创建右侧Y轴用于加速度
            self.ax4_accel = self.ax4.twinx()
            self.ax4_accel.set_ylabel('加速度 (m/s²)', fontsize=10, color='y')
            self.ax4_accel.tick_params(axis='y', labelcolor='y')
            self.ax4_accel.set_ylim([-5.0, 5.0])  # 初始范围，会自动调整

            # 在右侧Y轴上创建加速度线条
            self.line_accel, = self.ax4_accel.plot([], [], 'y-', label='车辆加速度 (Vehicle Acceleration)', linewidth=1.5)

            # 合并图例
            lines1, labels1 = self.ax4.get_legend_handles_labels()
            lines2, labels2 = self.ax4_accel.get_legend_handles_labels()
            self.ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            # 设置子图5：油门和刹车
            self.ax5.set_xlabel('时间 (s)', fontsize=10)
            self.ax5.set_ylabel('油门/刹车', fontsize=10)
            self.ax5.set_title('油门和刹车 (Throttle & Brake)', fontsize=12)
            self.ax5.grid(True, alpha=0.3)
            self.ax5.legend(loc='upper right')
            self.ax5.set_ylim([0.0, 1.0])  # 初始范围，会自动调整

            # 设置子图6：速度/加速度误差
            self.ax6.set_xlabel('时间 (s)', fontsize=10)
            self.ax6.set_ylabel('误差值', fontsize=10)
            self.ax6.set_title('速度/加速度误差 (Speed/Accel Error)', fontsize=12)
            self.ax6.grid(True, alpha=0.3)
            self.ax6.legend(loc='upper right')
            self.ax6.set_ylim([-5.0, 5.0])  # 初始范围，会自动调整

            plt.tight_layout()
            plt.show(block=False)
            
            # 立即刷新一次，确保窗口显示
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.get_logger().info(f"{GREEN}✓ 实时绘图已初始化（{self.plot_update_rate}Hz更新）{RESET}")

            # 启动绘图更新定时器
            plot_dt = 1.0 / self.plot_update_rate  # 转换为周期（秒）
            self.plot_timer = self.create_timer(plot_dt, self._update_plot)

        except Exception as e:
            self.get_logger().error(f"初始化绘图失败: {e}")
            self.enable_plotting = False

    def _update_plot(self):
        """更新误差绘图"""
        if not self.enable_plotting:
            return

        try:
            # 线程安全地复制数据
            with self.data_lock:
                time_data = list(self.error_time_history)
                cte_data = list(self.cross_track_error_history)
                he_data = list(self.heading_error_history)
                steer_data = list(self.normalized_steer_history)
                speed_data = list(self.vehicle_speed_history)
                accel_data = list(self.vehicle_accel_history)
                throttle_data = list(self.throttle_history)
                brake_data = list(self.brake_history)
                speed_error_data = list(self.speed_error_history)
                accel_error_data = list(self.accel_error_history)

            if len(time_data) == 0:
                return

            # 更新横向误差图
            self.line_cte.set_data(time_data, cte_data)

            # 更新航向误差图
            self.line_he.set_data(time_data, he_data)

            # 自动调整坐标轴范围
            time_range = [max(0, time_data[-1] - 30), time_data[-1] + 1]  # 显示最近30秒

            # 横向误差范围
            if len(cte_data) > 0:
                cte_range = [np.min(cte_data) - 0.2, np.max(cte_data) + 0.2]
                self.ax1.set_xlim(time_range)
                self.ax1.set_ylim(cte_range)

            # 航向误差范围
            if len(he_data) > 0:
                he_range = [np.min(he_data) - 0.2, np.max(he_data) + 0.2]
                self.ax2.set_xlim(time_range)
                self.ax2.set_ylim(he_range)

            # 归一化转向角范围
            if len(steer_data) > 0:
                steer_range = [np.min(steer_data) - 0.1, np.max(steer_data) + 0.1]
                # 限制在[-1, 1]范围内
                steer_range = [max(-1.0, steer_range[0]), min(1.0, steer_range[1])]
                self.ax3.set_xlim(time_range)
                self.ax3.set_ylim(steer_range)
                # 更新归一化转向角图
                self.line_normalized_steer.set_data(time_data, steer_data)

            # 速度和加速度范围（双Y轴）
            if len(speed_data) > 0:
                speed_range = [max(0.0, np.min(speed_data) - 0.5), np.max(speed_data) + 0.5]
                self.ax4.set_xlim(time_range)
                self.ax4.set_ylim(speed_range)
                # 更新速度图
                self.line_speed.set_data(time_data, speed_data)

            if len(accel_data) > 0:
                accel_range = [np.min(accel_data) - 0.5, np.max(accel_data) + 0.5]
                self.ax4_accel.set_ylim(accel_range)
                # 更新加速度图
                self.line_accel.set_data(time_data, accel_data)

            # 油门和刹车范围
            if len(throttle_data) > 0 or len(brake_data) > 0:
                # 计算油门和刹车的联合范围
                all_control = np.concatenate([throttle_data, brake_data]) if len(throttle_data) > 0 and len(brake_data) > 0 else (throttle_data if len(throttle_data) > 0 else brake_data)
                control_range = [max(0.0, np.min(all_control) - 0.05), min(1.0, np.max(all_control) + 0.05)]
                self.ax5.set_xlim(time_range)
                self.ax5.set_ylim(control_range)
                # 更新油门和刹车图
                if len(throttle_data) > 0:
                    self.line_throttle.set_data(time_data, throttle_data)
                if len(brake_data) > 0:
                    self.line_brake.set_data(time_data, brake_data)

            # 速度误差和加速度误差范围（第6个子图）
            if len(speed_error_data) > 0 and len(accel_error_data) > 0:
                # 计算两个误差的联合范围
                all_errors = np.concatenate([speed_error_data, accel_error_data])
                error_range = [np.min(all_errors) - 0.2, np.max(all_errors) + 0.2]
                self.ax6.set_xlim(time_range)
                self.ax6.set_ylim(error_range)
                # 更新速度误差和加速度误差图
                self.line_speed_error.set_data(time_data, speed_error_data)
                self.line_accel_error.set_data(time_data, accel_error_data)

            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            self.get_logger().warn(f"更新绘图失败: {e}")


def main(args=None):
    rclpy.init(args=args)
    plotter = VehiclePlotter()

    try:
        # 使用spin_once循环，以便matplotlib GUI事件循环可以运行
        while rclpy.ok():
            rclpy.spin_once(plotter, timeout_sec=0.01)  # 10ms超时，避免阻塞
            # 处理matplotlib GUI事件，使窗口能够显示和更新
            if plotter.enable_plotting and plotter.fig is not None:
                try:
                    plotter.fig.canvas.flush_events()  # 处理GUI事件
                except:
                    pass
    except KeyboardInterrupt:
        plotter.get_logger().info("正在关闭绘图节点...")
    except Exception as e:
        plotter.get_logger().error(f"绘图节点错误: {e}")
    finally:
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()