#!/usr/bin/env python3
"""CARLA车辆控制节点模块"""

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, Int32
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaEgoVehicleInfo
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import time
import threading
import carla

try:
    from map_load.msg import PathBoundary
except ImportError:
    PathBoundary = None

from .stanley_controller import StanleyController
from .pid_controller import PIDController
from .bicycle_model_ekf import BicycleModelEKF
from .constants import GREEN, CYAN, RESET


class CarlaVehicleControl(Node):
    """车辆控制节点"""
    
    def __init__(self):
        super().__init__('carla_vehicle_control')
        
        # 控制参数
        self.control_dt = 0.05  # 20Hz
        self.max_speed = 6.0  # m/s (约21.6 km/h)
        self.min_speed = 0.5
        self.max_acceleration = 2.0
        self.max_deceleration = -3.0
        
        # 初始化Stanley控制器
        # 这里的k参数过小会导致横向位置偏差对前轮转角的影响小，k参数过大会导致对横向误差较为敏感，容易发生震荡
        self.stanley = StanleyController(
            k=6,
            epsilon=0.3,
            max_steer=1.22,
            filter_alpha=0.2,
            lookahead_base=1.5,
            lookahead_gain=0.5
        )
        
        # 初始化PID速度控制器
        self.speed_controller = PIDController(
            kp=2.0, ki=0.15, kd=0.05, 
            dt=self.control_dt,
            output_limits=(-2.0, 2.0),
            integral_limit=1.0
        )
        
        # 加速度内环（输出油门/刹车）
        self.throttle_controller = PIDController(
            kp=0.15, ki=0.04, kd=0.0, 
            dt=self.control_dt,
            output_limits=(0.0, 1.0),  # 输出范围是油门/刹车（特别需要注意的是0.3为油门死区，1=全油门）
            integral_limit=10.0
        )

        self.brake_controller = PIDController(
            kp=0.05, ki=0.01, kd=0.0, 
            dt=self.control_dt,
            output_limits=(0.0, 1.0),
            integral_limit=5.0
        )
        
        # 车辆物理属性（从vehicle_info获取，如果无法获取则使用默认值）
        # CARLA官方默认值：轴距约2.7m，最大转向角约70度(1.22 rad)
        self.vehicle_wheelbase = 2.7  # m (轴距，默认值)
        self.vehicle_max_steer_angle = 1.22  # rad (最大转向角，默认值约70度)
        self.vehicle_wheel_count = 4  # 默认4轮
        self.vehicle_wheel_info = []  # 存储车轮信息
        self.vehicle_type_id = ""
        self.vehicle_info_received = False
        
        
        # 状态变量
        self.measured_x = 0.0
        self.measured_y = 0.0
        self.measured_yaw = 0.0
        self.measured_speed = 0.0  # m/s
        self.measured_accel = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.current_steer = 0.0    #control_loop更新后才可使用，当前模型假设转向角控制无误差
        
        self.waypoints = []
        self.current_waypoint_index = 0
        self.qp_path = []
        # self.qp_path_curvatures = []
        self.ref_nearest_idx = 0
        self.traj_nearest_idx = 0

        self.start_pose = None
        self.goal_pose = None
        self.wait_for_goal = True
        
        self.is_controlling = False
        self.is_spd_updated = False
        self.control_timer: Timer | None = None

        # 滤波系数
        self.filter_alpha_acc = 0.1
        self.filter_alpha_spd = 0.1
        self.filter_alpha_throttle = 0.1
        self.filter_alpha_brake = 0.6
        self.filter_alpha_x = 0.3  # 位置x滤波系数
        self.filter_alpha_y = 0.3  # 位置y滤波系数
        self.filter_alpha_yaw = 0.3  # 姿态yaw滤波系数

        # 滤波后的实际值
        self.filtered_actual_acc = 0.0  # 滤波后的纵向加速度
        self.filtered_actual_spd = 0.0  # 滤波后的速度
        self.filtered_actual_throttle = 0.3  # 滤波后的油门
        self.filtered_actual_brake = 0.0  # 滤波后的刹车

        # EKF相关参数
        self.ekf_x0 = [0.0, 0.0, 0.0]  # 初始状态 [x, y, phi]
        self.ekf_P0 = np.diag([1.0, 1.0, 0.5])  # 初始协方差矩阵
        self.ekf_Q = np.diag([0.1, 0.1, 0.01])  # 过程噪声协方差
        self.ekf_R = np.diag([0.5, 0.5, 0.05])  # 观测噪声协方差
        self.ekf_dt = 0.13  # 采样时间 (s)
        self.ekf = None
        self.last_control_input = np.array([0.0, 0.0])  # 上一次控制输入 [v, delta]

        # 油门刹车切换相关参数
        self.switch_threshold = 0.05
        
        # 路径曲率
        self.path_curvatures = []
        
        # 高斯噪声参数（标准差）
        self.enable_noise = True  # 噪声开关标志位，True=启用噪声，False=禁用噪声
        self.odom_noise_std_x = 0.025  # 位置x方向噪声标准差 (m)
        self.odom_noise_std_y = 0.01 # 位置y方向噪声标准差 (m)
        self.odom_noise_std_yaw = 0.01  # 姿态yaw噪声标准差 (rad)
        self.status_noise_std_velocity = 0.1  # 速度噪声标准差 (m/s)
        self.imu_noise_std_accel = 0.05  # 加速度噪声标准差 (m/s²)
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # 调试计数器
        self.log_counter = 0
        
        self.start_time = time.time()

        # 订阅器
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initialpose_callback,
            10
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        # 订阅修正后的起点位置（来自map_control）
        self.adjusted_pose_sub = self.create_subscription(
            PoseStamped,
            '/adjusted_initialpose',
            self.adjusted_pose_callback,
            10
        )
        self.waypoints_sub = self.create_subscription(
            Path,
            '/carla/ego_vehicle/waypoints',
            self.waypoints_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )
        self.status_sub = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10
        )
        self.vehicle_info_sub = self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.vehicle_info_callback,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/carla/ego_vehicle/imu',
            self.imu_callback,
            10
        )
        
        # 发布器
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )

        # 绘图数据发布器
        self.enable_plotting = True  # 可以通过参数控制是否启用绘图
        if self.enable_plotting:
            self.plot_data_pub = self.create_publisher(
                Float64MultiArray,
                '/vehicle_control/plot_data',
                10
            )
        
        # 路径平滑相关发布器和订阅器
        self.nearest_idx_pub = self.create_publisher(
            Int32,
            '/vehicle_control/nearest_idx',
            10
        )
        
        # 订阅路径增量更新
        self.path_update_index_sub = self.create_subscription(
            Int32,
            '/path_smoothing/update_index',
            self.path_update_index_callback,
            10
        )
        
        self.path_update_points_sub = self.create_subscription(
            Path,
            '/path_smoothing/update_points',
            self.path_update_points_callback,
            10
        )

        self.qp_xy_path_sub = self.create_subscription(
            Path,
            '/path_smoothing/qp_xy_path',
            self.qp_path_callback,
            10
        )
        
        # 状态变量：用于同步索引和点列表
        self.pending_update_index = None
        self.pending_update_points = None
        # PathBoundary 有效标志：若为 False（决策不一致等）则停车
        self.path_boundary_valid = True
        if PathBoundary is not None:
            self.path_boundary_sub = self.create_subscription(
                PathBoundary,
                '/path_smoothing/path_boundary',
                self.path_boundary_callback,
                10
            )
        else:
            self.path_boundary_sub = None

        # 参考线发布相关
        self.reference_path_pub = self.create_publisher(
            Path,
            '/vehicle_control/reference_path',
            10
        )
        self.waypoint_interval = 0.5  # 路径点间隔 (m)，方便后续调整
        self.reference_time_horizon = 5.0  # 参考线时间范围 (s)
        self.reference_path_counter = 0  # 用于控制发布频率（每隔两个控制周期）
        
        # CARLA连接
        self.carla_client = None
        self.carla_world = None
        self.ego_vehicle = None
        self._init_carla_connection()
        
        self.get_logger().info(f"{GREEN}✓ 车辆控制节点已启动{RESET}")
        self.get_logger().info("等待设置起点（2D Pose Estimate）...")
    
    def _init_carla_connection(self):
        """初始化CARLA连接"""
        try:
            self.carla_client = carla.Client('192.168.102.13', 2000)
            self.carla_client.set_timeout(5.0)
            self.carla_world = self.carla_client.get_world()
            self.get_logger().info(f"{GREEN}✓ CARLA连接成功{RESET}")
        except Exception as e:
            self.get_logger().error(f"CARLA连接失败: {e}")
    
    def _find_ego_vehicle(self):
        """查找ego车辆
        
        查找逻辑：
        1. 优先查找 role_name 包含 'ego' 或 'hero' 的车辆（CARLA标准ego车辆标识）
        2. 如果没找到，返回第一个车辆作为默认ego车辆
        """
        if self.carla_world is None:
            return None
        
        first_vehicle = None  # 记录第一个找到的车辆作为备选
        for actor in self.carla_world.get_actors():
            if 'vehicle' in actor.type_id:
                # 如果还没记录第一个车辆，先记录下来
                if first_vehicle is None:
                    first_vehicle = actor
                
                # 检查是否有ego/hero标识
                attrs = actor.attributes
                if 'role_name' in attrs and ('ego' in attrs['role_name'] or 'hero' in attrs['role_name']):
                    return actor  # 找到标准ego车辆，立即返回
        
        # 没找到标准ego车辆，返回第一个车辆
        return first_vehicle
    
    def adjusted_pose_callback(self, msg):
        """收到修正后的起点位置，移动CARLA车辆"""
        pose = msg.pose
        x, y = pose.position.x, pose.position.y
        _, _, yaw = euler_from_quaternion([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ])
        yaw_deg = math.degrees(yaw)
        
        self.get_logger().info(f"收到修正位置: ({x:.2f}, {y:.2f}), Yaw={yaw_deg:.1f}°")
        
        if self.ego_vehicle is None:
            self.ego_vehicle = self._find_ego_vehicle()
        
        if self.ego_vehicle is None:
            self.get_logger().error("未找到ego车辆，无法移动")
            return
        
        # 调试：打印车辆信息
        self.get_logger().info(f"车辆ID: {self.ego_vehicle.id}, 类型: {self.ego_vehicle.type_id}")
        old_loc = self.ego_vehicle.get_location()
        self.get_logger().info(f"移动前位置: ({old_loc.x:.2f}, {old_loc.y:.2f}, {old_loc.z:.2f})")
        
        # ROS -> CARLA 坐标转换
        transform = carla.Transform(
            carla.Location(x=x, y=-y, z=0.5),
            carla.Rotation(pitch=0, yaw=-yaw_deg, roll=0)
        )
        self.ego_vehicle.set_transform(transform)
        
        # 验证移动是否成功
        new_loc = self.ego_vehicle.get_location()
        new_transform = self.ego_vehicle.get_transform()
        new_yaw = new_transform.rotation.yaw/360.0*2*math.pi

        # 重新初始化EKF
        self.initialize_ekf(new_loc.x, new_loc.y, new_yaw)
        self.get_logger().info(f"移动后位置: ({new_loc.x:.2f}, {new_loc.y:.2f}, {new_loc.z:.2f})")
        self.get_logger().info(f"移动后航向角: {new_yaw:.2f}°")
        self.get_logger().info(f"{GREEN}✓ 车辆已移动到中心线: ({x:.2f}, {y:.2f}){RESET}")
    
    def status_callback(self, msg):
        """车辆状态回调"""
        raw_velocity = msg.velocity  # m/s
        if self.enable_noise:
            noise_velocity = np.random.normal(0.0, self.status_noise_std_velocity)
            self.measured_speed = raw_velocity + noise_velocity
        else:
            self.measured_speed = raw_velocity
        self.filtered_actual_spd = self.filter_alpha_spd * self.measured_speed + (1 - self.filter_alpha_spd) * self.filtered_actual_spd
        self.measured_speed = self.filtered_actual_spd
        self.is_spd_updated = True
    
    def imu_callback(self, msg):
        """IMU回调 - 获取纵向加速度"""
        # IMU消息中，linear_acceleration.x 是纵向加速度（车辆前进方向）
        # CARLA坐标系：x轴向前，y轴向左，z轴向上
        raw_longitudinal_accel = msg.linear_acceleration.x  # m/s²
        if self.enable_noise:
            noise_accel = np.random.normal(0.0, self.imu_noise_std_accel)
            longitudinal_accel = raw_longitudinal_accel + noise_accel
        else:
            longitudinal_accel = raw_longitudinal_accel
        self.filtered_actual_acc = self.filter_alpha_acc * longitudinal_accel + (1 - self.filter_alpha_acc) * self.filtered_actual_acc
        self.measured_accel = self.filtered_actual_acc
    
    def vehicle_info_callback(self, msg):
        """车辆信息回调 - 获取轴距、最大转向角、车轮等信息"""
        if self.vehicle_info_received:
            return  # 只处理一次，避免重复日志
        
        try:
            # 获取车辆类型ID
            self.vehicle_type_id = msg.id
            
            # 获取轴距（wheelbase）
            # CARLA的vehicle_info中，wheelbase可能在wheels信息中
            # 或者需要通过前后轮位置计算
            if hasattr(msg, 'wheels') and len(msg.wheels) > 0:
                self.vehicle_wheel_count = len(msg.wheels)
                self.vehicle_wheel_info = []
                
                # 提取车轮信息
                front_wheels = []
                rear_wheels = []
                
                for wheel in msg.wheels:
                    wheel_info = {
                        'position': {
                            'x': wheel.position.x if hasattr(wheel, 'position') else 0.0,
                            'y': wheel.position.y if hasattr(wheel, 'position') else 0.0,
                            'z': wheel.position.z if hasattr(wheel, 'position') else 0.0
                        },
                        'radius': wheel.radius if hasattr(wheel, 'radius') else 0.0,
                        'max_brake_torque': wheel.max_brake_torque if hasattr(wheel, 'max_brake_torque') else 0.0,
                        'max_handbrake_torque': wheel.max_handbrake_torque if hasattr(wheel, 'max_handbrake_torque') else 0.0,
                        'max_steer_angle': wheel.max_steer_angle if hasattr(wheel, 'max_steer_angle') else 0.0
                    }
                    self.vehicle_wheel_info.append(wheel_info)
                    
                    # 根据y坐标判断前后轮（CARLA坐标系：x向前，y向左）
                    y_pos = wheel_info['position']['y']
                    if abs(y_pos) > 0.5:  # 前轮通常在y方向有较大偏移
                        front_wheels.append(wheel_info)
                    else:
                        rear_wheels.append(wheel_info)
                
                # 计算轴距：前后轮中心之间的距离
                if front_wheels and rear_wheels:
                    front_x = sum(w['position']['x'] for w in front_wheels) / len(front_wheels)
                    rear_x = sum(w['position']['x'] for w in rear_wheels) / len(rear_wheels)
                    self.vehicle_wheelbase = abs(front_x - rear_x)
                else:
                    # 如果无法区分前后轮，使用默认值或估算
                    self.vehicle_wheelbase = 2.7  # 默认值
                
                # 获取最大转向角（通常前轮才有转向角）
                max_steer_angles = [w['max_steer_angle'] for w in self.vehicle_wheel_info if w['max_steer_angle'] > 0]
                if max_steer_angles:
                    self.vehicle_max_steer_angle = max(max_steer_angles)
                else:
                    # 如果消息中没有，使用默认值
                    self.vehicle_max_steer_angle = 1.22  # 默认值约70度（CARLA标准）
            else:
                # 如果没有wheels信息，使用默认值
                self.get_logger().warn("vehicle_info消息中没有wheels信息，使用默认值")
                self._use_default_vehicle_info()
                return
            
            self.vehicle_info_received = True
            
            # 打印车辆信息
            self.get_logger().info(
                f"{GREEN}=== 车辆信息已获取 ==={RESET}\n"
                f"车辆类型ID: {self.vehicle_type_id}\n"
                f"轴距: {self.vehicle_wheelbase:.2f} m\n"
                f"最大转向角: {math.degrees(self.vehicle_max_steer_angle):.1f}° ({self.vehicle_max_steer_angle:.3f} rad)\n"
                f"车轮数量: {self.vehicle_wheel_count}\n"
                f"车轮信息: {len(self.vehicle_wheel_info)} 个车轮"
            )
            
            # 打印每个车轮的详细信息
            for i, wheel in enumerate(self.vehicle_wheel_info):
                self.get_logger().info(
                    f"  车轮 {i+1}: 位置=({wheel['position']['x']:.2f}, "
                    f"{wheel['position']['y']:.2f}, {wheel['position']['z']:.2f}) m, "
                    f"半径={wheel['radius']:.2f} m, "
                    f"最大刹车扭矩={wheel['max_brake_torque']:.1f} Nm, "
                    f"最大转向角={math.degrees(wheel['max_steer_angle']):.1f}°"
                )
            
        except Exception as e:
            self.get_logger().error(f"处理车辆信息时出错: {e}，使用默认值")

    def _use_default_vehicle_info(self):
        """使用默认车辆信息"""
        self.vehicle_wheelbase = 2.7
        self.vehicle_max_steer_angle = 1.22
        self.vehicle_wheel_count = 4
        self.vehicle_info_received = True

    def initialize_ekf(self, x, y, yaw):
        """初始化EKF"""
        self.ekf_x0 = [x, y, yaw]
        self.ekf = BicycleModelEKF(
            x0=self.ekf_x0,
            P0=self.ekf_P0,
            Q=self.ekf_Q,
            R=self.ekf_R,
            l=self.vehicle_wheelbase,
            dt=self.ekf_dt,
            control_dt=self.control_dt
        )
        self.get_logger().info(f"EKF已初始化: 初始状态 {self.ekf_x0}")

    def odometry_callback(self, msg):
        """里程计回调 - 使用EKF进行状态估计"""
        with self.data_lock:
            # 获取原始测量值
            raw_x = msg.pose.pose.position.x
            raw_y = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            _, _, raw_yaw = euler_from_quaternion(
                [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            )

            # 添加噪声（模拟传感器噪声）
            if self.enable_noise:
                noise_x = np.random.normal(0.0, self.odom_noise_std_x)
                noise_y = np.random.normal(0.0, self.odom_noise_std_y)
                noise_yaw = np.random.normal(0.0, self.odom_noise_std_yaw)
                measured_x = raw_x + noise_x
                measured_y = raw_y + noise_y
                measured_yaw = raw_yaw + noise_yaw
            else:
                measured_x = raw_x
                measured_y = raw_y
                measured_yaw = raw_yaw

            # 初始化EKF（如果还未初始化）
            if self.ekf is None:
                self.initialize_ekf(measured_x, measured_y, measured_yaw)
                self.measured_x = measured_x
                self.measured_y = measured_y
                self.measured_yaw = measured_yaw
                return

            # 准备观测值 [x, y, phi]
            z = np.array([measured_x, measured_y, measured_yaw])

            # 执行EKF步骤 (预测 + 更新)
            # 使用上一次的控制输入进行预测
            self.ekf.is_updating = True
            x_est = self.ekf.step(self.last_control_input, z)

            # 更新车辆状态
            # self.measured_x = x_est[0]
            # self.measured_y = x_est[1]
            # self.measured_yaw = x_est[2]
            self.current_x = x_est[0]
            self.current_y = x_est[1]
            self.current_yaw = x_est[2]
            self.ekf.is_updating = False
    
    # FIXME: 采样点过于密集，当前间距0.1m
    def downsample_path(self, path, interval=1.0):
        if len(path) < 2:
            return path
        
        downsampled = [path[0]]
        last_point = np.array(path[0])
        dist_accumulated = 0.0
        
        for point in path[1:]:
            current_point = np.array(point)
            dist = np.linalg.norm(current_point - last_point)
            dist_accumulated += dist
            
            if dist_accumulated >= interval:
                downsampled.append(point)
                last_point = current_point
                dist_accumulated = 0.0
        
        # 添加最后一个点
        if downsampled[-1] != path[-1]:
            downsampled.append(path[-1])
        
        return downsampled

    def waypoints_callback(self, msg):
        """路径点回调"""
        new_waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            new_waypoints.append((x, y))
        
        # 路径降采样（间隔1米）
        # new_waypoints = self.downsample_path(new_waypoints, interval=0.5)
        
        with self.data_lock:
            self.waypoints = new_waypoints
            
            self.current_waypoint_index = 0
                    
        self.get_logger().info(f"{CYAN}✓ 收到路径规划，包含 {len(new_waypoints)} 个路径点{RESET}")
    
    def path_update_index_callback(self, msg):
        """路径更新索引回调"""
        self.pending_update_index = msg.data
        self._try_apply_path_update()
    
    def path_update_points_callback(self, msg):
        """路径更新点列表回调"""
        update_points = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            update_points.append((x, y))
        
        self.pending_update_points = update_points
        self._try_apply_path_update()

    def path_boundary_callback(self, msg):
        """PathBoundary 回调：决策无效时置标志，控制循环内会停车"""
        self.path_boundary_valid = msg.valid

    def qp_path_callback(self, msg):
        """QP XY 路径回调：PathData，供 Stanley 跟踪"""
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))
        with self.data_lock:
            self.qp_path = points
            # self.qp_path_curvatures = self.compute_curvatures(points) if len(points) >= 3 else []
            self.traj_nearest_idx = 0

    def _find_nearest_idx_on_path(self, path, x, y, hint_idx=0):
        """在路径上查找距 (x,y) 最近的段起点索引"""
        if len(path) < 2:
            return 0
        current_pos = np.array([x, y])
        min_dist = float('inf')
        nearest_idx = 0
        min_search = max(0, hint_idx - 50)
        max_search = min(len(path) - 1, hint_idx + 50)
        for i in range(min_search, max_search):
            if i + 1 >= len(path):
                break
            start = np.array(path[i])
            end = np.array(path[i + 1])
            seg = end - start
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            unit = seg / seg_len
            proj = np.clip(np.dot(current_pos - start, unit), 0, seg_len)
            nearest = start + proj * unit
            d = np.linalg.norm(current_pos - nearest)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        return nearest_idx

    def _try_apply_path_update(self):
        """尝试应用路径增量更新（需要同时收到索引和点列表）"""
        if self.pending_update_index is None or self.pending_update_points is None:
            return
        
        start_idx = self.pending_update_index
        update_points = self.pending_update_points
        
        # 如果没有需要更新的点，跳过
        if len(update_points) == 0:
            self.pending_update_index = None
            self.pending_update_points = None
            return
        
        with self.data_lock:
            # 局部更新waypoints
            if start_idx >= 0 and start_idx < len(self.waypoints):
                for i, point in enumerate(update_points):
                    global_idx = start_idx + i
                    if 0 <= global_idx < len(self.waypoints):
                        self.waypoints[global_idx] = point
                
        # 清除待处理状态
        self.pending_update_index = None
        self.pending_update_points = None
    
    def initialpose_callback(self, msg):
        """初始位置回调"""
        self.get_logger().info("收到起点，车辆已放置")
        self.start_pose = msg.pose.pose
        self.wait_for_goal = True
        
        # 停止当前控制
        if self.is_controlling and self.control_timer:
            self.control_timer.cancel()
            self.is_controlling = False
        
        # 发送停止命令
        self.publish_stop()
        
        # 重置状态
        with self.data_lock:
            self.waypoints = []
            self.current_waypoint_index = 0
            self.qp_path = []
            # self.qp_path_curvatures = []
            self.ref_nearest_idx = 0
            self.traj_nearest_idx = 0
            self.path_curvatures = []
        
        # 重置控制相关状态
        self.speed_controller.reset()
        self.throttle_controller.reset()
        self.brake_controller.reset()
        self.log_counter = 0
        
        # 重置Stanley控制器滤波状态
        self.stanley.filtered_cross_track_error = 0.0
        self.stanley.filtered_heading_error = 0.0

        # 重置EKF
        self.ekf = None
        self.last_control_input = np.array([0.0, 0.0])

        # 清空目标点
        self.goal_pose = None
        
        self.start_time = time.time()
        
        self.get_logger().info("✓ 车辆已停止，等待设置目标点...")
    
    def goal_pose_callback(self, msg):
        """目标点回调"""
        self.get_logger().info("收到目标点，准备启动车辆...")
        self.goal_pose = msg.pose
        self.wait_for_goal = False
        
        # 如果已有路径，开始控制
        if len(self.waypoints) > 0 and not self.is_controlling:
            self.start_control()
    
    def compute_curvatures(self, path, window_size=5):
        """计算路径曲率"""
        if len(path) < 3:
            return [0.0 for _ in path]
        
        n = len(path)
        curvatures = []
        
        for i in range(n):
            p_prev = np.array(path[i - 1], dtype=float)
            p = np.array(path[i], dtype=float)
            p_next = np.array(path[(i + 1) % n], dtype=float)
            
            ab = p - p_prev
            bc = p_next - p
            ca = p_prev - p_next
            
            lab = np.linalg.norm(ab)
            lbc = np.linalg.norm(bc)
            lca = np.linalg.norm(ca)
            
            denom = lab * lbc * lca
            if denom < 1e-6:
                curvatures.append(0.0)
                continue
            
            area = abs(np.cross(ab, (p_next - p_prev))) * 0.5
            kappa = 4.0 * area / denom

            # FIXME: 需要根据实际转向角方向判断曲率方向，检查一下这里除了影响前馈还影响哪些内容
            # 补充曲率方向（左转为负，右转为正，适配转向角方向）
            # 计算三点的叉积判断转向方向
            cross = (p[0]-p_prev[0])*(p_next[1]-p_prev[1]) - (p[1]-p_prev[1])*(p_next[0]-p_prev[0])
            if cross < 0:
                kappa = -kappa
            curvatures.append(float(kappa))
        
        return curvatures
    
    # FIXME: 后期考虑用交互模型，局部避障规划或speed profile规划速度
    def plan_speed(self):
        """速度规划"""
        planned_speed = 5
        reaction_stop_time = 5
        reaction_stop_distance = self.current_speed * reaction_stop_time
        
        # 接近终点减速
        if len(self.waypoints) > 0:
            goal = self.waypoints[-1]
            dist_to_goal = math.sqrt(
                (self.current_x - goal[0])**2 + (self.current_y - goal[1])**2
            )
            
            if dist_to_goal < reaction_stop_distance:
                # 线性减速
                planned_speed = min(planned_speed, dist_to_goal / reaction_stop_distance * planned_speed)

            if dist_to_goal < 0.5:
                planned_speed = 0.0
        
        return float(planned_speed)
    
    def publish_reference_path(self, waypoints, start_idx, current_speed):
        """
        发布车辆5秒内的参考线路径
        
        参数：
        waypoints: 完整路径点列表
        start_idx: 起始路径点索引（current_waypoint_index）
        current_speed: 当前车辆速度 (m/s)
        """
        # 如果速度太小，不发布
        if current_speed < 0.01:
            return
        
        if len(waypoints) == 0 or start_idx < 0 or start_idx >= len(waypoints):
            return
        
        # 计算5秒内行驶的距离
        distance_5s = current_speed * self.reference_time_horizon
        
        # 根据路径点间隔计算需要显示的路径点数量
        num_points_needed = int(distance_5s / self.waypoint_interval) + 1
        
        # 提取参考路径点（从start_idx开始）
        end_idx = min(start_idx + num_points_needed, len(waypoints))
        reference_waypoints = waypoints[start_idx:end_idx]
        
        # 如果没有路径点，不发布
        if len(reference_waypoints) == 0:
            return
        
        # 创建Path消息
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for waypoint in reference_waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(waypoint[0])
            pose_stamped.pose.position.y = float(waypoint[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        # 发布参考线
        self.reference_path_pub.publish(path_msg)
    
    def _update_error_visualization(self, cross_track_error, heading_error, normalized_steer=0.0, vehicle_speed=0.0, vehicle_accel=0.0, throttle=0.0, brake=0.0, speed_error=0.0, accel_error=0.0):
        """发布绘图数据到话题"""
        if not self.enable_plotting:
            return

        try:
            # 检查发布器是否已初始化
            if not hasattr(self, 'plot_data_pub') or self.plot_data_pub is None:
                return
            
            current_time = time.time() - self.start_time
            plot_data = Float64MultiArray()
            plot_data.data = [
                current_time, cross_track_error, heading_error, normalized_steer,
                vehicle_speed, vehicle_accel, throttle, brake, speed_error, accel_error
            ]
            self.plot_data_pub.publish(plot_data)
        except Exception as e:
            self.get_logger().warn(f"发布绘图数据失败: {e}")
    
    def publish_stop(self):
        """发送停止命令"""
        msg = CarlaEgoVehicleControl()
        msg.throttle = 0.0
        msg.steer = 0.0
        msg.brake = 1.0
        self.control_pub.publish(msg)
    
    def control_loop(self):
        """控制循环"""
        try:
            if not self.path_boundary_valid:
                self.publish_stop()
                return
            with self.data_lock:
                if not self.waypoints:
                    return
                waypoints = self.waypoints.copy()
                qp_path = self.qp_path.copy() if self.qp_path else []
                path_to_track = qp_path if qp_path else waypoints
                # path_curvatures = self.qp_path_curvatures if qp_path else self.path_curvatures
                start_idx = self.traj_nearest_idx if qp_path else self.ref_nearest_idx

                self.get_logger().info("------------------------")
                self.get_logger().info(f"第一个qp_path点: x={qp_path[0][0]}, y={qp_path[0][1]}")
                self.get_logger().info(f"第二个qp_path点: x={qp_path[1][0]}, y={qp_path[1][1]}")
                self.get_logger().info(f"第三个qp_path点: x={qp_path[2][0]}, y={qp_path[2][1]}")
                self.get_logger().info(f"ego当前位置: x={self.current_x}, y={self.current_y}")

                if not self.is_spd_updated:
                    self.current_speed += self.control_dt * self.measured_accel
                else:
                    self.current_speed = self.measured_speed
                    self.is_spd_updated = False

                # FIXME: self.ekf.is_updating python成员变量更新方式有待优化
                if not self.ekf.is_updating and self.is_controlling:
                    self.current_yaw += float(self.control_dt * self.current_speed * np.tan(self.current_steer) / self.vehicle_wheelbase)
                    self.current_yaw = self.ekf.normalize_angle(self.current_yaw)
                    self.current_x += self.control_dt * self.current_speed * np.cos(self.current_yaw)
                    self.current_y += self.control_dt * self.current_speed * np.sin(self.current_yaw)
                    # 高频更新预测状态
                    self.ekf.x_hat = np.array([self.current_x, self.current_y, self.current_yaw])
                else:
                    self.current_x = float(self.ekf.x_hat[0])
                    self.current_y = float(self.ekf.x_hat[1])
                    self.current_yaw = float(self.ekf.x_hat[2])
                    self.ekf.is_updating = False

                current_x = self.current_x
                current_y = self.current_y
                current_yaw = self.current_yaw
                current_v = self.current_speed


            # 检查是否到达目标
            if len(waypoints) > 0:
                goal = waypoints[-1]
                dist_to_goal = math.sqrt((current_x - goal[0])**2 + (current_y - goal[1])**2)
                
                if dist_to_goal < 1.0:
                    self.get_logger().info(f"{GREEN}🎯 已到达目标点！{RESET}")
                    self.publish_stop()
                    self.is_controlling = False
                    # ====== Humble适配：定时器取消+销毁 ======
                    if hasattr(self, 'control_timer') and self.control_timer is not None:
                        if not self.control_timer.is_canceled():
                            self.control_timer.cancel()
                    return
            
            # 使用Stanley控制器计算转向角
            steering_angle, target_point, heading_error, cross_track_error, curvature, nearest_idx = self.stanley.compute_steering(
                current_x, current_y, current_yaw, current_v, path_to_track, self.control_dt, self.vehicle_wheelbase, start_idx, forward_only=bool(qp_path)
            )

            self.get_logger().info(f"heading_error: {heading_error}, cross_track_error: {cross_track_error}, nearest_idx: {nearest_idx}")

            if qp_path:
                new_ref_idx = self._find_nearest_idx_on_path(waypoints, current_x, current_y, self.ref_nearest_idx)
                with self.data_lock:
                    self.traj_nearest_idx = nearest_idx
                    self.ref_nearest_idx = new_ref_idx
            else:
                with self.data_lock:
                    self.ref_nearest_idx = nearest_idx
                    self.current_waypoint_index = nearest_idx

            # 发布参考线最近索引供 path_smoother / vehicle_perception 使用
            nearest_idx_msg = Int32()
            nearest_idx_msg.data = self.ref_nearest_idx
            self.nearest_idx_pub.publish(nearest_idx_msg)

            # 发布参考线（每隔两个控制周期发布一次）
            self.reference_path_counter += 1
            if self.reference_path_counter >= 2:
                self.publish_reference_path(waypoints, self.ref_nearest_idx, current_v)
                self.reference_path_counter = 0

            # 速度规划
            planned_speed = self.plan_speed()

            # 更新EKF控制输入 [v, delta]
            # delta 是前轮转角，steering_angle 是 Stanley 控制器输出的转向角
            self.last_control_input = np.array([current_v, steering_angle])
            
            # PID速度控制
            speed_error = planned_speed - current_v
            acceleration = self.speed_controller.compute(speed_error)
            accel_error = acceleration - self.measured_accel
            self.current_steer = steering_angle

            # 转换为油门/刹车
            # FIXME: 输出为动力学模型输出，但输入是运动学模型输入，需要转换
            brake = 0.0

            if accel_error < -self.switch_threshold:
                brake = self.brake_controller.compute(-accel_error)
                self.filtered_actual_brake = self.filter_alpha_brake * brake + (1 - self.filter_alpha_brake) * self.filtered_actual_brake
                brake = self.filtered_actual_brake
                throttle = self.throttle_controller.compute(accel_error) + 0.3
                self.filtered_actual_throttle = self.filter_alpha_throttle * throttle + (1 - self.filter_alpha_throttle) * self.filtered_actual_throttle
                throttle = self.filtered_actual_throttle

            else:
                throttle = self.throttle_controller.compute(accel_error) + 0.3
                self.filtered_actual_throttle = self.filter_alpha_throttle * throttle + (1 - self.filter_alpha_throttle) * self.filtered_actual_throttle
                throttle = self.filtered_actual_throttle
                self.brake_controller.reset()

            # 发布控制命令
            # steering_angle 是弧度，CARLA steer 范围是 [-1, 1]
            # 假设最大转向角约为 70 度 (1.22 rad)
            max_steer_rad = 1.22  # CARLA 车辆最大转向角
            normalized_steer = steering_angle / max_steer_rad
            
            # CARLA转向方向：正值=右转，负值=左转
            # Stanley输出：正值=左转，负值=右转（标准右手坐标系）
            # 需要取反
            normalized_steer = -normalized_steer

            # 更新误差可视化（绘图）
            self._update_error_visualization(
                self.stanley.current_cross_track_error,
                self.stanley.current_heading_error,
                normalized_steer,
                current_v,
                self.measured_accel,
                throttle,
                brake,
                speed_error,
                accel_error
            )    

            # self.log_counter += 1
            
            msg = CarlaEgoVehicleControl()
            msg.throttle = float(np.clip(throttle, 0.0, 1.0))
            msg.steer = float(np.clip(normalized_steer, -1.0, 1.0))
            msg.brake = float(np.clip(brake, 0.0, 1.0))
            msg.gear = 1
            self.control_pub.publish(msg)
            self.is_controlling = True
                
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def start_control(self):
        """开始控制（适配Humble Timer，避免重复创建）"""
        if not self.waypoints:
            self.get_logger().warn("没有可用的路径点！")
            return
        
        # ====== Humble适配：定时器状态检查 ======
        # 1. 若定时器已存在且未取消，直接返回
        if self.control_timer is not None and not self.control_timer.is_canceled():
            # self.get_logger().warn("控制定时器已运行，无需重复创建")
            return
        
        # 重置控制状态
        self.current_waypoint_index = 0
        self.ref_nearest_idx = 0
        self.traj_nearest_idx = 0
        self.log_counter = 0
        # self.path_curvatures = self.compute_curvatures(self.waypoints) if len(self.waypoints) >= 3 else []
        self.start_time = time.time()
        
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

        self.get_logger().info(f"{GREEN}🚀 启动车辆控制！路径包含 {len(self.waypoints)} 个点{RESET}")

    def run(self):
        """主循环"""
        try:
            # ====== 主循环仅做自旋+启动条件检查 ======
            while rclpy.ok():
                # 自旋处理所有回调（定时器/话题订阅），超时1ms避免阻塞
                rclpy.spin_once(self, timeout_sec=0.001)
                
                # 仅在未控制时检查启动条件
                if not self.is_controlling:
                    # 等待初始位姿、目标位姿，且不等待目标
                    if self.start_pose is None or self.goal_pose is None or self.wait_for_goal:
                        continue
                    # 满足条件启动控制
                    self.start_control()
                

                
        except Exception as e:
            self.get_logger().error(f"主循环错误: {e}")
            raise
        finally:
            # ====== Humble定时器最终清理 ======
            if self.control_timer is not None:
                if not self.control_timer.is_canceled():
                    self.control_timer.cancel()
            self.publish_stop()
            self.get_logger().info(f"{GREEN}🔚 控制结束{RESET}")
