#!/usr/bin/env python3
"""CARLA车辆控制节点模块"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.timer import Timer
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray, Int32
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleInfo, CarlaEgoVehicleStatus
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion, quaternion_matrix
import math
import os
import numpy as np
import threading
import carla

try:
    from planning.msg import PathBoundary
except ImportError:
    PathBoundary = None

from .cg_lqr_controller import CgLqrController
from .control_params_loader import control_section, load_control_params, pid_from_section
from .pid_controller import PIDController
from .bicycle_model_ekf import BicycleModelEKF
from .constants import GREEN, CYAN, RESET


def _body_beta_and_speed_mag(vx: float, vy: float, beta_thresh: float) -> tuple[float, float]:
    """车体系纵/横向速度 → (侧偏角 β, 合速度)。"""
    v_mag = float(math.hypot(vx, vy))
    if v_mag < beta_thresh:
        return 0.0, v_mag
    return float(math.atan2(vy, vx)), v_mag


def _load_carla_control_params(logger=None):
    cfg = load_control_params()
    if logger:
        logger.info('Loaded vehicle control params from config')
    return cfg


class CarlaVehicleControl(Node):
    """车辆控制节点"""
    
    def __init__(self):
        super().__init__(
            'carla_vehicle_control',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )

        ctrl_cfg = _load_carla_control_params(self.get_logger())
        veh_cfg = ctrl_cfg['vehicle']
        loop_cfg = ctrl_cfg['control_loop']
        filt_cfg = ctrl_cfg['filter']
        ekf_cfg = ctrl_cfg['ekf']
        noise_cfg = ctrl_cfg['noise']

        self.control_dt = float(loop_cfg['control_dt'])
        self.max_speed = float(loop_cfg['max_speed'])
        self.min_speed = float(loop_cfg['min_speed'])
        self.max_acceleration = float(loop_cfg['max_acceleration'])
        self.max_deceleration = float(loop_cfg['max_deceleration'])
        self.dead_zone_throttle = float(loop_cfg['dead_zone_throttle'])
        self.beta_zero_speed_thresh_mps = float(loop_cfg['beta_zero_speed_thresh_mps'])
        self.switch_threshold = float(loop_cfg['switch_threshold'])
        self.goal_arrival_distance = float(loop_cfg['goal_arrival_distance'])
        self.waypoint_interval = float(loop_cfg['waypoint_interval'])
        self.reference_time_horizon = float(loop_cfg['reference_time_horizon'])
        self.reference_path_publish_every_n_cycles = int(loop_cfg['reference_path_publish_every_n_cycles'])

        self.vehicle_wheelbase = float(veh_cfg['wheelbase'])
        self.vehicle_max_steer_angle = float(veh_cfg['max_steer_angle'])
        self.vehicle_wheel_count = int(veh_cfg['wheel_count'])

        lqr_cfg = ctrl_cfg['lqr']
        self.lqr = CgLqrController(
            wheelbase=self.vehicle_wheelbase,
            dt=self.control_dt,
            lr_ratio=float(lqr_cfg['lr_ratio']),
            max_steer=float(lqr_cfg['max_steer']),
            max_speed=self.max_speed,
            max_acceleration=self.max_acceleration,
            q_x=float(lqr_cfg['q_x']),
            q_y=float(lqr_cfg['q_y']),
            q_theta=float(lqr_cfg['q_theta']),
            r_v=float(lqr_cfg['r_v']),
            r_delta=float(lqr_cfg['r_delta']),
            v_r_floor=float(lqr_cfg['v_r_floor']),
            curvature_dt=float(lqr_cfg['curvature_dt']),
        )

        self.speed_controller = PIDController(**pid_from_section(ctrl_cfg['speed_controller'], self.control_dt))
        self.throttle_controller = PIDController(**pid_from_section(ctrl_cfg['throttle_controller'], self.control_dt))
        self.brake_controller = PIDController(**pid_from_section(ctrl_cfg['brake_controller'], self.control_dt))
        
        # 车辆物理属性（从vehicle_info获取，如果无法获取则使用 config fallback）
        self.vehicle_wheel_info = []  # 存储车轮信息
        self.vehicle_type_id = ""
        self.vehicle_info_received = False
        
        
        # 状态变量
        self.measured_x = 0.0
        self.measured_y = 0.0
        self.measured_yaw = 0.0
        self.measured_speed = 0.0  # m/s
        self.raw_longitudinal_velocity = 0.0
        self.raw_lateral_velocity = 0.0
        self._body_vx = 0.0
        self._body_vy = 0.0
        self.current_beta = 0.0
        self.measured_accel = 0.0
        self.measured_lateral_accel = 0.0
        self.raw_longitudinal_accel = 0.0
        self.raw_lateral_accel = 0.0
        # 与 Odometry.pose.orientation 一致：child(车体)→parent(map) 旋转；用于世界系加速度→车体
        self._odom_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

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
        self.qp_nearest_idx = 0

        self.start_pose = None
        self.goal_pose = None
        self.wait_for_goal = True
        
        self.is_controlling = False
        self.is_arrived = False
        self.is_spd_updated = False
        self.control_timer: Timer | None = None

        self.filter_alpha_acc = float(filt_cfg['alpha_acc'])
        self.filter_alpha_spd = float(filt_cfg['alpha_spd'])
        self.filter_alpha_beta = float(filt_cfg['alpha_beta'])
        self.filter_alpha_throttle = float(filt_cfg['alpha_throttle'])
        self.filter_alpha_brake = float(filt_cfg['alpha_brake'])
        self.filter_alpha_x = float(filt_cfg['alpha_x'])
        self.filter_alpha_y = float(filt_cfg['alpha_y'])
        self.filter_alpha_yaw = float(filt_cfg['alpha_yaw'])

        self.filtered_actual_acc = 0.0
        self.filtered_actual_ay = 0.0
        self.filtered_actual_spd = 0.0
        self.filtered_actual_vy = 0.0
        self.filtered_actual_beta = 0.0
        self.filtered_actual_throttle = float(filt_cfg['initial_throttle'])
        self.filtered_actual_brake = 0.0

        self.ekf_x0 = [0.0, 0.0, 0.0]
        self.ekf_P0 = np.diag(ekf_cfg['P0_diag'])
        self.ekf_Q = np.diag(ekf_cfg['Q_diag'])
        self.ekf_R = np.diag(ekf_cfg['R_diag'])
        self.ekf_dt = float(ekf_cfg['dt'])
        self.ekf = None
        self.last_control_input = np.array([0.0, 0.0, 0.0])

        self.path_curvatures = []

        self.enable_noise = bool(noise_cfg['enable'])
        self.odom_noise_std_x = float(noise_cfg['odom_std_x'])
        self.odom_noise_std_y = float(noise_cfg['odom_std_y'])
        self.odom_noise_std_yaw = float(noise_cfg['odom_std_yaw'])
        self.status_noise_std_velocity = float(noise_cfg['status_std_velocity'])
        self.imu_noise_std_accel = float(noise_cfg['imu_std_accel'])
        
        # 线程锁
        self.data_lock = threading.Lock()

        # EgoPlanningTrajectory：按 header 锚定的相对时间 t 插值 v 与横向参考 (x,y,theta)
        self._speed_prof_t: list[float] = []
        self._speed_prof_v: list[float] = []
        self._traj_px: list[float] = []
        self._traj_py: list[float] = []
        self._traj_ptheta: list[float] = []
        self._traj_stamp_ns: int | None = None
        self._last_planning_anomaly_log_ns: int = 0

        # 调试计数器
        self.log_counter = 0
        
        self.start_time = self.get_clock().now()

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
        # 订阅修正后的起点位置（来自 global_path_planner）
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
        self.vehicle_info_sub = self.create_subscription(
            CarlaEgoVehicleInfo,
            '/carla/ego_vehicle/vehicle_info',
            self.vehicle_info_callback,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.ego_vehicle_status_sub = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.ego_vehicle_status_callback,
            10,
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

        try:
            from planning.msg import EgoPlanningTrajectory
            self._EgoPlanningTrajectory = EgoPlanningTrajectory
            self.ego_plan_traj_sub = self.create_subscription(
                EgoPlanningTrajectory,
                '/planning/ego_trajectory',
                self.ego_plan_traj_callback,
                10,
            )
        except ImportError:
            self._EgoPlanningTrajectory = None
            self.ego_plan_traj_sub = None
        
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
        self.reference_path_counter = 0
        
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

        with self.data_lock:
            self.qp_path = []
            self.ref_nearest_idx = 0
            self.qp_nearest_idx = 0
            self._speed_prof_t = []
            self._speed_prof_v = []
            self._traj_px = []
            self._traj_py = []
            self._traj_ptheta = []
            self._traj_stamp_ns = None
            self._odom_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            self._body_vx = 0.0
            self._body_vy = 0.0
            self.current_beta = 0.0
            self.filtered_actual_beta = 0.0
        self.speed_controller.reset()
        self.throttle_controller.reset()
        self.brake_controller.reset()
        self.is_arrived = False
        # 验证移动是否成功
        new_loc = self.ego_vehicle.get_location()
        new_transform = self.ego_vehicle.get_transform()
        new_yaw = new_transform.rotation.yaw/360.0*2*math.pi

        # 重新初始化EKF
        self.initialize_ekf(new_loc.x, new_loc.y, new_yaw)
        self.get_logger().info(f"移动后位置: ({new_loc.x:.2f}, {new_loc.y:.2f}, {new_loc.z:.2f})")
        self.get_logger().info(f"移动后航向角: {new_yaw:.2f}°")
        self.get_logger().info(f"{GREEN}✓ 车辆已移动到中心线: ({x:.2f}, {y:.2f}){RESET}")
    
    def ego_vehicle_status_callback(self, msg: CarlaEgoVehicleStatus) -> None:
        """世界系线加速度 → 车体系纵/横向，供 odom 间隙内积分车体系速度。"""
        try:
            aw = np.array(
                [
                    float(msg.acceleration.linear.x),
                    float(msg.acceleration.linear.y),
                    float(msg.acceleration.linear.z),
                ],
                dtype=np.float64,
            )
        except (AttributeError, TypeError, ValueError):
            return
        with self.data_lock:
            q = self._odom_quat_xyzw.copy()
        R_wb = quaternion_matrix([q[0], q[1], q[2], q[3]])[:3, :3]
        ab = R_wb.T @ aw
        raw_ax = float(ab[0])
        raw_ay = float(ab[1])
        self.raw_longitudinal_accel = raw_ax
        self.raw_lateral_accel = raw_ay
        if self.enable_noise:
            ax = raw_ax + np.random.normal(0.0, self.imu_noise_std_accel)
            ay = raw_ay + np.random.normal(0.0, self.imu_noise_std_accel)
        else:
            ax, ay = raw_ax, raw_ay
        self.filtered_actual_acc = (
            self.filter_alpha_acc * ax
            + (1.0 - self.filter_alpha_acc) * self.filtered_actual_acc
        )
        self.filtered_actual_ay = (
            self.filter_alpha_acc * ay
            + (1.0 - self.filter_alpha_acc) * self.filtered_actual_ay
        )
        self.measured_accel = self.filtered_actual_acc
        self.measured_lateral_accel = self.filtered_actual_ay

    def _refresh_filtered_beta(self) -> float:
        """由当前车体系速度算 β 并一阶低通，写入 current_beta。"""
        beta_raw, _ = _body_beta_and_speed_mag(
            self._body_vx, self._body_vy, self.beta_zero_speed_thresh_mps
        )
        self.filtered_actual_beta = (
            self.filter_alpha_beta * beta_raw
            + (1.0 - self.filter_alpha_beta) * self.filtered_actual_beta
        )
        self.current_beta = self.filtered_actual_beta
        return self.current_beta
    
    def vehicle_info_callback(self, msg):
        """车辆信息回调 - 获取轴距、最大转向角、车轮等信息"""
        if self.vehicle_info_received:
            return  # 只处理一次，避免重复日志
        
        try:
            # 获取车辆类型ID
            self.vehicle_type_id = msg.id
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
                    self.vehicle_wheelbase = float(control_section('vehicle')['wheelbase'])
                
                # 获取最大转向角（通常前轮才有转向角）
                max_steer_angles = [w['max_steer_angle'] for w in self.vehicle_wheel_info if w['max_steer_angle'] > 0]
                if max_steer_angles:
                    self.vehicle_max_steer_angle = max(max_steer_angles)
                else:
                    # 如果消息中没有，使用默认值
                    self.vehicle_max_steer_angle = float(control_section('vehicle')['max_steer_angle'])
            else:
                # 如果没有wheels信息，使用默认值
                self.get_logger().warn("vehicle_info消息中没有wheels信息，使用默认值")
                self._use_default_vehicle_info()
                return
            
            self.vehicle_info_received = True
            self.lqr.update_vehicle_params(
                wheelbase=self.vehicle_wheelbase,
                max_steer=self.vehicle_max_steer_angle,
            )

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
        """使用 config 中的默认车辆信息"""
        veh = control_section('vehicle')
        self.vehicle_wheelbase = float(veh['wheelbase'])
        self.vehicle_max_steer_angle = float(veh['max_steer_angle'])
        self.vehicle_wheel_count = int(veh['wheel_count'])
        self.vehicle_info_received = True
        self.lqr.update_vehicle_params(
            wheelbase=self.vehicle_wheelbase,
            max_steer=self.vehicle_max_steer_angle,
        )

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
            # twist 为车体系：x 纵向、y 横向；纵向仍供速度 PID，合速度+β 供运动学/EKF
            raw_vx = float(msg.twist.twist.linear.x)
            raw_vy = float(msg.twist.twist.linear.y)
            self.raw_longitudinal_velocity = raw_vx
            self.raw_lateral_velocity = raw_vy
            if self.enable_noise:
                vx_meas = raw_vx + np.random.normal(0.0, self.status_noise_std_velocity)
                vy_meas = raw_vy + np.random.normal(0.0, self.status_noise_std_velocity)
            else:
                vx_meas, vy_meas = raw_vx, raw_vy
            self.filtered_actual_spd = (
                self.filter_alpha_spd * vx_meas
                + (1.0 - self.filter_alpha_spd) * self.filtered_actual_spd
            )
            self.filtered_actual_vy = (
                self.filter_alpha_spd * vy_meas
                + (1.0 - self.filter_alpha_spd) * self.filtered_actual_vy
            )
            self._body_vx = self.filtered_actual_spd
            self._body_vy = self.filtered_actual_vy
            self.measured_speed = self.filtered_actual_spd
            self._refresh_filtered_beta()
            self.is_spd_updated = True

            # 获取原始测量值
            raw_x = msg.pose.pose.position.x
            raw_y = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            self._odom_quat_xyzw = np.array(
                [
                    float(orientation_q.x),
                    float(orientation_q.y),
                    float(orientation_q.z),
                    float(orientation_q.w),
                ],
                dtype=np.float64,
            )
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

    def ego_plan_traj_callback(self, msg):
        points = []
        t_list: list[float] = []
        v_list: list[float] = []
        x_list: list[float] = []
        y_list: list[float] = []
        th_list: list[float] = []
        for p in msg.points:
            points.append((p.x, p.y))
            t_list.append(float(p.t))
            v_list.append(float(p.v))
            x_list.append(float(p.x))
            y_list.append(float(p.y))
            th_list.append(float(p.theta))
        recv_ns = self.get_clock().now().nanoseconds
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        if stamp_ns <= 1e-6:
            stamp_ns = recv_ns
        with self.data_lock:
            self.qp_path = points
            if len(points) < 2:
                self.qp_nearest_idx = 0
            if (
                t_list
                and len(t_list) == len(v_list)
                and len(x_list) == len(t_list)
                and len(y_list) == len(t_list)
                and len(th_list) == len(t_list)
            ):
                self._speed_prof_t = t_list
                self._speed_prof_v = v_list
                self._traj_px = x_list
                self._traj_py = y_list
                self._traj_ptheta = th_list
                self._traj_stamp_ns = stamp_ns
            else:
                self.qp_nearest_idx = 0
                self._speed_prof_t = []
                self._speed_prof_v = []
                self._traj_px = []
                self._traj_py = []
                self._traj_ptheta = []
                self._traj_stamp_ns = None

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
        self.is_arrived = False
        
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
            self.ref_nearest_idx = 0
            self.qp_nearest_idx = 0
            self.path_curvatures = []
            self._speed_prof_t = []
            self._speed_prof_v = []
            self._traj_px = []
            self._traj_py = []
            self._traj_ptheta = []
            self._traj_stamp_ns = None
        
        # 重置控制相关状态
        self.speed_controller.reset()
        self.throttle_controller.reset()
        self.brake_controller.reset()
        self.log_counter = 0
        
        # 重置EKF
        self.ekf = None
        self.last_control_input = np.array([0.0, 0.0, 0.0])
        self._body_vx = 0.0
        self._body_vy = 0.0
        self.current_beta = 0.0
        self.filtered_actual_beta = 0.0
        with self.data_lock:
            self._odom_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # 清空目标点
        self.goal_pose = None
        
        self.get_logger().info("✓ 车辆已停止，等待设置目标点...")
    
    def goal_pose_callback(self, msg):
        """目标点回调"""
        self.get_logger().info("收到目标点，准备启动车辆...")
        self.goal_pose = msg.pose
        self.wait_for_goal = False
        
        # 如果已有路径，开始控制
        if len(self.waypoints) > 0 and not self.is_controlling:
            self.start_control()
    
    
    def plan_speed(self):
        """速度规划"""
        planned_speed = 1.5
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

            if dist_to_goal < self.goal_arrival_distance:
                planned_speed = 0.0
        
        return float(planned_speed)

    def _planning_exec_tau_sec(
        self, traj_stamp_ns: int | None, prof_t: list[float]
    ) -> tuple[float | None, float, float]:
        """
        与 _planned_speed_from_held_profile 相同的时间轴：tau_raw = now - stamp。
        返回 (tau_clamped, tau_raw, t_max)。无效时 (None, _, _)。
        """
        if traj_stamp_ns is None or not prof_t:
            return None, 0.0, 0.0
        now_ns = self.get_clock().now().nanoseconds
        tau_raw = (now_ns - traj_stamp_ns) * 1e-9
        t_max = float(max(prof_t))
        reasons: list[str] = []
        if tau_raw < 0.0:
            reasons.append("tau<0")
        if tau_raw > t_max:
            reasons.append("tau>t_last")
        tau = float(np.clip(tau_raw, 0.0, t_max))
        if reasons:
            if now_ns - self._last_planning_anomaly_log_ns >= 1_000_000_000:
                self.get_logger().error(
                    f"规划器时间异常（已夹紧插值）: {', '.join(reasons)}; "
                    f"tau_raw={tau_raw:.3f}s -> tau={tau:.3f}s, t_max={t_max:.3f}s"
                )
                self._last_planning_anomaly_log_ns = now_ns
        return tau, tau_raw, t_max

    def _planned_speed_from_held_profile(
        self,
        prof_t: list[float],
        prof_v: list[float],
        traj_stamp_ns: int | None,
        tau_clamped: float | None = None,
    ) -> float | None:
        """按轨迹 header 时钟与剖面点 (t,v) 线性插值期望速度；无有效剖面时返回 None。"""
        if not prof_t or len(prof_t) != len(prof_v):
            return None
        if tau_clamped is None:
            tau, _, _ = self._planning_exec_tau_sec(traj_stamp_ns, prof_t)
            if tau is None:
                return None
        else:
            tau = tau_clamped
        t_arr = np.asarray(prof_t, dtype=np.float64)
        v_arr = np.asarray(prof_v, dtype=np.float64)
        v = float(np.interp(tau, t_arr, v_arr))
        return float(np.clip(v, 0.0, self.max_speed))
    
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
    
    def _update_error_visualization(
        self,
        cross_track_error,
        heading_error,
        normalized_steer=0.0,
        vehicle_speed=0.0,
        vehicle_accel=0.0,
        throttle=0.0,
        brake=0.0,
        speed_error=0.0,
        accel_error=0.0,
    ):
        """发布绘图数据到话题"""
        if not self.enable_plotting:
            return

        try:
            # 检查发布器是否已初始化
            if not hasattr(self, 'plot_data_pub') or self.plot_data_pub is None:
                return
            
            elapsed = self.get_clock().now() - self.start_time
            current_time_s = float(elapsed.nanoseconds) * 1e-9
            plot_data = Float64MultiArray()
            plot_data.data = [
                current_time_s,
                float(cross_track_error),
                float(heading_error),
                float(normalized_steer),
                float(vehicle_speed),
                float(vehicle_accel),
                float(throttle),
                float(brake),
                float(speed_error),
                float(accel_error),
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
                speed_prof_t = list(self._speed_prof_t)
                speed_prof_v = list(self._speed_prof_v)
                traj_px = list(self._traj_px)
                traj_py = list(self._traj_py)
                traj_ptheta = list(self._traj_ptheta)
                traj_stamp_ns = self._traj_stamp_ns

                if not self.is_spd_updated:
                    self._body_vx += self.control_dt * self.measured_accel
                    self._body_vy += self.control_dt * self.measured_lateral_accel
                else:
                    self._body_vx = self.measured_speed
                    self._body_vy = self.filtered_actual_vy
                    self.is_spd_updated = False
                self.current_speed = self._body_vx
                self._refresh_filtered_beta()
                _, v_kin = _body_beta_and_speed_mag(
                    self._body_vx, self._body_vy, self.beta_zero_speed_thresh_mps
                )

                if not self.ekf.is_updating and self.is_controlling:
                    phi = float(self.current_yaw)
                    beta = float(self.current_beta)
                    phi_vel = phi + beta
                    self.current_yaw = self.ekf.normalize_angle(self.current_yaw)
                    self.current_x += float(
                        self.control_dt * v_kin * math.cos(phi_vel)
                    )
                    self.current_y += float(
                        self.control_dt * v_kin * math.sin(phi_vel)
                    )
                    self.ekf.x_hat = np.array(
                        [self.current_x, self.current_y, self.current_yaw]
                    )
                else:
                    self.current_x = float(self.ekf.x_hat[0])
                    self.current_y = float(self.ekf.x_hat[1])
                    self.current_yaw = float(self.ekf.x_hat[2])
                    self.ekf.is_updating = False

                current_x = self.current_x
                current_y = self.current_y
                current_yaw = self.current_yaw
                current_v = self.current_speed

            # 横向依赖带时间的 Ego 轨迹；无则停车（不再回退全局 waypoints）
            if (
                len(qp_path) < 2
                or traj_stamp_ns is None
                or len(speed_prof_t) < 2
                or len(traj_px) < 2
                or len(traj_py) < 2
                or len(traj_ptheta) < 2
            ):
                self.publish_stop()
                return

            tau_plan, _, _ = self._planning_exec_tau_sec(traj_stamp_ns, speed_prof_t)
            if tau_plan is None:
                self.publish_stop()
                return

            # 检查是否到达目标
            if len(waypoints) > 0:
                goal = waypoints[-1]
                dist_to_goal = math.sqrt((current_x - goal[0])**2 + (current_y - goal[1])**2)
                
                if dist_to_goal < self.goal_arrival_distance and not self.is_arrived:
                    self.get_logger().info(f"{GREEN}🎯 已到达目标点！{RESET}")
                    self.publish_stop()
                    self.is_arrived = True
                    self.is_controlling = False
                    # ====== Humble适配：定时器取消+销毁 ======
                    if hasattr(self, 'control_timer') and self.control_timer is not None:
                        if not self.control_timer.is_canceled():
                            self.control_timer.cancel()
                    return

            self.qp_nearest_idx = self._find_nearest_idx_on_path(
                qp_path, current_x, current_y, self.qp_nearest_idx
            )

            lqr_out = self.lqr.compute_control(
                current_x,
                current_y,
                current_yaw,
                speed_prof_t,
                traj_px,
                traj_py,
                traj_ptheta,
                speed_prof_v,
                tau_plan,
            )
            steering_angle = lqr_out["steering_angle"]
            target_point = lqr_out["ref_point"]
            ref_theta = lqr_out["ref_theta"]
            heading_error = lqr_out["heading_error"]
            cross_track_error = lqr_out["cross_track_error"]

            new_ref_idx = self._find_nearest_idx_on_path(
                waypoints, current_x, current_y, self.ref_nearest_idx
            )
            with self.data_lock:
                self.ref_nearest_idx = new_ref_idx

            # 发布参考线最近索引供 ref_line_smoother / planning_base 使用
            nearest_idx_msg = Int32()
            nearest_idx_msg.data = self.ref_nearest_idx
            self.nearest_idx_pub.publish(nearest_idx_msg)

            # 发布参考线（每隔两个控制周期发布一次）
            self.reference_path_counter += 1
            if self.reference_path_counter >= self.reference_path_publish_every_n_cycles:
                self.publish_reference_path(waypoints, self.ref_nearest_idx, current_v)
                self.reference_path_counter = 0

            # 速度：LQR 方案 A — v_r 来自轨迹插值，v_cmd = v_r + Δv 作为 PID 期望速度
            planned_speed = lqr_out["v_cmd"]

            # EKF 控制输入 [合速度, beta, delta]
            v_kin, _ = _body_beta_and_speed_mag(
                self._body_vx, self._body_vy, self.beta_zero_speed_thresh_mps
            )
            self.last_control_input = np.array([v_kin, self.current_beta, steering_angle])

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
                throttle = self.throttle_controller.compute(accel_error) + self.dead_zone_throttle
                self.filtered_actual_throttle = self.filter_alpha_throttle * throttle + (1 - self.filter_alpha_throttle) * self.filtered_actual_throttle
                throttle = self.filtered_actual_throttle

            else:
                throttle = self.throttle_controller.compute(accel_error) + self.dead_zone_throttle
                self.filtered_actual_throttle = self.filter_alpha_throttle * throttle + (1 - self.filter_alpha_throttle) * self.filtered_actual_throttle
                throttle = self.filtered_actual_throttle
                self.brake_controller.reset()

            # 发布控制命令
            # steering_angle 是弧度，CARLA steer 范围是 [-1, 1]
            max_steer_rad = self.vehicle_max_steer_angle
            normalized_steer = steering_angle / max_steer_rad

            # CARLA转向方向：正值=右转，负值=左转
            # LQR 输出：正值=左转，负值=右转（标准右手坐标系），需取反
            normalized_steer = -normalized_steer

            # 更新误差可视化（绘图）
            self._update_error_visualization(
                cross_track_error,
                heading_error,
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
        self.qp_nearest_idx = 0
        self.log_counter = 0
        self.start_time = self.get_clock().now()
        if not self.is_arrived:
            self.control_timer = self.create_timer(self.control_dt, self.control_loop)

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
