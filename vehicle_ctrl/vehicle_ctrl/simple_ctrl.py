#!/usr/bin/env python3
"""
车辆控制器 - 基于Stanley控制器
通过carla_ros_bridge与CARLA通信

功能：
1. 订阅RViz初始位姿和目标点
2. 订阅路径规划结果
3. 使用Stanley控制器进行横向控制
4. 使用PID控制器进行纵向控制
5. 发布车辆控制命令到CARLA
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaEgoVehicleInfo
from tf_transformations import euler_from_quaternion
import math
import numpy as np
from collections import deque
import time
import threading
import carla
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，支持交互式显示
import matplotlib.pyplot as plt

# 颜色定义
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"


class StanleyController:
    """
    Stanley横向控制器
    基于前轮反馈的路径跟踪控制
    """
    def __init__(self, k=0.8, epsilon=0.3, max_steer=0.5, 
                 filter_alpha=0.2, lookahead_base=1.5, lookahead_gain=0.5):
        """
        Args:
            k: 横向误差增益
            epsilon: 防止低速过度转向的极小常数
            max_steer: 最大转向角限制 (rad)
            filter_alpha: 误差低通滤波系数
            lookahead_base: 前视距离基础值 (m)
            lookahead_gain: 前视距离速度增益
        """
        self.k = k
        self.epsilon = epsilon
        self.max_steer = max_steer
        self.filter_alpha = filter_alpha
        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain
        
        # 滤波状态
        self.filtered_cross_track_error = 0.0
        self.filtered_heading_error = 0.0
        self.current_cross_track_error = 0.0
        self.current_heading_error = 0.0
        
    # FIXME: 可增加前馈控制，同时加PD阻尼抑制高速震荡。
    def compute_steering(self, current_x, current_y, current_yaw, current_v, path, dt=0.05):
        """
        计算转向角
        
        Returns:
            steering_angle: 转向角 (rad)
            target_point: 目标点坐标
            cross_track_error: 横向误差
        """
        if len(path) < 2:
            return 0.0, (current_x, current_y), 0.0
        
        current_pos = np.array([current_x, current_y])
        
        # 找到最近的路径段，同时保存段的完整信息以避免重复计算
        min_dist = float('inf')
        nearest_idx = 0
        target_point = path[0]
        nearest_seg_start = None
        nearest_seg_end = None
        nearest_seg_vec = None
        nearest_seg_len = None
        nearest_seg_unitvec = None
        
        for i in range(len(path) - 1):
            start_point = np.array(path[i])
            end_point = np.array(path[i + 1])
            
            line_vec = end_point - start_point
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                continue
                
            line_unitvec = line_vec / line_len
            vehicle_to_start = current_pos - start_point
            
            projection = np.dot(vehicle_to_start, line_unitvec)
            projection = np.clip(projection, 0, line_len)
            
            nearest_point = start_point + projection * line_unitvec
            dist = np.linalg.norm(current_pos - nearest_point)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                target_point = nearest_point
                # 保存最近段的完整信息
                nearest_seg_start = start_point
                nearest_seg_end = end_point
                nearest_seg_vec = line_vec
                nearest_seg_len = line_len
                nearest_seg_unitvec = line_unitvec
        
        # 计算横向误差（复用已保存的段信息）
        # 前视点从最近投影点开始，如果前视距离很小则等于最近投影点
        lookahead_point = target_point
        
        if nearest_seg_len is None:
            # 所有段都被跳过（长度为0），使用初始值
            cross_track_error = 0.0
            path_yaw = current_yaw
        else:
            vehicle_vec = current_pos - nearest_seg_start
            cross_track_error = np.cross(vehicle_vec, nearest_seg_vec) / nearest_seg_len
            
            # 计算前视距离
            lookahead_distance = self.lookahead_base + self.lookahead_gain * max(current_v, 0.0)
            
            # 计算前视点：从最近投影点开始，沿着路径向前累计距离
            # 如果前视距离只在一个线段内，直接计算；否则需要沿路径累计跨越多个线段
            if lookahead_distance > 1e-3:
                # 计算从target_point到最近线段终点的距离
                dist_to_seg_end = np.linalg.norm(nearest_seg_end - target_point)
                
                if lookahead_distance <= dist_to_seg_end:
                    # 前视距离只在一个线段内，直接计算
                    lookahead_point = target_point + nearest_seg_unitvec * lookahead_distance
                else:
                    # 前视距离跨越多个线段，需要沿路径累计距离
                    current_seg_idx = nearest_idx
                    remaining_distance = lookahead_distance - dist_to_seg_end
                    lookahead_point = nearest_seg_end
                    current_seg_idx += 1
                    
                    while remaining_distance > 1e-3 and current_seg_idx < len(path) - 1:
                        seg_start = np.array(path[current_seg_idx])
                        seg_end = np.array(path[current_seg_idx + 1])
                        seg_vec = seg_end - seg_start
                        seg_len = np.linalg.norm(seg_vec)
                        
                        if seg_len < 1e-6:
                            current_seg_idx += 1
                            continue
                        
                        seg_dir = seg_vec / seg_len
                        
                        if remaining_distance <= seg_len:
                            lookahead_point = seg_start + seg_dir * remaining_distance
                            remaining_distance = 0.0
                        else:
                            remaining_distance -= seg_len
                            lookahead_point = seg_end
                            current_seg_idx += 1
            
            path_yaw = math.atan2(lookahead_point[1] - current_y, lookahead_point[0] - current_x)
        
        # 航向误差
        heading_error = path_yaw - current_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # 保存原始误差值（用于可视化）
        self.current_cross_track_error = cross_track_error
        self.current_heading_error = heading_error
        
        # 低通滤波
        self.filtered_cross_track_error = (
            self.filter_alpha * cross_track_error + 
            (1.0 - self.filter_alpha) * self.filtered_cross_track_error
        )
        self.filtered_heading_error = (
            self.filter_alpha * heading_error + 
            (1.0 - self.filter_alpha) * self.filtered_heading_error
        )
        
        # Stanley控制律
        if current_v < 0.1:
            steer_correction = self.filtered_heading_error
        else:
            # FIXME: self.epsilon 是低速软化处理，但这里用在了非低速场景
            steer_correction = self.filtered_heading_error - np.arctan2(
                self.k * self.filtered_cross_track_error, 
                current_v + self.epsilon
            )
        
        steering_angle = np.clip(steer_correction, -self.max_steer, self.max_steer)
        
        return steering_angle, (float(lookahead_point[0]), float(lookahead_point[1])), cross_track_error


class PIDController:
    """PID速度控制器"""
    def __init__(self, kp, ki, kd, dt, output_limits=(-1, 1), integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def compute(self, error, rate_limit=None):
        proportional = self.kp * error
        
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        derivative = self.kd * (error - self.previous_error) / self.dt
        
        output = proportional + integral + derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        if rate_limit is not None:
            output_change = output - self.previous_output
            if abs(output_change) > rate_limit * self.dt:
                output = self.previous_output + np.sign(output_change) * rate_limit * self.dt
        
        self.previous_error = error
        self.previous_output = output
        
        return output


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
        self.stanley = StanleyController(
            k=0.8,
            epsilon=0.3,
            max_steer=0.5,
            filter_alpha=0.2,
            lookahead_base=1.5,
            lookahead_gain=0.5
        )
        
        # 初始化PID速度控制器
        self.speed_controller = PIDController(
            kp=2.0, ki=0.05, kd=0.5, 
            dt=self.control_dt,
            output_limits=(-2.0, 2.0),
            integral_limit=1.0
        )
        
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
            10
        )
        
        # 车辆物理属性（从vehicle_info获取，如果无法获取则使用默认值）
        # CARLA官方默认值：轴距约2.7m，最大转向角约70度(1.22 rad)
        self.vehicle_wheelbase = 2.7  # m (轴距，默认值)
        self.vehicle_max_steer_angle = 1.22  # rad (最大转向角，默认值约70度)
        self.vehicle_wheel_count = 4  # 默认4轮
        self.vehicle_wheel_info = []  # 存储车轮信息
        self.vehicle_type_id = ""
        self.vehicle_info_received = False
        
        # 发布器
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        
        # 状态变量
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_speed = 0.0  # m/s
        
        self.waypoints = []
        self.current_waypoint_index = 0
        
        self.start_pose = None
        self.goal_pose = None
        self.wait_for_goal = True
        
        self.is_controlling = False
        self.control_timer = None
        
        # 速度平滑
        self.filtered_speed = 0.0
        self.last_speed = 0.0
        self.speed_filter_alpha = 0.3
        
        # 路径曲率
        self.path_curvatures = []
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # 调试计数器
        self.log_counter = 0
        
        # 误差可视化数据存储
        self.error_history_size = 500  # 保存最近500个数据点
        self.error_time_history = deque(maxlen=self.error_history_size)
        self.cross_track_error_history = deque(maxlen=self.error_history_size)
        self.heading_error_history = deque(maxlen=self.error_history_size)
        self.filtered_cross_track_error_history = deque(maxlen=self.error_history_size)
        self.filtered_heading_error_history = deque(maxlen=self.error_history_size)
        self.error_plot_lock = threading.Lock()
        self.start_time = time.time()
        
        # 初始化实时绘图
        self.enable_plotting = True  # 可以通过参数控制是否启用绘图
        if self.enable_plotting:
            self._init_error_plot()
        
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
        import time
        time.sleep(0.1)
        new_loc = self.ego_vehicle.get_location()
        self.get_logger().info(f"移动后位置: ({new_loc.x:.2f}, {new_loc.y:.2f}, {new_loc.z:.2f})")
        self.get_logger().info(f"{GREEN}✓ 车辆已移动到中心线: ({x:.2f}, {y:.2f}){RESET}")
    
    def status_callback(self, msg):
        """车辆状态回调 - 获取速度"""
        self.vehicle_speed = msg.velocity  # m/s
    
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
    
    def odometry_callback(self, msg):
        """里程计回调"""
        with self.data_lock:
            self.vehicle_x = msg.pose.pose.position.x
            self.vehicle_y = msg.pose.pose.position.y
            
            orientation_q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion(
                [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            )
            self.vehicle_yaw = yaw
    
    def waypoints_callback(self, msg):
        """路径点回调"""
        new_waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            new_waypoints.append((x, y))
        
        with self.data_lock:
            self.waypoints = new_waypoints
            
            # 找到最近的路径点作为起始索引
            if len(new_waypoints) > 0:
                min_dist = float('inf')
                closest_idx = 0
                for i, wp in enumerate(new_waypoints):
                    dist = math.sqrt((self.vehicle_x - wp[0])**2 + (self.vehicle_y - wp[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                self.current_waypoint_index = closest_idx
            else:
                self.current_waypoint_index = 0
            
            # 计算路径曲率
            self.path_curvatures = self.compute_curvatures(new_waypoints)
        
        self.get_logger().info(f"{CYAN}✓ 收到路径规划，包含 {len(new_waypoints)} 个路径点{RESET}")
        
        # 如果已经设置了目标点，开始控制
        if not self.wait_for_goal and not self.is_controlling:
            self.start_control()
    
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
        
        self.speed_controller.reset()
        self.filtered_speed = 0.0
        self.last_speed = 0.0
        
        self.get_logger().info("✓ 车辆已停止，等待设置目标点...")
    
    def goal_pose_callback(self, msg):
        """目标点回调"""
        self.get_logger().info("收到目标点，准备启动车辆...")
        self.goal_pose = msg.pose
        self.wait_for_goal = False
        
        # 如果已有路径，开始控制
        if len(self.waypoints) > 0 and not self.is_controlling:
            self.start_control()
    
    def compute_curvatures(self, path):
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
            curvatures.append(float(kappa))
        
        return curvatures
    
    # FIXME: 后期考虑用交互模型，局部避障规划或speed profile规划速度
    def plan_speed(self):
        """速度规划"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return self.min_speed
        
        current_v = self.vehicle_speed
        
        # 基于曲率规划速度
        max_lateral_accel = 2.5
        preview_distance = 5.0 + 1.0 * max(current_v, 0.0)
        
        # 计算前瞻窗口内的最大曲率
        max_kappa = 0.0
        acc_dist = 0.0
        
        for i in range(self.current_waypoint_index, len(self.waypoints) - 1):
            if acc_dist > preview_distance:
                break
            
            if i < len(self.path_curvatures):
                max_kappa = max(max_kappa, abs(self.path_curvatures[i]))
            
            if i + 1 < len(self.waypoints):
                seg = np.linalg.norm(
                    np.array(self.waypoints[i + 1]) - np.array(self.waypoints[i])
                )
                acc_dist += seg
        
        # 基于曲率计算速度
        """
        FIXME: 
        取极大值容易受噪声影响，可以考虑取算数平均值; 
        且规划的速度不连续，max_speed和max_lateral_accel这两者应该由另一方定义，而非直接赋值为常数
        转弯需要设定一个速度下限值
        """
        if max_kappa < 1e-6:
            planned_speed = self.max_speed
        else:
            planned_speed = math.sqrt(max_lateral_accel / max_kappa)
        
        planned_speed = np.clip(planned_speed, self.min_speed, self.max_speed)
        
        # 接近终点减速
        if len(self.waypoints) > 0:
            goal = self.waypoints[-1]
            dist_to_goal = math.sqrt(
                (self.vehicle_x - goal[0])**2 + (self.vehicle_y - goal[1])**2
            )
            
            if dist_to_goal < 5.0:
                # 线性减速
                planned_speed = min(planned_speed, max(0.5, dist_to_goal / 5.0 * planned_speed))
        
        return float(planned_speed)
    
    def _init_error_plot(self):
        """初始化误差实时绘图"""
        try:
            # 设置matplotlib使用非阻塞后端
            plt.ion()  # 开启交互模式
            
            # 创建图形和子图
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle('车辆控制误差实时监控', fontsize=14, fontweight='bold')
            
            # 初始化线条
            # FIXME: 文字乱码
            self.line_cte, = self.ax1.plot([], [], 'b-', label='横向误差 (Cross Track Error)', linewidth=1.5)
            self.line_fcte, = self.ax1.plot([], [], 'r--', label='滤波后横向误差', linewidth=1.5)
            self.line_he, = self.ax2.plot([], [], 'g-', label='航向误差 (Heading Error)', linewidth=1.5)
            self.line_fhe, = self.ax2.plot([], [], 'm--', label='滤波后航向误差', linewidth=1.5)
            
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
            
            plt.tight_layout()
            plt.show(block=False)
            
            self.get_logger().info(f"{GREEN}✓ 误差实时绘图已启动{RESET}")
            
        except Exception as e:
            self.get_logger().error(f"初始化绘图失败: {e}")
            self.enable_plotting = False
    
    def _update_error_plot(self):
        """更新误差绘图"""
        if not self.enable_plotting:
            return
        
        try:
            with self.error_plot_lock:
                if len(self.error_time_history) == 0:
                    return
                
                # 转换为numpy数组
                time_data = np.array(self.error_time_history)
                cte_data = np.array(self.cross_track_error_history)
                fcte_data = np.array(self.filtered_cross_track_error_history)
                he_data = np.array(self.heading_error_history)
                fhe_data = np.array(self.filtered_heading_error_history)
                
                # 更新横向误差图
                self.line_cte.set_data(time_data, cte_data)
                self.line_fcte.set_data(time_data, fcte_data)
                
                # 更新航向误差图
                self.line_he.set_data(time_data, he_data)
                self.line_fhe.set_data(time_data, fhe_data)
                
                # 自动调整坐标轴范围
                if len(time_data) > 0:
                    time_range = [max(0, time_data[-1] - 30), time_data[-1] + 1]  # 显示最近30秒
                    
                    # 横向误差范围
                    if len(cte_data) > 0:
                        cte_range = [min(np.min(cte_data), np.min(fcte_data)) - 0.2,
                                    max(np.max(cte_data), np.max(fcte_data)) + 0.2]
                        self.ax1.set_xlim(time_range)
                        self.ax1.set_ylim(cte_range)
                    
                    # 航向误差范围
                    if len(he_data) > 0:
                        he_range = [min(np.min(he_data), np.min(fhe_data)) - 0.2,
                                   max(np.max(he_data), np.max(fhe_data)) + 0.2]
                        self.ax2.set_xlim(time_range)
                        self.ax2.set_ylim(he_range)
                
                # 刷新图形
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        except Exception as e:
            self.get_logger().warn(f"更新绘图失败: {e}")
    
    def _add_error_data(self, cross_track_error, heading_error, filtered_cross_track_error, filtered_heading_error):
        """添加误差数据到历史记录"""
        if not self.enable_plotting:
            return
        
        try:
            current_time = time.time() - self.start_time
            
            with self.error_plot_lock:
                self.error_time_history.append(current_time)
                self.cross_track_error_history.append(cross_track_error)
                self.heading_error_history.append(heading_error)
                self.filtered_cross_track_error_history.append(filtered_cross_track_error)
                self.filtered_heading_error_history.append(filtered_heading_error)
        except Exception as e:
            self.get_logger().warn(f"添加误差数据失败: {e}")
    
    def _update_error_visualization(self, cross_track_error, heading_error, filtered_cross_track_error, filtered_heading_error):
        """更新误差可视化（仅绘图，不发布消息）"""
        # 添加到绘图数据
        self._add_error_data(cross_track_error, heading_error, filtered_cross_track_error, filtered_heading_error)
        
        # 更新绘图（每5次更新一次，避免过于频繁）
        if len(self.error_time_history) % 5 == 0:
            self._update_error_plot()
    
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
            with self.data_lock:
                if not self.waypoints:
                    return
                
                waypoints = self.waypoints.copy()
                current_x = self.vehicle_x
                current_y = self.vehicle_y
                current_yaw = self.vehicle_yaw
            
            current_v = self.vehicle_speed
            
            # 检查是否到达目标
            if len(waypoints) > 0:
                goal = waypoints[-1]
                dist_to_goal = math.sqrt((current_x - goal[0])**2 + (current_y - goal[1])**2)
                
                if dist_to_goal < 1.0:
                    self.get_logger().info(f"{GREEN}🎯 已到达目标点！{RESET}")
                    self.publish_stop()
                    self.is_controlling = False
                    if self.control_timer:
                        self.control_timer.cancel()
                    return
            
            # 转换路径格式
            path_points = [(wp[0], wp[1]) for wp in waypoints]
            
            # 使用Stanley控制器计算转向角
            steering_angle, target_point, cross_track_error = self.stanley.compute_steering(
                current_x, current_y, current_yaw, current_v, path_points, self.control_dt
            )
            
            # 更新误差可视化（绘图）
            self._update_error_visualization(
                self.stanley.current_cross_track_error,
                self.stanley.current_heading_error,
                self.stanley.filtered_cross_track_error,
                self.stanley.filtered_heading_error
            )
            
            # 更新当前路径点索引
            # FIXME: 与compute_steering中的target_point一致，计算重复，且target_point计算复杂度较高，检索范围较大，考虑优化
            min_dist = float('inf')
            for i, wp in enumerate(waypoints):
                dist = math.sqrt((wp[0] - target_point[0])**2 + (wp[1] - target_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    self.current_waypoint_index = i
            
            # 速度规划
            planned_speed = self.plan_speed()
            
            # PID速度控制
            speed_error = planned_speed - current_v
            acceleration = self.speed_controller.compute(speed_error, rate_limit=3.0)
            
            # 速度平滑
            # FIXME: 这里的平滑计算没有任何意义，速度规划已经全交由self.plan_speed()处理
            # target_speed = current_v + acceleration * self.control_dt
            # max_speed_change = self.max_acceleration * self.control_dt
            # min_speed_change = self.max_deceleration * self.control_dt
            
            # speed_change = target_speed - self.last_speed
            # if speed_change > max_speed_change:
            #     target_speed = self.last_speed + max_speed_change
            # elif speed_change < min_speed_change:
            #     target_speed = self.last_speed + min_speed_change
            
            # self.filtered_speed = (
            #     self.speed_filter_alpha * target_speed + 
            #     (1.0 - self.speed_filter_alpha) * self.filtered_speed
            # )
            # self.filtered_speed = max(0.0, self.filtered_speed)
            # self.last_speed = self.filtered_speed
            
            # 转换为油门/刹车
            # FIXME: 输出为动力学模型输出，但输入是运动学模型输入，需要转换
            if acceleration >= 0:
                throttle = min(acceleration / self.max_acceleration, 1.0)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(abs(acceleration) / abs(self.max_deceleration), 1.0)
            
            # 接近终点时刹车
            if dist_to_goal < 3.0:
                brake = max(brake, (3.0 - dist_to_goal) / 3.0 * 0.5)
                throttle = throttle * (dist_to_goal / 3.0)
            
            # 发布控制命令
            # steering_angle 是弧度，CARLA steer 范围是 [-1, 1]
            # 假设最大转向角约为 70 度 (1.22 rad)
            max_steer_rad = 1.22  # CARLA 车辆最大转向角
            normalized_steer = steering_angle / max_steer_rad
            
            # CARLA转向方向：正值=右转，负值=左转
            # Stanley输出：正值=左转，负值=右转（标准右手坐标系）
            # 需要取反
            normalized_steer = -normalized_steer
            
            msg = CarlaEgoVehicleControl()
            msg.throttle = float(np.clip(throttle, 0.0, 1.0))
            msg.steer = float(np.clip(normalized_steer, -1.0, 1.0))
            msg.brake = float(np.clip(brake, 0.0, 1.0))
            self.control_pub.publish(msg)
            
            # 调试输出
            self.log_counter += 1
            if self.log_counter % 20 == 0:  # 每秒输出一次
                progress = (self.current_waypoint_index / len(waypoints)) * 100 if len(waypoints) > 0 else 0
                # 计算航向误差用于调试
                if len(waypoints) > self.current_waypoint_index:
                    target_wp = waypoints[min(self.current_waypoint_index + 1, len(waypoints) - 1)]
                    path_yaw_debug = math.atan2(target_wp[1] - current_y, target_wp[0] - current_x)
                    heading_err_debug = math.degrees(path_yaw_debug - current_yaw)
                else:
                    heading_err_debug = 0.0
                self.get_logger().info(
                    f"控制: {progress:.1f}% | 速度: {current_v*3.6:.1f} km/h | "
                    f"转向: {math.degrees(steering_angle):.1f}° (steer={normalized_steer:.3f}) | "
                    f"横向误差: {cross_track_error:.3f}m | 航向误差: {heading_err_debug:.1f}° | "
                    f"车辆Yaw: {math.degrees(current_yaw):.1f}° | 到终点: {dist_to_goal:.1f}m"
                )
                
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def start_control(self):
        """开始控制"""
        if not self.waypoints:
            self.get_logger().warn("没有可用的路径点！")
            return
        
        self.is_controlling = True
        self.current_waypoint_index = 0
        self.log_counter = 0
        self.speed_controller.reset()
        self.filtered_speed = 0.0
        self.last_speed = 0.0
        
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)
        
        self.get_logger().info(f"{GREEN}🚀 启动车辆控制！路径包含 {len(self.waypoints)} 个点{RESET}")
    
    def run(self):
        """主循环"""
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                
                if self.start_pose is None:
                    time.sleep(0.1)
                    continue
                
                if self.wait_for_goal or self.goal_pose is None:
                    time.sleep(0.1)
                    continue
                
                if not self.waypoints:
                    time.sleep(0.1)
                    continue
                
                if not self.is_controlling:
                    self.start_control()
                
                time.sleep(0.01)
                
        except Exception as e:
            self.get_logger().error(f"主循环错误: {e}")
            raise
        finally:
            self.publish_stop()


def main(args=None):
    rclpy.init(args=args)
    controller = CarlaVehicleControl()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.get_logger().info("正在关闭...")
    except Exception as e:
        controller.get_logger().error(f"错误: {e}")
    finally:
        if hasattr(controller, 'control_timer') and controller.control_timer:
            controller.control_timer.cancel()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
