#!/usr/bin/env python3
"""
路径平滑节点 - 使用二次规划方法对局部路径进行平滑

功能：
1. 订阅车辆当前位置索引（nearest_idx）和完整路径（waypoints）
2. 以10Hz频率对局部路径段进行QP平滑
3. 发布平滑后的路径段供可视化
4. 发布更新后的完整路径供车辆控制使用
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import scipy.sparse as sp
import time
import osqp


class PathSmootherNode(Node):
    """路径平滑节点"""
    
    def __init__(self):
        super().__init__('path_smoother_node')
        
        # 平滑参数配置
        self.smooth_w1 = 0.3   # 相似权重：控制与原始路径的接近程度
        self.smooth_w2 = 1.0   # 平滑权重：控制曲率平滑度
        self.smooth_w3 = 0.1   # 累减权重：控制点间距离均匀度
        self.smooth_deviation_limit = 0.5  # 最大允许偏差 (m)
        
        # 点间距（已知）
        self.point_spacing = 1  # 米
        
        # 前向/后向距离参数（参考Apollo）
        self.look_forward_time_sec = 8.0  # 秒
        self.look_forward_short_distance = 180.0  # 米
        self.look_backward_distance = 50.0  # 米
        
        # 默认前向距离（速度未接收时使用）
        self.default_forward_distance = 150.0  # 米
        
        # 平滑频率：10Hz
        self.smooth_interval = 0.1  # 0.1秒 = 10Hz
        
        # 状态变量
        self.current_waypoints = []
        self.current_nearest_idx = 0
        self.current_velocity = 0.0  # 当前速度 (m/s)
        self.last_nearest_idx = -1  # 上一次的nearest_idx，用于增量平滑
        self.last_smooth_end_idx = -1  # 上一次平滑的结束索引
        self.last_smooth_time = 0.0
        self.waypoints_received = False
        self.nearest_idx_received = False
        self.velocity_received = False
        
        # 订阅器
        self.waypoints_sub = self.create_subscription(
            Path,
            '/carla/ego_vehicle/waypoints',
            self.waypoints_callback,
            10
        )
        
        self.nearest_idx_sub = self.create_subscription(
            Int32,
            '/vehicle_control/nearest_idx',
            self.nearest_idx_callback,
            10
        )
        
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )
        
        # 发布器：发布增量更新（起始索引 + 更新的点列表）
        self.path_update_index_pub = self.create_publisher(
            Int32,
            '/path_smoothing/update_index',
            10
        )
        
        self.path_update_points_pub = self.create_publisher(
            Path,
            '/path_smoothing/update_points',
            10
        )
        
        # 发布器：发布平滑前的路径点Marker用于RViz可视化
        self.original_waypoints_marker_pub = self.create_publisher(
            MarkerArray,
            '/path_smoothing/original_waypoints',
            10
        )
        
        # 发布器：发布平滑后的路径点Marker用于RViz可视化
        self.smoothed_waypoints_marker_pub = self.create_publisher(
            MarkerArray,
            '/path_smoothing/smoothed_waypoints',
            10
        )
        
        # Marker ID计数器（用于唯一标识每个Marker）
        self.original_marker_id_counter = 0
        self.smoothed_marker_id_counter = 0
        
        # 定时器：10Hz频率执行平滑
        self.smooth_timer = self.create_timer(self.smooth_interval, self.smooth_path_callback)
        
        self.get_logger().info("路径平滑节点已启动")
    
    def waypoints_callback(self, msg):
        """路径点回调"""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            waypoints.append((x, y))
        
        # 如果路径发生变化，重置平滑状态
        if len(waypoints) != len(self.current_waypoints):
            self.last_nearest_idx = -1
            self.last_smooth_end_idx = -1
            # 重置Marker ID计数器
            self.original_marker_id_counter = 0
            # 发布初始waypoints的Marker到RViz（平滑前的路径点）
            self.publish_waypoints_markers(waypoints, is_initial=True)
        
        self.current_waypoints = waypoints
        self.waypoints_received = True
    
    def nearest_idx_callback(self, msg):
        """最近路径点索引回调"""
        self.current_nearest_idx = msg.data
        self.nearest_idx_received = True
    
    def odometry_callback(self, msg):
        """里程计回调：获取车辆速度"""
        # 从twist中获取速度（m/s）
        linear_velocity = msg.twist.twist.linear.x
        self.current_velocity = abs(linear_velocity)  # 取绝对值
        self.velocity_received = True
    
    def calculate_look_forward_distance(self, velocity):
        """
        计算前向距离（参考Apollo的LookForwardDistance）
        
        参数：
        velocity: 车辆速度 (m/s)
        
        返回：
        forward_distance: 前向距离 (m)
        """
        forward_distance = max(velocity * self.look_forward_time_sec, self.look_forward_short_distance)
        return forward_distance
    
    def distance_to_index(self, distance):
        """
        距离转索引（点间距0.25m）
        
        参数：
        distance: 距离 (m)
        
        返回：
        num_points: 索引数量
        """
        return int(distance / self.point_spacing)
    
    def index_to_distance(self, index):
        """
        索引转距离
        
        参数：
        index: 索引
        
        返回：
        distance: 距离 (m)
        """
        return index * self.point_spacing
    
    def build_second_diff_matrix(self, n):
        """构建二阶差分矩阵（针对交错排列的变量：[x1, y1, x2, y2, ...]）"""
        total_rows = 2 * (n-2)
        total_cols = 2 * n
        
        rows = []
        cols = []
        data = []
        
        # x坐标的二阶差分（在偶数行）
        for i in range(n-2):
            row_idx = 2 * i
            rows.append(row_idx)
            cols.append(2 * i)
            data.append(1)
            
            rows.append(row_idx)
            cols.append(2 * (i+1))
            data.append(-2)
            
            rows.append(row_idx)
            cols.append(2 * (i+2))
            data.append(1)
        
        # y坐标的二阶差分（在奇数行）
        for i in range(n-2):
            row_idx = 2 * i + 1
            rows.append(row_idx)
            cols.append(2 * i + 1)
            data.append(1)
            
            rows.append(row_idx)
            cols.append(2 * (i+1) + 1)
            data.append(-2)
            
            rows.append(row_idx)
            cols.append(2 * (i+2) + 1)
            data.append(1)
        
        D2 = sp.csc_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
        return D2
    
    def build_first_diff_matrix(self, n):
        """构建一阶差分矩阵（针对交错排列的变量）"""
        total_rows = 2 * (n-1)
        total_cols = 2 * n
        
        rows = []
        cols = []
        data = []
        
        # x坐标的一阶差分（在偶数行）
        for i in range(n-1):
            row_idx = 2 * i
            rows.append(row_idx)
            cols.append(2 * i)
            data.append(-1)
            
            rows.append(row_idx)
            cols.append(2 * (i+1))
            data.append(1)
        
        # y坐标的一阶差分（在奇数行）
        for i in range(n-1):
            row_idx = 2 * i + 1
            rows.append(row_idx)
            cols.append(2 * i + 1)
            data.append(-1)
            
            rows.append(row_idx)
            cols.append(2 * (i+1) + 1)
            data.append(1)
        
        D1 = sp.csc_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
        return D1
    
    def build_qp_matrices(self, original_points, w1=1.0, w2=0.1, w3=0.01):
        """
        构建QP问题的P和q矩阵
        
        代价函数：
        J = w1 * Σ(x_i - x_ir)^2 + (y_i - y_ir)^2        # 相似代价
            + w2 * Σ(x_i - 2x_{i+1} + x_{i+2})^2         # 平滑代价（二阶差分）
            + w3 * Σ(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 # 累减代价（一阶差分）
        """
        n = len(original_points)
        total_vars = 2 * n
        
        # 1. 相似代价的贡献
        P_similar = 2 * w1 * sp.eye(total_vars)
        q = -2 * w1 * original_points.flatten()
        
        # 2. 平滑代价的贡献（二阶差分）
        D2 = self.build_second_diff_matrix(n)
        P_smooth = 2 * w2 * (D2.T @ D2)
        
        # 3. 累减代价的贡献（一阶差分）
        D1 = self.build_first_diff_matrix(n)
        P_reduction = 2 * w3 * (D1.T @ D1)
        
        # 4. 合并所有项
        P = P_similar + P_smooth + P_reduction
        
        return P, q
    
    def add_path_constraints(self, original_points, overlap_indices=None, new_region_indices=None, deviation_limit=0.5, is_end_point=False):
        """
        为路径平滑问题增加线性约束
        
        参数：
        original_points: 原始路径点数组
        overlap_indices: 重叠区域点索引列表（等式约束），如果为None则没有重叠区域
        new_region_indices: 新区域点索引列表（不等式约束），如果为None则所有非重叠点都使用不等式约束
        deviation_limit: 最大允许偏差（仅用于新区域）
        is_end_point: 是否包含路径终点（终点必须使用等式约束）
        """
        n = len(original_points)
        total_vars = 2 * n
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        constraint_idx = 0
        
        # 确定需要等式约束的点（重叠区域 + 终点）
        equality_constraint_set = set()
        if overlap_indices is not None:
            equality_constraint_set.update(overlap_indices)
        if is_end_point and n > 0:
            equality_constraint_set.add(n - 1)  # 最后一个点
        
        # 确定新区域的点（如果指定了new_region_indices，则只对这些点使用不等式约束）
        if new_region_indices is not None:
            new_region_set = set(new_region_indices)
        else:
            # 如果没有指定，则所有非重叠、非终点的点都使用不等式约束
            new_region_set = set(range(n)) - equality_constraint_set
        
        # 1. 对重叠区域和终点进行等式约束（坐标不变）
        for idx in equality_constraint_set:
            if 0 <= idx < n:
                # x_idx = x_original
                rows.append(constraint_idx)
                cols.append(2*idx)
                data.append(1)
                l_list.append(original_points[idx, 0])
                u_list.append(original_points[idx, 0])
                constraint_idx += 1
                
                # y_idx = y_original
                rows.append(constraint_idx)
                cols.append(2*idx + 1)
                data.append(1)
                l_list.append(original_points[idx, 1])
                u_list.append(original_points[idx, 1])
                constraint_idx += 1
        
        # 2. 边界约束：对新区域点，不能偏离原始位置太远
        if deviation_limit > 0:
            for i in new_region_set:
                if 0 <= i < n:
                    # x_i 的双界约束
                    rows.append(constraint_idx)
                    cols.append(2*i)
                    data.append(1)
                    l_list.append(original_points[i, 0] - deviation_limit)
                    u_list.append(original_points[i, 0] + deviation_limit)
                    constraint_idx += 1
                    
                    # y_i 的双界约束
                    rows.append(constraint_idx)
                    cols.append(2*i + 1)
                    data.append(1)
                    l_list.append(original_points[i, 1] - deviation_limit)
                    u_list.append(original_points[i, 1] + deviation_limit)
                    constraint_idx += 1
        
        m = constraint_idx
        A = sp.csc_matrix((data, (rows, cols)), shape=(m, total_vars))
        l = np.array(l_list)
        u = np.array(u_list)
        
        return A, l, u
    
    def smooth_local_path(self, path_segment, overlap_indices=None, new_region_indices=None, is_end_point=False):
        """
        使用QP平滑局部路径段
        
        参数：
        path_segment: (n, 2) 局部路径点数组
        overlap_indices: 重叠区域点索引列表（在path_segment中的索引，等式约束）
        new_region_indices: 新区域点索引列表（在path_segment中的索引，不等式约束）
        is_end_point: 是否包含路径终点（终点必须使用等式约束）
        
        返回：
        smoothed_points: (n, 2) 平滑后的路径点，如果失败则返回原始路径
        """
        if len(path_segment) < 3:
            return path_segment
        
        try:
            # 转换为numpy数组
            original_points = np.array(path_segment)
            
            # 构建QP矩阵
            P, q = self.build_qp_matrices(
                original_points, 
                self.smooth_w1, 
                self.smooth_w2, 
                self.smooth_w3
            )
            
            # 添加约束
            A, l, u = self.add_path_constraints(
                original_points, 
                overlap_indices=overlap_indices,
                new_region_indices=new_region_indices,
                deviation_limit=self.smooth_deviation_limit,
                is_end_point=is_end_point
            )
            
            # 求解QP问题
            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
            result = prob.solve()
            
            # 检查求解状态
            if result.info.status_val == 1 and result.x is not None:  # 1表示solved
                smoothed_points = result.x.reshape(-1, 2)
                return smoothed_points
            else:
                return path_segment
        except Exception as e:
            self.get_logger().warn(f"路径平滑失败: {e}")
            return path_segment
    
    def get_local_path_segment(self, waypoints, nearest_idx, last_nearest_idx=-1, velocity=None):
        """
        获取局部路径段（支持增量平滑，基于距离而非点数）
        
        参数：
        waypoints: 当前完整路径
        nearest_idx: 当前最近路径点索引
        last_nearest_idx: 上一次的最近路径点索引（-1表示第一次）
        velocity: 车辆速度（m/s），如果为None则使用self.current_velocity或默认值
        
        返回：
        local_segment: 需要平滑的路径段（包含重叠区域+新区域）
        start_idx: 在完整路径中的起始索引（重叠区域起点）
        end_idx: 在完整路径中的结束索引
        overlap_indices: 重叠区域在local_segment中的索引列表（等式约束）
        new_region_indices: 新区域在local_segment中的索引列表（不等式约束）
        is_incremental: 是否是增量平滑
        is_end_point: 是否包含路径终点
        """
        n = len(waypoints)
        if n == 0:
            return None, None, None, None, None, False, False
        
        # 确保nearest_idx在有效范围内
        nearest_idx = max(0, min(nearest_idx, n - 1))
        
        # 获取速度（优先使用传入参数，否则使用当前速度，最后使用默认值）
        if velocity is None:
            if self.velocity_received:
                velocity = self.current_velocity
            else:
                velocity = 0.0  # 使用默认距离计算
        
        # 计算前向距离和平滑长度
        if velocity > 0:
            forward_distance = self.calculate_look_forward_distance(velocity)
        else:
            forward_distance = self.default_forward_distance
        
        overlap_length = forward_distance / 2.0  # 重叠区域为平滑长度的一半
        
        # 转换为索引
        backward_idx = self.distance_to_index(self.look_backward_distance)
        forward_idx = self.distance_to_index(forward_distance)
        overlap_idx = self.distance_to_index(overlap_length)
        
        # 第一次平滑：没有重叠区域
        if last_nearest_idx < 0:
            start_idx = max(0, nearest_idx - backward_idx)
            end_idx = min(n, nearest_idx + forward_idx + 1)
            
            # 确保至少3个点
            if end_idx - start_idx < 3:
                if start_idx == 0:
                    end_idx = min(n, start_idx + 3)
                elif end_idx == n:
                    start_idx = max(0, end_idx - 3)
                else:
                    end_idx = min(n, start_idx + 3)
            
            local_segment = waypoints[start_idx:end_idx]
            is_end_point = (end_idx >= n - 1)
            
            # 第一次平滑没有重叠区域，所有点都是新区域
            overlap_indices = None
            new_region_indices = list(range(len(local_segment)))
            
            return local_segment, start_idx, end_idx, overlap_indices, new_region_indices, False, is_end_point
        
        # 增量平滑：需要重叠区域
        # 检查是否前进
        if nearest_idx <= last_nearest_idx:
            # 后退或不变，不需要平滑
            return None, None, None, None, None, False, False
        
        # 使用保存的上次平滑结束索引
        if self.last_smooth_end_idx < 0:
            # 如果没有保存，则基于last_nearest_idx估算（向后兼容）
            last_smooth_end_idx = last_nearest_idx + forward_idx
        else:
            last_smooth_end_idx = self.last_smooth_end_idx
        
        # 计算重叠区域（从上次平滑的末尾向前）
        overlap_start_idx = max(0, last_smooth_end_idx - overlap_idx)
        overlap_end_idx = min(n - 1, last_smooth_end_idx)
        
        # 确保重叠区域至少有一个点
        if overlap_end_idx <= overlap_start_idx:
            overlap_start_idx = max(0, overlap_end_idx - 1)
        
        # 计算新区域的起点和终点
        new_region_start_idx = overlap_end_idx + 1
        new_region_end_idx = min(n - 1, nearest_idx + forward_idx)
        
        # 确保新区域至少有一个点
        if new_region_end_idx < new_region_start_idx:
            new_region_end_idx = min(n - 1, new_region_start_idx + 1)
        
        # 构建local_segment：重叠区域 + 新区域
        overlap_points = waypoints[overlap_start_idx:overlap_end_idx + 1]
        new_region_points = waypoints[new_region_start_idx:new_region_end_idx + 1]
        
        # 确保至少3个点
        if len(overlap_points) + len(new_region_points) < 3:
            return None, None, None, None, None, False, False
        
        local_segment = overlap_points + new_region_points
        
        # 计算索引范围（在local_segment中）
        num_overlap_points = len(overlap_points)
        overlap_indices = list(range(num_overlap_points))
        new_region_indices = list(range(num_overlap_points, len(local_segment)))
        
        # 判断是否包含终点
        is_end_point = (new_region_end_idx >= n - 1)
        
        # 计算在完整路径中的范围
        start_idx = overlap_start_idx
        end_idx = new_region_end_idx + 1
        
        return local_segment, start_idx, end_idx, overlap_indices, new_region_indices, True, is_end_point
    
    def smooth_path_callback(self):
        """定时器回调：执行路径平滑"""
        # 检查是否有必要的数据
        if not self.waypoints_received or not self.nearest_idx_received:
            return
        
        if len(self.current_waypoints) < 3:
            return
        
        # 检查时间间隔（10Hz频率控制）
        current_time = time.time()
        
        # 获取局部路径段（支持增量平滑）
        local_segment, start_idx, end_idx, overlap_indices, new_region_indices, is_incremental, is_end_point = self.get_local_path_segment(
            self.current_waypoints, 
            self.current_nearest_idx,
            self.last_nearest_idx,
            velocity=self.current_velocity if self.velocity_received else None
        )
        
        if local_segment is None or len(local_segment) < 3:
            self.last_nearest_idx = self.current_nearest_idx
            self.last_smooth_time = current_time
            return
        
        # 保存本次平滑的结束索引（用于下次增量平滑）
        self.last_smooth_end_idx = end_idx - 1  # end_idx是exclusive的，所以减1
        
        # 执行平滑
        smoothed_segment = self.smooth_local_path(
            local_segment, 
            overlap_indices=overlap_indices,
            new_region_indices=new_region_indices,
            is_end_point=is_end_point
        )
        
        # 确保返回的是numpy数组
        if not isinstance(smoothed_segment, np.ndarray):
            smoothed_segment = np.array(smoothed_segment)
        
        # 发布增量更新：起始索引 + 需要更新的点
        self.publish_path_update(
            start_idx, 
            smoothed_segment, 
            overlap_indices, 
            new_region_indices,
            is_incremental,
            is_end_point
        )
        
        # 更新状态
        self.last_nearest_idx = self.current_nearest_idx
        self.last_smooth_time = current_time
    
    def publish_path_update(self, start_idx, smoothed_segment, overlap_indices, new_region_indices, is_incremental, is_end_point):
        """
        发布路径增量更新
        
        参数：
        start_idx: 在全局路径中的起始索引（重叠区域起点）
        smoothed_segment: 平滑后的路径段（包含重叠区域+新区域）
        overlap_indices: 重叠区域索引列表（不发布）
        new_region_indices: 新区域索引列表（需要发布）
        is_incremental: 是否是增量平滑
        is_end_point: 是否包含路径终点
        """
        # 确定需要更新的点（只发布新区域，跳过重叠区域）
        if is_incremental and overlap_indices is not None:
            num_overlap_points = len(overlap_indices)
            points_to_update = smoothed_segment[num_overlap_points:]
            update_start_idx = start_idx + num_overlap_points
        else:
            # 第一次平滑：发布所有点
            points_to_update = smoothed_segment
            update_start_idx = start_idx
        
        # print(f"points_to_update: {len(points_to_update)}")
        # 如果没有需要更新的点，不发布（平滑失败的情况）
        if len(points_to_update) == 0:
            return
        
        # 发布起始索引
        index_msg = Int32()
        index_msg.data = update_start_idx
        self.path_update_index_pub.publish(index_msg)
        
        # 发布更新的点列表
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in points_to_update:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(point[0])
            pose_stamped.pose.position.y = float(point[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        self.path_update_points_pub.publish(path_msg)
        
        # 发布更新的路径点Marker到RViz
        self.publish_waypoints_markers(points_to_update, is_initial=False)
    
    def publish_waypoints_markers(self, waypoints, is_initial=False):
        """
        发布路径点Marker用于RViz可视化
        
        参数：
        waypoints: 路径点列表，可以是[(x,y), ...]或numpy数组
        is_initial: 是否是初始路径点（True表示平滑前的路径点，False表示平滑后的路径点）
        """
        if len(waypoints) == 0:
            return
        
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        
        # 根据is_initial选择发布器和命名空间
        if is_initial:
            # 平滑前的路径点
            publisher = self.original_waypoints_marker_pub
            ns = "original_waypoints"
            marker_id_counter = self.original_marker_id_counter
            
            # 如果是初始路径点，先清除之前的Marker
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = current_time
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.original_marker_id_counter = 0
            marker_id_counter = 0
        else:
            # 平滑后的路径点
            publisher = self.smoothed_waypoints_marker_pub
            ns = "smoothed_waypoints"
            marker_id_counter = self.smoothed_marker_id_counter
        
        # 为每个路径点创建Marker
        for waypoint in waypoints:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.ns = ns
            marker.id = marker_id_counter
            marker_id_counter += 1
            
            # 设置位置
            if isinstance(waypoint, (list, tuple)):
                marker.pose.position.x = float(waypoint[0])
                marker.pose.position.y = float(waypoint[1])
            else:
                # numpy数组
                marker.pose.position.x = float(waypoint[0])
                marker.pose.position.y = float(waypoint[1])
            
            marker.pose.position.z = 0.2  # 稍微抬高，避免与地面重叠
            marker.pose.orientation.w = 1.0
            
            # 设置大小和颜色
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # 平滑前的路径点用绿色，平滑后的路径点用红色
            if is_initial:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            marker.color.a = 1.0
            
            # 永久显示（lifetime=0表示永久）
            marker.lifetime.sec = 0
            
            marker_array.markers.append(marker)
        
        # 更新计数器
        if is_initial:
            self.original_marker_id_counter = marker_id_counter
        else:
            self.smoothed_marker_id_counter = marker_id_counter
        
        # 发布到对应的话题
        publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PathSmootherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n路径平滑节点已关闭")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
