#!/usr/bin/env python3
"""
地图显示与路径规划控制器 (Map Display and Path Planning Controller)

功能：
1. 地图显示：使用lanelet2库加载OSM文件并在RViz2上显示地图（道路边界、虚线、红绿灯等）
2. 路径规划：基于道路中心线使用A*算法规划路径
3. 自动调整：起始点和目标点自动调整到最近的道路中心线
4. 可视化：发布车辆、路径、waypoint等marker

主要特性：
- 使用lanelet2库加载地图
- 从lanelet2地图提取道路中心线用于路径规划
- 起点/终点自动对齐到道路中心线
- 路径完全沿着道路中心线
- A*算法基于道路拓扑搜索最优路径
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose, Twist, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.msg import Log as RosLog
import lanelet2
from lanelet2.projection import UtmProjector
import math
import time
import heapq
from collections import defaultdict
import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import numpy as np

class MapControlNode(Node):
    """
    1. 加载地图，提取中心线
    2. 根据各lanelet起点终点连接情况，用lanelet_id构建拓扑图
    3. 用A*算法做全局规划，并根据lanlet_id拼接路径段
    """
    def __init__(self):
        super().__init__('map_control_node')
        
        # OSM文件路径 - 使用ROS包资源路径，如果包未安装则使用相对路径
        try:
            package_share_directory = get_package_share_directory('map_load')
            self.osm_file = os.path.join(package_share_directory, 'resource', 'rviz_map_zitu._fixed.osm')
        except PackageNotFoundError:
            # 如果包未安装，使用相对于当前文件的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 从 map_load/map_load/map_control.py 到 map_load/resource/
            package_dir = os.path.dirname(os.path.dirname(current_dir))
            self.osm_file = os.path.join(package_dir, 'resource', 'rviz_map_zitu._fixed.osm')
            self.get_logger().warn(f"Package 'map_load' not found in ROS index. Using relative path: {self.osm_file}")
         	 
        # 使用lanelet2库加载地图
        self.lanelet_map = None
        
        # 地图和可视化发布器
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.map_publisher = self.create_publisher(MarkerArray, 'lanelet2_map', qos_profile)

        # 车辆状态
        self.initial_pose = None
        self.goal_pose = None

        self.vehicle_spawned = False

        self.current_pose = None
        self.current_velocity = Twist()
        
        # 路径发布控制
        self.path_published = False
        
        # 起点和终点的lanelet_id（用于路径规划）
        self.start_lanelet_id = None
        self.goal_lanelet_id = None
        
        # 停车检测
        self._stopped_since = None
        self._stop_speed_threshold_ms = 0.05
        self._stop_duration_sec = 0.1
        self.k_zeroval = 1e-4
        
        # 路径规划参数
        self.waypoints = []
        self.current_waypoint_index = 0
        self.path_resolution = 1  # 0.1米，提供足够密集的路径点用于精确控制
        self.waypoint_display_interval = 4  # 每隔4个点显示一个marker（2米间隔）
        
        # 加载地图并提取中心线
        self.load_map()  # 使用lanelet2库加载
        if self.lanelet_map is None:
            self.get_logger().error("❌ 地图加载失败，无法继续初始化")
            raise RuntimeError("地图加载失败")
        self.extract_centerlines()  # 使用lanelet2库提取中心线
        self.build_topology_graph()
        
        # 订阅初始位置和目标点
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            10
        )
        
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # 发布器
        self.status_pub = self.create_publisher(String, '/pose_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/carla/ego_vehicle/waypoints', qos_profile)
        self.adjusted_pose_pub = self.create_publisher(PoseStamped, '/adjusted_initialpose', 10)
        
        # 订阅车辆里程计
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )
        
        # 订阅 /rosout 捕捉停止日志
        self.rosout_sub = self.create_subscription(
            RosLog,
            '/rosout',
            self.rosout_callback,
            100
        )
        
        # 定时器
        self.map_timer = self.create_timer(1.0, self.publish_map)  # 1Hz发布地图
        
        self.get_logger().info("地图显示与路径规划控制器已启动")
        self.get_logger().info(f"地图已加载，包含 {len(self.centerlines)} 条中心线")
        self.get_logger().info("在RViz中使用以下工具:")
        self.get_logger().info("1. '2D Pose Estimate' 设置初始位置（自动调整到道路中心线）")
        self.get_logger().info("2. '2D Nav Goal' 设置目标点（自动调整到道路中心线）")
    
    def load_map(self):
        """使用lanelet2库加载地图"""
        try:
            # 使用UtmProjector加载地图
            projector = UtmProjector(lanelet2.io.Origin(0, 0))
            self.lanelet_map = lanelet2.io.load(self.osm_file, projector)
            self.get_logger().info(f"✅ 使用lanelet2库加载地图成功: {self.osm_file}")
        except Exception as e:
            self.get_logger().error(f"❌ lanelet2库加载地图失败: {str(e)}")
            raise RuntimeError(f"地图加载失败: {str(e)}")
    
    def _uniform_sample_path(self, path_points):
        """
        先计算路径总长度，然后根据 path_resolution 均匀采样
        
        Args:
            path_points: 原始路径点列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            均匀采样后的路径点列表
        """
        if len(path_points) < 2:
            return path_points
        
        # 1. 计算路径总长度和累积长度
        cumulative_lengths = [0.0]
        total_length = 0.0
        
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            seg_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += seg_length
            cumulative_lengths.append(total_length)
        
        # 如果总长度小于分辨率，直接返回起点和终点
        if total_length < self.path_resolution:
            return [path_points[0], path_points[-1]]
        
        # 2. 根据总长度和 path_resolution 计算采样点数
        # uniform_slice 的思路：将 [0, total_length] 均匀分割成 num+1 个点
        num_samples = max(1, int(total_length / self.path_resolution))
        
        # 3. 均匀采样：在 [0, total_length] 上均匀采样
        sampled_lengths = []
        if num_samples > 0:
            delta = self.path_resolution
            for i in range(num_samples + 1):
                sampled_lengths.append(i * delta)
        
        # 确保最后一个点是路径终点
        if len(sampled_lengths) > 0:
            sampled_lengths[-1] = total_length
        
        # 4. 根据采样点的弧长位置，在路径上找到对应的 (x, y) 坐标
        sampled_points = []
        
        for target_s in sampled_lengths:
            # 找到第一个 cumulative_lengths[i+1] >= target_s 的位置
            left, right = 0, len(cumulative_lengths) - 1
            seg_index = 0
            
            while left < right:
                mid = (left + right) // 2
                if cumulative_lengths[mid + 1] < target_s:
                    left = mid + 1
                else:
                    right = mid
            
            seg_index = left
            
            # 确保 seg_index 不越界
            if seg_index >= len(path_points) - 1:
                seg_index = len(path_points) - 2
            
            # 在当前段内插值
            p1 = path_points[seg_index]
            p2 = path_points[seg_index + 1]
            
            seg_start_s = cumulative_lengths[seg_index]
            seg_end_s = cumulative_lengths[seg_index + 1]
            
            if abs(seg_end_s - seg_start_s) < 1e-6:
                # 段长度为0，直接使用起点
                sampled_points.append(p1)
            else:
                # 计算插值比例
                ratio = (target_s - seg_start_s) / (seg_end_s - seg_start_s)
                ratio = max(0.0, min(1.0, ratio))  # 限制在 [0, 1] 范围内
                
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                sampled_points.append((x, y))
        
        return sampled_points
    
    def extract_centerlines(self):
        """提取lanelet的起始点和终点，用于构建拓扑图"""
        self.centerlines = {}
        self.lanelet_ids = []
        
        # 确保lanelet2库已加载
        if self.lanelet_map is None:
            self.get_logger().error("❌ lanelet2库未加载，无法提取中心线")
            return
                
        # FIXME: 直接将整条路径拼接成一个centerline去采样，不要一个个lanelet处理，否则可能导致路径间有间断
        # TODO: 先粗略的采样再线性插值到0.25m，否则路径会过拟合
        # TODO: plan_speed用简易的，当前曲率用前轮转角得到
        # TODO: 两个断开的点之间三次多项式插值
        for lanelet_ in self.lanelet_map.laneletLayer:
            # 只处理道路类型的 lanelet，排除标线、人行道等
            if "subtype" not in lanelet_.attributes.keys() or lanelet_.attributes["subtype"] == "road":
                lanelet_id = lanelet_.id
                self.lanelet_ids.append(lanelet_id)
                
                left_bound = [lanelet2.geometry.to2D(p) for p in lanelet_.leftBound]
                right_bound = [lanelet2.geometry.to2D(p) for p in lanelet_.rightBound]
                
                # 只计算起始点和终点
                min_len = min(len(left_bound), len(right_bound))
                if min_len > 0:
                    # 起始点
                    start_x = (left_bound[0].x + right_bound[0].x) / 2.0
                    start_y = (left_bound[0].y + right_bound[0].y) / 2.0
                    # 终点
                    end_x = (left_bound[-1].x + right_bound[-1].x) / 2.0
                    end_y = (left_bound[-1].y + right_bound[-1].y) / 2.0
                    
                    # 只保存起始点和终点
                    self.centerlines[lanelet_id] = [(start_x, start_y), (end_x, end_y)]
    
    def build_topology_graph(self):
        """构建Lanelet拓扑连接图 - 优先使用lanelet2库的拓扑信息，然后使用空间匹配"""
        self.graph = defaultdict(list)
        self.lanelet_data = {}
        
        # 使用空间匹配方法构建拓扑
        self.get_logger().info("使用中心线端点空间匹配构建拓扑连接...")        
        lanelet_ends = {}
        for lanelet_id, centerline in self.centerlines.items():
            if len(centerline) > 0:
                start_point = centerline[0]
                end_point = centerline[-1]
                lanelet_ends[lanelet_id] = (start_point, end_point)
        
        connection_threshold = 5.0  # 5米阈值
        spatial_connections = 0
        
        for lanelet1_id, (start1, end1) in lanelet_ends.items():
            for lanelet2_id, (start2, end2) in lanelet_ends.items():
                if lanelet1_id == lanelet2_id:
                    continue
                
                # 检查正向连接：lanelet1的终点到lanelet2的起点
                dist = math.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2)
                
                if dist < connection_threshold:
                    if lanelet2_id not in self.graph[lanelet1_id]:
                        self.graph[lanelet1_id].append(lanelet2_id)
                        spatial_connections += 1
                
                # 检查反向连接：lanelet1的起点到lanelet2的终点
                dist_reverse = math.sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2)
                if dist_reverse < connection_threshold:
                    if lanelet1_id not in self.graph[lanelet2_id]:
                        self.graph[lanelet2_id].append(lanelet1_id)
                        spatial_connections += 1
            
    def publish_map(self):
        """发布地图可视化markers到RViz2（使用lanelet2库）"""
        if self.lanelet_map is None:
            return
        
        marker_array = MarkerArray()
        marker_id = 0
        
        # 绘制lanelet边界（道路车道）
        lanelet_count = 0
        for lanelet_ in self.lanelet_map.laneletLayer:
            # 只处理道路类型的 lanelet
            if "subtype" not in lanelet_.attributes.keys() or lanelet_.attributes["subtype"] == "road":
                # 绘制左边界
                left_marker = Marker()
                left_marker.header.frame_id = "map"
                left_marker.header.stamp = self.get_clock().now().to_msg()
                left_marker.type = Marker.LINE_STRIP
                left_marker.action = Marker.ADD
                left_marker.scale.x = 0.3
                left_marker.color.a = 1.0
                left_marker.color.r = 0.0
                left_marker.color.g = 0.8
                left_marker.color.b = 0.0
                left_marker.ns = "left_bound"
                left_marker.id = marker_id
                
                for point in lanelet2.geometry.to2D(lanelet_.leftBound):
                    left_marker.points.append(Point(x=point.x, y=point.y, z=0.0))
                
                if len(left_marker.points) > 0:
                    marker_array.markers.append(left_marker)
                    marker_id += 1
                
                # 绘制右边界
                right_marker = Marker()
                right_marker.header.frame_id = "map"
                right_marker.header.stamp = self.get_clock().now().to_msg()
                right_marker.type = Marker.LINE_STRIP
                right_marker.action = Marker.ADD
                right_marker.scale.x = 0.3
                right_marker.color.a = 1.0
                right_marker.color.r = 0.0
                right_marker.color.g = 0.8
                right_marker.color.b = 0.0
                right_marker.ns = "right_bound"
                right_marker.id = marker_id
                
                for point in lanelet2.geometry.to2D(lanelet_.rightBound):
                    right_marker.points.append(Point(x=point.x, y=point.y, z=0.0))
                
                if len(right_marker.points) > 0:
                    marker_array.markers.append(right_marker)
                    marker_id += 1
                    lanelet_count += 1
        
        # 绘制虚线（从linestring layer获取）
        dashed_count = 0
        for linestring in self.lanelet_map.lineStringLayer:
            if ('type' in linestring.attributes.keys() and 
                'subtype' in linestring.attributes.keys() and
                linestring.attributes['type'] == 'line_thin' and 
                linestring.attributes['subtype'] == 'dashed'):
                points = []
                for point in lanelet2.geometry.to2D(linestring):
                    points.append((point.x, point.y))
                
                if len(points) >= 2:
                    dashed_marker = Marker()
                    dashed_marker.header.frame_id = "map"
                    dashed_marker.header.stamp = self.get_clock().now().to_msg()
                    dashed_marker.type = Marker.LINE_LIST
                    dashed_marker.action = Marker.ADD
                    dashed_marker.scale.x = 0.3
                    dashed_marker.color.a = 1.0
                    dashed_marker.color.r = 1.0
                    dashed_marker.color.g = 1.0
                    dashed_marker.color.b = 1.0
                    dashed_marker.ns = "dashed_lines"
                    dashed_marker.id = marker_id
                    
                    dash_length = 1.0
                    gap_length = 0.5
                    
                    total_length = 0.0
                    lengths = []
                    for i in range(len(points) - 1):
                        dx = points[i+1][0] - points[i][0]
                        dy = points[i+1][1] - points[i][1]
                        seg_len = math.sqrt(dx*dx + dy*dy)
                        lengths.append(seg_len)
                        total_length += seg_len
                    
                    current_dist = 0.0
                    drawing = True
                    next_switch = dash_length
                    
                    seg_idx = 0
                    seg_start_dist = 0.0
                    last_point = None
                    
                    while current_dist < total_length and seg_idx < len(points) - 1:
                        while seg_idx < len(lengths) and current_dist > seg_start_dist + lengths[seg_idx]:
                            seg_start_dist += lengths[seg_idx]
                            seg_idx += 1
                        
                        if seg_idx >= len(lengths):
                            break
                        
                        t = (current_dist - seg_start_dist) / lengths[seg_idx] if lengths[seg_idx] > 0 else 0
                        t = min(1.0, max(0.0, t))
                        
                        x = points[seg_idx][0] + t * (points[seg_idx+1][0] - points[seg_idx][0])
                        y = points[seg_idx][1] + t * (points[seg_idx+1][1] - points[seg_idx][1])
                        
                        current_point = Point(x=x, y=y, z=0.1)
                        
                        if drawing:
                            if last_point is None:
                                last_point = current_point
                            else:
                                dashed_marker.points.append(last_point)
                                dashed_marker.points.append(current_point)
                                last_point = current_point
                        else:
                            last_point = current_point
                        
                        if current_dist >= next_switch:
                            drawing = not drawing
                            next_switch += gap_length if drawing else dash_length
                        
                        current_dist += 0.1
                    
                    if len(dashed_marker.points) > 0:
                        marker_array.markers.append(dashed_marker)
                        marker_id += 1
                        dashed_count += 1
        
        # 绘制红绿灯（从point layer获取）
        traffic_light_count = 0
        for point in self.lanelet_map.pointLayer:
            if 'type' in point.attributes.keys() and point.attributes['type'] == 'traffic_light':
                traffic_light_marker = Marker()
                traffic_light_marker.header.frame_id = "map"
                traffic_light_marker.header.stamp = self.get_clock().now().to_msg()
                traffic_light_marker.type = Marker.SPHERE
                traffic_light_marker.action = Marker.ADD
                traffic_light_marker.ns = "traffic_lights"
                traffic_light_marker.id = marker_id
                
                point_2d = lanelet2.geometry.to2D(point)
                traffic_light_marker.pose.position.x = point_2d.x
                traffic_light_marker.pose.position.y = point_2d.y
                traffic_light_marker.pose.position.z = 3.0
                traffic_light_marker.pose.orientation.w = 1.0
                
                traffic_light_marker.scale.x = 1.0
                traffic_light_marker.scale.y = 1.0
                traffic_light_marker.scale.z = 1.0
                
                traffic_light_marker.color.a = 1.0
                traffic_light_marker.color.r = 1.0
                traffic_light_marker.color.g = 0.0
                traffic_light_marker.color.b = 0.0
                
                marker_array.markers.append(traffic_light_marker)
                marker_id += 1
                traffic_light_count += 1
        
        self.map_publisher.publish(marker_array)
    
    def astar_path_search(self, start_lanelet_id, goal_lanelet_id):
        """
        使用A*算法搜索从起点到终点的lanelet路径
        """
        # 检查起点和终点是否在拓扑图中
        if start_lanelet_id not in self.graph:
            self.get_logger().warn(f"起点lanelet {start_lanelet_id} 不在拓扑图中")
            return None, None
        
        if goal_lanelet_id not in self.graph:
            self.get_logger().warn(f"终点lanelet {goal_lanelet_id} 不在拓扑图中")
            return None, None
        
        # 如果起点和终点相同
        if start_lanelet_id == goal_lanelet_id:
            return [start_lanelet_id], ['forward']
        
        # 输出调试信息
        start_neighbors = self.graph[start_lanelet_id]
        goal_neighbors = self.graph[goal_lanelet_id]
        self.get_logger().info(f"🔍 A*搜索: 起点lanelet {start_lanelet_id} 有 {len(start_neighbors)} 个邻居: {start_neighbors[:5]}")
        self.get_logger().info(f"🔍 A*搜索: 终点lanelet {goal_lanelet_id} 有 {len(goal_neighbors)} 个邻居: {goal_neighbors[:5]}")
        
        # 检查起点和终点是否在同一个连通分量中（使用BFS）
        if not self._check_connectivity(start_lanelet_id, goal_lanelet_id):
            self.get_logger().error(f"❌ 起点lanelet {start_lanelet_id} 和终点lanelet {goal_lanelet_id} 不在同一个连通分量中！")
            return None, None
        
        # 简化的启发式函数
        def heuristic(lanelet_id):
            if lanelet_id == goal_lanelet_id:
                return 0.0
            return 1.0
        
        # A*搜索（简化版本）
        open_set = [(0, start_lanelet_id)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_lanelet_id] = 0
        visited = set()
        max_iterations = 500
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal_lanelet_id:
                # 重构路径
                path = [goal_lanelet_id]
                while current in came_from:
                    current = came_from[current]
                    path.insert(0, current)
                
                self.get_logger().info(f"✅ A*搜索成功，经过 {iterations} 次迭代，找到路径长度: {len(path)}")
                directions = ['forward'] * len(path)
                return path, directions
            
            if current not in self.graph:
                continue
                
            for neighbor in self.graph[current]:
                if neighbor in visited:
                    continue
                    
                tentative_g = g_score[current] + 1.0
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        if iterations >= max_iterations:
            self.get_logger().warn(f"❌ A*搜索达到最大迭代次数 {max_iterations}，未找到路径")
        else:
            self.get_logger().warn(f"❌ A*搜索完成，访问了 {len(visited)} 个节点，未找到路径")
            # 检查是否访问了终点
            if goal_lanelet_id in visited:
                self.get_logger().warn(f"⚠️ 终点lanelet {goal_lanelet_id} 被访问过，但路径重构失败")
            else:
                self.get_logger().warn(f"⚠️ 终点lanelet {goal_lanelet_id} 未被访问，说明无法从起点到达")
        
        return None, None  # 未找到路径
    
    def _check_connectivity(self, start_id, goal_id):
        """检查两个lanelet是否在同一个连通分量中（使用BFS）"""
        if start_id not in self.graph or goal_id not in self.graph:
            return False
        
        if start_id == goal_id:
            return True
        
        from collections import deque
        queue = deque([start_id])
        visited = {start_id}
        
        while queue:
            current = queue.popleft()
            
            if current == goal_id:
                return True
            
            if current in self.graph:
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return False
        
    
    def _compute_centerline_from_lanelet(self, lanelet_id):
        """从lanelet的leftBound和rightBound计算完整中心线"""
        # 在lanelet_map中找到对应的lanelet
        lanelet_ = None
        for ll in self.lanelet_map.laneletLayer:
            if ll.id == lanelet_id:
                lanelet_ = ll
                break
        
        if lanelet_ is None:
            # self.get_logger().warn(f"未找到lanelet_id={lanelet_id}")
            return []
        
        left_bound = [lanelet2.geometry.to2D(p) for p in lanelet_.leftBound]
        right_bound = [lanelet2.geometry.to2D(p) for p in lanelet_.rightBound]
        
        centerline = []
        min_len = min(len(left_bound), len(right_bound))
        
        for i in range(min_len):
            center_x = (left_bound[i].x + right_bound[i].x) / 2.0
            center_y = (left_bound[i].y + right_bound[i].y) / 2.0
            centerline.append((center_x, center_y))
        
        return centerline
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """计算点到线段的最短距离和最近点"""
        dx = x2 - x1
        dy = y2 - y1
        line_length_sq = dx * dx + dy * dy
        
        if line_length_sq < 1e-6:
            return math.sqrt((px - x1)**2 + (py - y1)**2), (x1, y1)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_length_sq))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        return dist, (proj_x, proj_y)
    
    def find_nearest_centerline_point(self, x, y):
        """找到离指定点最近的道路中心点（精确到线段），返回最近点、距离和lanelet_id"""
        min_dist = float('inf')
        nearest_point = None
        nearest_lanelet_id = None
                
        # 检查每条中心线的线段（centerlines现在只保存起始点和终点）
        for lanelet_id, centerline in self.centerlines.items():
            if len(centerline) >= 2:
                p1 = centerline[0]
                p2 = centerline[1]
                
                dist, proj_point = self.point_to_line_distance(
                    x, y, p1[0], p1[1], p2[0], p2[1]
                )
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = proj_point
                    nearest_lanelet_id = lanelet_id
        
        # 如果距离太远，给出详细警告和诊断信息
        if min_dist > 5.0:  # 超过5米就警告
            self.get_logger().warn(
                f"⚠️ 最近的中心线点距离 {min_dist:.2f} 米 ({min_dist*100:.1f}cm)，"
                f"lanelet_id={nearest_lanelet_id}，"
                f"点击位置: ({x:.2f}, {y:.2f})，"
                f"调整到: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f})"
            )
        
        return nearest_point, min_dist, nearest_lanelet_id
    
    def odom_callback(self, msg: Odometry):
        """监听车辆速度，检测车辆停止后将当前位置设为新的起点"""
        try:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            speed_ms = math.sqrt(vx*vx + vy*vy + vz*vz)
            now = self.get_clock().now().nanoseconds / 1e9

            if speed_ms < self._stop_speed_threshold_ms:
                if self._stopped_since is None:
                    self._stopped_since = now
                elif (now - self._stopped_since) >= self._stop_duration_sec:
                    if self.vehicle_spawned:
                        new_start_pose = Pose()
                        new_start_pose.position.x = msg.pose.pose.position.x
                        new_start_pose.position.y = msg.pose.pose.position.y
                        new_start_pose.position.z = 0.0
                        new_start_pose.orientation = msg.pose.pose.orientation
                        self.current_pose = new_start_pose
                            
                        self.goal_pose = None
                        self.waypoints = []
                        self.current_waypoint_index = 0
                        self.path_published = False
                        
                        self.get_logger().info("🚦 车辆已停止：已将当前位置设为新的起点，請在 RViz2 重新设置新的终点。")
                        self._stopped_since = now + 9999.0
            else:
                self._stopped_since = None
        except Exception:
            pass
    
    def rosout_callback(self, log: RosLog):
        """当 simple_ctrl.py 打印 RViz 风格位姿时，视为已停止并用该位姿作为新起点"""
        try:
            if not hasattr(log, 'name') or not hasattr(log, 'msg'):
                return
            if 'carla_vehicle_control' not in str(log.name):
                return
            text = str(log.msg)
            if 'Setting goal pose: Frame:map, Position(' not in text:
                return

            import re
            m = re.search(r"Position\(([-0-9\.]+),\s*([-0-9\.]+),\s*([-0-9\.]+)\).*Orientation\(0,\s*0,\s*([-0-9\.]+),\s*([-0-9\.]+)\)", text)
            if not m:
                return
            x = float(m.group(1))
            y = float(m.group(2))
            qz = float(m.group(4))
            qw = float(m.group(5))

            new_start_pose = Pose()
            new_start_pose.position.x = x
            new_start_pose.position.y = y
            new_start_pose.position.z = 0.0
            new_start_pose.orientation.x = 0.0
            new_start_pose.orientation.y = 0.0
            new_start_pose.orientation.z = qz
            new_start_pose.orientation.w = qw

            self.current_pose = new_start_pose
            self.vehicle_spawned = True

            self.goal_pose = None
            self.waypoints = []
            self.current_waypoint_index = 0
            self.path_published = False

            self.get_logger().info("📍 捕捉到 simple_ctrl 停止位姿，已更新起点；请在 RViz2 重新设置新的终点。")
        except Exception:
            pass
    
    def initial_pose_callback(self, msg):
        """处理初始位置回调 - 自动调整到道路中心线"""
        self.initial_pose = msg
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        yaw = self.quaternion_to_yaw(orientation)
        
        self.get_logger().info(
            f"收到初始位置: 位置({pose.x:.2f}, {pose.y:.2f}), 偏航角: {math.degrees(yaw):.1f}°"
        )
        
        nearest_point, distance, lanelet_id = self.find_nearest_centerline_point(pose.x, pose.y)
        
        # 保存起点的lanelet_id
        self.start_lanelet_id = lanelet_id
        
        if nearest_point is None:
            self.get_logger().error("❌ 未找到道路中心线！可能的原因：")
            self.get_logger().error("   1. 点击位置离所有道路太远")
            self.get_logger().error("   2. 地图数据有问题")
            self.get_logger().error("   3. 坐标系统不匹配")
            adjusted_pose = msg.pose.pose
        elif distance > 0.1:
            # 根据距离给出不同级别的警告
            if distance > 20.0:  # 超过20米
                self.get_logger().error(
                    f"❌ 起始点偏离道路中心线 {distance*100:.1f}cm ({distance:.2f}米)！"
                )
                self.get_logger().error("   这表示您在RViz中点击的位置离道路非常远")
                self.get_logger().error("   建议：在RViz中放大视图，点击时尽量靠近道路中心线")
            elif distance > 5.0:  # 超过5米
                self.get_logger().warn(
                    f"⚠️ 起始点偏离道路中心线 {distance*100:.1f}cm ({distance:.2f}米)"
                )
                self.get_logger().warn("   建议：在RViz中点击时尽量靠近道路中心线")
            else:
                self.get_logger().warn(
                    f"⚠️ 起始点偏离道路中心线 {distance*100:.1f}cm"
                )
            
            self.get_logger().info(
                f"📍 自动调整到最近的道路中心点: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f})"
            )
            
            adjusted_pose = Pose()
            adjusted_pose.position.x = nearest_point[0]
            adjusted_pose.position.y = nearest_point[1]
            adjusted_pose.position.z = 0.0
            adjusted_pose.orientation = msg.pose.pose.orientation
        else:
            self.get_logger().info(
                f"✓ 起始点在道路中心线上（偏离 {distance*100:.1f}cm）"
            )
            adjusted_pose = msg.pose.pose
        
        # 发布调整后的位置到 /adjusted_initialpose 话题
        adjusted_pose_msg = PoseStamped()
        adjusted_pose_msg.header.stamp = self.get_clock().now().to_msg()
        adjusted_pose_msg.header.frame_id = "map"
        adjusted_pose_msg.pose = adjusted_pose
        self.adjusted_pose_pub.publish(adjusted_pose_msg)
        self.get_logger().info(f"已发布调整后的初始位置到 /adjusted_initialpose: ({adjusted_pose.position.x:.2f}, {adjusted_pose.position.y:.2f})")
        
        self.spawn_vehicle_at_current_pose(adjusted_pose)
        self.publish_status("initial_pose_received")
    
    def goal_pose_callback(self, msg):
        """处理目标点回调 - 自动调整到道路中心线"""
        pose = msg.pose.position
        orientation = msg.pose.orientation
        
        yaw = self.quaternion_to_yaw(orientation)
        
        self.get_logger().info(
            f"收到目标点: 位置({pose.x:.2f}, {pose.y:.2f}), 偏航角: {math.degrees(yaw):.1f}°"
        )
        
        nearest_point, distance, lanelet_id = self.find_nearest_centerline_point(pose.x, pose.y)
        
        # 保存终点的lanelet_id
        self.goal_lanelet_id = lanelet_id
        
        if nearest_point is None:
            self.get_logger().error("❌ 未找到道路中心线！可能的原因：")
            self.get_logger().error("   1. 点击位置离所有道路太远")
            self.get_logger().error("   2. 地图数据有问题")
            self.get_logger().error("   3. 坐标系统不匹配")
            adjusted_goal_pose = msg
        elif distance > 0.1:
            # 根据距离给出不同级别的警告
            if distance > 20.0:  # 超过20米
                self.get_logger().error(
                    f"❌ 目标点偏离道路中心线 {distance*100:.1f}cm ({distance:.2f}米)！"
                )
                self.get_logger().error("   这表示您在RViz中点击的位置离道路非常远")
                self.get_logger().error("   建议：在RViz中放大视图，点击时尽量靠近道路中心线")
            elif distance > 5.0:  # 超过5米
                self.get_logger().warn(
                    f"⚠️ 目标点偏离道路中心线 {distance*100:.1f}cm ({distance:.2f}米)"
                )
                self.get_logger().warn("   建议：在RViz中点击时尽量靠近道路中心线")
            else:
                self.get_logger().warn(
                    f"⚠️ 目标点偏离道路中心线 {distance*100:.1f}cm"
                )
            
            self.get_logger().info(
                f"📍 自动调整目标点到最近的道路中心点: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f})"
            )
            
            adjusted_goal_pose = PoseStamped()
            adjusted_goal_pose.header = msg.header
            adjusted_goal_pose.pose.position.x = nearest_point[0]
            adjusted_goal_pose.pose.position.y = nearest_point[1]
            adjusted_goal_pose.pose.position.z = 0.0
            adjusted_goal_pose.pose.orientation = msg.pose.orientation
        else:
            self.get_logger().info(
                f"✓ 目标点在道路中心线上（偏离 {distance*100:.1f}cm）"
            )
            adjusted_goal_pose = msg
        
        self.goal_pose = adjusted_goal_pose
        
        if self.vehicle_spawned:
            self.plan_path()
            self.publish_status("goal_pose_received")
        else:
            self.get_logger().warn("请先设置初始位置生成自车")
    
    def spawn_vehicle_at_current_pose(self, pose):
        """在指定位置生成自车"""
        self.current_pose = pose
        self.vehicle_spawned = True
        
        self.current_velocity = Twist()
        self.waypoints = []
        self.current_waypoint_index = 0
        
        self.get_logger().info("🚗 自车（Tesla Model3）已在当前位置生成")
        self.get_logger().info(f"自车位置: ({pose.position.x:.2f}, {pose.position.y:.2f}), z={pose.position.z:.2f}")
        self.get_logger().info("等待目标点设置...")
    
    def _find_nearest_point_index_in_centerline(self, x, y, centerline):
        """找到centerline中最近点的索引"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for idx, point in enumerate(centerline):
            dist = math.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        return nearest_idx
    
    def plan_path(self):
        """基于道路中心线拓扑规划路径 - 使用A*算法"""
        if self.current_pose is None or self.goal_pose is None:
            return
            
        start_pose = self.current_pose.position
        goal_pose = self.goal_pose.pose.position
        
        self.waypoints = []
        self.current_waypoint_index = 0
        self.path_published = False
        
        # 使用全局变量中的lanelet_id
        start_lanelet_id = self.start_lanelet_id
        goal_lanelet_id = self.goal_lanelet_id
        
        self.get_logger().info(f"🔍 起点lanelet: {start_lanelet_id}, 终点lanelet: {goal_lanelet_id}")
        
        if start_lanelet_id is None or goal_lanelet_id is None:
            self.get_logger().error("❌ 起点或终点不在有效道路上")
            self.plan_straight_line_path(start_pose, goal_pose)
            return
        
        if start_lanelet_id == goal_lanelet_id:
            self.get_logger().info("✓ 起点和终点在同一lanelet，沿中心线采样")
            self.plan_path_in_lanelet(start_lanelet_id, start_pose, goal_pose)
            return
        
        # 检查是否在拓扑图中
        if start_lanelet_id not in self.graph:
            self.get_logger().error(f"❌ 起点lanelet {start_lanelet_id} 不在拓扑图中！")
            self.plan_straight_line_path(start_pose, goal_pose)
            return
            
        if goal_lanelet_id not in self.graph:
            self.get_logger().error(f"❌ 终点lanelet {goal_lanelet_id} 不在拓扑图中！")
            self.plan_straight_line_path(start_pose, goal_pose)
            return
        
        self.get_logger().info(f"🔍 开始A*搜索：Lanelet {start_lanelet_id} → {goal_lanelet_id}")
        lanelet_path, directions = self.astar_path_search(start_lanelet_id, goal_lanelet_id)
        
        if lanelet_path is None or directions is None:
            self.get_logger().error("❌ 无法找到从起点到终点的道路路径")
            # 输出详细的诊断信息
            if start_lanelet_id in self.graph:
                start_neighbors = self.graph[start_lanelet_id]
                self.get_logger().error(f"   起点lanelet {start_lanelet_id} 的邻居: {start_neighbors}")
            if goal_lanelet_id in self.graph:
                goal_neighbors = self.graph[goal_lanelet_id]
                self.get_logger().error(f"   终点lanelet {goal_lanelet_id} 的邻居: {goal_neighbors}")
            self.get_logger().warn("⚠️ 使用直线连接作为备用方案")
            self.plan_straight_line_path(start_pose, goal_pose)
            return
        
        self.get_logger().info(f"找到Lanelet路径: {lanelet_path}")
        
        self.waypoints = []
        self.waypoints.append((start_pose.x, start_pose.y))
        
        # 按需从lanelet的leftBound/rightBound重新计算中心线
        for i, lanelet_id in enumerate(lanelet_path):
            # 重新计算完整中心线
            centerline = self._compute_centerline_from_lanelet(lanelet_id)
            if len(centerline) == 0:
                continue
                
            direction = directions[i] if i < len(directions) else 'forward'
            
            if direction == 'reverse':
                centerline = list(reversed(centerline))
            
            if i == 0:
                # 第一个lanelet：从起点位置开始
                start_idx = self._find_nearest_point_index_in_centerline(
                    start_pose.x, start_pose.y, centerline
                )
                for point in centerline[start_idx:]:
                    self.waypoints.append(point)
            elif i == len(lanelet_path) - 1:
                # 最后一个lanelet：到终点位置结束
                goal_idx = self._find_nearest_point_index_in_centerline(
                    goal_pose.x, goal_pose.y, centerline
                )
                for point in centerline[:goal_idx + 1]:
                    self.waypoints.append(point)
            else:
                # 中间lanelet：添加全部点
                self.waypoints.extend(centerline)
        
        self.waypoints.append((goal_pose.x, goal_pose.y))
        
        self.waypoints = self._uniform_sample_path(self.waypoints)
        
        self.get_logger().info(f"路径规划完成，共生成 {len(self.waypoints)} 个路径点")
        
        self.publish_path()
    
    def plan_path_in_lanelet(self, lanelet_id, start_pose, goal_pose):
        """在同一lanelet内沿中心线规划路径"""
        self.waypoints = []
        self.waypoints.append((start_pose.x, start_pose.y))
        
        # 按需从lanelet的leftBound/rightBound重新计算中心线
        centerline = self._compute_centerline_from_lanelet(lanelet_id)
        
        if len(centerline) > 0:
            start_idx = self._find_nearest_point_index_in_centerline(
                start_pose.x, start_pose.y, centerline
            )
            goal_idx = self._find_nearest_point_index_in_centerline(
                goal_pose.x, goal_pose.y, centerline
            )
            
            # 确保索引顺序正确
            if start_idx <= goal_idx:
                for i in range(start_idx, goal_idx + 1):
                    self.waypoints.append(centerline[i])
            else:
                # 如果start_idx > goal_idx，说明需要反向
                for i in range(start_idx, -1, -1):
                    self.waypoints.append(centerline[i])
                for i in range(len(centerline) - 1, goal_idx - 1, -1):
                    if i != start_idx:
                        self.waypoints.append(centerline[i])
        
        self.waypoints.append((goal_pose.x, goal_pose.y))
        
        # 所有waypoints处理完成后，调用_uniform_sample_path进行均匀采样
        self.waypoints = self._uniform_sample_path(self.waypoints)
        
        self.publish_path()
    
    def plan_straight_line_path(self, start_pose, goal_pose):
        """使用直线路径作为备用方案"""
        dx = goal_pose.x - start_pose.x
        dy = goal_pose.y - start_pose.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:
            self.get_logger().warn("距离太近，无需规划路径")
            return
        
        num_waypoints = max(int(distance / self.path_resolution), 1)
        
        for i in range(num_waypoints + 1):
            ratio = i / num_waypoints
            waypoint_x = start_pose.x + dx * ratio
            waypoint_y = start_pose.y + dy * ratio
            self.waypoints.append((waypoint_x, waypoint_y))
        
        self.waypoints.append((goal_pose.x, goal_pose.y))
        self.publish_path()
    
    def publish_path(self):
        """发布规划路径，包含正确的方向信息（只发布一次）"""
        if self.path_published or not self.waypoints:
            return
        
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        
        for i, waypoint in enumerate(self.waypoints):
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = waypoint[0]
            pose_stamped.pose.position.y = waypoint[1]
            pose_stamped.pose.position.z = 0.0
            
            if i < len(self.waypoints) - 1:
                next_x, next_y = self.waypoints[i + 1]
                yaw = math.atan2(next_y - waypoint[1], next_x - waypoint[0])
            elif i > 0:
                prev_x, prev_y = self.waypoints[i - 1]
                yaw = math.atan2(waypoint[1] - prev_y, waypoint[0] - prev_x)
            else:
                yaw = 0.0
            
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
            
            path.poses.append(pose_stamped)
        
        self.path_pub.publish(path)
        self.path_published = True
        self.get_logger().info(f'✓ 发布路径到 /carla/ego_vehicle/waypoints，包含 {len(self.waypoints)} 个点')
    
    def quaternion_to_yaw(self, quat):
        """将四元数转换为偏航角"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def publish_status(self, status):
        """发布状态信息"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n地图显示与路径规划系统已关闭")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

