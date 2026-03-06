#!/usr/bin/env python3
"""CARLA车辆感知模块 - 处理障碍物检测和坐标转换"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Int32
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, Int32 as StdInt32
from derived_object_msgs.msg import ObjectArray
try:
    from map_load.msg import FrenetPath, SLBoundary, SLBoundaryArray
except ImportError:
    FrenetPath = None
    SLBoundary = None
    SLBoundaryArray = None
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import time
import threading

# 颜色常量（与 vehicle_ctrl.constants 保持一致）
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

try:
    from map_load import math_utils
except ImportError:
    math_utils = None


class VehiclePerception(Node):
    """车辆感知节点"""

    def __init__(self):
        super().__init__('vehicle_perception')

        # 控制参数
        self.control_dt = 0.05  # 20Hz
        self.point_spacing = 1.0  # 路径点间距 (m)

        # 路径与最近点状态（感知模块内部自行维护最近索引）
        self.current_waypoints = []
        self.waypoints_received = False
        self.last_nearest_idx = 0

        # 车辆状态（用于Frenet坐标转换）
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0

        # 订阅器
        # 订阅路径更新（与控制模块保持同步）
        self.waypoints_sub = self.create_subscription(
            Path,
            '/carla/ego_vehicle/waypoints',
            self.waypoints_callback,
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
        # 订阅里程计（用于自行计算最近路径段）
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )

        # 订阅车辆对象列表（CARLA ROS Bridge）
        self.objects_sub = self.create_subscription(
            ObjectArray,
            '/carla/ego_vehicle/objects',
            self.objects_callback,
            10
        )

        # 发布器
        # Frenet坐标发布器
        if FrenetPath is not None:
            self.frenet_path_pub = self.create_publisher(
                FrenetPath,
                '/path_smoothing/frenet_path',
                10
            )
        else:
            self.frenet_path_pub = None
            self.get_logger().warn("FrenetPath 消息类型未找到，Frenet 坐标发布功能将不可用")

        # SL边界数组发布器（批量发布所有障碍物）
        if SLBoundaryArray is not None:
            self.sl_boundary_array_pub = self.create_publisher(
                SLBoundaryArray,
                '/path_smoothing/sl_boundary_array',
                10
            )
        else:
            self.sl_boundary_array_pub = None
            self.get_logger().warn("SLBoundaryArray 消息类型未找到，SL边界数组发布功能将不可用")

        # 状态变量：用于同步索引和点列表
        self.pending_update_index = None
        self.pending_update_points = None

        # 最新障碍物列表（原始ObjectArray）
        self.latest_objects: ObjectArray | None = None

        # 定时器：以10Hz频率更新并发布 Frenet 坐标和 SL 边界
        self.perception_timer = self.create_timer(
            0.1,  # 10Hz
            self.update_frenet_coordinates
        )

        self.get_logger().info(f"{GREEN}✓ 车辆感知节点已启动{RESET}")

    def waypoints_callback(self, msg):
        """路径点回调"""
        waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            waypoints.append((x, y))

        with threading.Lock():
            self.current_waypoints = waypoints
            self.waypoints_received = True

        self.get_logger().info(f"{CYAN}✓ 感知模块收到路径规划，包含 {len(waypoints)} 个路径点{RESET}")

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

    def _try_apply_path_update(self):
        """尝试应用路径增量更新"""
        if self.pending_update_index is None or self.pending_update_points is None:
            return

        start_idx = self.pending_update_index
        update_points = self.pending_update_points

        if len(update_points) == 0:
            self.pending_update_index = None
            self.pending_update_points = None
            return

        with threading.Lock():
            if start_idx >= 0 and start_idx < len(self.current_waypoints):
                for i, point in enumerate(update_points):
                    global_idx = start_idx + i
                    if 0 <= global_idx < len(self.current_waypoints):
                        self.current_waypoints[global_idx] = point

        # 清除待处理状态
        self.pending_update_index = None
        self.pending_update_points = None

    def objects_callback(self, msg: ObjectArray):
        """车辆对象列表回调：缓存最新的障碍物信息"""
        # FIXME：检查object返回的数据是否正常，为什么boundingbox异常，是否是中点的问题？
        self.latest_objects = msg

    def odom_callback(self, msg):
        """里程计回调 - 获取车辆位置和姿态"""
        pose = msg.pose.pose
        self.current_x = pose.position.x
        self.current_y = pose.position.y

        orientation = pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

        linear_velocity = msg.twist.twist.linear.x
        self.current_speed = abs(linear_velocity)

    def update_frenet_coordinates(self):
        """更新Frenet坐标转换（感知模块内部自行搜索最近路径段）"""
        if not self.waypoints_received or len(self.current_waypoints) == 0:
            return

        # 1. 基于当前位姿和全局路径，搜索最近路径段索引（Stanley 风格）
        nearest_idx = self._find_nearest_index_on_path()

        # 2. 基于该最近索引截取局部参考线
        reference_segment = self.get_reference_path_segment(nearest_idx)
        if len(reference_segment) == 0:
            return

        # 3. 发布Frenet坐标（自车）
        self.publish_frenet_coordinates(reference_segment, nearest_idx)

        # 4. 发布SL边界（障碍物）
        self.publish_sl_boundaries(reference_segment)

    def get_reference_path_segment(self, center_idx: int):
        """
        获取参考线路径段：从 center_idx 前30m到后180m的范围

        返回：
        reference_segment: 参考线路径段（numpy数组）
        """
        if len(self.current_waypoints) == 0:
            return np.array([])

        # 计算索引范围
        backward_distance = 0.0  # 向前30m
        forward_distance = 180.0  # 向后180m

        backward_idx = int(backward_distance / self.point_spacing)
        forward_idx = int(forward_distance / self.point_spacing)

        n = len(self.current_waypoints)
        center_idx = max(0, min(center_idx, n - 1))

        # 计算起始和结束索引（end_idx 为包含的最后一个索引）
        start_idx = max(0, center_idx - backward_idx)
        end_idx = min(n - 1, center_idx + forward_idx)

        # 提取参考线段（包含 end_idx）
        reference_segment = np.array(self.current_waypoints[start_idx:end_idx + 1])

        return reference_segment

    def publish_frenet_coordinates(self, reference_path_segment, nearest_idx: int):
        """
        计算并发布Frenet坐标

        参数：
        reference_path_segment: 参考线路径段（numpy数组，shape为(n, 2)）
        """
        if self.frenet_path_pub is None:
            return

        if len(reference_path_segment) < 2:
            return

        try:
            # 对于自车，只需要简单的 (s, d)：
            # s 取固定 30m（相对于用于投影的参考线起点），
            # d 使用全局路径上最近段的几何横向误差
            backward_distance = 0.0
            s_value = backward_distance
            # 计算d：基于全局路径 self.current_waypoints 计算有符号横向误差，
            # 起始搜索索引使用本节点内部搜索得到的最近索引
            global_path = np.array(self.current_waypoints, dtype=float)
            d_value = self._compute_lateral_distance_to_path(global_path, nearest_idx)
            
            #FIXME: 查一下这里数据是否正常，查一下为什么没有红点显示

            # 创建FrenetPath消息
            frenet_msg = FrenetPath()
            frenet_msg.header.frame_id = "map"
            frenet_msg.header.stamp = self.get_clock().now().to_msg()

            # 填充数据：只发布单个车辆位置的 (s, d)
            frenet_msg.s = [float(s_value)]  # 车辆位置的 s（相对于参考线段起点）
            frenet_msg.d = [float(d_value)]  # 车辆位置的 d（横向偏差）

            # 导数信息暂不发布，但保留字段
            frenet_msg.s_dot = []
            frenet_msg.d_dot = []
            frenet_msg.s_ddot = []
            frenet_msg.d_ddot = []

            # 发布消息
            self.frenet_path_pub.publish(frenet_msg)

        except Exception as e:
            self.get_logger().warn(f"发布Frenet坐标失败: {e}")

    def _compute_lateral_distance_to_path(self, path_points: np.ndarray, start_idx: int | None = None) -> float:
        """
        计算当前车辆位置到参考线段的有符号横向距离
        """
        if len(path_points) < 2:
            return 0.0

        n = len(path_points)

        # 使用传入的起始索引或上一次的最近索引作为搜索起点
        if start_idx is None:
            nearest_idx = max(0, min(self.last_nearest_idx, n - 2))
        else:
            nearest_idx = max(0, min(start_idx, n - 2))

        vehicle_point = np.array([self.current_x, self.current_y], dtype=float)

        best_dist = float('inf')
        best_signed_dist = 0.0
        best_idx = nearest_idx

        # 只在附近若干段内搜索，避免全局扫描
        search_radius = 50
        start_seg = max(0, nearest_idx - search_radius)
        end_seg = min(n - 2, nearest_idx + search_radius)

        for i in range(start_seg, end_seg + 1):
            p0 = path_points[i]
            p1 = path_points[i + 1]
            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            seg_unit = seg_vec / seg_len

            # 投影到线段
            v = vehicle_point - p0
            proj_len = np.clip(np.dot(v, seg_unit), 0.0, seg_len)
            proj_point = p0 + proj_len * seg_unit

            # 有符号横向距离：使用2D叉乘确定左右
            rel = vehicle_point - proj_point
            # seg_vec = (dx, dy), rel = (rx, ry)
            cross = seg_vec[0] * rel[1] - seg_vec[1] * rel[0]
            signed_dist = np.sign(cross) * np.linalg.norm(rel)

            if abs(signed_dist) < best_dist:
                best_dist = abs(signed_dist)
                best_signed_dist = signed_dist
                best_idx = i

        # 更新内部最近索引，用于下一次 warm-start
        self.last_nearest_idx = best_idx

        return float(best_signed_dist)

    def _find_nearest_index_on_path(self) -> int:
        """
        在 self.current_waypoints 上搜索当前车辆最近的路径段索引，
        逻辑参考 StanleyController.compute_steering 中最近段搜索。
        """
        if len(self.current_waypoints) < 2:
            self.last_nearest_idx = 0
            return 0

        current_pos = np.array([self.current_x, self.current_y], dtype=float)
        n = len(self.current_waypoints)

        min_dist = float('inf')
        nearest_idx = self.last_nearest_idx

        # 仅在上一次附近窗口内搜索，加速计算
        min_search_idx = max(0, self.last_nearest_idx - 50)
        max_search_idx = min(n - 2, self.last_nearest_idx + 50)

        for i in range(min_search_idx, max_search_idx + 1):
            start_point = np.array(self.current_waypoints[i], dtype=float)
            end_point = np.array(self.current_waypoints[i + 1], dtype=float)

            line_vec = end_point - start_point
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                continue

            line_unitvec = line_vec / line_len
            vehicle_to_start = current_pos - start_point

            projection = np.dot(vehicle_to_start, line_unitvec)
            projection = np.clip(projection, 0.0, line_len)

            nearest_point = start_point + projection * line_unitvec
            dist = np.linalg.norm(current_pos - nearest_point)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        self.last_nearest_idx = nearest_idx
        return nearest_idx

    def publish_sl_boundaries(self, reference_path_segment):
        """
        计算并发布障碍物的SL边界

        参数：
        reference_path_segment: 参考线路径段
        """
        if self.sl_boundary_array_pub is None or math_utils is None:
            return

        if self.latest_objects is None or len(self.latest_objects.objects) == 0:
            return

        if len(reference_path_segment) < 2:
            return

        # 计算参考线的几何信息（headings, accumulated_s）
        headings, accumulated_s, kappas, dkappas = math_utils.compute_path_profile(
            reference_path_segment, self.point_spacing
        )

        ego_xy = np.array([self.current_x, self.current_y], dtype=float)

        # 收集所有障碍物的SLBoundary
        sl_boundaries_list = []
        
        for obj in self.latest_objects.objects:
            # 障碍物中心
            obj_xy = np.array([obj.pose.position.x, obj.pose.position.y], dtype=float)
            dist_to_ego = np.linalg.norm(obj_xy - ego_xy)
            if dist_to_ego > 100.0:
                continue

            # 只处理具有有效尺寸的box
            dims = obj.shape.dimensions
            if len(dims) < 2:
                continue

            length = float(dims[0])
            width = float(dims[1])

            # 计算障碍物yaw
            q = obj.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # 计算4个二维角点（世界坐标）
            corners_xy = self._compute_box_corners_2d(obj_xy, yaw, length, width)

            # 计算SLBoundary
            sl_boundary = self._compute_sl_boundary_for_box(
                corners_xy,
                reference_path_segment,
                headings,
                accumulated_s
            )

            if sl_boundary is None:
                continue

            # 创建SLBoundary消息并添加到列表
            boundary_msg = SLBoundary()
            boundary_msg.header.frame_id = "map"
            boundary_msg.header.stamp = self.get_clock().now().to_msg()

            boundary_msg.start_s = sl_boundary["start_s"]
            boundary_msg.end_s = sl_boundary["end_s"]
            boundary_msg.start_l = sl_boundary["start_l"]
            boundary_msg.end_l = sl_boundary["end_l"]
            boundary_msg.boundary_s = sl_boundary["boundary_s"]
            boundary_msg.boundary_l = sl_boundary["boundary_l"]
            boundary_msg.vehicle_id = str(obj.id)

            sl_boundaries_list.append(boundary_msg)

        # 一次性发布所有障碍物的SLBoundary数组
        if len(sl_boundaries_list) > 0 and self.sl_boundary_array_pub is not None:
            array_msg = SLBoundaryArray()
            array_msg.header.frame_id = "map"
            array_msg.header.stamp = self.get_clock().now().to_msg()
            array_msg.boundaries = sl_boundaries_list
            
            self.sl_boundary_array_pub.publish(array_msg)

    def _compute_box_corners_2d(self, center_xy: np.ndarray, yaw: float,
                                length: float, width: float) -> np.ndarray:
        """
        计算二维bounding box的4个角点（世界坐标系）
        长度沿x方向，宽度沿y方向
        """
        # 局部坐标系下的4个角点（逆时针）
        half_l = length / 2.0
        half_w = width / 2.0
        local_corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rot = np.array([[cos_yaw, -sin_yaw],
                        [sin_yaw, cos_yaw]])

        world_corners = (rot @ local_corners.T).T + center_xy
        return world_corners

    def _xy_to_sl_with_s_window(self,
                                point_xy: np.ndarray,
                                reference_path_segment: np.ndarray,
                                accumulated_s: np.ndarray,
                                s_start: float,
                                s_end: float) -> tuple[float, float] | None:
        """
        在 [s_start, s_end] 区间内搜索参考线的最近投影点，返回 (s, d)。
        """
        n = len(reference_path_segment)
        if n < 2:
            return None

        # 规范化s窗口
        if s_start > s_end:
            s_start, s_end = s_end, s_start

        path_length = accumulated_s[-1]
        # 与路径区间求交集
        s_start = max(0.0, s_start)
        s_end = min(path_length, s_end)
        if s_end - s_start < 1e-6:
            return None

        # 在 accumulated_s 上找到覆盖 [s_start, s_end] 的段索引范围
        # 先用 s/point_spacing 估算索引，再做小范围修正，避免全局线性扫描
        spacing = self.point_spacing if self.point_spacing > 1e-6 else 1.0
        # 估算起止索引（四舍五入）
        i_start_est = int(round(s_start / spacing))
        i_end_est = int(round(s_end / spacing))

        # 初步裁剪到合法范围 [0, n-2]
        i_start = max(0, min(i_start_est, n - 2))
        i_end = max(0, min(i_end_est, n - 2))

        # # 向左微调 i_start，直到 accumulated_s[i_start] <= s_start
        # while i_start > 0 and accumulated_s[i_start] > s_start:
        #     i_start -= 1

        # # 向右微调 i_end，直到 accumulated_s[i_end+1] >= s_end 或到达末尾
        # while i_end + 1 < n and accumulated_s[i_end + 1] < s_end:
        #     i_end += 1

        if i_start > i_end or i_end < 0:
            return None

        best_dist = float('inf')
        best_s = 0.0
        best_d = 0.0

        for i in range(i_start, i_end + 1):
            p0 = reference_path_segment[i]
            p1 = reference_path_segment[i + 1]
            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            seg_unit = seg_vec / seg_len

            v = point_xy - p0
            proj_len = np.clip(np.dot(v, seg_unit), 0.0, seg_len)
            proj_point = p0 + proj_len * seg_unit

            # 只接受s落在[s_start,s_end]窗口内的候选
            s_val = float(accumulated_s[i] + proj_len)
            if s_val < s_start - 1e-6 or s_val > s_end + 1e-6:
                continue

            dist = float(np.linalg.norm(point_xy - proj_point))
            if dist < best_dist:
                best_dist = dist

                rel = point_xy - proj_point
                cross = seg_vec[0] * rel[1] - seg_vec[1] * rel[0]
                d_val = np.sign(cross) * np.linalg.norm(rel)

                best_s = s_val
                best_d = float(d_val)

        if best_dist is float('inf'):
            return None

        return best_s, best_d

    def _xy_to_sl(self,
                  point_xy: np.ndarray,
                  reference_path_segment: np.ndarray,
                  accumulated_s: np.ndarray,
                  mode: str = "global",
                  warm_start_s: float | None = None,
                  s_window: tuple[float, float] | None = None) -> tuple[float, float] | None:
        """
        统一的 XYToSL 接口
          - mode='global' : 在整条参考线上搜索最近点
          - mode='warm'   : 使用 warm_start_s 附近的局部窗口搜索（当前未大规模使用）
          - mode='window' : 只在给定 [s_start, s_end] 窗口内搜索
        """
        n = len(reference_path_segment)
        if n < 2:
            return None

        path_length = accumulated_s[-1]

        if mode == "global":
            return self._xy_to_sl_with_s_window(
                point_xy,
                reference_path_segment,
                accumulated_s,
                0.0,
                float(path_length),
            )

        if mode == "window" and s_window is not None:
            s_start, s_end = s_window
            return self._xy_to_sl_with_s_window(
                point_xy,
                reference_path_segment,
                accumulated_s,
                float(s_start),
                float(s_end),
            )

        if mode == "warm" and warm_start_s is not None:
            # 简化版warm: 以warm_start_s为中心，给一个固定宽度窗口
            window_half = 20.0  # 可调
            s_start = float(warm_start_s) - window_half
            s_end = float(warm_start_s) + window_half
            return self._xy_to_sl_with_s_window(
                point_xy,
                reference_path_segment,
                accumulated_s,
                s_start,
                s_end,
            )

        # 默认回退到全局搜索
        return self._xy_to_sl_with_s_window(
            point_xy,
            reference_path_segment,
            accumulated_s,
            0.0,
            float(path_length),
        )

    def _compute_sl_boundary_for_box(self,
                                     corners_xy: np.ndarray,
                                     reference_path_segment: np.ndarray,
                                     headings: np.ndarray,
                                     accumulated_s: np.ndarray) -> dict | None:
        """
        对box的四个角点做XY→SL投影，构建SLBoundary（暂时不使用边中点投影）
        """
        n = len(reference_path_segment)
        if n < 2:
            return None

        num_corners = len(corners_xy)
        
        # 检查角点数量：必须是4个角点
        if num_corners != 4:
            print(
                f"障碍物bounding box角点数量异常：期望4个角点，实际为{num_corners}个。"
                f"当前仅支持矩形box（4个角点），跳过该障碍物。"
            )
            return None

        # 1. 根据自车位置选择离ego最近的角点作为起点，并rotate角点顺序
        ego_xy = np.array([self.current_x, self.current_y], dtype=float)
        obs_corners = corners_xy.copy()
        first_index = 0
        min_dist = float('inf')
        for i in range(num_corners):
            d = float(np.linalg.norm(obs_corners[i] - ego_xy))
            if d < min_dist:
                min_dist = d
                first_index = i
        if first_index != 0:
            obs_corners = np.roll(obs_corners, -first_index, axis=0)

        sl_points_s: list[float] = []
        sl_points_l: list[float] = []

        # 2. 第一个角点：使用全局搜索
        first_corner = obs_corners[0]
        first_res = self._xy_to_sl(
            first_corner,
            reference_path_segment,
            accumulated_s,
            mode="global",
        )
        if first_res is None:
            return None
        s0, l0 = first_res
        sl_points_s.append(s0)
        sl_points_l.append(l0)

        # 3. 后续角点：使用 hueristic s 窗口 [s_prev - 2*dist, s_prev + 2*dist]
        prev_s = s0
        for i in range(1, num_corners):
            p = obs_corners[i]
            prev_p = obs_corners[i - 1]
            distance = float(np.linalg.norm(p - prev_p))
            s_start = prev_s - 2.0 * distance
            s_end = prev_s + 2.0 * distance

            res = self._xy_to_sl(
                p,
                reference_path_segment,
                accumulated_s,
                mode="window",
                s_window=(s_start, s_end),
            )
            if res is None:
                return None
            s_i, l_i = res
            sl_points_s.append(s_i)
            sl_points_l.append(l_i)
            prev_s = s_i

        # 注意：暂时不使用边中点投影，仅使用原始四个角点

        if not sl_points_s:
            return None

        start_s = float(min(sl_points_s))
        end_s = float(max(sl_points_s))
        start_l = float(min(sl_points_l))
        end_l = float(max(sl_points_l))

        return {
            "start_s": start_s,
            "end_s": end_s,
            "start_l": start_l,
            "end_l": end_l,
            "boundary_s": [float(s) for s in sl_points_s],
            "boundary_l": [float(l) for l in sl_points_l],
        }


def main(args=None):
    rclpy.init(args=args)
    node = VehiclePerception()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n车辆感知节点已关闭")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

