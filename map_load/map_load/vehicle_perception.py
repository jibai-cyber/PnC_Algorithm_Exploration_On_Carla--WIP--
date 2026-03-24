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
from carla_msgs.msg import CarlaEgoVehicleInfo
try:
    from map_load.msg import FrenetPath, SLBoundary, SLBoundaryArray, PathBoundary
except ImportError:
    FrenetPath = None
    SLBoundary = None
    SLBoundaryArray = None
    PathBoundary = None
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import time
import threading
from .frenet_qp_planner import FrenetQPPlanner

GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"

# ---------- PathBoundary 与道路/坐标系参数 ----------
REFERENCE_LINE_HORIZON = 50.0  # 自车前方参考线/采样 s 范围 (m)
BACKWARD_DISTANCE = 30.0       # 自车后方参考线/采样 s 范围 (m)
ROAD_WIDTH_TOTAL = 8.0         # 双车道总宽 (m)
ROAD_LEFT_BOUNDARY_L = 6.0     # 道路左边界 l (右车道中心为 0)
ROAD_RIGHT_BOUNDARY_L = -2.0   # 道路右边界 l
PATH_BOUNDARY_S_RESOLUTION = 0.5  # s 方向采样间隔 (m)
K_ZERO_VAL = 1e-6              # 浮点比较/重叠判断容差

# ADC 边界与横向 buffer（用于在 PathBoundary 上收缩 ego 车辆中心可行域）
EGO_LAT_BUFFER = 0.3  # m
EGO_WIDTH_DEFAULT = 1.8  # m，若未能从车辆状态话题获取则使用该默认值
# 当在 s 方向上距离自车小于该值且无可行走廊时，才判定为 blocked 并触发停车
BLOCK_STOP_DISTANCE = 10.0  # m

# 可自定义的静态障碍物 ID 名称（按出现顺序依次分配）
# TODO: remove this after testing
CUSTOM_OBSTACLE_IDS = [
    "static_vehicle_1",
    "static_vehicle_2",
    "static_vehicle_3",
]


# 决策类型
class NudgeDecision:
    LEFT_NUDGE = "left_nudge"
    RIGHT_NUDGE = "right_nudge"
    BLOCKED = "blocked"

try:
    from map_load import math_utils
except ImportError:
    math_utils = None


# road_left_l, road_right_l 要大到保证可以包含obs的四个角点，否则该函数会出错
def _get_obs_ref_points_(
    corner_s: list, corner_l: list, decision: str,
    road_left_l: float, road_right_l: float,
):
    """
    按决策选取障碍物左右边界的 3 点参考（用于后续按 s 插值）。
    右 nudge: 锚点=离道路左边界最近；逆时针取剩余 3 点作为右边界。
    左 nudge: 锚点=离道路右边界最近；顺时针取剩余 3 点作为左边界。
    返回 (ref_left_s, ref_left_l, ref_right_s, ref_right_l)，各为长度 3 的列表。
    """
    n = 4
    if len(corner_s) != n or len(corner_l) != n:
        return ([], [], [], [])
    idx_anchor_left = min(range(n), key=lambda i: abs(corner_l[i] - road_right_l))
    idx_anchor_right = min(range(n), key=lambda i: abs(corner_l[i] - road_left_l))
    # 逆时针（与存储顺序一致）：(anchor+1)%4, (anchor+2)%4, (anchor+3)%4
    # 顺时针：(anchor+3)%4, (anchor+2)%4, (anchor+1)%4
    def cw_three(anchor):
        return [(anchor + 3) % n, (anchor + 2) % n, (anchor + 1) % n]
    def ccw_three(anchor):
        return [(anchor + 1) % n, (anchor + 2) % n, (anchor + 3) % n]
    
    ref_left_s = [corner_s[i] for i in cw_three(idx_anchor_left)]
    ref_left_l = [corner_l[i] for i in cw_three(idx_anchor_left)]
    ref_right_s = [corner_s[i] for i in ccw_three(idx_anchor_right)]
    ref_right_l = [corner_l[i] for i in ccw_three(idx_anchor_right)]
    return (ref_left_s, ref_left_l, ref_right_s, ref_right_l)


def _segment_l_at_s(sa: float, la: float, sb: float, lb: float, s: float):
    """
    线段 (sa,la)-(sb,lb) 在 s 处的 l。
    竖边 (sa==sb)：返回 (l_min, l_max) 两个值。
    斜边且 s 在段内：返回 (l_val, None) 表示单点。
    s 不在段内：返回 (None, None)。
    """
    if abs(sb - sa) <= K_ZERO_VAL:
        return (min(la, lb), max(la, lb))
    t = (s - sa) / (sb - sa)
    if t < 0 or t > 1:
        return (None, None)
    l_val = la + t * (lb - la)
    return (l_val, None)


def _get_obs_edge_l_at_s(ref_s: list, ref_l: list, s: float):
    """
    从 3 点参考 (两段) 在 s 处插值得到 l。
    竖边返回 (l_lo, l_hi)；斜边单点返回 (l_val, l_val) 供上层统一用 (lo, hi) 处理；否则 (None, None)。
    """
    if len(ref_s) != 3 or len(ref_l) != 3:
        return (None, None)
    for i in range(2):
        sa, sb = ref_s[i], ref_s[i + 1]
        la, lb = ref_l[i], ref_l[i + 1]
        lo, hi = _segment_l_at_s(sa, la, sb, lb, s)
        if lo is not None:
            return (lo, hi if hi is not None else lo)
    return (None, None)


def _get_obs_l_extent_at_s(obs: dict, s: float):
    """
    障碍物 4 角点在垂线 s 上的 l 范围 [l_min, l_max]。
    若当前 s 不在障碍物任一棱边的 [sa,sb] 内（即无 l 方向投影），返回 None。
    """
    cs = obs["corner_s"]
    cl = obs["corner_l"]
    n = 4
    l_vals = []
    for i in range(n):
        sa, la = cs[i], cl[i]
        sb, lb = cs[(i + 1) % n], cl[(i + 1) % n]
        lo, hi = _segment_l_at_s(sa, la, sb, lb, s)
        if lo is not None:
            l_vals.append(lo)
        if hi is not None:
            l_vals.append(hi)
    if not l_vals:
        return None
    return (min(l_vals), max(l_vals))


def _intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return max(a_lo, b_lo) <= min(a_hi, b_hi) + K_ZERO_VAL


def _path_bounds_from_two_elements(elements, a_key, b_key, ego_s, obs_info_list):
    """
    由当前 s 下选定的两个元素 a_key（提供上界）, b_key（提供下界）得到 path_lower, path_upper。
    返回值：(path_lower, path_upper)；若无法解析则返回 (None, None)。
    """
    obs_by_id = {o["id"]: o for o in obs_info_list}
    def get_upper(key):
        if key == "road_left":
            return ROAD_LEFT_BOUNDARY_L
        o = obs_by_id.get(key)
        if o is None:
            return None
        lo, hi = _get_obs_edge_l_at_s(o["ref_right_s"], o["ref_right_l"], ego_s)
        if lo is None:
            return None
        return max(lo, hi)
    def get_lower(key):
        if key == "road_right":
            return ROAD_RIGHT_BOUNDARY_L
        o = obs_by_id.get(key)
        if o is None:
            return None
        lo, hi = _get_obs_edge_l_at_s(o["ref_left_s"], o["ref_left_l"], ego_s)
        if lo is None:
            return None
        return min(lo, hi)
    u = get_upper(a_key)
    l = get_lower(b_key)
    if u is None or l is None:
        return (None, None)
    return (l, u)


class VehiclePerception(Node):
    """车辆感知节点"""

    def __init__(self):
        super().__init__('vehicle_perception')

        # 控制参数
        self.control_dt = 0.05  # 20Hz
        self.point_spacing = 1.0  # 路径点间距 (m)

        # 车辆参数（用于在 PathBoundary 上施加 ADC bound）
        self.ego_half_width = EGO_WIDTH_DEFAULT / 2.0
        self.vehicle_length = 4.5
        self.vehicle_wheelbase = 2.8
        self.vehicle_max_steering_angle = 0.6
        self.vehicle_info_received = False

        # 路径与最近点状态（感知模块内部自行维护最近索引）
        self.current_waypoints = []
        self.waypoints_received = False
        self.last_nearest_idx = 0

        # 车辆状态（用于Frenet坐标转换）
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0

        # 缓存未施加 ADC bound / lat_buffer 前的 PathBoundary，便于后续 ADCVertexConstraints 调用
        self.raw_path_s = []
        self.raw_path_l_upper = []
        self.raw_path_l_lower = []

        # 上周期 QP 解的缓存，用于判断复用条件和提供平滑初始状态
        self.prev_qp_solution = {
            "s": None,           # np.ndarray, shape (N,)
            "l": None,           # np.ndarray, shape (N,)
            "dl": None,          # np.ndarray, shape (N,)
            "ddl": None,         # np.ndarray, shape (N,)
            "obs_signature": None,  # str, 障碍物特征签名（id + decision）
            "obs_xy_map": None,  # dict[str, tuple[x, y]], 障碍物世界坐标，用于位置变化判断
            "path_boundary_valid": False,  # bool, 上周期 PathBoundary 是否有效
            "timestamp": None,   # float, ROS2 时间戳（秒）
            "ego_s_at_solve": 0.0,  # float, 求解时的 ego_s（局部坐标，通常为0）
            "reference_segment": None,  # np.ndarray, shape (M, 2)，上周期的参考线路径段
            "accumulated_s": None,      # np.ndarray, shape (M,)，上周期的累积弧长（局部坐标系）
        }
        # 复用判断阈值
        self.REUSE_MAX_TIME = 2.0          # 最大复用时长（秒）
        self.REUSE_MAX_L_DEV = 0.5         # 横向偏差阈值（米）
        self.REUSE_OBS_XY_TOL = 0.6        # 障碍物世界坐标变化容差（米）
        self.REUSE_MAX_EGO_S_JUMP = 5.0    # ego_s 跳变阈值（米，防止换参考线等）

        # 障碍物 ID 映射：Object.id(数值) -> 自定义 vehicle_id 字符串
        # 初次看到某个 Object.id 时，就从 CUSTOM_OBSTACLE_IDS 中取一个未用过的名称并绑定；
        # 之后该 Object.id 始终使用同一个名称。
        # TODO: remove this after testing
        self.object_id_map: dict[int, str] = {}

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

        self.adjusted_pose_sub = self.create_subscription(
            PoseStamped,
            '/adjusted_initialpose',
            self.adjusted_pose_callback,
            10,
        )

        # 订阅车辆对象列表（CARLA ROS Bridge）
        self.objects_sub = self.create_subscription(
            ObjectArray,
            '/carla/ego_vehicle/objects',
            self.objects_callback,
            10
        )

        # 订阅自车车辆信息以获取车长/轴距/最大转角/车宽（与控制模块使用相同话题，方案A）
        if CarlaEgoVehicleInfo is not None:
            self.ego_status_sub = self.create_subscription(
                CarlaEgoVehicleInfo,
                '/carla/ego_vehicle/vehicle_info',
                self.ego_info_callback,
                QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
        else:
            self.ego_status_sub = None

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

        # PathBoundary
        if PathBoundary is not None:
            self.path_boundary_pub = self.create_publisher(
                PathBoundary,
                '/path_smoothing/path_boundary',
                10
            )
        else:
            self.path_boundary_pub = None

        # QP 规划结果可视化：Frenet s-l Path
        self.qp_frenet_path_pub = self.create_publisher(
            Path,
            '/path_smoothing/qp_frenet_path',
            10,
        )

        # QP XY 路径（供控制模块 Stanley 跟踪）
        self.qp_xy_path_pub = self.create_publisher(
            Path,
            '/path_smoothing/qp_xy_path',
            10,
        )

        # 状态变量：用于接收 path_smoother 的 update_index / update_points 更新参考线
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

    def adjusted_pose_callback(self, msg: PoseStamped):
        """初始位姿调整后清空路径与 prev_qp_solution，避免复用上一条参考线的 QP 解。"""
        self.pending_update_index = None
        self.pending_update_points = None
        self.raw_path_s = []
        self.raw_path_l_upper = []
        self.raw_path_l_lower = []
        self.prev_qp_solution = {
            "s": None,
            "l": None,
            "dl": None,
            "ddl": None,
            "obs_signature": None,
            "obs_xy_map": None,
            "path_boundary_valid": False,
            "timestamp": None,
            "ego_s_at_solve": 0.0,
            "reference_segment": None,
            "accumulated_s": None,
        }
        with threading.Lock():
            self.current_waypoints = []
            self.waypoints_received = False
            self.last_nearest_idx = 0
        self.get_logger().info(
            f"{CYAN}✓ 感知：已清空路径与 QP 缓存{RESET}"
        )

    def ego_info_callback(self, msg):
        """
        自车车辆信息回调：从 CarlaEgoVehicleInfo.wheels 推断车宽、轴距、
        车长和最大转向角，供后续 Frenet QP 规划器使用。
        """
        if self.vehicle_info_received:
            return

        # 基本调试信息：确认回调确实被触发
        try:
            n_wheels = len(msg.wheels) if hasattr(msg, "wheels") else 0
        except Exception:
            n_wheels = -1
        self.get_logger().info(
            f"{CYAN}ego_info_callback 触发，wheels 数量 = {n_wheels}{RESET}"
        )

        if not hasattr(msg, "wheels") or not msg.wheels:
            self.get_logger().warn("vehicle_info消息中没有wheels信息，使用默认车辆参数")
            self._use_default_vehicle_info()
            return

        width_from_wheels = None
        wheel_x_positions = []
        wheel_radii = []
        wheel_steer_angles = []
        try:
            for w in msg.wheels:
                if hasattr(w, "position"):
                    if hasattr(w.position, "x"):
                        wheel_x_positions.append(float(w.position.x))
                if hasattr(w, "radius"):
                    wheel_radii.append(float(w.radius))
                if hasattr(w, "max_steer_angle") and float(w.max_steer_angle) > 0.0:
                    wheel_steer_angles.append(float(w.max_steer_angle))

            ys = [
                float(w.position.y)
                for w in msg.wheels
                if hasattr(w, "position") and hasattr(w.position, "y")
            ]
            if ys:
                max_y = max(ys)
                min_y = min(ys)
                est_width = max_y - min_y
                if est_width <= 0.0:
                    est_width = 2.0 * max(abs(max_y), abs(min_y))
                if est_width > 0.0:
                    width_from_wheels = est_width

            sorted_x = sorted(wheel_x_positions)
            if len(sorted_x) >= 2:
                half = max(1, len(sorted_x) // 2)
                rear_x = float(np.mean(sorted_x[:half]))
                front_x = float(np.mean(sorted_x[half:]))
                wheelbase_from_wheels = abs(front_x - rear_x)
                if wheelbase_from_wheels > 0.0:
                    self.vehicle_wheelbase = max(0.5, min(wheelbase_from_wheels, 5.0))

                    max_radius = max(wheel_radii) if wheel_radii else 0.0
                    # 车长：wheelbase + 轮半径贡献（radius 可能为 cm）
                    self.vehicle_length = max(
                        self.vehicle_wheelbase + 2.0 * max_radius / 100.0,
                        self.vehicle_wheelbase,
                    )
                    self.vehicle_length = max(1.5, min(self.vehicle_length, 8.0))

            if wheel_steer_angles:
                raw_angle = max(wheel_steer_angles)
                # CARLA 可能返回弧度或角度：>2 一般为角度(deg)，否则为弧度
                if raw_angle > 2.0:
                    raw_angle = math.radians(raw_angle)
                # 90°→tan=∞，限制为 70°≈1.22 rad
                self.vehicle_max_steering_angle = min(float(raw_angle), math.radians(70.0))
        except Exception:
            width_from_wheels = None
            self.get_logger().error(f"✗ vehicle_info 参数解析失败{RESET}")
            self._use_default_vehicle_info()
            return

        if width_from_wheels is not None:
            self.ego_half_width = width_from_wheels / 2.0
            self.get_logger().info(f"{GREEN}✓ 自车车宽: {width_from_wheels:.2f} m{RESET}")
        else:
            self.get_logger().warn("无法从vehicle_info推断车宽，保留默认值")

        self.vehicle_info_received = True
        self.get_logger().info(
            f"{GREEN}✓ 车辆参数已更新: "
            f"length={self.vehicle_length:.2f} m, "
            f"wheelbase={self.vehicle_wheelbase:.2f} m, "
            f"max_steering_angle={math.degrees(self.vehicle_max_steering_angle):.1f}°"
            f"{RESET}"
        )

    def _use_default_vehicle_info(self):
        """使用默认车辆信息。"""
        self.ego_half_width = EGO_WIDTH_DEFAULT / 2.0
        self.vehicle_length = 3.0
        self.vehicle_wheelbase = 2.8
        self.vehicle_max_steering_angle = 0.6
        self.vehicle_info_received = True

    # TODO: remove this after testing
    def _get_or_assign_vehicle_id(self, obj_id: int) -> str:
        """
        根据首次扫描顺序，将数值型 Object.id 映射为自定义的 vehicle_id 字符串。
        映射一旦建立，在节点生命周期内保持不变。
        """
        # 已有映射则直接返回
        if obj_id in self.object_id_map:
            return self.object_id_map[obj_id]

        # 还没有映射，则按顺序分配一个尚未使用的自定义 ID
        for name in CUSTOM_OBSTACLE_IDS:
            if name not in self.object_id_map.values():
                self.object_id_map[obj_id] = name
                return name

        # 自定义 ID 已用完：退回用原始数值字符串
        return str(obj_id)

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

        # 2. 基于该最近索引截取局部参考线（首尾截断）
        reference_segment, start_idx, end_idx = self.get_reference_path_segment(nearest_idx)
        if len(reference_segment) == 0:
            return

        # 3. 发布Frenet坐标（自车）
        self.publish_frenet_coordinates(reference_segment, nearest_idx)

        # 4. 发布SL边界（障碍物）与 PathBoundary
        self.publish_sl_boundaries(reference_segment, nearest_idx, start_idx)

    def get_reference_path_segment(self, center_idx: int):
        """
        获取参考线路径段：从 center_idx 前 BACKWARD_DISTANCE 到后 REFERENCE_LINE_HORIZON 的范围。
        首尾截断，保证在全局路径边界内。

        返回：
        reference_segment: 参考线路径段（numpy数组）
        start_idx: 在全局 waypoints 中的起始索引
        end_idx: 在全局 waypoints 中的结束索引（含）
        """
        if len(self.current_waypoints) == 0:
            return np.array([]), 0, 0

        backward_distance = BACKWARD_DISTANCE
        forward_distance = REFERENCE_LINE_HORIZON

        backward_idx = int(backward_distance / self.point_spacing)
        forward_idx = int(forward_distance / self.point_spacing)

        n = len(self.current_waypoints)
        center_idx = max(0, min(center_idx, n - 1))

        # 首尾截断（与 path_smoother 一致）
        start_idx = max(0, center_idx - backward_idx)
        end_idx = min(n - 1, center_idx + forward_idx)

        reference_segment = np.array(self.current_waypoints[start_idx:end_idx + 1])
        return reference_segment, start_idx, end_idx

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

    def publish_sl_boundaries(self, reference_path_segment, nearest_idx: int, start_idx: int):
        """
        计算并发布障碍物的SL边界

        参数：
        reference_path_segment: 参考线路径段
        nearest_idx: 自车在全局路径上的最近段索引
        start_idx: 参考线段在全局路径中的起始索引
        """
        if self.sl_boundary_array_pub is None or math_utils is None:
            return

        if len(reference_path_segment) < 2:
            return

        headings, accumulated_s_raw, kappas, dkappas, seg_len_flag = math_utils.compute_path_profile(
            reference_path_segment, self.point_spacing
        )
        if seg_len_flag:
            self.get_logger().error(f"✗ 路径点距过小/退化，s 计算可能不准确")
        # effective_backward = 自车到参考线段起点的弧长，裁剪掉假设后方 30m 的逻辑
        local_idx = min(nearest_idx - start_idx, len(accumulated_s_raw) - 1)
        local_idx = max(0, local_idx)
        s_ego_raw = float(accumulated_s_raw[local_idx])
        accumulated_s = accumulated_s_raw - s_ego_raw  # s=0 在自车位置

        ego_xy = np.array([self.current_x, self.current_y], dtype=float)

        # 收集所有障碍物的SLBoundary（无障碍物时为空，仍会发布 PathBoundary 全车道）
        sl_boundaries_list = []
        if self.latest_objects is not None and len(self.latest_objects.objects) > 0:
            for obj in self.latest_objects.objects:
                # 障碍物中心
                obj_xy = np.array([obj.pose.position.x, obj.pose.position.y], dtype=float)
                offset_vec = obj_xy - ego_xy
                dist_to_ego = np.linalg.norm(offset_vec)
                if dist_to_ego > 0.0:
                    # 参考线起点处切向方向 headings[0] 与 ego 前进方向近似一致
                    forward_yaw = float(headings[0]) if len(headings) > 0 else 0.0
                    forward_dir = np.array(
                        [math.cos(forward_yaw), math.sin(forward_yaw)], dtype=float
                    )
                    proj = float(np.dot(offset_vec, forward_dir))
                    if proj >= 0.0:
                        # 前方
                        if dist_to_ego > REFERENCE_LINE_HORIZON:
                            continue
                    else:
                        # 后方
                        if dist_to_ego > BACKWARD_DISTANCE:
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
                #boundary_msg.vehicle_id = str(obj.id)
                # TODO: remove this after testing
                boundary_msg.vehicle_id = self._get_or_assign_vehicle_id(obj.id)

                sl_boundaries_list.append(boundary_msg)

        # 一次性发布所有障碍物的SLBoundary数组
        if len(sl_boundaries_list) > 0 and self.sl_boundary_array_pub is not None:
            array_msg = SLBoundaryArray()
            array_msg.header.frame_id = "map"
            array_msg.header.stamp = self.get_clock().now().to_msg()
            array_msg.boundaries = sl_boundaries_list
            self.sl_boundary_array_pub.publish(array_msg)

        # 构建 obs 信息列表（每周期更新）并计算、发布 PathBoundary（无障碍物时也发布全车道边界）
        # 同时构建 id->obj_xy 的映射（从上面的循环结果中恢复，用于签名）
        obj_xy_map = {}  # oid -> (x, y) 世界坐标
        for obj in (self.latest_objects.objects if self.latest_objects else []):
            oid = self._get_or_assign_vehicle_id(obj.id)
            obj_xy = np.array([obj.pose.position.x, obj.pose.position.y], dtype=float)
            obj_xy_map[oid] = (float(obj_xy[0]), float(obj_xy[1]))

        obs_info_list = []
        for b in (sl_boundaries_list or []):
            oid = b.vehicle_id
            cs = list(b.boundary_s)
            cl = list(b.boundary_l)
            if len(cs) != 4 or len(cl) != 4:
                continue
            # 若整个障碍物在道路横向范围之外，则不参与 PathBoundary 计算
            if b.start_l > ROAD_LEFT_BOUNDARY_L + K_ZERO_VAL:
                continue
            if b.end_l < ROAD_RIGHT_BOUNDARY_L - K_ZERO_VAL:
                continue
            # 当前阶段：全部参与考虑，决策统一向左 nudge（保留接口）
            consider = True
            
            # TODO: remove this after testing
            
            decision = NudgeDecision.LEFT_NUDGE
            ref_left_s, ref_left_l, ref_right_s, ref_right_l = _get_obs_ref_points_(
                cs, cl, decision,
                100, -100,
            )
            obs_info_list.append({
                "id": oid,
                "consider": consider,
                "decision": decision,
                "corner_s": cs,
                "corner_l": cl,
                "start_s": b.start_s,
                "end_s": b.end_s,
                "ref_left_s": ref_left_s,
                "ref_left_l": ref_left_l,
                "ref_right_s": ref_right_s,
                "ref_right_l": ref_right_l,
                "obj_xy": obj_xy_map.get(oid, (None, None)),
            })
        if self.path_boundary_pub is not None:
            self._compute_and_publish_path_boundary(obs_info_list, kappas, accumulated_s, reference_path_segment)

    def _compute_obs_signature(self, obs_info_list: list) -> str:
        """
        生成障碍物特征签名，用于快速对比障碍物 ID 和决策是否发生变化。
        """
        parts = []
        for o in sorted(obs_info_list, key=lambda x: x["id"]):
            sig = f"{o['id']}:{o.get('decision', 'unknown')}"
            parts.append(sig)
        return "|".join(parts)

    def _check_obs_xy_reusable(self, obs_info_list: list) -> bool:
        """
        检查障碍物世界坐标是否变化在容差内。
        比较本周期的 obj_xy 与上周期的 obs_xy_map。
        返回 True 表示可以复用（所有障碍物位置变化都在容差内或数量/ID一致）。
        """
        prev = self.prev_qp_solution
        prev_xy_map = prev.get("obs_xy_map", None)
        if prev_xy_map is None:
            # 无上周期坐标数据，不能复用
            return False

        # 构建本周期的 xy_map
        current_xy_map = {}
        for o in obs_info_list:
            oid = o.get("id")
            obj_xy = o.get("obj_xy", (None, None))
            if obj_xy[0] is not None:
                current_xy_map[oid] = obj_xy

        # 检查 ID 集合是否一致
        if set(prev_xy_map.keys()) != set(current_xy_map.keys()):
            return False

        # 检查每个障碍物的位置变化
        for oid, prev_xy in prev_xy_map.items():
            curr_xy = current_xy_map.get(oid)
            if curr_xy is None:
                return False
            dist = math.hypot(curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1])
            if dist > self.REUSE_OBS_XY_TOL:
                return False

        return True

    def _check_path_reusable(
        self,
        obs_info_list: list,
        current_ego_s: float,
        current_ego_l: float,
    ) -> bool:
        """
        判断是否可以复用上周期的 PathBoundary 和 QP 解。
        检查条件：
        1. 上周期解有效（valid=True）
        2. 未超过最大复用时长
        3a. 障碍物签名一致（ID + decision 未变）
        3b. 障碍物世界坐标变化在容差内
        4. 自车横向偏差在容差内
        5. ego_s 无跳变（防止换参考线等）
        """
        prev = self.prev_qp_solution
        if not prev.get("path_boundary_valid", False):
            
            return False
        if prev["s"] is None:
            return False

        now = self.get_clock().now().nanoseconds / 1e9
        if prev["timestamp"] is None:
            return False
        # 条件2：最大复用时长
        if now - prev["timestamp"] > self.REUSE_MAX_TIME:
            return False

        # 条件5：ego_s 跳变检查（局部坐标系下 ego_s 应该接近上周期求解时的位置）
        if abs(current_ego_s - prev["ego_s_at_solve"]) > self.REUSE_MAX_EGO_S_JUMP:
            return False

        # 条件3a：障碍物签名对比（ID + decision）
        current_sig = self._compute_obs_signature(obs_info_list)
        if prev["obs_signature"] != current_sig:
            return False

        # 条件3b：障碍物世界坐标位置变化检查
        if not self._check_obs_xy_reusable(obs_info_list):
            return False

        # 条件4：自车横向偏差检查（在上周期路径上插值得到参考 l，与当前 l 比较）
        prev_s = prev["s"]
        prev_l = prev["l"]
        if prev_s is not None and prev_l is not None and len(prev_s) > 0:
            l_ref = float(np.interp(current_ego_s, prev_s, prev_l))
            if abs(current_ego_l - l_ref) > self.REUSE_MAX_L_DEV:
                return False

        return True

    def _get_init_state_from_prev_solution(
        self,
        ego_xy: np.ndarray,
        path_bound_lo: float,
        path_bound_hi: float,
        current_ego_l: float,
    ) -> tuple[float, float, float]:
        """
        从上周期 QP 解中插值获取当前 ego 位置在上周期参考线下的 (l, dl, ddl) 作为初始状态。
        使用本周期的 ego xy 坐标投影到上周期的参考线上，获取真实的 ego_s，再插值 QP 解。
        如果无上周期解或解无效，返回 (current_ego_l_clipped, 0.0, 0.0) 即使用当前 ego 的 l 值。
        返回的 l 会被裁剪到当前 path_bound 范围内。
        """
        prev = self.prev_qp_solution
        l_init = float(np.clip(current_ego_l, path_bound_lo, path_bound_hi))
        if prev["s"] is None or not prev.get("path_boundary_valid", False):
            # 无上周期解（首次运行），使用当前 ego 的 l 值
            return (l_init, 0.0, 0.0)

        prev_s = prev["s"]
        prev_l = prev["l"]
        prev_dl = prev["dl"]
        prev_ddl = prev["ddl"]
        prev_ref_seg = prev.get("reference_segment")
        prev_acc_s = prev.get("accumulated_s")

        if prev_s is None or len(prev_s) == 0 or prev_ref_seg is None or prev_acc_s is None:
            # 上周期数据不完整，使用当前 ego 的 l 值
            self.get_logger().warn("上周期数据不完整，使用当前 ego 的 l 值")
            return (l_init, 0.0, 0.0)

        # 将本周期的 ego xy 投影到上周期的参考线上
        ego_sl_res = self._xy_to_sl(ego_xy, prev_ref_seg, prev_acc_s)
        if ego_sl_res is None:
            # 投影失败，使用当前 ego 的 l 值
            self.get_logger().warn("投影失败，使用当前 ego 的 l 值")
            return (l_init, 0.0, 0.0)
        ego_s_on_prev = float(ego_sl_res[0])

        # 检查 ego_s 是否在上周期 QP 解的 s 范围内
        s_min = float(prev_s[0])
        s_max = float(prev_s[-1])
        if ego_s_on_prev < s_min - 1e-6 or ego_s_on_prev > s_max + 1e-6:
            # ego 超出上周期 QP 解范围，使用当前 ego 的 l 值
            self.get_logger().warn("ego 超出上周期 QP 解范围，使用当前 ego 的 l 值")
            return (l_init, 0.0, 0.0)

        # 在 ego_s_on_prev 处插值上周期的 QP 解
        l_ref = float(np.interp(ego_s_on_prev, prev_s, prev_l))
        dl_ref = float(np.interp(ego_s_on_prev, prev_s, prev_dl))
        ddl_ref = float(np.interp(ego_s_on_prev, prev_s, prev_ddl))

        # 裁剪到当前 path_bound 范围内
        l_init = float(np.clip(l_ref, path_bound_lo, path_bound_hi))
        # self.get_logger().warn(f"l_init: {l_init}, dl_ref: {dl_ref}, ddl_ref: {ddl_ref}")
        # self.get_logger().warn("-----------------------------------")

        return (l_init, dl_ref, ddl_ref)

    def _compute_and_publish_path_boundary(self, obs_info_list: list,
                                           kappas: np.ndarray,
                                           accumulated_s: np.ndarray,
                                           reference_path_segment: np.ndarray):
        """根据当前 obs 列表计算 PathBoundary 并发布，并在前 30m 调用 QP 生成局部避障路径。
        支持 QP 解复用，仅在必要时重新计算。
        """
        # 计算当前 ego 的 SL 坐标（用于复用判断和后续计算）
        ego_xy = np.array([self.current_x, self.current_y], dtype=float)
        ego_sl_res = self._xy_to_sl(ego_xy, reference_path_segment, accumulated_s)
        if ego_sl_res is not None:
            current_ego_s, current_ego_l = ego_sl_res[0], ego_sl_res[1]
        else:
            current_ego_s, current_ego_l = 0.0, 0.0

        # 检查是否可以复用上周期的 QP 解
        qp_reusable = self._check_path_reusable(obs_info_list, current_ego_s, current_ego_l)

        # s 从 ego 位置(0) 到局部参考线终点的弧长，起点/终点时做裁剪
        s_min_eff = 0.0
        s_max_eff = min(
            REFERENCE_LINE_HORIZON,
            max(0.0, float(accumulated_s[-1])) if len(accumulated_s) > 0 else REFERENCE_LINE_HORIZON
        )
        s_samples = np.arange(s_min_eff, s_max_eff + K_ZERO_VAL, PATH_BOUNDARY_S_RESOLUTION)
        if len(s_samples) == 0:
            return
        # 最终（已施加 ADC bound + lat_buffer）的 PathBoundary
        s_list = []
        l_upper_list = []
        l_lower_list = []
        raw_s_list = []
        raw_l_upper_list = []
        raw_l_lower_list = []
        valid = True
        prev_two = None

        for ego_s in s_samples:
            active = [
                o for o in obs_info_list
                if o["consider"] and o["start_s"] <= ego_s <= o["end_s"]
            ]
            # 道路左右边界
            road_left = (ROAD_LEFT_BOUNDARY_L, ROAD_LEFT_BOUNDARY_L, "road_left", None)
            road_right = (ROAD_RIGHT_BOUNDARY_L, ROAD_RIGHT_BOUNDARY_L, "road_right", None)
            # 先只对障碍物内部排序（当前 s 下无 l 投影的 obs 不参与）
            obs_elements = []
            for o in active:
                ext = _get_obs_l_extent_at_s(o, ego_s)
                if ext is None:
                    continue
                l_lo, l_hi = ext
                obs_elements.append((l_lo, l_hi, o["id"], o))
            # 按 l_upper 从大到小排序
            obs_elements.sort(key=lambda x: x[1], reverse=True)

            elements = [road_left] + obs_elements + [road_right]

            # 只保留 obs 的决策，按 l_upper 从大到小
            obs_order = [(e[2], e[3], e[0], e[1]) for e in elements if e[3] is not None]
            decisions = [item[1]["decision"] for item in obs_order]
            n_left = sum(1 for d in decisions if d == NudgeDecision.LEFT_NUDGE)
            n_right = sum(1 for d in decisions if d == NudgeDecision.RIGHT_NUDGE)
            change_count = 0
            for i in range(len(decisions) - 1):
                if decisions[i] != decisions[i + 1]:
                    change_count += 1
            if change_count > 1:
                valid = False
                self.get_logger().error(
                    "PathBoundary: 决策类型变化超过一次，请求停车"
                )
                break
            first_obs_decision = decisions[0] if decisions else None
            if first_obs_decision == NudgeDecision.LEFT_NUDGE and n_right > 0:
                valid = False
                self.get_logger().error(
                    "PathBoundary: 第一个 obs 为左 nudge 但存在右 nudge，请求停车"
                )
                break

            # 选取两个元素
            need_recompute = True
            if prev_two is not None:
                a_key, b_key = prev_two
                path_lo_old, path_hi_old = _path_bounds_from_two_elements(
                    elements, a_key, b_key, ego_s, obs_info_list
                )
                if path_lo_old is None:
                    need_recompute = True
                else:
                    def _obs_ended(obs_list, key, s):
                        if key in ("road_left", "road_right"):
                            return False
                        for o in obs_list:
                            if o["id"] == key:
                                return s > o["end_s"]
                        return False
                    left_obs_ended = _obs_ended(obs_info_list, a_key, ego_s) or _obs_ended(obs_info_list, b_key, ego_s)
                    if left_obs_ended:
                        need_recompute = True
                    else:
                        for o in active:
                            if o["id"] == a_key or o["id"] == b_key:
                                continue
                            ext = _get_obs_l_extent_at_s(o, ego_s)
                            if ext is None:
                                continue
                            l_lo, l_hi = ext
                            if _intervals_overlap(
                                path_lo_old, path_hi_old, l_lo, l_hi
                            ):
                                need_recompute = True
                                break
                        if not need_recompute:
                            # 先按道路几何边界裁剪
                            path_lo_old = max(path_lo_old, ROAD_RIGHT_BOUNDARY_L)
                            path_hi_old = min(path_hi_old, ROAD_LEFT_BOUNDARY_L)
                            # 保存收缩前（仅考虑道路与障碍物收缩）的 PathBoundary
                            raw_s_list.append(ego_s)
                            raw_l_lower_list.append(path_lo_old)
                            raw_l_upper_list.append(path_hi_old)
                            # 再施加 ADC bound 与横向 buffer，得到 ego 中心可行域
                            shrink = self.ego_half_width + EGO_LAT_BUFFER
                            adc_lo = path_lo_old + shrink
                            adc_hi = path_hi_old - shrink
                            if adc_hi <= adc_lo:
                                if ego_s <= BLOCK_STOP_DISTANCE:
                                    valid = False
                                    self.get_logger().error(
                                        f"PathBoundary: s={ego_s:.2f} 处 ADC 收缩后无可行走廊，判定 blocked（≤{BLOCK_STOP_DISTANCE:.1f}m 内）"
                                    )
                                    # self.get_logger().error(f"path_lo_old: {path_lo_old:.2f}, path_hi_old: {path_hi_old:.2f}, shrink: {shrink:.2f}, adc_lo: {adc_lo:.2f}, adc_hi: {adc_hi:.2f}")
                                    break
                                else:
                                    # 远处先仅告警，不立即判定整条 PathBoundary 失效
                                    self.get_logger().warn(
                                        f"PathBoundary: s={ego_s:.2f} 处 ADC 收缩后无可行走廊，暂不 blocked（>{BLOCK_STOP_DISTANCE:.1f}m）"
                                    )
                                    # self.get_logger().error(f"path_lo_old: {path_lo_old:.2f}, path_hi_old: {path_hi_old:.2f}, shrink: {shrink:.2f}, adc_lo: {adc_lo:.2f}, adc_hi: {adc_hi:.2f}")
                                    continue
                            s_list.append(ego_s)
                            l_upper_list.append(adc_hi)
                            l_lower_list.append(adc_lo)
                            continue

            if need_recompute:
                # 当 change_count==1 且两个 obs 都在 obs_elements 时，
                # 若某条边的 3 点插值在当前 s 不覆盖（_get_obs_edge_l_at_s 返回 None），_path_bounds_from_two_elements 会返回 (None, None)
                # 所以需要记录其 l 范围做 fallback，避免误用全道路
                fallback_lower = None
                fallback_upper = None
                if change_count == 1:
                    last_right_idx = None
                    first_left_idx = None
                    for i, (_, obs, l_lo, l_hi) in enumerate(obs_order):
                        if obs["decision"] == NudgeDecision.RIGHT_NUDGE:
                            last_right_idx = i
                        if first_left_idx is None and obs["decision"] == NudgeDecision.LEFT_NUDGE:
                            first_left_idx = i
                    if last_right_idx is not None and first_left_idx is not None:
                        elem_right = elements[1 + last_right_idx]
                        elem_left = elements[1 + first_left_idx]
                        fallback_upper = elem_right[0]
                        fallback_lower = elem_left[1]
                        prev_two = (elem_right[2], elem_left[2])
                    else:
                        path_upper = ROAD_LEFT_BOUNDARY_L
                        path_lower = ROAD_RIGHT_BOUNDARY_L
                        prev_two = ("road_left", "road_right")
                else:
                    if n_left > 0:
                        first_obs = obs_order[0]
                        path_upper = ROAD_LEFT_BOUNDARY_L
                        path_lower = first_obs[3]
                        prev_two = ("road_left", first_obs[1]["id"])
                    elif n_right > 0:
                        last_obs = obs_order[-1]
                        path_upper = last_obs[2]
                        path_lower = ROAD_RIGHT_BOUNDARY_L
                        prev_two = (last_obs[1]["id"], "road_right")
                    else:
                        path_upper = ROAD_LEFT_BOUNDARY_L
                        path_lower = ROAD_RIGHT_BOUNDARY_L
                        prev_two = ("road_left", "road_right")

                path_lower, path_upper = _path_bounds_from_two_elements(
                    elements, prev_two[0], prev_two[1], ego_s, obs_info_list
                )

                if path_upper is None or path_lower is None:
                    # n_left>0 or n_right>0 时不应回退到全道路；change_count==1 且已选出两障时用 extent 做 fallback
                    if fallback_lower is not None and fallback_upper is not None:
                        path_lower = fallback_lower
                        path_upper = fallback_upper
                    else:
                        path_upper = ROAD_LEFT_BOUNDARY_L
                        path_lower = ROAD_RIGHT_BOUNDARY_L

                # 先按道路边界裁剪
                path_lower = max(path_lower, ROAD_RIGHT_BOUNDARY_L)
                path_upper = min(path_upper, ROAD_LEFT_BOUNDARY_L)
                # 保存收缩前（仅考虑道路与障碍物收缩）的 PathBoundary
                raw_s_list.append(ego_s)
                raw_l_lower_list.append(path_lower)
                raw_l_upper_list.append(path_upper)
                # 再施加 ADC bound 和横向 buffer
                shrink = self.ego_half_width + EGO_LAT_BUFFER
                adc_lo = path_lower + shrink
                adc_hi = path_upper - shrink
                if adc_hi <= adc_lo:
                    if ego_s <= BLOCK_STOP_DISTANCE:
                        valid = False
                        self.get_logger().error(
                            f"PathBoundary: s={ego_s:.2f} 处 ADC 收缩后无可行走廊，判定 blocked（≤{BLOCK_STOP_DISTANCE:.1f}m 内）"
                        )
                        # self.get_logger().error(f"path_lo_old: {path_lower:.2f}, path_hi_old: {path_upper:.2f}, shrink: {shrink:.2f}, adc_lo: {adc_lo:.2f}, adc_hi: {adc_hi:.2f}")
                        break
                    else:
                        self.get_logger().warn(
                            f"PathBoundary: s={ego_s:.2f} 处 ADC 收缩后无可行走廊，暂不 blocked（>{BLOCK_STOP_DISTANCE:.1f}m）"
                        )
                        # self.get_logger().error(f"path_lo_old: {path_lower:.2f}, path_hi_old: {path_upper:.2f}, shrink: {shrink:.2f}, adc_lo: {adc_lo:.2f}, adc_hi: {adc_hi:.2f}")
                        continue

                s_list.append(ego_s)
                l_upper_list.append(adc_hi)
                l_lower_list.append(adc_lo)

        # 缓存未施加 ADC bound / lat_buffer 前的 PathBoundary，便于后续 ADCVertexConstraints 使用
        self.raw_path_s = raw_s_list
        self.raw_path_l_upper = raw_l_upper_list
        self.raw_path_l_lower = raw_l_lower_list

        # 发布 PathBoundary
        if self.path_boundary_pub is not None:
            msg = PathBoundary()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.s = [float(x) for x in s_list]
            msg.l_upper = [float(x) for x in l_upper_list]
            msg.l_lower = [float(x) for x in l_lower_list]
            msg.valid = valid
            self.path_boundary_pub.publish(msg)

        # 在前 30m 运行局部 QP，生成 Frenet QP 路径并通过 Path 更新控制端 waypoints
        try:
            if not valid:
                return
            if len(s_list) == 0:
                return
            # 只取 0–30m 区间
            s_arr = np.asarray(s_list, dtype=float)
            l_lower_arr = np.asarray(l_lower_list, dtype=float)
            l_upper_arr = np.asarray(l_upper_list, dtype=float)
            raw_s_arr = np.asarray(self.raw_path_s, dtype=float)
            raw_lo_arr = np.asarray(self.raw_path_l_lower, dtype=float)
            raw_hi_arr = np.asarray(self.raw_path_l_upper, dtype=float)
            mask = (s_arr >= 0.0) & (s_arr <= 30.0 + K_ZERO_VAL)
            if not mask.any():
                return
            s_qp = s_arr[mask]
            path_bound_qp = np.stack([l_lower_arr[mask], l_upper_arr[mask]], axis=1)
            raw_bound_qp = np.stack([raw_lo_arr[mask], raw_hi_arr[mask]], axis=1)
            if np.any(path_bound_qp[:, 0] > path_bound_qp[:, 1]) or np.any(raw_bound_qp[:, 0] > raw_bound_qp[:, 1]):
                return

            # 车辆参数合法性：wheelbase≈0 会导致 max_kappa 爆炸，FrenetQPPlanner 内部会 clamp
            if (self.vehicle_wheelbase < 0.3 or self.vehicle_length < 0.3
                    or self.ego_half_width <= 1e-6):
                self.get_logger().warn(
                    f"QP 跳过：车辆参数异常 wheelbase={self.vehicle_wheelbase:.3f} "
                    f"length={self.vehicle_length:.3f} half_w={self.ego_half_width:.3f}"
                )
                return

            # 判断是否复用上周期的 QP 解
            n_qp = s_qp.shape[0]
            prev = self.prev_qp_solution
            can_reuse_qp = (qp_reusable
                           and prev["s"] is not None
                           and prev["l"] is not None
                           and len(prev["s"]) == n_qp)

            if can_reuse_qp:
                # 复用上周期的 QP 解（长度匹配时）
                l_sol = prev["l"].copy()
                dl_sol = prev["dl"].copy() if prev["dl"] is not None else np.zeros_like(l_sol)
                ddl_sol = prev["ddl"].copy() if prev["ddl"] is not None else np.zeros_like(l_sol)
                self.get_logger().debug("QP 解复用：长度匹配，跳过重新求解")

            else:
                # 初始状态：从上周期 QP 解插值获取 (l, dl, ddl)，实现平滑衔接
                # 将本周期的 ego xy 投影到上周期的参考线上获取真实 ego_s，再插值 QP 解
                # TODO: 当前相当于是估计值，后续需要融合传感器数值进行卡尔曼滤波获得更精确的初值
                # self.get_logger().info("重新求解中")
                lo0, hi0 = path_bound_qp[0, 0], path_bound_qp[0, 1]
                l_init, dl_init, ddl_init = self._get_init_state_from_prev_solution(
                    ego_xy=ego_xy,  # 本周期的 ego xy 投影到上周期的参考线
                    path_bound_lo=lo0,
                    path_bound_hi=hi0,
                    current_ego_l=current_ego_l,  # 当前 ego 在当前参考线下的 l 值
                )

                delta_s_qp = float(PATH_BOUNDARY_S_RESOLUTION)

                # 参考曲率：按 accumulated_s 与 kappas 插值到 s_qp 上，裁剪避免极端值导致 QP 约束非法
                kappa_ref_qp = np.interp(s_qp, accumulated_s, kappas)
                kappa_max_safe = 1.0
                kappa_ref_qp = np.clip(np.nan_to_num(kappa_ref_qp, nan=0.0, posinf=kappa_max_safe, neginf=-kappa_max_safe), -kappa_max_safe, kappa_max_safe)

                planner = FrenetQPPlanner(
                    n=n_qp,
                    delta_s=delta_s_qp,
                    vehicle_length=self.vehicle_length,
                    vehicle_width=self.ego_half_width * 2.0,
                    max_steering_angle=self.vehicle_max_steering_angle,
                    max_steering_rate=0.5,
                    wheelbase=self.vehicle_wheelbase,
                )
                def _qp_debug(**kw):
                    reason = kw.get("reason", "?")
                    v = kw.get("v_current", 0.0)
                    self.get_logger().error(
                        f"QP 失败: {reason} | n={kw.get('n')} delta_s={kw.get('delta_s')} "
                        f"v={v:.2f} half_w={kw.get('half_width', 0):.2f} "
                        f"half_L={kw.get('half_length', 0):.2f} max_κ={kw.get('max_kappa', 0):.3f}"
                    )
                    pb = kw.get("path_boundary")
                    rpb = kw.get("raw_path_boundary")
                    kr = kw.get("kappa_ref")
                    if pb is not None and len(pb) > 0:
                        self.get_logger().error(
                            f"  path_bound[0]: l∈[{pb[0,0]:.3f},{pb[0,1]:.3f}] "
                            f"raw[0]: l∈[{rpb[0,0]:.3f},{rpb[0,1]:.3f}] κ_ref[0]={kr[0]:.4f}"
                        )
                    lv, uv = kw.get("l_vec"), kw.get("u_vec")
                    if lv is not None and uv is not None:
                        bad = np.where(lv > uv)[0]
                        if len(bad) > 0:
                            i = int(bad[0])
                            self.get_logger().error(f"  l>u 首处 idx={i} l[{i}]={lv[i]:.4e} u[{i}]={uv[i]:.4e}")

                sol = planner.solve(
                    path_boundary=path_bound_qp,
                    raw_path_boundary=raw_bound_qp,
                    kappa_ref=kappa_ref_qp,
                    v_current=self.current_speed,
                    l_init=l_init,
                    dl_init=dl_init,
                    ddl_init=ddl_init,
                    l_ref=np.zeros_like(s_qp),
                    debug_callback=_qp_debug,
                )
                if sol is None:
                    self.prev_qp_solution["path_boundary_valid"] = False
                    return
                l_sol, dl_sol, ddl_sol = sol

            # 保存 QP 相关状态供后续复用判断和平滑初始状态使用
            now_ts = self.get_clock().now().nanoseconds / 1e9
            obs_xy_map = {}
            for o in obs_info_list:
                oid = o.get("id")
                obj_xy = o.get("obj_xy", (None, None))
                if obj_xy[0] is not None:
                    obs_xy_map[oid] = (float(obj_xy[0]), float(obj_xy[1]))
            if can_reuse_qp:
                self.prev_qp_solution["obs_signature"] = self._compute_obs_signature(obs_info_list)
                self.prev_qp_solution["obs_xy_map"] = obs_xy_map
                self.prev_qp_solution["path_boundary_valid"] = valid
            else:
                self.prev_qp_solution = {
                    "s": np.asarray(s_qp, dtype=float).copy(),
                    "l": np.asarray(l_sol, dtype=float).copy(),
                    "dl": np.asarray(dl_sol, dtype=float).copy(),
                    "ddl": np.asarray(ddl_sol, dtype=float).copy(),
                    "obs_signature": self._compute_obs_signature(obs_info_list),
                    "obs_xy_map": obs_xy_map,
                    "path_boundary_valid": valid,
                    "timestamp": now_ts,
                    "ego_s_at_solve": 0.0,
                    "reference_segment": np.asarray(reference_path_segment, dtype=float).copy(),
                    "accumulated_s": np.asarray(accumulated_s, dtype=float).copy(),
                }

            # 可视化：发布 Frenet s-l Path
            # if self.qp_frenet_path_pub is not None:
            #     path_msg = Path()
            #     path_msg.header.frame_id = "map"
            #     path_msg.header.stamp = self.get_clock().now().to_msg()
            #     for si, li in zip(s_qp, l_sol):
            #         ps = PoseStamped()
            #         ps.header = path_msg.header
            #         ps.pose.position.x = float(si)
            #         ps.pose.position.y = float(li)
            #         ps.pose.position.z = 0.0
            #         ps.pose.orientation.w = 1.0
            #         path_msg.poses.append(ps)
            #     self.qp_frenet_path_pub.publish(path_msg)

            # 将 Frenet (s,l) 转换为 XY，发布到 qp_xy_path 供控制器 Stanley 跟踪
            if self.qp_xy_path_pub is None or can_reuse_qp:
                return
            # 计算 s→XY：在 reference_path_segment 上按 accumulated_s 插值
            xy_points: list[tuple[float, float]] = []
            acc_s = np.asarray(accumulated_s, dtype=float)
            pts = np.asarray(reference_path_segment, dtype=float)
            for si, li in zip(s_qp, l_sol):
                # 找到 si 所在的参考线段
                if si <= acc_s[0]:
                    idx = 0
                    t = 0.0
                elif si >= acc_s[-1]:
                    idx = len(acc_s) - 2
                    t = 1.0
                else:
                    idx = np.searchsorted(acc_s, si) - 1
                    idx = max(0, min(idx, len(acc_s) - 2))
                    ds = acc_s[idx + 1] - acc_s[idx]
                    t = 0.0 if abs(ds) < 1e-6 else float((si - acc_s[idx]) / ds)
                p0 = pts[idx]
                p1 = pts[idx + 1]
                seg = p1 - p0
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-6:
                    base_xy = p0
                    theta = 0.0
                else:
                    base_xy = p0 + t * seg
                    theta = math.atan2(seg[1], seg[0])
                # 左侧为正：法向量
                nx = -math.sin(theta)
                ny = math.cos(theta)
                x = float(base_xy[0] + li * nx)
                y = float(base_xy[1] + li * ny)
                xy_points.append((x, y))

            if not xy_points:
                return

            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for x, y in xy_points:
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.position.z = 0.0
                ps.pose.orientation.w = 1.0
                path_msg.poses.append(ps)
            self.qp_xy_path_pub.publish(path_msg)

        except Exception as e:
            self.get_logger().error(f"QP 路径规划或发布失败: {e}")

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

        # 与路径区间求交集
        path_min = float(accumulated_s[0])
        path_max = float(accumulated_s[-1])
        s_start = max(path_min, s_start)
        s_end = min(path_max, s_end)
        if s_end - s_start < 1e-6:
            return None

        # 在 accumulated_s 上找到覆盖 [s_start, s_end] 的段索引范围
        # 先用 s/point_spacing 估算索引，再做小范围修正，避免全局线性扫描
        spacing = self.point_spacing if self.point_spacing > 1e-6 else 1.0
        # 估算起止索引（四舍五入），注意要减去 path_min
        i_start_est = int(round((s_start - path_min) / spacing))
        i_end_est = int(round((s_end - path_min) / spacing))

        # 初步裁剪到合法范围 [0, n-2]
        i_start = max(0, min(i_start_est, n - 2))
        i_end = max(0, min(i_end_est, n - 2))

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

        path_min = float(accumulated_s[0])
        path_max = float(accumulated_s[-1])

        if mode == "global":
            # 在整段局部参考线上搜索最近点（s 范围为 [path_min, path_max]，例如 [-BACKWARD_DISTANCE, REFERENCE_LINE_HORIZON]）
            return self._xy_to_sl_with_s_window(
                point_xy,
                reference_path_segment,
                accumulated_s,
                path_min,
                path_max,
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
            path_min,
            path_max,
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

