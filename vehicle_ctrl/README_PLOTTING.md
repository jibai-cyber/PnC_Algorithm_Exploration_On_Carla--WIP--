# 车辆控制绘图架构说明

## 概述

车辆控制系统采用分离式架构，将控制逻辑和可视化完全分离，以确保控制周期不受绘图操作影响。

## 架构组成

### 1. 控制节点 (`simple_ctrl.py`)
- **功能**：车辆控制逻辑、路径跟踪、PID控制
- **发布话题**：
  - `/carla/ego_vehicle/vehicle_control_cmd`：车辆控制命令
  - `/vehicle_control/plot_data`：绘图数据（Float64MultiArray）
- **订阅话题**：
  - `/initialpose`：初始位置
  - `/goal_pose`：目标点
  - `/carla/ego_vehicle/waypoints`：路径规划结果
  - `/carla/ego_vehicle/odometry`：里程计数据
  - `/carla/ego_vehicle/vehicle_status`：车辆状态
  - `/carla/ego_vehicle/vehicle_info`：车辆信息
  - `/carla/ego_vehicle/imu`：IMU数据

### 2. 绘图节点 (`vehicle_plotter.py`)
- **功能**：实时可视化控制数据
- **订阅话题**：
  - `/vehicle_control/plot_data`：绘图数据
- **显示内容**：
  - 横向误差（Cross Track Error）
  - 航向误差（Heading Error）
  - 归一化转向角（Normalized Steer）
  - 车辆速度和加速度
  - 油门和刹车
  - 速度/加速度误差

## 数据格式

绘图数据使用 `std_msgs/Float64MultiArray` 格式：

```python
plot_data.data = [
    timestamp,              # 时间戳 (秒)
    cross_track_error,      # 横向误差 (米)
    heading_error,          # 航向误差 (弧度)
    normalized_steer,       # 归一化转向角 (-1~1)
    vehicle_speed,          # 车辆速度 (m/s)
    vehicle_accel,          # 车辆加速度 (m/s²)
    throttle,               # 油门 (0~1)
    brake,                  # 刹车 (0~1)
    speed_error,            # 速度误差
    accel_error             # 加速度误差
]
```

## 使用方法

### 1. 编译包
```bash
cd /path/to/ws
colcon build --packages-select vehicle_ctrl
source install/setup.bash
```

### 2. 启动控制节点
```bash
ros2 run vehicle_ctrl simple_ctrl
```

### 3. 启动绘图节点（在新终端）
```bash
export ROS_DOMAIN_ID=200
ros2 run vehicle_ctrl vehicle_plotter
```

## 性能优势

### 原架构问题
- 绘图操作在控制循环中执行
- matplotlib绘图耗时影响控制周期
- 实际控制周期可达0.16~0.17秒

### 新架构优势
- 控制周期精确为0.05秒，不受绘图影响
- 绘图节点独立运行，可调整更新频率
- 支持多节点部署和调试
- 更好的资源管理和模块化

## 配置参数

### 控制节点参数
- `enable_plotting`: 是否启用绘图数据发布（默认True）
- `control_dt`: 控制周期（默认0.05秒）

### 绘图节点参数
- `enable_plotting`: 是否启用绘图（默认True）
- `plot_update_rate`: 绘图更新频率（默认10Hz）
- `error_history_size`: 数据历史大小（默认500）