#!/usr/bin/env python3
import sys
import os
import glob
import time
import argparse
import math

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('/home/D/carla_simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def find_ego_vehicle(world, role_name='ego_vehicle'):
    """
    查找自车（ego vehicle）
    
    Args:
        world: CARLA世界对象
        role_name: 自车的role_name属性，默认为'hero'
    
    Returns:
        ego_vehicle: 自车对象，如果未找到返回None
    """
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        vehicle_role = vehicle.attributes.get('role_name', '')
        if vehicle_role == role_name:
            return vehicle
    return None


def list_all_vehicles(world):
    """
    列出所有车辆及其role_name（用于调试）
    """
    vehicles = world.get_actors().filter('vehicle.*')
    print("当前所有车辆:")
    for vehicle in vehicles:
        role_name = vehicle.attributes.get('role_name', 'None')
        print(f"  Vehicle ID: {vehicle.id}, role_name: '{role_name}'")


def calculate_position_ahead(world, ego_transform, distance):
    """
    计算自车正前方的道路位置
    
    Args:
        world: CARLA世界对象
        ego_transform: 自车的transform
        distance: 前方距离（米）
    
    Returns:
        spawn_transform: 前方道路位置的transform，如果找不到道路则返回计算位置
    """
    try:
        # 获取地图
        carla_map = world.get_map()
        
        # 获取自车当前位置的waypoint
        ego_waypoint = carla_map.get_waypoint(ego_transform.location)
        
        if ego_waypoint is None:
            # 如果找不到waypoint，使用简单计算
            logging.warning('无法找到自车位置的waypoint，使用简单计算')
            yaw_rad = math.radians(ego_transform.rotation.yaw)
            forward_vector = carla.Vector3D(
                math.cos(yaw_rad),
                math.sin(yaw_rad),
                0.0
            )
            new_location = ego_transform.location + forward_vector * distance
            return carla.Transform(new_location, ego_transform.rotation)
        
        # 沿着道路向前查找指定距离的waypoint
        current_waypoint = ego_waypoint
        accumulated_distance = 0.0
        step_size = min(5.0, distance / 2.0)  # 步长，不超过距离的一半
        
        # 沿着道路向前移动
        while accumulated_distance < distance:
            # 获取下一个waypoint（沿着当前道路方向）
            next_waypoints = current_waypoint.next(step_size)
            if not next_waypoints:
                # 如果没有下一个waypoint，尝试使用当前waypoint
                break
            
            # 选择最接近自车朝向的waypoint（确保是正前方）
            best_waypoint = next_waypoints[0]
            ego_yaw = ego_transform.rotation.yaw
            
            # 如果有多个waypoint，选择方向最接近的
            if len(next_waypoints) > 1:
                min_angle_diff = 180.0
                for wp in next_waypoints:
                    wp_yaw = wp.transform.rotation.yaw
                    angle_diff = abs(ego_yaw - wp_yaw)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_waypoint = wp
            
            # 计算到下一个waypoint的距离
            segment_distance = current_waypoint.transform.location.distance(best_waypoint.transform.location)
            
            if accumulated_distance + segment_distance >= distance:
                # 需要在这个segment中插值
                remaining_distance = distance - accumulated_distance
                if segment_distance > 0:
                    ratio = remaining_distance / segment_distance
                else:
                    ratio = 0
                
                # 插值计算位置
                start_loc = current_waypoint.transform.location
                end_loc = best_waypoint.transform.location
                interpolated_location = carla.Location(
                    start_loc.x + (end_loc.x - start_loc.x) * ratio,
                    start_loc.y + (end_loc.y - start_loc.y) * ratio,
                    end_loc.z  # 使用目标waypoint的z坐标（路面高度）
                )
                
                # 创建transform，使用目标waypoint的旋转（道路方向）
                spawn_transform = carla.Transform(interpolated_location, best_waypoint.transform.rotation)
                return spawn_transform
            
            accumulated_distance += segment_distance
            current_waypoint = best_waypoint
        
        # 如果已经到达或超过目标距离，使用当前waypoint
        spawn_transform = current_waypoint.transform
        return spawn_transform
        
    except Exception as e:
        # 如果出错，使用简单计算作为后备方案
        logging.warning(f'计算前方道路位置时出错: {str(e)}，使用简单计算')
        yaw_rad = math.radians(ego_transform.rotation.yaw)
        forward_vector = carla.Vector3D(
            math.cos(yaw_rad),
            math.sin(yaw_rad),
            0.0
        )
        new_location = ego_transform.location + forward_vector * distance
        return carla.Transform(new_location, ego_transform.rotation)


def create_following_scenario(host='localhost', port=2000, tm_port=8002, 
                              vehicle_filter='vehicle.*', 
                              ahead_distance=10.0, travel_distance=15.0,
                              role_name='ego_vehicle', max_wait_time=30,
                              position_jump_threshold=20.0):
    """
    创建跟车场景（检测自车位置大幅跳跃时生成环境车辆）
    
    Args:
        host: CARLA服务器地址
        port: CARLA服务器端口
        tm_port: Traffic Manager端口
        vehicle_filter: 车辆类型过滤
        ahead_distance: 在自车前方的距离（米），默认10米
        travel_distance: 车辆行驶距离（米），默认10米（减速行驶后停车）
        role_name: 自车的role_name属性
        max_wait_time: 最大等待自车出现的时间（秒）
        position_jump_threshold: 位置跳跃阈值（米），超过此值认为是新起点，默认20米
    """
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    
    vehicle = None
    traffic_manager = None
    last_ego_location = None  # 记录上一次检测到的自车位置
    blueprint = None
    
    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_synchronous_mode(True)
        
        # 获取车辆蓝图（只需获取一次）
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter(vehicle_filter)
        vehicle_blueprints = [bp for bp in vehicle_blueprints if bp.get_attribute('number_of_wheels').as_int() == 4]
        
        if not vehicle_blueprints:
            logging.error(f'未找到匹配的车辆蓝图: {vehicle_filter}')
            return
        
        # 保存蓝图库引用，每次生成时从库中获取新蓝图
        blueprint_template = vehicle_blueprints[0]
        
        print(f'场景2：跟车场景已启动')
        print(f'将在自车前方 {ahead_distance} 米生成环境车辆，手动控制减速行驶 {travel_distance} 米后停车')
        print(f'当自车位置跳跃超过 {position_jump_threshold} 米时生成环境车辆（支持RViz 2D Pose Estimate）')
        print(f'正在查找自车（role_name="{role_name}"）...')
        print('按 Ctrl+C 退出...')
        
        # 初始化变量
        start_location = None
        traveled_distance = 0.0
        stopping = False
        first_check = True
        
        # 主循环：检测自车位置大幅跳跃
        while True:
            world.wait_for_tick()
            
            # 查找自车
            ego_vehicle = find_ego_vehicle(world, role_name)
            
            if ego_vehicle is None:
                # 如果自车不存在，销毁环境车辆并重置记录
                if vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()
                    vehicle = None
                last_ego_location = None
                
                # 第一次检查时列出所有车辆（用于调试）
                if first_check:
                    list_all_vehicles(world)
                    print(f'\n提示：如果找不到自车，请检查Terminal 3生成的自车的role_name是否为"{role_name}"')
                    print('如果role_name不同，请使用 --role-name 参数指定正确的role_name')
                    first_check = False
                
                time.sleep(1)
                continue
            
            # 找到自车后，重置first_check标志
            if first_check:
                first_check = False
            
            # 获取自车当前位置
            ego_location = ego_vehicle.get_transform().location
            
            # 检查是否需要生成环境车辆
            need_generate = False
            
            if last_ego_location is None:
                # 第一次找到自车，需要生成场景
                need_generate = True
                print(f'找到自车，ID: {ego_vehicle.id}')
            else:
                # 检查位置是否发生大幅跳跃（2D Pose Estimate）
                position_jump = last_ego_location.distance(ego_location)
                if position_jump >= position_jump_threshold:
                    need_generate = True
                    print(f'检测到自车位置大幅跳跃 {position_jump:.2f} 米（可能是2D Pose Estimate设置新起点），生成环境车辆...')
            
            # 如果需要生成环境车辆
            if need_generate:
                # 销毁旧的环境车辆
                if vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()
                    vehicle = None
                
                # 记录新的自车位置
                last_ego_location = ego_location
                
                # 获取自车transform
                ego_transform = ego_vehicle.get_transform()
                print(f'自车位置: ({ego_transform.location.x:.2f}, {ego_transform.location.y:.2f}), 朝向: {ego_transform.rotation.yaw:.2f}°')
                
                # 计算正前方道路位置
                spawn_transform = calculate_position_ahead(world, ego_transform, ahead_distance)
                print(f'环境车辆生成位置（自车正前方 {ahead_distance} 米）: ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f}), 朝向: {spawn_transform.rotation.yaw:.2f}°')
                
                # 生成新车辆
                try:
                    # 每次从蓝图库获取新蓝图（避免属性冲突）
                    blueprint = blueprint_library.find(blueprint_template.id)
                    if blueprint is None:
                        logging.error(f'未找到蓝图: {blueprint_template.id}')
                        continue
                    
                    if blueprint.has_attribute('color'):
                        color = blueprint.get_attribute('color').recommended_values[0]
                        blueprint.set_attribute('color', color)
                    blueprint.set_attribute('role_name', 'scenario_vehicle')
                    
                    print(f'正在尝试在位置 ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f}, {spawn_transform.location.z:.2f}) 生成车辆...')
                    vehicle = world.spawn_actor(blueprint, spawn_transform)
                    
                    if vehicle is None:
                        logging.error('生成车辆失败：返回None')
                        continue
                    
                    print(f'✓ 环境车辆已生成，ID: {vehicle.id}')
                    
                    # 不使用自动驾驶，改为手动控制车辆向前行驶
                    vehicle.set_autopilot(False, tm_port)
                    
                    # 直接使用生成位置作为起始位置（这是最准确的位置）
                    start_location = spawn_transform.location
                    traveled_distance = 0.0
                    stopping = False
                    
                    # 等待一帧，确保车辆完全生成并初始化
                    world.wait_for_tick()
                    
                    # 验证车辆实际位置（用于调试）
                    actual_location = vehicle.get_transform().location
                    location_diff = start_location.distance(actual_location)
                    if location_diff > 1.0:  # 如果位置差异超过1米，给出警告
                        logging.warning(f'车辆实际位置与生成位置差异较大: {location_diff:.2f}米')
                        logging.warning(f'生成位置: ({start_location.x:.2f}, {start_location.y:.2f})')
                        logging.warning(f'实际位置: ({actual_location.x:.2f}, {actual_location.y:.2f})')
                    
                    # 立即开始控制车辆向前行驶
                    control = carla.VehicleControl()
                    control.throttle = 0.7  # 初始较大油门，确保车辆开始移动
                    control.brake = 0.0
                    control.steer = 0.0
                    vehicle.apply_control(control)
                    
                    print(f'✓ 车辆已启动，将手动控制减速行驶 {travel_distance} 米后停车')
                    print(f'  起始位置（生成位置）: ({start_location.x:.2f}, {start_location.y:.2f})')
                    print(f'  车辆实际位置: ({actual_location.x:.2f}, {actual_location.y:.2f})')
                    
                except Exception as e:
                    logging.error(f'生成环境车辆失败: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    vehicle = None
            
            # 如果环境车辆存在，手动控制其行为
            if vehicle is not None and vehicle.is_alive and start_location is not None:
                current_location = vehicle.get_transform().location
                traveled_distance = start_location.distance(current_location)
                
                
                # 获取车辆当前transform和速度
                vehicle_transform = vehicle.get_transform()
                vehicle_velocity = vehicle.get_velocity()
                current_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # m/s
                
                # 如果行驶距离超过设定值，开始减速停车
                if traveled_distance >= travel_distance and not stopping:
                    stopping = True
                    print(f'车辆已行驶 {traveled_distance:.2f} 米，开始减速停车...')
                
                # 控制车辆
                control = carla.VehicleControl()
                
                if stopping:
                    # 停车阶段：应用刹车
                    if current_speed < 0.1:  # 速度小于0.1 m/s，认为已停止
                        control.throttle = 0.0
                        control.brake = 1.0  # 完全刹车
                        control.hand_brake = True  # 拉手刹
                        control.steer = 0.0
                    else:
                        # 还在减速中
                        control.throttle = 0.0
                        control.brake = 0.8
                        control.steer = 0.0
                else:
                    # 行驶阶段：向前行驶
                    remaining_distance = travel_distance - traveled_distance
                    
                    if remaining_distance > 0:
                        # 根据剩余距离调整速度（减速行驶：越接近目标，速度越慢）
                        # 使用简单的线性减速
                        speed_factor = remaining_distance / travel_distance
                        
                        # 目标速度：最大约20 km/h，随剩余距离减少而降低
                        max_speed_ms = 20.0 / 3.6  # 约5.56 m/s
                        target_speed_ms = max_speed_ms * speed_factor
                        
                        # 根据当前速度和目标速度调整控制
                        if current_speed < target_speed_ms - 0.5:
                            # 需要加速（当前速度明显低于目标速度）
                            control.throttle = 0.7  # 较大油门加速
                            control.brake = 0.0
                        elif current_speed > target_speed_ms + 0.5:
                            # 需要减速（当前速度明显高于目标速度）
                            control.throttle = 0.0
                            control.brake = 0.3  # 轻微刹车
                        else:
                            # 速度接近目标，保持当前速度
                            if current_speed < 0.5:
                                # 如果速度很低，给一点油门
                                control.throttle = 0.4
                                control.brake = 0.0
                            else:
                                # 保持速度
                                control.throttle = 0.3  # 小油门维持速度
                                control.brake = 0.0
                        
                        # 保持直线行驶
                        control.steer = 0.0
                    else:
                        # 已经到达目标距离，开始停车
                        control.throttle = 0.0
                        control.brake = 0.8
                        control.steer = 0.0
                
                # 应用控制命令（每帧都应用，确保控制持续）
                vehicle.apply_control(control)
                
                # 每20帧打印一次状态（用于调试，避免输出过多）
                frame_count = getattr(vehicle, '_frame_count', 0)
                vehicle._frame_count = frame_count + 1
                if frame_count % 20 == 0:  # 每20帧打印一次
                    print(f'[调试] 车辆状态: 已行驶 {traveled_distance:.2f}米 / {travel_distance}米, 速度: {current_speed*3.6:.2f} km/h, 剩余: {max(0, travel_distance - traveled_distance):.2f}米, 油门: {control.throttle:.2f}, 刹车: {control.brake:.2f}')
            
            # 更新自车位置记录（用于检测下次位置跳跃）
            if ego_vehicle is not None:
                last_ego_location = ego_location
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print('\n正在停止场景...')
    except Exception as e:
        logging.error(f'场景执行出错: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if vehicle is not None and vehicle.is_alive:
            logging.info('正在销毁环境车辆...')
            vehicle.destroy()
        
        # 恢复同步模式设置
        if traffic_manager is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        print('\n场景2：跟车场景已清理完成')


def main():
    parser = argparse.ArgumentParser(description='场景2：跟车场景')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--tm-port', type=int, default=8002, help='Traffic Manager端口')
    parser.add_argument('--filterv', default='vehicle.*', help='车辆类型过滤')
    parser.add_argument('--ahead-distance', type=float, default=10.0,
                       help='在自车前方的距离（米），默认10米')
    parser.add_argument('--travel-distance', type=float, default=15.0,
                       help='车辆行驶距离（米），默认15米（减速行驶后停车）')
    parser.add_argument('--role-name', default='ego_vehicle',
                       help='自车的role_name属性，默认ego_vehicle')
    parser.add_argument('--max-wait-time', type=int, default=30,
                       help='最大等待自车出现的时间（秒），默认30秒')
    parser.add_argument('--position-jump-threshold', type=float, default=20.0,
                       help='位置跳跃阈值（米），超过此值认为是新起点，默认20米')
    
    args = parser.parse_args()
    
    create_following_scenario(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        vehicle_filter=args.filterv,
        ahead_distance=args.ahead_distance,
        travel_distance=args.travel_distance,
        role_name=args.role_name,
        max_wait_time=args.max_wait_time,
        position_jump_threshold=args.position_jump_threshold
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
