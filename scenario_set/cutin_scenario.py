#!/usr/bin/env python3
"""
场景3：插车场景
逻辑与 /home/eowyn/ws_lmy/ws_ads/src/carla_scenarios/carla_scenarios/cutin_scenario.py 对齐；
本脚本额外提供命令行入口与 get_trafficmanager bind 错误提示。
"""

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
        if vehicle.attributes.get('role_name') == role_name:
            return vehicle
    return None


def calculate_position_side_behind(transform, side_distance, behind_distance, left=True):
    """
    计算侧方后方位置

    Args:
        transform: 当前位置和朝向
        side_distance: 侧方距离（米）
        behind_distance: 后方距离（米）
        left: True表示左侧，False表示右侧

    Returns:
        new_transform: 侧方后方位置的transform
    """
    location = transform.location
    rotation = transform.rotation

    yaw_rad = math.radians(rotation.yaw)
    forward_vector = carla.Vector3D(
        math.cos(yaw_rad),
        math.sin(yaw_rad),
        0.0
    )

    if left:
        side_yaw_rad = yaw_rad + math.pi / 2
    else:
        side_yaw_rad = yaw_rad - math.pi / 2
    side_vector = carla.Vector3D(
        math.cos(side_yaw_rad),
        math.sin(side_yaw_rad),
        0.0
    )

    new_location = location - forward_vector * behind_distance + side_vector * side_distance
    new_transform = carla.Transform(new_location, rotation)
    return new_transform


def _ego_horizontal_speed_mps(ego_vehicle):
    v = ego_vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y)


def create_cutin_scenario(host='localhost', port=2000, tm_port=8000,
                          vehicle_filter='vehicle.*',
                          side_distance=4.0, behind_distance=10.0,
                          cutin_distance=25.0, cruise_distance=40.0,
                          left_side=True,
                          role_name='ego_vehicle', max_wait_time=30,
                          position_jump_threshold=20.0,
                          min_ego_speed_mps=0.5):
    """
    与 ws_lmy carla_scenarios.cutin_scenario.create_cutin_scenario 一致（参数、主循环、切入/巡航）。

    Args:
        host: CARLA服务器地址
        port: CARLA服务器端口
        tm_port: Traffic Manager端口
        vehicle_filter: 车辆类型过滤
        side_distance: 在自车侧方的距离（米），默认4米
        behind_distance: 在自车后方的距离（米），默认10米
        cutin_distance: 切入距离（米），车辆行驶这个距离后开始切入，默认25米
        cruise_distance: 切入后匀速行驶距离（米），行驶这个距离后停止，默认40米
        left_side: True表示左侧，False表示右侧，默认True（左侧）
        role_name: 自车的role_name属性
        max_wait_time: 保留参数（与 ws 一致，当前未在主循环使用）
        position_jump_threshold: 位置跳跃阈值（米），超过此值认为是新起点，默认20米
        min_ego_speed_mps: 自车水平速度达到 m/s 后才生成环境车，默认0.5
    """
    _ = max_wait_time  # 与 ws_lmy 声明一致
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    vehicle = None
    traffic_manager = None
    last_ego_location = None
    blueprint = None

    try:
        world = client.get_world()
        try:
            traffic_manager = client.get_trafficmanager(tm_port)
        except RuntimeError as e:
            if 'bind' in str(e).lower():
                print(
                    '\nTraffic Manager 初始化失败（bind error）：'
                    'tm_port=%s 本机监听端口已被占用（勿与 http.server 等共用 8000）。\n'
                    '排查：ss -tlnp | grep :%s\n' % (tm_port, tm_port),
                    flush=True,
                )
            raise
        traffic_manager.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter(vehicle_filter)
        vehicle_blueprints = [bp for bp in vehicle_blueprints if bp.get_attribute('number_of_wheels').as_int() == 4]

        if not vehicle_blueprints:
            logging.error(f'未找到匹配的车辆蓝图: {vehicle_filter}')
            return

        blueprint_template = vehicle_blueprints[0]

        side_str = "左侧" if left_side else "右侧"
        print(f'场景3：插车场景已启动（逻辑同 ws_lmy carla_scenarios）')
        print(f'将在自车{side_str} {side_distance} 米、后方 {behind_distance} 米生成环境车辆')
        print(f'环境车辆行驶 {cutin_distance} 米后切入自车前方，之后匀速行驶 {cruise_distance} 米后停止')
        print(f'当自车位置跳跃超过 {position_jump_threshold} 米时将重置并等待再次起步（支持RViz 2D Pose Estimate）')
        print(f'自车水平速度 ≥ {min_ego_speed_mps} m/s 时才生成环境车辆')
        print(f'正在查找自车（role_name="{role_name}"）...')
        print('按 Ctrl+C 退出...')

        start_location = None
        traveled_distance = 0.0
        cutting_in = False
        cutin_complete = False
        cutin_start_location = None
        cruise_distance_traveled = 0.0
        first_check = True
        ego_seen_logged = False
        waiting_speed_logged = False

        while True:
            world.wait_for_tick()

            ego_vehicle = find_ego_vehicle(world, role_name)

            if ego_vehicle is None:
                if vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()
                    vehicle = None
                last_ego_location = None
                cutting_in = False
                cutin_complete = False
                cutin_start_location = None
                cruise_distance_traveled = 0.0
                ego_seen_logged = False

                if first_check:
                    vehicles = world.get_actors().filter('vehicle.*')
                    print("当前所有车辆:")
                    for v in vehicles:
                        role_name_v = v.attributes.get('role_name', 'None')
                        print(f"  Vehicle ID: {v.id}, role_name: '{role_name_v}'")
                    print(f'\n提示：如果找不到自车，请检查Terminal 3生成的自车的role_name是否为"{role_name}"')
                    print('如果role_name不同，请使用 --role-name 参数指定正确的role_name')
                    first_check = False

                time.sleep(1)
                continue

            ego_location = ego_vehicle.get_transform().location
            ego_speed = _ego_horizontal_speed_mps(ego_vehicle)

            if last_ego_location is not None:
                jump_dist = last_ego_location.distance(ego_location)
                if jump_dist >= position_jump_threshold:
                    print(
                        f'检测到自车位置大幅跳跃 {jump_dist:.2f} 米（可能是2D Pose Estimate），'
                        '已移除环境车，等待自车速度达标后再生成...')
                    if vehicle is not None and vehicle.is_alive:
                        vehicle.destroy()
                        vehicle = None
                    cutting_in = False
                    cutin_complete = False
                    cutin_start_location = None
                    start_location = None
                    traveled_distance = 0.0
                    cruise_distance_traveled = 0.0
                    waiting_speed_logged = False

            if not ego_seen_logged:
                print(f'找到自车，ID: {ego_vehicle.id}')
                ego_seen_logged = True

            no_env_vehicle = vehicle is None or not vehicle.is_alive
            need_generate = no_env_vehicle and ego_speed >= min_ego_speed_mps

            if no_env_vehicle and ego_speed < min_ego_speed_mps and not waiting_speed_logged:
                print(
                    f'等待自车起步（当前水平速度 {ego_speed:.2f} m/s，'
                    f'需 ≥ {min_ego_speed_mps} m/s 后生成环境车）...')
                waiting_speed_logged = True

            if need_generate:
                waiting_speed_logged = False

            if need_generate:
                if vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()
                    vehicle = None

                last_ego_location = ego_location

                ego_transform = ego_vehicle.get_transform()
                print(f'自车位置: ({ego_transform.location.x:.2f}, {ego_transform.location.y:.2f})')

                spawn_transform = calculate_position_side_behind(
                    ego_transform, side_distance, behind_distance, left=left_side)
                side_str = "左侧" if left_side else "右侧"
                print(f'环境车辆生成位置（自车{side_str} {side_distance}米、后方 {behind_distance}米）: ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f})')

                try:
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

                    vehicle.set_autopilot(True, tm_port)
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, -10.0)

                    start_location = vehicle.get_transform().location
                    traveled_distance = 0.0
                    cutting_in = False
                    cutin_complete = False
                    cutin_start_location = None
                    cruise_distance_traveled = 0.0
                    print(f'✓ 车辆已启动自动驾驶，将在行驶 {cutin_distance} 米后切入自车前方')

                except Exception as e:
                    logging.error(f'生成环境车辆失败: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    vehicle = None

            if vehicle is not None and vehicle.is_alive and ego_vehicle is not None and ego_vehicle.is_alive and start_location is not None:
                current_location = vehicle.get_transform().location
                traveled_distance = start_location.distance(current_location)

                ego_current_transform = ego_vehicle.get_transform()

                yaw_rad = math.radians(ego_current_transform.rotation.yaw)
                forward_vector = carla.Vector3D(
                    math.cos(yaw_rad),
                    math.sin(yaw_rad),
                    0.0
                )

                if traveled_distance >= cutin_distance and not cutting_in:
                    cutting_in = True
                    print(f'车辆已行驶 {traveled_distance:.2f} 米，开始切入自车前方...')
                    vehicle.set_autopilot(False, tm_port)

                if cutting_in and not cutin_complete:
                    target_location = ego_current_transform.location + forward_vector * 20.0

                    direction = target_location - current_location
                    distance = math.sqrt(direction.x**2 + direction.y**2)

                    if distance > 1.0:
                        target_yaw = math.degrees(math.atan2(direction.y, direction.x))
                        current_yaw = vehicle.get_transform().rotation.yaw
                        angle_diff = target_yaw - current_yaw
                        while angle_diff > 180:
                            angle_diff -= 360
                        while angle_diff < -180:
                            angle_diff += 360

                        control = carla.VehicleControl()
                        control.throttle = 0.6
                        control.brake = 0.0
                        control.steer = max(-1.0, min(1.0, angle_diff / 30.0))
                        vehicle.apply_control(control)
                    else:
                        if not cutin_complete:
                            cutin_complete = True
                            cutin_start_location = current_location
                            cruise_distance_traveled = 0.0
                            print(f'车辆已切入自车前方，开始匀速行驶，将在行驶 {cruise_distance} 米后停止...')

                if cutin_complete:
                    if cutin_start_location is not None:
                        cruise_distance_traveled = cutin_start_location.distance(current_location)

                    if cruise_distance_traveled < cruise_distance:
                        vehicle_yaw = vehicle.get_transform().rotation.yaw
                        ego_yaw = ego_current_transform.rotation.yaw
                        angle_diff = ego_yaw - vehicle_yaw
                        while angle_diff > 180:
                            angle_diff -= 360
                        while angle_diff < -180:
                            angle_diff += 360

                        control = carla.VehicleControl()
                        control.throttle = 0.5
                        control.brake = 0.0
                        control.steer = max(-1.0, min(1.0, angle_diff / 30.0))
                        vehicle.apply_control(control)
                    else:
                        control = carla.VehicleControl()
                        control.throttle = 0.0
                        control.brake = 0.8
                        control.steer = 0.0
                        vehicle.apply_control(control)

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
        if vehicle is not None and vehicle.is_alive:
            logging.info('正在销毁环境车辆...')
            vehicle.destroy()

        if traffic_manager is not None:
            try:
                wfin = client.get_world()
                st = wfin.get_settings()
                st.synchronous_mode = False
                wfin.apply_settings(st)
            except Exception:
                pass

        print('\n场景3：插车场景已清理完成')


def main():
    parser = argparse.ArgumentParser(description='场景3：插车场景（ws_lmy carla_scenarios 逻辑）')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口')
    parser.add_argument('--tm-port', type=int, default=8000, help='Traffic Manager 端口')
    parser.add_argument('--filterv', default='vehicle.*', help='车辆类型过滤')
    parser.add_argument('--side-distance', type=float, default=4.0, help='侧向距离(m)，默认4')
    parser.add_argument('--behind-distance', type=float, default=10.0, help='后方距离(m)，默认10')
    parser.add_argument('--cutin-distance', type=float, default=25.0, help='行驶该距离后切入')
    parser.add_argument('--cruise-distance', type=float, default=40.0, help='切入后匀速距离')
    parser.add_argument('--right-side', action='store_true',
                        help='在自车右侧生成（默认左侧，与 ws_lmy 一致）')
    parser.add_argument('--role-name', default='ego_vehicle', help='自车 role_name')
    parser.add_argument('--max-wait-time', type=int, default=30, help='保留，与 ws API 一致')
    parser.add_argument('--position-jump-threshold', type=float, default=20.0, help='位置跳跃阈值(m)')
    parser.add_argument('--min-ego-speed', type=float, default=0.5,
                        help='自车水平速度≥该值(m/s)才生成环境车，与 ws min_ego_speed_mps 一致')

    args = parser.parse_args()

    create_cutin_scenario(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        vehicle_filter=args.filterv,
        side_distance=args.side_distance,
        behind_distance=args.behind_distance,
        cutin_distance=args.cutin_distance,
        cruise_distance=args.cruise_distance,
        left_side=not args.right_side,
        role_name=args.role_name,
        max_wait_time=args.max_wait_time,
        position_jump_threshold=args.position_jump_threshold,
        min_ego_speed_mps=args.min_ego_speed,
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
