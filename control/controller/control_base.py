#!/usr/bin/env python3
"""车辆控制主入口文件"""

import rclpy
from .carla_vehicle_control import CarlaVehicleControl


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
