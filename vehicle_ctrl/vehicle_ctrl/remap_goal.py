import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class GoalPoseRepublisher(Node):
    def __init__(self, role_name):
        super().__init__('goal_pose_republisher')  # 修改1: 更清晰的节点名
        self.role_name = role_name
        
        # 修改2: 添加调试日志显示角色名
        self.get_logger().info(f'Initializing GoalPoseRepublisher for role: {self.role_name}')
        
        # Subscribe to the /goal_pose topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )

        # Publisher to /carla/<ROLE_NAME>/goal
        self.publisher = self.create_publisher(
            PoseStamped,
            f'/carla/{self.role_name}/goal',
            10
        )
        
        # 修改3: 使用ROS2日志系统替代print
        self.get_logger().info("Goal pose re-mapping service started successfully...")
        
    def goal_pose_callback(self, msg):
        # 修改4: 更详细的日志记录，包含坐标信息
        self.get_logger().info(
            f'Received goal pose - X: {msg.pose.position.x:.2f}, '
            f'Y: {msg.pose.position.y:.2f}, '
            f'Z: {msg.pose.position.z:.2f}'
        )

        # Create a new PoseStamped message to publish
        new_msg = PoseStamped()

        # 修改5: 保留原始时间戳而不是覆盖为当前时间
        new_msg.header.stamp = msg.header.stamp  # 使用原始消息的时间戳
        new_msg.header.frame_id = 'map'  # 明确指定坐标系

        # 修改6: 添加高度验证和调整逻辑
        target_z = 0.2  # CARLA中合适的地面高度
        
        # 如果原始Z坐标不合理，使用默认值
        if msg.pose.position.z < -100 or msg.pose.position.z > 100:
            self.get_logger().warn(f'Unrealistic Z coordinate ({msg.pose.position.z}), using default: {target_z}')
            new_msg.pose.position.z = target_z
        else:
            new_msg.pose.position.z = msg.pose.position.z

        # 修改7: 复制所有位置和方向数据
        new_msg.pose.position.x = msg.pose.position.x
        new_msg.pose.position.y = msg.pose.position.y
        
        # 修改8: 验证方向数据，如果无效则使用默认值
        if (msg.pose.orientation.x == 0 and msg.pose.orientation.y == 0 and 
            msg.pose.orientation.z == 0 and msg.pose.orientation.w == 0):
            self.get_logger().warn('Received invalid orientation, using default orientation')
            new_msg.pose.orientation.w = 1.0  # 默认无旋转
        else:
            new_msg.pose.orientation = msg.pose.orientation

        # 修改9: 添加发布前的最终验证
        try:
            self.publisher.publish(new_msg)
            self.get_logger().info(
                f'Successfully published goal pose to /carla/{self.role_name}/goal - '
                f'X: {new_msg.pose.position.x:.2f}, Y: {new_msg.pose.position.y:.2f}, '
                f'Z: {new_msg.pose.position.z:.2f}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to publish goal pose: {str(e)}')

def main(args=None):
    rclpy.init(args=args)

    # 修改10: 添加参数验证和错误处理
    role_name = 'ego_vehicle'
    
    if not role_name or role_name.strip() == "":
        print("Error: Role name cannot be empty!")
        return

    try:
        goal_pose_republisher = GoalPoseRepublisher(role_name)
        rclpy.spin(goal_pose_republisher)
    except KeyboardInterrupt:
        # 修改11: 添加优雅退出的日志
        goal_pose_republisher.get_logger().info('Goal pose republisher shutting down due to keyboard interrupt...')
    except Exception as e:
        # 修改12: 添加异常处理
        print(f"Failed to initialize GoalPoseRepublisher: {str(e)}")
    finally:
        if 'goal_pose_republisher' in locals():
            goal_pose_republisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()