import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
import math
import time

class RvizGoalPublisher(Node):
    def __init__(self):
        super().__init__('rviz_goal_publisher')
        # 发布到 RViz2 2D Nav Goal 默认监听的话题
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
    def set_goal(self, x, y, z=0.0, yaw=0.0):
        """设置目标位置和方向"""
        goal_msg = PoseStamped()
        
        # 设置消息头
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        
        # 设置位置
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        
        # 设置方向
        goal_msg.pose.orientation = self.yaw_to_quaternion(yaw)
        
        # 发布目标
        self.goal_publisher.publish(goal_msg)
        print(f"✅ 已发布目标点到 /goal_pose 话题")
        print("请查看 RViz2 终端窗口，应该会显示类似:")
        print("[INFO] [rviz2]: Setting goal pose: Frame:map, Position(x, y, z), Orientation(...)")
        
    def yaw_to_quaternion(self, yaw):
        """将偏航角转换为四元数"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main():
    rclpy.init()
    goal_publisher = RvizGoalPublisher()
    
    print("🎯 RViz2 目标点发布器")
    print("=" * 50)
    print("注意: 确保 RViz2 正在运行并监听 /goal_pose 话题")
    print("RViz2 会在自己的终端输出目标点设置信息")
    print("=" * 50)
    
    try:
        # 获取用户输入
        x = float(input("请输入终点 X 坐标: "))
        y = float(input("请输入终点 Y 坐标: "))
        
        z_input = input("请输入终点 Z 坐标 (默认 0.0): ").strip()
        z = float(z_input) if z_input else 0.0
        
        yaw_input = input("请输入终点朝向角度 (弧度, 默认 0.0): ").strip()
        yaw = float(yaw_input) if yaw_input else 0.0
        
        # 设置并发布终点
        goal_publisher.set_goal(x, y, z, yaw)
        
        # 短暂保持节点运行确保消息发送
        print("等待消息发送...")
        time.sleep(0.5)
        rclpy.spin_once(goal_publisher, timeout_sec=1.0)
        
    except ValueError:
        print("❌ 错误: 请输入有效的数字")
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    
    finally:
        # 清理资源
        goal_publisher.destroy_node()
        rclpy.shutdown()
        print("程序已退出")

if __name__ == '__main__':
    main()