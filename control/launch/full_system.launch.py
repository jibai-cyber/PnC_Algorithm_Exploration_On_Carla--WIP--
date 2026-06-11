from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control',
            executable='control_base',
            name='control_base',
            output='screen',
        ),
        Node(
            package='planning',
            executable='ref_line_smoother_node',
            name='ref_line_smoother_node',
            output='screen',
        ),
        Node(
            package='planning',
            executable='planning_base',
            name='planning_base',
            output='screen',
        ),
        Node(
            package='planning',
            executable='speed_planner',
            name='speed_planner',
            output='screen',
        ),
    ])
