from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vehicle_ctrl',
            executable='simple_ctrl',
            name='simple_ctrl',
            output='screen',
        ),
        Node(
            package='map_load',
            executable='path_smoother_node',
            name='path_smoother_node',
            output='screen',
        ),
        Node(
            package='map_load',
            executable='vehicle_perception',
            name='vehicle_perception',
            output='screen',
        ),
    ])

