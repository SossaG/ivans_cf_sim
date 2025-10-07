from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            remappings=[('/cloud_in', '/crazyflie_real/pointcloud')],  # <-- your topic
            parameters=[
            {'frame_id': 'world'},
            {'resolution': 0.050},
            {'sensor_model/max_range': 3.5},
            {'publish_free_space': True},
            ]
        )
    ])
