from launch import LaunchDescription
from launch_ros.actions import Node

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    octo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cf3d_nav'), 'launch', 'octomap_mapping.launch.py')
        )
    )


    explore3d = Node(package='cf3d_nav',
            executable='explorer3d',
            name='explorer3d',
            output='screen',
            parameters=[
                {'cmd_vel_topic': '/auto_cmd_vel'},
                {'odom_topic': '/crazyflie_real/odom'},
                {'markers_topic': '/occupied_cells_vis_array'},
            ],)
    return LaunchDescription([octo, explore3d])
