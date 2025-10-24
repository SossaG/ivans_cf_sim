# cf3d_nav_full.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    # -------- Launch-time arguments (override from parent/master) --------
    world_frame_id_arg = DeclareLaunchArgument('world_frame_id', default_value='map')
    body_frame_id_arg  = DeclareLaunchArgument('body_frame_id',  default_value='crazyflie/odom')
    odom_topic_arg     = DeclareLaunchArgument('odom_topic',     default_value='/crazyflie/odom')
    scan_topic_arg     = DeclareLaunchArgument('scan_topic',     default_value='/crazyflie/scan')
    cloud_in_arg       = DeclareLaunchArgument('cloud_in',       default_value='/crazyflie/pointcloud')
    markers_topic_arg  = DeclareLaunchArgument('markers_topic',  default_value='/occupied_cells_vis_array')
    cmd_vel_topic_arg  = DeclareLaunchArgument('cmd_vel_topic',  default_value='/auto_cmd_vel')

    world_frame_id = LaunchConfiguration('world_frame_id')
    body_frame_id  = LaunchConfiguration('body_frame_id')
    odom_topic     = LaunchConfiguration('odom_topic')
    scan_topic     = LaunchConfiguration('scan_topic')
    cloud_in       = LaunchConfiguration('cloud_in')
    markers_topic  = LaunchConfiguration('markers_topic')
    cmd_vel_topic  = LaunchConfiguration('cmd_vel_topic')

    # -------- Include OctoMap (expects its launch to declare world_frame_id + cloud_in) --------
    octomap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('cf3d_nav'), 'launch', 'octomap_mapping.launch.py')
        ),
        launch_arguments={
            'world_frame_id': world_frame_id,
            'cloud_in':       cloud_in,
        }.items()
    )

    # -------- Explorer node (now fully configurable from above) --------
    explorer3d = Node(
        package='cf3d_nav',
        executable='explorer3d',
        name='explorer3d',
        output='screen',
        parameters=[
            # navigation topics
            {'cmd_vel_topic': cmd_vel_topic},
            {'odom_topic':    odom_topic},
            {'markers_topic': markers_topic},

            # frames (expose in case your node uses them internally)
            {'world_frame_id': world_frame_id},
            {'body_frame_id':  body_frame_id},

            # sensors (expose scan if your node subscribes to it)
            {'scan_topic':     scan_topic},
        ]
    )

    return LaunchDescription([
        world_frame_id_arg, body_frame_id_arg,
        odom_topic_arg, scan_topic_arg, cloud_in_arg,
        markers_topic_arg, cmd_vel_topic_arg,
        octomap_launch,
        explorer3d,
    ])
