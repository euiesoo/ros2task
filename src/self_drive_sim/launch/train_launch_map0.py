#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

WORLD_NAME = 'map0.world'
FLOORMAP_NAME = 'map0.npz'
POLLUTION_NAME = 'map0.json'
INIT_POS = (2.0, 0.0, 0.0, 1.5708) # x, y, z, Y
ACTORS = []

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('self_drive_sim')
    pkg_urdf = get_package_share_directory('namuhx-a1')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    xacro_file = os.path.join(pkg_urdf, 'urdf', 'a1.xacro')
    robot_desc = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    rviz_file = os.path.join(pkg_share_dir, 'launch', 'a1_single.rviz')

    world = os.path.join(pkg_share_dir, 'worlds', WORLD_NAME)
    floormap_file = os.path.join(pkg_share_dir, 'worlds', FLOORMAP_NAME)
    pollution_file = os.path.join(pkg_share_dir, 'worlds', POLLUTION_NAME)

    # Launch Descriptions
    gz_srv_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items(),
    )

    gz_cli_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
    )

    robot_name = "robot_0"

    robot_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=robot_name,
        parameters=[{
            'robot_description': robot_desc,
        }],
    )
    topic_name = f"/{robot_name}/robot_description"
    robot_spawn_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='urdf_spawner',
        namespace=robot_name,
        arguments=[
            '-topic', topic_name,
            '-entity', robot_name,
            '-x', str(INIT_POS[0]),
            '-y', str(INIT_POS[1]),
            '-z', str(INIT_POS[2]),
            '-Y', str(INIT_POS[3]),
            '-robot_namespace', robot_name,
            ],
        output='screen',
    )

    actor_collision_nodes = [
        Node(
            package='self_drive_sim',
            executable='actor_collision',
            namespace=actor['actor_name'],
            parameters=[actor],
            output='screen',
        ) for actor in ACTORS
    ]

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',  
        arguments=['-d', rviz_file],
        ros_arguments=['--log-level', 'WARN'],
        output='screen',
    )

    train_node = Node(
        package='self_drive_sim',
        executable='train',
        parameters=[{
            'robot_num': 1,
            'floormap_file': floormap_file,
            'pollution_file': pollution_file,
        }],
        output='screen',
    )

    return LaunchDescription([
        gz_srv_ld,
        gz_cli_ld,
        robot_pub_node,
        robot_spawn_node,
        *actor_collision_nodes,
        rviz_node,
        train_node,
    ])