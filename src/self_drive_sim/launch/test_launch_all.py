#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction, ExecuteProcess
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    test_map0_exec = ExecuteProcess(
        cmd=['ros2', 'launch', 'self_drive_sim', 'test_launch_map0.py'],
        output='screen',
    )

    test_map1_exec = ExecuteProcess(
        cmd=['ros2', 'launch', 'self_drive_sim', 'test_launch_map1.py'],
        output='screen',
    )

    test_map2_exec = ExecuteProcess(
        cmd=['ros2', 'launch', 'self_drive_sim', 'test_launch_map2.py'],
        output='screen',
    )

    test_map3_exec = ExecuteProcess(
        cmd=['ros2', 'launch', 'self_drive_sim', 'test_launch_map3.py'],
        output='screen',
    )

    shutdown_node_0 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=test_map0_exec,
            on_exit=[
                TimerAction(period=2.0, actions=[test_map1_exec])
            ],
        )
    )

    shutdown_node_1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=test_map1_exec,
            on_exit=[
                TimerAction(period=2.0, actions=[test_map2_exec])
            ],
        )
    )

    shutdown_node_2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=test_map2_exec,
            on_exit=[
                TimerAction(period=2.0, actions=[test_map3_exec])
            ],
        )
    )

    return LaunchDescription([
        test_map0_exec,
        shutdown_node_0,
        shutdown_node_1,
        shutdown_node_2,
    ])