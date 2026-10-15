#!/usr/bin/env python
############################################################################
# mpc_quadrotor_launch.py
# Launch file for the quaternion MPC inner-loop controller on PX4 x500.
#
# Usage (SITL):
#   ros2 launch px4_mpc mpc_quadrotor_launch.py \
#       trajectory_path:=/path/to/trajectory.npz
#
# Usage (hover, no trajectory file needed):
#   ros2 launch px4_mpc mpc_quadrotor_launch.py
############################################################################

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile


def generate_launch_description():

    # ── Launch arguments ───────────────────────────────────────────────
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='ROS 2 namespace for all nodes (useful for multi-vehicle)')

    traj_path_arg = DeclareLaunchArgument(
        'trajectory_path',
        default_value='/home/dinesh/trajectories/takeoff.npz',
        description='Absolute path to a pre-generated trajectory.npz file. '
                    'Leave empty to use the default static hover at z=3 m.')

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 for visualisation')

    namespace      = LaunchConfiguration('namespace')
    trajectory_path = LaunchConfiguration('trajectory_path')
    rviz_enabled   = LaunchConfiguration('rviz')

    return LaunchDescription([
        namespace_arg,
        traj_path_arg,
        rviz_arg,

        # ── MPC controller node ────────────────────────────────────────
        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='mpc_inner_quadrotor',       # entry_point in setup.py
            name='mpc_inner_quadrotor',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace':        namespace},
                {'trajectory_path':  trajectory_path},
            ]
        ),

        # ── Visualiser (px4_offboard package) ─────────────────────────
        Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer',
            parameters=[{'namespace': namespace}],
            condition=IfCondition(rviz_enabled)
        ),

        # ── RViz2 ─────────────────────────────────────────────────────
        OpaqueFunction(function=_launch_rviz),
    ])


# ──────────────────────────────────────────────────────────────────────────
def _patch_rviz_config(original_path: str, namespace: str) -> str:
    """Replace __NS__ placeholder in the RViz config with the actual namespace."""
    with open(original_path, 'r') as f:
        content = f.read()
    content = content.replace('__NS__', f'/{namespace}' if namespace else '')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.rviz')
    tmp.write(content.encode('utf-8'))
    tmp.close()
    return tmp.name


def _launch_rviz(context, *args, **kwargs):
    rviz_enabled = LaunchConfiguration('rviz').perform(context)
    if rviz_enabled.lower() not in ('true', '1', 'yes'):
        return []

    namespace    = LaunchConfiguration('namespace').perform(context)
    rviz_cfg     = os.path.join(
        get_package_share_directory('px4_mpc'), 'config.rviz')

    if not os.path.isfile(rviz_cfg):
        return []

    patched = _patch_rviz_config(rviz_cfg, namespace)
    return [
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', patched],
        )
    ]