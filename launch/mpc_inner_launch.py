#!/usr/bin/env python
############################################################################
# mpc_inner_launch.py
# Launch file for the quaternion MPC inner-loop controller on PX4 x500.
#
# Usage (SITL):
#   ros2 launch px4_mpc mpc_inner_launch.py \
#       trajectory_path:=/path/to/trajectory.npz
#
# Usage (hover, no trajectory file needed):
#   ros2 launch px4_mpc mpc_inner_launch.py
############################################################################

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
        default_value='',
        description='Path to a pre-generated trajectory .npz (~ is expanded). '
                    'Empty: static hover at (0, 0, 3). A missing file is an error.')

    rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='120.0',
        description='Control loop rate [Hz]; the MPC grid keeps a 0.4 s horizon with '
                    '20 stages (first stages = one period)')

    delay_arg = DeclareLaunchArgument(
        'actuation_delay',
        default_value='0.04',
        description='Delay [s] between state sample and command taking effect; the state '
                    'is propagated this far ahead before each solve (0 disables)')

    max_age_arg = DeclareLaunchArgument(
        'max_state_age',
        default_value='0.2',
        description='Max age [s] of PX4 state inputs; older inputs are propagated through '
                    'the commands sent since, beyond this the node stops commanding')

    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value='~/mpc_logs',
        description='Folder for per-step MPC run logs (empty string disables logging)')

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
        rate_arg,
        delay_arg,
        max_age_arg,
        log_dir_arg,
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
                {'control_rate':     ParameterValue(LaunchConfiguration('control_rate'), value_type=float)},
                {'actuation_delay':  ParameterValue(LaunchConfiguration('actuation_delay'), value_type=float)},
                {'max_state_age':    ParameterValue(LaunchConfiguration('max_state_age'), value_type=float)},
                {'log_dir':          ParameterValue(LaunchConfiguration('log_dir'), value_type=str)},
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