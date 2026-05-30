import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg   = get_package_share_directory('formation_control')
    tb3   = get_package_share_directory('turtlebot3_gazebo')
    world = os.path.join(pkg, 'worlds', 'formation_world.sdf')
    model = os.path.join(tb3, 'models', 'turtlebot3_burger', 'model.sdf')

    gz_server = ExecuteProcess(
        cmd=['gzserver', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world],
        output='screen'
    )
    gz_client = ExecuteProcess(cmd=['gzclient'], output='screen')

    def spawn(name, x, y):
        return TimerAction(period=6.0, actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', name, '-file', model,
                    '-x', str(x), '-y', str(y), '-z', '0.01',
                    '-robot_namespace', name
                ],
                output='screen'
            )
        ])

    def robot_node(exe, ns):
        return TimerAction(period=12.0, actions=[
            Node(
                package='formation_control',
                executable=exe,
                namespace=ns,
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ])

    def global_node(exe):
        return TimerAction(period=14.0, actions=[
            Node(
                package='formation_control',
                executable=exe,
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ])

    def consensus_node(ns):
        return TimerAction(period=14.0, actions=[
            Node(
                package='formation_control',
                executable='consensus_node.py',
                namespace=ns,
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ])

    return LaunchDescription([
        gz_server,
        gz_client,

        # Spawn robots at fixed start positions
        spawn('tb3_0',  0.0,  0.0),
        spawn('tb3_1', -0.55,  0.32),
        spawn('tb3_2', -0.55, -0.32),

        # Formation nodes
        robot_node('leader_explorer.py',    'tb3_0'),
        robot_node('follower_formation.py', 'tb3_1'),
        robot_node('follower_formation.py', 'tb3_2'),

        # Consensus node — one per robot
        consensus_node('tb3_0'),
        consensus_node('tb3_1'),
        consensus_node('tb3_2'),

        # Topology node — one global node
        #global_node('topology_node.py'),

        # Voronoi node — one global node
        #global_node('voronoi_coverage.py'),
    ])