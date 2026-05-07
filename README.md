# Multi-Robot Formation Control with Collision Avoidance and Coverage
This project uses 3 TurtleBot3 robots in a Gazebo simulation. One robot acts as the leader and moves automatically through 5 fixed points, while the other 2 robots follow behind in a triangle shape. The robots can avoid walls, pillars, and each other using a safety system based on math calculations. They also communicate with each other to work as a team, cover the area properly, and maintain a stable network connection, all at the same time.

# System Requirements
1. Operating System: Ubuntu 22.04
2. ROS: ROS2 Humble
3. Simulator: Gazebo

# System Architecture
```
tb3_ws/
└── src/
    └── formation_control/
        │
        ├── formation_control/
        │   ├── __init__.py
        │   └── scripts/
        │       ├── __init__.py
        │       ├── leader_explorer.py
        │       ├── follower_formation.py
        │       ├── consensus_node.py
        │       ├── topology_node.py
        │       ├── voronoi_coverage.py
        │       └── pose_publisher.py
        ├── launch/
        │   └── formation_sim.launch.py
        ├── worlds/
        │   └── formation_world.sdf
        ├── package.xml
        └── setup.py
```
# Features
1. CBF + QP Collision Avoidance
2. LiDAR Obstacle Detection
3. Triangle Formation Control
4. Waypoint Navigation + Tangent Avoidance
5. Distributed Consensus
6. Topology Control
7. Voronoi Coverage

# Commands
### Build:
```
cd ~/tb3_ws && colcon build && source install/setup.bash
```
### Launch:
```
ros2 launch formation_control formation_sim.launch.py
```
