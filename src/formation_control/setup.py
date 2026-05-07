from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'formation_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shchad',
    maintainer_email='shchad@todo.todo',
    description='Multi-robot formation with CBF, consensus, topology, Voronoi',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pose_publisher.py = formation_control.scripts.pose_publisher:main',
            'leader_explorer.py = formation_control.scripts.leader_explorer:main',
            'follower_formation.py = formation_control.scripts.follower_formation:main',
            'consensus_node.py = formation_control.scripts.consensus_node:main',
            'topology_node.py = formation_control.scripts.topology_node:main',
            'voronoi_coverage.py = formation_control.scripts.voronoi_coverage:main',
        ],
    },
)