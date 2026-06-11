from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json') if os.path.exists('config') else []),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.osm') + glob('maps/*.pcd') if os.path.exists('maps') else []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Vehicle control package for CARLA simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_base = controller.control_base:main',
            'ego_state_plotter = controller.ego_state_plotter:main',
            'odom_rect_marker = controller.odom_rect_marker:main',
            'remap_goal = controller.remap_goal:main',
        ],
    },
)
