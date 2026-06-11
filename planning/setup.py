from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.osm')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Map display and path planning controller using lanelet2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_path_planner_node = modified_EM_planner.global_path_planner:main',
            'ref_line_smoother_node = modified_EM_planner.ref_line_smoother:main',
            'planning_base = modified_EM_planner.planning_base:main',
            'speed_planner = modified_EM_planner.speed_planner:main',
        ],
    },
)
