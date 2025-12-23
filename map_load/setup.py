from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'map_load'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.osm')),
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
            'map_control_node = map_load.map_control:main',
        ],
    },
)

