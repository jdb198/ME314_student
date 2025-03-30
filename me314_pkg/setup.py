#!/usr/bin/env python3
import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'me314_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        # For ROS 2 to discover the package
        ('share/ament_index/resource_index/packages', ['resource/me314_pkg']),
        (os.path.join('share', package_name), ['package.xml']),
        # Optionally copy over anything in resource/
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
        # If you want your launch files discoverable via setup.py instead of CMake:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'xarm',                
        'numpy',
        'scipy',
        'xarm-python-sdk',
    ],
    zip_safe=True,
    maintainer='Alex Qiu',
    maintainer_email='aqiu34@stanford.edu',
    description='A ROS2 package integrating the XArm ROS2 library and the XArm Python SDK for ME 314 Robotic Dexterity.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'xarm_commander = me314_pkg.me314_py.xarm_planner.xarm_commander:main',
        ],
    },
)
