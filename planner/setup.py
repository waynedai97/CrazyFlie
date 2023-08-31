import os
from glob import glob
from setuptools import setup

package_name = 'planner_env'
submodules = 'planner_env/modules'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, submodules],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/config.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jimmy',
    maintainer_email='todo@todo.org',
    description='This package acts as a bridge between the path planner and crazyflie drones',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_env = planner_env.planner_env:main'
        ],
    },
)
