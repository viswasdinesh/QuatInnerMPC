import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'px4_mpc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # RViz config used by the launch file (optional)
        (os.path.join('share', package_name), glob('*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Dinesh',
    maintainer_email='viswasdinesh28@gmail.com',
    description='Quaternion inner-loop NMPC (acados) for PX4 multirotors.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'mpc_inner_quadrotor = px4_mpc.mpc_inner_quadrotor:main',
        ],
    },
)
