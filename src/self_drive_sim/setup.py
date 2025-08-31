import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'self_drive_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'agent'), glob('agent/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='ms.kang@elicer.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f"actor_collision = {package_name}.actor_collision:main",
            f"train = {package_name}.train:main",
            f"test = {package_name}.test:main",
            f"debug = {package_name}.debug:main",
        ],
    },
)
