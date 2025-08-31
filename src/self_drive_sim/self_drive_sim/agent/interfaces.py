import numpy as np
from typing import Tuple, TypedDict
from self_drive_sim.simulation.floor_map import MapInfo

# Type hints
class Observation(TypedDict):
    sensor_lidar_front: np.ndarray
    sensor_lidar_back: np.ndarray
    sensor_tof_left: np.ndarray
    sensor_tof_right: np.ndarray
    sensor_camera: np.ndarray
    sensor_ray: float
    sensor_pollution: float
    air_sensor_pollution: np.ndarray
    disp_position: Tuple[float, float]
    disp_angle: float

class Info(TypedDict):
    robot_position: Tuple[float, float]
    robot_angle: float
    collided: bool
    all_pollution: np.ndarray