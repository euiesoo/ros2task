import math
import random
import json
import time
from typing import List, Tuple

import rclpy
from rclpy.wait_for_message import wait_for_message
from rclpy.node import Node, Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import numpy as np
from cv_bridge import CvBridge

from self_drive_sim.simulation.floor_map import FloorMap
from self_drive_sim.simulation.pollution_manager import PollutionManager
from self_drive_sim.agent.interfaces import Observation, Info

SRV_WAIT_PERIOD = 1.0

DELTA_TIME = 0.1
PURIFY_STARTUP_TIME = 1.0 # 청정 동작을 시작한 후 1초가 지나야 실제로 청정이 시작됨
HARD_COLLISION_THR = 0.7 # 강한 충돌로 판정되는 로봇 속력

MAX_LIN_SPD = 1.0
MAX_ANG_SPD = 2.0

COLLISION_SENSOR_NUM = 4
COLLISION_SENSOR_THR = 0.25 # 충돌 판정 거리 (중심에서). 0.2 ~ 0.3
COLLISION_COOLDOWN = 5.0 # 한 번 충돌하면 N초 동안은 횟수 추가 x, N초 이상 충돌이 지속되면 별개의 충돌로 취급

STATION_RANGE = 0.5 # 도킹 스테이션 감지 거리 (시뮬레이션 종료)

class SensorData():
    def __init__(self):
        self.lidar_front = np.array([])
        self.lidar_back = np.array([])
        self.tof_left = np.array([])
        self.tof_right = np.array([])
        self.camera = np.array([])
        self.ray = 0.0

        self.bridge = CvBridge() # tof data decoder

    def callback_lidar_front(self, msg: LaserScan):
        self.lidar_front = np.array(msg.ranges)
    
    def callback_lidar_back(self, msg: LaserScan):
        self.lidar_back = np.array(msg.ranges)
    
    def callback_tof_left(self, msg: Image):
        cv2 = self.bridge.imgmsg_to_cv2(msg)
        self.tof_left = np.array(cv2)
    
    def callback_tof_right(self, msg: Image):
        cv2 = self.bridge.imgmsg_to_cv2(msg)
        self.tof_right = np.array(cv2)

    def callback_camera(self, msg: Image):
        cv2 = self.bridge.imgmsg_to_cv2(msg)
        self.camera = np.array(cv2)
    
    def callback_ray(self, msg: LaserScan):
        self.ray = msg.ranges[1] # pick middle one from 3

class Robot():
    robot_name: str
    purify_speed: float
    sensor_data: SensorData
    cmd_vel_pub: Publisher

    purify_duration: float
    pos: Tuple[float, float]
    vel: Tuple[float, float]
    angle: float
    last_pos: Tuple[float, float]
    last_angle: float

    disp_std_pos = 0.01
    disp_std_angle = 0.01

    time_since_last_collision: float
    weak_collision_cnt: int
    purify_amount: float

    terminated: bool
    done: bool
    done_time: float

    def __init__(self):
        self.purify_duration = 0.0
        self.pos = (0.0, 0.0)
        self.last_pos = (0.0, 0.0)
        self.angle = 0.0
        self.last_angle = 0.0

        self.collision_lidar_data = None
        self.time_since_last_collision = COLLISION_COOLDOWN
        self.weak_collision_cnt = 0
        self.purify_amount = 0.0

        self.terminated = False
        self.done = False
        self.done_time = 1800.0
    
    def reset(self):
        self.last_pos = (0.0, 0.0)
        self.last_angle = 0.0
        self.purify_duration = 0.0
        self.time_since_last_collision = COLLISION_COOLDOWN
        self.weak_collision_cnt = 0
        self.purify_amount = 0.0
        self.terminated = False
        self.done = False
        self.done_time = 1800.0

    def callback_odom(self, msg: Odometry):
        self.pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            )
        
        x, y, z, w = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        
        self.angle = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)) # yaw

        self.vel = (
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )
    
    def callback_collision_lidar(self, index: int, msg: LaserScan):
        data = np.array(msg.ranges)
        if self.collision_lidar_data is None:
            self.collision_lidar_data = np.zeros((
                COLLISION_SENSOR_NUM, data.size
            ))
        self.collision_lidar_data[index] = data
    
    def move(self, linear, angular):
        # send control command to robot (diff drive, twist control)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_vel_pub.publish(cmd)
    
    def detetct_collision(self):
        # detect collision every step
        # 0: no collision, 1: weak collision 2: strong collision
        if self.collision_lidar_data is None:
            return False
        collided = np.any(self.collision_lidar_data < COLLISION_SENSOR_THR)

        if collided:
            speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
            if speed > HARD_COLLISION_THR:
                # hard collision; terminate immediately
                self.time_since_last_collision = 0.0
                return 2
            else:
                if self.time_since_last_collision < COLLISION_COOLDOWN:
                    self.time_since_last_collision += DELTA_TIME
                else:
                    self.weak_collision_cnt += 1
                    self.time_since_last_collision = 0.0
                return 1
        else:
            return 0

    def get_disp_pos(self):
        # calculate position displacement since last get_disp
        # add noise and return
        dx = self.pos[0] - self.last_pos[0]
        dy = self.pos[1] - self.last_pos[1]

        dx *= random.gauss(1.0, self.disp_std_pos)
        dy *= random.gauss(1.0, self.disp_std_pos)

        self.last_pos = (self.pos[0], self.pos[1])

        return (dx, dy)
    
    def get_disp_angle(self):
        # calculate angle displacement since last get_disp
        # add noise and return
        dth = self.angle - self.last_angle
        dth *= random.gauss(1.0, self.disp_std_angle)
        dth = (dth + math.pi) % (2 * math.pi) - math.pi

        self.last_angle = self.angle

        return dth

class GazeboEnv(Node):
    def __init__(self, random_seed=None, debug_mode=False):
        super().__init__('env')
        self.declare_parameter('robot_num', 1)
        self.declare_parameter('floormap_file', "")
        self.declare_parameter('pollution_file', "")
        self.declare_parameter('result_file_name', "")
        self.result_file_name = self.get_parameter('result_file_name').get_parameter_value().string_value

        self.debug_mode = debug_mode

        if random_seed is None:
            self.random_seed = int(time.time() * 10)
        else:
            self.random_seed = random_seed
        random.seed(self.random_seed)

        self.steps = 0
        self.fm = FloorMap.from_file(
            self.get_parameter('floormap_file').get_parameter_value().string_value)
        self.pm = PollutionManager(self.fm)

        # load data from json file
        pollution_sources, pollution_end_time, time_bank, has_sensor = load_pollution_json(
            self.get_parameter('pollution_file').get_parameter_value().string_value)
        
        self.pollution_sources = pollution_sources
        self.pollution_end_time = pollution_end_time
        self.time_bank = time_bank
        self.pm.has_sensor = np.array(has_sensor, dtype=bool)

        self.robot_num = self.get_parameter('robot_num').get_parameter_value().integer_value

        self.robots: List[Robot] = []
        for i in range(self.robot_num):
            robot_name = f"robot_{i}"
            purify_speed = 1 # -(pollution level)/s while purifying
            sensor_data = SensorData()

            # sensor data subscriptions
            self.create_subscription(
                LaserScan,
                f"{robot_name}/base_sensor_lidar_front_controller/out",
                sensor_data.callback_lidar_front,
                10,
                )
            self.create_subscription(
                LaserScan,
                f"{robot_name}/base_sensor_lidar_back_controller/out",
                sensor_data.callback_lidar_back,
                10,
                )
            self.create_subscription(
                Image,
                f"{robot_name}/base_sensor_tof_left_sensor/depth/image_raw",
                sensor_data.callback_tof_left,
                10,
                )
            self.create_subscription(
                Image,
                f"{robot_name}/base_sensor_tof_right_sensor/depth/image_raw",
                sensor_data.callback_tof_right,
                10,
                )
            self.create_subscription(
                Image,
                f"{robot_name}/base_sensor_camera_sensor/image_raw",
                sensor_data.callback_camera,
                10,
                )
            self.create_subscription(
                LaserScan,
                f"{robot_name}/base_sensor_ray_controller/out",
                sensor_data.callback_ray,
                10,
                )

            # bot control publication
            cmd_vel_pub = self.create_publisher(Twist, f"{robot_name}/cmd_vel", 10)

            robot = Robot()
            robot.robot_name = robot_name
            robot.purify_speed = purify_speed
            robot.sensor_data = sensor_data
            robot.cmd_vel_pub = cmd_vel_pub
            self.create_subscription(
                Odometry,
                f"{robot_name}/odom",
                robot.callback_odom,
                10,
            )
            for i in range(COLLISION_SENSOR_NUM):
                self.create_subscription(
                    LaserScan,
                    f"{robot_name}/collision_ray_{i}_controller/out",
                    lambda x, idx=i: robot.callback_collision_lidar(idx, x),
                    10,
                )

            self.robots.append(robot)

        # simulation time
        self.sim_time = 0.0
        self.clock_qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # simulation service clients
        self.pause_cli = self.create_client(Empty, "/pause_physics")
        self.unpause_cli = self.create_client(Empty, "/unpause_physics")
        self.reset_cli = self.create_client(Empty, "/reset_world")

        # wait for services
        for cli in self.clients:
            while not cli.wait_for_service(timeout_sec=SRV_WAIT_PERIOD):
                self.get_logger().info(f"service ({cli.srv_name}) not available, waiting again...")

    def advance_simulation(self, steps: int = 1):
        # advance simulation (unpause pause)
        while not self.unpause_cli.wait_for_service(timeout_sec=SRV_WAIT_PERIOD):
            self.get_logger().info(f"service ({self.unpause_cli.srv_name}) not available, waiting again...")
        while not self.pause_cli.wait_for_service(timeout_sec=SRV_WAIT_PERIOD):
            self.get_logger().info(f"service ({self.pause_cli.srv_name}) not available, waiting again...")
        
        future = self.unpause_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        for i in range(steps):
            succ = False
            while not succ:
                succ, msg = wait_for_message(
                    Clock,
                    self,
                    "/clock",
                    qos_profile=self.clock_qos_profile,
                    time_to_wait=10.0,
                    )
                if not succ:
                    self.get_logger().info("/clock wait timeout (10.0s), trying unpuase again...")

                    future = self.unpause_cli.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(self, future)

                    # raise RuntimeError("Failed to get clock message")
            
            self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

        future = self.pause_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

    def reset(self):
        while not self.reset_cli.wait_for_service(timeout_sec=SRV_WAIT_PERIOD):
            self.get_logger().info(f"service ({self.reset_cli.srv_name}) not available, waiting again...")
        
        # reset and pause the world
        future = self.reset_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        wait_steps = random.randint(10, 30)
        self.advance_simulation(wait_steps) # wait 1sec

        self.pm.reset()
        self.pm.clear_sources()
        num_rooms = self.fm.num_rooms
        shuffle_idx = random.sample(range(num_rooms), num_rooms)
        for source in self.pollution_sources:
            self.pm.add_source(
                shuffle_idx[source['room_idx']], # randomly permute sources
                source['speed'],
                source['start_time'],
                source['end_time'],
            )
        self.steps = 0
        for robot in self.robots:
            robot.reset()
            robot.get_disp_pos()
            robot.get_disp_angle()

    def step(self, actions):
        """
        actions: [[MODE, LINEAR, ANGULAR] for each robot].
        if MODE is 0, robot is moving. Twist control with linear/angular speed.
        if MODE is 1, robot is purifying air. LINEAR, ANGULAR is ignored.
        """
        
        # commit actions
        for i in range(self.robot_num):
            robot = self.robots[i]
            action = actions[i]

            if self.debug_mode:
                mode = 0 if robot.vel[0]**2 + robot.vel[1]**2 > 0.1 else 1
            else:
                mode = action[0] if action[0] == 0 or action[0] == 1 else 0
            linear = max(-MAX_LIN_SPD, min(float(action[1]), MAX_LIN_SPD))
            angular = max(-MAX_ANG_SPD, min(float(action[2]), MAX_ANG_SPD))
            action = (mode, linear, angular)

            if not self.debug_mode:
                if action[0] == 0:
                    robot.move(action[1], action[2])
                else:
                    robot.move(0.0, 0.0)

            if action[0] == 0:
                # Moving
                robot.purify_duration = 0
            else:
                # Purifying
                if robot.purify_duration < PURIFY_STARTUP_TIME:
                    # 청정 동작을 시작한 후 1초가 지나야 실제로 청정이 시작됨
                    robot.purify_duration += DELTA_TIME
                else:
                    robot.purify_amount -= self.pm.add_pollution_pos(
                        pos=robot.pos,
                        amount=-robot.purify_speed * DELTA_TIME,
                        )

        # simulate for 1 step
        self.advance_simulation()
        self.pm.step(DELTA_TIME) # step pollution
        self.update_pollution_display()

        # get situation
        results: List[Tuple[Observation, Info, bool, bool]] = [() for _ in range(self.robot_num)]

        for i in range(self.robot_num):
            robot = self.robots[i]

            observation: Observation = {} # data for model input
            info: Info = {} # additional data - only used for reward calculation during training. not provided during test (inference)
            terminated = False
            done = False

            collided = robot.detetct_collision()

            # observation
            observation['sensor_lidar_front'] = robot.sensor_data.lidar_front.copy()
            observation['sensor_lidar_back'] = robot.sensor_data.lidar_back.copy()
            observation['sensor_tof_left'] = robot.sensor_data.tof_left.copy()
            observation['sensor_tof_right'] = robot.sensor_data.tof_right.copy()
            observation['sensor_camera'] = robot.sensor_data.camera.copy()
            observation['sensor_ray'] = robot.sensor_data.ray
            observation['sensor_pollution'] = self.pm.get_pollution_pos(robot.pos) # robot sensor data
            observation['air_sensor_pollution'] = self.pm.get_pollutions_masked() # air sensor data
            observation['disp_position'] = robot.get_disp_pos()
            observation['disp_angle'] = robot.get_disp_angle()

            # info
            info['robot_position'] = robot.pos
            info['robot_angle'] = robot.angle
            info['collided'] = collided > 0
            info['all_pollution'] = self.pm.get_pollutions() # all polution data

            if collided == 2:
                # hard collision: terminate
                terminated = True
                robot.terminated = True
            elif math.dist(robot.pos, self.fm.station_pos) < STATION_RANGE:
                # done: park in station
                done = True
                robot.done = True
                robot.done_time = self.steps * DELTA_TIME

            results[i] = (observation, info, terminated, done)

        self.steps += 1

        return results
    
    def get_map_info(self, robot_idx=0):
        return self.fm.to_map_info(
            self.pollution_end_time,
            self.robots[robot_idx].pos,
            self.robots[robot_idx].angle,
            )
    
    def get_score_data(self, robot_idx):
        robot: Robot = self.robots[robot_idx]

        early_return = robot.done_time < self.pollution_end_time

        score = 0.0
        purify_score = 0.0
        time_score = 0.0
        msg = ""

        if robot.terminated:
            msg = "Score: 0 (terminated)"
        elif early_return:
            msg = "Score: 0 (returned too early)"
        else:
            purify_score = self.calculate_purify_score()
            time_score = self.calculate_time_score(robot)
            score = purify_score + time_score

            msg = f"Score: {score:.2f} (pollution: {purify_score:.2f}, time: {time_score:.2f})"

            if not self.check_purify_completion():
                pollution_left = []
                pollution = self.pm.pollution

                for i in range(self.pm.num_rooms):
                    if pollution[i] > 0:
                        pollution_left.append(f"{self.fm.room_names[i]}: {pollution[i]:.2f}")
                msg += f"\nair pollution left - {', '.join(pollution_left)}"

            if robot.weak_collision_cnt > 0:
                msg += f"\n{robot.weak_collision_cnt} weak collisions"

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "purify_score": purify_score,
            "time_score": time_score,
            "done": robot.done,
            "terminated": robot.terminated,
            "early_return": early_return,
            "done_time": robot.done_time,
            "collisions": robot.weak_collision_cnt,
            "purify_amount": robot.purify_amount,
            "message": msg,
        }
    
    def check_purify_completion(self):
        return np.all(self.pm.pollution == 0.0)
    
    def calculate_purify_score(self):
        purify_score_max = 50

        pollution_total = 0.0
        for source in self.pm.sources:
            pollution_total += source['speed'] * (source['end_time'] - source['start_time'])

        pollution_left = self.pm.pollution.sum()

        return purify_score_max * (1 - pollution_left / pollution_total)
        
    def calculate_time_score(self, robot: Robot):
        time_score_max = 50
        collision_panelty = 30

        score = 0

        adjusted_time = robot.done_time + collision_panelty * robot.weak_collision_cnt

        if adjusted_time < self.time_bank:
            if adjusted_time < self.time_bank - self.pollution_end_time:
                # somehow arrived before pollution_end_time
                factor = 1
            else:
                factor = (self.time_bank - adjusted_time) / (self.time_bank - self.pollution_end_time)
            score += time_score_max * factor
        
        return score
    
    def update_pollution_display(self):
        # update visual indicator in gazebo
        pass

    def close(self):
        while not self.pause_cli.wait_for_service(timeout_sec=SRV_WAIT_PERIOD):
            self.get_logger().info(f"service ({self.pause_cli.srv_name}) not available, waiting again...")
        future = self.pause_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        self.get_logger.info("Simulation Closed.")

def load_pollution_json(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pollution_sources = data['sources']
    pollution_end_time = data['pollution_end_time']
    time_bank = data['time_bank']
    has_sensor = data['has_sensor']
    
    return (pollution_sources, pollution_end_time, time_bank, has_sensor)