from self_drive_sim.agent.interfaces import Observation, Info, MapInfo
import numpy as np
import math
from collections import deque
from enum import Enum

class RobotState(Enum):
    EXPLORING = 1
    MOVING_TO_TARGET = 2
    PURIFYING = 3
    RETURNING_TO_DOCK = 4
    OBSTACLE_AVOIDANCE = 5

class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0
        
        # Map information
        self.map_info = None
        self.current_pos = (0, 0)
        self.current_angle = 0
        self.target_pos = None
        
        # State management
        self.state = RobotState.EXPLORING
        self.visited_cells = set()
        self.room_clean_status = {}  # room_id -> is_clean
        self.purify_timer = 0
        self.purify_threshold = 50  # Start purifying when pollution > threshold
        
        # Path planning
        self.path_queue = deque()
        self.exploration_targets = deque()
        
        # Control parameters
        self.max_linear_speed = 0.8
        self.max_angular_speed = 1.5
        self.position_tolerance = 0.3
        self.angle_tolerance = 0.2
        
        # Safety parameters
        self.obstacle_distance_threshold = 0.5
        self.collision_avoidance_active = False
        self.avoidance_timer = 0
        
        # Purification parameters
        self.min_purify_time = 30  # Minimum time to spend purifying
        self.current_room_pollution = 0
        
    def initialize_map(self, map_info: MapInfo):
        """Initialize map information and plan initial exploration"""
        self.map_info = map_info
        self.current_pos = map_info.starting_pos
        self.current_angle = map_info.starting_angle
        
        # Initialize room clean status
        for i in range(map_info.num_rooms):
            self.room_clean_status[i] = False
            
        # Generate exploration targets for each room
        self._generate_exploration_targets()
        
        self.log(f"Initialized map with {map_info.num_rooms} rooms")
        self.log(f"Starting position: {self.current_pos}")
        self.log(f"Station position: {map_info.station_pos}")
        self.log(f"Pollution end time: {map_info.pollution_end_time}")

    def _generate_exploration_targets(self):
        """Generate target positions for each room"""
        self.exploration_targets.clear()
        
        for room_id in range(self.map_info.num_rooms):
            room_cells = self.map_info.get_cells_in_room(room_id)
            if room_cells:
                # Find center of room
                center_x = sum(cell[0] for cell in room_cells) / len(room_cells)
                center_y = sum(cell[1] for cell in room_cells) / len(room_cells)
                center_pos = self.map_info.grid2pos((center_x, center_y))
                
                self.exploration_targets.append((room_id, center_pos))
                self.log(f"Room {room_id} ({self.map_info.room_names[room_id]}) center: {center_pos}")

    def act(self, observation: Observation):
        """Main action selection logic"""
        self._update_position_estimate(observation)
        self._analyze_sensors(observation)
        
        # State machine
        action = self._execute_state_machine(observation)
        
        # Log status periodically
        if self.steps % 100 == 0:
            self._log_status(observation)
            
        self.steps += 1
        return action

    def _update_position_estimate(self, observation: Observation):
        """Update position estimate using displacement data"""
        disp_x, disp_y = observation['disp_position']
        disp_angle = observation['disp_angle']
        
        # Update position (simple dead reckoning - could be improved with SLAM)
        self.current_pos = (
            self.current_pos[0] + disp_x,
            self.current_pos[1] + disp_y
        )
        self.current_angle = (self.current_angle + disp_angle) % (2 * math.pi)

    def _analyze_sensors(self, observation: Observation):
        """Analyze sensor data for obstacles and pollution"""
        # Check LiDAR for obstacles
        lidar_front = observation['sensor_lidar_front']
        lidar_back = observation['sensor_lidar_back']
        
        # Find minimum distance in front sector
        front_min_dist = np.min(lidar_front[80:160])  # Front 80 degrees
        
        # Check if obstacle avoidance is needed
        if front_min_dist < self.obstacle_distance_threshold:
            self.collision_avoidance_active = True
            self.avoidance_timer = 0
        elif self.avoidance_timer > 20:  # Clear avoidance after 2 seconds
            self.collision_avoidance_active = False
            
        # Update pollution information
        self.current_room_pollution = observation['sensor_pollution']

    def _execute_state_machine(self, observation: Observation):
        """Execute current state and return action"""
        
        if self.collision_avoidance_active:
            return self._obstacle_avoidance_behavior(observation)
        
        if self.state == RobotState.EXPLORING:
            return self._exploration_behavior(observation)
        elif self.state == RobotState.MOVING_TO_TARGET:
            return self._move_to_target_behavior(observation)
        elif self.state == RobotState.PURIFYING:
            return self._purification_behavior(observation)
        elif self.state == RobotState.RETURNING_TO_DOCK:
            return self._return_to_dock_behavior(observation)
        else:
            return (0, 0, 0)

    def _exploration_behavior(self, observation: Observation):
        """Explore rooms and identify areas needing cleaning"""
        
        # Check if current area needs cleaning
        current_room_id = self._get_current_room()
        if current_room_id >= 0 and self.current_room_pollution > self.purify_threshold:
            self.state = RobotState.PURIFYING
            self.purify_timer = 0
            return (1, 0, 0)  # Start purifying
        
        # Check if all rooms are clean and pollution period ended
        if self._all_rooms_clean() and self.steps * 0.1 > self.map_info.pollution_end_time:
            self.state = RobotState.RETURNING_TO_DOCK
            self.target_pos = self.map_info.station_pos
            return self._move_to_target_behavior(observation)
        
        # Move to next exploration target
        if not self.exploration_targets:
            self._generate_exploration_targets()
            
        if self.exploration_targets:
            room_id, target_pos = self.exploration_targets[0]
            self.target_pos = target_pos
            self.state = RobotState.MOVING_TO_TARGET
            return self._move_to_target_behavior(observation)
        
        return (0, 0, 0)

    def _move_to_target_behavior(self, observation: Observation):
        """Move towards target position"""
        if self.target_pos is None:
            self.state = RobotState.EXPLORING
            return (0, 0, 0)
        
        # Calculate distance and angle to target
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)
        
        # Check if we've reached the target
        if distance < self.position_tolerance:
            if self.state == RobotState.RETURNING_TO_DOCK:
                return (0, 0, 0)  # Mission complete
            else:
                # Reached exploration target
                if self.exploration_targets:
                    self.exploration_targets.popleft()
                self.state = RobotState.EXPLORING
                return self._exploration_behavior(observation)
        
        # Calculate control commands
        angle_diff = self._normalize_angle(target_angle - self.current_angle)
        
        # If angle difference is large, rotate first
        if abs(angle_diff) > self.angle_tolerance:
            angular_speed = self._clamp(angle_diff * 2.0, -self.max_angular_speed, self.max_angular_speed)
            return (0, 0, angular_speed)
        else:
            # Move forward
            linear_speed = min(distance * 2.0, self.max_linear_speed)
            angular_speed = self._clamp(angle_diff, -0.5, 0.5)
            return (0, linear_speed, angular_speed)

    def _purification_behavior(self, observation: Observation):
        """Purify current area"""
        self.purify_timer += 1
        
        # Check if purification is complete or sufficient time has passed
        if (self.current_room_pollution < 5 or 
            self.purify_timer > self.min_purify_time):
            
            current_room_id = self._get_current_room()
            if current_room_id >= 0:
                self.room_clean_status[current_room_id] = True
                self.log(f"Completed cleaning room {current_room_id}")
            
            self.state = RobotState.EXPLORING
            return self._exploration_behavior(observation)
        
        return (1, 0, 0)  # Continue purifying

    def _return_to_dock_behavior(self, observation: Observation):
        """Return to docking station"""
        self.target_pos = self.map_info.station_pos
        return self._move_to_target_behavior(observation)

    def _obstacle_avoidance_behavior(self, observation: Observation):
        """Avoid obstacles using sensor data"""
        self.avoidance_timer += 1
        
        lidar_front = observation['sensor_lidar_front']
        tof_left = observation['sensor_tof_left']
        tof_right = observation['sensor_tof_right']
        
        # Simple obstacle avoidance: find direction with most free space
        left_distances = lidar_front[160:241]  # Left side
        right_distances = lidar_front[0:80]    # Right side
        
        left_avg = np.mean(left_distances[left_distances > 0.1])
        right_avg = np.mean(right_distances[right_distances > 0.1])
        
        # Choose direction with more free space
        if left_avg > right_avg:
            return (0, 0.3, 0.8)  # Turn left and move slowly
        else:
            return (0, 0.3, -0.8)  # Turn right and move slowly

    def _get_current_room(self):
        """Get current room ID based on position"""
        grid_pos = self.map_info.pos2grid(self.current_pos)
        return self.map_info.get_room_id(grid_pos[0], grid_pos[1])

    def _all_rooms_clean(self):
        """Check if all rooms have been cleaned"""
        return all(self.room_clean_status.values())

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _clamp(self, value, min_val, max_val):
        """Clamp value between min and max"""
        return max(min_val, min(value, max_val))

    def _log_status(self, observation: Observation):
        """Log current status"""
        current_room = self._get_current_room()
        pollution_levels = observation['air_sensor_pollution']
        
        status_msg = f"Step {self.steps}, State: {self.state.name}, "
        status_msg += f"Room: {current_room}, Pollution: {self.current_room_pollution:.1f}, "
        status_msg += f"Position: ({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f})"
        
        self.log(status_msg)
        
        # Log room pollution levels
        if not np.isnan(pollution_levels).all():
            room_status = []
            for i, pollution in enumerate(pollution_levels):
                if not np.isnan(pollution):
                    clean_status = "✓" if self.room_clean_status.get(i, False) else "✗"
                    room_status.append(f"{self.map_info.room_names[i]}: {pollution:.1f}{clean_status}")
            if room_status:
                self.log(f"Room status: {', '.join(room_status)}")

    def learn(self, observation: Observation, info: Info, action, 
              next_observation: Observation, next_info: Info, 
              terminated, done):
        """Learning function - can be used for reinforcement learning"""
        # Update position with accurate info during training
        if 'robot_position' in info:
            self.current_pos = info['robot_position']
            self.current_angle = info['robot_angle']
        
        # Learn from collision events
        if info.get('collided', False):
            self.log("Collision detected - adjusting behavior")
            self.collision_avoidance_active = True
            self.avoidance_timer = 0

    def reset(self):
        """Reset agent state for new episode"""
        self.steps = 0
        self.state = RobotState.EXPLORING
        self.visited_cells.clear()
        self.room_clean_status.clear()
        self.purify_timer = 0
        self.path_queue.clear()
        self.exploration_targets.clear()
        self.target_pos = None
        self.collision_avoidance_active = False
        self.avoidance_timer = 0
        
        if self.map_info:
            self.current_pos = self.map_info.starting_pos
            self.current_angle = self.map_info.starting_angle
            for i in range(self.map_info.num_rooms):
                self.room_clean_status[i] = False

    def log(self, msg):
        """Log message using ROS logger"""
        self.logger(str(msg))
