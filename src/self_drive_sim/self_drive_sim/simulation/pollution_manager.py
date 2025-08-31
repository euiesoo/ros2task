import numpy as np
from self_drive_sim.simulation.floor_map import FloorMap

"""
공기 오염/청정 로직
FloorMap 기반으로 작동하며, GazeboEnv에서 사용
실제로 Gazebo와 통신하는건 GazeboEnv에서 구현 예정 (오염 표시기 등)
"""

class PollutionManager():
    def __init__(self, floor_map: FloorMap):
        """
        floor_map: FloorMap object
        """

        self.floor_map = floor_map
        self.num_rooms = floor_map.get_room_count()
        self.pollution = np.zeros(self.num_rooms, dtype=float)  # pollution level per room
        self.sources = []  # list of dicts: {room, speed, start, end}

        self.pollution = np.zeros(self.num_rooms, dtype=float)
        self.sim_time = 0.0

        self.has_sensor = np.array([True] * self.num_rooms)

    def add_source(self, room_idx, speed, start_time, end_time):
        """
        Add a pollution source (or sink if speed<0).
        room: int, room index
        speed: float, pollution per second (can be negative)
        start_time: float, seconds
        end_time: float, seconds
        """
        self.sources.append({
            'room_idx': room_idx,
            'speed': speed,
            'start_time': start_time,
            'end_time': end_time
        })
    
    def clear_sources(self):
        self.sources = []

    def set_pollution_room(self, room_idx, value):
        """Set pollution in a room to a specific value."""
        if room_idx == -1:
            return
        
        if value < 0:
            value = 0
        self.pollution[room_idx] = value

    def add_pollution_room(self, room_idx, amount):
        """Increase pollution in a room by amount (can be negative)."""
        if room_idx == -1:
            return 0
        
        delta = max(-self.pollution[room_idx], amount)

        new_val = self.pollution[room_idx] + delta
        self.set_pollution_room(room_idx, new_val)

        return delta

    def set_pollution_pos(self, pos, value):
        """Set pollution in the room containing pos(x, y) to value. Raise error if non-room."""
        x, y = self.floor_map.pos2grid(pos)
        room_idx = self.floor_map.get_room_id(x, y)
        self.set_pollution_room(room_idx, value)

    def add_pollution_pos(self, pos, amount):
        """Increase pollution in the room containing pos(x, y) by amount. Raise error if non-room."""
        x, y = self.floor_map.pos2grid(pos)
        room_idx = self.floor_map.get_room_id(x, y)
        return self.add_pollution_room(room_idx, amount)

    def get_pollution_pos(self, pos):
        """Set pollution in the room containing pos(x, y) to value. Raise error if non-room."""
        x, y = self.floor_map.pos2grid(pos)
        room_idx = self.floor_map.get_room_id(x, y)
        if room_idx == -1:
            return None
        return self.pollution[room_idx]
    
    def get_pollutions(self):
        return self.pollution.copy()
    
    def get_pollutions_masked(self):
        masked = self.pollution.copy()
        masked[~self.has_sensor] = np.nan

        return masked
    
    def reset(self):
        """Initialize/reset simulation, clear all pollution levels."""
        self.pollution = np.zeros(self.num_rooms, dtype=float)
        self.sim_time = 0.0

    def step(self, dt=0.1):
        """
        Simulate a step of dt seconds. Update pollution levels according to sources.
        """
        self.sim_time += dt
        for src in self.sources:
            if src['start_time'] <= self.sim_time < src['end_time']:
                self.add_pollution_room(
                    room_idx=src['room_idx'],
                    amount=src['speed'] * dt,
                    )