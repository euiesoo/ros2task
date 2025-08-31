import numpy as np
import math

"""
매핑 정보
격자 형태로 벽, 방 정보를 저장하며, 모든 정보는 에이전트에게 오픈됨
"""

class FloorMap():
    @classmethod
    def from_file(cls, path):
        """Load FloorMap from a .npz file and check room connectivity."""
        data = np.load(path, allow_pickle=True)
        wall_grid = data['wall_grid']
        room_grid = data['room_grid']
        num_rooms = int(data['num_rooms']) if 'num_rooms' in data else 0
        grid_size = data['grid_size']
        # in npz file origin is stored in (y, x)
        grid_origin = data['grid_origin']
        grid_origin[0], grid_origin[1] = grid_origin[1], grid_origin[0]
        station_grid = data['station']
        station_grid[0], station_grid[1] = station_grid[1], station_grid[0]
        room_names = data['room_names'] if 'room_names' in data else [f"Room {i}" for i in range(num_rooms)]
        obj = cls(
            wall_grid.shape,
            wall_grid=wall_grid,
            room_grid=room_grid,
            num_rooms=num_rooms,
            grid_size=grid_size,
            grid_origin=grid_origin,
            station_grid=station_grid,
            room_names=room_names,
            )
        # Check connectivity for each room (ignore -1)
        for room_id in np.unique(room_grid):
            if room_id == -1:
                continue
            cells = np.argwhere(room_grid == room_id)
            if len(cells) == 0:
                continue
            # BFS from first cell
            visited = set()
            queue = [tuple(cells[0])]
            visited.add(tuple(cells[0]))
            cell_set = set(map(tuple, cells))
            while queue:
                y, x = queue.pop(0)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if (ny, nx) in cell_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
            if len(visited) != len(cells):
                print(f"Warning: Room {room_id} is not fully connected (has {len(cells)} cells, but only {len(visited)} are connected).")
        return obj

    def __init__(
            self,
            grid_shape,
            wall_grid=None,
            room_grid=None,
            num_rooms=0,
            grid_size=0.5,
            grid_origin=(0.0, 0.0),
            station_grid=(0.0, 0.0),
            room_names=None,
            ):
        """
        grid_shape: tuple (height, width) of the map
        wall_grid: optional, numpy array of shape (H, W), bool, True if wall
        room_grid: optional, numpy array of shape (H, W), int, room id for each cell
        num_rooms: optional, number of rooms (if known)
        """
        self.height, self.width = grid_shape
        if wall_grid is not None:
            self.wall_grid = wall_grid.astype(bool)
        else:
            self.wall_grid = np.zeros(grid_shape, dtype=bool)
        if room_grid is not None:
            self.room_grid = room_grid.astype(int)
        else:
            self.room_grid = np.full(grid_shape, -1, dtype=int)  # -1: not assigned
        self.num_rooms = num_rooms if num_rooms > 0 else (np.max(self.room_grid) + 1 if np.any(self.room_grid >= 0) else 0)
        self.grid_size = grid_size
        self.grid_origin = grid_origin
        self.station_pos = self.grid2pos(station_grid)
        if room_names is not None:
            self.room_names = room_names
        else:
            self.room_names = [f"Room {i}" for i in range(self.num_rooms)]

    def is_wall(self, x, y):
        """Return True if cell (x, y) is a wall."""
        xi = math.floor(x)
        yi = math.floor(y)
        
        if xi < 0 or xi >= self.height or yi < 0 or yi >= self.width:
            return False
        
        return self.wall_grid[xi, yi]

    def get_room_id(self, x, y):
        """Return room id for cell (x, y), or -1 if not assigned."""
        xi = math.floor(x)
        yi = math.floor(y)
        
        if xi < 0 or xi >= self.height or yi < 0 or yi >= self.width:
            return -1
        
        return self.room_grid[xi, yi]

    def get_cells_in_room(self, room_id):
        """Return list of (x, y) tuples for all cells in the given room."""
        xs, ys = np.where(self.room_grid == room_id)
        return list(zip(xs, ys))

    def get_room_count(self):
        return self.num_rooms

    def grid2pos(self, grid):
        """
        convert grid position to world position
        """
        x = (grid[0] - self.grid_origin[0]) * self.grid_size
        y = (grid[1] - self.grid_origin[1]) * self.grid_size

        return (x, y)
    
    def pos2grid(self, pos):
        """
        convert world position to grid position
        """
        x = pos[0] / self.grid_size + self.grid_origin[0]
        y = pos[1] / self.grid_size + self.grid_origin[1]

        return (x, y)
    
    def to_map_info(self, pollution_end_time: float, starting_pos, starting_angle):
        """
        Convert this FloorMap to a MapInfo object.
        pollution_end_time: float, time when the last pollution source stops emitting.
        """
        return MapInfo(self, pollution_end_time, starting_pos, starting_angle)

class MapInfo:
    def __init__(self, floor_map: FloorMap, pollution_end_time: float, starting_pos=(0, 0), starting_angle=0.0):
        self.height = floor_map.height
        self.width = floor_map.width
        self.wall_grid = floor_map.wall_grid.copy()
        self.room_grid = floor_map.room_grid.copy()
        self.num_rooms = floor_map.num_rooms
        self.grid_size = floor_map.grid_size
        self.grid_origin = floor_map.grid_origin
        self.station_pos = floor_map.station_pos
        self.room_names = floor_map.room_names

        self.pollution_end_time = pollution_end_time
        self.starting_pos = starting_pos
        self.starting_angle = starting_angle

    def is_wall(self, x, y):
        """Return True if cell (x, y) is a wall."""
        xi = math.floor(x)
        yi = math.floor(y)
        
        if xi < 0 or xi >= self.height or yi < 0 or yi >= self.width:
            return False
        
        return self.wall_grid[xi, yi]

    def get_room_id(self, x, y):
        """Return room id for cell (x, y), or -1 if not assigned."""
        xi = math.floor(x)
        yi = math.floor(y)
        
        if xi < 0 or xi >= self.height or yi < 0 or yi >= self.width:
            return -1
        
        return self.room_grid[xi, yi]

    def get_cells_in_room(self, room_id):
        """Return list of (x, y) tuples for all cells in the given room."""
        xs, ys = np.where(self.room_grid == room_id)
        return list(zip(xs, ys))

    def grid2pos(self, grid):
        """
        convert grid position to world position
        """
        x = (grid[0] - self.grid_origin[0]) * self.grid_size
        y = (grid[1] - self.grid_origin[1]) * self.grid_size

        return (x, y)
    
    def pos2grid(self, pos):
        """
        convert world position to grid position
        """
        x = pos[0] / self.grid_size + self.grid_origin[0]
        y = pos[1] / self.grid_size + self.grid_origin[1]

        return (x, y)