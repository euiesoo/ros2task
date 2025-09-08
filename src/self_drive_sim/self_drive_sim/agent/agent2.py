import math, heapq, numpy as np
from typing import List, Tuple, Optional

# --- 패키지 임포트: launch로 실행할 때와 단독 실행 모두 지원 ---
try:
    from self_drive_sim.agent.interfaces import Observation, Info
    from self_drive_sim.simulation.floor_map import MapInfo
except Exception:
    from interfaces import Observation, Info          # fallback
    from floor_map import MapInfo                     # fallback

Grid = Tuple[int, int]


class Agent:
    """
    Map0: A*로 Room(0) 이동 → 방 내부 지그재그 커버리지(오염 감지 시 MODE=1) → 스테이션 복귀
    """

    # ===== 하이퍼파라미터 (안정 튜닝 값) =====
    GRID_NEIGHBORS = [(1,0),(-1,0),(0,1),(0,-1)]
    WP_REACH_DIST_M = 0.22         # waypoint 도달 허용치(셀 0.2m보다 약간 큼)
    MAX_LINEAR = 0.32              # m/s
    ANG_KP = 0.95                  # yaw 오차 게인(너무 크면 빙빙 돎)
    NEAR_SLOW_DIST = 0.35          # 가까우면 속도 감속
    CLEAN_THRESHOLD = 0.35         # 오염도 임계
    CLEAN_DWELL_STEPS = 8          # 청정 유지 스텝

    def __init__(self, logger=print):
        self.logger = logger
        self.map: Optional[MapInfo] = None

        self.grid_origin = (0.0, 0.0)
        self.grid_size = 0.2

        self.path: List[Grid] = []
        self.wp_idx: int = 0

        self.phase = "GO_ROOM"          # GO_ROOM -> COVER_CLEAN -> RETURN -> DOCKED
        self.room_id = 0
        self.cover_cells: List[Grid] = []
        self.cover_idx = 0

        self.station_world = (0.0, 0.0)
        self.station_grid: Grid = (0, 0)

        self.clean_hold = 0
        self._last_cmd = (0, 0.0, 0.0)  # (MODE, v, w)을 항상 보관

    # ==================== 초기화 ====================
    def initialize_map(self, map_info: MapInfo):
        self.map = map_info
        self.grid_origin = tuple(map_info.grid_origin)
        self.grid_size = float(map_info.grid_size)
        self.station_world = tuple(map_info.station_pos)
        self.station_grid = self.pos2grid(self.station_world)

        # 커버리지 순서 생성
        self.cover_cells = self._build_snake_coverage(map_info.room_grid, map_info.wall_grid, self.room_id)
        self.cover_idx = 0

        # 시작/첫 목표
        start_world = getattr(map_info, "starting_pos", self.station_world)
        start_grid = self.pos2grid(start_world)
        first_goal = self.cover_cells[0] if self.cover_cells else start_grid

        self.path = self._a_star(start_grid, first_goal)
        if not self.path:
            # 문턱 쪽 가장 가까운 셀로 폴백
            if self.cover_cells:
                first_goal = min(self.cover_cells, key=lambda c: abs(c[0]-self.station_grid[0])+abs(c[1]-self.station_grid[1]))
                self.path = self._a_star(start_grid, first_goal)
        self.wp_idx = 0
        self.phase = "GO_ROOM"

        self.logger(f"[Agent] init: start={start_grid}, goal={first_goal}, path_len={len(self.path)}")

    # ==================== 좌표 변환 ====================
    def grid2pos(self, cell: Grid) -> Tuple[float, float]:
        gx, gy = cell
        wx = (gx + 0.5 - self.grid_origin[0]) * self.grid_size
        wy = (gy + 0.5 - self.grid_origin[1]) * self.grid_size
        return (wx, wy)

    def pos2grid(self, pos: Tuple[float, float]) -> Grid:
        x, y = pos
        gx = int(round(x / self.grid_size + self.grid_origin[0] - 0.5))
        gy = int(round(y / self.grid_size + self.grid_origin[1] - 0.5))
        return (gx, gy)

    # ==================== A* ====================
    def _a_star(self, start: Grid, goal: Grid) -> List[Grid]:
        wall = self.map.wall_grid  # bool[H,W] (True=벽)
        H, W = wall.shape

        def inb(x: int, y: int) -> bool:  return 0 <= x < W and 0 <= y < H
        def free(x: int, y: int) -> bool:  return inb(x,y) and (not wall[y, x])
        def h(a: Grid, b: Grid) -> int:    return abs(a[0]-b[0]) + abs(a[1]-b[1])

        openq: List[Tuple[int, Grid]] = []
        heapq.heappush(openq, (0, start))
        came = {start: None}
        g = {start: 0}

        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal: break
            for dx, dy in self.GRID_NEIGHBORS:
                nx, ny = cur[0]+dx, cur[1]+dy
                if not free(nx, ny): continue
                cost = g[cur] + 1
                n = (nx, ny)
                if n not in g or cost < g[n]:
                    g[n] = cost
                    heapq.heappush(openq, (cost + h(n, goal), n))
                    came[n] = cur

        if goal not in came:
            self.logger("[Agent] A*: no route")
            return []

        # 경로 복원
        path: List[Grid] = []
        node: Optional[Grid] = goal
        while node is not None:
            path.append(node)
            node = came.get(node)
        path.reverse()
        return path

    # ==================== 커버리지 생성 ====================
    def _build_snake_coverage(self, room_grid: np.ndarray, wall_grid: np.ndarray, room_id: int) -> List[Grid]:
        H, W = room_grid.shape
        cells = [(x,y) for y in range(H) for x in range(W) if room_grid[y,x]==room_id and not wall_grid[y,x]]
        if not cells: return []
        xs, ys = [c[0] for c in cells], [c[1] for c in cells]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

        order: List[Grid] = []
        for y in range(ymin, ymax+1):
            row = [(x,y) for x in range(xmin, xmax+1) if room_grid[y,x]==room_id and not wall_grid[y,x]]
            if not row: continue
            row = sorted(row, key=(lambda c:  c[0] if (y-ymin)%2==0 else -c[0]))
            order.extend(row)

        # 너무 촘촘하면 1칸씩만 취함
        thinned = []
        for i, c in enumerate(order):
            if i==0 or (abs(c[0]-thinned[-1][0])+abs(c[1]-thinned[-1][1])>=1):
                thinned.append(c)
        return thinned

    # ==================== 행동 ====================
    def act(self, observation: Observation) -> Tuple[int, float, float]:
        x, y, yaw = self._read_pose(observation)
        pollution = observation.get("sensor_pollution", 0.0)

        # 청정 유지
        if self.clean_hold > 0:
            self.clean_hold -= 1
            self._last_cmd = (1, 0.0, 0.0)
            return self._last_cmd

        # 상태머신
        if self.phase == "GO_ROOM":
            if self._follow_path(x, y, yaw):
                self.phase = "COVER_CLEAN"
                self.cover_idx = 0
                if self.cover_idx < len(self.cover_cells):
                    self._plan_to(self.cover_cells[self.cover_idx], current_world=(x, y))

        elif self.phase == "COVER_CLEAN":
            if pollution > self.CLEAN_THRESHOLD and self._inside_room(x, y):
                self.clean_hold = self.CLEAN_DWELL_STEPS
                self._last_cmd = (1, 0.0, 0.0)
                return self._last_cmd

            if self._follow_path(x, y, yaw):
                self.cover_idx += 1
                if self.cover_idx < len(self.cover_cells):
                    self._plan_to(self.cover_cells[self.cover_idx], current_world=(x, y))
                else:
                    self.phase = "RETURN"
                    self._plan_to(self.station_grid, current_world=(x, y))

        elif self.phase == "RETURN":
            if self._follow_path(x, y, yaw):
                self.phase = "DOCKED"
                self._last_cmd = (0, 0.0, 0.0)
                return self._last_cmd

        # 👉 중요: 마지막 이동 명령 반환
        return self._last_cmd

    def learn(self, observation: Observation, info: Info, action,
              next_observation: Observation, next_info: Info,
              terminated: bool, done: bool):
        pass

    # ==================== 내부 유틸 ====================
    def _read_pose(self, obs: Observation) -> Tuple[float, float, float]:
        if "pose" in obs:
            x,y,yaw = obs["pose"];        return float(x),float(y),float(yaw)
        if "disp_position" in obs and "disp_angle" in obs:
            x,y = obs["disp_position"];   return float(x),float(y),float(obs["disp_angle"])
        if "position" in obs and "angle" in obs:
            x,y = obs["position"];        return float(x),float(y),float(obs["angle"])
        return 0.0, 0.0, 0.0

    def _inside_room(self, x: float, y: float) -> bool:
        gx, gy = self.pos2grid((x, y))
        H, W = self.map.room_grid.shape
        return (0 <= gy < H and 0 <= gx < W and self.map.room_grid[gy, gx] == self.room_id)

    def _plan_to(self, goal_grid: Grid, current_world: Optional[Tuple[float,float]]=None):
        if current_world is not None:
            cur_grid = self.pos2grid(current_world)
        else:
            cur_grid = self.station_grid
        self.path = self._a_star(cur_grid, goal_grid)
        self.wp_idx = 0
        self.logger(f"[Plan] {cur_grid} -> {goal_grid}, len={len(self.path)}")

    def _follow_path(self, x: float, y: float, yaw: float) -> bool:
        if not self.path:             # 경로 없음 → 목표 달성으로 간주
            self._last_cmd = (0, 0.0, 0.0)
            return True

        gx, gy = self.path[self.wp_idx]
        wx, wy = self.grid2pos((gx, gy))
        dx, dy = (wx - x), (wy - y)
        dist = math.hypot(dx, dy)

        # waypoint 도달 판정
        if dist < self.WP_REACH_DIST_M:
            if self.wp_idx < len(self.path) - 1:
                self.wp_idx += 1
                # 다음 step에서 계속 추종
                self._last_cmd = (0, 0.0, 0.0)
                return False
            else:
                self._last_cmd = (0, 0.0, 0.0)
                return True

        # 제어 명령 생성
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = ((tgt_yaw - yaw + math.pi) % (2*math.pi)) - math.pi
        lin = self.MAX_LINEAR * (0.6 if dist < self.NEAR_SLOW_DIST else 1.0)
        ang = self.ANG_KP * yaw_err

        self._last_cmd = (0, float(lin), float(ang))
        return False

    def reset(self):
        self.path.clear()
        self.wp_idx = 0
        self.phase = "GO_ROOM"
        self.cover_idx = 0
        self.clean_hold = 0
        self._last_cmd = (0, 0.0, 0.0)
