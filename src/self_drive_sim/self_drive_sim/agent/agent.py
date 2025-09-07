# 이동식 공기청정기 테스트 2025.09

import numpy as np
import math
import heapq
from typing import List, Tuple, Optional
import os, sys

# sys.path.append('home/elicer/ros_ws/src/self_drive_sim/self_drive_sim/agent')
# sys.path.append(os.path.join(os.path.dirname(__file__),"..", "self_drive_sim"))

from interfaces import Observation, Info
from floor_map import MapInfo

Grid = Tuple[int, int]

class Agent:
    """ mission
    Map0: 방(0)까지 A* 이동 → 방 내부 커버리지(오염 감지 시 MODE=1 청정) → 스테이션 복귀
    """

    # ===== 파라미터 정의=====
    GRID_NEIGHBORS = [(1,0), (-1,0), (0,1), (0,-1)]  # 4방향
    WP_REACH_DIST_M = 0.18        # 웨이포인트 도달 판정 (grid_size=0.2m 기준)
    MAX_LINEAR = 0.35             # [m/s]
    ANG_KP = 1.2                  # 각오차 게인
    NEAR_SLOW_DIST = 0.35         # 근접 시 감속
    CLEAN_THRESHOLD = 0.35        # 오염도 임계값 (관측치 > 임계 → MODE=1)
    CLEAN_DWELL_STEPS = 8         # 청정 유지 스텝 (센서 변동 안정화 목적)

    def __init__(self, logger=print):
        self.logger = logger
        self.map: Optional[MapInfo] = None

        # 좌표 변환에 사용
        self.grid_origin = (0.0, 0.0)
        self.grid_size = 0.2

        # 경로/웨이포인트
        self.path: List[Grid] = []
        self.wp_idx: int = 0

        # 상태머신
        self.phase = "GO_ROOM"     # GO_ROOM -> COVER_CLEAN -> RETURN -> DOCKED

        # 커버리지 대상
        self.room_id = 0
        self.cover_cells: List[Grid] = []   # 방 내부 지그재그 순회 셀들
        self.cover_idx = 0

        # 스테이션
        self.station_world = (0.0, 0.0)
        self.station_grid: Grid = (0, 0)

        # 청정 유지
        self.clean_hold = 0

    # ==================== 초기화 ====================
    def initialize_map(self, map_info: MapInfo):
        self.map = map_info
        self.grid_origin = tuple(map_info.grid_origin)  # (ox, oy)
        self.grid_size = float(map_info.grid_size)
        self.station_world = tuple(map_info.station_pos)
        self.station_grid = self.pos2grid(self.station_world)

        # 방 0 셀 수집
        self.cover_cells = self._build_snake_coverage(map_info.room_grid, map_info.wall_grid, self.room_id)
        self.cover_idx = 0

        # 시작 그리드
        start_world = getattr(map_info, "starting_pos", self.station_world)
        start_grid = self.pos2grid(start_world)

        # 첫 목표: 방 커버리지의 첫 셀
        first_goal = self.cover_cells[0] if self.cover_cells else start_grid
        self.path = self._a_star(start_grid, first_goal)
        self.wp_idx = 0
        self.phase = "GO_ROOM"

        self.logger(f"[Agent] init: start={start_grid}, first_goal={first_goal}, path_len={len(self.path)}")

    # ==================== 좌표 변환 ====================
    def grid2pos(self, cell: Grid) -> Tuple[float, float]:
        """grid index -> world (셀 중심)"""
        gx, gy = cell
        wx = (gx + 0.5 - self.grid_origin[0]) * self.grid_size
        wy = (gy + 0.5 - self.grid_origin[1]) * self.grid_size
        return (wx, wy)

    def pos2grid(self, pos: Tuple[float, float]) -> Grid:
        """world -> grid index (반올림)"""
        x, y = pos
        gx = int(round(x / self.grid_size + self.grid_origin[0] - 0.5))
        gy = int(round(y / self.grid_size + self.grid_origin[1] - 0.5))
        return (gx, gy)

    # ==================== A* 경로계획 ====================
    def _a_star(self, start: Grid, goal: Grid) -> List[Grid]:
        wall = self.map.wall_grid  # bool[h, w] (True=벽)
        H, W = wall.shape

        def inb(x: int, y: int) -> bool:
            return 0 <= x < W and 0 <= y < H

        def free(x: int, y: int) -> bool:
            return inb(x, y) and (not wall[y, x])

        def h(a: Grid, b: Grid) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        openq: List[Tuple[int, Grid]] = []
        heapq.heappush(openq, (0, start))
        came = {start: None}
        g = {start: 0}

        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                break
            for dx, dy in self.GRID_NEIGHBORS:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not free(nx, ny):
                    continue
                cost = g[cur] + 1
                n = (nx, ny)
                if n not in g or cost < g[n]:
                    g[n] = cost
                    f = cost + h(n, goal)
                    heapq.heappush(openq, (f, n))
                    came[n] = cur

        if goal not in came:
            self.logger("[Agent] A*: no route; returning []")
            return []

        # 복원
        path: List[Grid] = []
        node: Optional[Grid] = goal
        while node is not None:
            path.append(node)
            node = came.get(node)
        path.reverse()
        return path

    # ==================== 커버리지 경로 생성 ====================
    def _build_snake_coverage(self, room_grid: np.ndarray, wall_grid: np.ndarray, room_id: int) -> List[Grid]:
        """
        방 번호(room_id)에 해당하는 셀만 대상으로 지그재그 순회 순서를 만든다.
        - 벽(True)은 제외
        - 방의 bounding box 를 따라 행 단위로 스네이크
        """
        H, W = room_grid.shape
        cells = [(x, y) for y in range(H) for x in range(W)
                 if (room_grid[y, x] == room_id and not wall_grid[y, x])]

        if not cells:
            return []

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        order: List[Grid] = []
        for y in range(ymin, ymax + 1):
            row = [(x, y) for x in range(xmin, xmax + 1)
                   if room_grid[y, x] == room_id and not wall_grid[y, x]]
            if not row:
                continue
            if (y - ymin) % 2 == 0:
                row = sorted(row, key=lambda c: c[0])     # → 방향
            else:
                row = sorted(row, key=lambda c: -c[0])    # ← 방향
            order.extend(row)

        # 너무 촘촘하면 1칸씩만 취함(충돌/요동 방지)
        thinned = []
        for i, c in enumerate(order):
            if i == 0 or (abs(c[0]-thinned[-1][0]) + abs(c[1]-thinned[-1][1]) >= 1):
                thinned.append(c)
        return thinned

    # ==================== 행동 ====================
    def act(self, observation: Observation) -> Tuple[int, float, float]:
        # --- 현재 pose 읽기 ---
        x, y, yaw = self._read_pose(observation)

        # --- 오염 센서 ---
        pollution = observation.get("sensor_pollution", 0.0)

        # --- 청정 유지 로직 ---
        if self.clean_hold > 0:
            self.clean_hold -= 1
            return (1, 0.0, 0.0)  # MODE=1 유지

        # --- 상태 전이 & 목표 경로 유지 ---
        if self.phase == "GO_ROOM":
            # 방 커버리지 첫 셀까지 도달하면 커버리지 시작
            if self._follow_path(x, y, yaw):
                self.phase = "COVER_CLEAN"
                self.cover_idx = 0
                # 다음 목표로 경로 재계획
                if self.cover_idx < len(self.cover_cells):
                    self._plan_to(self.cover_cells[self.cover_idx])

        elif self.phase == "COVER_CLEAN":
            # 오염이 높으면 청정
            if pollution > self.CLEAN_THRESHOLD and self._inside_room(x, y):
                self.clean_hold = self.CLEAN_DWELL_STEPS
                return (1, 0.0, 0.0)

            # 현재 목표 웨이포인트가 끝났으면 다음 커버리지 셀로
            if self._follow_path(x, y, yaw):
                self.cover_idx += 1
                if self.cover_idx < len(self.cover_cells):
                    self._plan_to(self.cover_cells[self.cover_idx])
                else:
                    # 커버리지 완료 → 스테이션 복귀
                    self.phase = "RETURN"
                    self._plan_to(self.station_grid)

        elif self.phase == "RETURN":
            if self._follow_path(x, y, yaw):
                self.phase = "DOCKED"
                return (0, 0.0, 0.0)

        # 기본 값(이동 중 행동은 _follow_path 내에서 결정)
        return (0, 0.0, 0.0)

    def learn(self, observation: Observation, info: Info, action,
              next_observation: Observation, next_info: Info,
              terminated: bool, done: bool):
        # 비학습형   
        pass

    # ==================== 내부 유틸 ====================
    def _read_pose(self, obs: Observation) -> Tuple[float, float, float]:
        # 다양한 키를 허용
        if "pose" in obs:
            x, y, yaw = obs["pose"]
            return float(x), float(y), float(yaw)
        if "disp_position" in obs and "disp_angle" in obs:
            x, y = obs["disp_position"]
            return float(x), float(y), float(obs["disp_angle"])
        if "position" in obs and "angle" in obs:
            x, y = obs["position"]
            return float(x), float(y), float(obs["angle"])
        # 안전장치
        return 0.0, 0.0, 0.0

    def _inside_room(self, x: float, y: float) -> bool:
        """현재가 방(room_id) 내부인지 여부"""
        gx, gy = self.pos2grid((x, y))
        H, W = self.map.room_grid.shape
        if 0 <= gy < H and 0 <= gx < W:
            return self.map.room_grid[gy, gx] == self.room_id
        return False

    def _plan_to(self, goal_grid: Grid):
        cur_grid = self._safe_current_grid()
        self.path = self._a_star(cur_grid, goal_grid)
        self.wp_idx = 0

    def _safe_current_grid(self) -> Grid:
        # 현재 위치를 월드->그리드로 변환 (가끔 경계값 튐 방지)
        # 실제 pose는 act()에서만 읽을 수 있으므로, 가장 최근 plan의 시작점이 비었으면 station 기준
        if self.path and 0 <= self.wp_idx < len(self.path):
            return self.path[self.wp_idx]
        return self.station_grid

    def _follow_path(self, x: float, y: float, yaw: float) -> bool:
        """
        경로를 따라 1 step 제어 명령을 보낸다.
        return: True면 '경로의 마지막 웨이포인트'에 도달(=목표 달성)
        """
        if not self.path:
            return True

        # 현재 웨이포인트
        gx, gy = self.path[self.wp_idx]
        wx, wy = self.grid2pos((gx, gy))

        dx, dy = (wx - x), (wy - y)
        dist = math.hypot(dx, dy)

        # 웨이포인트 도달 → 다음
        if dist < self.WP_REACH_DIST_M:
            if self.wp_idx < len(self.path) - 1:
                self.wp_idx += 1
                return False
            else:
                # 경로 최종 도달
                return True

        # 진행 명령 산출
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = ((tgt_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi

        lin = self.MAX_LINEAR * (0.55 if dist < self.NEAR_SLOW_DIST else 1.0)
        ang = self.ANG_KP * yaw_err

        # 이동 명령 송신
        self._issue_move(lin, ang)
        return False

    # --- 시뮬레이터와의 인터페이스: 여기서는 (mode, v, w) 반환 대신 안에서만 저장 ---
    def _issue_move(self, linear: float, angular: float):
        # act()의 반환값만 사용하는 구조라면, 여기서 값을 저장했다가 act()에서 꺼내는 방식으로 바꿀 수 있음(??)
        self._last_cmd = (0, float(linear), float(angular))
        # trick: act()의 기본 반환(0,0,0)을 덮기 위해, 호출 직후 값을 가져가도록 설계하려면
        # 구조를 바꿔야 한다. 여기선 _follow_path 호출 직후 return을 하지 않고,
        # act() 말미의 기본 반환을 (self._last_cmd)로 해도 된다(테스트용)
        pass

    def reset(self):
        self.path = []
        self.wp_idx = 0
        self.phase = "GO_ROOM"
        self.cover_idx = 0
        self.clean_hold = 0

if __name__ == '__main__':
    ros_test = Agent()
    ros_test._a_star([3,4],[10,8])
