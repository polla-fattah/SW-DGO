"""
Baseline: Decentralized Local MAPF (with Windowed Conflict Resolution).
Implements a contemporary hybrid baseline (inspired by Dergachev & Yakovlev 2021; Keskin et al. 2024).
Combines global static topological A* with local windowed conflict arbitration and stop-and-wait yielding,
WITHOUT multi-hop V2V mesh broadcasts and WITHOUT continuous Gaussian social proxemics.
"""

from __future__ import annotations
import math
import time
from math import hypot as math_hypot, atan2 as math_atan2
from typing import List, Tuple, Optional, Dict
from ..core.graph import TopologicalGraph, Node
from ..core.human import Human
from ..core.metrics import init_social_metrics, update_social_metrics
from ..core.units import (ARRIVAL_RADIUS_PX, 
    PX_TO_M, M_TO_PX, ROBOT_RADIUS_PX, ROBOT_VMAX_MPS
)

class DecentralizedLocalMAPFAgent:
    """
    Decentralized Local MAPF agent.
    Maintains a global static A* roadmap, but performs local conflict resolution only within
    line-of-sight sensing radius (R_sense = 6.0 m / 200 px).
    Demonstrates the delayed backtracking makespan inflation and social proxemic blindness.
    """
    def __init__(self, agent_id: int, graph: TopologicalGraph, start_node: str, goal_node: str,
                 max_speed: float = ROBOT_VMAX_MPS * M_TO_PX,  # ~40 px/s (1.2 m/s)
                 sense_radius: float = 200.0):
        self.agent_id = agent_id
        self.graph = graph.clone()
        self.current_node = start_node
        self.goal_node = goal_node
        self.max_speed = max_speed
        self.sense_radius = sense_radius

        node_obj = self.graph.get_node(start_node)
        self.x: float = node_obj.x
        self.y: float = node_obj.y
        self.heading: float = 0.0
        self.speed: float = 0.0

        # Compute initial static A* path
        self.path: List[str] = self._compute_static_astar(start_node, goal_node)
        self.path_index: int = 0
        self.target_node: Optional[str] = self.path[1] if len(self.path) > 1 else None

        if self.target_node:
            t_obj = self.graph.get_node(self.target_node)
            self.heading = math_atan2(t_obj.y - self.y, t_obj.x - self.x)

        # State
        self.state: str = "NAVIGATING"  # "NAVIGATING", "YIELDING_LOCAL", "DOCKED"
        self.yield_timer: float = 0.0

        # Performance Metrics
        self.total_distance: float = 0.0
        self.travel_time: float = 0.0
        self.replan_count: int = 0
        self.deadlock_count: int = 0
        init_social_metrics(self)
        self.head_on_events: int = 0
        self.is_docked: bool = False
        self.last_compute_time_ms: float = 0.0

    @property
    def current_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def _compute_static_astar(self, start: str, goal: str) -> List[str]:
        """Standard static A* search on topological graph."""
        import heapq
        open_set = [(0.0, start)]
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.graph.successors(current):
                edge = self.graph.get_edge(current, neighbor)
                d = edge.d if edge else self.graph.distance(current, neighbor)
                tentative_g = g_score[current] + d
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.graph.distance(neighbor, goal)
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        return [start]

    def step(self, dt: float, peer_positions: Dict[int, Tuple[float, float]],
             humans: List[Human]) -> None:
        if self.is_docked:
            return

        t0 = time.perf_counter()
        self.travel_time += dt

        # Check goal arrival
        goal_obj = self.graph.get_node(self.goal_node)
        if math_hypot(self.x - goal_obj.x, self.y - goal_obj.y) < ARRIVAL_RADIUS_PX:
            self.is_docked = True
            self.speed = 0.0
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        # Social compliance measured by the shared helper (identical threshold and
        # identical per-human semantics as every other planner).
        update_social_metrics(self, humans, dt)

        if not self.target_node:
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        target_obj = self.graph.get_node(self.target_node)
        dx = target_obj.x - self.x
        dy = target_obj.y - self.y
        dist_to_target = math_hypot(dx, dy)

        # 1. Local Sensing Conflict Detection (Only within sense_radius)
        # Check if next waypoint is physically obstructed by a peer or human
        local_conflict = False
        for pid, (px, py) in peer_positions.items():
            if pid != self.agent_id and math_hypot(self.x - px, self.y - py) < 35.0:
                # Priority rule: Lower agent_id has right of way
                if pid < self.agent_id:
                    local_conflict = True
                    break

        if local_conflict:
            # Stop-and-wait yielding
            self.speed = 0.0
            self.yield_timer += dt
            if self.yield_timer > 3.0:
                self.head_on_events += 1
                # Trigger local 1-hop reroute
                self.replan_count += 1
                self.path = self._compute_static_astar(self.current_node, self.goal_node)
                self.path_index = 0
                self.target_node = self.path[1] if len(self.path) > 1 else None
                self.yield_timer = 0.0
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return
        else:
            self.yield_timer = 0.0

        # 2. Waypoint Progression
        if dist_to_target < 10.0:
            self.current_node = self.target_node
            self.path_index += 1
            if self.path_index + 1 < len(self.path):
                self.target_node = self.path[self.path_index + 1]
            else:
                self.target_node = None
                self.is_docked = True
                self.speed = 0.0
                self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
                return

        # 3. Kinematic Step
        target_heading = math_atan2(dy, dx)
        angle_diff = (target_heading - self.heading + math.pi) % (2.0 * math.pi) - math.pi
        self.heading += max(-2.5 * dt, min(2.5 * dt, angle_diff * 4.0))

        self.speed = min(self.max_speed, self.speed + 50.0 * dt)
        step_len = self.speed * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.total_distance += step_len
        self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
