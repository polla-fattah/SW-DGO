"""
Baseline 1: Static A* Planner without Mesh Communication or Human Proxemics.
Only plans on baseline physical distances D(u, v) and does not receive V2V mesh updates.
"""

from __future__ import annotations
import math
import time
import heapq
from typing import Dict, List, Tuple, Optional
from ..core.graph import TopologicalGraph
from ..core.human import Human, ProxemicsField
from ..core.metrics import init_social_metrics, update_social_metrics
from ..core.units import (
    PX_TO_M, M_TO_PX, ROBOT_RADIUS_PX, ROBOT_VMAX_MPS
)

class StaticAStarAgent:
    """
    Trolley agent using traditional static A* routing.
    Lacks mesh communication and human proxemic awareness.
    """
    def __init__(self, agent_id: int, graph: TopologicalGraph, start_node: str, goal_node: str,
                 max_speed: float = ROBOT_VMAX_MPS * M_TO_PX):  # ~40 px/s (1.2 m/s)
        self.agent_id = agent_id
        self.graph = graph
        self.current_node = start_node
        self.goal_node = goal_node
        self.max_speed = max_speed

        node_obj = self.graph.get_node(start_node)
        self.x: float = node_obj.x
        self.y: float = node_obj.y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.heading: float = 0.0

        # Plan static path
        self.path = self._plan_astar(start_node, goal_node)
        self.path_index = 0

        # Metrics
        self.total_distance: float = 0.0
        self.travel_time: float = 0.0
        self.deadlock_count: int = 0
        init_social_metrics(self)
        self.head_on_events: int = 0
        self.is_docked: bool = False
        self.last_compute_time_ms: float = 0.0

    @property
    def current_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def _plan_astar(self, start: str, goal: str) -> List[str]:
        open_set: List[Tuple[float, int, str]] = []
        heapq.heappush(open_set, (0.0, 0, start))
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {node: math.inf for node in self.graph.nodes}
        g_score[start] = 0.0
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.graph.successors(current):
                edge = self.graph.get_edge(current, neighbor)
                d = edge.d if edge else self.graph.distance(current, neighbor)
                tentative_g = g_score[current] + d
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.graph.distance(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return [start]

    def step(self, dt: float, humans: List[Human], prox_field: ProxemicsField,
             current_sim_time: float) -> None:
        if self.is_docked:
            return

        t0 = time.perf_counter()
        self.travel_time += dt

        # Social compliance measured by the shared helper (identical threshold
        # and identical per-human semantics as every other planner).
        update_social_metrics(self, humans, dt)

        if self.path_index >= len(self.path) - 1:
            self.is_docked = True
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        target_node_id = self.path[self.path_index + 1]
        target_obj = self.graph.get_node(target_node_id)
        dx = target_obj.x - self.x
        dy = target_obj.y - self.y
        dist = math.hypot(dx, dy)

        if dist < 6.0:
            self.path_index += 1
            self.current_node = target_node_id
            if self.current_node == self.goal_node:
                self.is_docked = True
        else:
            self.heading = math.atan2(dy, dx)
            step_dist = min(dist, self.max_speed * dt)
            self.x += math.cos(self.heading) * step_dist
            self.y += math.sin(self.heading) * step_dist
            self.total_distance += step_dist

        self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
