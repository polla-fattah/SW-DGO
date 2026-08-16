"""
Baseline: Optimal Reciprocal Collision Avoidance (ORCA / RVO).
Implements van den Berg et al. (2008) Velocity Obstacles and Half-Plane Linear Programming constraints.
Mathematically models the distinct ORCA failure mode in single-file corridors:
Constraint Infeasibility / Velocity Oscillation (mutual stop v -> 0).
"""

from __future__ import annotations
import math
import time
from math import hypot as math_hypot, sqrt as math_sqrt, atan2 as math_atan2, cos as math_cos, sin as math_sin
from typing import List, Tuple, Optional
from ..core.human import Human
from ..core.metrics import init_social_metrics, update_social_metrics
from ..core.units import (ARRIVAL_RADIUS_PX, 
    PX_TO_M, M_TO_PX, ROBOT_RADIUS_PX, ROBOT_VMAX_MPS
)

class ORCAAgent:
    """
    Optimal Reciprocal Collision Avoidance (ORCA) agent.
    Computes reciprocal velocity obstacle half-planes and selects admissible velocity
    closest to preferred velocity v_pref subject to linear half-plane constraints.
    """
    def __init__(self, agent_id: int, start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                 max_speed: float = ROBOT_VMAX_MPS * M_TO_PX,  # ~40 px/s (1.2 m/s)
                 time_horizon: float = 2.0, radius: float = ROBOT_RADIUS_PX):
        self.agent_id = agent_id
        self.x, self.y = start_pos
        self.goal_x, self.goal_y = goal_pos
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.max_speed = max_speed
        self.time_horizon = time_horizon
        self.radius = radius
        self.heading: float = 0.0

        # Performance Metrics
        self.total_distance: float = 0.0
        self.travel_time: float = 0.0
        self.deadlock_count: int = 0
        init_social_metrics(self)
        self.head_on_events: int = 0
        self.is_docked: bool = False

    @property
    def current_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def _compute_orca_halfplane(self, rel_pos: Tuple[float, float], rel_vel: Tuple[float, float],
                                combined_radius: float, time_horizon: float, reciprocity: float = 0.5
                                ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Computes the ORCA linear constraint half-plane (point on line p_line, unit normal n)
        for an obstacle/peer at relative position rel_pos.
        """
        px, py = rel_pos
        vx, vy = rel_vel
        dist = math_hypot(px, py)

        inv_tau = 1.0 / time_horizon

        # Vector pointing toward relative position center
        if dist > combined_radius:
            # Cone projection
            w_x = vx - inv_tau * px
            w_y = vy - inv_tau * py
            w_len = math_hypot(w_x, w_y)
            
            # Normal to cone
            dot_product = w_x * px + w_y * py
            if dot_product < 0.0 and (dot_product * dot_product) > (dist * dist * w_len * w_len * (combined_radius / dist) ** 2):
                # Project on cut-off circle
                unit_w_x = w_x / max(1e-5, w_len)
                unit_w_y = w_y / max(1e-5, w_len)
                u_x = (combined_radius * inv_tau - w_len) * unit_w_x
                u_y = (combined_radius * inv_tau - w_len) * unit_w_y
                n_x = unit_w_x
                n_y = unit_w_y
            else:
                # Project on legs of cone
                leg_len = math_sqrt(max(0.0, dist * dist - combined_radius * combined_radius))
                if (px * vy - py * vx) > 0.0:
                    # Left leg
                    n_x = (px * leg_len - py * combined_radius) / (dist * dist)
                    n_y = (py * leg_len + px * combined_radius) / (dist * dist)
                else:
                    # Right leg
                    n_x = -(px * leg_len + py * combined_radius) / (dist * dist)
                    n_y = -(py * leg_len - px * combined_radius) / (dist * dist)
                
                u_x = -(vx * n_x + vy * n_y) * n_x
                u_y = -(vx * n_x + vy * n_y) * n_y
        else:
            # Collision already occurring / severe penetration
            inv_time_step = 1.0 / 0.05
            w_x = vx - inv_time_step * px
            w_y = vy - inv_time_step * py
            w_len = math_hypot(w_x, w_y)
            unit_w_x = w_x / max(1e-5, w_len)
            unit_w_y = w_y / max(1e-5, w_len)
            u_x = (combined_radius * inv_time_step - w_len) * unit_w_x
            u_y = (combined_radius * inv_time_step - w_len) * unit_w_y
            n_x = unit_w_x
            n_y = unit_w_y

        # Line point: v_A + reciprocity * u
        point_x = self.vx + reciprocity * u_x
        point_y = self.vy + reciprocity * u_y

        return ((point_x, point_y), (n_x, n_y))

    def step(self, dt: float, peer_positions: Optional[List[Tuple[float, float]]] = None,
             humans: Optional[List[Human]] = None,
             shelf_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
             peer_agents: Optional[List[ORCAAgent]] = None) -> None:
        if self.is_docked:
            return

        t0 = time.perf_counter()
        self.travel_time += dt
        humans = humans or []
        shelf_bounds = shelf_bounds or []

        # 1. Preferred Velocity toward Goal: v_pref
        dx_goal = self.goal_x - self.x
        dy_goal = self.goal_y - self.y
        dist_goal = math_hypot(dx_goal, dy_goal)

        if dist_goal < ARRIVAL_RADIUS_PX:
            self.is_docked = True
            self.vx = 0.0
            self.vy = 0.0
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        v_pref_x = (dx_goal / dist_goal) * self.max_speed
        v_pref_y = (dy_goal / dist_goal) * self.max_speed

        orca_lines = []

        # 2. Build ORCA Half-Planes from Peer Trolleys (Reciprocity = 0.5)
        if peer_agents:
            for peer in peer_agents:
                if peer.agent_id == self.agent_id or peer.is_docked:
                    continue
                rel_pos = (peer.x - self.x, peer.y - self.y)
                rel_vel = (self.vx - peer.vx, self.vy - peer.vy)
                combined_r = self.radius * 2.2
                if 0.001 < math_hypot(rel_pos[0], rel_pos[1]) < 80.0:
                    line = self._compute_orca_halfplane(rel_pos, rel_vel, combined_r, self.time_horizon, reciprocity=0.5)
                    orca_lines.append(line)
        elif peer_positions:
            for px, py in peer_positions:
                rel_pos = (px - self.x, py - self.y)
                d_peer = math_hypot(rel_pos[0], rel_pos[1])
                if 0.001 < d_peer < 80.0:  # Exclude self
                    rel_vel = (self.vx, self.vy)
                    combined_r = self.radius * 2.2
                    line = self._compute_orca_halfplane(rel_pos, rel_vel, combined_r, self.time_horizon, reciprocity=0.5)
                    orca_lines.append(line)

        # Social compliance measured by the shared helper (identical threshold and
        # identical per-human semantics as every other planner).
        update_social_metrics(self, humans, dt)

        # 3. Build ORCA Half-Planes from Dynamic Humans (Reciprocity = 1.0, non-reciprocating)
        for human in humans:
            d_h = math_hypot(self.x - human.x, self.y - human.y)

            if d_h < 70.0:
                rel_pos = (human.x - self.x, human.y - self.y)
                rel_vel = (self.vx - human.vx, self.vy - human.vy)
                combined_r = self.radius + human.radius + 10.0
                line = self._compute_orca_halfplane(rel_pos, rel_vel, combined_r, self.time_horizon, reciprocity=1.0)
                orca_lines.append(line)

        # 4. Build ORCA Half-Planes from Static Shelf Fixtures (Reciprocity = 1.0, stationary)
        for min_x, min_y, max_x, max_y in shelf_bounds:
            cx = min_x if self.x < min_x else (max_x if self.x > max_x else self.x)
            cy = min_y if self.y < min_y else (max_y if self.y > max_y else self.y)
            obs_dist = math_hypot(self.x - cx, self.y - cy)
            if obs_dist < 40.0:
                rel_pos = (cx - self.x, cy - self.y)
                rel_vel = (self.vx, self.vy)
                combined_r = self.radius + 6.0
                line = self._compute_orca_halfplane(rel_pos, rel_vel, combined_r, time_horizon=1.0, reciprocity=1.0)
                orca_lines.append(line)

        # 5. 2D Linear Program: Select v_new satisfying all half-planes (v - p) . n >= 0
        best_vx, best_vy = v_pref_x, v_pref_y
        feasible = True

        for (px, py), (nx, ny) in orca_lines:
            if ((best_vx - px) * nx + (best_vy - py) * ny) < 0.0:
                dot = (px - best_vx) * nx + (py - best_vy) * ny
                best_vx += dot * nx
                best_vy += dot * ny

        # Check speed constraint ||v|| <= max_speed
        best_speed = math_hypot(best_vx, best_vy)
        if best_speed > self.max_speed:
            best_vx = (best_vx / best_speed) * self.max_speed
            best_vy = (best_vy / best_speed) * self.max_speed
            best_speed = self.max_speed

        # Verify whether all constraints are satisfied
        for (px, py), (nx, ny) in orca_lines:
            if ((best_vx - px) * nx + (best_vy - py) * ny) < -1.0:
                feasible = False
                break

        # Failure Mode: Constraint Infeasibility in narrow corridors (v -> 0 stop/oscillation)
        if not feasible or best_speed < 1.0:
            if dist_goal > 30.0:
                self.stalled_ticks += 1
            self.vx = 0.0
            self.vy = 0.0
        else:
            self.vx = best_vx
            self.vy = best_vy
            self.heading = math_atan2(self.vy, self.vx)

        step_dist = math_hypot(self.vx * dt, self.vy * dt)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.total_distance += step_dist
        self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
