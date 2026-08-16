"""
Baseline: Classical Artificial Potential Fields (APF).
Implements Khatib (1986) continuous potential field navigation with attractive goal forces
and repulsive obstacle forces from static shelf walls, dynamic pedestrians, and peer robots.
Mathematically models the classical failure mode: Local Potential Minima Traps in concave 90° corners.
"""

from __future__ import annotations
import math
import time
from math import hypot as math_hypot, exp as math_exp
from typing import List, Tuple, Optional
from ..core.human import Human
from ..core.metrics import init_social_metrics, update_social_metrics
from ..core.units import (ARRIVAL_RADIUS_PX, 
    PX_TO_M, M_TO_PX, ROBOT_RADIUS_PX, ROBOT_VMAX_MPS
)

class ArtificialPotentialFieldAgent:
    """
    Classical Artificial Potential Fields (APF) reactive agent.
    Steers along the negative gradient of the total potential field:
    U_total(p) = U_att(p) + sum(U_rep_shelves) + sum(U_rep_humans) + sum(U_rep_peers)
    F_net = -grad(U_total)
    """
    def __init__(self, agent_id: int, start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                 max_speed: float = ROBOT_VMAX_MPS * M_TO_PX,  # ~40 px/s (1.2 m/s)
                 k_att: float = 1.0, k_rep_obs: float = 350.0, k_rep_human: float = 450.0,
                 d0_shelf: float = 30.0, d0_human: float = 40.0):
        self.agent_id = agent_id
        self.x, self.y = start_pos
        self.goal_x, self.goal_y = goal_pos
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.max_speed = max_speed
        self.heading: float = 0.0

        # Potential field parameters
        self.k_att = k_att
        self.k_rep_obs = k_rep_obs
        self.k_rep_human = k_rep_human
        self.d0_shelf = d0_shelf   # Influence distance for static shelves (~0.9 m)
        self.d0_human = d0_human   # Influence distance for humans (~1.2 m)

        # Performance Metrics
        self.total_distance: float = 0.0
        self.travel_time: float = 0.0
        self.deadlock_count: int = 0
        init_social_metrics(self)
        self.head_on_events: int = 0
        self.shelf_corner_scrapes: int = 0
        self.is_docked: bool = False
        self.last_compute_time_ms: float = 0.0

    @property
    def current_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def step(self, dt: float, peer_positions: Optional[List[Tuple[float, float]]] = None,
             humans: Optional[List[Human]] = None,
             shelf_bounds: Optional[List[Tuple[float, float, float, float]]] = None) -> None:
        if self.is_docked:
            return

        t0 = time.perf_counter()
        self.travel_time += dt
        peer_positions = peer_positions or []
        humans = humans or []
        shelf_bounds = shelf_bounds or []

        # 1. Attractive Force toward Goal: F_att = -k_att * (p - p_goal)
        dx_goal = self.goal_x - self.x
        dy_goal = self.goal_y - self.y
        dist_goal = math_hypot(dx_goal, dy_goal)

        if dist_goal < ARRIVAL_RADIUS_PX:
            self.is_docked = True
            self.vx = 0.0
            self.vy = 0.0
            return

        # Normalized attractive vector scaled by max speed
        f_att_x = (dx_goal / dist_goal) * self.max_speed * self.k_att
        f_att_y = (dy_goal / dist_goal) * self.max_speed * self.k_att

        f_rep_x = 0.0
        f_rep_y = 0.0

        # 2. Repulsive Force from Static Shelf Boundaries
        for min_x, min_y, max_x, max_y in shelf_bounds:
            cx = min_x if self.x < min_x else (max_x if self.x > max_x else self.x)
            cy = min_y if self.y < min_y else (max_y if self.y > max_y else self.y)
            d_obs = math_hypot(self.x - cx, self.y - cy)

            if 0.001 < d_obs < self.d0_shelf:
                # F_rep = k_rep * (1/d - 1/d0) * (1/d^2) * (p - p_obs)/d
                rep_mag = self.k_rep_obs * (1.0 / d_obs - 1.0 / self.d0_shelf) * (1.0 / (d_obs * d_obs))
                rep_mag = min(rep_mag, self.max_speed * 3.0)  # Numerical clamp
                f_rep_x += ((self.x - cx) / d_obs) * rep_mag
                f_rep_y += ((self.y - cy) / d_obs) * rep_mag

        # Social compliance measured by the shared helper (identical threshold and
        # identical per-human semantics as every other planner).
        update_social_metrics(self, humans, dt)

        # 3. Repulsive Force from Dynamic Human Pedestrians
        for human in humans:
            d_h = math_hypot(self.x - human.x, self.y - human.y)

            if 0.001 < d_h < self.d0_human:
                rep_mag = self.k_rep_human * (1.0 / d_h - 1.0 / self.d0_human) * (1.0 / (d_h * d_h))
                rep_mag = min(rep_mag, self.max_speed * 3.0)
                f_rep_x += ((self.x - human.x) / d_h) * rep_mag
                f_rep_y += ((self.y - human.y) / d_h) * rep_mag

        # 4. Repulsive Force from Peer Trolleys (Reynolds Separation)
        for px, py in peer_positions:
            d_peer = math_hypot(self.x - px, self.y - py)
            if 0.001 < d_peer < 35.0:
                rep_mag = self.max_speed * 2.0 * ((35.0 - d_peer) / 35.0)
                f_rep_x += ((self.x - px) / d_peer) * rep_mag
                f_rep_y += ((self.y - py) / d_peer) * rep_mag

        # Net Force Vector
        fx_net = f_att_x + f_rep_x
        fy_net = f_att_y + f_rep_y
        f_mag = math_hypot(fx_net, fy_net)

        # Failure Mode: Local Potential Minimum Trap (F_net -> 0 while far from goal)
        if f_mag < 1.5 and dist_goal > 30.0:
            self.stalled_ticks += 1
            # In local minimum trap, velocity decays to near zero
            self.vx = 0.0
            self.vy = 0.0
        else:
            # Velocity update
            if f_mag > self.max_speed:
                fx_net = (fx_net / f_mag) * self.max_speed
                fy_net = (fy_net / f_mag) * self.max_speed

            self.vx = fx_net
            self.vy = fy_net
            self.heading = math.atan2(self.vy, self.vx)

        step_dist = math_hypot(self.vx * dt, self.vy * dt)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.total_distance += step_dist
        self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
