"""
Human Pedestrian Model and Shelf-Aware Navigation with True Anisotropic Gaussian Proxemics.
Models dynamic shoppers that strictly respect supermarket shelf boundaries,
navigate through aisles/crossways, and possess directional asymmetric Gaussian personal-space fields.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from d2ro.core.units import (
    PX_TO_M, M_TO_PX, HUMAN_RADIUS_PX,
    PROXEMIC_SIGMA_FRONT_PX, PROXEMIC_SIGMA_SIDE_PX, PROXEMIC_SIGMA_REAR_PX, PROXEMIC_AMPLITUDE
)

@dataclass
class Human:
    """Represents a human shopper navigating retail aisles with directional orientation."""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0     # Facing orientation in radians
    speed: float = 1.0
    state: str = "browsing"  # "browsing" or "walking"
    state_timer: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    radius: float = HUMAN_RADIUS_PX  # Physical body radius in pixels (0.36 m)

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def update(self, dt: float, bounds: Tuple[float, float, float, float],
               shelves: Optional[List[Tuple[float, float, float, float]]] = None,
               aisle_x_coords: Optional[List[float]] = None,
               crossway_y_coords: Optional[List[float]] = None) -> None:
        """
        Updates pedestrian state, kinematics, and facing heading while ensuring pedestrians
        strictly navigate within open corridors and never penetrate solid shelf fixtures.
        """
        self.state_timer -= dt
        min_x, min_y, max_x, max_y = bounds

        if self.state_timer <= 0:
            if self.state == "browsing":
                self.state = "walking"
                self.state_timer = random.uniform(3.0, 7.0)
                if aisle_x_coords and crossway_y_coords and random.random() < 0.6:
                    self.target_x = random.choice(aisle_x_coords)
                    self.target_y = random.uniform(crossway_y_coords[0], crossway_y_coords[-1])
                elif aisle_x_coords and crossway_y_coords:
                    self.target_x = random.uniform(aisle_x_coords[0], aisle_x_coords[-1])
                    self.target_y = random.choice(crossway_y_coords)
                else:
                    self.target_x = random.uniform(min_x + 50, max_x - 50)
                    self.target_y = random.uniform(min_y + 50, max_y - 50)
            else:
                self.state = "browsing"
                self.state_timer = random.uniform(2.0, 5.0)
                self.vx = 0.0
                self.vy = 0.0
                # When browsing, turn toward nearest shelf to face products
                if shelves:
                    min_dist = float('inf')
                    for sx1, sy1, sx2, sy2 in shelves:
                        cx = (sx1 + sx2) / 2.0
                        cy = (sy1 + sy2) / 2.0
                        d = math.hypot(self.x - cx, self.y - cy)
                        if d < min_dist:
                            min_dist = d
                            self.heading = math.atan2(cy - self.y, cx - self.x)

        if self.state == "walking":
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = math.hypot(dx, dy)
            speed_pxps = self.speed * M_TO_PX  # 1.0 m/s ~ 33.3 px/s
            if dist > 6.0:
                self.vx = (dx / dist) * speed_pxps
                self.vy = (dy / dist) * speed_pxps
                self.heading = math.atan2(self.vy, self.vx)
                self.x += self.vx * dt
                self.y += self.vy * dt
            else:
                self.state = "browsing"
                self.state_timer = random.uniform(2.0, 4.0)

        # 1. Hard Shelf Collision Resolution (Pedestrians cannot penetrate solid shelves)
        if shelves:
            for min_sx, min_sy, max_sx, max_sy in shelves:
                expanded_min_x = min_sx - self.radius
                expanded_max_x = max_sx + self.radius
                expanded_min_y = min_sy - self.radius
                expanded_max_y = max_sy + self.radius

                if (expanded_min_x <= self.x <= expanded_max_x and
                    expanded_min_y <= self.y <= expanded_max_y):
                    d_left = self.x - expanded_min_x
                    d_right = expanded_max_x - self.x
                    d_top = self.y - expanded_min_y
                    d_bottom = expanded_max_y - self.y

                    # Find minimum distance to push out
                    min_d = d_left
                    if d_right < min_d: min_d = d_right
                    if d_top < min_d: min_d = d_top
                    if d_bottom < min_d: min_d = d_bottom

                    if min_d == d_left:
                        self.x = expanded_min_x
                        self.vx = -abs(self.vx)
                    elif min_d == d_right:
                        self.x = expanded_max_x
                        self.vx = abs(self.vx)
                    elif min_d == d_top:
                        self.y = expanded_min_y
                        self.vy = -abs(self.vy)
                    else:
                        self.y = expanded_max_y
                        self.vy = abs(self.vy)

        # 2. Store Perimeter Boundary Clamping
        if self.x < min_x + 15.0:
            self.x = min_x + 15.0
        elif self.x > max_x - 15.0:
            self.x = max_x - 15.0

        if self.y < min_y + 15.0:
            self.y = min_y + 15.0
        elif self.y > max_y - 15.0:
            self.y = max_y - 15.0


class ProxemicsField:
    """
    Computes true Asymmetric Anisotropic 2D Gaussian personal-space discomfort fields
    based on Hall's Proxemics and HA-VLN 2.0 guidelines.
    
    Front personal space: sigma_front = 1.35 m (45 px)
    Lateral personal space: sigma_side = 0.90 m (30 px)
    Rear personal space: sigma_rear = 0.60 m (20 px)
    """
    def __init__(self, amplitude: float = PROXEMIC_AMPLITUDE,
                 sigma_front: float = PROXEMIC_SIGMA_FRONT_PX,
                 sigma_side: float = PROXEMIC_SIGMA_SIDE_PX,
                 sigma_rear: float = PROXEMIC_SIGMA_REAR_PX):
        self.amplitude = amplitude
        self.sigma_front = sigma_front
        self.sigma_side = sigma_side
        self.sigma_rear = sigma_rear

    def compute_penalty_at_point(self, x: float, y: float, humans: Any) -> float:
        """
        Evaluates the asymmetric anisotropic 2D Gaussian discomfort at coordinate (x, y)
        transformed into each pedestrian's local body orientation frame.
        """
        if humans is None:
            return 0.0
        if isinstance(humans, list):
            human_list = humans
        elif isinstance(humans, (tuple, set)):
            human_list = list(humans)
        else:
            human_list = [humans]

        sig_f = self.sigma_front
        sig_r = self.sigma_rear
        sig_s = self.sigma_side
        amp = self.amplitude

        total_penalty = 0.0
        max_cutoff = 3.5 * sig_f
        max_cutoff_sq = max_cutoff * max_cutoff

        for h in human_list:
            if not (hasattr(h, 'x') and hasattr(h, 'y') and hasattr(h, 'heading')):
                continue
            try:
                hx = float(h.x)
                hy = float(h.y)
                hheading = float(h.heading)
            except Exception:
                continue

            dx = x - hx
            dy = y - hy
            raw_dist_sq = dx * dx + dy * dy

            if raw_dist_sq > max_cutoff_sq:
                continue

            cos_h = math.cos(hheading)
            sin_h = math.sin(hheading)
            x_ego = -sin_h * dx + cos_h * dy
            y_ego = cos_h * dx + sin_h * dy

            sigma_long = sig_f if y_ego >= 0.0 else sig_r
            sigma_lat = sig_s

            term1 = x_ego / sigma_lat
            term2 = y_ego / sigma_long
            exponent = -0.5 * (term1 * term1 + term2 * term2)
            if exponent > -12.0:
                total_penalty += amp * math.exp(exponent)

        return total_penalty

    def compute_edge_segment_penalty(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                     humans: Any, num_samples: int = 6) -> float:
        r"""
        Integrates the continuous 2D anisotropic Gaussian discomfort field along corridor segment (p1 -> p2).
        Uses trapezoidal numerical quadrature over segment length L (meters):
        \int_0^L H_prox(s) ds \approx (L / N) * (0.5*H_0 + 0.5*H_N + sum_{i=1}^{N-1} H_i)
        """
        if not humans:
            return 0.0

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        seg_len_px = math.hypot(x2 - x1, y2 - y1)
        seg_len_m = seg_len_px * PX_TO_M

        if seg_len_px < 1e-5:
            return self.compute_penalty_at_point(x1, y1, humans) * PX_TO_M

        H_sum = 0.0
        for step_idx in range(num_samples + 1):
            t_val = step_idx / float(num_samples)
            sx = (1.0 - t_val) * x1 + t_val * x2
            sy = (1.0 - t_val) * y1 + t_val * y2
            pt_penalty = self.compute_penalty_at_point(sx, sy, humans)
            weight = 0.5 if (step_idx == 0 or step_idx == num_samples) else 1.0
            H_sum += weight * pt_penalty

        integral = (seg_len_m / float(num_samples)) * H_sum
        return integral
