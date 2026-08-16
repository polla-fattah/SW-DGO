"""
Topological Graph Representation for SW-DGO Framework.
Models nodes (aisle junctions/waypoints) and directed edges carrying four weighted
soft cost terms minimised subject to one hard reservation constraint:

    C(u, v, t) = w_D * D(u, v) + w_M * W_mesh(u, v, t)
               + w_H * H_prox(v, t) + w_S * S_trolley(v, t)
    subject to (u, v) not reserved by a preceding peer

Reservation is a feasibility constraint rather than a weighted term: a reserved edge
is unavailable at any coefficient, and an unreserved one contributes zero. See the
note on Edge.weight_r for why the field is retained as a documented no-op.
"""

from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

from d2ro.core.units import (
    WEIGHT_DISTANCE_WD, WEIGHT_MESH_WM, WEIGHT_PROXEMIC_WH,
    WEIGHT_MUTEX_LOCK_WR, WEIGHT_TROLLEY_WS, PX_TO_M,
    ROBOT_RADIUS_PX, SHELF_CLEARANCE_MARGIN_PX, TROLLEY_CLEARANCE_AMPLITUDE
)

@dataclass
class Node:
    """Represents a topological waypoint in the service environment."""
    id: str
    x: float
    y: float
    is_docking_bay: bool = False
    is_charging_station: bool = False

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def pos_m(self) -> Tuple[float, float]:
        """Returns coordinate in meters (SI metric)."""
        return (self.x * PX_TO_M, self.y * PX_TO_M)

@dataclass
class Edge:
    """
    Directed edge carrying the four weighted soft cost terms of SW-DGO:
    C(u, v, t) = w_D * D(u, v) + w_M * W_mesh(u, v, t) + w_H * H_prox(v, t)
               + w_S * S_trolley
    A reserved edge is infeasible and returns infinite cost before these apply.
    """
    u: str
    v: str
    d: float  # Baseline physical Euclidean distance / traversal length
    is_single_file: bool = False  # True if corridor cannot accommodate bidirectional traffic
    w_mesh: float = 0.0      # Distributed mesh congestion penalty
    h_prox: float = 0.0      # Asymmetric human proxemic discomfort penalty
    r_lock: float = 0.0      # Directional corridor lock penalty (0.0 or math.inf)
    s_trolley: float = 0.0   # Kinetic vehicle clearance envelope penalty (static + dynamic)
    s_clearance: float = 0.0 # Static geometric component of S_trolley (fixture clearance)

    # Weight coefficients
    weight_d: float = WEIGHT_DISTANCE_WD
    weight_m: float = WEIGHT_MESH_WM
    weight_h: float = WEIGHT_PROXEMIC_WH
    # DEPRECATED and unused: reservation is a feasibility constraint, not a weighted
    # term (see `cost`). Retained only so that existing constructor calls do not
    # break; it has no effect on any cost and must not be reported as a weight.
    weight_r: float = WEIGHT_MUTEX_LOCK_WR
    weight_s: float = WEIGHT_TROLLEY_WS

    # Active lock metadata
    lock_owner: Optional[int] = None  # Agent ID holding current lock
    lock_expiry: float = 0.0          # Simulation timestamp when lock expires

    @property
    def cost(self) -> float:
        r"""
        Composite traversal cost: four weighted soft terms, subject to one hard
        feasibility constraint.

            minimise  w_D*d + w_M*w_mesh + w_H*h_prox + w_S*s_trolley
            subject to e not in E_reserved(t)

        The reservation is a CONSTRAINT, not a weighted objective term. An earlier
        formulation wrote it as w_R * r_lock and described w_R as the fifth tunable
        weight, but r_lock only ever takes the values 0 or infinity: when it is
        infinity the edge is unavailable and no finite multiplier changes that, and
        when it is 0 the product is 0 for every w_R. The coefficient was therefore
        inert in every reachable state, which is why perturbing it moved no measured
        quantity. Modelling reservation as feasibility rather than as cost states
        what the code has always done.
        """
        if self.r_lock == math.inf:
            return math.inf
        try:
            total_cost = (
                float(self.weight_d) * float(self.d) +
                float(self.weight_m) * float(self.w_mesh) +
                float(self.weight_h) * float(self.h_prox) +
                float(self.weight_s) * float(self.s_trolley)
            )
            return max(0.001, total_cost)
        except (TypeError, ValueError):
            return max(0.001, float(self.weight_d) * float(self.d))

class TopologicalGraph:
    """
    Directed graph modeling indoor facility topology with dynamic edge costs.
    Supports predecessor/successor lookups for incremental D* Lite heuristic search.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self._succ: Dict[str, List[str]] = {}
        self._pred: Dict[str, List[str]] = {}

    def clone(self) -> TopologicalGraph:
        """Creates an independent deep copy of the topological graph for decentralized agent memory."""
        new_g = TopologicalGraph()
        for node_id, node in self.nodes.items():
            new_g.nodes[node_id] = Node(
                id=node.id, x=node.x, y=node.y,
                is_docking_bay=node.is_docking_bay,
                is_charging_station=node.is_charging_station
            )
            new_g._succ[node_id] = list(self._succ.get(node_id, []))
            new_g._pred[node_id] = list(self._pred.get(node_id, []))

        for key, edge in self.edges.items():
            new_g.edges[key] = Edge(
                u=edge.u, v=edge.v, d=edge.d,
                is_single_file=edge.is_single_file,
                w_mesh=edge.w_mesh, h_prox=edge.h_prox,
                r_lock=edge.r_lock, s_trolley=edge.s_trolley,
                s_clearance=edge.s_clearance,
                weight_d=edge.weight_d, weight_m=edge.weight_m,
                weight_h=edge.weight_h, weight_r=edge.weight_r,
                weight_s=edge.weight_s, lock_owner=edge.lock_owner,
                lock_expiry=edge.lock_expiry
            )
        return new_g

    def add_node(self, node_id: str, x: float, y: float, is_docking_bay: bool = False) -> Node:
        node = Node(id=node_id, x=x, y=y, is_docking_bay=is_docking_bay)
        self.nodes[node_id] = node
        if node_id not in self._succ:
            self._succ[node_id] = []
        if node_id not in self._pred:
            self._pred[node_id] = []
        return node

    def add_edge(self, u: str, v: str, is_single_file: bool = False, bidirectional: bool = True) -> None:
        """Adds a directed edge (or bidirectional pair) between node u and node v."""
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must exist before creating an edge.")

        p_u = self.nodes[u]
        p_v = self.nodes[v]
        dist_px = math.hypot(p_u.x - p_v.x, p_u.y - p_v.y)
        dist_m = dist_px * PX_TO_M  # Edge distance in meters (SI metric)

        # Forward edge
        edge_uv = Edge(u=u, v=v, d=dist_m, is_single_file=is_single_file)
        self.edges[(u, v)] = edge_uv
        if v not in self._succ[u]:
            self._succ[u].append(v)
        if u not in self._pred[v]:
            self._pred[v].append(u)

        if bidirectional:
            edge_vu = Edge(u=v, v=u, d=dist_m, is_single_file=is_single_file)
            self.edges[(v, u)] = edge_vu
            if u not in self._succ[v]:
                self._succ[v].append(u)
            if v not in self._pred[u]:
                self._pred[u].append(v)

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def get_edge(self, u: str, v: str) -> Optional[Edge]:
        return self.edges.get((u, v), None)

    def get_cost(self, u: str, v: str) -> float:
        edge = self.get_edge(u, v)
        if edge is None:
            return math.inf
        return edge.cost

    def successors(self, u: str) -> List[str]:
        return self._succ.get(u, [])

    def predecessors(self, v: str) -> List[str]:
        return self._pred.get(v, [])

    def distance(self, u: str, v: str) -> float:
        """Euclidean distance heuristic between two nodes in meters (SI metric)."""
        n1 = self.nodes[u]
        n2 = self.nodes[v]
        return math.hypot(n1.x - n2.x, n1.y - n2.y) * PX_TO_M

    def update_mesh_penalty(self, u: str, v: str, penalty: float) -> bool:
        """Updates mesh congestion penalty on an edge. Returns True if modified."""
        edge = self.get_edge(u, v)
        if edge and edge.w_mesh != penalty:
            edge.w_mesh = penalty
            return True
        return False

    def update_proxemic_penalty(self, v: str, penalty: float) -> List[Tuple[str, str]]:
        """Updates human proxemic penalty for all incoming edges to node v."""
        changed_edges = []
        for u in self.predecessors(v):
            edge = self.get_edge(u, v)
            if edge and abs(edge.h_prox - penalty) > 1e-3:
                edge.h_prox = penalty
                changed_edges.append((u, v))
        return changed_edges

    def decay_mesh_penalties(self, dt: float, decay_rate: float = 0.1386294) -> List[Tuple[str, str]]:
        """Decays mesh congestion penalties exponentially over time: w_mesh(t + dt) = w_mesh(t) * exp(-decay_rate * dt)."""
        if not isinstance(self.edges, dict):
            return []
        changed_edges = []
        decay_factor = math.exp(-decay_rate * dt)
        for key, edge in list(self.edges.items()):
            if isinstance(edge, Edge) and edge.w_mesh > 0.0:
                old_val = edge.w_mesh
                new_val = edge.w_mesh * decay_factor
                if new_val < 0.01:
                    new_val = 0.0
                edge.w_mesh = new_val
                if math.fabs(old_val - edge.w_mesh) > 0.5:
                    changed_edges.append((edge.u, edge.v))
        return changed_edges

    @staticmethod
    def _clearance_to_rects(px: float, py: float,
                            rects: List[Tuple[float, float, float, float]]) -> float:
        """Euclidean distance (px) from point to the nearest rectangular fixture."""
        best = math.inf
        for x1, y1, x2, y2 in rects:
            cx = x1 if px < x1 else (x2 if px > x2 else px)
            cy = y1 if py < y1 else (y2 if py > y2 else py)
            d = math.hypot(px - cx, py - cy)
            if d < best:
                best = d
        return best

    def compute_clearance_penalties(self, shelves: List[Tuple[float, float, float, float]],
                                    num_samples: int = 8,
                                    amplitude: float = TROLLEY_CLEARANCE_AMPLITUDE) -> None:
        r"""
        Computes the STATIC geometric component of the kinetic safety envelope
        $S_{\text{trolley}}$ for every edge, and seeds ``s_trolley`` with it.

        For each edge the required chassis envelope is
        $r_{\text{req}} = r_{\text{robot}} + \delta_{\text{shelf}}$. The normalised
        clearance deficit at a sample point is
        $\phi(s) = \max\left(0, (r_{\text{req}} - d_{\text{clear}}(s)) / r_{\text{req}}\right)$,
        and the edge penalty is its trapezoidal line integral over the segment length
        $L$ (metres):
        $S^{\text{static}}_{\text{trolley}}(u,v) = A_s \int_0^L \phi(s)\,ds$.

        This makes narrow, fixture-flanked aisles genuinely more expensive to traverse
        at the graph-optimisation level, rather than being corrected only reactively.
        """
        if not shelves:
            return

        r_req = ROBOT_RADIUS_PX + SHELF_CLEARANCE_MARGIN_PX
        for edge in self.edges.values():
            n_u = self.nodes.get(edge.u)
            n_v = self.nodes.get(edge.v)
            if n_u is None or n_v is None:
                continue

            seg_len_m = math.hypot(n_u.x - n_v.x, n_u.y - n_v.y) * PX_TO_M
            phi_sum = 0.0
            for i in range(num_samples + 1):
                t = i / float(num_samples)
                sx = (1.0 - t) * n_u.x + t * n_v.x
                sy = (1.0 - t) * n_u.y + t * n_v.y
                d_clear = self._clearance_to_rects(sx, sy, shelves)
                phi = max(0.0, (r_req - d_clear) / r_req) if d_clear < r_req else 0.0
                phi_sum += (0.5 if i in (0, num_samples) else 1.0) * phi

            edge.s_clearance = amplitude * (seg_len_m / float(num_samples)) * phi_sum
            edge.s_trolley = edge.s_clearance

    def decay_proxemic_penalties(self, dt: float, decay_rate: float = 0.1386294) -> List[Tuple[str, str]]:
        """
        Decays remembered proxemic observations exponentially.

        Without this the agent forgets congestion the instant it leaves sensing
        range, which makes a purely local planner oscillate in and out of a blocked
        aisle. Giving local observations the SAME memory model as mesh alerts means
        any measured benefit of V2V telemetry is attributable to its RANGE, not to
        the no-mesh condition being handicapped by instant amnesia.
        """
        changed = []
        factor = math.exp(-decay_rate * dt)
        for edge in self.edges.values():
            if edge.h_prox > 0.0:
                old = edge.h_prox
                new = edge.h_prox * factor
                if new < 0.01:
                    new = 0.0
                edge.h_prox = new
                # Only report changes large enough to matter for routing.
                if math.fabs(old - new) > 0.5:
                    changed.append((edge.u, edge.v))
        return changed

    def clean_expired_locks(self, current_time: float) -> List[Tuple[str, str]]:
        """Releases expired corridor locks."""
        if not isinstance(self.edges, dict):
            return []
        unlocked_edges = []
        for key, edge in list(self.edges.items()):
            if isinstance(edge, Edge) and edge.lock_owner is not None and current_time >= edge.lock_expiry:
                edge.lock_owner = None
                edge.r_lock = 0.0
                unlocked_edges.append((edge.u, edge.v))
        return unlocked_edges
