"""
Autonomous Trolley Agent for SW-DGO Framework.
Implements non-holonomic kinematics, dynamic inter-trolley safety clearance envelopes (S_trolley),
anti-tailgating following distances, expanded shelf margin collision solvers, and human proxemics.
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from .graph import TopologicalGraph
from .dstar_lite import DStarLite
from .mesh_network import MeshNetwork, MessageType, MeshPacket
from .human import Human, ProxemicsField
from .metrics import init_social_metrics, update_social_metrics
from .units import (
    ROBOT_RADIUS_PX, SHELF_CLEARANCE_MARGIN_PX, FOLLOWING_DISTANCE_GAP_PX,
    V2V_MESH_COMM_RANGE_PX, ROBOT_VMAX_MPS, ROBOT_VMAX_PXPS, ROBOT_WMAX_RADPS, M_TO_PX,
    TROLLEY_PEER_AMPLITUDE, TROLLEY_PEER_SIGMA_PX, INTIMATE_RADIUS_PX,
    HEAD_ON_CONFLICT_RADIUS_PX, MESH_ALERT_EQUIV_M, MESH_FOLLOW_BLOCK_EQUIV_M,
    SENSING_RADIUS_PX, V2V_DECAY_RATE_PER_SEC,
    ARRIVAL_RADIUS_PX,
    CORRIDOR_LOCK_LEASE_S as LOCK_LEASE_S
)

class TrolleyAgent:
    """
    Autonomous Mobile Shopping Trolley (Int-Cart).
    Executes decentralized SW-DGO routing logic with non-holonomic kinematics,
    active inter-trolley safety clearance envelopes, shelf-margin safety zones, and social yielding.
    """
    def __init__(self, agent_id: int, graph: TopologicalGraph, start_node: str, goal_node: str,
                 mesh_net: MeshNetwork, max_speed: float = ROBOT_VMAX_PXPS, max_omega: float = ROBOT_WMAX_RADPS,
                 comm_radius: float = V2V_MESH_COMM_RANGE_PX,
                 enable_mesh: bool = True, enable_lock: bool = True,
                 enable_prox: bool = True, enable_safety: bool = True,
                 lock_priority: float = 1.0, static_route: bool = False,
                 enable_yield: bool = True,
                 enable_safety_cost: Optional[bool] = None,
                 enable_safety_controller: Optional[bool] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.agent_id = agent_id
        self.graph = graph.clone()
        self.current_node = start_node
        self.goal_node = goal_node
        self.mesh_net = mesh_net

        # Ablation / Feature Flags
        self.enable_mesh = enable_mesh
        self.enable_lock = enable_lock
        self.enable_prox = enable_prox
        self.enable_safety = enable_safety

        # `enable_safety` historically switched TWO different things at once: the
        # w_S graph-cost term, and the reactive safety controller (clearance bubble,
        # shelf margin, following gap, inter-trolley slowing). Ablating it therefore
        # measured planner and controller together and could not attribute the
        # effect to either. They are now separable; both default to `enable_safety`
        # so existing behaviour is unchanged unless a caller asks otherwise.
        self.enable_safety_cost = (enable_safety if enable_safety_cost is None
                                   else enable_safety_cost)
        self.enable_safety_controller = (enable_safety if enable_safety_controller
                                         is None else enable_safety_controller)

        # Freezes the high-level route: the path is solved once against the initial
        # cost field and never re-solved, no matter how edge costs subsequently move.
        #
        # This exists to support a MATCHED-CONTROLLER shortest-path baseline. The
        # standalone Static A* baseline steers by snapping its heading straight at the
        # next waypoint, so comparing it against D2RO confounds "the route was chosen
        # differently" with "the vehicle turns differently". An agent constructed with
        # static_route=True and the social terms disabled shares D2RO's executor
        # exactly -- same angular-rate limit, collision geometry, yielding layer and
        # arrival test -- and differs ONLY in that its route is a fixed shortest path.
        # The difference between the two therefore isolates SW-DGO itself.
        self.static_route = static_route

        # Whether the agent stops for pedestrians. Separate from enable_prox, which
        # governs only the H_prox COST TERM: an agent can route socially without
        # yielding, or yield without routing socially.
        #
        # The matched-controller baseline must set this False. A frozen-route agent
        # that yields has no recourse when a pedestrian occupies its only path -- it
        # stops, cannot re-route, and stalls until the clock expires, which measures
        # the absence of replanning rather than the cost of social routing. Static A*
        # drives through people; the matched arm must do the same, differing from it
        # only in the vehicle model. Social exposure is still MEASURED either way.
        self.enable_yield = enable_yield

        # Non-holonomic Kinematics
        node_obj = self.graph.get_node(start_node)
        self.x: float = node_obj.x
        self.y: float = node_obj.y
        self.heading: float = 0.0
        self.speed: float = 0.0
        self.max_speed = max_speed
        self.max_omega = max_omega

        # Safety Envelopes (Physical Body Radius vs Kinetic Clearance Bubble in SI units)
        self.radius: float = ROBOT_RADIUS_PX                    # Physical chassis radius (0.40 m / 13.3 px)
        _ctl = self.enable_safety_controller
        self.safety_bubble_radius: float = 26.0 if _ctl else 0.0 # Kinetic safety clearance envelope (0.78 m)
        self.shelf_margin: float = SHELF_CLEARANCE_MARGIN_PX if _ctl else 0.0 # Minimum distance maintained from shelf edges (0.54 m / 18 px)
        self.following_gap: float = FOLLOWING_DISTANCE_GAP_PX if _ctl else 0.0 # Anti-tailgating gap (1.08 m / 36 px)

        # Ablating S_trolley must remove the term from the graph objective as well as
        # from reactive motion correction; otherwise the "w/o safety" arm is not a true
        # ablation of the w_S component of Eq. (3).
        if not self.enable_safety_cost:
            for edge in self.graph.edges.values():
                edge.s_clearance = 0.0
                edge.s_trolley = 0.0

        # Per-agent weight overrides for the sensitivity study (Eq. 3). Applied to
        # this agent's OWN cloned graph, so one agent's weights never leak into a
        # peer's cost field, and applied BEFORE the first solve so the initial route
        # already reflects them.
        self.weights = {"w_D": None, "w_M": None, "w_H": None,
                        "w_R": None, "w_S": None}
        if weights:
            self.weights.update(weights)
            field = {"w_D": "weight_d", "w_M": "weight_m", "w_H": "weight_h",
                     "w_R": "weight_r", "w_S": "weight_s"}
            for edge in self.graph.edges.values():
                for key, attr in field.items():
                    val = self.weights.get(key)
                    if val is not None:
                        setattr(edge, attr, float(val))

        # Latency tracking
        self.last_compute_time_ms: float = 0.15

        # High-level planning
        self.planner = DStarLite(self.graph, start_node, goal_node)
        self.planner.compute_shortest_path()
        self.target_node: Optional[str] = self.planner.get_next_waypoint()

        if self.target_node:
            t_obj = self.graph.get_node(self.target_node)
            self.heading = math.atan2(t_obj.y - self.y, t_obj.x - self.x)

        # State machine: "NAVIGATING", "WAITING_LOCK", "YIELDING_HUMAN", "FOLLOWING_CART", "DOCKED"
        self.state: str = "NAVIGATING"
        self.active_lock_edge: Optional[Tuple[str, str]] = None
        self.wait_timer: float = 0.0

        # ---- Distributed corridor reservation state (Sec. III-D) ----------- #
        # Claim tuple per Eq. (9): <owner, direction, t_acquire, t_expire, priority>.
        # Claims are totally ordered by (priority, t_acquire, agent_id); the smallest
        # claim wins. Ordering by t_acquire gives FIFO service, which is what rules
        # out starvation; agent_id breaks exactly-simultaneous ties deterministically.
        self.lock_priority: float = lock_priority
        self.lock_claims: Dict[Tuple[str, str], Dict[int, Tuple[float, float, int]]] = {}
        self.pending_corridor: Optional[Tuple[str, str]] = None
        # lock_wait_time is a PER-EPISODE timer: _release_corridor() resets it once the
        # corridor is granted, which is correct for the state machine but destroys the
        # quantity the experiments actually want. total_lock_wait_time accumulates
        # across the whole mission and is never reset, so end-of-run reads are valid.
        self.lock_wait_time: float = 0.0
        self.total_lock_wait_time: float = 0.0
        self.lock_grants_received: int = 0
        self.lock_denials_received: int = 0
        self.yield_timer: float = 0.0
        self.peer_block_timer: float = 0.0

        # ---- Performance metrics ------------------------------------------ #
        # Event counters and exposure accumulators are kept strictly separate.
        # A per-tick counter answers "how long was the robot too close?", while an
        # event counter answers "how many distinct encounters occurred?". Reporting a
        # tick count under an event name is what produced figures such as "2094
        # deadlocks per trial" in the previous revision.
        self.total_distance: float = 0.0
        self.travel_time: float = 0.0
        self.replan_count: int = 0
        # Per-repair D* Lite timings. Kept as a series, not a running mean, so the
        # analysis can report median/p95/max: a mean alone hides the tail, and the
        # tail is what determines whether a control deadline is ever missed.
        self.replan_latencies_ms: List[float] = []

        # Genuine routing deadlocks: no admissible route exists and it is not
        # attributable to orderly yielding at a reserved corridor.
        self.deadlock_count: int = 0
        # Discrete head-on encounters in single-file corridors (entry-triggered).
        self.head_on_events: int = 0
        # Control cycles spent stalled behind a peer (congestion, not deadlock).
        self.congestion_events: int = 0

        # Social compliance: distinct encounters vs cumulative exposure time.
        init_social_metrics(self)
        self._peers_in_conflict: set = set()

        # Fixture contact, split the same way human proxemics already is: a tick
        # counter answers "how long was the chassis inside the clearance envelope?",
        # an event counter answers "how many distinct contacts occurred?". The single
        # legacy counter conflated them, so a value of ~193 was reported as 193
        # "scrapes" when it is 193 control cycles requiring a clearance correction.
        self.shelf_contact_ticks: int = 0     # exposure (control cycles)
        self.shelf_contact_events: int = 0    # distinct fixture contacts
        self.min_shelf_clearance_px: float = math.inf
        self._shelves_in_contact: set = set()
        self.is_docked: bool = False
        self.last_compute_time_ms: float = 0.0

        self.mesh_net.register_agent(self.agent_id, self)

    @property
    def current_pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def _repair(self) -> None:
        """
        Executes ONE D* Lite repair and records how long it took.

        Timing lives here, around `compute_shortest_path()` alone, rather than
        around the whole control step. `last_compute_time_ms` measures the entire
        `step()` -- mesh handling, proxemic and safety updates, yielding, collision
        correction and motion -- and is recorded on every tick including ticks where
        no repair happens. Averaging that quantity and calling it "D* Lite repair
        latency" measures the wrong thing; the two are reported separately.

        The initial full solve in __init__ is deliberately NOT recorded here: it is a
        from-scratch solve, not an incremental repair, and averaging it into the
        repair series would flatter the incremental claim.
        """
        t = time.perf_counter()
        self.planner.compute_shortest_path()
        self.replan_latencies_ms.append((time.perf_counter() - t) * 1000.0)
        self.replan_count += 1

    def process_inbound_mesh(self, current_time: float = math.inf) -> bool:
        # current_time MUST be threaded through: fetch_inbound defaults to +inf, which
        # delivers every queued packet immediately regardless of its deliver_at stamp.
        # Omitting it silently disables the mesh's latency model -- the network can
        # simulate delay, but the agent never experiences any.
        packets = self.mesh_net.fetch_inbound(self.agent_id, current_time=current_time)
        cost_changed = False

        for pkt in packets:
            u, v = pkt.edge
            if pkt.msg_type == MessageType.CONGESTION_ALERT:
                if self.graph.update_mesh_penalty(u, v, pkt.cost_penalty):
                    self.planner.notify_edge_cost_change(u, v)
                    cost_changed = True
            elif pkt.msg_type == MessageType.LOCK_REQUEST:
                edge = self.graph.get_edge(u, v)
                if edge and edge.is_single_file:
                    corridor = self._corridor_key(u, v)
                    peer_claim = (pkt.priority, pkt.timestamp, pkt.sender_id)
                    self._register_claim(corridor, peer_claim)

                    # Answer the request explicitly: grant if we do not out-rank the
                    # peer, deny if our own outstanding claim is stronger.
                    mine = self.lock_claims.get(corridor, {}).get(self.agent_id)
                    contested = mine is not None and mine < peer_claim
                    self.mesh_net.broadcast(
                        sender_id=self.agent_id,
                        msg_type=(MessageType.LOCK_DENY if contested
                                  else MessageType.LOCK_GRANT),
                        edge=(u, v),
                        priority=self.lock_priority,
                        ttl=3,
                        current_time=pkt.timestamp,
                        target_id=pkt.sender_id
                    )
                    cost_changed = True

            elif pkt.msg_type in (MessageType.LOCK_GRANT, MessageType.LOCK_DENY):
                if pkt.target_id == self.agent_id:
                    if pkt.msg_type == MessageType.LOCK_GRANT:
                        self.lock_grants_received += 1
                    else:
                        self.lock_denials_received += 1
                        # Record the denier's stronger standing so the total order
                        # converges even if we never saw its original request.
                        self._register_claim(
                            self._corridor_key(u, v),
                            (pkt.priority, pkt.timestamp, pkt.sender_id)
                        )
                        cost_changed = True

            elif pkt.msg_type == MessageType.LOCK_RELEASE:
                corridor = self._corridor_key(u, v)
                if self.lock_claims.get(corridor, {}).pop(pkt.sender_id, None) is not None:
                    cost_changed = True

        if self._apply_lock_costs(0.0):
            cost_changed = True

        return cost_changed

    def update_human_proxemics(self, humans: Any, prox_field: ProxemicsField) -> bool:
        cost_changed = False
        if humans is None or not hasattr(self.graph, "edges"):
            return False

        if not hasattr(self, "_prox_update_counter"):
            self._prox_update_counter = 0
        self._prox_update_counter += 1
        if self._prox_update_counter % 3 != 0:
            return False
        if isinstance(humans, list):
            human_list = humans
        elif isinstance(humans, (tuple, set)):
            human_list = list(humans)
        else:
            human_list = [humans]

        # Onboard perception is line-of-sight bounded: only humans the agent can
        # actually observe may enter its own cost field. Congestion beyond this
        # radius is knowable solely through V2V telemetry, which is precisely the
        # perception horizon the mesh is claimed to extend.
        human_list = [h for h in human_list
                      if hasattr(h, "x")
                      and math.hypot(self.x - h.x, self.y - h.y) <= SENSING_RADIUS_PX]

        if not isinstance(self.graph.edges, dict):
            return False

        for key, edge in list(self.graph.edges.items()):
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            u, v = key
            nu = self.graph.get_node(u)
            nv = self.graph.get_node(v)
            if nu is None or nv is None:
                continue
            d_u = math.hypot(self.x - nu.x, self.y - nu.y)
            d_v = math.hypot(self.x - nv.x, self.y - nv.y)
            d_near = d_u if d_u < d_v else d_v
            if d_near <= 200.0:
                mx = (nu.x + nv.x) * 0.5
                my = (nu.y + nv.y) * 0.5
                edge_len = math.hypot(nu.x - nv.x, nu.y - nv.y)
                cutoff = edge_len * 0.5 + 90.0
                has_nearby = False
                for h in human_list:
                    if hasattr(h, "x") and math.hypot(h.x - mx, h.y - my) <= cutoff:
                        has_nearby = True
                        break

                if has_nearby:
                    seg_penalty = prox_field.compute_edge_segment_penalty((nu.x, nu.y), (nv.x, nv.y), human_list)
                else:
                    seg_penalty = 0.0

                # Observations RAISE the remembered penalty immediately; they never
                # clear it. Forgetting is handled by decay_proxemic_penalties, so an
                # agent that merely looks away does not instantly un-learn a blockage.
                if seg_penalty > edge.h_prox + 0.05:
                    edge.h_prox = seg_penalty
                    self.planner.notify_edge_cost_change(u, v)
                    cost_changed = True
        return cost_changed

    def update_trolley_safety_costs(self, peer_agents: Optional[List[TrolleyAgent]]) -> bool:
        r"""
        Updates the kinetic safety envelope term $S_{\text{trolley}}(v,t)$ on the agent's
        own graph, so that the safety component genuinely participates in SW-DGO graph
        optimisation (Eq. 3) rather than acting only as post-planning motion correction.

        $S_{\text{trolley}}(u,v,t) = S^{\text{static}}_{\text{clearance}}(u,v)
          + A_t \sum_{j \neq i} \exp\!\left(-\frac{\|p_v - p_j(t)\|^2}{2\sigma_t^2}\right)$

        The static term encodes fixture clearance; the dynamic sum encodes peer trolley
        occupancy of the edge endpoint. Returns True if any edge cost changed.
        """
        if not self.enable_safety:
            return False

        # Throttle: recompute every 3rd tick (matches proxemics cadence)
        if not hasattr(self, "_safety_update_counter"):
            self._safety_update_counter = 0
        self._safety_update_counter += 1
        if self._safety_update_counter % 3 != 0:
            return False

        peers = [p for p in (peer_agents or [])
                 if p.agent_id != self.agent_id and not p.is_docked]

        cost_changed = False
        two_sigma_sq = 2.0 * (TROLLEY_PEER_SIGMA_PX ** 2)
        cutoff_sq = (3.0 * TROLLEY_PEER_SIGMA_PX) ** 2

        for (u, v), edge in self.graph.edges.items():
            n_v = self.graph.nodes.get(v)
            if n_v is None:
                continue

            peer_pen = 0.0
            for p in peers:
                d_sq = (n_v.x - p.x) ** 2 + (n_v.y - p.y) ** 2
                if d_sq < cutoff_sq:
                    peer_pen += TROLLEY_PEER_AMPLITUDE * math.exp(-d_sq / two_sigma_sq)

            new_s = edge.s_clearance + peer_pen
            if math.fabs(edge.s_trolley - new_s) > 1.0:
                edge.s_trolley = new_s
                self.planner.notify_edge_cost_change(u, v)
                cost_changed = True

        return cost_changed

    def resolve_shelf_collisions(self, shelves: Optional[List[Tuple[float, float, float, float]]]) -> None:
        """
        Hard collision & clearance buffer clamping against rectangular shelves.
        Enforces shelf_margin so trolley corners never scrape or slam into shelf walls.
        """
        if not shelves:
            return

        contact_now = set()
        for idx, (sx1, sy1, sx2, sy2) in enumerate(shelves):
            # Clearance to the bare fixture, independent of the margin, so the figure
            # remains comparable across configurations with different envelopes.
            cx = max(sx1, min(sx2, self.x))
            cy = max(sy1, min(sy2, self.y))
            self.min_shelf_clearance_px = min(self.min_shelf_clearance_px,
                                              math.hypot(self.x - cx, self.y - cy))

            min_x = sx1 - self.shelf_margin
            max_x = sx2 + self.shelf_margin
            min_y = sy1 - self.shelf_margin
            max_y = sy2 + self.shelf_margin

            if min_x <= self.x <= max_x and min_y <= self.y <= max_y:
                contact_now.add(idx)
                self.shelf_contact_ticks += 1
                dx_left = self.x - min_x
                dx_right = max_x - self.x
                dy_bottom = self.y - min_y
                dy_top = max_y - self.y

                min_escape = min(dx_left, dx_right, dy_bottom, dy_top)
                if min_escape == dx_left:
                    self.x = min_x - 1.0
                elif min_escape == dx_right:
                    self.x = max_x + 1.0
                elif min_escape == dy_bottom:
                    self.y = min_y - 1.0
                else:
                    self.y = max_y + 1.0

        # A contact is counted once on entry, not once per tick spent inside.
        self.shelf_contact_events += len(contact_now - self._shelves_in_contact)
        self._shelves_in_contact = contact_now

    def check_inter_trolley_safety(self, peer_agents: Optional[List[TrolleyAgent]], dt: float) -> bool:
        """
        Inter-trolley safety clearance (S_trolley):
        - Prevents tailgating and inter-agent crowding.
        - Maintains smooth kinetic spacing without freezing.
        """
        if not peer_agents or not self.enable_safety_controller:
            return False

        must_slow_down = False
        for other in peer_agents:
            if other.agent_id == self.agent_id or other.is_docked:
                continue

            dist = math.hypot(self.x - other.x, self.y - other.y)

            # 1. Kinetic Safety Bubble Clearance (Chassis Separation Bubble)
            min_sep = self.safety_bubble_radius * 2.0  # 52 px (1.56 m)
            if dist < min_sep and dist > 0.1:
                overlap = (min_sep - dist) * 0.5
                repulse_x = ((self.x - other.x) / dist) * overlap
                repulse_y = ((self.y - other.y) / dist) * overlap
                # Right-hand passing bias in wide corridors
                lat_x = -(other.y - self.y) / dist * 0.8
                lat_y = (other.x - self.x) / dist * 0.8
                self.x += repulse_x + lat_x
                self.y += repulse_y + lat_y

            # 2. Anti-Tailgating Following Distance Gap
            if dist < self.following_gap and dist > 0.1:
                dx = other.x - self.x
                dy = other.y - self.y
                forward_x = math.cos(self.heading)
                forward_y = math.sin(self.heading)
                dot = forward_x * dx + forward_y * dy
                if dot > 0.0:  # Other cart is directly in front along heading vector
                    must_slow_down = True

        if must_slow_down:
            self.state = "FOLLOWING_CART"
            self.peer_block_timer += dt
            # Modulate speed to match a safe following crawl (0.8 m/s)
            self.speed = min(self.speed, 0.8 * M_TO_PX)

            self.stalled_ticks += 1

            # Being held up behind a slower cart is congestion, not deadlock: the
            # route still exists and the agent still progresses.
            if self.peer_block_timer > 1.8 and self.target_node:
                self.congestion_events += 1
                if self.enable_mesh:
                    self.graph.update_mesh_penalty(self.current_node, self.target_node,
                                                   MESH_FOLLOW_BLOCK_EQUIV_M)
                    self.planner.notify_edge_cost_change(self.current_node, self.target_node)
                    self._repair()
                    self.target_node = self.planner.get_next_waypoint()
                self.peer_block_timer = 0.0
            return False  # Still allow forward crawl step

        self.peer_block_timer = 0.0
        if self.state == "FOLLOWING_CART":
            self.state = "NAVIGATING"
        return False

    def check_human_collision_and_yield(self, humans: Any, dt: float,
                                        current_sim_time: float) -> bool:
        if humans is None:
            self.yield_timer = 0.0
            return False
        if isinstance(humans, list):
            human_list = humans
        elif isinstance(humans, (tuple, set)):
            human_list = list(humans)
        else:
            human_list = [humans]

        # Social compliance measured by the shared helper, so D2RO and every baseline
        # are scored against an identical threshold and identical semantics.
        update_social_metrics(self, human_list, dt)

        # Socially blind agents are still scored: measurement is unconditional,
        # only the behavioural response is switched off.
        if not self.enable_yield:
            self.yield_timer = 0.0
            return False

        yield_required = False
        for human in human_list:
            if not hasattr(human, "x"):
                continue
            dist = math.hypot(self.x - human.x, self.y - human.y)
            if dist < INTIMATE_RADIUS_PX and dist > 0.1:
                push_dist = INTIMATE_RADIUS_PX - dist
                self.x -= ((human.x - self.x) / dist) * (push_dist * 0.5)
                self.y -= ((human.y - self.y) / dist) * (push_dist * 0.5)

            if dist < 38.0:
                dx = human.x - self.x
                dy = human.y - self.y
                dot = math.cos(self.heading) * dx + math.sin(self.heading) * dy
                if dot > 0.0:
                    yield_required = True

        if yield_required:
            self.state = "YIELDING_HUMAN"
            self.yield_timer += dt
            self.speed = 0.0

            if self.yield_timer > 0.8 and self.target_node:
                # Congestion propagation is the V2V mechanism itself: when W_mesh is
                # ablated the agent must neither transmit nor apply the penalty to its
                # own graph, otherwise the "w/o mesh" arm still enjoys mesh-derived
                # rerouting and the ablation measures nothing.
                if self.enable_mesh:
                    self.broadcast_congestion(self.current_node, self.target_node,
                                              penalty=MESH_ALERT_EQUIV_M,
                                              current_time=current_sim_time)
                    self._repair()
                    self.target_node = self.planner.get_next_waypoint()
                self.yield_timer = 0.0
            return True

        self.yield_timer = 0.0
        return False

    def step(self, dt: float, humans: List[Human], prox_field: ProxemicsField,
             current_sim_time: float = 0.0,
             shelves: Optional[List[Tuple[float, float, float, float]]] = None,
             peer_agents: Optional[List[TrolleyAgent]] = None) -> None:
        """Main non-holonomic kinematic D2RO execution tick."""
        if self.is_docked:
            return

        t0 = time.perf_counter()
        self.travel_time += dt

        # 1. Process V2V Mesh, Proxemics & Kinetic Safety Envelope
        mesh_changed = self.process_inbound_mesh(current_sim_time) if self.enable_mesh else False
        prox_changed = self.update_human_proxemics(humans, prox_field) if self.enable_prox else False
        safety_changed = self.update_trolley_safety_costs(peer_agents) if self.enable_safety_cost else False

        # 1b. Temporal forgetting of remembered congestion.
        # This MUST operate on the agent's own cloned graph: that is the graph D* Lite
        # plans on. Decaying the shared layout graph instead would leave the penalties
        # that actually drive planning undecayed forever.
        # Applied on a 0.5 s cadence rather than every tick: decay is smooth and slow
        # (5 s half-life), so per-tick notification would force a full replan 20x a
        # second, inflating both replan counts and measured latency for no change in
        # the resulting route.
        decay_changed = False
        self._decay_accum = getattr(self, "_decay_accum", 0.0) + dt
        if self._decay_accum >= 0.5:
            step = self._decay_accum
            self._decay_accum = 0.0
            for (du, dv) in self.graph.decay_mesh_penalties(
                    step, decay_rate=V2V_DECAY_RATE_PER_SEC):
                self.planner.notify_edge_cost_change(du, dv)
                decay_changed = True
            for (du, dv) in self.graph.decay_proxemic_penalties(
                    step, decay_rate=V2V_DECAY_RATE_PER_SEC):
                self.planner.notify_edge_cost_change(du, dv)
                decay_changed = True

        # 2. Incremental Replan
        # A static-route agent never re-solves: its route is frozen at construction.
        if (mesh_changed or prox_changed or safety_changed or decay_changed) \
                and not self.static_route:
            self._repair()
            self.target_node = self.planner.get_next_waypoint()

        # 3. Check Docking Arrival (Multi-cart return bay queue)
        goal_obj = self.graph.get_node(self.goal_node)
        if self.current_node == self.goal_node or math.hypot(self.x - goal_obj.x, self.y - goal_obj.y) < ARRIVAL_RADIUS_PX:
            self.is_docked = True
            self.state = "DOCKED"
            if self.active_lock_edge and self.enable_lock:
                self._release_corridor(current_sim_time)
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        # 4. Check Human Yielding
        if self.check_human_collision_and_yield(humans, dt, current_sim_time):
            self.resolve_shelf_collisions(shelves)
            self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
            return

        # 5. Check Inter-Trolley Kinetic Safety Clearance (Anti-Tailgating)
        if self.enable_safety_controller:
            self.check_inter_trolley_safety(peer_agents, dt)

        # 6. Waypoint & Corridor Lock Verification
        if self.target_node is None:
            self._repair()
            self.target_node = self.planner.get_next_waypoint()
            if self.target_node is None:
                # No route may simply mean the only corridor onward is reserved by a
                # stronger peer. That is orderly yielding, not a deadlock, and must not
                # be counted as one.
                if self.enable_lock and self._blocked_by_peer_lock():
                    self.state = "WAITING_LOCK"
                    self.wait_timer += dt
                    self.lock_wait_time += dt
                    self.total_lock_wait_time += dt
                    self.speed = 0.0
                else:
                    self.deadlock_count += 1
                self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
                return

        edge = self.graph.get_edge(self.current_node, self.target_node)
        if self.enable_lock and edge and edge.is_single_file:
            self._purge_expired_claims(current_sim_time)
            corridor = self._corridor_key(self.current_node, self.target_node)

            # Announce intent, then defer to the total order over observed claims.
            self._request_corridor(self.current_node, self.target_node, current_sim_time)
            self._apply_lock_costs(current_sim_time)

            if not self._holds_corridor(corridor):
                self.state = "WAITING_LOCK"
                self.wait_timer += dt
                self.lock_wait_time += dt
                self.total_lock_wait_time += dt
                self.speed = 0.0
                if self.wait_timer > 1.8:
                    # Yield persistently blocked: let D* Lite divert around the corridor.
                    self._repair()
                    self.target_node = self.planner.get_next_waypoint()
                    self.wait_timer = 0.0
                self.resolve_shelf_collisions(shelves)
                self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0
                return
            else:
                # We hold the strongest claim: take ownership of the corridor.
                self.active_lock_edge = (self.current_node, self.target_node)
                self.wait_timer = 0.0
        elif not self.enable_lock and edge and edge.is_single_file:
            # With locks ablated, opposing agents may both commit to the same aisle.
            # Count each head-on encounter ONCE, on entry, rather than once per tick.
            conflicting_now = set()
            if peer_agents:
                for peer in peer_agents:
                    if peer.agent_id == self.agent_id or peer.is_docked:
                        continue
                    d_peer = math.hypot(self.x - peer.x, self.y - peer.y)
                    if d_peer < HEAD_ON_CONFLICT_RADIUS_PX:
                        heading_delta = abs((self.heading - peer.heading + math.pi)
                                            % (2 * math.pi) - math.pi)
                        if heading_delta > math.pi * 0.5:  # genuinely opposed
                            conflicting_now.add(peer.agent_id)
                            self.speed = 0.0
            self.head_on_events += len(conflicting_now - self._peers_in_conflict)
            self._peers_in_conflict = conflicting_now

        if self.state != "FOLLOWING_CART":
            self.state = "NAVIGATING"
        self.wait_timer = 0.0

        # Track shelf scrapes if safety envelopes are ablated
        if not self.enable_safety_controller and shelves:
            contact_now = set()
            for idx, (min_sx, min_sy, max_sx, max_sy) in enumerate(shelves):
                cx = max(min_sx, min(max_sx, self.x))
                cy = max(min_sy, min(max_sy, self.y))
                gap = math.hypot(self.x - cx, self.y - cy)
                self.min_shelf_clearance_px = min(self.min_shelf_clearance_px, gap)
                if gap < 14.0:
                    contact_now.add(idx)
                    self.shelf_contact_ticks += 1
            self.shelf_contact_events += len(contact_now - self._shelves_in_contact)
            self._shelves_in_contact = contact_now

        # 7. Non-Holonomic Kinematics (Bounded Steering Angle & Differential Turning)
        target_obj = self.graph.get_node(self.target_node)
        dx = target_obj.x - self.x
        dy = target_obj.y - self.y
        dist = math.hypot(dx, dy)

        if dist < 8.0:  # Waypoint arrived
            if self.active_lock_edge and self.active_lock_edge != (self.current_node, self.target_node):
                self._release_corridor(current_sim_time)

            self.current_node = self.target_node
            if self.current_node == self.goal_node:
                self.is_docked = True
                self.state = "DOCKED"
                return

            self.planner.update_start(self.current_node)
            self._repair()
            self.target_node = self.planner.get_next_waypoint()
        else:
            desired_heading = math.atan2(dy, dx)
            angle_diff = (desired_heading - self.heading + math.pi) % (2 * math.pi) - math.pi

            max_turn = self.max_omega * dt
            turn_step = max(-max_turn, min(max_turn, angle_diff))
            self.heading += turn_step

            # Unicycle forward motion with corner deceleration
            alignment = max(0.25, math.cos(angle_diff))
            target_speed = (self.speed if self.state == "FOLLOWING_CART" else self.max_speed) * alignment
            self.speed = min(dist / dt, target_speed)

            step_dist = self.speed * dt
            self.x += math.cos(self.heading) * step_dist
            self.y += math.sin(self.heading) * step_dist
            self.total_distance += step_dist

        # 8. Clamp against solid shelf walls with safety margin
        self.resolve_shelf_collisions(shelves)
        self.last_compute_time_ms = (time.perf_counter() - t0) * 1000.0

    def broadcast_congestion(self, u: str, v: str, penalty: float, current_time: float) -> None:
        self.graph.update_mesh_penalty(u, v, penalty)
        self.planner.notify_edge_cost_change(u, v)
        self.mesh_net.broadcast(
            sender_id=self.agent_id,
            msg_type=MessageType.CONGESTION_ALERT,
            edge=(u, v),
            cost_penalty=penalty,
            ttl=3,
            current_time=current_time
        )

    # ---------------------------------------------------------------- #
    # Distributed corridor reservation protocol
    # ---------------------------------------------------------------- #
    @staticmethod
    def _corridor_key(u: str, v: str) -> Tuple[str, str]:
        """
        A single-file corridor is one physical resource regardless of travel
        direction, so (u,v) and (v,u) must map to the same reservation key.
        """
        return (u, v) if u <= v else (v, u)

    def _purge_expired_claims(self, current_time: float) -> None:
        """Drops claims past their lease, so a failed holder cannot block forever."""
        for corridor, claims in list(self.lock_claims.items()):
            for aid, (prio, t_acq, owner) in list(claims.items()):
                if current_time - t_acq > LOCK_LEASE_S:
                    del claims[aid]
            if not claims:
                del self.lock_claims[corridor]

    def _register_claim(self, corridor: Tuple[str, str], claim: Tuple[float, float, int]) -> None:
        self.lock_claims.setdefault(corridor, {})[claim[2]] = claim

    def _winner(self, corridor: Tuple[str, str]) -> Optional[Tuple[float, float, int]]:
        """Returns the strongest outstanding claim under the total order."""
        claims = self.lock_claims.get(corridor)
        return min(claims.values()) if claims else None

    def _holds_corridor(self, corridor: Tuple[str, str]) -> bool:
        w = self._winner(corridor)
        return w is not None and w[2] == self.agent_id

    def _blocked_by_peer_lock(self) -> bool:
        """
        True if every onward single-file corridor from the current node is currently
        reserved by a stronger peer. Distinguishes orderly lock yielding from a genuine
        routing deadlock, so the two are never conflated in the metrics.
        """
        for nxt in self.graph.successors(self.current_node):
            edge = self.graph.get_edge(self.current_node, nxt)
            if edge is None or not edge.is_single_file:
                continue
            winner = self._winner(self._corridor_key(self.current_node, nxt))
            if winner is not None and winner[2] != self.agent_id:
                return True
        return False

    def _request_corridor(self, u: str, v: str, current_time: float) -> None:
        """
        Issues a reservation request. Ownership is NOT assumed here: the agent may
        only enter once it is the winner of the total order over all claims it has
        observed, which is what makes this mutual exclusion rather than a unilateral
        local assignment.
        """
        corridor = self._corridor_key(u, v)
        if self.agent_id in self.lock_claims.get(corridor, {}):
            return  # request already outstanding

        claim = (self.lock_priority, current_time, self.agent_id)
        self._register_claim(corridor, claim)
        self.pending_corridor = corridor
        self.mesh_net.broadcast(
            sender_id=self.agent_id,
            msg_type=MessageType.LOCK_REQUEST,
            edge=(u, v),
            priority=self.lock_priority,
            ttl=3,
            current_time=current_time
        )

    def _release_corridor(self, current_time: float) -> None:
        if not self.active_lock_edge:
            return
        u, v = self.active_lock_edge
        corridor = self._corridor_key(u, v)
        self.lock_claims.get(corridor, {}).pop(self.agent_id, None)

        edge = self.graph.get_edge(u, v)
        if edge:
            edge.lock_owner = None
            edge.r_lock = 0.0
            self.planner.notify_edge_cost_change(u, v)

        self.mesh_net.broadcast(
            sender_id=self.agent_id,
            msg_type=MessageType.LOCK_RELEASE,
            edge=(u, v),
            ttl=3,
            current_time=current_time
        )
        self.active_lock_edge = None
        self.pending_corridor = None
        self.lock_wait_time = 0.0

    def _apply_lock_costs(self, current_time: float) -> bool:
        """
        Projects the reservation state onto edge costs: a corridor claimed by a
        stronger peer becomes infinite-cost for this agent, so D* Lite naturally
        diverts through a parallel aisle or a turnout alcove.
        """
        changed = False
        for corridor, claims in self.lock_claims.items():
            if not claims:
                continue
            winner = min(claims.values())
            blocked = (winner[2] != self.agent_id)
            for (a, b) in ((corridor[0], corridor[1]), (corridor[1], corridor[0])):
                edge = self.graph.get_edge(a, b)
                if edge is None or not edge.is_single_file:
                    continue
                new_lock = math.inf if blocked else 0.0
                if edge.r_lock != new_lock:
                    edge.r_lock = new_lock
                    edge.lock_owner = winner[2]
                    edge.lock_expiry = winner[1] + LOCK_LEASE_S
                    self.planner.notify_edge_cost_change(a, b)
                    changed = True
        return changed
