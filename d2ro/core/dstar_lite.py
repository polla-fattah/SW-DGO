"""
D* Lite Incremental Graph Search Algorithm.
Implements Koenig & Likhachev (2002) optimized for dynamic, multi-agent retail routing.
"""

from __future__ import annotations
import math
import heapq
from typing import Dict, List, Tuple, Optional, Set
from .graph import TopologicalGraph

class PriorityQueue:
    """
    Priority Queue supporting O(log N) insert, pop, and key updates with lazy deletion.
    Keys are 2-tuples (k1, k2) compared lexicographically.
    """
    def __init__(self):
        self._heap: List[Tuple[Tuple[float, float], int, str]] = []
        self._entries: Dict[str, Tuple[float, float]] = {}
        self._counter: int = 0  # Tie-breaker to prevent node comparison errors

    def insert(self, item: str, priority: Tuple[float, float]) -> None:
        self._counter += 1
        self._entries[item] = priority
        heapq.heappush(self._heap, (priority, self._counter, item))

    def remove(self, item: str) -> None:
        if item in self._entries:
            del self._entries[item]

    def update(self, item: str, priority: Tuple[float, float]) -> None:
        self.insert(item, priority)

    def pop(self) -> str:
        while self._heap:
            priority, _, item = heapq.heappop(self._heap)
            if item in self._entries and self._entries[item] == priority:
                del self._entries[item]
                return item
        raise KeyError("Pop from empty priority queue")

    def top_key(self) -> Tuple[float, float]:
        while self._heap:
            priority, _, item = self._heap[0]
            if item in self._entries and self._entries[item] == priority:
                return priority
            heapq.heappop(self._heap)
        return (math.inf, math.inf)

    def contains(self, item: str) -> bool:
        return item in self._entries

    def is_empty(self) -> bool:
        while self._heap:
            priority, _, item = self._heap[0]
            if item in self._entries and self._entries[item] == priority:
                return False
            heapq.heappop(self._heap)
        return True


class DStarLite:
    """
    Incremental heuristic search engine (D* Lite).
    Plans from s_start to s_goal and rapidly repairs paths as edge costs change.
    """
    def __init__(self, graph: TopologicalGraph, s_start: str, s_goal: str):
        self.graph = graph
        self.s_start = s_start
        self.s_last = s_start
        self.s_goal = s_goal

        self.g: Dict[str, float] = {}
        self.rhs: Dict[str, float] = {}
        self.km: float = 0.0
        self.queue = PriorityQueue()

        # Smallest distance weight anywhere in this agent's graph. See _heuristic.
        self._w_d_min = self._min_distance_weight()

        self._initialize()

    def _min_distance_weight(self) -> float:
        """
        The smallest w_D present on any edge of this graph.

        Cached at construction because the weights are set once, before the first
        solve, and never varied during a run. If that ever changes, this must be
        recomputed whenever a weight changes.
        """
        weights = [float(getattr(e, "weight_d", 1.0))
                   for e in getattr(self.graph, "edges", {}).values()]
        w = min(weights) if weights else 1.0
        # A non-positive distance weight would make the heuristic useless (and a
        # zero-cost path arbitrarily long), so clamp to a small positive value.
        return w if w > 0.0 else 1.0

    def _heuristic(self, u: str, v: str) -> float:
        r"""
        Admissible lower bound on the true cost between two vertices.

        The traversal cost of an edge is

            w_D*d + w_M*w_mesh + w_H*h_prox + w_S*s_trolley

        with every term after the first non-negative. A lower bound on any path is
        therefore obtained by keeping only the distance term at its smallest
        possible coefficient, giving

            h(u,v) = min(w_D) * ||p_u - p_v||.

        Scaling by min(w_D) is not cosmetic. Plain Euclidean distance is admissible
        only while w_D >= 1: at w_D = 0.5 an unobstructed edge costs 0.5*d while the
        heuristic claims d, so h > c* and the search may return a suboptimal path.
        The weight-sensitivity study evaluates w_D at 0.5 and 0.75, so the unscaled
        heuristic was inadmissible for exactly those configurations.
        """
        return self._w_d_min * self.graph.distance(u, v)

    def _get_g(self, u: str) -> float:
        return self.g.get(u, math.inf)

    def _get_rhs(self, u: str) -> float:
        return self.rhs.get(u, math.inf)

    def _calculate_key(self, u: str) -> Tuple[float, float]:
        min_g_rhs = min(self._get_g(u), self._get_rhs(u))
        k1 = min_g_rhs + self._heuristic(self.s_start, u) + self.km
        k2 = min_g_rhs
        return (k1, k2)

    def _initialize(self) -> None:
        """Initializes g and rhs values."""
        self.g.clear()
        self.rhs.clear()
        self.km = 0.0
        self.queue = PriorityQueue()

        self.rhs[self.s_goal] = 0.0
        self.queue.insert(self.s_goal, self._calculate_key(self.s_goal))

    def _update_vertex(self, u: str) -> None:
        if u != self.s_goal:
            # rhs(u) = min_{s' in Succ(u)} (c(u, s') + g(s'))
            succ_costs = []
            for s_prime in self.graph.successors(u):
                c = self.graph.get_cost(u, s_prime)
                succ_costs.append(c + self._get_g(s_prime))
            self.rhs[u] = min(succ_costs) if succ_costs else math.inf

        in_queue = self.queue.contains(u)
        g_val = self._get_g(u)
        rhs_val = self._get_rhs(u)

        if not math.isclose(g_val, rhs_val, abs_tol=1e-6) and in_queue:
            self.queue.update(u, self._calculate_key(u))
        elif not math.isclose(g_val, rhs_val, abs_tol=1e-6) and not in_queue:
            self.queue.insert(u, self._calculate_key(u))
        elif math.isclose(g_val, rhs_val, abs_tol=1e-6) and in_queue:
            self.queue.remove(u)

    def compute_shortest_path(self, max_iterations: int = 10000) -> bool:
        """Executes D* Lite priority queue expansion."""
        iterations = 0
        while (self.queue.top_key() < self._calculate_key(self.s_start) or 
               not math.isclose(self._get_rhs(self.s_start), self._get_g(self.s_start), abs_tol=1e-6)):
            if self.queue.is_empty() or iterations >= max_iterations:
                return False  # No path exists or exceeded iteration budget

            iterations += 1
            k_old = self.queue.top_key()
            u = self.queue.pop()
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self.queue.insert(u, k_new)
            elif self._get_g(u) > self._get_rhs(u):
                self.g[u] = self._get_rhs(u)
                for s in self.graph.predecessors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = math.inf
                for s in self.graph.predecessors(u) + [u]:
                    self._update_vertex(s)

        return True

    def notify_edge_cost_change(self, u: str, v: str) -> None:
        """Notifies planner of an updated edge cost (u, v) and marks vertex inconsistent."""
        self._update_vertex(u)

    def update_start(self, new_start: str) -> None:
        """Advances agent position along trajectory and accumulates heuristic key shift."""
        if new_start == self.s_start:
            return
        self.km += self._heuristic(self.s_start, new_start)
        self.s_last = new_start
        self.s_start = new_start

    def get_next_waypoint(self) -> Optional[str]:
        """Returns immediate successor node on the optimal trajectory."""
        if self.s_start == self.s_goal:
            return self.s_goal

        successors = self.graph.successors(self.s_start)
        if not successors:
            return None

        best_cost = math.inf
        best_next = None

        for s_prime in successors:
            c = self.graph.get_cost(self.s_start, s_prime)
            total = c + self._get_g(s_prime)
            if total < best_cost:
                best_cost = total
                best_next = s_prime

        if best_cost == math.inf:
            return None
        return best_next

    def extract_full_path(self, max_length: int = 100) -> List[str]:
        """Extracts complete sequence of waypoints from s_start to s_goal."""
        path = [self.s_start]
        curr = self.s_start
        visited = {curr}

        while curr != self.s_goal and len(path) < max_length:
            successors = self.graph.successors(curr)
            best_cost = math.inf
            best_next = None
            for s_prime in successors:
                c = self.graph.get_cost(curr, s_prime)
                total = c + self._get_g(s_prime)
                if total < best_cost:
                    best_cost = total
                    best_next = s_prime

            if best_next is None or best_next in visited or best_cost == math.inf:
                break
            path.append(best_next)
            visited.add(best_next)
            curr = best_next

        return path
