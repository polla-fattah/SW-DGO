"""
Randomised optimality validation for the incremental D* Lite implementation.

The theoretical claim the manuscript makes is that incremental repair yields the
same solution a fresh search would. That is only credible if it is tested against
an independent optimum after MULTIPLE start advances with intervening cost
changes -- a single move cannot expose an incorrect k_m accumulation.

Each trial: build a random connected graph, then repeatedly
  (1) perturb a random set of edge costs (both increases and decreases),
  (2) advance the start along the current plan,
  (3) compare the D* Lite cost-to-go against a fresh Dijkstra optimum.
"""

import heapq
import math
import random
import unittest

from d2ro.core.graph import TopologicalGraph
from d2ro.core.dstar_lite import DStarLite


def dijkstra_cost(graph: TopologicalGraph, start: str, goal: str) -> float:
    """Independent reference optimum over the graph's current edge costs."""
    dist = {start: 0.0}
    pq = [(0.0, start)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == goal:
            return d
        for v in graph.successors(u):
            c = graph.get_cost(u, v)
            if c == math.inf:
                continue
            nd = d + c
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist.get(goal, math.inf)


def random_grid(rng, cols=4, rows=4, spacing=100.0):
    """A 4-connected grid: dense enough to have many competing routes."""
    g = TopologicalGraph()
    for i in range(cols):
        for j in range(rows):
            g.add_node(f"n{i}_{j}", i * spacing + rng.uniform(-8, 8),
                       j * spacing + rng.uniform(-8, 8))
    for i in range(cols):
        for j in range(rows):
            if i + 1 < cols:
                g.add_edge(f"n{i}_{j}", f"n{i+1}_{j}", bidirectional=True)
            if j + 1 < rows:
                g.add_edge(f"n{i}_{j}", f"n{i}_{j+1}", bidirectional=True)
    return g


class TestDStarLiteOptimality(unittest.TestCase):
    def test_matches_dijkstra_after_repeated_moves_and_cost_changes(self):
        rng = random.Random(20260815)
        goal = "n3_3"

        for trial in range(150):
            g = random_grid(rng)
            start = "n0_0"
            planner = DStarLite(g, start, goal)
            planner.compute_shortest_path()

            for move in range(5):
                # (1) Perturb a handful of edges, both upward and downward.
                for _ in range(rng.randint(1, 5)):
                    key = rng.choice(list(g.edges.keys()))
                    edge = g.edges[key]
                    edge.w_mesh = rng.choice([0.0, rng.uniform(5.0, 120.0)])
                    planner.notify_edge_cost_change(*key)

                planner.compute_shortest_path()

                # (2) Compare against an independent optimum from the current start.
                reference = dijkstra_cost(g, planner.s_start, goal)
                incremental = planner._get_g(planner.s_start)

                self.assertAlmostEqual(
                    incremental, reference, delta=1e-6,
                    msg=(f"trial {trial} move {move}: D* Lite g={incremental:.4f} "
                         f"but Dijkstra optimum={reference:.4f} "
                         f"from {planner.s_start}")
                )

                # (3) Advance the start one step along the current plan.
                nxt = planner.get_next_waypoint()
                if nxt is None or nxt == goal:
                    break
                planner.update_start(nxt)

    def test_extracted_path_cost_equals_optimum(self):
        """The path actually followed must cost what the optimum costs."""
        rng = random.Random(4242)
        goal = "n3_3"

        for trial in range(80):
            g = random_grid(rng)
            planner = DStarLite(g, "n0_0", goal)
            planner.compute_shortest_path()

            for _ in range(rng.randint(1, 6)):
                key = rng.choice(list(g.edges.keys()))
                g.edges[key].w_mesh = rng.uniform(0.0, 90.0)
                planner.notify_edge_cost_change(*key)
            planner.compute_shortest_path()

            path = planner.extract_full_path()
            self.assertEqual(path[-1], goal, "extracted path must reach the goal")

            walked = sum(g.get_cost(path[i], path[i + 1])
                         for i in range(len(path) - 1))
            self.assertAlmostEqual(
                walked, dijkstra_cost(g, "n0_0", goal), delta=1e-6,
                msg=f"trial {trial}: followed path is not the optimum"
            )

    def test_recovers_when_blocked_edge_reopens(self):
        """Cost decreases must be repaired, not only cost increases."""
        g = random_grid(random.Random(7))
        planner = DStarLite(g, "n0_0", "n3_3")
        planner.compute_shortest_path()

        key = ("n0_0", "n1_0")
        g.edges[key].w_mesh = 10_000.0
        planner.notify_edge_cost_change(*key)
        planner.compute_shortest_path()
        self.assertAlmostEqual(planner._get_g("n0_0"),
                               dijkstra_cost(g, "n0_0", "n3_3"), delta=1e-6)

        g.edges[key].w_mesh = 0.0
        planner.notify_edge_cost_change(*key)
        planner.compute_shortest_path()
        self.assertAlmostEqual(planner._get_g("n0_0"),
                               dijkstra_cost(g, "n0_0", "n3_3"), delta=1e-6)


class TestAdmissibilityUnderDistanceWeight(unittest.TestCase):
    """
    Optimality must hold across the whole weight grid the sensitivity study uses,
    not merely at nominal weights.

    The previous suite compared against Dijkstra over many randomised scenarios but
    always at w_D = 1, so it could not detect a heuristic that only becomes
    inadmissible below 1. Plain Euclidean distance overestimates the true cost as
    soon as w_D < 1 -- an unobstructed edge costs w_D*d while the heuristic claims
    d -- and an overestimating heuristic silently returns suboptimal paths rather
    than failing loudly.
    """

    W_D_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]

    def _apply_distance_weight(self, graph, w_d):
        for e in graph.edges.values():
            e.weight_d = w_d

    def test_optimal_for_every_distance_weight_in_the_sensitivity_grid(self):
        rng = random.Random(20260816)
        for w_d in self.W_D_GRID:
            for trial in range(30):
                g = random_grid(rng)
                self._apply_distance_weight(g, w_d)
                start, goal = "n0_0", "n3_3"

                planner = DStarLite(g, start, goal)
                planner.compute_shortest_path()

                got = planner.g.get(start, math.inf)
                want = dijkstra_cost(g, start, goal)
                self.assertAlmostEqual(
                    got, want, places=6,
                    msg=(f"w_D={w_d} trial={trial}: D* Lite returned {got}, "
                         f"Dijkstra optimum is {want}"))

    def test_optimal_after_cost_changes_at_low_distance_weight(self):
        """The failure mode needs repair cycles, not just a first solve."""
        rng = random.Random(7771)
        for w_d in (0.5, 0.75):
            for trial in range(25):
                g = random_grid(rng)
                self._apply_distance_weight(g, w_d)
                planner = DStarLite(g, "n0_0", "n3_3")
                planner.compute_shortest_path()

                for _ in range(4):
                    for (u, v), e in list(g.edges.items()):
                        if rng.random() < 0.25:
                            e.h_prox = rng.uniform(0.0, 4.0)
                            planner.notify_edge_cost_change(u, v)
                    planner.compute_shortest_path()

                    got = planner.g.get(planner.s_start, math.inf)
                    want = dijkstra_cost(g, planner.s_start, "n3_3")
                    self.assertAlmostEqual(
                        got, want, places=6,
                        msg=f"w_D={w_d} trial={trial}: {got} vs Dijkstra {want}")

    def test_heuristic_never_exceeds_true_cost(self):
        """Admissibility directly: h(u,goal) <= c*(u,goal) for every vertex."""
        rng = random.Random(31337)
        for w_d in self.W_D_GRID:
            g = random_grid(rng)
            self._apply_distance_weight(g, w_d)
            planner = DStarLite(g, "n0_0", "n3_3")
            planner.compute_shortest_path()
            for u in g.nodes:
                h = planner._heuristic(u, "n3_3")
                c_star = dijkstra_cost(g, u, "n3_3")
                if c_star == math.inf:
                    continue
                self.assertLessEqual(
                    h, c_star + 1e-9,
                    msg=(f"w_D={w_d}: heuristic {h:.6f} exceeds true cost "
                         f"{c_star:.6f} from {u} -- heuristic is inadmissible"))


if __name__ == "__main__":
    unittest.main()
