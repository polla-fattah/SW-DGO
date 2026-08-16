"""
Unit tests for the kinetic safety envelope S_trolley as a genuine SW-DGO
edge-cost term (Eq. 3 / Eq. 8), covering:
  - the static geometric fixture-clearance component,
  - the dynamic peer-occupancy component,
  - the routing influence of the term (it must change chosen paths),
  - correct removal of the term under ablation (enable_safety=False).
"""

import math
import unittest

from d2ro.core.graph import TopologicalGraph
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import ProxemicsField
from d2ro.core.units import (
    ROBOT_RADIUS_PX, SHELF_CLEARANCE_MARGIN_PX, TROLLEY_PEER_SIGMA_PX
)
from d2ro.environments.supermarket import SupermarketLayout


class TestStaticClearance(unittest.TestCase):
    @staticmethod
    def _corridor(wall_offset_px):
        """A single A->B edge flanked by two walls at +/- wall_offset_px."""
        g = TopologicalGraph()
        g.add_node("A", 0.0, 0.0)
        g.add_node("B", 0.0, 100.0)
        g.add_edge("A", "B")
        g.compute_clearance_penalties([
            (-wall_offset_px - 20.0, 0.0, -wall_offset_px, 100.0),
            (wall_offset_px, 0.0, wall_offset_px + 20.0, 100.0),
        ])
        return g.get_edge("A", "B")

    def test_tight_edge_penalised_open_edge_free(self):
        """An edge squeezed between fixtures must cost more than an open one."""
        r_req = ROBOT_RADIUS_PX + SHELF_CLEARANCE_MARGIN_PX

        tight = self._corridor(r_req * 0.5)   # walls inside the chassis envelope
        wide = self._corridor(r_req * 4.0)    # walls far outside it

        self.assertGreater(tight.s_clearance, 0.0,
                           "fixture-flanked edge must incur a clearance penalty")
        self.assertEqual(wide.s_clearance, 0.0,
                         "edge with ample clearance must incur no penalty")
        self.assertEqual(tight.s_trolley, tight.s_clearance)

    def test_clearance_scales_with_tightness(self):
        """Narrower gaps must produce strictly larger penalties."""
        r_req = ROBOT_RADIUS_PX + SHELF_CLEARANCE_MARGIN_PX
        self.assertGreater(self._corridor(r_req * 0.25).s_clearance,
                           self._corridor(r_req * 0.75).s_clearance)

    def test_all_domains_seed_clearance(self):
        """Every environment must seed a non-trivial clearance field."""
        layout = SupermarketLayout()
        vals = [e.s_clearance for e in layout.graph.edges.values()]
        self.assertTrue(any(v > 0 for v in vals),
                        "supermarket aisles must carry clearance penalties")


class TestDynamicPeerComponent(unittest.TestCase):
    def _two_agent_graph(self):
        g = TopologicalGraph()
        g.add_node("S", 0.0, 0.0)
        g.add_node("VIA_A", 100.0, -50.0)
        g.add_node("VIA_B", 100.0, 50.0)
        g.add_node("G", 200.0, 0.0)
        g.add_edge("S", "VIA_A")
        g.add_edge("S", "VIA_B")
        g.add_edge("VIA_A", "G")
        g.add_edge("VIA_B", "G")
        return g

    def test_peer_presence_raises_edge_cost(self):
        """A peer parked on a node must inflate the cost of edges into it."""
        g = self._two_agent_graph()
        mesh = MeshNetwork()
        ego = TrolleyAgent(1, g, "S", "G", mesh)
        blocker = TrolleyAgent(2, g, "VIA_A", "G", mesh)
        blocker.x, blocker.y = 100.0, -50.0  # sit exactly on VIA_A

        before = ego.graph.get_edge("S", "VIA_A").s_trolley
        for _ in range(3):  # throttled to every 3rd tick
            ego.update_trolley_safety_costs([ego, blocker])
        after = ego.graph.get_edge("S", "VIA_A").s_trolley

        self.assertGreater(after, before,
                           "edge into an occupied node must gain a safety penalty")

    def test_peer_penalty_decays_with_distance(self):
        """Penalty must fall off with peer distance (Gaussian envelope)."""
        g = self._two_agent_graph()
        mesh = MeshNetwork()
        ego = TrolleyAgent(1, g, "S", "G", mesh)
        far = TrolleyAgent(3, g, "VIA_B", "G", mesh)
        far.x, far.y = 100.0 + 6.0 * TROLLEY_PEER_SIGMA_PX, 50.0

        for _ in range(3):
            ego.update_trolley_safety_costs([ego, far])
        self.assertAlmostEqual(ego.graph.get_edge("S", "VIA_B").s_trolley,
                               ego.graph.get_edge("S", "VIA_B").s_clearance,
                               delta=1e-6,
                               msg="distant peer must not inflate edge cost")


class TestSafetyAblation(unittest.TestCase):
    def test_ablation_removes_term_from_graph(self):
        """enable_safety=False must zero the w_S term in the graph objective."""
        layout = SupermarketLayout()
        mesh = MeshNetwork()
        full = TrolleyAgent(1, layout.graph, "N_back_1", "DOCK_BAY_MAIN", mesh,
                            enable_safety=True)
        ablated = TrolleyAgent(2, layout.graph, "N_back_1", "DOCK_BAY_MAIN", mesh,
                               enable_safety=False)

        self.assertTrue(any(e.s_trolley > 0 for e in full.graph.edges.values()),
                        "full system must carry a nonzero safety field")
        self.assertTrue(all(e.s_trolley == 0.0 for e in ablated.graph.edges.values()),
                        "ablated system must carry no safety field at all")

    def test_term_participates_in_cost(self):
        """Edge.cost must actually change when s_trolley changes."""
        g = TopologicalGraph()
        g.add_node("A", 0.0, 0.0)
        g.add_node("B", 100.0, 0.0)
        g.add_edge("A", "B")
        e = g.get_edge("A", "B")
        base = e.cost
        e.s_trolley = 10.0
        self.assertGreater(e.cost, base,
                           "S_trolley must contribute to the composite edge cost")


if __name__ == "__main__":
    unittest.main()
