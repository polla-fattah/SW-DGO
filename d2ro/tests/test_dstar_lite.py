"""
Unit Tests for D* Lite Incremental Search Engine.
"""

import unittest
from d2ro.core.graph import TopologicalGraph
from d2ro.core.dstar_lite import DStarLite

class TestDStarLite(unittest.TestCase):
    def test_initial_shortest_path(self):
        """Verify D* Lite computes initial optimal path on a simple 2x2 grid."""
        g = TopologicalGraph()
        g.add_node("A", 0.0, 0.0)
        g.add_node("B", 10.0, 0.0)
        g.add_node("C", 0.0, 10.0)
        g.add_node("D", 10.0, 10.0)

        g.add_edge("A", "B", bidirectional=True)
        g.add_edge("B", "D", bidirectional=True)
        g.add_edge("A", "C", bidirectional=True)
        g.add_edge("C", "D", bidirectional=True)

        planner = DStarLite(g, s_start="A", s_goal="D")
        success = planner.compute_shortest_path()
        self.assertTrue(success)

        path = planner.extract_full_path()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "D")
        self.assertIn(path[1], ("B", "C"))

    def test_incremental_edge_blockage_replanning(self):
        """Verify D* Lite dynamically routes around an edge when its cost inflates."""
        g = TopologicalGraph()
        g.add_node("A", 0.0, 0.0)
        g.add_node("B", 10.0, 0.0)
        g.add_node("D", 20.0, 0.0)
        g.add_node("C", 0.0, 15.0)
        g.add_node("E", 20.0, 15.0)

        g.add_edge("A", "B", bidirectional=True)
        g.add_edge("B", "D", bidirectional=True)
        g.add_edge("A", "C", bidirectional=True)
        g.add_edge("C", "E", bidirectional=True)
        g.add_edge("E", "D", bidirectional=True)

        planner = DStarLite(g, s_start="A", s_goal="D")
        planner.compute_shortest_path()
        path1 = planner.extract_full_path()
        self.assertEqual(path1, ["A", "B", "D"])

        # Dynamically block edge (B, D) with heavy congestion
        g.update_mesh_penalty("B", "D", penalty=500.0)
        g.update_mesh_penalty("D", "B", penalty=500.0)
        planner.notify_edge_cost_change("B", "D")
        planner.notify_edge_cost_change("D", "B")

        planner.compute_shortest_path()
        path2 = planner.extract_full_path()
        self.assertEqual(path2, ["A", "C", "E", "D"])

    def test_start_node_advancement(self):
        """Verify moving start node updates key accumulator km properly."""
        g = TopologicalGraph()
        g.add_node("N1", 0.0, 0.0)
        g.add_node("N2", 10.0, 0.0)
        g.add_node("N3", 20.0, 0.0)

        g.add_edge("N1", "N2", bidirectional=True)
        g.add_edge("N2", "N3", bidirectional=True)

        planner = DStarLite(g, s_start="N1", s_goal="N3")
        planner.compute_shortest_path()
        self.assertEqual(planner.get_next_waypoint(), "N2")

        planner.update_start("N2")
        planner.compute_shortest_path()
        self.assertEqual(planner.get_next_waypoint(), "N3")

if __name__ == "__main__":
    unittest.main()
