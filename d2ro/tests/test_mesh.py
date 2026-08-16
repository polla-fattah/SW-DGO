"""
Unit Tests for V2V Ad-Hoc Mesh Communication and Penalty Decay.
"""

import unittest
from d2ro.core.graph import TopologicalGraph
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent

class TestMeshNetwork(unittest.TestCase):
    def test_mesh_broadcast_and_reception(self):
        """Verify trolley A broadcasts congestion alert and trolley B receives it within range."""
        g = TopologicalGraph()
        g.add_node("N1", 0.0, 0.0)
        g.add_node("N2", 50.0, 0.0)
        g.add_node("N3", 100.0, 0.0)
        g.add_edge("N1", "N2", bidirectional=True)
        g.add_edge("N2", "N3", bidirectional=True)

        mesh = MeshNetwork(comm_radius=200.0)

        agent1 = TrolleyAgent(agent_id=1, graph=g, start_node="N1", goal_node="N3", mesh_net=mesh)
        agent2 = TrolleyAgent(agent_id=2, graph=g, start_node="N3", goal_node="N1", mesh_net=mesh)

        # Agent 1 broadcasts blockage on (N1, N2)
        agent1.broadcast_congestion("N1", "N2", penalty=100.0, current_time=1.0)

        # Agent 2 checks inbound queue
        inbound_changed = agent2.process_inbound_mesh()
        self.assertTrue(inbound_changed)
        self.assertEqual(agent2.graph.get_edge("N1", "N2").w_mesh, 100.0)

    def test_mesh_penalty_temporal_decay(self):
        """Verify dynamic congestion penalties decay over time."""
        g = TopologicalGraph()
        g.add_node("A", 0.0, 0.0)
        g.add_node("B", 10.0, 0.0)
        g.add_edge("A", "B", bidirectional=True)

        g.update_mesh_penalty("A", "B", penalty=50.0)
        self.assertEqual(g.get_edge("A", "B").w_mesh, 50.0)

        # Decay over 5 seconds at lambda=0.1386294 (5s half-life)
        g.decay_mesh_penalties(dt=5.0, decay_rate=0.1386294)
        self.assertAlmostEqual(g.get_edge("A", "B").w_mesh, 25.0, delta=0.1)

        # Decay further over 60 seconds until 0.0
        g.decay_mesh_penalties(dt=60.0, decay_rate=0.1386294)
        self.assertEqual(g.get_edge("A", "B").w_mesh, 0.0)

if __name__ == "__main__":
    unittest.main()
