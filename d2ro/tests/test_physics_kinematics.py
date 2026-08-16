"""
Unit tests for physics integration, SI unit scaling, D* Lite multi-node movement,
exponential mesh decay, and trapezoidal proxemic line integrals.
"""

import math
import unittest
from d2ro.core.units import (
    PX_TO_M, M_TO_PX, PHYSICS_DT, ROBOT_VMAX_MPS, ROBOT_VMAX_PXPS
)
from d2ro.core.graph import TopologicalGraph
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.dstar_lite import DStarLite
from d2ro.core.human import Human, ProxemicsField
from d2ro.baselines import StaticAStarAgent, ArtificialPotentialFieldAgent, ORCAAgent, DecentralizedLocalMAPFAgent

class TestPhysicsKinematics(unittest.TestCase):
    def test_physical_velocity_alignment(self):
        """Verify all agents commanded at V_max travel exactly 1.20 m in 1 simulated second (20 ticks)."""
        g = TopologicalGraph()
        n1 = g.add_node("N1", 0.0, 0.0)
        n2 = g.add_node("N2", 1000.0, 0.0)  # 30 meters ahead
        g.add_edge("N1", "N2")

        mesh = MeshNetwork()

        # 1. TrolleyAgent (D2RO)
        agent = TrolleyAgent(1, g, "N1", "N2", mesh, max_speed=ROBOT_VMAX_PXPS)
        for _ in range(20):  # 20 ticks @ dt=0.05s = 1.0s
            agent.step(PHYSICS_DT, [], ProxemicsField())

        d2ro_dist_m = agent.total_distance * PX_TO_M
        self.assertAlmostEqual(d2ro_dist_m, 1.20, delta=0.05)

        # 2. Static A* Agent
        astar_agent = StaticAStarAgent(2, g, "N1", "N2", max_speed=ROBOT_VMAX_PXPS)
        for _ in range(20):
            astar_agent.step(PHYSICS_DT, [], ProxemicsField(), 0.0)

        astar_dist_m = astar_agent.total_distance * PX_TO_M
        self.assertAlmostEqual(astar_dist_m, 1.20, delta=0.05)

        # 3. APF Agent
        apf_agent = ArtificialPotentialFieldAgent(3, (0.0, 0.0), (1000.0, 0.0), max_speed=ROBOT_VMAX_PXPS)
        for _ in range(20):
            apf_agent.step(PHYSICS_DT)

        apf_dist_m = apf_agent.total_distance * PX_TO_M
        self.assertAlmostEqual(apf_dist_m, 1.20, delta=0.05)

        # 4. ORCA Agent
        orca_agent = ORCAAgent(4, (0.0, 0.0), (1000.0, 0.0), max_speed=ROBOT_VMAX_PXPS)
        for _ in range(20):
            orca_agent.step(PHYSICS_DT)

        orca_dist_m = orca_agent.total_distance * PX_TO_M
        self.assertAlmostEqual(orca_dist_m, 1.20, delta=0.05)

    def test_dstar_lite_multinode_km(self):
        """Verify D* Lite accumulated heuristic km shift across multi-node movements N1 -> N2 -> N3 -> N4."""
        g = TopologicalGraph()
        g.add_node("N1", 0.0, 0.0)
        g.add_node("N2", 100.0, 0.0)
        g.add_node("N3", 200.0, 0.0)
        g.add_node("N4", 300.0, 0.0)
        g.add_edge("N1", "N2")
        g.add_edge("N2", "N3")
        g.add_edge("N3", "N4")

        dstar = DStarLite(g, "N1", "N4")
        dstar.compute_shortest_path()
        self.assertEqual(dstar.km, 0.0)

        # Move N1 -> N2 (100 px = 3.0 m)
        dstar.update_start("N2")
        self.assertAlmostEqual(dstar.km, 3.0, delta=1e-4)

        # Move N2 -> N3 (100 px = 3.0 m)
        dstar.update_start("N3")
        self.assertAlmostEqual(dstar.km, 6.0, delta=1e-4)

        # Move N3 -> N4 (100 px = 3.0 m)
        dstar.update_start("N4")
        self.assertAlmostEqual(dstar.km, 9.0, delta=1e-4)

    def test_exponential_mesh_decay(self):
        """Verify exponential mesh decay on topological graph edges."""
        g = TopologicalGraph()
        g.add_node("N1", 0.0, 0.0)
        g.add_node("N2", 100.0, 0.0)
        g.add_edge("N1", "N2")

        g.update_mesh_penalty("N1", "N2", 100.0)
        self.assertEqual(g.get_edge("N1", "N2").w_mesh, 100.0)

        # Decay for 5.0 s at lambda = 0.1386294 (5s half-life) -> 100 * 0.5 = 50.0
        g.decay_mesh_penalties(5.0, decay_rate=0.1386294)
        val = g.get_edge("N1", "N2").w_mesh
        self.assertAlmostEqual(val, 50.0, delta=0.1)

    def test_trapezoidal_proxemic_integral(self):
        """Verify trapezoidal numerical integration of proxemics field over edge length."""
        pf = ProxemicsField()
        human = Human(id=1, x=50.0, y=10.0, heading=math.pi / 2.0)
        
        p1 = (0.0, 0.0)
        p2 = (100.0, 0.0)
        
        # 100 px = 3.0 m segment
        integral = pf.compute_edge_segment_penalty(p1, p2, [human])
        self.assertGreater(integral, 0.0, "Integral over human proxemic field should be positive")

if __name__ == "__main__":
    unittest.main()
