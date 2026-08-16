"""
Unit Tests for Hospital Layout & Head-On Corridor Negotiation.
"""

import unittest
from d2ro.environments.hospital import HospitalLayout, HospitalScenarioSuite
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import ProxemicsField

class TestHospitalFramework(unittest.TestCase):
    def test_hospital_layout_graph_connectivity(self):
        """Verify that all hospital departments (ER, Wards, OR, MRI, ICU) are reachable."""
        layout = HospitalLayout()
        self.assertIn("N_ER_TRIAGE", layout.graph.nodes)
        self.assertIn("N_OR_SUITE", layout.graph.nodes)
        self.assertIn("N_MRI_CT", layout.graph.nodes)
        self.assertIn("N_ICU_DISCHARGE", layout.graph.nodes)

    def test_hospital_head_on_scenario_initialization(self):
        """Verify Scenario B initializes head-on pushchair encounters correctly."""
        layout = HospitalLayout()
        pushchairs, humans, desc = HospitalScenarioSuite.get_scenario("B", layout)
        self.assertEqual(len(pushchairs), 3)
        self.assertEqual(pushchairs[0]["start"], "N_WARD_A_NORTH")
        self.assertEqual(pushchairs[0]["goal"], "N_WARD_A_SOUTH")
        self.assertEqual(pushchairs[1]["start"], "N_WARD_A_SOUTH")
        self.assertEqual(pushchairs[1]["goal"], "N_WARD_A_NORTH")

if __name__ == "__main__":
    unittest.main()
