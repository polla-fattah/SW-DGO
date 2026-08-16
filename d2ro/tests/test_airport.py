"""
Unit Tests for Airport Terminal Layout & Autonomous Luggage Trolley Routing.
"""

import unittest
from d2ro.environments.airport import AirportLayout, AirportScenarioSuite

class TestAirportFramework(unittest.TestCase):
    def test_airport_layout_graph_connectivity(self):
        """Verify that open check-in concourses, security lanes, plaza, and gate piers are connected."""
        layout = AirportLayout()
        self.assertIn("N_CHK_80_80", layout.graph.nodes)
        self.assertIn("N_SEC_NORTH", layout.graph.nodes)
        self.assertIn("N_PLZ_510_100", layout.graph.nodes)
        self.assertIn("N_GATE_A1", layout.graph.nodes)
        self.assertIn("N_GATE_B2", layout.graph.nodes)
        self.assertIn("TROLLEY_DEPOT_MAIN", layout.graph.nodes)

    def test_airport_scenario_generation(self):
        """Verify airport scenario suites generate valid dynamic passenger crowds."""
        layout = AirportLayout()
        trolleys, humans, desc = AirportScenarioSuite.get_scenario("A", layout)
        self.assertEqual(len(humans), 16)
        self.assertEqual(len(trolleys), 4)

if __name__ == "__main__":
    unittest.main()
