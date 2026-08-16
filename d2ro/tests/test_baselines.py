import unittest
from d2ro.environments.supermarket import SupermarketLayout
from d2ro.core.human import Human
from d2ro.baselines import (
    StaticAStarAgent, ArtificialPotentialFieldAgent,
    ORCAAgent, DecentralizedLocalMAPFAgent
)

class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.env = SupermarketLayout()
        self.start_node = "N_back_0"
        self.goal_node = "DOCK_BAY_MAIN"
        self.start_pos = self.env.graph.get_node(self.start_node).pos
        self.goal_pos = self.env.graph.get_node(self.goal_node).pos
        self.humans = [Human(id=1, x=295.0, y=200.0, heading=0.0)]

    def test_static_astar_agent(self):
        agent = StaticAStarAgent(agent_id=1, graph=self.env.graph,
                                 start_node=self.start_node, goal_node=self.goal_node)
        self.assertEqual(agent.current_node, self.start_node)
        self.assertFalse(agent.is_docked)
        agent.step(dt=0.05, humans=self.humans, prox_field=None, current_sim_time=0.0)
        self.assertGreater(agent.travel_time, 0.0)

    def test_artificial_potential_fields_agent(self):
        agent = ArtificialPotentialFieldAgent(agent_id=2, start_pos=self.start_pos, goal_pos=self.goal_pos)
        self.assertFalse(agent.is_docked)
        agent.step(dt=0.05, peer_positions=[], humans=self.humans, shelf_bounds=self.env.shelf_bounds)
        self.assertGreater(agent.travel_time, 0.0)
        self.assertGreater(agent.total_distance, 0.0)

    def test_orca_agent(self):
        agent = ORCAAgent(agent_id=3, start_pos=self.start_pos, goal_pos=self.goal_pos)
        self.assertFalse(agent.is_docked)
        agent.step(dt=0.05, peer_positions=[], humans=self.humans, shelf_bounds=self.env.shelf_bounds)
        self.assertGreater(agent.travel_time, 0.0)
        self.assertGreater(agent.total_distance, 0.0)

    def test_decentralized_local_mapf_agent(self):
        agent = DecentralizedLocalMAPFAgent(agent_id=4, graph=self.env.graph,
                                            start_node=self.start_node, goal_node=self.goal_node)
        self.assertFalse(agent.is_docked)
        agent.step(dt=0.05, peer_positions={}, humans=self.humans)
        self.assertGreater(agent.travel_time, 0.0)

if __name__ == "__main__":
    unittest.main()
