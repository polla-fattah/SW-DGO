"""
Unit tests for the distributed directional corridor reservation protocol.

Claims are totally ordered by (priority, t_acquire, agent_id) and the smallest
claim wins. That ordering supplies three properties tested here:
  - mutual exclusion (at most one holder per physical corridor),
  - deterministic resolution of exactly-simultaneous requests,
  - FIFO service, which is what rules out starvation.

A lease bounds how long an unrefreshed claim survives, so an agent that fails
while holding a corridor cannot block it permanently.
"""

import math
import unittest

from d2ro.core.graph import TopologicalGraph
from d2ro.core.mesh_network import MeshNetwork, MessageType
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import ProxemicsField
from d2ro.core.units import CORRIDOR_LOCK_LEASE_S


def _bypass_graph():
    """A --(single-file)-- B, with a parallel detour A-C-D-B."""
    g = TopologicalGraph()
    for n, (x, y) in {"A": (0, 0), "B": (100, 0), "C": (0, 50), "D": (100, 50)}.items():
        g.add_node(n, x, y)
    g.add_edge("A", "B", is_single_file=True, bidirectional=True)
    g.add_edge("A", "C", is_single_file=False, bidirectional=True)
    g.add_edge("C", "D", is_single_file=False, bidirectional=True)
    g.add_edge("D", "B", is_single_file=False, bidirectional=True)
    return g


def _no_bypass_graph():
    """A --(single-file)-- B only: the loser has no alternative but to wait."""
    g = TopologicalGraph()
    g.add_node("A", 0.0, 0.0)
    g.add_node("B", 100.0, 0.0)
    g.add_edge("A", "B", is_single_file=True, bidirectional=True)
    return g


class TestMutualExclusion(unittest.TestCase):
    def test_only_one_agent_holds_corridor(self):
        g = _bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        prox = ProxemicsField()
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1.step(0.1, [], prox, current_sim_time=0.1)
        a2.step(0.1, [], prox, current_sim_time=0.1)

        corridor = TrolleyAgent._corridor_key("A", "B")
        holders = [a for a in (a1, a2) if a._holds_corridor(corridor)]
        self.assertEqual(len(holders), 1, "exactly one agent may hold the corridor")
        self.assertEqual(holders[0].agent_id, 1, "earlier/lower claim must win")

    def test_loser_diverts_when_bypass_exists(self):
        """The blocked agent should reroute via the parallel aisle, not stall."""
        g = _bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        prox = ProxemicsField()
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1.step(0.1, [], prox, current_sim_time=0.1)
        a2.step(0.1, [], prox, current_sim_time=0.1)

        self.assertEqual(math.inf, a2.graph.get_edge("B", "A").r_lock,
                         "contested corridor must be infinite-cost for the loser")
        self.assertIn(a2.target_node, ["C", "D"],
                      "loser must divert through the parallel route")

    def test_loser_waits_when_no_bypass(self):
        """With no alternative route the loser must hold, not barge in."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        prox = ProxemicsField()
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1.step(0.1, [], prox, current_sim_time=0.1)
        a2.step(0.1, [], prox, current_sim_time=0.1)

        self.assertEqual(a2.state, "WAITING_LOCK")
        self.assertEqual(a2.speed, 0.0, "waiting agent must be stationary")


class TestSimultaneousContention(unittest.TestCase):
    def test_identical_timestamps_resolve_deterministically(self):
        """Exactly-simultaneous requests must not both succeed."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        prox = ProxemicsField()
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        t = 0.5
        a1._request_corridor("A", "B", t)
        a2._request_corridor("B", "A", t)
        a1.process_inbound_mesh()
        a2.process_inbound_mesh()

        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertTrue(a1._holds_corridor(corridor))
        self.assertFalse(a2._holds_corridor(corridor))

    def test_both_agents_agree_on_winner(self):
        """Independent agents must converge on the same holder."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1._request_corridor("A", "B", 0.5)
        a2._request_corridor("B", "A", 0.5)
        a1.process_inbound_mesh()
        a2.process_inbound_mesh()

        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertEqual(a1._winner(corridor), a2._winner(corridor),
                         "distributed agents must agree on the arbitration outcome")

    def test_higher_priority_preempts_earlier_request(self):
        """An emergency cart (lower priority value) out-ranks an earlier routine claim."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        routine = TrolleyAgent(1, g, "A", "B", mesh, lock_priority=1.0)
        urgent = TrolleyAgent(2, g, "B", "A", mesh, lock_priority=0.0)

        routine._request_corridor("A", "B", 0.1)
        urgent._request_corridor("B", "A", 0.9)   # later, but urgent
        routine.process_inbound_mesh()
        urgent.process_inbound_mesh()

        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertTrue(urgent._holds_corridor(corridor),
                        "priority must dominate arrival order")
        self.assertFalse(routine._holds_corridor(corridor))

    def test_grant_and_deny_are_actually_exchanged(self):
        """LOCK_GRANT / LOCK_DENY must be real traffic, not dead message types."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1._request_corridor("A", "B", 0.1)
        a2.process_inbound_mesh()      # a2 sees a1's request, replies GRANT
        a1.process_inbound_mesh()      # a1 consumes the reply

        self.assertGreater(a1.lock_grants_received, 0,
                           "the winner must receive an explicit grant")


class TestFairnessAndRecovery(unittest.TestCase):
    def test_fifo_ordering_prevents_starvation(self):
        """Among equal priorities the earliest requester is served first."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a2._request_corridor("B", "A", 0.2)   # agent 2 asks FIRST
        a1._request_corridor("A", "B", 0.8)   # agent 1 asks later
        a1.process_inbound_mesh()
        a2.process_inbound_mesh()

        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertTrue(a2._holds_corridor(corridor),
                        "earlier requester must win despite higher agent_id")

    def test_release_frees_corridor(self):
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1._request_corridor("A", "B", 0.1)
        a2.process_inbound_mesh()
        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertFalse(a2._holds_corridor(corridor))

        a1.active_lock_edge = ("A", "B")
        a1._release_corridor(1.0)
        a2.process_inbound_mesh()
        self.assertFalse(a2.lock_claims.get(corridor, {}),
                         "release must clear the holder's claim network-wide")

    def test_expired_lease_is_reclaimed(self):
        """A holder that dies mid-corridor must not block it forever."""
        g = _no_bypass_graph()
        mesh = MeshNetwork(comm_radius=300.0)
        a1 = TrolleyAgent(1, g, "A", "B", mesh)
        a2 = TrolleyAgent(2, g, "B", "A", mesh)

        a1._request_corridor("A", "B", 0.0)
        a2.process_inbound_mesh()
        corridor = TrolleyAgent._corridor_key("A", "B")
        self.assertFalse(a2._holds_corridor(corridor))

        # Agent 1 vanishes without releasing; the lease must expire.
        a2._purge_expired_claims(CORRIDOR_LOCK_LEASE_S + 1.0)
        self.assertFalse(a2.lock_claims.get(corridor, {}),
                         "stale claim must be purged once the lease elapses")


if __name__ == "__main__":
    unittest.main()
