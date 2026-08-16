"""
Regression tests for three measurement defects found in round-2 review.

Each of these was a case where the code computed something real but the
experiment recorded something else, so the manuscript described a quantity that
was never measured. They are unit-tested here because none of them is visible in
aggregate results -- a wrong-but-plausible number looks exactly like a right one.

  * lock wait time was destroyed on corridor release before the run ended;
  * D* Lite repair latency was reported from a whole-control-step timer;
  * modelled mesh latency was bypassed because the agent never passed sim time.
"""

import math
import unittest

from d2ro.core.agent import TrolleyAgent
from d2ro.core.graph import TopologicalGraph
from d2ro.core.human import ProxemicsField
from d2ro.core.mesh_network import MeshNetwork, MessageType


def _corridor_graph():
    """A --(single-file)-- B only: a blocked agent must wait, it cannot divert."""
    g = TopologicalGraph()
    g.add_node("A", 0.0, 0.0)
    g.add_node("B", 100.0, 0.0)
    g.add_edge("A", "B", is_single_file=True, bidirectional=True)
    return g


def _open_graph():
    g = TopologicalGraph()
    for n, (x, y) in {"A": (0, 0), "B": (100, 0), "C": (200, 0)}.items():
        g.add_node(n, x, y)
    g.add_edge("A", "B", bidirectional=True)
    g.add_edge("B", "C", bidirectional=True)
    return g


class TestLockWaitAccumulation(unittest.TestCase):
    """`total_lock_wait_time` must survive a corridor release."""

    def test_release_does_not_destroy_the_total(self):
        g = _corridor_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "B", mesh)

        a.lock_wait_time = 3.0
        a.total_lock_wait_time = 3.0
        a.active_lock_edge = ("A", "B")

        a._release_corridor(current_time=1.0)

        # The per-episode timer is reset by design; the mission total is not.
        self.assertEqual(a.lock_wait_time, 0.0)
        self.assertEqual(a.total_lock_wait_time, 3.0,
                         "corridor release destroyed the accumulated wait time, "
                         "which is what the experiments read at end of run")

    def test_total_accumulates_across_two_separate_waits(self):
        g = _corridor_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "B", mesh)

        for _ in range(2):
            a.lock_wait_time = 0.0
            for _ in range(4):
                a.lock_wait_time += 0.05
                a.total_lock_wait_time += 0.05
            a.active_lock_edge = ("A", "B")
            a._release_corridor(current_time=1.0)

        self.assertAlmostEqual(a.total_lock_wait_time, 0.4, places=6)


class TestRepairLatencyInstrumentation(unittest.TestCase):
    """Repair latency must come from `compute_shortest_path`, not the whole step."""

    def test_series_starts_empty_and_excludes_the_initial_solve(self):
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "C", mesh)
        # __init__ performs a from-scratch solve, which is not an incremental repair.
        self.assertEqual(a.replan_latencies_ms, [],
                         "the initial full solve must not be counted as a repair")

    def test_each_repair_records_exactly_one_timing(self):
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "C", mesh)

        before_count = a.replan_count
        a._repair()
        a._repair()

        self.assertEqual(len(a.replan_latencies_ms), 2)
        self.assertEqual(a.replan_count, before_count + 2,
                         "replan_count must advance with every repair, so that the "
                         "reported replan total matches the timed repair series")
        for t in a.replan_latencies_ms:
            self.assertGreaterEqual(t, 0.0)

    def test_repair_timing_is_not_the_control_step_timing(self):
        """The two quantities are distinct and must not be conflated."""
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "C", mesh)
        a.step(0.05, [], ProxemicsField(), current_sim_time=0.0,
               shelves=[], peer_agents=[a])
        # A control step always records a step time; it need not have repaired.
        self.assertGreater(a.last_compute_time_ms, 0.0)
        self.assertLessEqual(len(a.replan_latencies_ms), a.replan_count)


class TestMeshLatencyReachesTheAgent(unittest.TestCase):
    """A packet in flight must not be visible to the agent before it lands."""

    def test_agent_does_not_receive_a_packet_before_deliver_at(self):
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=1000.0, latency_s=1.0)
        receiver = TrolleyAgent(1, g, "A", "C", mesh)
        sender = TrolleyAgent(2, g, "C", "A", mesh)

        mesh.broadcast(sender_id=sender.agent_id,
                       msg_type=MessageType.CONGESTION_ALERT,
                       edge=("A", "B"), ttl=3, current_time=0.0)

        # 0.5 s < 1.0 s latency: still in flight.
        receiver.process_inbound_mesh(current_time=0.5)
        self.assertTrue(mesh.inbound_queues.get(receiver.agent_id),
                        "packet was consumed before its deliver_at time, so the "
                        "modelled network latency never affects the agent")

        # After the latency has elapsed it must arrive.
        receiver.process_inbound_mesh(current_time=1.5)
        self.assertFalse(mesh.inbound_queues.get(receiver.agent_id),
                         "packet failed to arrive after its deliver_at time")

    def test_step_threads_simulation_time_into_the_mesh(self):
        """Regression: step() must not fall back to the +inf default."""
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=1000.0, latency_s=5.0)
        receiver = TrolleyAgent(1, g, "A", "C", mesh)
        sender = TrolleyAgent(2, g, "C", "A", mesh)

        mesh.broadcast(sender_id=sender.agent_id,
                       msg_type=MessageType.CONGESTION_ALERT,
                       edge=("A", "B"), ttl=3, current_time=0.0)

        receiver.step(0.05, [], ProxemicsField(), current_sim_time=0.1,
                      shelves=[], peer_agents=[receiver])

        self.assertTrue(mesh.inbound_queues.get(receiver.agent_id),
                        "step() delivered a packet that is still in flight; "
                        "simulation time is not reaching fetch_inbound()")


class TestStaticRouteBaseline(unittest.TestCase):
    """The matched-controller baseline must never re-solve its route."""

    def test_static_route_agent_does_not_replan(self):
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        a = TrolleyAgent(1, g, "A", "C", mesh, static_route=True,
                         enable_mesh=False, enable_prox=False, enable_lock=False)
        prox = ProxemicsField()
        for i in range(40):
            a.step(0.05, [], prox, current_sim_time=i * 0.05,
                   shelves=[], peer_agents=[a])
        self.assertEqual(a.replan_latencies_ms, [],
                         "a frozen-route agent performed a cost-triggered repair")

    def test_static_route_shares_the_kinematic_limits(self):
        """It is only a *matched* controller if the vehicle model is identical."""
        g = _open_graph()
        mesh = MeshNetwork(comm_radius=500.0)
        d2ro = TrolleyAgent(1, g, "A", "C", mesh)
        matched = TrolleyAgent(2, g, "A", "C", mesh, static_route=True,
                               enable_mesh=False, enable_prox=False, enable_lock=False)
        self.assertEqual(matched.max_omega, d2ro.max_omega)
        self.assertEqual(matched.max_speed, d2ro.max_speed)
        self.assertEqual(matched.radius, d2ro.radius)
        self.assertEqual(matched.shelf_margin, d2ro.shelf_margin)


if __name__ == "__main__":
    unittest.main()
