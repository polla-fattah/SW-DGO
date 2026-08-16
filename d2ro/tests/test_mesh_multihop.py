"""
Unit tests for the multi-hop V2V mesh: TTL-bounded relaying, duplicate
suppression, per-hop latency, and stochastic packet loss.

The central property under test is horizon extension: an alert must reach a peer
that lies FAR outside the originator's own radio range, by being relayed through
an intermediate node. Without this, the anticipatory-communication claim reduces
to plain single-hop broadcast.
"""

import math
import unittest

from d2ro.core.mesh_network import MeshNetwork, MessageType


class _Stub:
    """Minimal stand-in exposing the current_pos interface the mesh requires."""
    def __init__(self, agent_id, x, y):
        self.agent_id = agent_id
        self.x = float(x)
        self.y = float(y)

    @property
    def current_pos(self):
        return (self.x, self.y)


def _chain_network(radius=100.0, **kw):
    """A -- B -- C strung out so that A and C are NOT in direct range."""
    net = MeshNetwork(comm_radius=radius, **kw)
    a, b, c = _Stub(1, 0, 0), _Stub(2, 90, 0), _Stub(3, 180, 0)
    for s in (a, b, c):
        net.register_agent(s.agent_id, s)
    return net, a, b, c


class TestMultiHopForwarding(unittest.TestCase):
    def test_out_of_range_peer_reached_via_relay(self):
        """C is 180 px from A (radius 100) and must still receive the alert."""
        net, a, b, c = _chain_network()
        self.assertGreater(math.hypot(c.x - a.x, c.y - a.y), net.comm_radius,
                           "test premise: C must be outside A's direct range")

        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      cost_penalty=500.0, ttl=3, current_time=0.0)

        self.assertEqual(len(net.fetch_inbound(2)), 1, "B receives directly")
        relayed = net.fetch_inbound(3)
        self.assertEqual(len(relayed), 1, "C must receive via relay through B")
        self.assertEqual(relayed[0].hops, 2, "packet must show two hops")
        self.assertEqual(relayed[0].cost_penalty, 500.0)

    def test_ttl_bounds_propagation(self):
        """TTL=1 permits only a single hop; C must not be reached."""
        net, a, b, c = _chain_network()
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=1, current_time=0.0)
        self.assertEqual(len(net.fetch_inbound(2)), 1, "B still receives directly")
        self.assertEqual(len(net.fetch_inbound(3)), 0,
                         "TTL=1 must not reach the two-hop peer")

    def test_ttl_decrements_along_chain(self):
        net, a, b, c = _chain_network()
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        self.assertEqual(net.fetch_inbound(2)[0].ttl, 2)
        self.assertEqual(net.fetch_inbound(3)[0].ttl, 1)

    def test_originator_never_receives_own_packet(self):
        net, a, b, c = _chain_network()
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        self.assertEqual(len(net.fetch_inbound(1)), 0,
                         "flooding must not echo back to the originator")


class TestDuplicateSuppression(unittest.TestCase):
    def test_each_node_receives_exactly_one_copy(self):
        """Fully-connected cluster: flooding must not multiply copies."""
        net = MeshNetwork(comm_radius=1000.0)
        # The mesh holds agents weakly, so the caller must keep them alive -- exactly
        # as the simulation runner does with its own agent list.
        stubs = [_Stub(i, i * 10.0, 0.0) for i in range(1, 6)]
        for s in stubs:
            net.register_agent(s.agent_id, s)
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        for i in range(2, 6):
            self.assertEqual(len(net.fetch_inbound(i)), 1,
                             f"agent {i} must receive exactly one copy")
        self.assertGreater(net.duplicates_suppressed, 0,
                           "suppression counter must record avoided duplicates")


class TestLatency(unittest.TestCase):
    def test_packet_not_readable_before_arrival(self):
        net, a, b, c = _chain_network(latency_s=0.5)
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        self.assertEqual(len(net.fetch_inbound(3, current_time=0.4)), 0,
                         "two-hop packet must not arrive before 2 x latency")
        self.assertEqual(len(net.fetch_inbound(3, current_time=1.0)), 1,
                         "two-hop packet must arrive by 2 x latency")

    def test_further_hops_arrive_later(self):
        net, a, b, c = _chain_network(latency_s=0.5)
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        self.assertEqual(len(net.fetch_inbound(2, current_time=0.6)), 1,
                         "one-hop packet arrives after a single latency")


class TestPacketLoss(unittest.TestCase):
    def test_total_loss_delivers_nothing(self):
        net, a, b, c = _chain_network(packet_loss_rate=1.0)
        net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                      ttl=3, current_time=0.0)
        self.assertEqual(len(net.fetch_inbound(2)), 0)
        self.assertEqual(len(net.fetch_inbound(3)), 0)
        self.assertGreater(net.packets_lost, 0)

    def test_partial_loss_is_reproducible(self):
        """Loss must be driven by a dedicated RNG so runs stay deterministic."""
        def run():
            net, a, b, c = _chain_network(packet_loss_rate=0.5)
            delivered = 0
            for t in range(40):
                net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                              ttl=3, current_time=float(t))
                delivered += len(net.fetch_inbound(3))
            return delivered

        self.assertEqual(run(), run(),
                         "identical seeds must give identical loss realisations")

    def test_loss_does_not_perturb_global_rng(self):
        import random as _r
        _r.seed(42)
        baseline = [_r.random() for _ in range(5)]

        _r.seed(42)
        net, a, b, c = _chain_network(packet_loss_rate=0.5)
        for t in range(10):
            net.broadcast(1, MessageType.CONGESTION_ALERT, ("U", "V"),
                          ttl=3, current_time=float(t))
        after = [_r.random() for _ in range(5)]
        self.assertEqual(baseline, after,
                         "mesh loss must not consume the global RNG stream")


if __name__ == "__main__":
    unittest.main()
