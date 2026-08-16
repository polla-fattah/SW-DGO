"""
V2V Ad-Hoc Mesh Communication Network Simulator.

Implements a store-and-forward multi-hop ad-hoc broadcast network with:
  - TTL-bounded multi-hop forwarding (each relay decrements the hop budget),
  - duplicate suppression via globally unique packet identifiers,
  - per-hop transmission latency,
  - stochastic per-link packet loss drawn from a dedicated RNG stream.

The dedicated RNG stream is important for reproducibility: enabling packet loss
must not perturb the global ``random`` sequence that drives scenario generation,
otherwise the loss model would silently change every other stochastic quantity.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Set

class MessageType(Enum):
    CONGESTION_ALERT = auto()
    LOCK_REQUEST = auto()
    LOCK_GRANT = auto()
    LOCK_DENY = auto()
    LOCK_RELEASE = auto()
    HEARTBEAT = auto()

@dataclass
class MeshPacket:
    """Telemetry packet exchanged over the V2V mesh network."""
    msg_type: MessageType
    sender_id: int
    edge: Tuple[str, str]
    cost_penalty: float = 0.0
    priority: float = 0.0  # Arbitration key (lower = stronger claim)
    ttl: int = 3           # Remaining multi-hop forwarding budget
    timestamp: float = 0.0
    packet_id: int = 0
    hops: int = 0          # Number of relays traversed so far
    deliver_at: float = 0.0  # Simulation time at which this copy becomes readable
    target_id: Optional[int] = None  # For unicast replies (grants/denials)

class MeshNetwork:
    """
    Decentralized V2V Mesh Network Simulator.

    Packets flood outward from the originator: every peer within ``comm_radius``
    receives a copy, and any peer receiving a packet with remaining TTL relays it
    onward from its own position. This is what allows an alert to reach agents
    far beyond the originator's own radio range, which is the mechanism the
    anticipatory horizon-extension claim depends upon.
    """
    def __init__(self, comm_radius: float = 300.0, packet_loss_rate: float = 0.0,
                 latency_s: float = 0.0, seed: int = 12345):
        self.comm_radius = comm_radius
        self.packet_loss_rate = packet_loss_rate
        self.latency_s = latency_s
        self.agents: Dict[int, Any] = {}
        self.inbound_queues: Dict[int, List[MeshPacket]] = {}

        # Telemetry counters
        self.total_packets_transmitted: int = 0
        self.packets_lost: int = 0
        self.duplicates_suppressed: int = 0
        self.relay_transmissions: int = 0
        self.packet_counter: int = 0

        self._seen: Dict[int, Set[int]] = {}
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_agent(self, agent_id: int, agent_ref: Any) -> None:
        # Strong reference. The agent <-> mesh cycle this creates is reclaimed by the
        # cyclic garbage collector between trials, so it needs no special handling.
        # (An earlier weakref-based variant was tried and reverted: it destabilised
        # long runs without solving any real leak.)
        self.agents[agent_id] = agent_ref
        self.inbound_queues[agent_id] = []
        self._seen[agent_id] = set()

    def _resolve(self, agent_id: int):
        """Returns a registered agent (strong reference variant)."""
        return self.agents.get(agent_id)

    def live_agents(self):
        """Yields (agent_id, agent) for every still-live registered agent."""
        for aid in list(self.agents.keys()):
            agent = self._resolve(aid)
            if agent is not None:
                yield aid, agent

    def unregister_agent(self, agent_id: int) -> None:
        self.agents.pop(agent_id, None)
        self.inbound_queues.pop(agent_id, None)
        self._seen.pop(agent_id, None)

    # ------------------------------------------------------------------ #
    # Transmission
    # ------------------------------------------------------------------ #
    def broadcast(self, sender_id: int, msg_type: MessageType, edge: Tuple[str, str],
                  cost_penalty: float = 0.0, priority: float = 0.0, ttl: int = 3,
                  current_time: float = 0.0,
                  target_id: Optional[int] = None) -> int:
        """
        Originates a packet and floods it through the mesh.
        Returns the number of direct (first-hop) recipients.
        """
        if self._resolve(sender_id) is None:
            return 0

        self.packet_counter += 1
        packet = MeshPacket(
            msg_type=msg_type, sender_id=sender_id, edge=edge,
            cost_penalty=cost_penalty, priority=priority, ttl=ttl,
            timestamp=current_time, packet_id=self.packet_counter,
            hops=0, deliver_at=current_time, target_id=target_id
        )
        # The originator must never relay its own packet back to itself.
        self._seen.setdefault(sender_id, set()).add(packet.packet_id)
        return self._flood(packet, sender_id, current_time)

    def _flood(self, packet: MeshPacket, from_id: int, current_time: float) -> int:
        """Delivers one hop outward from ``from_id``, then recursively relays."""
        source = self._resolve(from_id)
        if source is None or packet.ttl <= 0:
            return 0

        sx, sy = source.current_pos
        direct = 0
        newly_reached: List[int] = []

        for peer_id, peer in list(self.live_agents()):
            if peer_id == from_id:
                continue
            seen = self._seen.setdefault(peer_id, set())
            if packet.packet_id in seen:
                self.duplicates_suppressed += 1
                continue

            px, py = peer.current_pos
            if math.hypot(sx - px, sy - py) > self.comm_radius:
                continue

            # Per-link stochastic loss (dedicated RNG stream)
            if self.packet_loss_rate > 0.0 and self._rng.random() < self.packet_loss_rate:
                self.packets_lost += 1
                continue

            hop_no = packet.hops + 1
            arrival = current_time + hop_no * self.latency_s
            seen.add(packet.packet_id)
            self.inbound_queues[peer_id].append(
                replace(packet, ttl=packet.ttl - 1, hops=hop_no, deliver_at=arrival)
            )
            self.total_packets_transmitted += 1
            if hop_no > 1:
                self.relay_transmissions += 1
            direct += 1
            newly_reached.append(peer_id)

        # Store-and-forward: each newly reached node relays onward if TTL remains.
        if packet.ttl - 1 > 0:
            for relay_id in newly_reached:
                self._flood(replace(packet, ttl=packet.ttl - 1, hops=packet.hops + 1),
                            relay_id, current_time)

        return direct

    # ------------------------------------------------------------------ #
    # Reception
    # ------------------------------------------------------------------ #
    def fetch_inbound(self, agent_id: int, current_time: float = math.inf) -> List[MeshPacket]:
        """
        Retrieves packets that have finished propagating by ``current_time``.
        Packets still in flight (per-hop latency) remain queued for a later tick.
        """
        queue = self.inbound_queues.get(agent_id, [])
        if current_time is None or current_time == math.inf:
            self.inbound_queues[agent_id] = []
            return queue

        ready = [p for p in queue if p.deliver_at <= current_time]
        self.inbound_queues[agent_id] = [p for p in queue if p.deliver_at > current_time]
        return ready

    def reset_seen(self) -> None:
        """Clears duplicate-suppression memory (used between independent trials)."""
        for k in self._seen:
            self._seen[k] = set()
