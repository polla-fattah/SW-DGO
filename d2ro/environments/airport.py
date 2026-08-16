"""
Realistic Airport Terminal Layout & Autonomous Luggage Trolley Fleet Routing.
Models international airport terminal architecture featuring:
- Massive Open-Plan Check-in Concourse
- Security Checkpoint Chokepoint Corridors
- Duty-Free & Central Retail Plaza (Open Space with Island Pods)
- Long Narrow Boarding Gate Piers (Gates A1-A4 & Gates B1-B4)
- Multi-Bay Luggage Trolley Stacking Depots
- Heavy dynamic passenger crowds with Gaussian proxemic repulsion

All labels use generic, academic terminology suitable for scientific publications.
Dimensions are optimized to fit comfortably on all standard displays without off-screen clipping.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from ..core.graph import TopologicalGraph
from ..core.human import Human

@dataclass
class AirportStructure:
    """Represents a generic terminal architectural fixture."""
    x: float
    y: float
    w: float
    h: float
    name: str = "Structure"
    zone_type: str = "checkin"  # "checkin", "security", "retail", "gate", "depot"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class AirportLayout:
    """
    Airport Terminal Layout combining wide open concourses with narrow gate corridors.
    Optimized for compact, responsive viewing (980 x 580).
    """
    def __init__(self):
        self.graph = TopologicalGraph()
        self.structures: List[AirportStructure] = []
        self.trolley_depots: List[str] = []

        self.width = 980.0
        self.height = 580.0
        self.bounds = (0.0, 0.0, self.width, self.height)

        # Coordinate Keylines
        self.x_checkin_hall = 120.0       # Wide Open Check-in Concourse
        self.x_security_choke = 300.0     # Narrow Security Screening
        self.x_dutyfree_plaza = 510.0     # Wide Open Central Plaza
        self.x_gate_pier_a = 740.0        # Pier A (Gates A1-A3)
        self.x_gate_pier_b = 880.0        # Pier B (Gates B1-B3)

        self.y_north = 80.0
        self.y_mid = 260.0
        self.y_south = 440.0
        self.y_depot = 510.0

        self._build_airport_topology()
        # Seed the static geometric component of S_trolley (Eq. 8).
        self.graph.compute_clearance_penalties(self.obstacle_bounds)

    @property
    def obstacle_bounds(self) -> List[Tuple[float, float, float, float]]:
        """Uniform accessor for solid fixtures (shared across all three domains)."""
        return [s.bounds for s in self.structures]

    def _build_airport_topology(self) -> None:
        # 1. WAYPOINT ROADMAP
        # Check-in Concourse Open Grid (Wide Open Space)
        chk_xs = [80.0, 150.0, 220.0]
        chk_ys = [80.0, 200.0, 320.0, 440.0]
        for gx in chk_xs:
            for gy in chk_ys:
                self.graph.add_node(f"N_CHK_{int(gx)}_{int(gy)}", gx, gy)

        # Interconnect Open Check-in Concourse (Multi-Directional Mesh)
        for i, gx in enumerate(chk_xs):
            for j, gy in enumerate(chk_ys):
                if i < len(chk_xs) - 1:
                    self.graph.add_edge(f"N_CHK_{int(gx)}_{int(gy)}", f"N_CHK_{int(chk_xs[i+1])}_{int(gy)}", is_single_file=False, bidirectional=True)
                if j < len(chk_ys) - 1:
                    self.graph.add_edge(f"N_CHK_{int(gx)}_{int(gy)}", f"N_CHK_{int(gx)}_{int(chk_ys[j+1])}", is_single_file=False, bidirectional=True)
                if i < len(chk_xs) - 1 and j < len(chk_ys) - 1:
                    self.graph.add_edge(f"N_CHK_{int(gx)}_{int(gy)}", f"N_CHK_{int(chk_xs[i+1])}_{int(chk_ys[j+1])}", is_single_file=False, bidirectional=True)

        # Security Checkpoint Chokepoints (Narrow Bottlenecks)
        self.graph.add_node("N_SEC_NORTH", self.x_security_choke, 160.0)
        self.graph.add_node("N_SEC_MID", self.x_security_choke, 260.0)
        self.graph.add_node("N_SEC_SOUTH", self.x_security_choke, 360.0)

        # Connect Check-in to Security
        self.graph.add_edge("N_CHK_220_200", "N_SEC_NORTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_CHK_220_200", "N_SEC_MID", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_CHK_220_320", "N_SEC_SOUTH", is_single_file=False, bidirectional=True)

        # Central Retail Plaza (Open Space Grid)
        plz_xs = [390.0, 510.0, 630.0]
        plz_ys = [100.0, 200.0, 300.0, 420.0]
        for px in plz_xs:
            for py in plz_ys:
                self.graph.add_node(f"N_PLZ_{int(px)}_{int(py)}", px, py)

        for i, px in enumerate(plz_xs):
            for j, py in enumerate(plz_ys):
                if i < len(plz_xs) - 1:
                    self.graph.add_edge(f"N_PLZ_{int(px)}_{int(py)}", f"N_PLZ_{int(plz_xs[i+1])}_{int(py)}", is_single_file=False, bidirectional=True)
                if j < len(plz_ys) - 1:
                    self.graph.add_edge(f"N_PLZ_{int(px)}_{int(py)}", f"N_PLZ_{int(px)}_{int(plz_ys[j+1])}", is_single_file=False, bidirectional=True)
                if i < len(plz_xs) - 1 and j < len(plz_ys) - 1:
                    self.graph.add_edge(f"N_PLZ_{int(px)}_{int(py)}", f"N_PLZ_{int(plz_xs[i+1])}_{int(plz_ys[j+1])}", is_single_file=False, bidirectional=True)

        # Connect Security to Plaza
        self.graph.add_edge("N_SEC_NORTH", "N_PLZ_390_100", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_SEC_MID", "N_PLZ_390_200", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_SEC_SOUTH", "N_PLZ_390_300", is_single_file=False, bidirectional=True)

        # Boarding Gate Piers (Narrow Single-File Gate Corridors)
        # Pier A (North Gate Corridors)
        self.graph.add_node("N_GATE_A1", self.x_gate_pier_a, 80.0)
        self.graph.add_node("N_GATE_A2", self.x_gate_pier_a, 180.0)
        self.graph.add_node("N_PIER_A_HUB", self.x_gate_pier_a, 260.0)

        self.graph.add_edge("N_GATE_A1", "N_GATE_A2", is_single_file=True, bidirectional=True)
        self.graph.add_edge("N_GATE_A2", "N_PIER_A_HUB", is_single_file=True, bidirectional=True)

        # Pier B (South Gate Corridors)
        self.graph.add_node("N_PIER_B_HUB", self.x_gate_pier_b, 260.0)
        self.graph.add_node("N_GATE_B1", self.x_gate_pier_b, 340.0)
        self.graph.add_node("N_GATE_B2", self.x_gate_pier_b, 440.0)

        self.graph.add_edge("N_PIER_B_HUB", "N_GATE_B1", is_single_file=True, bidirectional=True)
        self.graph.add_edge("N_GATE_B1", "N_GATE_B2", is_single_file=True, bidirectional=True)

        # Connect Plaza to Piers
        self.graph.add_edge("N_PLZ_630_200", "N_GATE_A2", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_PLZ_630_200", "N_PIER_A_HUB", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_PIER_A_HUB", "N_PIER_B_HUB", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_PLZ_630_300", "N_GATE_B1", is_single_file=False, bidirectional=True)

        # Trolley Return Depots (Clearly visible inside viewport)
        self.graph.add_node("TROLLEY_DEPOT_MAIN", 150.0, self.y_depot, is_docking_bay=True)
        self.graph.add_node("TROLLEY_DEPOT_PIER", 510.0, self.y_depot, is_docking_bay=True)
        self.trolley_depots.extend(["TROLLEY_DEPOT_MAIN", "TROLLEY_DEPOT_PIER"])

        self.graph.add_edge("N_CHK_150_440", "TROLLEY_DEPOT_MAIN", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_PLZ_510_420", "TROLLEY_DEPOT_PIER", is_single_file=False, bidirectional=True)

        # 2. SOLID GENERIC ACADEMIC TERMINAL FIXTURES
        # Check-in Desk Islands (Generic non-commercial labels)
        self.structures.append(AirportStructure(30, 110, 35, 60, name="Check-in Bank 1", zone_type="checkin"))
        self.structures.append(AirportStructure(30, 230, 35, 60, name="Check-in Bank 2", zone_type="checkin"))
        self.structures.append(AirportStructure(30, 350, 35, 60, name="Check-in Bank 3", zone_type="checkin"))

        # Security Scanner Blocks
        self.structures.append(AirportStructure(265, 60, 20, 80, name="Security Wall N", zone_type="security"))
        self.structures.append(AirportStructure(265, 190, 20, 50, name="X-Ray Lane 1", zone_type="security"))
        self.structures.append(AirportStructure(265, 290, 20, 50, name="X-Ray Lane 2", zone_type="security"))
        self.structures.append(AirportStructure(265, 390, 20, 80, name="Security Wall S", zone_type="security"))

        # Central Retail Islands (Generic non-commercial retail zones)
        self.structures.append(AirportStructure(430, 130, 50, 50, name="Retail Kiosk 1", zone_type="retail"))
        self.structures.append(AirportStructure(560, 130, 50, 50, name="Retail Kiosk 2", zone_type="retail"))
        self.structures.append(AirportStructure(430, 330, 50, 50, name="Dining Pod A", zone_type="retail"))
        self.structures.append(AirportStructure(560, 330, 50, 50, name="Dining Pod B", zone_type="retail"))

        # Boarding Gate Lounges
        self.structures.append(AirportStructure(775, 60, 45, 50, name="Gate A1 Lounge", zone_type="gate"))
        self.structures.append(AirportStructure(775, 160, 45, 50, name="Gate A2 Lounge", zone_type="gate"))
        self.structures.append(AirportStructure(915, 320, 45, 50, name="Gate B1 Lounge", zone_type="gate"))
        self.structures.append(AirportStructure(915, 420, 45, 50, name="Gate B2 Lounge", zone_type="gate"))


class AirportScenarioSuite:
    """
    Airport benchmark scenarios with dense dynamic pedestrian flows and narrow pier bottlenecks.
    """
    @staticmethod
    def get_scenario(scenario_id: str, layout: AirportLayout) -> Tuple[List[Dict], List[Human], str]:
        if scenario_id == "A" or scenario_id == "open_concourse_crowd":
            desc = "Airport Scenario A: High-Density Check-in Concourse. Trolleys navigate across the open concourse through 16 dynamic passengers using continuous Gaussian proxemics."
            trolleys = [
                {"id": 1, "start": "N_CHK_80_80", "goal": "TROLLEY_DEPOT_MAIN"},
                {"id": 2, "start": "N_CHK_220_80", "goal": "TROLLEY_DEPOT_MAIN"},
                {"id": 3, "start": "N_PLZ_510_100", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 4, "start": "N_GATE_A1", "goal": "TROLLEY_DEPOT_PIER"},
            ]
            # No random.seed() here: the caller owns the seed. This scenario
            # previously reset it to a constant, which made all 100 cross-domain
            # trials byte-identical (n=1 reported as n=100). Scenarios A in the
            # supermarket and hospital suites likewise leave the seed alone.
            humans = []
            for i in range(16):
                humans.append(Human(
                    id=i + 1,
                    x=random.uniform(70.0, 240.0),
                    y=random.uniform(90.0, 430.0),
                    speed=random.uniform(0.6, 1.2),
                    state="walking"
                ))
            return trolleys, humans, desc

        elif scenario_id == "B" or scenario_id == "pier_head_on":
            desc = "Airport Scenario B: Narrow Gate Pier A Head-On Encounter. Trolley 1 returns down Pier A while passenger flow moves upward; Trolley 1 claims mutex lock and resolves conflict smoothly."
            trolleys = [
                {"id": 1, "start": "N_GATE_A1", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 2, "start": "N_PIER_A_HUB", "goal": "N_GATE_A1"},
                {"id": 3, "start": "N_GATE_B2", "goal": "TROLLEY_DEPOT_PIER"},
            ]
            humans = [
                Human(1, layout.x_gate_pier_a, 130.0, speed=0.4, state="browsing"),
                Human(2, layout.x_gate_pier_a, 220.0, speed=0.3, state="browsing"),
                Human(3, 480.0, 260.0, speed=0.9, state="walking"),
            ]
            return trolleys, humans, desc

        elif scenario_id == "C" or scenario_id == "security_bottleneck":
            desc = "Airport Scenario C: Security Checkpoint Surge Alert. Center security screening lane is congested; leading trolley broadcasts V2V CONGESTION_ALERT diverting fleet via North Lane."
            trolleys = [
                {"id": 1, "start": "N_CHK_80_200", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 2, "start": "N_CHK_150_200", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 3, "start": "N_CHK_80_320", "goal": "TROLLEY_DEPOT_MAIN"},
                {"id": 4, "start": "N_GATE_A2", "goal": "TROLLEY_DEPOT_PIER"},
            ]
            humans = [
                Human(1, layout.x_security_choke, 240.0, speed=0.0, state="browsing"),
                Human(2, layout.x_security_choke, 260.0, speed=0.0, state="browsing"),
                Human(3, layout.x_security_choke, 280.0, speed=0.0, state="browsing"),
            ]
            return trolleys, humans, desc

        elif scenario_id == "D" or scenario_id == "duty_free_meandering":
            desc = "Airport Scenario D: Retail Plaza Meander. Passengers wander between central retail kiosks; autonomous trolleys dynamically calculate fluid detour paths."
            trolleys = [
                {"id": 1, "start": "N_SEC_NORTH", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 2, "start": "N_SEC_SOUTH", "goal": "N_GATE_B2"},
                {"id": 3, "start": "N_GATE_A1", "goal": "TROLLEY_DEPOT_MAIN"},
            ]
            humans = [
                Human(1, 450.0, 200.0, speed=1.1, state="walking"),
                Human(2, 550.0, 230.0, speed=1.0, state="walking"),
                Human(3, 500.0, 320.0, speed=1.2, state="walking"),
                Human(4, 420.0, 380.0, speed=0.9, state="walking"),
            ]
            return trolleys, humans, desc

        else:
            desc = "Airport Scenario E: International Peak Rush Hour. 6 autonomous luggage trolleys collecting across Check-in, Security, Plaza, and Gates amid 18 dynamic passengers."
            trolleys = [
                {"id": 1, "start": "N_CHK_80_80", "goal": "TROLLEY_DEPOT_MAIN"},
                {"id": 2, "start": "N_CHK_220_440", "goal": "TROLLEY_DEPOT_MAIN"},
                {"id": 3, "start": "N_PLZ_390_100", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 4, "start": "N_GATE_A1", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 5, "start": "N_GATE_B2", "goal": "TROLLEY_DEPOT_PIER"},
                {"id": 6, "start": "N_PLZ_630_420", "goal": "TROLLEY_DEPOT_MAIN"},
            ]
            random.seed(404)
            humans = []
            for i in range(18):
                humans.append(Human(
                    id=i + 1,
                    x=random.uniform(60.0, layout.width - 60.0),
                    y=random.uniform(60.0, layout.height - 80.0),
                    speed=random.uniform(0.6, 1.3)
                ))
            return trolleys, humans, desc
