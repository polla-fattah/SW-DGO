"""
Realistic Hospital Environment & Autonomous Pushchair Routing Roadmap.
Models clinical hospital architecture for autonomous patient pushchair fleets:
- Emergency Department (ER & Triage)
- Operating Theatres (OR Surgical Suite)
- Radiology / MRI / CT Imaging Suite
- Inpatient Patient Wards A & B
- Central Nurse Station & Transfer Concourse
- Sterile Narrow Transfer Corridors & Alcove Holding Bays (for head-on yielding)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from ..core.graph import TopologicalGraph
from ..core.human import Human

@dataclass
class HospitalRoom:
    """Represents a clinical department, surgical room, or structural wall fixture."""
    x: float
    y: float
    w: float
    h: float
    name: str = "Room"
    dept_type: str = "ward"  # "er", "or", "mri", "icu", "ward", "nurse_station", "alcove"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class HospitalLayout:
    """
    Hospital Architectural Layout with multi-corridor topological graph.
    Features sterile narrow corridors, emergency turnouts, and holding alcoves.
    """
    def __init__(self):
        self.graph = TopologicalGraph()
        self.rooms: List[HospitalRoom] = []
        self.holding_alcoves: List[str] = []

        # Hospital Dimensions
        self.width = 1160.0
        self.height = 860.0
        self.bounds = (0.0, 0.0, self.width, self.height)

        # Coordinate Keylines
        self.x_er = 140.0             # Left Emergency Wing
        self.x_ward_a = 340.0         # Inpatient Ward A Corridor
        self.x_central_hub = 580.0     # Central Nurse Station & Elevator Concourse
        self.x_ward_b = 820.0         # Inpatient Ward B Corridor
        self.x_or_mri = 1020.0        # Right Surgical OR & MRI Wing

        self.y_north_hall = 110.0     # North Transfer Corridor
        self.y_mid_hall = 390.0       # Central Cross-Concourse
        self.y_south_hall = 670.0     # South Inpatient Promenade

        self._build_hospital_topology()
        # Seed the static geometric component of S_trolley (Eq. 8).
        self.graph.compute_clearance_penalties(self.obstacle_bounds)

    @property
    def obstacle_bounds(self) -> List[Tuple[float, float, float, float]]:
        """Uniform accessor for solid fixtures (shared across all three domains)."""
        return [r.bounds for r in self.rooms]

    def _build_hospital_topology(self) -> None:
        # 1. WAYPOINT NODES
        # Emergency Wing Nodes
        self.graph.add_node("N_ER_TRIAGE", self.x_er, self.y_north_hall)
        self.graph.add_node("N_ER_TRAUMA", self.x_er, self.y_mid_hall)
        self.graph.add_node("N_ER_AMBULANCE", self.x_er, self.y_south_hall)

        # Ward A Corridor Nodes (Narrow Single-File Section between North and Mid)
        self.graph.add_node("N_WARD_A_NORTH", self.x_ward_a, self.y_north_hall)
        self.graph.add_node("N_WARD_A_MID", self.x_ward_a, self.y_mid_hall)
        self.graph.add_node("N_WARD_A_SOUTH", self.x_ward_a, self.y_south_hall)
        # Ward A Holding Alcove (Turnout bay for passing)
        self.graph.add_node("N_ALCOVE_A", self.x_ward_a + 60.0, 250.0)

        # Central Hub Nodes (Nurse Station & Elevators - Wide Arterial)
        self.graph.add_node("N_CENTRAL_NORTH", self.x_central_hub, self.y_north_hall)
        self.graph.add_node("N_NURSE_STATION", self.x_central_hub, self.y_mid_hall)
        self.graph.add_node("N_CENTRAL_SOUTH", self.x_central_hub, self.y_south_hall)

        # Ward B Corridor Nodes (Narrow Single-File Section between Mid and South)
        self.graph.add_node("N_WARD_B_NORTH", self.x_ward_b, self.y_north_hall)
        self.graph.add_node("N_WARD_B_MID", self.x_ward_b, self.y_mid_hall)
        self.graph.add_node("N_WARD_B_SOUTH", self.x_ward_b, self.y_south_hall)
        # Ward B Holding Alcove
        self.graph.add_node("N_ALCOVE_B", self.x_ward_b - 60.0, 530.0)

        # Surgical OR & Radiology/MRI Nodes
        self.graph.add_node("N_OR_SUITE", self.x_or_mri, self.y_north_hall)
        self.graph.add_node("N_MRI_CT", self.x_or_mri, self.y_mid_hall)
        self.graph.add_node("N_ICU_DISCHARGE", self.x_or_mri, self.y_south_hall)

        # 2. CORRIDOR EDGES
        # Horizontal Thoroughfares (Wide Non-single-file main corridors)
        # North Corridor: ER Triage <-> Ward A <-> Central <-> Ward B <-> OR Suite
        self.graph.add_edge("N_ER_TRIAGE", "N_WARD_A_NORTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_A_NORTH", "N_CENTRAL_NORTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_CENTRAL_NORTH", "N_WARD_B_NORTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_B_NORTH", "N_OR_SUITE", is_single_file=False, bidirectional=True)

        # Mid Corridor: ER Trauma <-> Ward A <-> Nurse Station <-> Ward B <-> MRI/CT
        self.graph.add_edge("N_ER_TRAUMA", "N_WARD_A_MID", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_A_MID", "N_NURSE_STATION", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_NURSE_STATION", "N_WARD_B_MID", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_B_MID", "N_MRI_CT", is_single_file=False, bidirectional=True)

        # South Corridor: ER Ambulance <-> Ward A <-> Central South <-> Ward B <-> ICU
        self.graph.add_edge("N_ER_AMBULANCE", "N_WARD_A_SOUTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_A_SOUTH", "N_CENTRAL_SOUTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_CENTRAL_SOUTH", "N_WARD_B_SOUTH", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_WARD_B_SOUTH", "N_ICU_DISCHARGE", is_single_file=False, bidirectional=True)

        # Vertical Department Aisles (Strictly Single-File Narrow Transfer Corridors)
        # Emergency Wing
        self.graph.add_edge("N_ER_TRIAGE", "N_ER_TRAUMA", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_ER_TRAUMA", "N_ER_AMBULANCE", is_single_file=False, bidirectional=True)

        # Ward A (Narrow Transfer Corridor with Alcove Turnout)
        self.graph.add_edge("N_WARD_A_NORTH", "N_WARD_A_MID", is_single_file=True, bidirectional=True)
        self.graph.add_edge("N_WARD_A_MID", "N_WARD_A_SOUTH", is_single_file=True, bidirectional=True)
        # Alcove A Connections
        self.graph.add_edge("N_WARD_A_NORTH", "N_ALCOVE_A", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_ALCOVE_A", "N_WARD_A_MID", is_single_file=False, bidirectional=True)

        # Central Nurse Concourse (Wide dual-lane corridor)
        self.graph.add_edge("N_CENTRAL_NORTH", "N_NURSE_STATION", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_NURSE_STATION", "N_CENTRAL_SOUTH", is_single_file=False, bidirectional=True)

        # Ward B (Narrow Transfer Corridor with Alcove Turnout)
        self.graph.add_edge("N_WARD_B_NORTH", "N_WARD_B_MID", is_single_file=True, bidirectional=True)
        self.graph.add_edge("N_WARD_B_MID", "N_WARD_B_SOUTH", is_single_file=True, bidirectional=True)
        # Alcove B Connections
        self.graph.add_edge("N_WARD_B_MID", "N_ALCOVE_B", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_ALCOVE_B", "N_WARD_B_SOUTH", is_single_file=False, bidirectional=True)

        # Surgical & MRI Wing (Sterile Single-File Airway)
        self.graph.add_edge("N_OR_SUITE", "N_MRI_CT", is_single_file=True, bidirectional=True)
        self.graph.add_edge("N_MRI_CT", "N_ICU_DISCHARGE", is_single_file=True, bidirectional=True)

        # 3. CLINICAL ROOMS & SOLID WALL FIXTURES
        # Emergency Department Rooms (Left wing between corridors)
        self.rooms.append(HospitalRoom(20, 160, 90, 180, name="ER Triage Bay", dept_type="er"))
        self.rooms.append(HospitalRoom(20, 440, 90, 180, name="Ambulance Bay", dept_type="er"))

        # Inpatient Ward Rooms Block A (Between x_er and x_ward_a)
        self.rooms.append(HospitalRoom(190, 160, 100, 180, name="Ward 101-104", dept_type="ward"))
        self.rooms.append(HospitalRoom(190, 440, 100, 180, name="Ward 105-108", dept_type="ward"))

        # Central Hub Fixtures (Elevators & Pharmacy between Ward A and Central Concourse)
        self.rooms.append(HospitalRoom(420, 160, 100, 180, name="Elevator Bank", dept_type="nurse_station"))
        self.rooms.append(HospitalRoom(420, 440, 100, 180, name="Pharmacy & Lab", dept_type="nurse_station"))

        # Inpatient Ward Rooms Block B (Between Central Concourse and Ward B)
        self.rooms.append(HospitalRoom(680, 160, 100, 180, name="Ward 201-204", dept_type="ward"))
        self.rooms.append(HospitalRoom(680, 440, 100, 180, name="Ward 205-208", dept_type="ward"))

        # Surgical OR & Radiology Wing (Right wing beyond x_or_mri)
        self.rooms.append(HospitalRoom(1060, 160, 80, 180, name="Operating OR-1", dept_type="or"))
        self.rooms.append(HospitalRoom(1060, 440, 80, 180, name="ICU Intensive Care", dept_type="icu"))


class HospitalScenarioSuite:
    """
    Clinical benchmark scenarios demonstrating multi-path planning,
    head-on patient transfer resolution, and emergency priority routing.
    """
    @staticmethod
    def get_scenario(scenario_id: str, layout: HospitalLayout) -> Tuple[List[Dict], List[Human], str]:
        if scenario_id == "A" or scenario_id == "emergency_reroute":
            desc = "Hospital Scenario A: Urgent Trauma Transfer to Operating OR-1. Ward A corridor is congested with medical staff; pushchair detects Gaussian fields and routes through Central Concourse."
            pushchairs = [
                {"id": 1, "start": "N_ER_TRAUMA", "goal": "N_OR_SUITE", "is_emergency": True},
                {"id": 2, "start": "N_WARD_A_SOUTH", "goal": "N_MRI_CT", "is_emergency": False},
                {"id": 3, "start": "N_WARD_B_SOUTH", "goal": "N_ER_AMBULANCE", "is_emergency": False},
            ]
            # Staff crowding Ward A North corridor
            humans = [
                Human(1, layout.x_ward_a, 200.0, speed=0.3, state="browsing"),
                Human(2, layout.x_ward_a, 240.0, speed=0.2, state="browsing"),
                Human(3, layout.x_ward_a, 280.0, speed=0.2, state="browsing"),
                Human(4, layout.x_central_hub + 50, layout.y_mid_hall, speed=0.9, state="walking"),
                Human(5, layout.x_ward_b, 480.0, speed=0.8, state="walking"),
            ]
            return pushchairs, humans, desc

        elif scenario_id == "B" or scenario_id == "head_on_encounter":
            desc = "Hospital Scenario B: Real Head-On Clinical Corridor Encounter. Pushchair 1 (moving patient to MRI) meets Pushchair 2 (returning patient to Ward A) in narrow single-file corridor. Pushchair 1 acquires mutex lock; Pushchair 2 yields into Alcove Turnout Bay to let P1 pass."
            pushchairs = [
                {"id": 1, "start": "N_WARD_A_NORTH", "goal": "N_WARD_A_SOUTH", "is_emergency": False},
                {"id": 2, "start": "N_WARD_A_SOUTH", "goal": "N_WARD_A_NORTH", "is_emergency": False},
                {"id": 3, "start": "N_ER_TRIAGE", "goal": "N_ICU_DISCHARGE", "is_emergency": False},
            ]
            humans = [
                Human(1, layout.x_central_hub, layout.y_north_hall, speed=0.8),
                Human(2, layout.x_ward_b, layout.y_south_hall, speed=0.9),
            ]
            return pushchairs, humans, desc

        elif scenario_id == "C" or scenario_id == "sterile_or_lock":
            desc = "Hospital Scenario C: Sterile Surgical OR Airlock. Pushchair 1 transfers post-op patient from OR to ICU. Trailing pushchairs receive V2V mesh lock broadcast and hold in Central Concourse."
            pushchairs = [
                {"id": 1, "start": "N_OR_SUITE", "goal": "N_ICU_DISCHARGE", "is_emergency": True},
                {"id": 2, "start": "N_MRI_CT", "goal": "N_OR_SUITE", "is_emergency": False},
                {"id": 3, "start": "N_ER_TRAUMA", "goal": "N_MRI_CT", "is_emergency": False},
            ]
            humans = [
                Human(1, layout.x_ward_b, 250.0, speed=0.6),
                Human(2, layout.x_central_hub, 450.0, speed=0.7),
            ]
            return pushchairs, humans, desc

        elif scenario_id == "D" or scenario_id == "code_blue":
            desc = "Hospital Scenario D: Code Blue Resuscitation Alert. Emergency cart broadcasts high-priority V2V signal; standard patient pushchairs yield to the corridor edge."
            pushchairs = [
                {"id": 1, "start": "N_ER_AMBULANCE", "goal": "N_OR_SUITE", "is_emergency": True},
                {"id": 2, "start": "N_WARD_A_MID", "goal": "N_ICU_DISCHARGE", "is_emergency": False},
                {"id": 3, "start": "N_WARD_B_NORTH", "goal": "N_ER_TRAUMA", "is_emergency": False},
                {"id": 4, "start": "N_CENTRAL_SOUTH", "goal": "N_MRI_CT", "is_emergency": False},
            ]
            humans = [
                Human(1, 400.0, layout.y_mid_hall, speed=1.2, state="walking"),
                Human(2, 700.0, layout.y_mid_hall, speed=1.1, state="walking"),
                Human(3, 580.0, 250.0, speed=0.9, state="walking"),
            ]
            return pushchairs, humans, desc

        else:
            desc = "Hospital Scenario E: Shift Change & Full Patient Transit Rush. 6 autonomous pushchairs transferring patients between ER, Wards, OR, and MRI with dynamic medical staff."
            pushchairs = [
                {"id": 1, "start": "N_ER_TRIAGE", "goal": "N_OR_SUITE", "is_emergency": True},
                {"id": 2, "start": "N_WARD_A_NORTH", "goal": "N_MRI_CT", "is_emergency": False},
                {"id": 3, "start": "N_WARD_A_SOUTH", "goal": "N_ICU_DISCHARGE", "is_emergency": False},
                {"id": 4, "start": "N_WARD_B_SOUTH", "goal": "N_ER_TRAUMA", "is_emergency": False},
                {"id": 5, "start": "N_WARD_B_NORTH", "goal": "N_ER_AMBULANCE", "is_emergency": False},
                {"id": 6, "start": "N_NURSE_STATION", "goal": "N_OR_SUITE", "is_emergency": False},
            ]
            random.seed(202)
            humans = []
            for i in range(12):
                humans.append(Human(
                    id=i + 1,
                    x=random.uniform(100.0, layout.width - 100.0),
                    y=random.uniform(80.0, layout.height - 80.0),
                    speed=random.uniform(0.7, 1.4)
                ))
            return pushchairs, humans, desc
