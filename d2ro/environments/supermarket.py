"""
Realistic Retail Supermarket Layout and Benchmark Scenarios.
Models authentic supermarket architecture including:
- Front Checkout Bank & Multi-Bay Cart Return Depots
- Grocery Center Aisle Grid (Aisles 1 to 6) with End-Cap Island Displays
- Transverse Action Alley (Middle Arterial Promenade)
- Perimeter Fresh Produce & Bakery Thoroughfares
- Consistent cart return destinations to Cart Depots across all scenarios
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from ..core.graph import TopologicalGraph
from ..core.human import Human

@dataclass
class ShelfObstacle:
    """Rectangular fixture/shelf obstacle with semantic retail department labeling."""
    x: float
    y: float
    w: float
    h: float
    name: str = "Shelf"
    category: str = "grocery"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

class SupermarketLayout:
    """
    Realistic Supermarket Environment with multi-department architecture.
    """
    def __init__(self):
        self.graph = TopologicalGraph()
        self.shelves: List[ShelfObstacle] = []
        self.docking_bays: List[str] = []

        # Coordinate System & Major Thoroughfares
        self.start_x = 180.0
        self.aisle_spacing = 130.0
        self.num_aisles = 6
        
        # Y Coordinate Levels
        self.y_back_promenade = 90.0      # Rear bakery/dairy corridor (wide)
        self.y_shelf_top_start = 140.0
        self.y_action_alley = 300.0       # Middle Action Alley cross-promenade (wide)
        self.y_shelf_bot_end = 460.0
        self.y_front_concourse = 510.0    # Front concourse before checkouts (wide)
        self.y_checkout_registers = 590.0 # Checkout lane register bank
        self.y_cart_depot = 660.0         # Cart return docking depot

        self.bounds = (0.0, 0.0, self.start_x + (self.num_aisles + 1) * self.aisle_spacing + 80.0, self.y_cart_depot + 90.0)

        self._build_realistic_store()
        # Seed the static geometric component of S_trolley (Eq. 8) so that fixture
        # clearance participates in graph optimisation, not only reactive correction.
        self.graph.compute_clearance_penalties(self.obstacle_bounds)

    @property
    def obstacle_bounds(self) -> List[Tuple[float, float, float, float]]:
        """Uniform accessor for solid fixtures (shared across all three domains)."""
        return [s.bounds for s in self.shelves]

    @property
    def width(self) -> float:
        return self.bounds[2]

    @property
    def height(self) -> float:
        return self.bounds[3]

    @property
    def shelf_bounds(self) -> List[Tuple[float, float, float, float]]:
        return [s.bounds for s in self.shelves]

    def _build_realistic_store(self) -> None:
        """Constructs authentic retail topological roadmap and solid fixtures."""
        
        # 1. WAYPOINT NODES AT KEY JUNCTIONS
        for i in range(self.num_aisles):
            ax = self.start_x + i * self.aisle_spacing
            self.graph.add_node(f"N_back_{i}", ax, self.y_back_promenade)
            self.graph.add_node(f"N_mid_{i}", ax, self.y_action_alley)
            self.graph.add_node(f"N_front_{i}", ax, self.y_front_concourse)

        # Left Produce & Right Deli Perimeter Nodes
        x_left_produce = self.start_x - 100.0
        x_right_deli = self.start_x + (self.num_aisles - 1) * self.aisle_spacing + 100.0

        self.graph.add_node("N_produce_back", x_left_produce, self.y_back_promenade)
        self.graph.add_node("N_produce_mid", x_left_produce, self.y_action_alley)
        self.graph.add_node("N_produce_front", x_left_produce, self.y_front_concourse)

        self.graph.add_node("N_deli_back", x_right_deli, self.y_back_promenade)
        self.graph.add_node("N_deli_mid", x_right_deli, self.y_action_alley)
        self.graph.add_node("N_deli_front", x_right_deli, self.y_front_concourse)

        # Cart Return Depots (Front of store)
        self.graph.add_node("DOCK_BAY_MAIN", self.start_x + 2 * self.aisle_spacing, self.y_cart_depot, is_docking_bay=True)
        self.graph.add_node("DOCK_BAY_EXPRESS", self.start_x + 4 * self.aisle_spacing, self.y_cart_depot, is_docking_bay=True)
        self.docking_bays.extend(["DOCK_BAY_MAIN", "DOCK_BAY_EXPRESS"])

        # 2. CORRIDOR EDGES
        # Vertical Grocery Aisles (Narrow Single-File Corridors)
        for i in range(self.num_aisles):
            self.graph.add_edge(f"N_back_{i}", f"N_mid_{i}", is_single_file=True, bidirectional=True)
            self.graph.add_edge(f"N_mid_{i}", f"N_front_{i}", is_single_file=True, bidirectional=True)

        # Left & Right Perimeter Thoroughfares
        self.graph.add_edge("N_produce_back", "N_produce_mid", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_produce_mid", "N_produce_front", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_deli_back", "N_deli_mid", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_deli_mid", "N_deli_front", is_single_file=False, bidirectional=True)

        # Horizontal Arterial Promenades
        self.graph.add_edge("N_produce_back", f"N_back_0", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_produce_mid", f"N_mid_0", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_produce_front", f"N_front_0", is_single_file=False, bidirectional=True)

        for i in range(self.num_aisles - 1):
            self.graph.add_edge(f"N_back_{i}", f"N_back_{i+1}", is_single_file=False, bidirectional=True)
            self.graph.add_edge(f"N_mid_{i}", f"N_mid_{i+1}", is_single_file=False, bidirectional=True)
            self.graph.add_edge(f"N_front_{i}", f"N_front_{i+1}", is_single_file=False, bidirectional=True)

        self.graph.add_edge(f"N_back_{self.num_aisles-1}", "N_deli_back", is_single_file=False, bidirectional=True)
        self.graph.add_edge(f"N_mid_{self.num_aisles-1}", "N_deli_mid", is_single_file=False, bidirectional=True)
        self.graph.add_edge(f"N_front_{self.num_aisles-1}", "N_deli_front", is_single_file=False, bidirectional=True)

        # Connect Front Concourse to Cart Return Depots through checkout lanes
        for i in range(self.num_aisles):
            self.graph.add_edge(f"N_front_{i}", "DOCK_BAY_MAIN", is_single_file=False, bidirectional=True)
            self.graph.add_edge(f"N_front_{i}", "DOCK_BAY_EXPRESS", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_produce_front", "DOCK_BAY_MAIN", is_single_file=False, bidirectional=True)
        self.graph.add_edge("N_deli_front", "DOCK_BAY_EXPRESS", is_single_file=False, bidirectional=True)

        # 3. SOLID RETAIL FIXTURES & SHELVING BLOCKS
        shelf_width = self.aisle_spacing - 46.0
        h_upper_shelf = (self.y_action_alley - self.y_shelf_top_start) - 30.0
        h_lower_shelf = (self.y_shelf_bot_end - (self.y_action_alley + 30.0))

        # Grocery Center Shelves with End-Cap Displays
        for i in range(self.num_aisles - 1):
            sx = self.start_x + i * self.aisle_spacing + 23.0
            self.shelves.append(ShelfObstacle(sx, self.y_shelf_top_start + 15.0, shelf_width, h_upper_shelf, name=f"Aisle {i+1}A", category="grocery"))
            self.shelves.append(ShelfObstacle(sx, self.y_action_alley + 30.0, shelf_width, h_lower_shelf, name=f"Aisle {i+1}B", category="grocery"))

        # Produce Display Islands (positioned along left perimeter wall, clear of x=80 aisle)
        self.shelves.append(ShelfObstacle(15.0, 180.0, 40.0, 80.0, name="Organic Produce", category="island"))
        self.shelves.append(ShelfObstacle(15.0, 350.0, 40.0, 80.0, name="Fresh Fruits", category="island"))

        # Bakery & Deli Counters (positioned along right perimeter wall, clear of x_right_deli aisle)
        self.shelves.append(ShelfObstacle(x_right_deli + 35.0, 180.0, 60.0, 100.0, name="Artisan Bakery", category="deli"))
        self.shelves.append(ShelfObstacle(x_right_deli + 35.0, 350.0, 60.0, 100.0, name="Prepared Foods", category="deli"))

        # Checkout Register Booths (Placed between aisle checkout lanes)
        for i in range(self.num_aisles - 1):
            rx = self.start_x + (i + 0.5) * self.aisle_spacing - 14.0
            self.shelves.append(ShelfObstacle(rx, self.y_checkout_registers - 20.0, 28.0, 40.0, name=f"Register {i+1}", category="checkout"))


class ScenarioSuite:
    """
    Generates realistic operational scenarios for autonomous retail fleet routing.
    In all scenarios, all returning trolleys navigate towards the front Cart Return Depots.
    """
    @staticmethod
    def get_scenario(scenario_id: str, layout: SupermarketLayout) -> Tuple[List[Dict], List[Human], str]:
        if scenario_id == "A" or scenario_id == "crowded_aisle":
            desc = "Scenario A: Heavy Shopper Congestion in Grocery Aisle 3. Carts detect continuous Gaussian personal-space fields and proactively divert through open parallel aisles to Cart Depot."
            trolleys = [
                {"id": 1, "start": "N_back_2", "goal": "DOCK_BAY_MAIN"},
                {"id": 2, "start": "N_back_4", "goal": "DOCK_BAY_EXPRESS"},
                {"id": 3, "start": "N_back_0", "goal": "DOCK_BAY_MAIN"},
                {"id": 4, "start": "N_back_5", "goal": "DOCK_BAY_EXPRESS"},
            ]
            ax3 = layout.start_x + 2 * layout.aisle_spacing
            humans = [
                Human(1, ax3, 190.0, speed=0.2, state="browsing"),
                Human(2, ax3 + 8, 230.0, speed=0.3, state="browsing"),
                Human(3, ax3 - 6, 270.0, speed=0.2, state="browsing"),
                Human(4, ax3, 380.0, speed=0.3, state="browsing"),
                Human(5, ax3, 420.0, speed=0.2, state="browsing"),
                Human(6, layout.start_x + 1 * layout.aisle_spacing, 250.0, speed=0.9),
                Human(7, layout.start_x + 3 * layout.aisle_spacing, 390.0, speed=0.8),
            ]
            return trolleys, humans, desc

        elif scenario_id == "B" or scenario_id == "head_on_lock":
            desc = "Scenario B: Single-File Corridor Mutex Lock. Trolley 1 (top of Aisle 2) & Trolley 2 (bottom of Aisle 2) enter single-file corridor head-on. T1 claims lock, T2 detects active lock and smoothly reroutes."
            trolleys = [
                {"id": 1, "start": "N_back_1", "goal": "DOCK_BAY_MAIN"},
                {"id": 2, "start": "N_front_1", "goal": "N_back_3"},
                {"id": 3, "start": "N_back_3", "goal": "DOCK_BAY_EXPRESS"},
            ]
            humans = [
                Human(1, layout.start_x + 0 * layout.aisle_spacing, 220.0, speed=0.8),
                Human(2, layout.start_x + 3 * layout.aisle_spacing, 400.0, speed=0.9),
            ]
            return trolleys, humans, desc

        elif scenario_id == "C" or scenario_id == "mesh_blockage":
            desc = "Scenario C: Sudden Pallet Blockage in Aisle 1. Trolley 1 encounters a fallen restock box, broadcasts CONGESTION_ALERT over V2V mesh, and trailing fleet detours to Cart Depot before arriving."
            trolleys = [
                {"id": 1, "start": "N_back_0", "goal": "DOCK_BAY_MAIN"},
                {"id": 2, "start": "N_back_0", "goal": "DOCK_BAY_MAIN"},
                {"id": 3, "start": "N_back_1", "goal": "DOCK_BAY_MAIN"},
                {"id": 4, "start": "N_back_5", "goal": "DOCK_BAY_EXPRESS"},
            ]
            ax0 = layout.start_x + 0 * layout.aisle_spacing
            humans = [
                Human(1, ax0, 210.0, speed=0.0, state="browsing"),
                Human(2, ax0, 240.0, speed=0.0, state="browsing"),
                Human(3, ax0, 270.0, speed=0.0, state="browsing"),
            ]
            return trolleys, humans, desc

        elif scenario_id == "D" or scenario_id == "social_crossing":
            desc = "Scenario D: Action Alley Cross-Traffic. Dynamic shoppers cross perpendicular through Action Alley; trolleys smoothly decelerate, yield politely, and resume to Cart Depot."
            trolleys = [
                {"id": 1, "start": "N_back_0", "goal": "DOCK_BAY_MAIN"},
                {"id": 2, "start": "N_back_2", "goal": "DOCK_BAY_MAIN"},
                {"id": 3, "start": "N_back_4", "goal": "DOCK_BAY_EXPRESS"},
            ]
            humans = [
                Human(1, 200.0, layout.y_action_alley, speed=1.2, state="walking"),
                Human(2, 650.0, layout.y_action_alley, speed=1.1, state="walking"),
                Human(3, 400.0, layout.y_front_concourse, speed=1.0, state="walking"),
                Human(4, 750.0, layout.y_front_concourse, speed=1.2, state="walking"),
            ]
            return trolleys, humans, desc

        else:
            desc = "Scenario E: Supermarket Rush Hour. 6 autonomous trolleys navigating across Produce, Bakery, and Grocery aisles to Cart Depots amid 12 dynamic shoppers."
            trolleys = [
                {"id": 1, "start": "N_produce_back", "goal": "DOCK_BAY_MAIN"},
                {"id": 2, "start": "N_back_1", "goal": "DOCK_BAY_MAIN"},
                {"id": 3, "start": "N_back_2", "goal": "DOCK_BAY_MAIN"},
                {"id": 4, "start": "N_back_4", "goal": "DOCK_BAY_EXPRESS"},
                {"id": 5, "start": "N_deli_back", "goal": "DOCK_BAY_EXPRESS"},
                {"id": 6, "start": "N_mid_3", "goal": "DOCK_BAY_EXPRESS"},
            ]
            random.seed(101)
            humans = []
            min_x, min_y, max_x, max_y = layout.bounds
            for i in range(12):
                humans.append(Human(
                    id=i + 1,
                    x=random.uniform(min_x + 90, max_x - 90),
                    y=random.uniform(min_y + 90, max_y - 90),
                    speed=random.uniform(0.7, 1.3)
                ))
            return trolleys, humans, desc
