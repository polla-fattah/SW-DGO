"""
Unified Interactive Multi-Domain Desktop Simulator for D²RO / SW-DGO Framework.
Authors: Polla Fattah, et al.

Master Launcher features:
1. Multi-Domain Selection: Supermarket (Retail), Hospital (Clinical ER/OR), Airport (Open Concourse).
2. Live Algorithm Toggles: D²RO (SW-DGO Proposed) vs. Static A* vs. Reactive Avoidance (ORCA / Potential Field).
3. Real-Time Visual Layers: 2D Gaussian Proxemic Discomfort Halos (H_prox), V2V Mesh Broadcast Waves (W_mesh),
   Trolley Kinetic Safety Clearance Envelopes (S_trolley), Turnout Alcove Locks (R_lock), and Trajectory Trails.
4. Interactive Obstacle Spawning: Click anywhere on the floorplan to spawn real-time human crowds or aisle blocks.
"""

from __future__ import annotations
import os
import sys
import math
import time
import random
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple, Optional, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from d2ro.environments.supermarket import SupermarketLayout, ScenarioSuite as SupermarketScenarios
from d2ro.environments.hospital import HospitalLayout, HospitalScenarioSuite
from d2ro.environments.airport import AirportLayout, AirportScenarioSuite
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import Human, ProxemicsField
from d2ro.baselines import (
    StaticAStarAgent, ArtificialPotentialFieldAgent,
    ORCAAgent, DecentralizedLocalMAPFAgent
)

class UnifiedD2ROApp:
    """Master Unified Graphical Simulator for D²RO Autonomous Fleet Multi-Agent Routing."""
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("D²RO: Socially-Weighted Distributed Graph Optimization (SW-DGO) Simulator")
        self.root.geometry("1280x950")
        self.root.minsize(1100, 850)
        self.root.configure(bg="#0b1120")

        # Simulation State
        self.selected_domain = "supermarket"  # "supermarket", "hospital", "airport"
        self.selected_algorithm = "d2ro"       # "d2ro", "static_astar", "apf", "orca", "local_mapf"
        self.selected_scenario = "A"
        self.is_running = True
        self.sim_speed = 1.0
        self.dt = 0.05
        self.sim_time = 0.0

        # Visual Layer Toggles
        self.show_graph_var = tk.BooleanVar(value=True)
        self.show_proxemics_var = tk.BooleanVar(value=True)
        self.show_safety_bubble_var = tk.BooleanVar(value=True)
        self.show_mesh_waves_var = tk.BooleanVar(value=True)
        self.show_trajectories_var = tk.BooleanVar(value=True)

        # Simulation Entities
        self.layout: Any = None
        self.prox_field = ProxemicsField(amplitude=450.0)
        self.mesh_net: Optional[MeshNetwork] = None
        self.d2ro_agents: List[TrolleyAgent] = []
        self.astar_agents: List[StaticAStarAgent] = []
        self.apf_agents: List[ArtificialPotentialFieldAgent] = []
        self.orca_agents: List[ORCAAgent] = []
        self.mapf_agents: List[DecentralizedLocalMAPFAgent] = []
        self.humans: List[Human] = []
        self.shelf_boxes: List[Tuple[float, float, float, float]] = []
        self.scenario_desc = ""

        # Trajectory History for trails
        self.trajectory_trails: Dict[int, List[Tuple[float, float]]] = {}
        self.mesh_wave_animations: List[Dict[str, Any]] = []

        self._build_ui()
        self.load_environment(self.selected_domain, self.selected_scenario)
        self._sim_loop()

    # ==========================================================================
    # UI CONSTRUCTION
    # ==========================================================================
    def _build_ui(self) -> None:
        # 1. Header Bar
        header_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=8)
        header_frame.pack(fill=tk.X)

        title_box = tk.Frame(header_frame, bg="#0f172a")
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box, text="D²RO Autonomous Fleet Multi-Agent Routing Simulator",
            font=("Segoe UI", 15, "bold"), fg="#38bdf8", bg="#0f172a"
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_box, text="Socially-Weighted Distributed Graph Optimization (SW-DGO) | D* Lite + V2V Mesh + Proxemics",
            font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a"
        )
        subtitle_lbl.pack(anchor="w")

        # Domain Selector Buttons (Right side of header)
        domain_box = tk.Frame(header_frame, bg="#0f172a")
        domain_box.pack(side=tk.RIGHT)

        self.domain_buttons: Dict[str, tk.Button] = {}
        domains = [
            ("supermarket", "🛒 Supermarket Retail"),
            ("hospital", "🏥 Clinical Hospital"),
            ("airport", "✈️ Airport Terminal")
        ]
        for dom_key, dom_label in domains:
            btn = tk.Button(
                domain_box, text=dom_label, font=("Segoe UI", 9, "bold"),
                bg="#1e293b", fg="#cbd5e1", activebackground="#2563eb", activeforeground="#ffffff",
                relief=tk.FLAT, padx=12, pady=6, cursor="hand2",
                command=lambda k=dom_key: self._on_select_domain(k)
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.domain_buttons[dom_key] = btn

        # 2. Controls & Scenario Ribbon
        ribbon_frame = tk.Frame(self.root, bg="#1e293b", padx=16, pady=6)
        ribbon_frame.pack(fill=tk.X, pady=(2, 0))

        # Algorithm Selector
        alg_box = tk.Frame(ribbon_frame, bg="#1e293b")
        alg_box.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(alg_box, text="Algorithm:", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1e293b").pack(side=tk.LEFT, padx=(0, 4))
        self.alg_combo = ttk.Combobox(
            alg_box, values=[
                "D²RO (SW-DGO Proposed)",
                "Static A* (Baseline)",
                "Artificial Potential Fields (APF)",
                "Optimal Reciprocal Collision Avoidance (ORCA)",
                "Decentralized Local MAPF (Hybrid)"
            ],
            state="readonly", width=28, font=("Segoe UI", 9)
        )
        self.alg_combo.current(0)
        self.alg_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)
        self.alg_combo.pack(side=tk.LEFT)

        # Scenario Buttons Frame
        self.scenario_frame = tk.Frame(ribbon_frame, bg="#1e293b")
        self.scenario_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Layer Checkboxes (Right side of ribbon)
        layer_box = tk.Frame(ribbon_frame, bg="#1e293b")
        layer_box.pack(side=tk.RIGHT)

        tk.Checkbutton(layer_box, text="Graph", variable=self.show_graph_var, bg="#1e293b", fg="#cbd5e1",
                       selectcolor="#0f172a", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(layer_box, text="Proxemics Halo", variable=self.show_proxemics_var, bg="#1e293b", fg="#cbd5e1",
                       selectcolor="#0f172a", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(layer_box, text="Safety Bubble", variable=self.show_safety_bubble_var, bg="#1e293b", fg="#cbd5e1",
                       selectcolor="#0f172a", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(layer_box, text="V2V Mesh Waves", variable=self.show_mesh_waves_var, bg="#1e293b", fg="#cbd5e1",
                       selectcolor="#0f172a", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(layer_box, text="Trails", variable=self.show_trajectories_var, bg="#1e293b", fg="#cbd5e1",
                       selectcolor="#0f172a", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)

        # 3. Scenario Banner Description
        self.desc_lbl = tk.Label(
            self.root, text="", font=("Segoe UI", 9, "italic"),
            fg="#38bdf8", bg="#0f172a", padx=16, pady=4, anchor="w"
        )
        self.desc_lbl.pack(fill=tk.X)

        # 4. Interactive Simulation Canvas
        canvas_container = tk.Frame(self.root, bg="#0b1120")
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self.canvas = tk.Canvas(
            canvas_container, bg="#020617", highlightthickness=1,
            highlightbackground="#1e293b", cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # 5. Bottom Status Bar & Telemetry HUD
        bottom_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=8)
        bottom_frame.pack(fill=tk.X)

        # Control Buttons
        btn_box = tk.Frame(bottom_frame, bg="#0f172a")
        btn_box.pack(side=tk.LEFT)

        self.play_btn = tk.Button(
            btn_box, text="⏸ Pause", font=("Segoe UI", 9, "bold"),
            bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8", activeforeground="#ffffff",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self._toggle_play
        )
        self.play_btn.pack(side=tk.LEFT, padx=3)

        restart_btn = tk.Button(
            btn_box, text="🔄 Reset", font=("Segoe UI", 9, "bold"),
            bg="#334155", fg="#ffffff", activebackground="#475569", activeforeground="#ffffff",
            relief=tk.FLAT, padx=12, pady=4, cursor="hand2", command=self._restart_current_scenario
        )
        restart_btn.pack(side=tk.LEFT, padx=3)

        # Speed Selector
        tk.Label(btn_box, text="Speed:", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#0f172a").pack(side=tk.LEFT, padx=(12, 4))
        for spd in [1.0, 2.0, 4.0]:
            spd_btn = tk.Button(
                btn_box, text=f"{int(spd)}x", font=("Segoe UI", 8, "bold"),
                bg="#1e293b", fg="#cbd5e1", relief=tk.FLAT, padx=6, pady=2, cursor="hand2",
                command=lambda s=spd: self._set_speed(s)
            )
            spd_btn.pack(side=tk.LEFT, padx=1)

        # Telemetry HUD
        self.telemetry_lbl = tk.Label(
            bottom_frame, text="", font=("Consolas", 9, "bold"),
            fg="#38bdf8", bg="#1e293b", padx=12, pady=5, relief=tk.RIDGE
        )
        self.telemetry_lbl.pack(side=tk.RIGHT)

    # ==========================================================================
    # DOMAIN & SCENARIO LOADING
    # ==========================================================================
    def _on_select_domain(self, domain_key: str) -> None:
        self.selected_domain = domain_key
        for k, btn in self.domain_buttons.items():
            if k == domain_key:
                btn.configure(bg="#2563eb", fg="#ffffff")
            else:
                btn.configure(bg="#1e293b", fg="#cbd5e1")
        self.load_environment(domain_key, "A")

    def _on_algorithm_change(self, event=None) -> None:
        val = self.alg_combo.get()
        if "D²RO" in val:
            self.selected_algorithm = "d2ro"
        elif "Static" in val:
            self.selected_algorithm = "static_astar"
        elif "Potential" in val:
            self.selected_algorithm = "apf"
        elif "ORCA" in val:
            self.selected_algorithm = "orca"
        elif "Local MAPF" in val:
            self.selected_algorithm = "local_mapf"
        self._restart_current_scenario()

    def _update_scenario_tabs(self) -> None:
        for child in self.scenario_frame.winfo_children():
            child.destroy()

        if self.selected_domain == "supermarket":
            scenarios = [
                ("A", "Scenario A: Aisle 3 Crowd"),
                ("B", "Scenario B: Head-On Mutex"),
                ("C", "Scenario C: Sudden Blockage"),
                ("D", "Scenario D: Action Alley"),
                ("E", "Scenario E: Full Rush Hour")
            ]
        elif self.selected_domain == "hospital":
            scenarios = [
                ("A", "Scenario A: ER Emergency"),
                ("B", "Scenario B: Sterile OR Lock"),
                ("C", "Scenario C: Ward Congestion"),
                ("D", "Scenario D: Turnout Yielding")
            ]
        else:  # airport
            scenarios = [
                ("A", "Scenario A: Security Surge"),
                ("B", "Scenario B: Gate Pier Bottleneck"),
                ("C", "Scenario C: Open Plaza Flow"),
                ("D", "Scenario D: Peak Arrival Waves")
            ]

        self.scenario_tab_buttons = {}
        for s_key, s_label in scenarios:
            btn = tk.Button(
                self.scenario_frame, text=s_label, font=("Segoe UI", 8, "bold"),
                bg="#0f172a" if s_key == self.selected_scenario else "#1e293b",
                fg="#38bdf8" if s_key == self.selected_scenario else "#94a3b8",
                relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
                command=lambda k=s_key: self.load_environment(self.selected_domain, k)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.scenario_tab_buttons[s_key] = btn

    def load_environment(self, domain_key: str, scenario_key: str) -> None:
        self.selected_domain = domain_key
        self.selected_scenario = scenario_key
        self.sim_time = 0.0
        self.trajectory_trails.clear()
        self.mesh_wave_animations.clear()

        self._update_scenario_tabs()

        # Instantiate Domain Architecture & Bounds
        if domain_key == "supermarket":
            self.layout = SupermarketLayout()
            trolley_cfgs, self.humans, self.scenario_desc = SupermarketScenarios.get_scenario(scenario_key, self.layout)
            self.shelf_boxes = [s.bounds for s in self.layout.shelves]
        elif domain_key == "hospital":
            self.layout = HospitalLayout()
            trolley_cfgs, self.humans, self.scenario_desc = HospitalScenarioSuite.get_scenario(scenario_key, self.layout)
            self.shelf_boxes = [r.bounds for r in self.layout.rooms]
        else:  # airport
            self.layout = AirportLayout()
            trolley_cfgs, self.humans, self.scenario_desc = AirportScenarioSuite.get_scenario(scenario_key, self.layout)
            self.shelf_boxes = [s.bounds for s in self.layout.structures]

        self.desc_lbl.configure(text=f"[{domain_key.upper()} — {self.selected_algorithm.upper()}] {self.scenario_desc}")

        # Initialize Algorithms
        self.mesh_net = MeshNetwork(comm_radius=350.0)
        self.d2ro_agents = []
        self.astar_agents = []
        self.apf_agents = []
        self.orca_agents = []
        self.mapf_agents = []

        for c in trolley_cfgs:
            cid = c["id"]
            start_n = c["start"]
            goal_n = c["goal"]
            self.trajectory_trails[cid] = []
            s_node = self.layout.graph.get_node(start_n)
            g_node = self.layout.graph.get_node(goal_n)

            # 1. D2RO Agent
            agent = TrolleyAgent(cid, self.layout.graph, start_n, goal_n, self.mesh_net)
            self.d2ro_agents.append(agent)

            # 2. Static A* Agent
            astar_ag = StaticAStarAgent(cid, self.layout.graph, start_n, goal_n)
            self.astar_agents.append(astar_ag)

            # 3. Artificial Potential Fields Agent
            apf_ag = ArtificialPotentialFieldAgent(cid, (s_node.x, s_node.y), (g_node.x, g_node.y))
            self.apf_agents.append(apf_ag)

            # 4. ORCA Agent
            orca_ag = ORCAAgent(cid, (s_node.x, s_node.y), (g_node.x, g_node.y))
            self.orca_agents.append(orca_ag)

            # 5. Decentralized Local MAPF Agent
            mapf_ag = DecentralizedLocalMAPFAgent(cid, self.layout.graph, start_n, goal_n)
            self.mapf_agents.append(mapf_ag)

    def _restart_current_scenario(self) -> None:
        self.load_environment(self.selected_domain, self.selected_scenario)

    def _toggle_play(self) -> None:
        self.is_running = not self.is_running
        self.play_btn.configure(
            text="▶ Play" if not self.is_running else "⏸ Pause",
            bg="#10b981" if not self.is_running else "#2563eb"
        )

    def _set_speed(self, speed: float) -> None:
        self.sim_speed = speed

    def _on_canvas_click(self, event) -> None:
        """Spawn dynamic human crowd or obstacle on floorplan click."""
        cx, cy = float(event.x), float(event.y)
        new_id = len(self.humans) + 100
        new_human = Human(new_id, cx, cy, speed=0.5, state="browsing")
        self.humans.append(new_human)

        # Trigger V2V Mesh alert animation
        self.mesh_wave_animations.append({
            "x": cx, "y": cy, "radius": 10.0, "max_radius": 180.0, "color": "#f59e0b"
        })

    # ==========================================================================
    # SIMULATION & RENDERING LOOP (60 FPS)
    # ==========================================================================
    def _sim_loop(self) -> None:
        if self.is_running:
            eff_dt = self.dt * self.sim_speed

            # 1. Update Dynamic Humans
            for h in self.humans:
                h.update(eff_dt, self.layout.bounds, self.shelf_boxes)

            # 2. Update Active Fleet Based on Selected Algorithm
            if self.selected_algorithm == "d2ro":
                for a in self.d2ro_agents:
                    a.step(eff_dt, self.humans, self.prox_field, current_sim_time=self.sim_time,
                           shelves=self.shelf_boxes, peer_agents=self.d2ro_agents)
                    self.trajectory_trails[a.agent_id].append((a.x, a.y))
                    if len(self.trajectory_trails[a.agent_id]) > 60:
                        self.trajectory_trails[a.agent_id].pop(0)

                self.layout.graph.decay_mesh_penalties(eff_dt, decay_rate=2.0)

            elif self.selected_algorithm == "static_astar":
                for a in self.astar_agents:
                    a.step(eff_dt, self.humans, self.prox_field, current_sim_time=self.sim_time)
                    self.trajectory_trails[a.agent_id].append((a.x, a.y))
                    if len(self.trajectory_trails[a.agent_id]) > 60:
                        self.trajectory_trails[a.agent_id].pop(0)

            elif self.selected_algorithm == "apf":
                peer_pos = [a.current_pos for a in self.apf_agents]
                for a in self.apf_agents:
                    a.step(eff_dt, peer_pos, self.humans, self.shelf_boxes)
                    self.trajectory_trails[a.agent_id].append((a.x, a.y))
                    if len(self.trajectory_trails[a.agent_id]) > 60:
                        self.trajectory_trails[a.agent_id].pop(0)

            elif self.selected_algorithm == "orca":
                peer_pos = [a.current_pos for a in self.orca_agents]
                for a in self.orca_agents:
                    a.step(eff_dt, peer_pos, self.humans, self.shelf_boxes)
                    self.trajectory_trails[a.agent_id].append((a.x, a.y))
                    if len(self.trajectory_trails[a.agent_id]) > 60:
                        self.trajectory_trails[a.agent_id].pop(0)

            elif self.selected_algorithm == "local_mapf":
                peer_dict = {a.agent_id: a.current_pos for a in self.mapf_agents}
                for a in self.mapf_agents:
                    a.step(eff_dt, peer_dict, self.humans)
                    self.trajectory_trails[a.agent_id].append((a.x, a.y))
                    if len(self.trajectory_trails[a.agent_id]) > 60:
                        self.trajectory_trails[a.agent_id].pop(0)

            # 3. Update V2V Mesh Broadcast Animations
            for wave in self.mesh_wave_animations[:]:
                wave["radius"] += 8.0 * self.sim_speed
                if wave["radius"] >= wave["max_radius"]:
                    self.mesh_wave_animations.remove(wave)

            self.sim_time += eff_dt

        # 4. Render Frame
        self._render_scene()
        self._update_telemetry_hud()

        # Schedule next tick at ~60 FPS (16ms)
        self.root.after(16, self._sim_loop)

    def _render_scene(self) -> None:
        self.canvas.delete("all")

        # 1. Render Domain Architectural Fixtures (Shelves, Walls, Depots)
        self._render_environment_geometry()

        # 2. Render Topological Roadmap Graph (Nodes & Corridors)
        if self.show_graph_var.get():
            self._render_topological_graph()

        # 3. Render 2D Gaussian Proxemic Discomfort Halos
        if self.show_proxemics_var.get():
            self._render_proxemic_halos()

        # 4. Render Dynamic Pedestrians
        self._render_humans()

        # 5. Render Trajectory Trails
        if self.show_trajectories_var.get():
            self._render_trails()

        # 6. Render V2V Mesh Wave Animations
        if self.show_mesh_waves_var.get() and self.selected_algorithm == "d2ro":
            self._render_mesh_waves()

        # 7. Render Active Multi-Agent Fleet
        self._render_agents()

    def _render_environment_geometry(self) -> None:
        if self.selected_domain == "supermarket":
            # Aisles and fixtures
            for shelf in self.layout.shelves:
                min_x, min_y, max_x, max_y = shelf.bounds
                if shelf.category == "aisle":
                    fill_c, out_c = "#1e293b", "#334155"
                elif shelf.category == "checkout":
                    fill_c, out_c = "#312e81", "#4f46e5"
                else:
                    fill_c, out_c = "#064e3b", "#059669"
                self.canvas.create_rectangle(min_x, min_y, max_x, max_y, fill=fill_c, outline=out_c, width=2)
                self.canvas.create_text((min_x + max_x) / 2, (min_y + max_y) / 2, text=shelf.name,
                                       fill="#94a3b8", font=("Segoe UI", 7, "bold"))

        elif self.selected_domain == "hospital":
            for room in self.layout.rooms:
                min_x, min_y, max_x, max_y = room.bounds
                if room.dept_type == "alcove":
                    fill_c, out_c = "#064e3b", "#10b981"  # Emerald alcove
                elif room.dept_type == "or" or room.dept_type == "sterile_or":
                    fill_c, out_c = "#450a0a", "#ef4444"  # Red sterile zone
                elif room.dept_type == "er":
                    fill_c, out_c = "#7c2d12", "#f97316"  # Orange ER trauma
                else:
                    fill_c, out_c = "#1e293b", "#334155"
                self.canvas.create_rectangle(min_x, min_y, max_x, max_y, fill=fill_c, outline=out_c, width=2)
                self.canvas.create_text((min_x + max_x) / 2, (min_y + max_y) / 2, text=room.name,
                                       fill="#94a3b8", font=("Segoe UI", 7, "bold"))

        else:  # airport
            for struct in self.layout.structures:
                min_x, min_y, max_x, max_y = struct.bounds
                if struct.zone_type == "security":
                    fill_c, out_c = "#451a03", "#f59e0b"  # Amber security
                elif struct.zone_type == "gate" or struct.zone_type == "gate_pier":
                    fill_c, out_c = "#172554", "#3b82f6"  # Blue gates
                elif struct.zone_type == "retail":
                    fill_c, out_c = "#064e3b", "#10b981"  # Green retail
                else:
                    fill_c, out_c = "#1e293b", "#334155"
                self.canvas.create_rectangle(min_x, min_y, max_x, max_y, fill=fill_c, outline=out_c, width=2)
                self.canvas.create_text((min_x + max_x) / 2, (min_y + max_y) / 2, text=struct.name,
                                       fill="#94a3b8", font=("Segoe UI", 7, "bold"))

    def _render_topological_graph(self) -> None:
        graph = self.layout.graph
        for (u, v), edge in graph.edges.items():
            nu = graph.get_node(u)
            nv = graph.get_node(v)
            edge_c = "#1e293b"
            w = 1
            if edge.r_lock == math.inf:
                edge_c = "#ef4444"
                w = 3
            elif edge.w_mesh > 50.0:
                edge_c = "#f59e0b"
                w = 2
            elif edge.h_prox > 30.0:
                edge_c = "#ec4899"
                w = 2

            self.canvas.create_line(nu.x, nu.y, nv.x, nv.y, fill=edge_c, width=w)

        for nid, node in graph.nodes.items():
            n_col = "#10b981" if node.is_docking_bay else "#475569"
            r = 5 if node.is_docking_bay else 3
            self.canvas.create_oval(node.x - r, node.y - r, node.x + r, node.y + r, fill=n_col, outline="")

    def _render_proxemic_halos(self) -> None:
        for h in self.humans:
            # 2D Gaussian personal-space halo
            self.canvas.create_oval(
                h.x - 38, h.y - 38, h.x + 38, h.y + 38,
                fill="", outline="#ec4899", width=1, dash=(3, 3)
            )
            # Intimate personal boundary (0.8m)
            self.canvas.create_oval(
                h.x - 24, h.y - 24, h.x + 24, h.y + 24,
                fill="#831843", outline="#f43f5e", width=1
            )

    def _render_humans(self) -> None:
        for h in self.humans:
            self.canvas.create_oval(
                h.x - 10, h.y - 10, h.x + 10, h.y + 10,
                fill="#ec4899", outline="#ffffff", width=1
            )

    def _render_trails(self) -> None:
        colors = ["#38bdf8", "#fbbf24", "#34d399", "#a78bfa", "#f87171"]
        for cid, pts in self.trajectory_trails.items():
            c = colors[(cid - 1) % len(colors)]
            for i in range(1, len(pts)):
                self.canvas.create_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], fill=c, width=2)

    def _render_mesh_waves(self) -> None:
        for wave in self.mesh_wave_animations:
            r = wave["radius"]
            self.canvas.create_oval(
                wave["x"] - r, wave["y"] - r, wave["x"] + r, wave["y"] + r,
                outline=wave["color"], width=2
            )

    def _render_agents(self) -> None:
        agent_list = (
            self.d2ro_agents if self.selected_algorithm == "d2ro" else
            self.astar_agents if self.selected_algorithm == "static_astar" else
            self.apf_agents if self.selected_algorithm == "apf" else
            self.orca_agents if self.selected_algorithm == "orca" else
            self.mapf_agents
        )

        colors = ["#38bdf8", "#fbbf24", "#34d399", "#a78bfa", "#f87171"]

        for idx, a in enumerate(agent_list):
            c = colors[idx % len(colors)]

            # 1. Trolley Kinetic Safety Clearance Bubble (S_trolley)
            if self.show_safety_bubble_var.get() and hasattr(a, "safety_bubble_radius"):
                self.canvas.create_oval(
                    a.x - a.safety_bubble_radius, a.y - a.safety_bubble_radius,
                    a.x + a.safety_bubble_radius, a.y + a.safety_bubble_radius,
                    outline=c, width=1, dash=(4, 4)
                )

            # 2. Physical Directional Chassis
            r = 12.0
            self.canvas.create_oval(
                a.x - r, a.y - r, a.x + r, a.y + r,
                fill=c, outline="#ffffff", width=2
            )

            # Heading arrow
            hx = a.x + math.cos(a.heading) * 16.0
            hy = a.y + math.sin(a.heading) * 16.0
            self.canvas.create_line(a.x, a.y, hx, hy, fill="#ffffff", width=2, arrow=tk.LAST)

            # ID Label
            state_str = getattr(a, "state", "NAV")
            self.canvas.create_text(
                a.x, a.y - 18, text=f"T{a.agent_id} [{state_str[:4]}]",
                fill="#ffffff", font=("Segoe UI", 8, "bold")
            )

    def _update_telemetry_hud(self) -> None:
        agent_list = (
            self.d2ro_agents if self.selected_algorithm == "d2ro" else
            self.astar_agents if self.selected_algorithm == "static_astar" else
            self.apf_agents if self.selected_algorithm == "apf" else
            self.orca_agents if self.selected_algorithm == "orca" else
            self.mapf_agents
        )

        docked = sum(1 for a in agent_list if a.is_docked)
        total = len(agent_list)
        deadlocks = sum(a.deadlock_count for a in agent_list)
        violations = sum(a.proxemic_violations for a in agent_list)
        packets = self.mesh_net.total_packets_transmitted if self.mesh_net and self.selected_algorithm == "d2ro" else 0

        hud_text = (
            f"⏱ Sim Time: {self.sim_time:5.1f}s | "
            f"Fleet: {docked}/{total} Docked | "
            f"Deadlocks: {deadlocks} | "
            f"Proxemic Violations: {violations} | "
            f"V2V Packets: {packets}"
        )
        self.telemetry_lbl.configure(text=hud_text)

def main():
    root = tk.Tk()
    app = UnifiedD2ROApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
