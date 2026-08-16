"""
Native Python Desktop GUI Simulator for Autonomous Hospital Pushchair Fleet.
Renders clinical hospital layout (ER, OR, MRI, Wards, Nurse Stations, Alcoves),
directional pushchairs with kinetic safety envelopes, real-time path trajectories, and head-on conflict resolution.
"""

from __future__ import annotations
import math
import tkinter as tk
from typing import List, Dict, Tuple, Optional
from ..environments.hospital import HospitalLayout, HospitalScenarioSuite
from ..core.mesh_network import MeshNetwork
from ..core.agent import TrolleyAgent
from ..core.human import Human, ProxemicsField

class HospitalSimApp:
    """
    Native Python GUI Application for Hospital Autonomous Pushchair Fleet Simulation.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hospital Autonomous Pushchair Fleet Simulator (SW-DGO / D²RO)")
        self.root.geometry("1220x920")
        self.root.configure(bg="#0b1329")

        # Simulation parameters
        self.layout = HospitalLayout()
        self.prox_field = ProxemicsField(amplitude=500.0, sigma=42.0)
        self.current_scenario_key = "B"
        self.is_running = True
        self.dt = 0.05
        self.sim_time = 0.0

        # Entities
        self.mesh_net: Optional[MeshNetwork] = None
        self.agents: List[TrolleyAgent] = []
        self.humans: List[Human] = []
        self.emergency_flags: Dict[int, bool] = {}
        self.scenario_desc = ""

        # Geometry helpers
        self.room_boxes = [r.bounds for r in self.layout.rooms]
        self.aisle_x = [self.layout.x_er, self.layout.x_ward_a, self.layout.x_central_hub, self.layout.x_ward_b, self.layout.x_or_mri]
        self.crossway_y = [self.layout.y_north_hall, self.layout.y_mid_hall, self.layout.y_south_hall]

        self._create_widgets()
        self.load_scenario("B")
        self._sim_loop()

    def _create_widgets(self) -> None:
        top_frame = tk.Frame(self.root, bg="#0b1329", pady=8)
        top_frame.pack(fill=tk.X, padx=16)

        title_lbl = tk.Label(top_frame, text="Autonomous Hospital Pushchair Simulator — Multi-Path & Head-On Resolution",
                             font=("Segoe UI", 15, "bold"), fg="#38bdf8", bg="#0b1329")
        title_lbl.pack(anchor="w")

        # Scenario Tabs
        tab_frame = tk.Frame(top_frame, bg="#0b1329", pady=6)
        tab_frame.pack(fill=tk.X)

        self.tab_buttons: Dict[str, tk.Button] = {}
        scenarios = [
            ("A", "Scenario A: Urgent Trauma Reroute"),
            ("B", "Scenario B: Real Head-On Encounter (Alcove Yield)"),
            ("C", "Scenario C: Sterile OR Lock"),
            ("D", "Scenario D: Code Blue Priority"),
            ("E", "Scenario E: Full Shift Change Rush")
        ]

        for key, label in scenarios:
            btn = tk.Button(tab_frame, text=label, font=("Segoe UI", 9, "bold"),
                            bg="#1e293b", fg="#94a3b8", activebackground="#0284c7",
                            activeforeground="#ffffff", relief=tk.FLAT, padx=10, pady=4,
                            command=lambda k=key: self.load_scenario(k))
            btn.pack(side=tk.LEFT, padx=3)
            self.tab_buttons[key] = btn

        # Banner Description
        self.desc_lbl = tk.Label(self.root, text="", font=("Segoe UI", 10),
                                 fg="#7dd3fc", bg="#1e293b", padx=12, pady=5, anchor="w")
        self.desc_lbl.pack(fill=tk.X, padx=16, pady=2)

        # Simulation Canvas
        min_x, min_y, max_x, max_y = self.layout.bounds
        self.canvas = tk.Canvas(self.root, width=int(max_x - min_x) + 20, height=int(max_y - min_y) + 20,
                               bg="#050a18", highlightthickness=1, highlightbackground="#1e3a8a")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Bottom Controls & Telemetry
        bottom_frame = tk.Frame(self.root, bg="#0b1329", pady=4)
        bottom_frame.pack(fill=tk.X, padx=16)

        ctrl_frame = tk.Frame(bottom_frame, bg="#0b1329")
        ctrl_frame.pack(side=tk.LEFT)

        self.play_btn = tk.Button(ctrl_frame, text="Pause", font=("Segoe UI", 9, "bold"),
                                  bg="#0284c7", fg="#ffffff", padx=12, pady=3, relief=tk.FLAT,
                                  command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)

        restart_btn = tk.Button(ctrl_frame, text="Restart", font=("Segoe UI", 9, "bold"),
                                bg="#334155", fg="#ffffff", padx=12, pady=3, relief=tk.FLAT,
                                command=lambda: self.load_scenario(self.current_scenario_key))
        restart_btn.pack(side=tk.LEFT, padx=3)

        self.telemetry_lbl = tk.Label(bottom_frame, text="", font=("Consolas", 9),
                                      fg="#f8fafc", bg="#1e293b", padx=10, pady=4)
        self.telemetry_lbl.pack(side=tk.RIGHT)

    def load_scenario(self, key: str) -> None:
        self.current_scenario_key = key
        self.sim_time = 0.0

        for k, btn in self.tab_buttons.items():
            if k == key:
                btn.configure(bg="#0284c7", fg="#ffffff")
            else:
                btn.configure(bg="#1e293b", fg="#94a3b8")

        self.layout = HospitalLayout()
        pushchair_cfgs, self.humans, self.scenario_desc = HospitalScenarioSuite.get_scenario(key, self.layout)
        self.desc_lbl.configure(text=self.scenario_desc)

        self.mesh_net = MeshNetwork(comm_radius=400.0)
        self.agents = []
        self.emergency_flags = {}

        for cfg in pushchair_cfgs:
            speed = 3.2 if cfg.get("is_emergency", False) else 2.5
            agent = TrolleyAgent(
                agent_id=cfg["id"],
                graph=self.layout.graph,
                start_node=cfg["start"],
                goal_node=cfg["goal"],
                mesh_net=self.mesh_net,
                max_speed=speed
            )
            self.agents.append(agent)
            self.emergency_flags[cfg["id"]] = cfg.get("is_emergency", False)

    def _toggle_play(self) -> None:
        self.is_running = not self.is_running
        self.play_btn.configure(text="Pause" if self.is_running else "Play")

    def _on_canvas_click(self, event: tk.Event) -> None:
        click_x, click_y = event.x, event.y
        nearest_node = None
        min_d = 999999.0
        for nid, n in self.layout.graph.nodes.items():
            d = math.hypot(click_x - n.x, click_y - n.y)
            if d < min_d:
                min_d = d
                nearest_node = nid

        if nearest_node and min_d < 50.0:
            for a in self.agents:
                for succ in self.layout.graph.successors(nearest_node):
                    a.broadcast_congestion(nearest_node, succ, penalty=600.0, current_time=self.sim_time)
            print(f"[Hospital Event] Medical Cart Obstacle Placed near {nearest_node}")

    def _sim_loop(self) -> None:
        if self.is_running and self.agents:
            self.sim_time += self.dt

            for h in self.humans:
                h.update(self.dt, self.layout.bounds, self.room_boxes, self.aisle_x, self.crossway_y)

            for a in self.agents:
                a.step(self.dt, self.humans, self.prox_field, current_sim_time=self.sim_time,
                       shelves=self.room_boxes, peer_agents=self.agents)

            self.layout.graph.decay_mesh_penalties(self.dt, decay_rate=2.0)

        self._render()
        self.root.after(30, self._sim_loop)

    def _rotate_point(self, px: float, py: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        nx = cx + (px * cos_a - py * sin_a)
        ny = cy + (px * sin_a + py * cos_a)
        return (nx, ny)

    def _render(self) -> None:
        self.canvas.delete("all")

        # 1. Draw Clinical Rooms & Department Zones
        dept_colors = {
            "er": ("#450a0a", "#dc2626", "#fca5a5"),
            "or": ("#083344", "#0891b2", "#67e8f9"),
            "mri": ("#1e1b4b", "#4f46e5", "#a5b4fc"),
            "icu": ("#2e1065", "#9333ea", "#d8b4fe"),
            "ward": ("#064e3b", "#059669", "#6ee7b7"),
            "nurse_station": ("#0f172a", "#3b82f6", "#93c5fd")
        }

        for r in self.layout.rooms:
            bg_clr, border_clr, txt_clr = dept_colors.get(r.dept_type, ("#1e293b", "#475569", "#cbd5e1"))
            self.canvas.create_rectangle(r.x, r.y, r.x + r.w, r.y + r.h,
                                        fill=bg_clr, outline=border_clr, width=1.8)
            self.canvas.create_text(r.x + r.w/2, r.y + r.h/2, text=r.name,
                                   fill=txt_clr, font=("Segoe UI", 9, "bold"))

        # 2. Draw Corridor Graph Lines
        for (u, v), edge in self.layout.graph.edges.items():
            nu = self.layout.graph.get_node(u)
            nv = self.layout.graph.get_node(v)
            color = "#f43f5e" if edge.is_single_file else "#334155"
            dash = (4, 4) if edge.is_single_file else ()
            w = 2.5 if edge.is_single_file else 1.5
            self.canvas.create_line(nu.x, nu.y, nv.x, nv.y, fill=color, width=w, dash=dash)

        # 3. Draw Holding Alcove Bay Markers
        for alcove_node in ["N_ALCOVE_A", "N_ALCOVE_B"]:
            if alcove_node in self.layout.graph.nodes:
                an = self.layout.graph.get_node(alcove_node)
                self.canvas.create_rectangle(an.x - 14, an.y - 14, an.x + 14, an.y + 14,
                                            outline="#f59e0b", width=1.5, dash=(2, 2))
                self.canvas.create_text(an.x, an.y - 18, text="TURNOUT ALCOVE", fill="#f59e0b", font=("Segoe UI", 7, "bold"))

        # 4. Draw Waypoint Nodes
        for nid, n in self.layout.graph.nodes.items():
            self.canvas.create_oval(n.x - 4, n.y - 4, n.x + 4, n.y + 4, fill="#38bdf8", outline="")

        # 5. Draw Active Planned Path Trajectory Lines
        agent_colors = ["#38bdf8", "#f43f5e", "#a855f7", "#34d399", "#fbbf24", "#ec4899"]
        for idx, a in enumerate(self.agents):
            if a.is_docked:
                continue
            clr = "#f43f5e" if self.emergency_flags.get(a.agent_id, False) else agent_colors[idx % len(agent_colors)]
            path = a.planner.extract_full_path()
            if len(path) > 1:
                if a.target_node:
                    tn = self.layout.graph.get_node(a.target_node)
                    self.canvas.create_line(a.x, a.y, tn.x, tn.y, fill=clr, width=1.8, dash=(2, 2))
                for p_idx in range(len(path) - 1):
                    p_u = self.layout.graph.get_node(path[p_idx])
                    p_v = self.layout.graph.get_node(path[p_idx + 1])
                    self.canvas.create_line(p_u.x, p_u.y, p_v.x, p_v.y, fill=clr, width=1.8, dash=(3, 3))

        # 6. Draw Pedestrians (Doctors, Nurses with Gaussian Bubbles)
        for h in self.humans:
            self.canvas.create_oval(h.x - 42, h.y - 42, h.x + 42, h.y + 42,
                                   outline="#f59e0b", width=1, dash=(3, 3))
            self.canvas.create_oval(h.x - 18, h.y - 18, h.x + 18, h.y + 18,
                                   fill="#78350f", outline="")
            self.canvas.create_oval(h.x - 6, h.y - 6, h.x + 6, h.y + 6,
                                   fill="#fbbf24", outline="#ffffff", width=1.5)

        # 7. Draw Autonomous Patient Pushchairs + Kinetic Safety Clearance Rings
        for a in self.agents:
            is_emergency = self.emergency_flags.get(a.agent_id, False)
            color = "#0284c7"
            badge_text = "PATIENT"
            if is_emergency:
                color = "#dc2626"
                badge_text = "EMERGENCY"
            if a.is_docked:
                color = "#10b981"
                badge_text = "ARRIVED"
            elif a.state == "YIELDING_HUMAN":
                color = "#d97706"
                badge_text = "YIELD"
            elif a.state == "WAITING_LOCK":
                color = "#7c3aed"
                badge_text = "ALCOVE WAIT"
            elif a.state == "FOLLOWING_CART":
                color = "#06b6d4"
                badge_text = "SPACING"

            # Draw Pushchair Safety Ring (S_trolley)
            if not a.is_docked:
                self.canvas.create_oval(a.x - a.safety_bubble_radius, a.y - a.safety_bubble_radius,
                                       a.x + a.safety_bubble_radius, a.y + a.safety_bubble_radius,
                                       outline=color, width=1, dash=(2, 2))

            chair_poly = [
                (12, 6),
                (12, -6),
                (6, -8),
                (-10, -8),
                (-10, 8),
                (6, 8),
            ]

            world_poly = []
            for lx, ly in chair_poly:
                wx, wy = self._rotate_point(lx, ly, a.x, a.y, a.heading)
                world_poly.extend([wx, wy])

            self.canvas.create_polygon(world_poly, fill=color, outline="#ffffff", width=1.8)

            # Seated Patient Circle
            px, py = self._rotate_point(-2, 0, a.x, a.y, a.heading)
            self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#ffffff", outline="")

            # Rear Push Handle Bar
            h1 = self._rotate_point(-11, -7, a.x, a.y, a.heading)
            h2 = self._rotate_point(-11, 7, a.x, a.y, a.heading)
            self.canvas.create_line(h1[0], h1[1], h2[0], h2[1], fill="#e2e8f0", width=2.5)

            lbl_color = "#fca5a5" if is_emergency else "#ffffff"
            self.canvas.create_text(a.x, a.y - 18, text=f"P{a.agent_id} ({badge_text})",
                                   fill=lbl_color, font=("Segoe UI", 8, "bold"))

        # 8. Telemetry Display
        replans = sum(a.replan_count for a in self.agents)
        packets = self.mesh_net.total_packets_transmitted if self.mesh_net else 0
        docked = sum(1 for a in self.agents if a.is_docked)
        yielding = sum(1 for a in self.agents if a.state in ["YIELDING_HUMAN", "FOLLOWING_CART"])
        
        telemetry_text = f"Hospital Sim Time: {self.sim_time:.1f}s | Replans: {replans} | V2V Packets: {packets} | Safe Yielding: {yielding} | Transits Completed: {docked}/{len(self.agents)}"
        self.telemetry_lbl.configure(text=telemetry_text)


def launch_hospital_gui():
    root = tk.Tk()
    app = HospitalSimApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_hospital_gui()
