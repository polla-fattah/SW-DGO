"""
Native Python Desktop GUI Simulator for Autonomous Airport Luggage Trolley Fleet.
Renders open-plan check-in concourses, retail plazas, narrow gate piers, and heavy dynamic crowds.
All labeling is generic and scientific. Canvas dimensions are compact and 100% visible on all displays.
"""

from __future__ import annotations
import math
import tkinter as tk
from typing import List, Dict, Tuple, Optional
from ..environments.airport import AirportLayout, AirportScenarioSuite
from ..core.mesh_network import MeshNetwork
from ..core.agent import TrolleyAgent
from ..core.human import Human, ProxemicsField

class AirportSimApp:
    """
    Native Python GUI Application for Airport Autonomous Luggage Trolley Fleet Simulation.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Airport Autonomous Luggage Trolley Simulator (SW-DGO / D²RO)")
        self.root.geometry("1020x720")
        self.root.configure(bg="#030712")

        # Simulation parameters
        self.layout = AirportLayout()
        self.prox_field = ProxemicsField(amplitude=480.0, sigma=36.0)
        self.current_scenario_key = "A"
        self.is_running = True
        self.dt = 0.05
        self.sim_time = 0.0

        # Entities
        self.mesh_net: Optional[MeshNetwork] = None
        self.agents: List[TrolleyAgent] = []
        self.humans: List[Human] = []
        self.scenario_desc = ""

        # Geometry helpers
        self.struct_boxes = [s.bounds for s in self.layout.structures]
        self.aisle_x = [80.0, 150.0, 220.0, self.layout.x_security_choke, 390.0, 510.0, 630.0, self.layout.x_gate_pier_a, self.layout.x_gate_pier_b]
        self.crossway_y = [80.0, 180.0, 260.0, 340.0, 440.0, 510.0]

        self._create_widgets()
        self.load_scenario("A")
        self._sim_loop()

    def _create_widgets(self) -> None:
        top_frame = tk.Frame(self.root, bg="#030712", pady=6)
        top_frame.pack(fill=tk.X, padx=12)

        title_lbl = tk.Label(top_frame, text="Autonomous Airport Trolley Simulator — Open Concourses & Heavy Crowds",
                             font=("Segoe UI", 14, "bold"), fg="#38bdf8", bg="#030712")
        title_lbl.pack(anchor="w")

        # Scenario Tabs
        tab_frame = tk.Frame(top_frame, bg="#030712", pady=4)
        tab_frame.pack(fill=tk.X)

        self.tab_buttons: Dict[str, tk.Button] = {}
        scenarios = [
            ("A", "Scenario A: Open Check-in Concourse (16 Travelers)"),
            ("B", "Scenario B: Gate Pier A Head-On Encounter"),
            ("C", "Scenario C: Security Chokepoint Surge Alert"),
            ("D", "Scenario D: Retail Plaza Meander"),
            ("E", "Scenario E: Peak Rush Hour")
        ]

        for key, label in scenarios:
            btn = tk.Button(tab_frame, text=label, font=("Segoe UI", 8, "bold"),
                            bg="#111827", fg="#9ca3af", activebackground="#0284c7",
                            activeforeground="#ffffff", relief=tk.FLAT, padx=8, pady=3,
                            command=lambda k=key: self.load_scenario(k))
            btn.pack(side=tk.LEFT, padx=2)
            self.tab_buttons[key] = btn

        # Banner Description
        self.desc_lbl = tk.Label(self.root, text="", font=("Segoe UI", 9),
                                 fg="#7dd3fc", bg="#111827", padx=10, pady=4, anchor="w")
        self.desc_lbl.pack(fill=tk.X, padx=12, pady=2)

        # Simulation Canvas (Compact & 100% visible on screen)
        min_x, min_y, max_x, max_y = self.layout.bounds
        self.canvas = tk.Canvas(self.root, width=int(max_x - min_x) + 10, height=int(max_y - min_y) + 10,
                               bg="#090d16", highlightthickness=1, highlightbackground="#1f2937")
        self.canvas.pack(padx=12, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Bottom Controls & Telemetry
        bottom_frame = tk.Frame(self.root, bg="#030712", pady=4)
        bottom_frame.pack(fill=tk.X, padx=12)

        ctrl_frame = tk.Frame(bottom_frame, bg="#030712")
        ctrl_frame.pack(side=tk.LEFT)

        self.play_btn = tk.Button(ctrl_frame, text="Pause", font=("Segoe UI", 9, "bold"),
                                  bg="#0284c7", fg="#ffffff", padx=10, pady=2, relief=tk.FLAT,
                                  command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=2)

        restart_btn = tk.Button(ctrl_frame, text="Restart", font=("Segoe UI", 9, "bold"),
                                bg="#374151", fg="#ffffff", padx=10, pady=2, relief=tk.FLAT,
                                command=lambda: self.load_scenario(self.current_scenario_key))
        restart_btn.pack(side=tk.LEFT, padx=2)

        self.telemetry_lbl = tk.Label(bottom_frame, text="", font=("Consolas", 9),
                                      fg="#f8fafc", bg="#111827", padx=10, pady=3)
        self.telemetry_lbl.pack(side=tk.RIGHT)

    def load_scenario(self, key: str) -> None:
        self.current_scenario_key = key
        self.sim_time = 0.0

        for k, btn in self.tab_buttons.items():
            if k == key:
                btn.configure(bg="#0284c7", fg="#ffffff")
            else:
                btn.configure(bg="#111827", fg="#9ca3af")

        self.layout = AirportLayout()
        trolley_cfgs, self.humans, self.scenario_desc = AirportScenarioSuite.get_scenario(key, self.layout)
        self.desc_lbl.configure(text=self.scenario_desc)

        self.mesh_net = MeshNetwork(comm_radius=350.0)
        self.agents = []

        for cfg in trolley_cfgs:
            agent = TrolleyAgent(
                agent_id=cfg["id"],
                graph=self.layout.graph,
                start_node=cfg["start"],
                goal_node=cfg["goal"],
                mesh_net=self.mesh_net,
                max_speed=2.6
            )
            self.agents.append(agent)

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

        if nearest_node and min_d < 45.0:
            for a in self.agents:
                for succ in self.layout.graph.successors(nearest_node):
                    a.broadcast_congestion(nearest_node, succ, penalty=MESH_ALERT_EQUIV_M, current_time=self.sim_time)
            print(f"[Airport Event] Dynamic Obstacle Placed near {nearest_node}")

    def _sim_loop(self) -> None:
        if self.is_running and self.agents:
            self.sim_time += self.dt

            for h in self.humans:
                h.update(self.dt, self.layout.bounds, self.struct_boxes, self.aisle_x, self.crossway_y)

            for a in self.agents:
                a.step(self.dt, self.humans, self.prox_field, current_sim_time=self.sim_time, shelves=self.struct_boxes)
                # Hard perimeter boundary clamping to prevent going off screen
                a.x = max(20.0, min(self.layout.width - 20.0, a.x))
                a.y = max(20.0, min(self.layout.height - 20.0, a.y))

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

        # Zone Backgrounds
        # 1. Check-in Concourse
        self.canvas.create_rectangle(15, 35, 250, 480, fill="#0f172a", outline="#1e293b", width=1, stipple="gray25")
        self.canvas.create_text(130, 48, text="CHECK-IN CONCOURSE", fill="#38bdf8", font=("Segoe UI", 8, "bold"))

        # 2. Security Screening Zone
        self.canvas.create_rectangle(255, 35, 340, 480, fill="#1c1917", outline="#44403c", width=1, stipple="gray25")
        self.canvas.create_text(298, 48, text="SECURITY", fill="#f59e0b", font=("Segoe UI", 7, "bold"))

        # 3. Central Retail Plaza
        self.canvas.create_rectangle(345, 35, 680, 480, fill="#064e3b", outline="#047857", width=1, stipple="gray25")
        self.canvas.create_text(510, 48, text="CENTRAL RETAIL & DINING PLAZA", fill="#10b981", font=("Segoe UI", 8, "bold"))

        # 4. Gate Piers A & B
        self.canvas.create_rectangle(690, 35, 965, 480, fill="#1e1b4b", outline="#3730a3", width=1, stipple="gray25")
        self.canvas.create_text(825, 48, text="BOARDING GATES (PIER A & B)", fill="#818cf8", font=("Segoe UI", 8, "bold"))

        # 1. Draw Terminal Fixtures & Structures
        zone_colors = {
            "checkin": ("#1e293b", "#0284c7", "#7dd3fc"),
            "security": ("#451a03", "#d97706", "#fde68a"),
            "retail": ("#022c22", "#059669", "#6ee7b7"),
            "gate": ("#172554", "#3b82f6", "#bfdbfe")
        }

        for s in self.layout.structures:
            bg_clr, border_clr, txt_clr = zone_colors.get(s.zone_type, ("#1f2937", "#4b5563", "#e5e7eb"))
            self.canvas.create_rectangle(s.x, s.y, s.x + s.w, s.y + s.h,
                                        fill=bg_clr, outline=border_clr, width=1.5)
            self.canvas.create_text(s.x + s.w/2, s.y + s.h/2, text=s.name,
                                   fill=txt_clr, font=("Segoe UI", 7, "bold"))

        # 2. Draw Corridor Roadmap
        for (u, v), edge in self.layout.graph.edges.items():
            nu = self.layout.graph.get_node(u)
            nv = self.layout.graph.get_node(v)
            color = "#f43f5e" if edge.is_single_file else "#374151"
            dash = (4, 4) if edge.is_single_file else ()
            w = 2.0 if edge.is_single_file else 1.0
            self.canvas.create_line(nu.x, nu.y, nv.x, nv.y, fill=color, width=w, dash=dash)

        # 3. Draw Nodes & Trolley Depots
        for nid, n in self.layout.graph.nodes.items():
            if n.is_docking_bay:
                self.canvas.create_oval(n.x - 9, n.y - 9, n.x + 9, n.y + 9, fill="#10b981", outline="#ffffff", width=1.5)
                self.canvas.create_text(n.x, n.y + 14, text="TROLLEY DEPOT", fill="#10b981", font=("Segoe UI", 7, "bold"))
            else:
                self.canvas.create_oval(n.x - 3, n.y - 3, n.x + 3, n.y + 3, fill="#38bdf8", outline="")

        # 4. Draw Active Path Lines
        agent_colors = ["#38bdf8", "#f43f5e", "#a855f7", "#34d399", "#fbbf24", "#ec4899"]
        for idx, a in enumerate(self.agents):
            if a.is_docked:
                continue
            clr = agent_colors[idx % len(agent_colors)]
            path = a.planner.extract_full_path()
            if len(path) > 1:
                if a.target_node:
                    tn = self.layout.graph.get_node(a.target_node)
                    self.canvas.create_line(a.x, a.y, tn.x, tn.y, fill=clr, width=1.5, dash=(2, 2))
                for p_idx in range(len(path) - 1):
                    p_u = self.layout.graph.get_node(path[p_idx])
                    p_v = self.layout.graph.get_node(path[p_idx + 1])
                    self.canvas.create_line(p_u.x, p_u.y, p_v.x, p_v.y, fill=clr, width=1.5, dash=(3, 3))

        # 5. Draw Passengers with Gaussian Halos
        for h in self.humans:
            self.canvas.create_oval(h.x - 36, h.y - 36, h.x + 36, h.y + 36,
                                   outline="#f59e0b", width=1, dash=(3, 3))
            self.canvas.create_oval(h.x - 16, h.y - 16, h.x + 16, h.y + 18,
                                   fill="#78350f", outline="")
            self.canvas.create_oval(h.x - 5, h.y - 5, h.x + 5, h.y + 5,
                                   fill="#fbbf24", outline="#ffffff", width=1.5)

        # 6. Draw Directional Luggage Trolley Chassis
        for a in self.agents:
            color = "#0ea5e9"
            badge_text = ""
            if a.is_docked:
                color = "#10b981"
                badge_text = "STACKED"
            elif a.state == "YIELDING_HUMAN":
                color = "#d97706"
                badge_text = "YIELD"
            elif a.state == "WAITING_LOCK":
                color = "#7c3aed"
                badge_text = "PIER LOCK"

            cart_poly = [
                (12, 0),     # Front nose
                (9, 7),      # Front right frame
                (-9, 7),     # Rear right
                (-9, -7),    # Rear left
                (9, -7),     # Front left frame
            ]

            world_poly = []
            for lx, ly in cart_poly:
                wx, wy = self._rotate_point(lx, ly, a.x, a.y, a.heading)
                world_poly.extend([wx, wy])

            self.canvas.create_polygon(world_poly, fill=color, outline="#ffffff", width=1.6)

            # 2 luggage suitcases inside cart
            s1_p = self._rotate_point(-3, 0, a.x, a.y, a.heading)
            s2_p = self._rotate_point(3, 0, a.x, a.y, a.heading)
            self.canvas.create_rectangle(s1_p[0] - 2.5, s1_p[1] - 3, s1_p[0] + 2.5, s1_p[1] + 3, fill="#ffffff", outline="")
            self.canvas.create_rectangle(s2_p[0] - 2.5, s2_p[1] - 3, s2_p[0] + 2.5, s2_p[1] + 3, fill="#ffffff", outline="")

            # Rear push handle
            h1 = self._rotate_point(-10, -6, a.x, a.y, a.heading)
            h2 = self._rotate_point(-10, 6, a.x, a.y, a.heading)
            self.canvas.create_line(h1[0], h1[1], h2[0], h2[1], fill="#e5e7eb", width=2.0)

            # Labels
            self.canvas.create_text(a.x, a.y - 15, text=f"T{a.agent_id}",
                                   fill="#ffffff", font=("Segoe UI", 8, "bold"))
            if badge_text:
                self.canvas.create_text(a.x, a.y + 15, text=badge_text,
                                       fill=color, font=("Segoe UI", 7, "bold"))

        # 7. Telemetry Display
        replans = sum(a.replan_count for a in self.agents)
        packets = self.mesh_net.total_packets_transmitted if self.mesh_net else 0
        docked = sum(1 for a in self.agents if a.is_docked)
        yielding = sum(1 for a in self.agents if a.state == "YIELDING_HUMAN")
        
        telemetry_text = f"Airport Time: {self.sim_time:.1f}s | Replans: {replans} | V2V Packets: {packets} | Yielding: {yielding} | Stacked: {docked}/{len(self.agents)}"
        self.telemetry_lbl.configure(text=telemetry_text)


def launch_airport_gui():
    root = tk.Tk()
    app = AirportSimApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_airport_gui()
