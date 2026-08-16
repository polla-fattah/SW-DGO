"""
Native Python Desktop GUI Visualizer for D²RO / SW-DGO Framework.
Renders realistic multi-department supermarket architecture, directional shopping cart chassis,
trolley kinetic safety clearance envelopes (S_trolley), dynamic trajectory trails, and interactive scenario switching.
"""

from __future__ import annotations
import math
import tkinter as tk
from typing import List, Dict, Tuple, Optional
from ..environments.supermarket import SupermarketLayout, ScenarioSuite
from ..core.mesh_network import MeshNetwork
from ..core.agent import TrolleyAgent
from ..core.human import Human, ProxemicsField

class SupermarketSimApp:
    """
    Native Python GUI Application for D²RO Fleet Simulation.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("D²RO Real-World Supermarket Fleet Simulator")
        self.root.geometry("1180x890")
        self.root.configure(bg="#0f172a")

        # Simulation parameters
        self.layout = SupermarketLayout()
        self.prox_field = ProxemicsField()
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
        self.shelf_boxes = [s.bounds for s in self.layout.shelves]
        self.aisle_x = [self.layout.start_x + i * self.layout.aisle_spacing for i in range(self.layout.num_aisles)]
        self.aisle_x.extend([self.layout.start_x - 100.0, self.layout.start_x + (self.layout.num_aisles - 1) * self.layout.aisle_spacing + 100.0])
        self.crossway_y = [
            self.layout.y_back_promenade,
            self.layout.y_action_alley,
            self.layout.y_front_concourse,
            self.layout.y_cart_depot
        ]

        self._create_widgets()
        self.load_scenario("A")
        self._sim_loop()

    def _create_widgets(self) -> None:
        top_frame = tk.Frame(self.root, bg="#0f172a", pady=8)
        top_frame.pack(fill=tk.X, padx=16)

        title_lbl = tk.Label(top_frame, text="D²RO Autonomous Trolley Simulator — Supermarket Architecture",
                             font=("Segoe UI", 15, "bold"), fg="#ffffff", bg="#0f172a")
        title_lbl.pack(anchor="w")

        # Scenario Tabs
        tab_frame = tk.Frame(top_frame, bg="#0f172a", pady=6)
        tab_frame.pack(fill=tk.X)

        self.tab_buttons: Dict[str, tk.Button] = {}
        scenarios = [
            ("A", "Scenario A: Aisle 3 Crowd"),
            ("B", "Scenario B: Head-On Lock"),
            ("C", "Scenario C: Sudden V2V Blockage"),
            ("D", "Scenario D: Action Alley Crossing"),
            ("E", "Scenario E: Full Rush Hour")
        ]

        for key, label in scenarios:
            btn = tk.Button(tab_frame, text=label, font=("Segoe UI", 9, "bold"),
                            bg="#1e293b", fg="#94a3b8", activebackground="#2563eb",
                            activeforeground="#ffffff", relief=tk.FLAT, padx=10, pady=4,
                            command=lambda k=key: self.load_scenario(k))
            btn.pack(side=tk.LEFT, padx=3)
            self.tab_buttons[key] = btn

        # Banner Description
        self.desc_lbl = tk.Label(self.root, text="", font=("Segoe UI", 10),
                                 fg="#38bdf8", bg="#1e293b", padx=12, pady=5, anchor="w")
        self.desc_lbl.pack(fill=tk.X, padx=16, pady=2)

        # Simulation Canvas
        min_x, min_y, max_x, max_y = self.layout.bounds
        self.canvas = tk.Canvas(self.root, width=int(max_x - min_x) + 40, height=int(max_y - min_y) + 40,
                               bg="#0b1120", highlightthickness=1, highlightbackground="#334155")
        self.canvas.pack(padx=16, pady=6)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Bottom Controls & Telemetry
        bottom_frame = tk.Frame(self.root, bg="#0f172a", pady=4)
        bottom_frame.pack(fill=tk.X, padx=16)

        ctrl_frame = tk.Frame(bottom_frame, bg="#0f172a")
        ctrl_frame.pack(side=tk.LEFT)

        self.play_btn = tk.Button(ctrl_frame, text="Pause", font=("Segoe UI", 9, "bold"),
                                  bg="#2563eb", fg="#ffffff", padx=12, pady=3, relief=tk.FLAT,
                                  command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)

        restart_btn = tk.Button(ctrl_frame, text="Restart", font=("Segoe UI", 9, "bold"),
                                bg="#475569", fg="#ffffff", padx=12, pady=3, relief=tk.FLAT,
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
                btn.configure(bg="#2563eb", fg="#ffffff")
            else:
                btn.configure(bg="#1e293b", fg="#94a3b8")

        self.layout = SupermarketLayout()
        trolley_cfgs, self.humans, self.scenario_desc = ScenarioSuite.get_scenario(key, self.layout)
        self.desc_lbl.configure(text=self.scenario_desc)

        self.mesh_net = MeshNetwork(comm_radius=350.0)
        self.agents = []
        for cfg in trolley_cfgs:
            agent = TrolleyAgent(
                agent_id=cfg["id"],
                graph=self.layout.graph,
                start_node=cfg["start"],
                goal_node=cfg["goal"],
                mesh_net=self.mesh_net
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
            print(f"[User Click] Spawned Dynamic Blockage near {nearest_node}")

    def _sim_loop(self) -> None:
        if self.is_running and self.agents:
            self.sim_time += self.dt

            for h in self.humans:
                h.update(self.dt, self.layout.bounds, self.shelf_boxes, self.aisle_x, self.crossway_y)

            for a in self.agents:
                a.step(self.dt, self.humans, self.prox_field, current_sim_time=self.sim_time,
                       shelves=self.shelf_boxes, peer_agents=self.agents)

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

        # Department Background Zones
        self.canvas.create_rectangle(30, 70, self.layout.start_x - 30, self.layout.y_front_concourse + 20,
                                    fill="#064e3b", outline="#047857", width=1, stipple="gray25")
        self.canvas.create_text(self.layout.start_x - 100, 75, text="PRODUCE & FRESH", fill="#10b981", font=("Segoe UI", 9, "bold"))

        x_deli = self.layout.start_x + (self.layout.num_aisles - 1) * self.layout.aisle_spacing + 100.0
        self.canvas.create_rectangle(x_deli - 80, 70, x_deli + 80, self.layout.y_front_concourse + 20,
                                    fill="#4c0519", outline="#9f1239", width=1, stipple="gray25")
        self.canvas.create_text(x_deli, 75, text="DELI & BAKERY", fill="#f43f5e", font=("Segoe UI", 9, "bold"))

        self.canvas.create_text(self.layout.start_x + 2.5 * self.layout.aisle_spacing, self.layout.y_action_alley - 12,
                               text="— CENTRAL ACTION ALLEY (TRANSVERSE PROMENADE) —", fill="#64748b", font=("Segoe UI", 8, "bold"))

        # 1. Draw Aisles & Crossways
        for (u, v), edge in self.layout.graph.edges.items():
            nu = self.layout.graph.get_node(u)
            nv = self.layout.graph.get_node(v)
            color = "#f43f5e" if edge.is_single_file else "#475569"
            dash = (4, 4) if edge.is_single_file else ()
            w = 2 if edge.is_single_file else 1.5
            self.canvas.create_line(nu.x, nu.y, nv.x, nv.y, fill=color, width=w, dash=dash)

        # 2. Draw Shelves & Store Fixtures
        for s in self.layout.shelves:
            fill_color = "#1e293b"
            border_color = "#334155"
            text_color = "#94a3b8"
            if s.category == "island":
                fill_color = "#064e3b"
                border_color = "#059669"
                text_color = "#6ee7b7"
            elif s.category == "deli":
                fill_color = "#4c0519"
                border_color = "#e11d48"
                text_color = "#fda4af"
            elif s.category == "checkout":
                fill_color = "#312e81"
                border_color = "#6366f1"
                text_color = "#c7d2fe"

            self.canvas.create_rectangle(s.x, s.y, s.x + s.w, s.y + s.h,
                                        fill=fill_color, outline=border_color, width=1.5)
            self.canvas.create_text(s.x + s.w/2, s.y + s.h/2, text=s.name,
                                   fill=text_color, font=("Segoe UI", 8, "bold"))

        # 3. Draw Nodes & Cart Depots
        for nid, n in self.layout.graph.nodes.items():
            if n.is_docking_bay:
                self.canvas.create_oval(n.x - 11, n.y - 11, n.x + 11, n.y + 11, fill="#10b981", outline="#ffffff", width=1.8)
                self.canvas.create_text(n.x, n.y + 18, text="CART DEPOT", fill="#10b981", font=("Segoe UI", 9, "bold"))
            else:
                self.canvas.create_oval(n.x - 3.5, n.y - 3.5, n.x + 3.5, n.y + 3.5, fill="#38bdf8", outline="")

        # 4. Draw Active Planned Trajectory Lines for each Cart
        agent_colors = ["#3b82f6", "#06b6d4", "#a855f7", "#ec4899", "#f59e0b", "#10b981"]
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

        # 5. Draw Humans with Gaussian Halos
        for h in self.humans:
            self.canvas.create_oval(h.x - 38, h.y - 38, h.x + 38, h.y + 38,
                                   outline="#f97316", width=1, dash=(3, 3))
            self.canvas.create_oval(h.x - 18, h.y - 18, h.x + 18, h.y + 18,
                                   fill="#7c2d12", outline="")
            self.canvas.create_oval(h.x - 6, h.y - 6, h.x + 6, h.y + 6,
                                   fill="#f97316", outline="#ffffff", width=1.5)

        # 6. Draw Directional Shopping Trolley Chassis + Kinetic Safety Bubble
        for a in self.agents:
            color = "#3b82f6"
            badge_text = ""
            if a.is_docked:
                color = "#10b981"
            elif a.state == "YIELDING_HUMAN":
                color = "#f59e0b"
                badge_text = "YIELD"
            elif a.state == "WAITING_LOCK":
                color = "#a855f7"
                badge_text = "LOCK WAIT"
            elif a.state == "FOLLOWING_CART":
                color = "#06b6d4"
                badge_text = "SAFE SPACING"

            # Draw Trolley Kinetic Safety Clearance Envelope (S_trolley)
            if not a.is_docked:
                self.canvas.create_oval(a.x - a.safety_bubble_radius, a.y - a.safety_bubble_radius,
                                       a.x + a.safety_bubble_radius, a.y + a.safety_bubble_radius,
                                       outline=color, width=1, dash=(2, 2))

            # Directional shopping cart chassis vertices
            local_cart_poly = [
                (14, 0),    # Front nose tip
                (10, 8),    # Front right basket corner
                (-10, 9),   # Rear right corner
                (-10, -9),  # Rear left corner
                (10, -8),   # Front left basket corner
            ]

            world_poly = []
            for lx, ly in local_cart_poly:
                wx, wy = self._rotate_point(lx, ly, a.x, a.y, a.heading)
                world_poly.extend([wx, wy])

            self.canvas.create_polygon(world_poly, fill=color, outline="#ffffff", width=1.8)

            # Rear handle bar
            h_left = self._rotate_point(-10, -7, a.x, a.y, a.heading)
            h_right = self._rotate_point(-10, 7, a.x, a.y, a.heading)
            self.canvas.create_line(h_left[0], h_left[1], h_right[0], h_right[1], fill="#e2e8f0", width=2.5)

            # Label & State badge
            self.canvas.create_text(a.x, a.y - 18, text=f"T{a.agent_id}",
                                   fill="#ffffff", font=("Segoe UI", 9, "bold"))
            if badge_text:
                self.canvas.create_text(a.x, a.y + 18, text=badge_text,
                                       fill=color, font=("Segoe UI", 8, "bold"))

        # 7. Telemetry display
        replans = sum(a.replan_count for a in self.agents)
        packets = self.mesh_net.total_packets_transmitted if self.mesh_net else 0
        docked = sum(1 for a in self.agents if a.is_docked)
        yielding = sum(1 for a in self.agents if a.state in ["YIELDING_HUMAN", "FOLLOWING_CART"])
        
        telemetry_text = f"Time: {self.sim_time:.1f}s | Replans: {replans} | Mesh Pkts: {packets} | Safe Yielding: {yielding} | Docked: {docked}/{len(self.agents)}"
        self.telemetry_lbl.configure(text=telemetry_text)


def launch_gui():
    root = tk.Tk()
    app = SupermarketSimApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
