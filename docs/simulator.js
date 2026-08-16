/**
 * D²RO / SW-DGO Web Simulator Engine (Pyodide WebAssembly + HTML5 Canvas Bridge)
 */

class WebSimulator {
  constructor() {
    this.pyodide = null;
    this.canvas = document.getElementById("simCanvas");
    this.ctx = this.canvas.getContext("2d");

    // UI state
    this.currentEnv = "supermarket";
    this.currentScenario = "A";
    this.currentAblation = "d2ro";
    this.isRunning = true; // Auto-start running on load once ready
    this.speedMultiplier = 1.0;
    this.dt = 0.05;

    // Latest state frame from Python
    this.state = null;

    // Click block animation ripples
    this.ripples = [];

    // Telemetry DOM Elements
    this.dom = {
      simTime: document.getElementById("telSimTime"),
      fleetStatus: document.getElementById("telFleetStatus"),
      safetyEnvelope: document.getElementById("telSafetyEnvelope"),
      v2vPackets: document.getElementById("telV2vPackets"),
      humanYields: document.getElementById("telHumanYields"),
      corridorLocks: document.getElementById("telCorridorLocks"),
      loadingOverlay: document.getElementById("loadingOverlay"),
      loadingText: document.getElementById("loadingText"),
      playBtn: document.getElementById("playBtn"),
      scenarioDesc: document.getElementById("scenarioDescBanner")
    };

    this.initPyodide();
  }

  async initPyodide() {
    try {
      this.updateLoading("Initializing WebAssembly engine...", "Loading Pyodide CPython v0.26 runtime");
      this.pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/"
      });

      this.updateLoading("Loading scientific packages...", "Installing NumPy for kinematic calculations");
      await this.pyodide.loadPackage("numpy");

      this.updateLoading("Writing D²RO core modules...", "Injecting SW-DGO routing engine into browser VFS");

      // Write bundled python source files into Pyodide virtual filesystem
      if (window.D2RO_PYTHON_FILES) {
        for (const [filePath, content] of Object.entries(window.D2RO_PYTHON_FILES)) {
          const parts = filePath.split("/");
          let curr = "";
          for (let i = 0; i < parts.length - 1; i++) {
            curr += (curr ? "/" : "") + parts[i];
            try {
              this.pyodide.FS.mkdir(curr);
            } catch (e) {
              // Directory may already exist
            }
          }
          this.pyodide.FS.writeFile(filePath, content);
        }
      }

      this.updateLoading("Bootstrapping Python simulation bridge...", "Compiling D* Lite & V2V mesh components");

      const pyBridgeCode = `
import math
import sys
import json

from d2ro.environments.supermarket import SupermarketLayout, ScenarioSuite as SupermarketScenarios
from d2ro.environments.airport import AirportLayout, AirportScenarioSuite
from d2ro.environments.hospital import HospitalLayout, HospitalScenarioSuite
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import ProxemicsField

class PySimEngine:
    def __init__(self):
        self.env_name = "supermarket"
        self.scenario_key = "A"
        self.ablation_mode = "d2ro"
        self.sim_time = 0.0
        self.layout = None
        self.agents = []
        self.humans = []
        self.mesh_net = None
        self.prox_field = ProxemicsField()
        self.scenario_desc = ""
        self.total_yielding_events = 0
        self.total_lock_waits = 0

    def load_scenario(self, env_name, scenario_key, ablation_mode="d2ro"):
        self.env_name = env_name
        self.scenario_key = scenario_key
        self.ablation_mode = ablation_mode
        self.sim_time = 0.0
        self.total_yielding_events = 0
        self.total_lock_waits = 0

        if env_name == "supermarket":
            self.layout = SupermarketLayout()
            cfgs, self.humans, self.scenario_desc = SupermarketScenarios.get_scenario(scenario_key, self.layout)
        elif env_name == "airport":
            self.layout = AirportLayout()
            cfgs, self.humans, self.scenario_desc = AirportScenarioSuite.get_scenario(scenario_key, self.layout)
        elif env_name == "hospital":
            self.layout = HospitalLayout()
            cfgs, self.humans, self.scenario_desc = HospitalScenarioSuite.get_scenario(scenario_key, self.layout)

        self.mesh_net = MeshNetwork(comm_radius=350.0)
        self.agents = []

        # Ablation mode flags.
        #
        # "static_a_star" reproduces the manuscript's MATCHED-CONTROLLER baseline,
        # whose whole purpose is to differ from the proposed method in routing
        # policy ONLY. It therefore keeps the safety envelope (identical vehicle
        # dynamics) and drops the mesh (no peer information):
        #
        #     enable_mesh=False, enable_prox=False, enable_lock=False,
        #     enable_safety=True, static_route=True, enable_yield=False
        #
        # An earlier version of this demo had two of those backwards -- it left the
        # mesh ENABLED and switched the safety envelope OFF -- so the baseline shown
        # here was neither the published arm nor a fair comparison: it was a socially
        # blind planner that also drove differently. Kept in sync with
        # run_experiments.py::run_baseline_comparison.
        is_matched = (ablation_mode == "static_a_star")
        enable_mesh = (ablation_mode != "no_mesh") and not is_matched
        enable_prox = not is_matched
        enable_lock = not is_matched
        enable_safety = True
        static_route = is_matched
        enable_yield = not is_matched

        for cfg in cfgs:
            agent = TrolleyAgent(
                agent_id=cfg["id"],
                graph=self.layout.graph,
                start_node=cfg["start"],
                goal_node=cfg["goal"],
                mesh_net=self.mesh_net,
                enable_mesh=enable_mesh,
                enable_lock=enable_lock,
                enable_prox=enable_prox,
                enable_safety=enable_safety,
                static_route=static_route,
                enable_yield=enable_yield
            )
            self.agents.append(agent)

    def step(self, dt):
        if not self.agents:
            return
        self.sim_time += dt

        # Obstacles / shelf geometry helpers
        shelf_boxes = []
        aisle_x = []
        crossway_y = []

        if self.env_name == "supermarket":
            shelf_boxes = [s.bounds for s in self.layout.shelves]
            aisle_x = [self.layout.start_x + i * self.layout.aisle_spacing for i in range(self.layout.num_aisles)]
            crossway_y = [self.layout.y_back_promenade, self.layout.y_action_alley, self.layout.y_front_concourse, self.layout.y_cart_depot]
        elif self.env_name == "airport":
            shelf_boxes = [s.bounds for s in self.layout.structures]
            aisle_x = [80.0, 150.0, 220.0, self.layout.x_security_choke, 390.0, 510.0, 630.0, self.layout.x_gate_pier_a, self.layout.x_gate_pier_b]
            crossway_y = [80.0, 180.0, 260.0, 340.0, 440.0, 510.0]
        elif self.env_name == "hospital":
            shelf_boxes = [r.bounds for r in self.layout.rooms]
            aisle_x = [self.layout.x_er, self.layout.x_ward_a, self.layout.x_central_hub, self.layout.x_ward_b, self.layout.x_or_mri]
            crossway_y = [self.layout.y_north_hall, self.layout.y_mid_hall, self.layout.y_south_hall]

        # Update pedestrians
        for h in self.humans:
            h.update(dt, self.layout.bounds, shelf_boxes, aisle_x, crossway_y)

        # Update autonomous trolley agents
        for a in self.agents:
            old_state = a.state
            a.step(dt, self.humans, self.prox_field, current_sim_time=self.sim_time,
                   shelves=shelf_boxes, peer_agents=self.agents)

            if a.state == "YIELDING_HUMAN" and old_state != "YIELDING_HUMAN":
                self.total_yielding_events += 1
            if a.state == "WAITING_LOCK" and old_state != "WAITING_LOCK":
                self.total_lock_waits += 1

        self.layout.graph.decay_mesh_penalties(dt, decay_rate=2.0)

    def trigger_blockage(self, click_x, click_y):
        nearest_node = None
        min_d = 999999.0
        for nid, n in self.layout.graph.nodes.items():
            d = math.hypot(click_x - n.x, click_y - n.y)
            if d < min_d:
                min_d = d
                nearest_node = nid

        if nearest_node and min_d < 60.0:
            for a in self.agents:
                for succ in self.layout.graph.successors(nearest_node):
                    a.broadcast_congestion(nearest_node, succ, penalty=20.0, current_time=self.sim_time)
            return nearest_node
        return None

    def get_state(self):
        # Package full JSON serializable state for JS renderer
        nodes = [{"id": n.id, "x": n.x, "y": n.y, "dock": n.is_docking_bay} for n in self.layout.graph.nodes.values()]
        edges = []
        for (u, v), e in self.layout.graph.edges.items():
            nu = self.layout.graph.nodes[u]
            nv = self.layout.graph.nodes[v]
            edges.append({
                "u": u, "v": v, "x1": nu.x, "y1": nu.y, "x2": nv.x, "y2": nv.y,
                "single": e.is_single_file, "w_mesh": e.w_mesh, "cost": e.cost
            })

        fixtures = []
        if self.env_name == "supermarket":
            fixtures = [{"x": s.x, "y": s.y, "w": s.w, "h": s.h, "name": s.name, "cat": s.category} for s in self.layout.shelves]
        elif self.env_name == "airport":
            fixtures = [{"x": s.x, "y": s.y, "w": s.w, "h": s.h, "name": s.name, "cat": s.zone_type} for s in self.layout.structures]
        elif self.env_name == "hospital":
            fixtures = [{"x": s.x, "y": s.y, "w": s.w, "h": s.h, "name": s.name, "cat": s.dept_type} for s in self.layout.rooms]

        agents_data = []
        for a in self.agents:
            path_nodes = a.planner.extract_full_path()
            path_coords = [{"x": self.layout.graph.nodes[nid].x, "y": self.layout.graph.nodes[nid].y} for nid in path_nodes if nid in self.layout.graph.nodes]
            agents_data.append({
                "id": a.agent_id,
                "x": a.x, "y": a.y,
                "heading": a.heading,
                "velocity": a.speed,
                "state": a.state,
                "is_docked": a.is_docked,
                "bubble_radius": a.safety_bubble_radius,
                "path": path_coords,
                "target_node": a.target_node
            })

        humans_data = [{"id": h.id, "x": h.x, "y": h.y, "heading": h.heading, "state": h.state} for h in self.humans]

        return json.dumps({
            "env": self.env_name,
            "scenario": self.scenario_key,
            "desc": self.scenario_desc,
            "bounds": self.layout.bounds,
            "sim_time": self.sim_time,
            "nodes": nodes,
            "edges": edges,
            "fixtures": fixtures,
            "agents": agents_data,
            "humans": humans_data,
            "v2v_packets": self.mesh_net.total_packets_transmitted,
            "yielding_events": self.total_yielding_events,
            "lock_waits": self.total_lock_waits
        })

py_engine = PySimEngine()
py_engine.load_scenario("supermarket", "A", "d2ro")
`;

      await this.pyodide.runPythonAsync(pyBridgeCode);

      // Hide loading screen
      this.dom.loadingOverlay.style.opacity = "0";
      setTimeout(() => {
        this.dom.loadingOverlay.style.display = "none";
      }, 300);

      this.bindEvents();
      this.loadScenario("supermarket", "A", "d2ro");
      this.dom.playBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;
      this.startLoop();
    } catch (err) {
      console.error(err);
      this.updateLoading("Initialization Error", err.message || "Failed to start Python WebAssembly context");
    }
  }

  updateLoading(title, subtext) {
    this.dom.loadingText.textContent = title;
    document.querySelector(".loading-subtext").textContent = subtext;
  }

  loadScenario(env, scenario, ablation) {
    this.currentEnv = env || this.currentEnv;
    this.currentScenario = scenario || this.currentScenario;
    this.currentAblation = ablation || this.currentAblation;

    this.pyodide.runPython(
      `py_engine.load_scenario("${this.currentEnv}", "${this.currentScenario}", "${this.currentAblation}")`
    );

    this.fetchState();
    this.resizeCanvas();
    this.render();
  }

  fetchState() {
    const raw = this.pyodide.runPython("py_engine.get_state()");
    this.state = JSON.parse(raw);
    this.updateTelemetry();
  }

  resizeCanvas() {
    if (!this.state || !this.state.bounds) return;
    const [minX, minY, maxX, maxY] = this.state.bounds;
    const width = Math.ceil(maxX - minX);
    const height = Math.ceil(maxY - minY);

    // High-DPI scaling
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = width + "px";
    this.canvas.style.height = height + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  bindEvents() {
    // Environment Buttons
    document.querySelectorAll(".env-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".env-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const env = btn.dataset.env;
        this.loadScenario(env, "A", this.currentAblation);
      });
    });

    // Scenario Tabs
    document.querySelectorAll(".scen-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".scen-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const scen = btn.dataset.scen;
        this.loadScenario(this.currentEnv, scen, this.currentAblation);
      });
    });

    // Ablation Mode Buttons
    document.querySelectorAll(".ablation-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".ablation-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const mode = btn.dataset.mode;
        this.loadScenario(this.currentEnv, this.currentScenario, mode);
      });
    });

    // Play / Pause
    this.dom.playBtn.addEventListener("click", () => {
      this.isRunning = !this.isRunning;
      this.dom.playBtn.innerHTML = this.isRunning
        ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`
        : `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg> Play`;
    });

    // Step
    document.getElementById("stepBtn").addEventListener("click", () => {
      this.isRunning = false;
      this.dom.playBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg> Play`;
      this.stepSim();
    });

    // Restart
    document.getElementById("restartBtn").addEventListener("click", () => {
      this.loadScenario(this.currentEnv, this.currentScenario, this.currentAblation);
    });

    // Speed Slider
    document.getElementById("speedSlider").addEventListener("input", (e) => {
      this.speedMultiplier = parseFloat(e.target.value);
      document.getElementById("speedVal").textContent = this.speedMultiplier.toFixed(1) + "x";
    });

    // Canvas click to spawn dynamic blockage
    this.canvas.addEventListener("click", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const hitNode = this.pyodide.runPython(`py_engine.trigger_blockage(${clickX}, ${clickY})`);
      if (hitNode) {
        this.ripples.push({ x: clickX, y: clickY, radius: 10, maxRadius: 50, alpha: 1.0 });
      }
    });
  }

  stepSim() {
    const effectiveDt = this.dt * this.speedMultiplier;
    this.pyodide.runPython(`py_engine.step(${effectiveDt})`);
    this.fetchState();
  }

  startLoop() {
    const loop = (now) => {
      if (this.isRunning) {
        this.stepSim();
      }
      this.updateRipples();
      this.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  updateRipples() {
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      const r = this.ripples[i];
      r.radius += 2;
      r.alpha -= 0.04;
      if (r.alpha <= 0 || r.radius >= r.maxRadius) {
        this.ripples.splice(i, 1);
      }
    }
  }

  updateTelemetry() {
    if (!this.state) return;
    this.dom.simTime.textContent = this.state.sim_time.toFixed(1) + " s";

    const activeCount = this.state.agents.filter((a) => !a.is_docked).length;
    this.dom.fleetStatus.textContent = `${activeCount} Active / ${this.state.agents.length - activeCount} Docked`;

    this.dom.v2vPackets.textContent = this.state.v2v_packets;
    this.dom.humanYields.textContent = this.state.yielding_events;
    this.dom.corridorLocks.textContent = this.state.lock_waits;
    this.dom.scenarioDesc.textContent = this.state.desc;

    // Safety bubble
    const activeAgent = this.state.agents.find((a) => !a.is_docked);
    if (activeAgent) {
      const rMeters = (activeAgent.bubble_radius * 0.03).toFixed(2);
      this.dom.safetyEnvelope.textContent = `${rMeters} m (${activeAgent.bubble_radius.toFixed(0)} px)`;
    } else {
      this.dom.safetyEnvelope.textContent = "1.00 m (33 px)";
    }
  }

  render() {
    if (!this.state) return;

    const ctx = this.ctx;
    const [minX, minY, maxX, maxY] = this.state.bounds;

    ctx.clearRect(0, 0, maxX - minX, maxY - minY);

    // 1. Background Grid & Zones
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, maxX - minX, maxY - minY);

    // Draw Department background zone accents
    if (this.state.env === "supermarket") {
      // Fresh produce area
      ctx.fillStyle = "rgba(6, 78, 59, 0.25)";
      ctx.strokeStyle = "#047857";
      ctx.lineWidth = 1;
      ctx.fillRect(30, 70, 120, 440);
      ctx.strokeRect(30, 70, 120, 440);
      ctx.fillStyle = "#10b981";
      ctx.font = "bold 11px Segoe UI, sans-serif";
      ctx.fillText("PRODUCE & FRESH", 40, 90);

      // Deli Bakery
      const rightX = maxX - 120;
      ctx.fillStyle = "rgba(76, 5, 25, 0.25)";
      ctx.strokeStyle = "#9f1239";
      ctx.fillRect(rightX, 70, 90, 440);
      ctx.strokeRect(rightX, 70, 90, 440);
      ctx.fillStyle = "#f43f5e";
      ctx.fillText("DELI & BAKERY", rightX + 5, 90);

      // Action Alley text
      ctx.fillStyle = "#475569";
      ctx.font = "bold 10px Segoe UI, sans-serif";
      ctx.fillText("— CENTRAL ACTION ALLEY (TRANSVERSE PROMENADE) —", 320, 290);
    }

    // 2. Graph Edges
    for (const edge of this.state.edges) {
      ctx.beginPath();
      ctx.moveTo(edge.x1, edge.y1);
      ctx.lineTo(edge.x2, edge.y2);

      if (edge.w_mesh > 0.1) {
        ctx.strokeStyle = "#f43f5e";
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 4]);
      } else if (edge.single) {
        ctx.strokeStyle = "#a855f7";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
      } else {
        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 3. Fixtures & Shelves
    for (const fix of this.state.fixtures) {
      ctx.lineWidth = 1.5;

      let fillStyle = "#1e293b";
      let borderStyle = "#334155";
      let textStyle = "#94a3b8";

      if (fix.cat === "island" || fix.cat === "retail") {
        fillStyle = "#064e3b";
        borderStyle = "#059669";
        textStyle = "#6ee7b7";
      } else if (fix.cat === "deli" || fix.cat === "er" || fix.cat === "or") {
        fillStyle = "#4c0519";
        borderStyle = "#e11d48";
        textStyle = "#fda4af";
      } else if (fix.cat === "checkout" || fix.cat === "ward" || fix.cat === "gate") {
        fillStyle = "#312e81";
        borderStyle = "#6366f1";
        textStyle = "#c7d2fe";
      }

      ctx.fillStyle = fillStyle;
      ctx.fillRect(fix.x, fix.y, fix.w, fix.h);
      ctx.strokeStyle = borderStyle;
      ctx.strokeRect(fix.x, fix.y, fix.w, fix.h);

      ctx.fillStyle = textStyle;
      ctx.font = "bold 10px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(fix.name, fix.x + fix.w / 2, fix.y + fix.h / 2);
    }

    // 4. Graph Nodes & Depots
    for (const n of this.state.nodes) {
      if (n.dock) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, 11, 0, 2 * Math.PI);
        ctx.fillStyle = "#10b981";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "#10b981";
        ctx.font = "bold 10px Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("DEPOT", n.x, n.y + 20);
      } else {
        ctx.beginPath();
        ctx.arc(n.x, n.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "#38bdf8";
        ctx.fill();
      }
    }

    // 5. Active Planned Trajectory Trails
    const agentColors = ["#3b82f6", "#06b6d4", "#a855f7", "#ec4899", "#f59e0b", "#10b981"];
    for (let idx = 0; idx < this.state.agents.length; idx++) {
      const a = this.state.agents[idx];
      if (a.is_docked || !a.path || a.path.length <= 1) continue;

      const clr = agentColors[idx % agentColors.length];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      for (const p of a.path) {
        ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = clr;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 6. Pedestrians & Anisotropic Proxemic Halos
    for (const h of this.state.humans) {
      // Outer proxemic halo (1.35m front space)
      ctx.beginPath();
      ctx.ellipse(h.x, h.y, 38, 26, h.heading, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(249, 115, 22, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Inner personal space
      ctx.beginPath();
      ctx.arc(h.x, h.y, 16, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(124, 45, 18, 0.6)";
      ctx.fill();

      // Pedestrian Body core
      ctx.beginPath();
      ctx.arc(h.x, h.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = "#f97316";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Heading indicator nose
      ctx.beginPath();
      ctx.moveTo(h.x, h.y);
      ctx.lineTo(h.x + Math.cos(h.heading) * 12, h.y + Math.sin(h.heading) * 12);
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // 7. Directional Autonomous Trolley Chassis & Safety Envelopes
    for (let idx = 0; idx < this.state.agents.length; idx++) {
      const a = this.state.agents[idx];

      let mainColor = "#3b82f6";
      let statusLabel = "";

      if (a.is_docked) {
        mainColor = "#10b981";
      } else if (a.state === "YIELDING_HUMAN") {
        mainColor = "#f59e0b";
        statusLabel = "YIELD";
      } else if (a.state === "WAITING_LOCK") {
        mainColor = "#a855f7";
        statusLabel = "LOCK WAIT";
      } else if (a.state === "FOLLOWING_CART") {
        mainColor = "#06b6d4";
        statusLabel = "SAFE SPACING";
      }

      // Draw Trolley Kinetic Safety Clearance Envelope (S_trolley)
      if (!a.is_docked) {
        ctx.beginPath();
        ctx.arc(a.x, a.y, a.bubble_radius, 0, 2 * Math.PI);
        ctx.strokeStyle = mainColor;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Rotated Directional Chassis Polygon
      const localCartPoly = [
        [14, 0],   // Nose tip
        [10, 8],   // Front right
        [-10, 9],  // Rear right
        [-10, -9], // Rear left
        [10, -8]   // Front left
      ];

      ctx.save();
      ctx.translate(a.x, a.y);
      ctx.rotate(a.heading);

      ctx.beginPath();
      ctx.moveTo(localCartPoly[0][0], localCartPoly[0][1]);
      for (let i = 1; i < localCartPoly.length; i++) {
        ctx.lineTo(localCartPoly[i][0], localCartPoly[i][1]);
      }
      ctx.closePath();

      ctx.fillStyle = mainColor;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Center chassis core marker
      ctx.beginPath();
      ctx.arc(0, 0, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      ctx.restore();

      // Draw Status Text Badge
      if (statusLabel) {
        ctx.fillStyle = mainColor;
        ctx.font = "bold 9px Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(statusLabel, a.x, a.y - 18);
      }
    }

    // 8. Click ripples for dynamic blockages
    for (const r of this.ripples) {
      ctx.beginPath();
      ctx.arc(r.x, r.y, r.radius, 0, 2 * Math.PI);
      ctx.strokeStyle = `rgba(244, 63, 94, ${r.alpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

// Instantiate simulator once DOM loads
window.addEventListener("DOMContentLoaded", () => {
  window.simApp = new WebSimulator();
});
