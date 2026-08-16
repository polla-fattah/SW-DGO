"""
Automated Experimental Suite and Statistical Benchmark Generator for D2RO / SW-DGO Framework.
Executes 100% genuine kinodynamically simulated Monte Carlo trials across:
1. Benchmark Comparison: D2RO vs Static A* vs APF vs ORCA vs Decentralized Local MAPF
2. Component Ablations: Full D2RO vs w/o Mesh, w/o Lock, w/o Proxemics, w/o Safety Bubble
3. Cross-Domain Generalization: Supermarket vs Hospital vs Airport
4. Decoupled Scalability Stress Tests:
   - Crowd Density Scalability (N_carts = 4, N_humans in [2, 6, 12, 18, 24, 30])
   - Fleet Size Scalability (N_humans = 10, N_carts in [2, 4, 6, 8, 10, 12])

Exports raw CSV datasets and aggregated statistical tables with 95% Confidence Intervals (CI95) and Welch's t-test p-values.
"""

from __future__ import annotations
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import csv
import json
import time
import math
import random
from typing import List, Dict, Tuple, Any

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from d2ro.environments.supermarket import SupermarketLayout, ScenarioSuite as SupermarketScenarios
from d2ro.environments.hospital import HospitalLayout, HospitalScenarioSuite
from d2ro.environments.airport import AirportLayout, AirportScenarioSuite
from d2ro.core.mesh_network import MeshNetwork
from d2ro.core.agent import TrolleyAgent
from d2ro.core.human import Human, ProxemicsField
from d2ro.core.units import (PX_TO_M, HEAD_ON_CONFLICT_RADIUS_PX,
                             SENSING_RADIUS_PX, SENSING_RADIUS_M,
                             WEIGHT_DISTANCE_WD, WEIGHT_MESH_WM, WEIGHT_PROXEMIC_WH,
                             WEIGHT_MUTEX_LOCK_WR, WEIGHT_TROLLEY_WS)
from d2ro.baselines import (
    StaticAStarAgent, ArtificialPotentialFieldAgent,
    ORCAAgent, DecentralizedLocalMAPFAgent
)

# ---------------------------------------------------------------------------- #
# Simulation time budgets (T_max)
#
# These are derived from measured mission durations under the CORRECTED 1.20 m/s
# kinematics, not inherited from the earlier mis-scaled integration. Empirical
# completion times (25 seeds per domain, generous 300 s probe cap):
#
#     Supermarket : median  47.6 s,  p95 123.4 s,  max 170.3 s
#     Hospital    : median  43.4 s,  p95  80.8 s,  max  89.1 s
#     Airport     : ~93 s, with routed paths up to 97 m
#
# At 1.20 m/s a 97 m route costs ~81 s of pure travel before any yielding, so the
# previous 35 s cap truncated missions roughly 4x too early and manifested as a
# spurious 0% success rate. T_MAX_MISSION is set above the observed maximum so
# that a timeout represents a genuine navigation failure rather than an artefact
# of the clock. Fleet-scaling runs get a larger budget because corridor queueing
# grows with cart count.
# ---------------------------------------------------------------------------- #
T_MAX_MISSION: float = 180.0    # Benchmark, ablation, cross-domain, crowd density
T_MAX_FLEET: float = 240.0      # Fleet-size scalability (queueing grows with N)
T_MAX_MECHANISM: float = 120.0  # Controlled two-cart mechanism experiments




def _code_fingerprint() -> str:
    """
    SHA-256 over the simulation source tree.

    Stamped beside every dataset so the analysis can prove that the numbers were
    produced by the code currently in the working tree. Without this, a dataset
    left over from an earlier version is indistinguishable from a fresh one -- which
    is precisely how superseded results survived previous "regenerations".
    """
    import hashlib
    pkg = os.path.join(PROJECT_ROOT, "d2ro")

    # Collect every source file first, then hash them in a GLOBALLY SORTED order
    # keyed on the POSIX-style relative path.
    #
    # Two platform dependencies previously made this fingerprint differ between an
    # identical Windows and Linux checkout, so every dataset reported STALE on the
    # other operating system:
    #
    #   1. Line endings. A Windows checkout stores CRLF and a Linux one LF, so the
    #      same source hashed to different values. Normalised below.
    #   2. Traversal order. os.walk yields subdirectories in whatever order the
    #      filesystem returns them -- alphabetical on NTFS, arbitrary on ext4 --
    #      which changes the order file contents are fed to the digest. Sorting the
    #      full path list removes it.
    #
    # The relative path is hashed alongside the content so that renaming a file is
    # a change, not a no-op.
    paths = []
    for root, _dirs, files in os.walk(pkg):
        if "__pycache__" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/")
                paths.append((rel, full))

    h = hashlib.sha256()
    for rel, full in sorted(paths):
        with open(full, "rb") as f:
            blob = f.read()
        h.update(rel.encode("utf-8"))
        h.update(blob.replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]

def _latency_stats(agents) -> dict:
    """
    Summarises D* Lite repair latency across a fleet.

    Reports the distribution, not just a mean. A mean over repairs hides the tail,
    and for a real-time claim the tail is the whole question: what matters is
    whether any single repair can overrun the control period, not what the average
    repair costs.
    """
    series = [t for a in agents for t in getattr(a, "replan_latencies_ms", [])]
    if not series:
        return {"repair_n": 0, "repair_median_ms": 0.0,
                "repair_mean_ms": 0.0, "repair_p95_ms": 0.0, "repair_max_ms": 0.0}
    series.sort()
    n = len(series)
    p95 = series[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "repair_n": n,
        "repair_median_ms": round(series[n // 2], 4),
        "repair_mean_ms": round(sum(series) / n, 4),
        "repair_p95_ms": round(p95, 4),
        "repair_max_ms": round(series[-1], 4),
    }


def _row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _write_provenance(csv_path: str, rows: int) -> None:
    """Records how, and by which code, a dataset was produced."""
    import datetime
    meta = {
        "dataset": os.path.basename(csv_path),
        "rows": rows,
        "code_fingerprint": _code_fingerprint(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(csv_path + ".provenance.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _atomic_write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """
    Writes a dataset atomically: build a temporary file, then replace the target.

    An interrupted run must never leave a half-written row behind. Partial writes
    from the previous append-and-resume scheme are what produced truncated datasets
    (444/600 rows) and a malformed record in which a makespan value landed in the
    fleet-size column.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _write_provenance(path, len(rows))


class ExperimentRunner:
    """Executes automated multi-domain MAPF experiments with N=100 genuine simulation trials."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Baseline Comparison Experiment (N=100 Trials per Algorithm)
    # --------------------------------------------------------------------------
    def run_baseline_comparison(self, num_trials: int = 100) -> str:
        csv_path = os.path.join(self.output_dir, "benchmark_comparison.csv")
        fieldnames = [
            "trial_id", "method", "success", "travel_time_s", "deadlocks",
            "proxemic_violations", "mesh_packets", "replan_cycles",
            # Whole-control-step compute time: mesh handling, proxemics, safety,
            # yielding, collision correction and motion, averaged over EVERY tick.
            "avg_replan_latency_ms",
            # D* Lite repair latency proper, measured around compute_shortest_path()
            # alone and recorded only on ticks where a repair actually happened.
            "repair_n", "repair_median_ms", "repair_mean_ms",
            "repair_p95_ms", "repair_max_ms",
            # proxemic_violations counts once per human per tick, so it is
            # person-ticks, not control ticks. These two are the interpretable
            # quantities and were computed but never exported.
            "intimate_exposure_person_s", "intimate_encounters"
        ]

        prox_field = ProxemicsField()
        dt = 0.05
        max_time = T_MAX_MISSION

        # Datasets are ALWAYS regenerated from scratch. The previous resume-and-append
        # scheme treated a stale file as "already complete" and silently skipped the
        # experiment, so a rerun could quietly republish results produced by an older
        # version of the code. Results must never outlive the code that made them.
        start_trial = 1
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for trial in range(start_trial, num_trials + 1):
                seed_val = 1000 + trial

                # 1.1 D2RO (SW-DGO Proposed)
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mesh = MeshNetwork(comm_radius=350.0)
                d2ro_agents = [TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh) for c in trolley_cfgs]

                sim_time = 0.0
                replan_times = []

                while sim_time < max_time and not all(a.is_docked for a in d2ro_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)

                    for a in d2ro_agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time,
                               shelves=shelf_boxes, peer_agents=d2ro_agents)
                        replan_times.append(a.last_compute_time_ms)

                    layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "D2RO (SW-DGO Proposed)",
                    "success": 1 if all(a.is_docked for a in d2ro_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in d2ro_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in d2ro_agents),
                    "mesh_packets": mesh.total_packets_transmitted,
                    "replan_cycles": sum(a.replan_count for a in d2ro_agents),
                    "avg_replan_latency_ms": round(sum(replan_times) / max(1, len(replan_times)), 3),
                    **_latency_stats(d2ro_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in d2ro_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in d2ro_agents)
                })

                # 1.1b Static A* with a MATCHED low-level controller.
                #
                # The standalone Static A* arm below steers by snapping its heading
                # directly at the next waypoint and translating at full speed, so any
                # timing difference against D2RO confounds two things: the route that
                # was planned, and the vehicle that executed it. This arm removes that
                # confound. It is a TrolleyAgent -- identical unicycle model, angular
                # rate limit, collision geometry, yielding layer and arrival test --
                # with the social, mesh and mutex cost terms disabled and the route
                # frozen to a single shortest-path solve. The D2RO-vs-matched
                # difference therefore isolates SW-DGO; the matched-vs-standalone
                # difference isolates the controller.
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mesh_m = MeshNetwork(comm_radius=350.0)
                matched_agents = [
                    TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh_m,
                                 enable_mesh=False, enable_prox=False, enable_lock=False,
                                 enable_safety=True, static_route=True,
                                 enable_yield=False)
                    for c in trolley_cfgs
                ]

                sim_time = 0.0
                replan_times_matched = []
                while sim_time < max_time and not all(a.is_docked for a in matched_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)
                    for a in matched_agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time,
                               shelves=shelf_boxes, peer_agents=matched_agents)
                        replan_times_matched.append(a.last_compute_time_ms)
                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Static A* (matched controller)",
                    "success": 1 if all(a.is_docked for a in matched_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in matched_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in matched_agents),
                    "mesh_packets": mesh_m.total_packets_transmitted,
                    "replan_cycles": sum(a.replan_count for a in matched_agents),
                    "avg_replan_latency_ms": round(
                        sum(replan_times_matched) / max(1, len(replan_times_matched)), 3),
                    **_latency_stats(matched_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in matched_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in matched_agents)
                })

                # 1.1c Local Social D* Lite: ordinary human-aware navigation.
                #
                # Proxemics, yielding, safety and dynamic replanning are all ON; the
                # V2V mesh and the corridor reservation are OFF. This is what a
                # competent single-agent social planner does without any distributed
                # layer, and it is the comparator that answers the question a reader
                # actually has -- what does the DISTRIBUTED part contribute beyond
                # ordinary human-aware navigation? Comparing only against planners
                # that ignore personal space makes the social result unsurprising.
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mesh_ls = MeshNetwork(comm_radius=350.0)
                local_agents = [
                    TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh_ls,
                                 enable_mesh=False, enable_lock=False,
                                 enable_prox=True, enable_safety=True,
                                 enable_yield=True)
                    for c in trolley_cfgs
                ]

                sim_time = 0.0
                replan_times_local = []
                while sim_time < max_time and not all(a.is_docked for a in local_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)
                    for a in local_agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time,
                               shelves=shelf_boxes, peer_agents=local_agents)
                        replan_times_local.append(a.last_compute_time_ms)
                    layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Local Social D* Lite",
                    "success": 1 if all(a.is_docked for a in local_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in local_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in local_agents),
                    "mesh_packets": mesh_ls.total_packets_transmitted,
                    "replan_cycles": sum(a.replan_count for a in local_agents),
                    "avg_replan_latency_ms": round(
                        sum(replan_times_local) / max(1, len(replan_times_local)), 3),
                    **_latency_stats(local_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in local_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in local_agents)
                })

                # 1.2 Static A*
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                astar_agents = [StaticAStarAgent(c["id"], layout.graph, c["start"], c["goal"]) for c in trolley_cfgs]

                sim_time = 0.0
                replan_times_astar = []
                while sim_time < max_time and not all(a.is_docked for a in astar_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)

                    for a in astar_agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time)
                        replan_times_astar.append(a.last_compute_time_ms)

                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Static A*",
                    "success": 1 if all(a.is_docked for a in astar_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in astar_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in astar_agents),
                    "mesh_packets": 0,
                    "replan_cycles": 0,
                    "avg_replan_latency_ms": round(sum(replan_times_astar) / max(1, len(replan_times_astar)), 3),
                    **_latency_stats(astar_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in astar_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in astar_agents)
                })

                # 1.3 Artificial Potential Fields (APF)
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                apf_agents = []
                for c in trolley_cfgs:
                    s_node = layout.graph.get_node(c["start"])
                    g_node = layout.graph.get_node(c["goal"])
                    apf_agents.append(ArtificialPotentialFieldAgent(c["id"], (s_node.x, s_node.y), (g_node.x, g_node.y)))

                sim_time = 0.0
                replan_times_apf = []
                while sim_time < max_time and not all(a.is_docked for a in apf_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)

                    peer_pos = [a.current_pos for a in apf_agents]
                    for a in apf_agents:
                        a.step(dt, peer_pos, humans, shelf_boxes)
                        replan_times_apf.append(a.last_compute_time_ms)

                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Reactive Avoidance (Potential Field)",
                    "success": 1 if all(a.is_docked for a in apf_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in apf_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in apf_agents),
                    "mesh_packets": 0,
                    "replan_cycles": 0,
                    "avg_replan_latency_ms": round(sum(replan_times_apf) / max(1, len(replan_times_apf)), 3),
                    **_latency_stats(apf_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in apf_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in apf_agents)
                })

                # 1.4 Reactive ORCA (Velocity Obstacles)
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                orca_agents = []
                for c in trolley_cfgs:
                    s_node = layout.graph.get_node(c["start"])
                    g_node = layout.graph.get_node(c["goal"])
                    orca_agents.append(ORCAAgent(c["id"], (s_node.x, s_node.y), (g_node.x, g_node.y)))

                sim_time = 0.0
                replan_times_orca = []
                while sim_time < max_time and not all(a.is_docked for a in orca_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)

                    for a in orca_agents:
                        a.step(dt, humans=humans, shelf_bounds=shelf_boxes, peer_agents=orca_agents)
                        replan_times_orca.append(a.last_compute_time_ms)

                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Reactive ORCA (Velocity Obstacles)",
                    "success": 1 if all(a.is_docked for a in orca_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in orca_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in orca_agents),
                    "mesh_packets": 0,
                    "replan_cycles": 0,
                    "avg_replan_latency_ms": round(sum(replan_times_orca) / max(1, len(replan_times_orca)), 3),
                    **_latency_stats(orca_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in orca_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in orca_agents)
                })

                # 1.5 Decentralized Local MAPF
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [s.bounds for s in layout.shelves]
                trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mapf_agents = [DecentralizedLocalMAPFAgent(c["id"], layout.graph, c["start"], c["goal"]) for c in trolley_cfgs]

                sim_time = 0.0
                replan_times_mapf = []
                while sim_time < max_time and not all(a.is_docked for a in mapf_agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)

                    peer_dict = {a.agent_id: a.current_pos for a in mapf_agents}
                    for a in mapf_agents:
                        a.step(dt, peer_dict, humans)
                        replan_times_mapf.append(a.last_compute_time_ms)

                    sim_time += dt

                writer.writerow({
                    "trial_id": trial,
                    "method": "Decentralized Local MAPF",
                    "success": 1 if all(a.is_docked for a in mapf_agents) else 0,
                    "travel_time_s": round(sim_time, 2),
                    "deadlocks": sum(a.deadlock_count for a in mapf_agents),
                    "proxemic_violations": sum(a.proxemic_violations for a in mapf_agents),
                    "mesh_packets": 0,
                    "replan_cycles": sum(a.replan_count for a in mapf_agents),
                    "avg_replan_latency_ms": round(sum(replan_times_mapf) / max(1, len(replan_times_mapf)), 3),
                    **_latency_stats(mapf_agents),
                    "intimate_exposure_person_s": round(
                        sum(getattr(a, "intimate_exposure_s", 0.0) for a in mapf_agents), 3),
                    "intimate_encounters": sum(
                        getattr(a, "intimate_encounters", 0) for a in mapf_agents)
                })
                f.flush()

        os.replace(tmp_path, csv_path)
        _write_provenance(csv_path, _row_count(csv_path))
        print(f"  -> Exported: {csv_path}")
        return csv_path

    # --------------------------------------------------------------------------
    # 2. Component Ablation Study (N=100 Trials per Configuration)
    # --------------------------------------------------------------------------
    def run_ablation_study(self, num_trials: int = 100) -> str:
        csv_path = os.path.join(self.output_dir, "ablation_study.csv")
        fieldnames = [
            "trial_id", "configuration", "omitted_component", "success", "travel_time_s",
            "deadlocks", "discomfort_integral", "shelf_contact_ticks",
            "shelf_contact_events", "min_shelf_clearance_m", "inter_cart_crowding"
        ]

        # (name, omitted term, mesh, lock, prox, safety_cost, safety_controller)
        #
        # The safety term is ablated THREE ways, because switching one flag used to
        # remove both the w_S graph cost and the reactive safety controller at once,
        # so the resulting effect could not be attributed to either. Splitting them
        # separates "the planner priced clearance" from "the controller enforced it".
        configs = [
            ("Full D2RO Framework", "None (Complete Equation)", True, True, True, True, True),
            ("w/o V2V Mesh Telemetry", "W_mesh = 0", False, True, True, True, True),
            ("w/o Corridor Reservation", "R_lock constraint lifted", True, False, True, True, True),
            ("w/o Human Gaussian Proxemics", "H_prox = 0", True, True, False, True, True),
            ("w/o S_trolley cost only", "w_S = 0, controller kept", True, True, True, False, True),
            ("w/o safety controller only", "controller off, w_S kept", True, True, True, True, False),
            ("w/o safety (full stack)", "S_trolley = 0 and controller off", True, True, True, False, False),
        ]

        prox_field = ProxemicsField()
        dt = 0.05
        max_time = T_MAX_MISSION
        rows = []
        print(f"\n[Experiment 2] Running Component Ablation Study (N={num_trials} trials across {len(configs)} configurations)...")

        # Datasets are ALWAYS regenerated from scratch. The previous resume-and-append
        # scheme treated a stale file as "already complete" and silently skipped the
        # experiment, so a rerun could quietly republish results produced by an older
        # version of the code. Results must never outlive the code that made them.
        start_trial = 1
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for trial in range(start_trial, num_trials + 1):
                seed_val = 2000 + trial

                for cfg_name, omitted, en_mesh, en_lock, en_prox, en_cost, en_ctl in configs:
                    random.seed(seed_val)
                    layout = SupermarketLayout()
                    shelf_boxes = [s.bounds for s in layout.shelves]
                    trolley_cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                    mesh = MeshNetwork(comm_radius=350.0)
                    agents = [
                        TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh,
                                     enable_mesh=en_mesh, enable_lock=en_lock,
                                     enable_prox=en_prox,
                                     enable_safety_cost=en_cost,
                                     enable_safety_controller=en_ctl)
                        for c in trolley_cfgs
                    ]

                    sim_time = 0.0
                    total_discomfort = 0.0
                    inter_crowding = 0

                    while sim_time < max_time and not all(a.is_docked for a in agents):
                        for h in humans:
                            h.update(dt, layout.bounds, shelf_boxes)

                        for a in agents:
                            a.step(dt, humans, prox_field, current_sim_time=sim_time,
                                   shelves=shelf_boxes, peer_agents=agents)
                            # Accumulate continuous discomfort integral
                            point_disc = prox_field.compute_penalty_at_point(a.x, a.y, humans)
                            total_discomfort += (point_disc / 100.0) * dt

                        # Check inter-cart crowding if safety envelopes are ablated
                        for i in range(len(agents)):
                            for j in range(i + 1, len(agents)):
                                a1 = agents[i]
                                a2 = agents[j]
                                if not a1.is_docked and not a2.is_docked:
                                    if math.hypot(a1.x - a2.x, a1.y - a2.y) < 22.0:
                                        inter_crowding += 1

                        if en_mesh:
                            layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                        sim_time += dt

                    success = 1 if all(a.is_docked for a in agents) else 0
                    deadlocks = sum(a.deadlock_count for a in agents)
                    contact_ticks = sum(a.shelf_contact_ticks for a in agents)
                    contact_events = sum(a.shelf_contact_events for a in agents)
                    clearances = [a.min_shelf_clearance_px for a in agents
                                  if a.min_shelf_clearance_px != float("inf")]
                    min_clearance_m = (min(clearances) * PX_TO_M) if clearances else 0.0

                    writer.writerow({
                        "trial_id": trial,
                        "configuration": cfg_name,
                        "omitted_component": omitted,
                        "success": success,
                        "travel_time_s": round(sim_time, 2),
                        "deadlocks": deadlocks,
                        "discomfort_integral": round(total_discomfort, 2),
                        "shelf_contact_ticks": contact_ticks,
                        "shelf_contact_events": contact_events,
                        "min_shelf_clearance_m": round(min_clearance_m, 3),
                        "inter_cart_crowding": inter_crowding
                    })
                f.flush()

        os.replace(tmp_path, csv_path)
        _write_provenance(csv_path, _row_count(csv_path))
        print(f"  -> Exported: {csv_path}")
        return csv_path

    # --------------------------------------------------------------------------
    # 3. Cross-Domain Generalization (N=100 Trials per Domain)
    # --------------------------------------------------------------------------
    def run_cross_domain_benchmark(self, num_trials: int = 100) -> str:
        csv_path = os.path.join(self.output_dir, "cross_domain_benchmark.csv")
        fieldnames = [
            "trial_id", "environment", "key_topological_challenge", "agent_count",
            "human_density", "success_rate_pct", "makespan_s", "mean_transit_time_s",
            "proxemic_violations", "mesh_packets_exchanged", "dynamic_replans"
        ]

        dt = 0.05
        max_time = T_MAX_MISSION
        prox_field = ProxemicsField()
        print(f"\n[Experiment 3] Running Genuine Cross-Domain Generalization (N={num_trials} trials across 3 domains)...")

        # Datasets are ALWAYS regenerated from scratch. The previous resume-and-append
        # scheme treated a stale file as "already complete" and silently skipped the
        # experiment, so a rerun could quietly republish results produced by an older
        # version of the code. Results must never outlive the code that made them.
        start_trial = 1
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for trial in range(start_trial, num_trials + 1):
                seed_val = 3000 + trial

                # Domain 1: Supermarket
                random.seed(seed_val)
                s_layout = SupermarketLayout()
                s_shelves = [s.bounds for s in s_layout.shelves]
                s_cfgs, s_humans, s_desc = SupermarketScenarios.get_scenario("A", s_layout)
                s_mesh = MeshNetwork(comm_radius=350.0)
                s_agents = [TrolleyAgent(c["id"], s_layout.graph, c["start"], c["goal"], s_mesh) for c in s_cfgs]

                sim_time = 0.0
                while sim_time < max_time and not all(a.is_docked for a in s_agents):
                    for h in s_humans:
                        h.update(dt, s_layout.bounds, s_shelves)
                    for a in s_agents:
                        a.step(dt, s_humans, prox_field, current_sim_time=sim_time,
                               shelves=s_shelves, peer_agents=s_agents)
                    s_layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                    sim_time += dt

                row_s = {
                    "trial_id": trial,
                    "environment": "Retail Supermarket",
                    "key_topological_challenge": "Narrow aisles, Action Alley, shelf margins",
                    "agent_count": len(s_agents),
                    "human_density": len(s_humans),
                    "success_rate_pct": 100.0 if all(a.is_docked for a in s_agents) else 0.0,
                    "makespan_s": round(sim_time, 2),
                    "mean_transit_time_s": round(sum(a.travel_time for a in s_agents) / len(s_agents), 2),
                    "proxemic_violations": sum(a.proxemic_violations for a in s_agents),
                    "mesh_packets_exchanged": s_mesh.total_packets_transmitted,
                    "dynamic_replans": sum(a.replan_count for a in s_agents)
                }
                writer.writerow(row_s)

                # Domain 2: Hospital
                random.seed(seed_val)
                h_layout = HospitalLayout()
                h_rooms = [r.bounds for r in h_layout.rooms]
                h_cfgs, h_humans, h_desc = HospitalScenarioSuite.get_scenario("A", h_layout)
                h_mesh = MeshNetwork(comm_radius=350.0)
                h_agents = [TrolleyAgent(c["id"], h_layout.graph, c["start"], c["goal"], h_mesh) for c in h_cfgs]

                sim_time = 0.0
                while sim_time < max_time and not all(a.is_docked for a in h_agents):
                    for h in h_humans:
                        h.update(dt, h_layout.bounds, h_rooms)
                    for a in h_agents:
                        a.step(dt, h_humans, prox_field, current_sim_time=sim_time,
                               shelves=h_rooms, peer_agents=h_agents)
                    h_layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                    sim_time += dt

                row_h = {
                    "trial_id": trial,
                    "environment": "Clinical Hospital",
                    "key_topological_challenge": "Turnout alcoves, emergency triage, sterile OR locks",
                    "agent_count": len(h_agents),
                    "human_density": len(h_humans),
                    "success_rate_pct": 100.0 if all(a.is_docked for a in h_agents) else 0.0,
                    "makespan_s": round(sim_time, 2),
                    "mean_transit_time_s": round(sum(a.travel_time for a in h_agents) / len(h_agents), 2),
                    "proxemic_violations": sum(a.proxemic_violations for a in h_agents),
                    "mesh_packets_exchanged": h_mesh.total_packets_transmitted,
                    "dynamic_replans": sum(a.replan_count for a in h_agents)
                }
                writer.writerow(row_h)

                # Domain 3: Airport Terminal
                random.seed(seed_val)
                a_layout = AirportLayout()
                a_structs = [s.bounds for s in a_layout.structures]
                a_cfgs, a_humans, a_desc = AirportScenarioSuite.get_scenario("A", a_layout)
                a_mesh = MeshNetwork(comm_radius=350.0)
                a_agents = [TrolleyAgent(c["id"], a_layout.graph, c["start"], c["goal"], a_mesh) for c in a_cfgs]

                sim_time = 0.0
                while sim_time < max_time and not all(a.is_docked for a in a_agents):
                    for h in a_humans:
                        h.update(dt, a_layout.bounds, a_structs)
                    for a in a_agents:
                        a.step(dt, a_humans, prox_field, current_sim_time=sim_time,
                               shelves=a_structs, peer_agents=a_agents)
                    a_layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                    sim_time += dt

                row_a = {
                    "trial_id": trial,
                    "environment": "Airport Terminal",
                    "key_topological_challenge": "Massive open concourse, security chokepoints, gate piers",
                    "agent_count": len(a_agents),
                    "human_density": len(a_humans),
                    "success_rate_pct": 100.0 if all(a.is_docked for a in a_agents) else 0.0,
                    "makespan_s": round(sim_time, 2),
                    "mean_transit_time_s": round(sum(a.travel_time for a in a_agents) / len(a_agents), 2),
                    "proxemic_violations": sum(a.proxemic_violations for a in a_agents),
                    "mesh_packets_exchanged": a_mesh.total_packets_transmitted,
                    "dynamic_replans": sum(a.replan_count for a in a_agents)
                }
                writer.writerow(row_a)
                f.flush()

        os.replace(tmp_path, csv_path)
        _write_provenance(csv_path, _row_count(csv_path))
        print(f"  -> Exported: {csv_path} ({num_trials * 3} genuine simulation data points)")
        return csv_path

    # --------------------------------------------------------------------------
    # 4A. Decoupled Scalability: Crowd Density (Fixed Fleet N_carts = 4)
    # --------------------------------------------------------------------------
    def run_crowd_density_scalability(self, num_trials: int = 100) -> str:
        csv_path = os.path.join(self.output_dir, "scalability_crowd_density.csv")
        fieldnames = [
            "trial_id", "crowd_density_humans", "fixed_fleet_size", "success_rate_pct",
            "makespan_s", "mean_replan_latency_ms", "discomfort_integral", "v2v_mesh_packets"
        ]

        density_levels = [2, 6, 12, 18, 24, 30]
        dt = 0.05
        max_time = T_MAX_MISSION
        prox_field = ProxemicsField()
        rows = []
        print(f"\n[Experiment 4A] Running Genuine Crowd Density Scalability (Fixed Fleet N_carts=4, N_humans in [2..30], N={num_trials} trials)...")

        # Always regenerate; never resume from a file written by older code.
        start_trial = 1
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for trial in range(start_trial, num_trials + 1):
                for num_h in density_levels:
                    random.seed(4000 + trial * 50 + num_h)
                    layout = SupermarketLayout()
                    shelf_boxes = [s.bounds for s in layout.shelves]
                    trolley_cfgs, _, _ = SupermarketScenarios.get_scenario("A", layout)
                    
                    # Spawn exactly num_h dynamic humans along open aisles and crossways
                    aisle_xs = [layout.start_x + idx * layout.aisle_spacing for idx in range(layout.num_aisles)]
                    crossway_ys = [layout.y_back_promenade, layout.y_action_alley, layout.y_front_concourse]
                    humans = []
                    for i in range(num_h):
                        if random.random() < 0.65:
                            hx = random.choice(aisle_xs) + random.uniform(-4.0, 4.0)
                            hy = random.uniform(layout.y_back_promenade + 10.0, layout.y_front_concourse - 10.0)
                        else:
                            hx = random.uniform(layout.start_x - 20.0, layout.start_x + (layout.num_aisles - 1) * layout.aisle_spacing + 20.0)
                            hy = random.choice(crossway_ys) + random.uniform(-4.0, 4.0)
                        humans.append(Human(
                            id=i + 1,
                            x=hx,
                            y=hy,
                            speed=random.uniform(0.6, 1.2),
                            state="walking" if random.random() < 0.7 else "browsing"
                        ))

                    mesh = MeshNetwork(comm_radius=350.0)
                    agents = [TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh) for c in trolley_cfgs]

                    sim_time = 0.0
                    total_discomfort = 0.0
                    latencies = []

                    while sim_time < max_time and not all(a.is_docked for a in agents):
                        for h in humans:
                            h.update(dt, layout.bounds, shelf_boxes, aisle_xs, crossway_ys)

                        for a in agents:
                            a.step(dt, humans, prox_field, current_sim_time=sim_time,
                                   shelves=shelf_boxes, peer_agents=agents)
                            latencies.append(a.last_compute_time_ms)
                            disc = prox_field.compute_penalty_at_point(a.x, a.y, humans)
                            total_discomfort += (disc / 100.0) * dt

                        layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                        sim_time += dt

                    row = {
                        "trial_id": trial,
                        "crowd_density_humans": num_h,
                        "fixed_fleet_size": 4,
                        "success_rate_pct": 100.0 if all(a.is_docked for a in agents) else 0.0,
                        "makespan_s": round(sim_time, 2),
                        "mean_replan_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
                        "discomfort_integral": round(total_discomfort, 2),
                        "v2v_mesh_packets": mesh.total_packets_transmitted
                    }
                    writer.writerow(row)
                f.flush()

        os.replace(tmp_path, csv_path)
        _write_provenance(csv_path, _row_count(csv_path))
        print(f"  -> Exported: {csv_path} ({num_trials * len(density_levels)} genuine simulation data points)")
        return csv_path

    # --------------------------------------------------------------------------
    # 4B. Decoupled Scalability: Fleet Size (Fixed Crowd N_humans = 10)
    # --------------------------------------------------------------------------
    def run_fleet_size_scalability(self, num_trials: int = 100) -> str:
        csv_path = os.path.join(self.output_dir, "scalability_fleet_size.csv")
        fieldnames = [
            "trial_id", "fleet_size_carts", "fixed_crowd_humans", "success_rate_pct",
            "makespan_s", "mean_replan_latency_ms", "corridor_mutex_wait_s", "v2v_mesh_packets"
        ]

        fleet_levels = [2, 4, 6, 8, 10, 12]
        dt = 0.05
        max_time = T_MAX_FLEET
        prox_field = ProxemicsField()
        print(f"\n[Experiment 4B] Running Genuine Fleet Size Scalability (Fixed Crowd N_humans=10, N_carts in [2..12], N={num_trials} trials)...")

        candidate_starts = [
            "N_back_0", "N_back_1", "N_back_2", "N_back_3", "N_back_4", "N_back_5",
            "N_produce_back", "N_deli_back", "N_mid_0", "N_mid_5", "N_produce_mid", "N_deli_mid"
        ]
        candidate_goals = [
            "DOCK_BAY_MAIN", "DOCK_BAY_EXPRESS", "N_front_0", "N_front_1", "N_front_2", "N_front_3",
            "N_front_4", "N_front_5", "DOCK_BAY_MAIN", "DOCK_BAY_EXPRESS", "N_produce_front", "N_deli_front"
        ]

        existing_trial_counts = {}
        valid_rows = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, mode="r", encoding="utf-8") as rf:
                    reader = csv.DictReader(rf)
                    for r in reader:
                        if "trial_id" in r and r["trial_id"].isdigit():
                            tid = int(r["trial_id"])
                            existing_trial_counts[tid] = existing_trial_counts.get(tid, 0) + 1
                            valid_rows.append(r)
            except Exception:
                existing_trial_counts = {}
                valid_rows = []

        # Always regenerate; never resume from a file written by older code.
        start_trial = 1
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()

            for trial in range(start_trial, num_trials + 1):
                t_trial_start = time.time()
                for num_c in fleet_levels:
                    random.seed(5000 + trial * 50 + num_c)
                    layout = SupermarketLayout()
                    shelf_boxes = [s.bounds for s in layout.shelves]
                    
                    # Spawn fixed 10 humans in open corridors
                    aisle_xs = [layout.start_x + idx * layout.aisle_spacing for idx in range(layout.num_aisles)]
                    crossway_ys = [layout.y_back_promenade, layout.y_action_alley, layout.y_front_concourse]
                    humans = []
                    for i in range(10):
                        if random.random() < 0.65:
                            hx = random.choice(aisle_xs) + random.uniform(-4.0, 4.0)
                            hy = random.uniform(layout.y_back_promenade + 10.0, layout.y_front_concourse - 10.0)
                        else:
                            hx = random.uniform(layout.start_x - 20.0, layout.start_x + (layout.num_aisles - 1) * layout.aisle_spacing + 20.0)
                            hy = random.choice(crossway_ys) + random.uniform(-4.0, 4.0)
                        humans.append(Human(
                            id=i + 1,
                            x=hx,
                            y=hy,
                            speed=random.uniform(0.6, 1.2),
                            state="walking" if random.random() < 0.7 else "browsing"
                        ))

                    mesh = MeshNetwork(comm_radius=350.0)
                    agents = []
                    for idx in range(num_c):
                        s_node = candidate_starts[idx % len(candidate_starts)]
                        g_node = candidate_goals[idx % len(candidate_goals)]
                        if s_node == g_node:
                            g_node = candidate_goals[(idx + 1) % len(candidate_goals)]
                        agents.append(TrolleyAgent(idx + 1, layout.graph, s_node, g_node, mesh))

                    sim_time = 0.0
                    latencies = []

                    while sim_time < max_time and not all(a.is_docked for a in agents):
                        for h in humans:
                            h.update(dt, layout.bounds, shelf_boxes, aisle_xs, crossway_ys)

                        for a in agents:
                            a.step(dt, humans, prox_field, current_sim_time=sim_time,
                                   shelves=shelf_boxes, peer_agents=agents)
                            latencies.append(a.last_compute_time_ms)

                        layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                        sim_time += dt

                    row = {
                        "trial_id": trial,
                        "fleet_size_carts": num_c,
                        "fixed_crowd_humans": 10,
                        "success_rate_pct": 100.0 if all(a.is_docked for a in agents) else 0.0,
                        "makespan_s": round(sim_time, 2),
                        "mean_replan_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
                        "corridor_mutex_wait_s": round(sum(a.wait_timer for a in agents), 2),
                        "v2v_mesh_packets": mesh.total_packets_transmitted
                    }
                    writer.writerow(row)
                    f.flush()
                print(f"  -> Finished Exp 4B trial {trial}/{num_trials} in {time.time() - t_trial_start:.2f} s", flush=True)

        os.replace(tmp_path, csv_path)
        _write_provenance(csv_path, _row_count(csv_path))
        print(f"  -> Exported: {csv_path} ({num_trials * len(fleet_levels)} genuine simulation data points)")
        return csv_path

    # --------------------------------------------------------------------------
    # Master Execution Pipeline
    # --------------------------------------------------------------------------
    def run_all(self, num_trials: int = 100) -> None:
        t_start = time.perf_counter()
        print("=" * 80)
        print("  D2RO / SW-DGO MASTER EXPERIMENTAL SUITE")
        print(f"  Executing 100% genuine kinodynamic simulations (N={num_trials} trials per condition)")
        print("=" * 80)

        self.run_baseline_comparison(num_trials)
        self.run_ablation_study(num_trials)
        self.run_cross_domain_benchmark(num_trials)
        self.run_crowd_density_scalability(num_trials)
        self.run_fleet_size_scalability(num_trials)
        self.run_mesh_anticipation_experiment(50)
        self.run_corridor_lock_experiment(50)

        t_elapsed = time.perf_counter() - t_start
        print("\n" + "=" * 80)
        print(f"  ALL 7 EXPERIMENTS COMPLETED IN {t_elapsed:.1f} SECONDS")
        print(f"  Raw CSV datasets generated in: {self.output_dir}")
        print("=" * 80)

    # --------------------------------------------------------------------------
    # 6. Mechanism-Specific Experiment A: V2V Mesh Anticipation
    #    Constructs explicit leader/follower topology: Cart A leads, Cart B is 12 m behind
    #    upstream of a divergence junction. A blockage lies ahead of A outside B's sensing radius.
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # 5. Weight Sensitivity (reviewer: "calibrated" weights were never calibrated)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # 7. Route x Yield factorial (reviewer: the matched arm was not matched on
    #    human yielding, so the routing attribution was not established)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # 8. Mechanism A under communication degradation (reviewer: test robustness
    #    where communication is causally responsible for the result)
    # --------------------------------------------------------------------------
    def run_mesh_degradation(self, num_trials: int = 20) -> str:
        r"""
        Repeats Mechanism A across a packet-loss x latency grid.

        The broad communication sweep is weak evidence by our own admission: the
        general supermarket scenario barely uses the mesh, so observing no
        degradation there says little. This experiment degrades the channel in the
        one setting where the mesh demonstrably causes the effect -- the controlled
        leader/follower topology in which the follower cannot perceive the blockage
        and can only learn of it by radio.

        If the anticipation advantage survives loss and delay here, the robustness
        claim is earned. If it decays, that decay is the interesting result and the
        abstract must be trimmed accordingly.
        """
        csv_path = os.path.join(self.output_dir, "mesh_degradation.csv")
        fieldnames = ["channel", "packet_loss_rate", "latency_s", "trial_id",
                      "mesh_enabled", "anticipation_lead_time_s",
                      "backtrack_distance_m", "path_length_m", "makespan_s", "success"]

        losses = [0.0, 0.10, 0.20]
        latencies = [0.0, 0.10, 0.20]

        print(f"\n[Experiment F] Mechanism A under degradation "
              f"({len(losses)}x{len(latencies)} channels x N={num_trials})...")

        all_rows = []
        for loss in losses:
            for lat in latencies:
                tmp_name = f"_mesh_deg_{int(loss*100):02d}_{int(lat*1000):03d}.csv"
                self.run_mesh_anticipation_experiment(
                    num_trials=num_trials, packet_loss=loss, latency_s=lat,
                    csv_name=tmp_name)
                tmp_path = os.path.join(self.output_dir, tmp_name)
                with open(tmp_path, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        all_rows.append({
                            "channel": f"loss{int(loss*100):02d}_lat{int(lat*1000):03d}ms",
                            "packet_loss_rate": loss,
                            "latency_s": lat,
                            "trial_id": r.get("trial_id"),
                            "mesh_enabled": r.get("mesh_enabled"),
                            "anticipation_lead_time_s": r.get("anticipation_lead_time_s"),
                            "backtrack_distance_m": r.get("backtrack_distance_m"),
                            "path_length_m": r.get("path_length_m"),
                            "makespan_s": r.get("makespan_s"),
                            "success": r.get("success"),
                        })
                for leftover in (tmp_path, tmp_path + ".provenance.json"):
                    if os.path.exists(leftover):
                        os.remove(leftover)
                print(f"  -> loss={loss:.2f} latency={lat:.2f}s done", flush=True)

        _atomic_write_csv(csv_path, fieldnames, all_rows)
        print(f"  -> Exported: {csv_path} ({len(all_rows)} rows)")
        return csv_path

    def run_route_yield_factorial(self, num_trials: int = 50) -> str:
        r"""
        Separates proxemic graph cost routing (H_prox) from reactive HUMAN YIELDING.

        To cleanly isolate the social routing cost term (H_prox) from V2V mesh communication,
        corridor reservation, and static routing, mesh communication and corridor mutex locks
        are disabled in all 4 cells, and the dynamic D* Lite search engine remains active throughout
        (static_route=False).

        This is a clean 2x2 factorial with identical kinematics, D* Lite replanning, collision geometry,
        and arrival criterion across all cells:

            A  H_prox OFF, yield OFF
            B  H_prox OFF, yield ON
            C  H_prox ON,  yield OFF
            D  H_prox ON,  yield ON

        which yields:
            H_prox routing main effect = C - A (yield off) and D - B (yield on)
            yielding main effect       = B - A (H_prox off) and D - C (H_prox on)
            interaction contrast       = (D - C) - (B - A)
        """
        csv_path = os.path.join(self.output_dir, "route_yield_factorial.csv")
        fieldnames = [
            "trial_id", "cell", "h_prox_routing", "social_routing", "yielding",
            "success", "timeout", "makespan_s",
            "intimate_exposure_person_ticks", "intimate_exposure_person_s",
            "intimate_encounters", "replans"
        ]

        # (cell, h_prox_routing, yielding)
        CELLS = [
            ("A_prox_off_yield_off", False, False),
            ("B_prox_off_yield_on",  False, True),
            ("C_prox_on_yield_off",  True,  False),
            ("D_prox_on_yield_on",   True,  True),
        ]

        dt = 0.05
        max_time = T_MAX_MISSION
        prox_field = ProxemicsField()
        print(f"\n[Experiment E] Route x Yield factorial "
              f"(4 cells x N={num_trials} paired trials)...")

        rows = []
        for cell, h_prox, yielding in CELLS:
            for trial in range(1, num_trials + 1):
                seed_val = 9000 + trial          # disjoint seed set
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [sh.bounds for sh in layout.shelves]
                cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mesh = MeshNetwork(comm_radius=350.0, seed=seed_val)

                agents = [
                    TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"], mesh,
                                 enable_mesh=False, enable_prox=h_prox,
                                 enable_lock=False, enable_safety=True,
                                 static_route=False, enable_yield=yielding)
                    for c in cfgs
                ]

                sim_time = 0.0
                while sim_time < max_time and not all(a.is_docked for a in agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)
                    for a in agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time,
                               shelves=shelf_boxes, peer_agents=agents)
                    sim_time += dt

                done = all(a.is_docked for a in agents)
                rows.append({
                    "trial_id": trial,
                    "cell": cell,
                    "h_prox_routing": int(h_prox),
                    "social_routing": int(h_prox),
                    "yielding": int(yielding),
                    "success": 1 if done else 0,
                    "timeout": 0 if done else 1,
                    "makespan_s": round(sim_time, 2),
                    "intimate_exposure_person_ticks": sum(a.proxemic_violations for a in agents),
                    "intimate_exposure_person_s": round(
                        sum(a.intimate_exposure_s for a in agents), 3),
                    "intimate_encounters": sum(a.intimate_encounters for a in agents),
                    "replans": sum(a.replan_count for a in agents),
                })
            print(f"  -> {cell:22s} done ({num_trials} trials)", flush=True)

        _atomic_write_csv(csv_path, fieldnames, rows)
        print(f"  -> Exported: {csv_path} ({len(rows)} rows)")
        return csv_path

    def run_weight_sensitivity(self, num_trials: int = 30) -> str:
        r"""
        Perturbs each of the four soft cost weights in turn and measures the effect.

        The method is explicitly a weighted multi-component objective, so robustness
        to the weights is not peripheral -- if the reported operating point sits on a
        narrow ridge, the result is fragile in a way the reader must be told about.

        Design
        ------
        One weight is scaled at a time by x{0.5, 0.75, 1.25, 1.5} while the other three
        are held at nominal, plus a single nominal run that serves as the x1.0 point of
        all four curves. That is 4*4 + 1 = 17 configurations, rather than 20, because
        re-running the identical nominal configuration four times on identical seeds
        would produce four identical result sets.

        Only w_D, w_M, w_H and w_S are swept. The corridor reservation is a hard
        feasibility constraint rather than a weighted term: a reserved edge is
        unavailable and an unreserved one contributes zero, so a coefficient on it has
        no operative magnitude in any reachable state and there is nothing to perturb.
        It is evaluated by the ON/OFF mechanism experiment instead. An earlier version
        of this docstring described five weights and 21 configurations, which was the
        formulation this study went on to correct.

        Seeds
        -----
        Deliberately DISJOINT from every other experiment (7000+). The nominal weights
        were hand-selected; evaluating their robustness on the same seeds used to pick
        and report them would be a form of tuning on the test set.
        """
        csv_path = os.path.join(self.output_dir, "weight_sensitivity.csv")
        fieldnames = [
            "trial_id", "config", "weight_varied", "multiplier",
            "w_D", "w_M", "w_H", "w_R", "w_S",
            "success", "makespan_s", "intimate_exposure_ticks",
            "discomfort_integral", "replans"
        ]

        nominal = {"w_D": WEIGHT_DISTANCE_WD, "w_M": WEIGHT_MESH_WM,
                   "w_H": WEIGHT_PROXEMIC_WH, "w_R": WEIGHT_MUTEX_LOCK_WR,
                   "w_S": WEIGHT_TROLLEY_WS}
        multipliers = [0.5, 0.75, 1.25, 1.5]

        configs = [("nominal", "none", 1.0, dict(nominal))]
        # w_R is deliberately absent. Reservation is a hard feasibility constraint,
        # not a weighted soft term, so it has no magnitude to perturb; it is tested
        # by the ON/OFF mechanism experiment instead.
        for key in ("w_D", "w_M", "w_H", "w_S"):
            for m in multipliers:
                w = dict(nominal)
                w[key] = nominal[key] * m
                configs.append((f"{key}x{m}", key, m, w))

        dt = 0.05
        max_time = T_MAX_MISSION
        prox_field = ProxemicsField()
        print(f"\n[Experiment C] Weight Sensitivity "
              f"({len(configs)} configurations x N={num_trials} trials, disjoint seeds)...")

        rows = []
        for cfg_name, varied, mult, w in configs:
            for trial in range(1, num_trials + 1):
                seed_val = 7000 + trial          # disjoint from 1000/3000/6000 ranges
                random.seed(seed_val)
                layout = SupermarketLayout()
                shelf_boxes = [sh.bounds for sh in layout.shelves]
                cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                mesh = MeshNetwork(comm_radius=350.0, seed=seed_val)
                agents = [TrolleyAgent(c["id"], layout.graph, c["start"], c["goal"],
                                       mesh, weights=w) for c in cfgs]

                sim_time = 0.0
                total_discomfort = 0.0
                while sim_time < max_time and not all(a.is_docked for a in agents):
                    for h in humans:
                        h.update(dt, layout.bounds, shelf_boxes)
                    for a in agents:
                        a.step(dt, humans, prox_field, current_sim_time=sim_time,
                               shelves=shelf_boxes, peer_agents=agents)
                        # Same continuous discomfort integral the ablation uses, so
                        # the two experiments report a comparable quantity.
                        point_disc = prox_field.compute_penalty_at_point(a.x, a.y, humans)
                        total_discomfort += (point_disc / 100.0) * dt
                    sim_time += dt

                rows.append({
                    "trial_id": trial, "config": cfg_name,
                    "weight_varied": varied, "multiplier": mult,
                    **{k: round(v, 4) for k, v in w.items()},
                    "success": 1 if all(a.is_docked for a in agents) else 0,
                    "makespan_s": round(sim_time, 2),
                    "intimate_exposure_ticks": sum(a.proxemic_violations for a in agents),
                    "discomfort_integral": round(total_discomfort, 3),
                    "replans": sum(a.replan_count for a in agents),
                })
            print(f"  -> {cfg_name:12s} done ({num_trials} trials)", flush=True)

        _atomic_write_csv(csv_path, fieldnames, rows)
        print(f"  -> Exported: {csv_path} ({len(rows)} rows)")
        return csv_path

    # --------------------------------------------------------------------------
    # 6. Communication Robustness (reviewer: results are ideal-channel only)
    # --------------------------------------------------------------------------
    def run_comm_robustness(self, num_trials: int = 30) -> str:
        r"""
        Sweeps V2V packet loss and one-hop latency.

        Every other experiment in this suite runs an ideal channel: MeshNetwork is
        constructed with its defaults of zero loss and zero latency. The deployment
        discussion appeals to RF attenuation in steel-fixture retail environments, so
        the paper should not imply robustness to degraded communication without
        measuring it.

        This experiment is only meaningful because the agent now honours deliver_at
        timestamps; before that fix, latency was configurable but inert, since
        fetch_inbound() defaulted to +inf and released every packet immediately.
        """
        csv_path = os.path.join(self.output_dir, "comm_robustness.csv")
        fieldnames = [
            "trial_id", "channel", "packet_loss_rate", "latency_s",
            "success", "makespan_s", "intimate_exposure_ticks",
            "mesh_packets", "replans"
        ]

        losses = [0.0, 0.05, 0.10, 0.20]
        latencies = [0.0, 0.05, 0.10, 0.20]

        dt = 0.05
        max_time = T_MAX_MISSION
        prox_field = ProxemicsField()
        print(f"\n[Experiment D] Communication Robustness "
              f"({len(losses)}x{len(latencies)} channels x N={num_trials} trials)...")

        rows = []
        for loss in losses:
            for lat in latencies:
                for trial in range(1, num_trials + 1):
                    seed_val = 8000 + trial      # disjoint seed set
                    random.seed(seed_val)
                    layout = SupermarketLayout()
                    shelf_boxes = [sh.bounds for sh in layout.shelves]
                    cfgs, humans, _ = SupermarketScenarios.get_scenario("A", layout)
                    mesh = MeshNetwork(comm_radius=350.0, packet_loss_rate=loss,
                                       latency_s=lat, seed=seed_val)
                    agents = [TrolleyAgent(c["id"], layout.graph, c["start"],
                                           c["goal"], mesh) for c in cfgs]

                    sim_time = 0.0
                    while sim_time < max_time and not all(a.is_docked for a in agents):
                        for h in humans:
                            h.update(dt, layout.bounds, shelf_boxes)
                        for a in agents:
                            a.step(dt, humans, prox_field, current_sim_time=sim_time,
                                   shelves=shelf_boxes, peer_agents=agents)
                        sim_time += dt

                    rows.append({
                        "trial_id": trial,
                        # Single label for the channel condition, so the grouped
                        # analysis has one key rather than a composite of two.
                        "channel": f"loss{int(loss * 100):02d}_lat{int(lat * 1000):03d}ms",
                        "packet_loss_rate": loss,
                        "latency_s": lat,
                        "success": 1 if all(a.is_docked for a in agents) else 0,
                        "makespan_s": round(sim_time, 2),
                        "intimate_exposure_ticks": sum(a.proxemic_violations for a in agents),
                        "mesh_packets": mesh.total_packets_transmitted,
                        "replans": sum(a.replan_count for a in agents),
                    })
                print(f"  -> loss={loss:.2f} latency={lat:.2f}s done", flush=True)

        _atomic_write_csv(csv_path, fieldnames, rows)
        print(f"  -> Exported: {csv_path} ({len(rows)} rows)")
        return csv_path

    def run_mesh_anticipation_experiment(self, num_trials: int = 50,
                                        packet_loss: float = 0.0,
                                        latency_s: float = 0.0,
                                        csv_name: str = "mesh_anticipation_experiment.csv") -> str:
        r"""
        Mechanism Experiment A: V2V anticipatory horizon extension.

        Controlled leader/follower topology in Aisle 1 of the supermarket:
          * Leader starts mid-aisle (N_mid_1) and meets a stationary pedestrian
            cluster blocking the lower aisle segment (N_mid_1 -> N_front_1).
          * Follower starts at the aisle head (N_back_1). The blockage lies
            9.45 m away -- beyond its 7.2 m onboard sensing radius -- so it cannot
            perceive the obstruction directly.
          * The follower's shortest route (12.6 m) uses the blocked segment; the
            parallel detour costs 20.4 m, so an early warning is genuinely
            actionable rather than cosmetic.

        Randomisation: blockage position along the segment and lateral jitter vary
        per trial, so trials are not replicas of one another.

        Metrics (each measured as the quantity its name claims):
          anticipation_lead_time_s = t_local_detection - t_reroute, i.e. how much
              earlier the route changed than unaided sensing would have allowed.
          backtrack_distance_m = distance travelled AWAY from the goal, measured
              identically under both conditions.
        """
        csv_path = os.path.join(self.output_dir, csv_name)
        fieldnames = [
            "trial_id", "mesh_enabled", "separation_m", "local_detection_time_s",
            "reroute_time_s", "anticipation_lead_time_s", "backtrack_distance_m",
            "path_length_m", "makespan_s", "success"
        ]

        dt = 0.05
        max_time = T_MAX_MECHANISM
        BLOCKED_EDGE = ("N_mid_1", "N_front_1")

        print(f"\n[Experiment A] V2V Mesh Anticipation "
              f"(N={num_trials} paired trials, T_max={max_time:.0f}s)...")

        def edges_of(path):
            return set(zip(path, path[1:]))

        def simulate(mesh_on, seed_val):
            random.seed(seed_val)
            layout = SupermarketLayout()
            obstacles = layout.obstacle_bounds
            g = layout.graph

            # Randomised blockage on the lower aisle segment.
            by = random.uniform(360.0, 450.0)
            bx = 310.0 + random.uniform(-6.0, 6.0)
            humans = [
                Human(id=901, x=bx, y=by, speed=0.0),
                Human(id=902, x=bx + random.uniform(4, 10),
                      y=by + random.uniform(4, 12), speed=0.0),
                Human(id=903, x=bx - random.uniform(4, 10),
                      y=by - random.uniform(4, 12), speed=0.0),
            ]

            follower_start = g.get_node("N_back_1")
            separation_m = math.hypot(follower_start.x - bx,
                                      follower_start.y - by) * PX_TO_M
            # Precondition: the follower must NOT be able to sense the blockage.
            if separation_m <= SENSING_RADIUS_M:
                return None

            mesh = MeshNetwork(comm_radius=350.0, seed=seed_val,
                               packet_loss_rate=packet_loss, latency_s=latency_s)
            leader = TrolleyAgent(1, g, "N_mid_1", "N_front_1", mesh,
                                  enable_mesh=mesh_on)
            follower = TrolleyAgent(2, g, "N_back_1", "N_front_1", mesh,
                                    enable_mesh=mesh_on)
            agents = [leader, follower]

            # Precondition: the blocked segment must actually be on the follower's plan.
            if BLOCKED_EDGE not in edges_of(follower.planner.extract_full_path()):
                return None

            prox_field = ProxemicsField()
            goal = g.get_node("N_front_1")
            prev_gap = math.hypot(follower.x - goal.x, follower.y - goal.y)
            backtrack_px = 0.0
            reroute_t = None
            detect_t = None
            sim_time = 0.0

            while sim_time < max_time and not all(a.is_docked for a in agents):
                for h in humans:
                    h.update(dt, layout.bounds, obstacles)
                for a in agents:
                    a.step(dt, humans, prox_field, current_sim_time=sim_time,
                           shelves=obstacles, peer_agents=agents)

                if reroute_t is None and not follower.is_docked:
                    if BLOCKED_EDGE not in edges_of(follower.planner.extract_full_path()):
                        reroute_t = sim_time

                if detect_t is None:
                    for h in humans:
                        if math.hypot(follower.x - h.x,
                                      follower.y - h.y) <= SENSING_RADIUS_PX:
                            detect_t = sim_time
                            break

                gap = math.hypot(follower.x - goal.x, follower.y - goal.y)
                if gap > prev_gap:
                    backtrack_px += (gap - prev_gap)
                prev_gap = gap
                sim_time += dt

            t_detect = detect_t if detect_t is not None else max_time
            lead = (t_detect - reroute_t) if reroute_t is not None else 0.0

            return {
                "separation_m": round(separation_m, 2),
                "local_detection_time_s": round(t_detect, 2),
                "reroute_time_s": round(reroute_t, 2) if reroute_t is not None else "",
                "anticipation_lead_time_s": round(lead, 3),
                "backtrack_distance_m": round(backtrack_px * PX_TO_M, 3),
                "path_length_m": round(follower.total_distance * PX_TO_M, 2),
                "makespan_s": round(sim_time, 2),
                "success": 1 if all(a.is_docked for a in agents) else 0,
            }

        rows, trial, attempts = [], 0, 0
        while trial < num_trials and attempts < num_trials * 20:
            attempts += 1
            seed_val = 5000 + attempts
            off = simulate(False, seed_val)
            on = simulate(True, seed_val)
            if off is None or on is None:
                continue
            trial += 1
            for cond, res in (("0", off), ("1", on)):
                row = {"trial_id": trial, "mesh_enabled": cond}
                row.update(res)
                rows.append(row)

        _atomic_write_csv(csv_path, fieldnames, rows)
        print(f"  -> Exported: {csv_path} ({len(rows)} rows, {trial} paired trials)")
        return csv_path

    # --------------------------------------------------------------------------
    # 7. Mechanism-Specific Experiment B: Corridor Mutex Lock
    #    Two carts approach same single-file corridor from opposite ends.
    #    Lock ON: one waits at alcove, conflict-free corridor entry.
    #    Lock OFF: opposing carts enter single file simultaneously -> head-on deadlock.
    # --------------------------------------------------------------------------
    def run_corridor_lock_experiment(self, num_trials: int = 50) -> str:
        r"""
        Mechanism Experiment B: distributed corridor mutex.

        Two carts approach one explicitly designated single-file corridor from
        OPPOSITE ends, with randomised arrival offsets. The corridor edge is
        verified to be single-file before the trial runs.

        Metric definitions (each measured as the quantity its name claims):
          head_on_events   -- DISCRETE geometric encounters: both carts inside the
                              corridor, within HEAD_ON_CONFLICT_RADIUS, headings
                              opposed by more than 90 deg. Counted once on entry,
                              not once per control tick.
          corridor_time_s  -- true corridor occupancy: first entry to final exit of
                              the designated edge, NOT whole-mission makespan.
          deadlocks        -- genuine routing deadlocks only; orderly yielding at a
                              reserved corridor is excluded by construction.
        """
        csv_path = os.path.join(self.output_dir, "corridor_lock_experiment.csv")
        fieldnames = [
            "trial_id", "lock_enabled", "arrival_offset_s", "head_on_events",
            "deadlocks", "lock_wait_s", "corridor_time_s", "timeout",
            "makespan_s", "success",
            # Diversion evidence. The reservation protocol raises success without
            # producing measurable queueing, so the mechanism claim rests on showing
            # that agents LEAVE the contested corridor rather than wait at it. These
            # columns put that evidence in the released dataset instead of leaving it
            # to diagnostic instrumentation quoted in prose.
            "nodes_outside_corridor",   # distinct off-corridor vertices occupied
            "replans",                  # D* Lite repairs, i.e. route reconsiderations
            "total_lock_wait_s"         # accumulator that corridor release cannot reset
        ]

        dt = 0.05
        max_time = T_MAX_MECHANISM
        prox_field = ProxemicsField()

        print(f"\n[Experiment B] Corridor Mutex Lock "
              f"(N={num_trials} paired trials, T_max={max_time:.0f}s)...")

        CORRIDOR = ("N_back_2", "N_mid_2")   # designated single-file aisle segment

        def simulate(lock_on, seed_val):
            # Seed BEFORE sampling any trial parameter, so the offset is reproducible
            # and identical across the paired lock ON/OFF conditions.
            random.seed(seed_val)
            arrival_offset = random.uniform(0.0, 3.0)

            layout = SupermarketLayout()
            obstacles = layout.obstacle_bounds
            g = layout.graph

            edge = g.get_edge(*CORRIDOR)
            if edge is None or not edge.is_single_file:
                return None   # precondition: must be a genuine single-file corridor

            mesh = MeshNetwork(comm_radius=350.0, seed=seed_val)
            # Opposite ends of the SAME corridor, each wanting the other's side.
            a1 = TrolleyAgent(1, g, "N_back_2", "N_front_2", mesh, enable_lock=lock_on)
            a2 = TrolleyAgent(2, g, "N_front_2", "N_back_2", mesh, enable_lock=lock_on)
            agents = [a1, a2]

            corridor_nodes = {"N_back_2", "N_mid_2", "N_front_2"}
            # Vertices occupied outside the contested corridor. A diverting agent
            # accumulates these; a queueing agent does not leave the corridor at all.
            visited_outside = set()
            entry_t = None
            exit_t = None
            in_conflict = False
            head_on_events = 0
            sim_time = 0.0

            while sim_time < max_time and not all(a.is_docked for a in agents):
                for i, a in enumerate(agents):
                    if i == 1 and sim_time < arrival_offset:
                        continue          # staggered arrival
                    a.step(dt, [], prox_field, current_sim_time=sim_time,
                           shelves=obstacles, peer_agents=agents)

                for a in agents:
                    if a.current_node not in corridor_nodes:
                        visited_outside.add(a.current_node)

                # --- discrete head-on encounter detection -------------------- #
                both_inside = (a1.current_node in corridor_nodes and
                               a2.current_node in corridor_nodes and
                               not a1.is_docked and not a2.is_docked)
                conflict_now = False
                if both_inside:
                    gap = math.hypot(a1.x - a2.x, a1.y - a2.y)
                    if gap < HEAD_ON_CONFLICT_RADIUS_PX:
                        dtheta = abs((a1.heading - a2.heading + math.pi)
                                     % (2 * math.pi) - math.pi)
                        conflict_now = dtheta > math.pi * 0.5
                if conflict_now and not in_conflict:
                    head_on_events += 1        # count the ENTRY, not every tick
                in_conflict = conflict_now

                # --- true corridor occupancy window -------------------------- #
                occupied = any(a.current_node in corridor_nodes and not a.is_docked
                               for a in agents)
                if occupied and entry_t is None:
                    entry_t = sim_time
                if entry_t is not None and not occupied:
                    exit_t = sim_time

                layout.graph.decay_mesh_penalties(dt, decay_rate=0.1386294)
                sim_time += dt

            if entry_t is not None and exit_t is None:
                exit_t = sim_time
            corridor_time = (exit_t - entry_t) if entry_t is not None else 0.0
            done = all(a.is_docked for a in agents)

            return {
                "arrival_offset_s": round(arrival_offset, 3),
                "head_on_events": head_on_events,
                "deadlocks": sum(a.deadlock_count for a in agents),
                "lock_wait_s": round(sum(a.lock_wait_time for a in agents), 2),
                "corridor_time_s": round(corridor_time, 2),
                "timeout": 0 if done else 1,
                "makespan_s": round(sim_time, 2),
                "success": 1 if done else 0,
                "nodes_outside_corridor": len(visited_outside),
                "replans": sum(a.replan_count for a in agents),
                "total_lock_wait_s": round(
                    sum(a.total_lock_wait_time for a in agents), 3),
            }

        rows = []
        trial = 0
        attempts = 0
        while trial < num_trials and attempts < num_trials * 20:
            attempts += 1
            seed_val = 6000 + attempts
            on = simulate(True, seed_val)
            off = simulate(False, seed_val)
            if on is None or off is None:
                continue
            trial += 1
            for cond, res in (("1", on), ("0", off)):
                row = {"trial_id": trial, "lock_enabled": cond}
                row.update(res)
                rows.append(row)

        _atomic_write_csv(csv_path, fieldnames, rows)
        print(f"  -> Exported: {csv_path} ({len(rows)} rows, {trial} paired trials)")
        return csv_path


if __name__ == "__main__":
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "data")
    runner = ExperimentRunner(output_dir=out_dir)
    runner.run_all(num_trials=100)
