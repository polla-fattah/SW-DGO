# SW-DGO — Socially-Weighted Distributed Graph Optimization

**Multi-agent route planning for autonomous service fleets in crowded, human-shared indoor space.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-green)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightblue)](LICENSE-DATA)

> **▶ [Try the simulator in your browser](https://polla-fattah.github.io/SW-DGO/)** — no install.
> It runs the *actual* Python planner compiled to WebAssembly via Pyodide, not a
> JavaScript re-implementation, so what you watch is the code in this repository.

---

## What problem this solves

Retail shopping carts, hospital pushchairs and airport luggage trolleys increasingly
drive themselves through spaces full of people who are under no obligation to get out
of the way. Classical planners handle this badly in two specific ways:

- **They treat pedestrians as obstacles to clear, not people whose personal space
  carries a cost.** A shortest-path planner will drive straight through a browsing
  shopper's intimate zone because nothing in its objective says not to.
- **They have no mechanism for symmetric conflicts.** Two carts entering the same
  single-file aisle from opposite ends is not a collision-avoidance problem; it is a
  coordination problem, and reactive avoidance can only oscillate.

SW-DGO addresses both by pricing sociability *inside the graph search* rather than
bolting a reactive layer on top.

## The formulation

Each agent maintains its own directed cost graph and minimises a **four-term edge
traversal cost** subject to **one hard feasibility constraint**:

```
minimise  C_i(u,v,t) = w_D · D(u,v)          # Euclidean distance
                     + w_M · W_mesh(u,v,t)    # V2V anticipatory congestion field
                     + w_H · H_prox(v,t)      # human proxemics (2D asymmetric Gaussian)
                     + w_S · S_trolley(v,t)   # non-holonomic clearance envelope

subject to  (u,v) ∉ E_reserved(t)             # directional corridor reservation
```

solved by **D\* Lite** incremental repair, so a changing world re-plans only the
affected subgraph rather than the whole map.

Reservation is a *constraint*, not a fifth weighted term. A reserved edge is
unavailable at any coefficient and an unreserved one contributes zero, so a weight on
it would have no operative magnitude in any reachable state.

## What the experiments show

4,650 kinodynamic trials across eleven experiments, in three synthetic topologies
(supermarket, hospital, airport). Every number below is regenerated from the raw CSVs
in [`experiments/data/`](experiments/data/) by [the analysis
pipeline](analysis/scripts/analyze_results.py); none is typed by hand.

| Planner | Success | Makespan (s) | Intimate exposure (person-s, median) |
|:--------|:-------:|:------------:|:------------------------------------:|
| **SW-DGO (proposed)** | 99.0% | 47.18 ± 13.40 | **0.00** |
| Local Social D\* Lite | 100.0% | **39.06 ± 15.12** | **0.00** |
| Static A\* (matched controller) | 100.0% | 19.20 | 6.40 |
| Static A\* | 100.0% | **18.00** | 6.40 |
| Artificial Potential Fields | 100.0% | 34.54 | 10.18 |

**Social routing does the work.** A 2×2 factorial with the mesh and reservation
disabled in every cell isolates the proxemic cost term: enabling it alone cuts
intimate-space exposure by 6.40 person-seconds (95% CI [−6.48, −6.31],
*p* = 1.5 × 10⁻⁹), at a cost of 19.03 s of travel time. Reactive yielding adds no
statistically detectable reduction on top of it (*p* = 0.59).

**The distributed layer is conditional, and we say so.** An ordinary human-aware
planner with no mesh and no reservation matches the full system's social compliance
exactly (*p* = 1) while finishing ~8 s sooner. The distributed machinery earns its
cost only under the topologies it was designed for — a blockage outside a follower's
sensing range (anticipatory rerouting 10.7 s early) and a single-file corridor
entered from both ends (success 36% → 88%). A deployment that never produces those
should run the local planner.

## Repository layout

```
d2ro/                        Simulation package
├── core/
│   ├── agent.py             TrolleyAgent: D* Lite + the four cost terms + V2V mesh
│   ├── dstar_lite.py        Incremental replanner (admissible scaled heuristic)
│   ├── graph.py             Directed cost graph; reservation as a hard constraint
│   ├── human.py             Pedestrian model + 2D asymmetric Gaussian proxemics
│   ├── mesh_network.py      V2V broadcast with exponential time-decay, loss, latency
│   ├── metrics.py           Exposure/encounter instrumentation (person-time)
│   └── units.py             SI calibration: px↔m, speeds, cost normalisation
├── baselines/               Static A*, APF, ORCA, decentralised local MAPF
├── environments/            Supermarket, hospital (turnout alcoves), airport
├── sim/run_experiments.py   All eleven experiments
└── tests/                   67 tests across 14 files

docs/                        Browser simulator (Pyodide/WASM) — served by GitHub Pages
scripts/                     Desktop demos; web-bundle build and freshness check
experiments/data/            Raw CSVs, each with a provenance fingerprint
analysis/scripts/            The single statistics pipeline
```

## Quick start

```bash
pip install -r requirements.txt
```

Watch it run:

```bash
python scripts/demo_supermarket.py
```

Reproduce every dataset (~30 minutes):

```bash
PYTHONDONTWRITEBYTECODE=1 python run_full_suite.py
python analysis/scripts/analyze_results.py
```

Run the tests:

```bash
python -m pytest d2ro/tests/ -q
```

## How reproducibility is enforced

Two mechanisms, both mechanical rather than aspirational.

**Dataset provenance.** Every CSV is written alongside a `.provenance.json` recording
a SHA-256 fingerprint of the entire simulation package that produced it. The analysis
pipeline recomputes that fingerprint and **refuses to emit results for any dataset
whose code has changed since**. Editing anything under `d2ro/` — even a comment —
marks every dataset stale and forces regeneration. That is deliberate: it is the only
way to guarantee that reported numbers came from the code you are reading.

The fingerprint sorts paths globally and normalises line endings, so a Windows and a
Linux checkout agree. Both were real bugs before they were fixed.

**Web bundle freshness.** The browser demo executes a snapshot of the Python sources
embedded in `docs/python_bundle.js`. Nothing about editing the package refreshes it,
so the demo could silently drift from the code. The bundle now records a fingerprint
of its sources and `scripts/check_web_bundle.py` fails on mismatch, in CI:

```bash
python scripts/check_web_bundle.py     # OK, or tells you to rebuild
python scripts/build_web_bundle.py     # rebuild after changing d2ro/
```

## Known limitations

Stated plainly, because they bound what the results mean.

- **Pedestrians are non-reciprocal.** Simulated humans walk, browse and avoid
  fixtures, but do not perceive or react to the robots. Reported exposure is an upper
  bound on what a real crowd — which would partly step aside — would produce.
- **Weights are hand-selected.** `[w_D, w_M, w_H, w_S] = [1.0, 1.5, 2.0, 1.2]`.
  Perturbing each by ×0.5–1.5 holds success at 97–100%, so the operating point is a
  plateau, but no calibration procedure was run and no optimality is claimed. The
  sweep is one-at-a-time; interactions are untested.
- **Sensing is range-limited, not occlusion-aware.** Pedestrians are selected by
  Euclidean distance with no ray-casting against shelves.
- **Fleet scaling is bounded.** Success falls to 78% at twelve carts in a 36 × 24 m
  floorplan as mesh traffic grows super-linearly.
- **ORCA and local MAPF are our own implementations** and complete 0% of missions
  under the common arrival criterion. They are reported as properties of *our
  implementations*, not of the algorithms, and no conclusion depends on them.
- **Everything is simulation.** No physical fleet; localisation is assumed, not
  estimated.
- **We report *when* the distributed layer helps, not *how often*.** The base rate at
  which the enabling topologies arise in a real facility is unmeasured, so per-event
  effects cannot yet be converted into an expected benefit.

## Future work

Ordered roughly by how much they would change the conclusions.

1. **Base-rate measurement** — instrument how frequently out-of-sight blockages and
   contested single-file corridors actually occur, turning the conditional result into
   a quantitative deployment rule.
2. **Joint weight calibration** — treat it as multi-objective (a Pareto front over
   makespan and social exposure) rather than a one-at-a-time sweep.
3. **Reciprocal pedestrian model** — recorded trajectories or a social-force model, so
   exposure reflects a two-sided interaction.
4. **Baseline validation** — reproduce the ORCA implementation against RVO2 on a
   canonical benchmark.
5. **Mesh traffic suppression** — relevance filtering and spatial scoping, to lift the
   fleet-size ceiling.
6. **Heading-augmented search** — put turn cost into the graph state space rather than
   only the motion layer.
7. **Heterogeneous fleets** — vehicle-specific agility weights in the reservation
   tuple, for mixed pods, scrubbers and wheelchairs.

## Licence

| Scope | Licence |
|:------|:--------|
| Source code | [MIT](LICENSE) |
| Simulation datasets | [CC BY 4.0](LICENSE-DATA) |

The code and data are openly licensed precisely so that every reported number can be
independently reproduced and checked.

## Citation

A manuscript describing this work is under peer review. Citation details will be
added on acceptance; until then, please cite this repository.
