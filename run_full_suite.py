"""
Full experimental suite driver.

Regenerates every dataset in experiments/data/ using the corrected kinematics,
normalised cost terms, bounded perception, multi-hop mesh and distributed
corridor reservation protocol.

Robustness
----------
Each experiment runs in its OWN subprocess. Two reasons:

  * Isolation. A crash in one experiment must not abort the others, and must not
    leave the suite silently reporting success with missing data. The previously
    published scalability dataset was truncated to 444 of 600 rows exactly this
    way -- the run died partway and nobody noticed.
  * Fresh interpreter state. Long runs on Windows have been observed to abort with
    an access violation after several hundred sub-runs; a per-experiment process
    keeps memory and stack usage bounded.

Every dataset is written atomically and always regenerated from scratch, so a
rerun can never republish results produced by an older version of the code.

Usage
-----
    python run_full_suite.py              # run everything
    python run_full_suite.py --only 4A    # run one experiment
    python run_full_suite.py --list       # show experiment keys
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

OUT = os.path.join(BASE, "experiments", "data")
MAX_ATTEMPTS = 5

# key -> (human-readable name, runner method, trial count, expected row count)
EXPERIMENTS = {
    "1":  ("Benchmark comparison",      "run_baseline_comparison",        100, 700),
    "2":  ("Component ablation",        "run_ablation_study",             100, 700),
    "3":  ("Cross-domain benchmark",    "run_cross_domain_benchmark",     100, 300),
    "4A": ("Crowd-density scalability", "run_crowd_density_scalability",  100, 600),
    "4B": ("Fleet-size scalability",    "run_fleet_size_scalability",     100, 600),
    "A":  ("Mesh anticipation",         "run_mesh_anticipation_experiment", 50, 100),
    "B":  ("Corridor mutex lock",       "run_corridor_lock_experiment",     50, 100),
    # 21 weight configurations x 30 trials; 16 channel conditions x 30 trials.
    "C":  ("Weight sensitivity",        "run_weight_sensitivity",           30, 510),
    "D":  ("Communication robustness",  "run_comm_robustness",              30, 480),
    "E":  ("Route x Yield factorial",   "run_route_yield_factorial",        50, 200),
    "F":  ("Mesh under degradation",    "run_mesh_degradation",             20, 360),
}


def _run_one(key: str) -> None:
    """
    Executes a single experiment in this process, on the MAIN thread.

    An earlier revision ran the work inside a worker thread with an enlarged stack.
    That turned out to make matters worse, not better: identical workloads that
    complete cleanly on the main thread aborted with an access violation inside the
    worker. Isolation is provided by the per-experiment subprocess instead.
    """
    name, method, trials, _ = EXPERIMENTS[key]
    from d2ro.sim.run_experiments import ExperimentRunner
    runner = ExperimentRunner(output_dir=OUT)
    getattr(runner, method)(trials)


def _spawn(key: str) -> int:
    """Runs one experiment in a fresh subprocess; returns its exit code."""
    return subprocess.call([sys.executable, "-u", os.path.abspath(__file__),
                            "--child", key], cwd=BASE)


def _row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single experiment key")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (name, _, trials, rows) in EXPERIMENTS.items():
            print(f"  {k:3s}  {name:30s} trials={trials:4d}  expected_rows={rows}")
        return

    if args.child:
        _run_one(args.child)
        return

    keys = [args.only] if args.only else list(EXPERIMENTS)
    results = []
    t_all = time.time()

    for key in keys:
        name = EXPERIMENTS[key][0]
        print(f"\n{'=' * 70}\n>>> [{key}] {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        # Hardware on this workstation is intermittently faulty, so a failed run is
        # retried several times before the dataset is declared incomplete. Retries are
        # safe: every experiment regenerates its dataset from scratch and writes it
        # atomically, so a crashed attempt cannot leave partial rows behind.
        code = _spawn(key)
        attempt = 1
        while code != 0 and attempt < MAX_ATTEMPTS:
            attempt += 1
            print(f"    [retry {attempt}/{MAX_ATTEMPTS}] {name} exited {code}", flush=True)
            code = _spawn(key)
        elapsed = time.time() - t0
        results.append((key, name, code, elapsed))
        status = "done" if code == 0 else f"FAILED (exit {code})"
        print(f"    [{status}] {name} in {elapsed:.1f}s", flush=True)

    print(f"\n{'=' * 70}\nSUMMARY  (total {time.time() - t_all:.1f}s)\n{'=' * 70}", flush=True)
    ok = True
    for key, name, code, elapsed in results:
        expected = EXPERIMENTS[key][3]
        # map key -> output file for a completeness check
        fname = {
            "1": "benchmark_comparison.csv",
            "2": "ablation_study.csv",
            "3": "cross_domain_benchmark.csv",
            "4A": "scalability_crowd_density.csv",
            "4B": "scalability_fleet_size.csv",
            "A": "mesh_anticipation_experiment.csv",
            "B": "corridor_lock_experiment.csv",
            "C": "weight_sensitivity.csv",
            "D": "comm_robustness.csv",
            "E": "route_yield_factorial.csv",
            "F": "mesh_degradation.csv",
        }[key]
        rows = _row_count(os.path.join(OUT, fname))
        complete = (code == 0 and rows == expected)
        ok = ok and complete
        flag = "OK " if complete else "BAD"
        print(f"  [{flag}] {key:3s} {name:30s} rows={rows:4d}/{expected:<4d} "
              f"exit={code} {elapsed:7.1f}s", flush=True)

    print("\nAll datasets complete." if ok else
          "\nSOME DATASETS ARE INCOMPLETE - do not publish these results.", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
