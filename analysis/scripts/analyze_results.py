"""
Single statistical analysis pipeline for the D2RO / SW-DGO experiments.

This is the ONLY place statistics are computed. It replaces the two previous
pipelines, which disagreed with one another: one used exact Student-t intervals
while the other used a 1.96 x SEM normal approximation and described an
independent-samples Welch test as a "paired Welch's t-test".

Methodology
-----------
Every experiment reuses the same seed across conditions, so trials are naturally
PAIRED. The analysis exploits that pairing rather than discarding it, and picks a
test appropriate to each outcome type:

  binary outcome (mission success)
      McNemar's exact test on discordant pairs, plus a Wilson score interval for
      each rate. A t-test on a 0/1 variable is not appropriate and is not used.

  continuous outcome (makespan, corridor time)
      Paired t-test when the paired differences pass a normality screen
      (Shapiro-Wilk), otherwise the Wilcoxon signed-rank test. The choice made is
      recorded per comparison so it is auditable.

  count / exposure outcome (intimate exposure, head-on events)
      These are zero-inflated and heavily skewed -- a mean +/- SD summary of
      "0.59 +/- 5.90" is not describing a symmetric distribution. Reported with
      medians and IQR, compared with a paired bootstrap of the mean difference
      (BCa-free percentile interval) and a Wilcoxon test.

Effect sizes accompany every comparison, because a p-value alone says nothing
about magnitude. Multiplicity is controlled with Holm-Bonferroni across the
family of comparisons within each experiment.

Outputs
-------
  experiments/data/analysis_results.json   machine-readable, the single source
                                           of truth consumed by the table and
                                           figure generators
  experiments/data/analysis_report.md      human-readable summary

Datasets that are missing or incomplete are reported as such. They are never
silently skipped and never back-filled with values from an earlier run.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")

_ALLOW_UNVERIFIED = False
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260815

# The simulation control step. Exposure counters increment once per human per tick,
# so a tick count multiplied by this is person-seconds exactly.
CONTROL_DT_S = 0.05

# Expected row counts; a dataset that does not match is treated as incomplete.
EXPECTED_ROWS = {
    "benchmark_comparison.csv": 700,   # 7 planners x 100 paired trials
    "ablation_study.csv": 700,   # 7 configurations x 100 trials
    "cross_domain_benchmark.csv": 300,
    "scalability_crowd_density.csv": 600,
    "scalability_fleet_size.csv": 600,
    "mesh_anticipation_experiment.csv": 100,
    "corridor_lock_experiment.csv": 100,
    "weight_sensitivity.csv": 510,
    "comm_robustness.csv": 480,
    "route_yield_factorial.csv": 200,
    "mesh_degradation.csv": 360,
}

D2RO_LABEL = "D2RO (SW-DGO Proposed)"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def current_code_fingerprint() -> str:
    """SHA-256 over the simulation source tree (must match run_experiments.py)."""
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

def load(name: str) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """
    Returns (rows, status).

    Status is 'ok' only when the dataset exists, has the expected number of rows,
    AND carries a provenance stamp matching the current source tree. A dataset
    produced by superseded code is reported as STALE rather than analysed: results
    must never outlive the code that generated them.
    """
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None, "missing"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected = EXPECTED_ROWS.get(name)
    if expected is not None and len(rows) != expected:
        return rows, f"incomplete: {len(rows)}/{expected}"

    prov_path = path + ".provenance.json"
    if not os.path.exists(prov_path):
        status = "unverified: no provenance stamp (generated by older code)"
        return rows, (status if not _ALLOW_UNVERIFIED else "provisional: " + status)
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("code_fingerprint") != current_code_fingerprint():
        status = (f"STALE: generated by code {prov.get('code_fingerprint')}, "
                  f"current is {current_code_fingerprint()}")
        return rows, (status if not _ALLOW_UNVERIFIED else "provisional: " + status)
    return rows, "ok"


def fnum(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def describe(values: Sequence[float]) -> Dict[str, Any]:
    """Mean/SD with Student-t CI, plus median/IQR for skewed outcomes."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = arr.size
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and sem > 0:
        lo, hi = stats.t.interval(1 - ALPHA, df=n - 1, loc=mean, scale=sem)
    else:
        lo = hi = mean
    q1, med, q3 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    return {
        "n": n, "mean": mean, "sd": sd, "sem": sem,
        "ci95": [float(lo), float(hi)],
        "median": med, "iqr": [q1, q3],
        "min": float(arr.min()), "max": float(arr.max()),
        "zero_fraction": float((arr == 0).mean()),
    }


def wilson_interval(successes: int, n: int) -> List[float]:
    """Wilson score interval - valid for proportions near 0 and 1, unlike normal."""
    if n == 0:
        return [0.0, 0.0]
    z = stats.norm.ppf(1 - ALPHA / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


# --------------------------------------------------------------------------- #
# Paired inference
# --------------------------------------------------------------------------- #
def paired_continuous(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """
    Paired comparison of a continuous outcome.

    Uses a paired t-test when the paired differences look approximately normal,
    otherwise Wilcoxon signed-rank. The test actually used is reported.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "test": "insufficient data"}

    xa = np.array([p[0] for p in pairs], dtype=float)
    xb = np.array([p[1] for p in pairs], dtype=float)
    diff = xa - xb

    if np.allclose(diff, 0):
        return {"n_pairs": len(pairs), "test": "identical", "p": 1.0,
                "mean_difference": 0.0, "cohens_dz": 0.0}

    normal = True
    if 3 <= diff.size <= 5000:
        try:
            normal = stats.shapiro(diff).pvalue > 0.05
        except Exception:
            normal = False

    if normal:
        res = stats.ttest_rel(xa, xb)
        test = "paired t-test"
    else:
        res = stats.wilcoxon(xa, xb, zero_method="zsplit")
        test = "Wilcoxon signed-rank"

    sd_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    dz = float(diff.mean() / sd_diff) if sd_diff > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "test": test,
        "statistic": float(res.statistic),
        "p": float(res.pvalue),
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "cohens_dz": dz,
        "normal_differences": bool(normal),
    }


def paired_bootstrap(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Percentile bootstrap CI for the mean paired difference (skewed counts)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n_pairs": len(pairs)}
    diff = np.array([p[0] - p[1] for p in pairs], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, diff.size, size=(BOOTSTRAP_RESAMPLES, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "n_pairs": len(pairs),
        "mean_difference": float(diff.mean()),
        "ci95_difference": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def mcnemar(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    """Exact McNemar test on paired binary outcomes."""
    pairs = [(int(x), int(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n01 = sum(1 for x, y in pairs if x == 0 and y == 1)
    n10 = sum(1 for x, y in pairs if x == 1 and y == 0)
    disc = n01 + n10
    p = 1.0 if disc == 0 else float(stats.binomtest(n10, disc, 0.5).pvalue)
    return {
        "n_pairs": len(pairs),
        "test": "McNemar (exact)",
        "b_only": n01, "a_only": n10, "discordant": disc, "p": p,
    }


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values across a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[key] = running
    return adjusted


# --------------------------------------------------------------------------- #
# Experiment analyses
# --------------------------------------------------------------------------- #
def analyse_benchmark() -> Dict[str, Any]:
    rows, status = load("benchmark_comparison.csv")
    out: Dict[str, Any] = {"status": status}
    if rows is None:
        return out

    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][int(r["trial_id"])] = r
    trials = sorted(set.intersection(*[set(v) for v in by.values()])) if by else []
    out["n_trials"] = len(trials)

    def series(method, key):
        return [fnum(by[method][t], key) for t in trials]

    def succ(method):
        return [int(by[method][t]["success"]) for t in trials]

    groups = {}
    for method in by:
        s = succ(method)
        tt_all = series(method, "travel_time_s")
        tt_ok = [by[method][t] for t in trials if by[method][t]["success"] == "1"]
        groups[method] = {
            "success_rate": 100.0 * sum(s) / len(s) if s else 0.0,
            "success_ci95": [100 * x for x in wilson_interval(sum(s), len(s))],
            "makespan_all": describe(tt_all),
            "makespan_successful": describe([fnum(r, "travel_time_s") for r in tt_ok]),
            "deadlocks": describe(series(method, "deadlocks")),
            # person-ticks (legacy: one increment per human per tick)
            "intimate_exposure": describe(series(method, "proxemic_violations")),
            # the interpretable units: person-seconds inside the boundary, and
            # distinct inward boundary crossings
            "exposure_person_s": describe(series(method, "intimate_exposure_person_s")),
            "intimate_encounters": describe(series(method, "intimate_encounters")),
            # Whole control step: mesh, proxemics, safety, yielding, motion --
            # averaged over EVERY tick, including ticks with no repair. This is
            # controller-step compute time, NOT D* Lite repair latency.
            "step_compute_ms": describe(series(method, "avg_replan_latency_ms")),
            # Kept under the old key so existing consumers do not silently break.
            "replan_latency_ms": describe(series(method, "avg_replan_latency_ms")),
            # D* Lite repair latency proper: measured around compute_shortest_path()
            # and recorded only on ticks where a repair actually occurred. The tail
            # matters more than the mean for a real-time claim, so p95 and max are
            # carried through rather than collapsed into an average.
            "repair_median_ms": describe(series(method, "repair_median_ms")),
            "repair_p95_ms": describe(series(method, "repair_p95_ms")),
            "repair_max_ms": describe(series(method, "repair_max_ms")),
            "repair_count": describe(series(method, "repair_n")),
            "mesh_packets": describe(series(method, "mesh_packets")),
        }
    out["groups"] = groups

    comparisons, raw_p = {}, {}
    if D2RO_LABEL in by:
        for method in by:
            if method == D2RO_LABEL:
                continue
            c = {
                "success": mcnemar(succ(D2RO_LABEL), succ(method)),
                "makespan": paired_continuous(series(D2RO_LABEL, "travel_time_s"),
                                              series(method, "travel_time_s")),
                # Person-seconds, not the raw tick count in `proxemic_violations`.
                # The Wilcoxon p is identical either way -- ranks are invariant
                # under a positive rescale -- but the effect size and its interval
                # then carry the unit the manuscript actually reasons in, instead
                # of a count that only means something at dt = 0.05 s.
                "intimate_exposure": {
                    **paired_continuous(series(D2RO_LABEL, "intimate_exposure_person_s"),
                                        series(method, "intimate_exposure_person_s")),
                    "bootstrap": paired_bootstrap(
                        series(D2RO_LABEL, "intimate_exposure_person_s"),
                        series(method, "intimate_exposure_person_s")),
                },
            }
            comparisons[method] = c
            raw_p[f"{method}|success"] = c["success"]["p"]
            raw_p[f"{method}|makespan"] = c["makespan"].get("p", 1.0)
            raw_p[f"{method}|intimate"] = c["intimate_exposure"].get("p", 1.0)

    out["comparisons"] = comparisons
    out["holm_adjusted_p"] = holm(raw_p) if raw_p else {}
    return out


def analyse_grouped(fname: str, group_key: str,
                    metrics: Dict[str, str]) -> Dict[str, Any]:
    """Generic per-group descriptive analysis (ablation, cross-domain, scaling)."""
    rows, status = load(fname)
    out: Dict[str, Any] = {"status": status}
    if rows is None:
        return out
    by = defaultdict(list)
    for r in rows:
        by[r[group_key]].append(r)
    groups = {}
    for name, rs in by.items():
        entry = {"n": len(rs)}
        for label, col in metrics.items():
            if col in rs[0]:
                entry[label] = describe([fnum(r, col) for r in rs])
        # Person-ticks are an artefact of the integration step: the counter
        # increments once per human per tick, so multiplying by dt yields person-
        # seconds exactly. Exporting the converted series keeps every exposure
        # figure and table in the one unit the manuscript reasons about, rather
        # than leaving a tick count for a caption to explain away.
        if "intimate_exposure_ticks" in rs[0]:
            entry["exposure_person_s"] = describe(
                [fnum(r, "intimate_exposure_ticks") * CONTROL_DT_S for r in rs])
        if "success" in rs[0]:
            s = [int(r["success"]) for r in rs]
            entry["success_rate"] = 100.0 * sum(s) / len(s)
            entry["success_ci95"] = [100 * x for x in wilson_interval(sum(s), len(s))]
        if "success_rate_pct" in rs[0]:
            vals = [fnum(r, "success_rate_pct") for r in rs]
            ok = sum(1 for v in vals if v and v > 50)
            entry["success_rate"] = 100.0 * ok / len(vals)
            entry["success_ci95"] = [100 * x for x in wilson_interval(ok, len(vals))]
        groups[name] = entry
    out["groups"] = groups
    return out


def analyse_paired_mechanism(fname: str, cond_key: str, on_value: str,
                             metrics: List[str]) -> Dict[str, Any]:
    """Paired ON/OFF mechanism experiment (mesh anticipation, corridor lock)."""
    rows, status = load(fname)
    out: Dict[str, Any] = {"status": status}
    if rows is None:
        return out

    on, off = {}, {}
    for r in rows:
        (on if r[cond_key] == on_value else off)[int(r["trial_id"])] = r
    trials = sorted(set(on) & set(off))
    out["n_pairs"] = len(trials)
    if not trials:
        return out

    out["conditions"] = {}
    for label, src in (("on", on), ("off", off)):
        entry = {}
        for m in metrics:
            if m in src[trials[0]]:
                entry[m] = describe([fnum(src[t], m) for t in trials])
        if "success" in src[trials[0]]:
            s = [int(src[t]["success"]) for t in trials]
            entry["success_rate"] = 100.0 * sum(s) / len(s)
            entry["success_ci95"] = [100 * x for x in wilson_interval(sum(s), len(s))]
        out["conditions"][label] = entry

    comps, raw_p = {}, {}
    for m in metrics:
        if m not in on[trials[0]]:
            continue
        a = [fnum(on[t], m) for t in trials]
        b = [fnum(off[t], m) for t in trials]
        comps[m] = {**paired_continuous(a, b), "bootstrap": paired_bootstrap(a, b)}
        raw_p[m] = comps[m].get("p", 1.0)
    if "success" in on[trials[0]]:
        comps["success"] = mcnemar([int(on[t]["success"]) for t in trials],
                                   [int(off[t]["success"]) for t in trials])
        raw_p["success"] = comps["success"]["p"]

    out["comparisons"] = comps
    out["holm_adjusted_p"] = holm(raw_p) if raw_p else {}
    return out


def analyse_factorial() -> Dict[str, Any]:
    r"""
    2x2 factorial: proxemic graph cost (H_prox) x reactive yielding.

    Every cell runs the same dynamic D* Lite planner with mesh and reservation
    disabled, so the only things that vary are the two named factors. The cells are

        A  H_prox OFF, yield OFF        C  H_prox ON,  yield OFF
        B  H_prox OFF, yield ON         D  H_prox ON,  yield ON

    Cell summaries alone cannot support a claim about a main effect or an
    interaction, so this function computes the pre-specified paired contrasts and
    tests them. Trials are seed-paired across cells, which is what licenses the
    pairing: trial t in cell A and trial t in cell D see the same pedestrian
    realisation.

        routing effect      C - A  (yield off)      D - B  (yield on)
        yielding effect     B - A  (H_prox off)     D - C  (H_prox on)
        interaction         (D - C) - (B - A)

    Exposure is heavily zero-inflated once H_prox is on, so each contrast carries a
    signed-rank test (paired_continuous falls back from t to Wilcoxon when the
    differences are not normal, and reports which it used), a percentile bootstrap
    CI on the mean difference, and the median [IQR] of the difference itself.
    Binary success gets the paired treatment it needs -- exact McNemar -- rather
    than a comparison of two independent percentages.

    p-values within each outcome family are Holm-adjusted; the manuscript quotes the
    adjusted values.
    """
    rows, ver = load("route_yield_factorial.csv")
    out: Dict[str, Any] = {"dataset": "route_yield_factorial.csv", "status": ver}

    if not rows:
        return out

    by_cell: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_cell[r["cell"]][int(r["trial_id"])] = r

    cells = ["A_prox_off_yield_off", "B_prox_off_yield_on",
             "C_prox_on_yield_off", "D_prox_on_yield_on"]
    if not any(c in by_cell for c in cells):
        # Datasets written before the factor was cleaned up used these names.
        cells = ["A_frozen_noyield", "B_frozen_yield",
                 "C_social_noyield", "D_social_yield"]

    def exposure(row: Dict[str, str]) -> float:
        """Person-seconds. Older datasets carry only person-ticks at dt = 0.05 s."""
        if "intimate_exposure_person_s" in row:
            return fnum(row, "intimate_exposure_person_s")
        return fnum(row, "intimate_exposure_person_ticks") * 0.05

    def exposure_rate(row: Dict[str, str]) -> float:
        """
        Person-seconds of intimate-space exposure per MINUTE of mission.

        Total exposure confounds two different things: how intrusive the robot is
        while it is present, and how long it is present. Cells differ enormously on
        the second -- a cell whose missions mostly time out spends nine times longer
        in the environment than one that completes, and accumulates person-seconds
        for the whole of it. Reading the total difference as a per-moment social
        effect would therefore overstate it.

        Normalising by mission duration separates the two. The total remains the
        operationally meaningful quantity (a pedestrian crowded for 50 person-seconds
        does not care that the robot was also failing), so both are reported.
        """
        m = fnum(row, "makespan_s")
        return exposure(row) / m * 60.0 if m and m > 0 else 0.0

    groups: Dict[str, Dict[str, Any]] = {}
    for c in cells:
        cell_rows = list(by_cell.get(c, {}).values())
        if not cell_rows:
            continue
        n = len(cell_rows)
        succ = [int(r["success"]) for r in cell_rows]
        lo, hi = wilson_interval(sum(succ), n)
        groups[c] = {
            "n": n,
            "success_rate": sum(succ) / n * 100.0,
            # Reported as a percentage, matching success_rate. Its absence is what
            # previously made the generated table print a fabricated [0.0, 0.0].
            "success_ci95": [lo * 100.0, hi * 100.0],
            "makespan": describe([fnum(r, "makespan_s") for r in cell_rows]),
            "exposure_person_s": describe([exposure(r) for r in cell_rows]),
            "exposure_rate_per_min": describe([exposure_rate(r) for r in cell_rows]),
            "encounters": describe([fnum(r, "intimate_encounters") for r in cell_rows]),
        }
    out["groups"] = groups

    if len(cells) != 4 or not all(c in by_cell for c in cells):
        return out
    trials = sorted(by_cell[cells[0]].keys())
    if not all(all(t in by_cell[c] for t in trials) for c in cells):
        # Unbalanced cells would silently break the pairing; say so rather than
        # reporting a contrast computed over a different set of trials per cell.
        out["contrasts_note"] = "cells are not trial-balanced; contrasts omitted"
        return out

    A, B, C, D = (by_cell[c] for c in cells)

    def vec(cell: Dict[int, Dict[str, str]], f) -> List[float]:
        return [f(cell[t]) for t in trials]

    def contrast(diff: Sequence[float]) -> Dict[str, Any]:
        """A paired contrast tested against zero, with a CI and a distribution."""
        zeros = [0.0] * len(diff)
        stat = paired_continuous(list(diff), zeros)
        return {
            **describe(list(diff)),
            "test": stat.get("test"),
            "p": stat.get("p"),
            "cohens_dz": stat.get("cohens_dz"),
            "bootstrap": paired_bootstrap(list(diff), zeros),
        }

    contrasts: Dict[str, Any] = {}
    for outcome, getter in (("exposure", exposure),
                            ("exposure_rate", exposure_rate),
                            ("makespan", lambda r: fnum(r, "makespan_s"))):
        a, b, c, d = (vec(x, getter) for x in (A, B, C, D))
        defs = {
            f"routing_effect_yield_off_{outcome}": [ci - ai for ai, ci in zip(a, c)],
            f"routing_effect_yield_on_{outcome}":  [di - bi for bi, di in zip(b, d)],
            f"yielding_effect_prox_off_{outcome}": [bi - ai for ai, bi in zip(a, b)],
            f"yielding_effect_prox_on_{outcome}":  [di - ci for ci, di in zip(c, d)],
            f"interaction_{outcome}": [(di - ci) - (bi - ai)
                                       for ai, bi, ci, di in zip(a, b, c, d)],
        }
        computed = {k: contrast(v) for k, v in defs.items()}
        adj = holm({k: v["p"] for k, v in computed.items() if v.get("p") is not None})
        for k, v in computed.items():
            if k in adj:
                v["p_holm"] = adj[k]
        contrasts.update(computed)
    out["contrasts"] = contrasts

    # Paired binary success. The reviewer is right that comparing 100% against 12%
    # as two independent proportions ignores the seed pairing that the design went
    # to the trouble of establishing.
    succ_vec = {name: [int(cell[t]["success"]) for t in trials]
                for name, cell in zip("ABCD", (A, B, C, D))}
    pairs = {
        "routing_effect_yield_off_success": ("C", "A"),
        "routing_effect_yield_on_success":  ("D", "B"),
        "yielding_effect_prox_off_success": ("B", "A"),
        "yielding_effect_prox_on_success":  ("D", "C"),
    }
    succ_tests = {k: mcnemar(succ_vec[x], succ_vec[y]) for k, (x, y) in pairs.items()}
    adj = holm({k: v["p"] for k, v in succ_tests.items()})
    for k, v in succ_tests.items():
        v["p_holm"] = adj[k]
    out["success_contrasts"] = succ_tests

    return out


def analyse_degradation_paired() -> Dict[str, Any]:
    r"""
    Analyses Mechanism A under communication degradation by pairing Mesh ON and Mesh OFF
    trials at each channel condition (loss rate x latency).
    Calculates paired deltas:
      delta_lead_time_s = T_Mesh_ON - T_Mesh_OFF
      delta_backtrack_m = Backtrack_Mesh_OFF - Backtrack_Mesh_ON
      delta_makespan_s = Makespan_Mesh_OFF - Makespan_Mesh_ON
    along with 95% CIs and p-values.
    """
    rows, ver = load("mesh_degradation.csv")
    out: Dict[str, Any] = {"dataset": "mesh_degradation.csv", "status": ver}

    if not rows:
        return out

    by_chan: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        chan = r["channel"]
        tid = int(r["trial_id"])
        arm = str(r["mesh_enabled"])
        by_chan[chan][(tid, arm)] = r

    channels: Dict[str, Dict[str, Any]] = {}
    for chan, records in by_chan.items():
        trials = sorted(list(set(t for t, a in records.keys())))
        paired_records = []
        for t in trials:
            if (t, "1") in records and (t, "0") in records:
                paired_records.append((records[(t, "1")], records[(t, "0")]))

        if not paired_records:
            continue

        n_pairs = len(paired_records)
        delta_lead = [fnum(on, "anticipation_lead_time_s") - fnum(off, "anticipation_lead_time_s") for on, off in paired_records]
        delta_backtrack = [fnum(off, "backtrack_distance_m") - fnum(on, "backtrack_distance_m") for on, off in paired_records]
        delta_makespan = [fnum(off, "makespan_s") - fnum(on, "makespan_s") for on, off in paired_records]

        on_lead = [fnum(on, "anticipation_lead_time_s") for on, _ in paired_records]
        off_lead = [fnum(off, "anticipation_lead_time_s") for _, off in paired_records]

        channels[chan] = {
            "n_pairs": n_pairs,
            "loss_rate": float(paired_records[0][0].get("packet_loss_rate", 0)),
            "latency_s": float(paired_records[0][0].get("latency_s", 0)),
            "on_lead_time": describe(on_lead),
            "off_lead_time": describe(off_lead),
            "delta_lead_time": {**describe(delta_lead), "bootstrap": paired_bootstrap(delta_lead, [0.0]*len(delta_lead))},
            "delta_backtrack": {**describe(delta_backtrack), "bootstrap": paired_bootstrap(delta_backtrack, [0.0]*len(delta_backtrack))},
            "delta_makespan": {**describe(delta_makespan), "bootstrap": paired_bootstrap(delta_makespan, [0.0]*len(delta_makespan))},
        }

    out["channels"] = channels

    # ---------------------------------------------------------------------- #
    # Does the mesh advantage actually change with the channel?
    #
    # Reporting a per-channel effect answers "is the mesh still helping here",
    # but the manuscript wants to claim a tolerance threshold, which is a claim
    # about the DIFFERENCE BETWEEN channels. That needs its own contrast.
    #
    # The design supports it: trial t at 10% loss and trial t at 0% loss share a
    # seed and a pedestrian realisation, so the paired mesh effects can themselves
    # be differenced across channels, pairing on (trial, other factor). A negative
    # value means the mesh advantage shrank when the channel got worse.
    # ---------------------------------------------------------------------- #
    delta_at: Dict[Tuple[float, float, int], float] = {}
    for chan, records in by_chan.items():
        for (t, arm) in records:
            if arm != "1" or (t, "0") not in records:
                continue
            on, off = records[(t, "1")], records[(t, "0")]
            loss = float(on.get("packet_loss_rate", 0))
            lat = float(on.get("latency_s", 0))
            delta_at[(loss, lat, t)] = (fnum(on, "anticipation_lead_time_s")
                                        - fnum(off, "anticipation_lead_time_s"))

    losses = sorted({k[0] for k in delta_at})
    latencies = sorted({k[1] for k in delta_at})

    def across(levels: List[float], axis: int, ref: float) -> Dict[str, Any]:
        """Each level against the best channel, pairing on trial and the other axis."""
        res: Dict[str, Any] = {}
        for lv in levels:
            if lv == ref:
                continue
            keys = [k for k in delta_at
                    if k[axis] == lv and (k[:axis] + (ref,) + k[axis + 1:]) in delta_at]
            if len(keys) < 3:
                continue
            here = [delta_at[k] for k in sorted(keys)]
            there = [delta_at[k[:axis] + (ref,) + k[axis + 1:]] for k in sorted(keys)]
            diff = [x - y for x, y in zip(here, there)]
            stat = paired_continuous(diff, [0.0] * len(diff))
            res[f"{lv:g}"] = {
                **describe(diff),
                "test": stat.get("test"),
                "p": stat.get("p"),
                "bootstrap": paired_bootstrap(diff, [0.0] * len(diff)),
            }
        adj = holm({k: v["p"] for k, v in res.items() if v.get("p") is not None})
        for k, v in res.items():
            if k in adj:
                v["p_holm"] = adj[k]
        return res

    if losses:
        out["lead_time_vs_loss"] = {"reference": losses[0],
                                    "levels": across(losses, 0, losses[0])}
    if latencies:
        out["lead_time_vs_latency"] = {"reference": latencies[0],
                                       "levels": across(latencies, 1, latencies[0])}
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(d: Optional[Dict[str, Any]], places: int = 2) -> str:
    if not d or d.get("n", 0) == 0:
        return "n/a"
    return f"{d['mean']:.{places}f} ± {d['sd']:.{places}f}"


def build_report(results: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# D²RO — Statistical Analysis Report\n")
    L.append("Generated by `paper/scripts/analyze_results.py`, the single "
             "statistics pipeline. All values derive from the raw CSVs in "
             "`experiments/data/`; none are entered by hand.\n")

    L.append("## Dataset availability\n")
    L.append("| Dataset | Status |")
    L.append("|:--|:--|")
    for key, res in results.items():
        if isinstance(res, dict) and "status" in res:
            flag = "✅" if res["status"] == "ok" else "⚠️"
            L.append(f"| {key} | {flag} {res['status']} |")
    L.append("")

    bench = results.get("benchmark", {})
    if bench.get("groups"):
        L.append(f"## Comparative benchmark (N = {bench.get('n_trials', 0)} paired trials)\n")
        L.append("| Method | Success | Makespan, successful (s) | Deadlocks | "
                 "Intimate exposure (ticks), median [IQR] |")
        L.append("|:--|--:|--:|--:|--:|")
        for m, g in bench["groups"].items():
            ex = g["intimate_exposure"]
            lo, hi = g["success_ci95"]
            L.append(
                f"| {m} | {g['success_rate']:.1f}% [{lo:.1f}, {hi:.1f}] "
                f"| {fmt(g['makespan_successful'])} | {fmt(g['deadlocks'])} "
                f"| {ex.get('median', 0):.0f} [{ex.get('iqr', [0, 0])[0]:.0f}, "
                f"{ex.get('iqr', [0, 0])[1]:.0f}] |")
        L.append("")
        L.append("Intimate exposure is reported as median [IQR] because the "
                 "distribution is zero-inflated and strongly right-skewed; a "
                 "mean ± SD summary would misrepresent it.\n")

        if bench.get("comparisons"):
            L.append("### Paired comparisons against D²RO\n")
            L.append("| Baseline | Outcome | Test | Effect | p (Holm-adj.) |")
            L.append("|:--|:--|:--|--:|--:|")
            adj = bench.get("holm_adjusted_p", {})
            for m, c in bench["comparisons"].items():
                s = c["success"]
                L.append(f"| {m} | success | {s['test']} | "
                         f"{s['a_only']} vs {s['b_only']} discordant | "
                         f"{adj.get(f'{m}|success', s['p']):.3g} |")
                mk = c["makespan"]
                L.append(f"| {m} | makespan | {mk.get('test','-')} | "
                         f"Δ={mk.get('mean_difference',0):.2f}s, dz={mk.get('cohens_dz',0):.2f} | "
                         f"{adj.get(f'{m}|makespan', mk.get('p',1)):.3g} |")
                ie = c["intimate_exposure"]
                bs = ie.get("bootstrap", {})
                ci = bs.get("ci95_difference", [0, 0])
                L.append(f"| {m} | intimate exposure | {ie.get('test','-')} | "
                         f"Δ={bs.get('mean_difference',0):.1f} "
                         f"[{ci[0]:.1f}, {ci[1]:.1f}] | "
                         f"{adj.get(f'{m}|intimate', ie.get('p',1)):.3g} |")
            L.append("")
            L.append("Each p-value belongs to one specific outcome. A single "
                     "significance statement is never attached to several "
                     "unrelated metrics at once.\n")

    for key, title in (("ablation", "Component ablation"),
                       ("cross_domain", "Cross-domain generalisation"),
                       ("crowd_density", "Crowd-density scalability"),
                       ("fleet_size", "Fleet-size scalability")):
        res = results.get(key, {})
        if not res.get("groups"):
            L.append(f"## {title}\n\n_Dataset {res.get('status', 'missing')} — "
                     f"not analysed._\n")
            continue
        L.append(f"## {title}\n")
        first = next(iter(res["groups"].values()))
        cols = [c for c in first if c not in ("n", "success_ci95")]
        L.append("| Group | n | " + " | ".join(cols) + " |")
        L.append("|:--|--:|" + "--:|" * len(cols))
        for name, g in res["groups"].items():
            cells = []
            for c in cols:
                v = g.get(c)
                cells.append(f"{v:.1f}%" if c == "success_rate" and isinstance(v, float)
                             else fmt(v) if isinstance(v, dict) else str(v))
            L.append(f"| {name} | {g.get('n','-')} | " + " | ".join(cells) + " |")
        L.append("")

    for key, title in (("mesh_anticipation", "Mechanism A — V2V mesh anticipation"),
                       ("corridor_lock", "Mechanism B — corridor mutex lock")):
        res = results.get(key, {})
        if not res.get("conditions"):
            L.append(f"## {title}\n\n_Dataset {res.get('status','missing')} — "
                     f"not analysed._\n")
            continue
        L.append(f"## {title} (N = {res['n_pairs']} paired trials)\n")
        on, off = res["conditions"]["on"], res["conditions"]["off"]
        keys = [k for k in on if k not in ("success_rate", "success_ci95")]
        L.append("| Metric | ON | OFF | Test | p (Holm-adj.) |")
        L.append("|:--|--:|--:|:--|--:|")
        adj = res.get("holm_adjusted_p", {})
        for k in keys:
            c = res.get("comparisons", {}).get(k, {})
            L.append(f"| {k} | {fmt(on.get(k))} | {fmt(off.get(k))} | "
                     f"{c.get('test','-')} | {adj.get(k, c.get('p', float('nan'))):.3g} |")
        if "success_rate" in on:
            sc = res.get("comparisons", {}).get("success", {})
            L.append(f"| success rate | {on['success_rate']:.1f}% | "
                     f"{off['success_rate']:.1f}% | {sc.get('test','-')} | "
                     f"{adj.get('success', sc.get('p', float('nan'))):.3g} |")
        L.append("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    import sys
    # --allow-unverified lets a dataset produced by superseded code be analysed so
    # that provisional artefacts can be inspected. The status string is preserved
    # verbatim, so every downstream table and figure stays labelled PROVISIONAL.
    allow = "--allow-unverified" in sys.argv
    global _ALLOW_UNVERIFIED
    _ALLOW_UNVERIFIED = allow

    results: Dict[str, Any] = {
        "benchmark": analyse_benchmark(),
        "ablation": analyse_grouped(
            "ablation_study.csv", "configuration",
            {"makespan": "travel_time_s", "discomfort": "discomfort_integral",
             "deadlocks": "deadlocks",
             "shelf_contact_ticks": "shelf_contact_ticks",
             "shelf_contact_events": "shelf_contact_events"}),
        "cross_domain": analyse_grouped(
            "cross_domain_benchmark.csv", "environment",
            {"makespan": "makespan_s", "transit": "mean_transit_time_s",
             "intimate_exposure": "proxemic_violations", "replans": "dynamic_replans"}),
        "crowd_density": analyse_grouped(
            "scalability_crowd_density.csv", "crowd_density_humans",
            {"makespan": "makespan_s", "replan_latency_ms": "mean_replan_latency_ms",
             "corridor_mutex_wait_s": "corridor_mutex_wait_s",
             "mesh_packets": "v2v_mesh_packets"}),
        "fleet_size": analyse_grouped(
            "scalability_fleet_size.csv", "fleet_size_carts",
            {"makespan": "makespan_s", "replan_latency_ms": "mean_replan_latency_ms",
             "corridor_mutex_wait_s": "corridor_mutex_wait_s",
             "mesh_packets": "v2v_mesh_packets"}),
        # Weight sensitivity: one group per configuration. Grouping by `config`
        # keeps each perturbed weight identifiable; the nominal row serves as the
        # x1.0 point shared by all five curves.
        "weight_sensitivity": analyse_grouped(
            "weight_sensitivity.csv", "config",
            {"makespan": "makespan_s",
             "intimate_exposure": "intimate_exposure_ticks",
             "discomfort": "discomfort_integral", "replans": "replans"}),
        # Communication robustness: grouped by the channel condition itself.
        "comm_robustness": analyse_grouped(
            "comm_robustness.csv", "channel",
            {"makespan": "makespan_s",
             "intimate_exposure": "intimate_exposure_ticks",
             "mesh_packets": "mesh_packets", "replans": "replans"}),
        # Route x Yield factorial: one group per cell. The cells are compared in
        # the manuscript rather than here, because the quantities of interest are
        # differences and an interaction, not per-cell descriptives alone.
        "route_yield_factorial": analyse_factorial(),
        # Mechanism A repeated per channel condition. Grouped by channel AND arm so
        # the anticipation advantage can be tracked as the link degrades.
        "mesh_degradation": analyse_degradation_paired(),
        "mesh_anticipation": analyse_paired_mechanism(
            "mesh_anticipation_experiment.csv", "mesh_enabled", "1",
            ["anticipation_lead_time_s", "backtrack_distance_m",
             "path_length_m", "makespan_s"]),
        "corridor_lock": analyse_paired_mechanism(
            "corridor_lock_experiment.csv", "lock_enabled", "1",
            # total_lock_wait_s supersedes lock_wait_s: the latter is a per-episode
            # timer that corridor release resets, so an end-of-run read of it is
            # always ~0 regardless of how long the agent actually waited. Both are
            # reported so the correction is visible rather than silent.
            ["head_on_events", "deadlocks", "lock_wait_s", "total_lock_wait_s",
             "corridor_time_s", "makespan_s",
             # Diversion evidence: off-corridor vertices occupied, and the number of
             # route reconsiderations that took the agent there.
             "nodes_outside_corridor", "replans"]),
    }

    json_path = os.path.join(DATA_DIR, "analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    md_path = os.path.join(DATA_DIR, "analysis_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(results))

    print("Statistical analysis complete.")
    for key, res in results.items():
        if isinstance(res, dict) and "status" in res:
            print(f"  {key:20s} {res['status']}")
    print(f"\n  -> {json_path}")
    print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
