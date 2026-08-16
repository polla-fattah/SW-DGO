"""
Bundle the d2ro Python sources into docs/python_bundle.js for Pyodide (WASM).

The browser demo runs the *actual* simulation code, not a re-implementation, which
is the point of it -- but it runs a SNAPSHOT. Nothing about editing `d2ro/` updates
`docs/python_bundle.js`, so the demo can silently keep executing code that no longer
matches the repository, and a visitor comparing the demo against the paper would be
comparing against a version that no longer exists.

The bundle therefore carries a fingerprint of the sources it was built from, and
`scripts/check_web_bundle.py` recomputes that fingerprint from the working tree and
fails when the two differ. Running it in CI is what makes this self-correcting: the
answer to "how do I remember to rebuild the bundle every time" is that you do not
have to, because the build breaks until you do.

Usage:
    python scripts/build_web_bundle.py          # rebuild the bundle
    python scripts/check_web_bundle.py          # is the committed bundle current?
"""
import hashlib
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
TARGET_JS = os.path.join(DOCS_DIR, "python_bundle.js")

FILES_TO_BUNDLE = [
    "d2ro/__init__.py",
    "d2ro/core/__init__.py",
    "d2ro/core/units.py",
    "d2ro/core/graph.py",
    "d2ro/core/grid_map.py",
    "d2ro/core/dstar_lite.py",
    "d2ro/core/mesh_network.py",
    "d2ro/core/human.py",
    "d2ro/core/metrics.py",
    "d2ro/core/agent.py",
    "d2ro/environments/__init__.py",
    "d2ro/environments/supermarket.py",
    "d2ro/environments/airport.py",
    "d2ro/environments/hospital.py",
]


def read_sources() -> dict:
    """The bundled files, keyed by POSIX-style repo-relative path."""
    out = {}
    for rel_path in FILES_TO_BUNDLE:
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(full_path):
            raise SystemExit(
                f"ERROR: {rel_path} is listed in FILES_TO_BUNDLE but does not exist. "
                f"The demo would silently ship without it.")
        with open(full_path, "r", encoding="utf-8") as f:
            out[rel_path.replace("\\", "/")] = f.read()
    return out


def fingerprint(sources: dict) -> str:
    """
    SHA-256 over the bundled sources, path-then-content in sorted path order.

    Deliberately the same construction as the dataset provenance fingerprint in
    d2ro/sim/run_experiments.py: paths sorted globally so filesystem traversal
    order cannot matter, and CRLF normalised to LF so a Windows checkout and a
    Linux one agree. Both platform dependencies were real bugs there before they
    were fixed, and there is no reason to repeat them here.
    """
    h = hashlib.sha256()
    for rel in sorted(sources):
        h.update(rel.encode("utf-8"))
        h.update(sources[rel].replace("\r\n", "\n").encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> None:
    sources = read_sources()
    for rel, content in sources.items():
        print(f"Bundled {rel} ({len(content)} bytes)")

    fp = fingerprint(sources)
    js_content = (
        "// Auto-generated SW-DGO Python source bundle for Pyodide (WASM).\n"
        "// Do not edit by hand: run `python scripts/build_web_bundle.py`.\n"
        f"// Source fingerprint: {fp}\n"
        f"window.D2RO_BUNDLE_FINGERPRINT = {json.dumps(fp)};\n"
        f"window.D2RO_PYTHON_FILES = {json.dumps(sources, indent=2)};\n"
    )
    with open(TARGET_JS, "w", encoding="utf-8", newline="") as f:
        f.write(js_content)
    print(f"\nGenerated {TARGET_JS} ({len(js_content)} bytes)")
    print(f"Source fingerprint: {fp}")


if __name__ == "__main__":
    main()
