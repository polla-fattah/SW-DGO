"""
Is the committed web bundle built from the current Python sources?

The browser demo executes a snapshot of `d2ro/` embedded in
`docs/python_bundle.js`. Editing the package does not update that snapshot, so
without a check the demo drifts away from the repository silently -- and a visitor
who compares the demo's behaviour against the manuscript would be comparing against
code that no longer exists.

This recomputes the fingerprint from the working tree and compares it with the one
recorded in the bundle. Exit status is non-zero on drift, so CI can hold the line
instead of a human having to remember.

Usage:
    python scripts/check_web_bundle.py
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_web_bundle import TARGET_JS, fingerprint, read_sources  # noqa: E402


def recorded_fingerprint(path: str) -> str:
    """The fingerprint the bundle claims, or '' if it predates the check."""
    if not os.path.exists(path):
        return ""
    text = io.open(path, encoding="utf-8").read(4096)
    m = re.search(r"window\.D2RO_BUNDLE_FINGERPRINT\s*=\s*(\"[0-9a-f]+\")", text)
    return json.loads(m.group(1)) if m else ""


def main() -> int:
    if not os.path.exists(TARGET_JS):
        print("FAIL  docs/python_bundle.js is missing entirely.")
        print("      Run: python scripts/build_web_bundle.py")
        return 1

    current = fingerprint(read_sources())
    recorded = recorded_fingerprint(TARGET_JS)

    if not recorded:
        print("FAIL  the bundle carries no fingerprint, so it predates this check "
              "and its freshness cannot be established.")
        print("      Run: python scripts/build_web_bundle.py")
        return 1

    if recorded != current:
        print(f"FAIL  the web demo is running stale code.")
        print(f"      bundle was built from : {recorded}")
        print(f"      d2ro/ currently hashes: {current}")
        print("      Run: python scripts/build_web_bundle.py")
        return 1

    print(f"OK    web bundle matches d2ro/ ({current})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
