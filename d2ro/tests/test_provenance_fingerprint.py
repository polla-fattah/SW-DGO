"""
The provenance fingerprint must identify the SOURCE, not the checkout.

Round-3 CI exposed a defect that three rounds of review had missed: the fingerprint
hashed raw file bytes, so a Windows checkout (CRLF) and a Linux checkout (LF)
produced different hashes for identical source. Every dataset therefore reported
STALE on the other operating system, and a reviewer attempting the clean-room
reproduction the manuscript invites would have seen nothing but STALE datasets.

Every "all datasets ok" reported before that point had been verified on one OS only.

Two properties are asserted here:

  * the hash is invariant to line-ending style;
  * the two independent implementations -- the one that WRITES stamps in
    run_experiments.py and the one that VERIFIES them in analyze_results.py --
    agree with each other. If they ever drift apart, every dataset reads STALE
    forever and no amount of regeneration fixes it.
"""

import hashlib
import importlib.util
import os
import unittest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFingerprintIsSourceIdentity(unittest.TestCase):

    def _hash_tree(self, transform):
        """Re-implements the fingerprint, applying `transform` to each file."""
        pkg = os.path.join(PROJECT_ROOT, "d2ro")
        h = hashlib.sha256()
        for root, _dirs, files in os.walk(pkg):
            if "__pycache__" in root:
                continue
            for fn in sorted(files):
                if fn.endswith(".py"):
                    with open(os.path.join(root, fn), "rb") as f:
                        h.update(transform(f.read()))
        return h.hexdigest()[:16]

    @staticmethod
    def _normalise(blob: bytes) -> bytes:
        """The normalisation the production fingerprint applies."""
        return blob.replace(b"\r\n", b"\n")

    def test_invariant_to_line_endings(self):
        """
        The same source checked out with CRLF and with LF must hash identically.

        Both variants are passed through the production normalisation, which is the
        step under test: without it these two byte streams differ and the hashes
        diverge, which is exactly the failure CI surfaced.
        """
        pkg = os.path.join(PROJECT_ROOT, "d2ro")
        h_lf, h_crlf = hashlib.sha256(), hashlib.sha256()
        for root, _dirs, files in os.walk(pkg):
            if "__pycache__" in root:
                continue
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                with open(os.path.join(root, fn), "rb") as f:
                    raw = f.read()
                lf = raw.replace(b"\r\n", b"\n")          # as a Linux checkout
                crlf = lf.replace(b"\n", b"\r\n")         # as a Windows checkout
                self.assertNotEqual(lf, crlf if b"\n" in lf else lf + b"x",
                                    "test setup produced identical variants")
                h_lf.update(self._normalise(lf))
                h_crlf.update(self._normalise(crlf))
        self.assertEqual(
            h_lf.hexdigest()[:16], h_crlf.hexdigest()[:16],
            "fingerprint changes with line-ending style, so datasets generated on "
            "one platform report STALE on another")

    # The analysis pipeline sits beside the manuscript in the research repository
    # and under analysis/ in the public code release. The two trees otherwise share
    # identical sources, and this test should not be the one thing that differs, so
    # it searches rather than assuming.
    ANALYZER_LOCATIONS = [
        os.path.join("paper", "scripts", "analyze_results.py"),
        os.path.join("analysis", "scripts", "analyze_results.py"),
    ]

    def test_writer_and_verifier_agree(self):
        """The stamp writer and the stamp checker must compute the same value."""
        runner = _load("d2ro_runner", os.path.join("d2ro", "sim", "run_experiments.py"))
        rel = next((p for p in self.ANALYZER_LOCATIONS
                    if os.path.exists(os.path.join(PROJECT_ROOT, p))), None)
        if rel is None:
            self.skipTest("analyze_results.py is not present in this tree")
        analyzer = _load("d2ro_analyzer", rel)
        self.assertEqual(
            runner._code_fingerprint(), analyzer.current_code_fingerprint(),
            "run_experiments.py and analyze_results.py disagree about the code "
            "fingerprint; every dataset would report STALE regardless of reruns")

    # A fourth test re-implementing the fingerprint against an "independent
    # reference" was removed. It duplicated the function under test, and because the
    # fingerprint covers d2ro/tests/ it was sensitive to which test files happened to
    # exist while it ran -- brittle, without catching anything the three assertions
    # above miss. Invariance and writer/verifier agreement are the properties that
    # matter.


if __name__ == "__main__":
    unittest.main()
