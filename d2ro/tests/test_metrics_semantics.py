"""
Tests for metric semantics: discrete events must not be conflated with per-tick
exposure, and every planner must be scored against an identical threshold.
"""

import unittest

from d2ro.core.metrics import init_social_metrics, update_social_metrics
from d2ro.core.units import INTIMATE_RADIUS_PX, PHYSICS_DT
from d2ro.core.human import Human


class _Robot:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y
        init_social_metrics(self)


class TestEventVsExposure(unittest.TestCase):
    def test_sustained_proximity_is_one_encounter_many_ticks(self):
        """Standing next to a person for 100 ticks is ONE encounter, not 100."""
        r = _Robot()
        h = Human(id=7, x=INTIMATE_RADIUS_PX * 0.5, y=0.0)
        for _ in range(100):
            update_social_metrics(r, [h], PHYSICS_DT)

        self.assertEqual(r.intimate_encounters, 1,
                         "sustained proximity must count as a single encounter")
        self.assertEqual(r.proxemic_violations, 100,
                         "exposure ticks must still accumulate every tick")
        self.assertAlmostEqual(r.intimate_exposure_s, 100 * PHYSICS_DT, places=6)

    def test_reentry_counts_as_new_encounter(self):
        r = _Robot()
        near = Human(id=1, x=INTIMATE_RADIUS_PX * 0.5, y=0.0)
        far = Human(id=1, x=INTIMATE_RADIUS_PX * 10.0, y=0.0)

        update_social_metrics(r, [near], PHYSICS_DT)   # enter
        update_social_metrics(r, [far], PHYSICS_DT)    # leave
        update_social_metrics(r, [near], PHYSICS_DT)   # re-enter

        self.assertEqual(r.intimate_encounters, 2)

    def test_distinct_people_count_separately(self):
        r = _Robot()
        a = Human(id=1, x=INTIMATE_RADIUS_PX * 0.4, y=0.0)
        b = Human(id=2, x=0.0, y=INTIMATE_RADIUS_PX * 0.4)
        update_social_metrics(r, [a, b], PHYSICS_DT)
        self.assertEqual(r.intimate_encounters, 2)

    def test_outside_boundary_is_not_counted(self):
        r = _Robot()
        h = Human(id=1, x=INTIMATE_RADIUS_PX * 1.5, y=0.0)
        update_social_metrics(r, [h], PHYSICS_DT)
        self.assertEqual(r.intimate_encounters, 0)
        self.assertEqual(r.proxemic_violations, 0)
        self.assertEqual(r.intimate_exposure_s, 0.0)


class TestThresholdConsistency(unittest.TestCase):
    def test_all_planners_share_one_threshold(self):
        """
        Regression guard: the previous revision measured D2RO at 25-26 px and the
        baselines at 26.67 px, so the social comparison was against different
        boundaries. No planner may hard-code its own value.
        """
        import inspect
        from d2ro.baselines import (StaticAStarAgent, ArtificialPotentialFieldAgent,
                                    ORCAAgent, DecentralizedLocalMAPFAgent)
        from d2ro.core import agent as agent_mod

        for mod in (StaticAStarAgent, ArtificialPotentialFieldAgent,
                    ORCAAgent, DecentralizedLocalMAPFAgent, agent_mod.TrolleyAgent):
            src = inspect.getsource(inspect.getmodule(mod))
            self.assertIn("update_social_metrics(self", src,
                          f"{mod.__name__} must use the shared social accounting")
            # Only metrics.py may increment the exposure counter; a planner doing its
            # own counting is exactly how the thresholds drifted apart before.
            self.assertNotIn("self.proxemic_violations +=", src,
                             f"{mod.__name__} must not count violations itself")

    def test_threshold_matches_documented_value(self):
        self.assertAlmostEqual(INTIMATE_RADIUS_PX * 0.03, 0.80, places=6)


if __name__ == "__main__":
    unittest.main()
