"""
Shared safety- and social-metric accounting.

Every planner (D2RO and all baselines) must measure social compliance with an
IDENTICAL definition and an IDENTICAL threshold, otherwise the comparison
reports differences in instrumentation rather than differences in behaviour.

Two distinct quantities are tracked, and they must never be conflated:

  * ``intimate_encounters``  -- discrete boundary crossings inward. Answers
    "how many times did the robot intrude on someone's personal space?"
  * ``intimate_exposure_s``  -- accumulated seconds spent inside the boundary.
    Answers "for how long was the robot too close?"

A per-tick counter reported under an event name is what previously produced
uninterpretable figures such as "2094 deadlocks per trial".
"""

from __future__ import annotations
import math
from typing import Any, Iterable

from .units import INTIMATE_RADIUS_PX


def init_social_metrics(agent: Any) -> None:
    """Installs the shared metric fields on any planner instance."""
    agent.proxemic_violations = 0      # per-tick exposure count (legacy name)
    agent.intimate_encounters = 0      # discrete inward boundary crossings
    agent.intimate_exposure_s = 0.0    # cumulative seconds inside the boundary
    agent.stalled_ticks = 0            # control cycles with no viable motion
    agent._humans_inside = set()


def update_social_metrics(agent: Any, humans: Iterable[Any], dt: float) -> int:
    """
    Updates the shared social-compliance metrics for one control tick.
    Returns the number of humans currently inside the intimate boundary.
    """
    if not humans:
        agent._humans_inside = set()
        return 0

    inside_now = set()
    for h in humans:
        if not (hasattr(h, "x") and hasattr(h, "y")):
            continue
        if math.hypot(agent.x - h.x, agent.y - h.y) < INTIMATE_RADIUS_PX:
            inside_now.add(getattr(h, "id", id(h)))
            agent.proxemic_violations += 1
            agent.intimate_exposure_s += dt

    previously = getattr(agent, "_humans_inside", set())
    agent.intimate_encounters += len(inside_now - previously)
    agent._humans_inside = inside_now
    return len(inside_now)
