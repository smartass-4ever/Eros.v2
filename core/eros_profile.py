"""
eros_profile.py — runtime feature profile (transparent, reversible).

EROS_PROFILE selects how much of the architecture runs:

  full  (default) — every subsystem active (original behavior).
  sharp           — disables non-load-bearing background / "inner life" modules
                    that do NOT affect the main response, so the live system is
                    tighter and higher signal-to-noise.

Nothing is deleted. `sharp` only skips subsystems at runtime; set
EROS_PROFILE=full (or unset it) to restore everything.

Under `sharp`, DISABLED (background/low-signal):
    - consciousness_growth   : the self-awareness/metacognition float counters
    - rem_engine             : REM subconscious consolidation pass
    - imagination            : counterfactual imagination injections
    - self_reflection        : context-triggered inner-voice composer
    - unified_self_systems   : the aggregated self/growth hub
    - proactive_scheduler    : time-of-day greeting/check-in loop

Under `sharp`, KEPT (load-bearing / the real interior):
    perception, emotion + hybrid appraisal, emotional clock, the game-theory
    decision core, orchestration + synthesis, memory, safety, interior
    telemetry, curiosity drive, personality, and constitutional beliefs.
"""

import os

PROFILE = os.environ.get("EROS_PROFILE", "full").strip().lower()

# Subsystems that the 'sharp' profile turns off.
_SHARP_DISABLED = frozenset({
    "consciousness_growth",
    "rem_engine",
    "imagination",
    "self_reflection",
    "unified_self_systems",
    "proactive_scheduler",
})


def sharp() -> bool:
    """True if the sharp profile is active."""
    return PROFILE == "sharp"


def enabled(subsystem: str) -> bool:
    """True if `subsystem` should run under the active profile."""
    if PROFILE == "sharp" and subsystem in _SHARP_DISABLED:
        return False
    return True
