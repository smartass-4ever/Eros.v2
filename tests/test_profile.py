"""
Tests for the runtime feature profile (core/eros_profile.py).

Verify that 'sharp' disables exactly the intended background modules and keeps
every load-bearing subsystem, and that 'full' (the default) enables everything.
The module reads EROS_PROFILE once at import, so we reload it per case.
"""

import os
import sys
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

LOAD_BEARING = [
    "perception", "emotion", "game_theory", "orchestration",
    "memory", "safety", "telemetry", "curiosity", "personality", "beliefs",
]
SHARP_OFF = [
    "consciousness_growth", "rem_engine", "imagination",
    "self_reflection", "unified_self_systems", "proactive_scheduler",
]


def _load(profile_value):
    if profile_value is None:
        os.environ.pop("EROS_PROFILE", None)
    else:
        os.environ["EROS_PROFILE"] = profile_value
    import eros_profile
    return importlib.reload(eros_profile)


def test_default_is_full_and_enables_everything():
    p = _load(None)
    assert p.sharp() is False
    for s in LOAD_BEARING + SHARP_OFF:
        assert p.enabled(s) is True, f"{s} should be enabled under full"


def test_sharp_disables_only_background_modules():
    p = _load("sharp")
    assert p.sharp() is True
    for s in SHARP_OFF:
        assert p.enabled(s) is False, f"{s} should be OFF under sharp"


def test_sharp_keeps_all_load_bearing():
    p = _load("sharp")
    for s in LOAD_BEARING:
        assert p.enabled(s) is True, f"{s} must stay ON under sharp"


def test_unknown_profile_behaves_like_full():
    p = _load("banana")
    for s in SHARP_OFF:
        assert p.enabled(s) is True  # only 'sharp' disables anything


def test_case_and_whitespace_insensitive():
    p = _load("  SHARP  ")
    assert p.sharp() is True


def teardown_module(module):
    os.environ.pop("EROS_PROFILE", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
    os.environ.pop("EROS_PROFILE", None)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
