"""
Regression test for the emotion-as-control-signal property.

Locks in the core claim demonstrated by tools/emotion_ablation.py: on an
emotionally charged turn, clamping the emotional signal to neutral must change
the decision core's chosen motive. If this ever stops being true, emotion has
silently stopped driving behavior.

Runs against the deterministic (no-API-key) decision core, so it is stable.
"""

import os
import sys
import io
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from game_theory_decision import GameTheoryDecisionEngine, build_context_signals  # noqa: E402

NEUTRAL = {"valence": 0.0, "arousal": 0.5, "intensity": 0.0}


def _decide(engine, msg, emotion):
    signals = build_context_signals(
        user_input=msg,
        emotional_state=emotion,
        relationship_data={"trust_level": 0.5, "interaction_count": 5},
        curiosity_data={}, memory_data={}, belief_data={},
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.decide(signals)


def test_clamping_emotion_changes_the_decision_on_a_charged_turn():
    engine = GameTheoryDecisionEngine()
    msg = "my mom is in the hospital and i don't know what to do"
    intact = _decide(engine, msg, {"valence": -0.9, "arousal": 0.6, "intensity": 0.9})
    clamped = _decide(engine, msg, dict(NEUTRAL))
    assert intact.primary is not None and clamped.primary is not None
    assert intact.primary != clamped.primary, (
        f"emotion did not change the decision: intact={intact.primary}, clamped={clamped.primary}"
    )
    # Specifically, distress should raise warmth as the winning motive.
    assert intact.primary.value == "warmth"


def test_warmth_score_tracks_emotional_intensity():
    engine = GameTheoryDecisionEngine()
    msg = "i feel completely overwhelmed"
    hi = _decide(engine, msg, {"valence": -0.8, "arousal": 0.5, "intensity": 0.9})
    lo = _decide(engine, msg, dict(NEUTRAL))
    assert hi.player_scores.warmth > lo.player_scores.warmth


def test_positive_emotion_also_shifts_scores():
    # The signal is directional, not just "distress": positive affect moves warmth too.
    engine = GameTheoryDecisionEngine()
    msg = "this is the best day ever"
    happy = _decide(engine, msg, {"valence": 0.8, "arousal": 0.7, "intensity": 0.7})
    flat = _decide(engine, msg, dict(NEUTRAL))
    assert happy.player_scores.warmth != flat.player_scores.warmth


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
