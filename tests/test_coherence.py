"""
Coherence property tests.

Locks in the claim demonstrated by tools/coherence_demo.py: a single
`synthesize_for_expression` call yields ONE object from which both the spoken
response and the action decision are read, so speech and action cannot diverge.

Runs against the real orchestrator, deterministic/offline (no API key).
"""

import os
import sys
import io
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from cognitive_orchestrator import (  # noqa: E402
    CognitiveOrchestrator, AttentionPriority, CognitiveLoad,
)

CAPS = {"cloud_search": True, "cloud_weather": True, "cloud_news": True,
        "cloud_stocks": True, "local_node": False}


def _synth(msg, emotion, priority=AttentionPriority.MEDIUM, load=None):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = CognitiveOrchestrator()
        return orch.synthesize_for_expression(
            priority=priority,
            cognitive_load=load or CognitiveLoad(0.3, 0.2, 0.2, emotion.get("intensity", 0.2)),
            all_cognitive_outputs={
                "user_input": msg,
                "emotional_state": emotion,
                "available_capabilities": CAPS,
                "has_local_node": CAPS["local_node"],
                "user_relationship": {"trust_level": 0.6, "interaction_count": 12},
                "conversation_history": [],
            },
        )


def test_single_object_carries_both_projections():
    ctx = _synth("what's the weather in Tokyo right now?",
                 {"valence": 0.1, "arousal": 0.4, "intensity": 0.2})
    # Speech projection exists...
    assert ctx.response_mode is not None
    assert len(ctx.to_expression_prompt()) > 0
    # ...and the action projection exists on the SAME object.
    assert ctx.action_request is not None
    assert ctx.action_request.get("action_type") == "check_weather"


def test_no_action_claim_without_request():
    # The structural invariant that prevents "say yes, then don't do it".
    for msg, emo in [
        ("i've been feeling really lonely lately", {"valence": -0.6, "arousal": 0.3, "intensity": 0.7}),
        ("what's the weather in Tokyo?", {"valence": 0.1, "arousal": 0.4, "intensity": 0.2}),
        ("play some jazz for me", {"valence": 0.4, "arousal": 0.5, "intensity": 0.3}),
    ]:
        ctx = _synth(msg, emo)
        if ctx.should_take_action:
            assert ctx.action_request is not None, f"claims action but no request: {msg!r}"


def test_pure_conversation_has_no_action():
    ctx = _synth("i've been feeling really lonely lately",
                 {"valence": -0.6, "arousal": 0.3, "intensity": 0.7},
                 priority=AttentionPriority.CRITICAL)
    assert ctx.should_take_action is False
    assert ctx.action_request is None


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
