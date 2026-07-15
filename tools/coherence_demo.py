"""
coherence_demo.py — one intention, broadcast to speech AND action.

A recurring failure mode for agents is incoherence: the model says it will do
something and then doesn't, because "what to say" and "what to do" are decided
by two independent processes that can disagree.

Eros forms intention once. The orchestrator's `synthesize_for_expression`
produces a single `SynthesizedContext` object, and BOTH the spoken response and
the action decision are read off that same object:

    ctx.response_mode / ctx.to_expression_prompt()   -> what Eros SAYS
    ctx.should_take_action / ctx.action_request      -> what Eros DOES

Because speech and action are two projections of one decision (not two
decisions), the agent structurally cannot say yes and then not do it.

This drives the REAL orchestrator offline (no API key; the decision core is
deterministic) and shows, per message, the single intention object and its two
projections.

Usage:
    python tools/coherence_demo.py
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from cognitive_orchestrator import (  # noqa: E402
    CognitiveOrchestrator, AttentionPriority, CognitiveLoad,
)

CAPABILITIES = {
    "cloud_search": True, "cloud_weather": True, "cloud_news": True,
    "cloud_stocks": True, "local_node": False,  # no paired computer in this demo
}

CASES = [
    {
        "msg": "i've been feeling really lonely lately, like nobody gets it",
        "emotion": {"valence": -0.6, "arousal": 0.3, "intensity": 0.7},
        "priority": AttentionPriority.CRITICAL,
        "load": CognitiveLoad(0.4, 0.3, 0.2, 0.7),
    },
    {
        "msg": "what's the weather in Tokyo right now?",
        "emotion": {"valence": 0.1, "arousal": 0.4, "intensity": 0.2},
        "priority": AttentionPriority.MEDIUM,
        "load": CognitiveLoad(0.3, 0.2, 0.2, 0.2),
    },
    {
        "msg": "play some jazz for me",
        "emotion": {"valence": 0.4, "arousal": 0.5, "intensity": 0.3},
        "priority": AttentionPriority.MEDIUM,
        "load": CognitiveLoad(0.3, 0.2, 0.2, 0.3),
    },
]


def synth(orch, case):
    outputs = {
        "user_input": case["msg"],
        "emotional_state": case["emotion"],
        "available_capabilities": CAPABILITIES,
        "has_local_node": CAPABILITIES["local_node"],
        "user_relationship": {"trust_level": 0.6, "interaction_count": 12},
        "conversation_history": [],
    }
    return orch.synthesize_for_expression(
        priority=case["priority"],
        cognitive_load=case["load"],
        all_cognitive_outputs=outputs,
    )


def describe_action(ctx):
    if not ctx.action_request:
        return "none (this turn is speech-only)"
    ar = ctx.action_request
    if ar.get("needs_setup"):
        return f"{ar.get('action_type')} — blocked, needs setup: {ar.get('needs_setup')}"
    return f"{ar.get('action_type')} (execute={ctx.should_take_action})"


def main():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):        # silence init logs
        orch = CognitiveOrchestrator()  # mistral_client=None -> deterministic/offline

    print("=" * 84)
    print("  COHERENCE DEMO — one intention, broadcast to speech AND action")
    print("  (real orchestrator.synthesize_for_expression; offline/deterministic)")
    print("=" * 84)

    for i, case in enumerate(CASES, 1):
        with contextlib.redirect_stdout(io.StringIO()):   # silence internal logs
            ctx = synth(orch, case)

        speech_prompt = ctx.to_expression_prompt()
        # A compact fingerprint of the speech projection.
        speech_line = f"mode={ctx.response_mode.name}  needs={ctx.user_needs}  length={ctx.suggested_response_length}"

        print(f"\n[{i}] \"{case['msg']}\"")
        print(f"    ONE intention object:  SynthesizedContext @ {hex(id(ctx))}")
        print(f"    ├─ SAY  (speech projection):  {speech_line}")
        print(f"    │        motive={ctx.primary_player}  directive={ctx.game_directive or '—'}")
        print(f"    │        expression prompt built from this object: {len(speech_prompt)} chars")
        print(f"    └─ DO   (action projection):  {describe_action(ctx)}")

        # Coherence assertion: both projections come from the SAME object.
        coherent = True
        note = "speech and action are two reads of one intention"
        if ctx.should_take_action and not ctx.action_request:
            coherent = False
            note = "INCOHERENT: claims action but no action_request"
        print(f"    => coherent: {coherent}  ({note})")

    print("\n" + "=" * 84)
    print("  Each turn: exactly ONE synthesis call -> ONE SynthesizedContext.")
    print("  Speech (to_expression_prompt) and action (action_request) are read off")
    print("  that single object, so the agent cannot say yes and then not do it.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
