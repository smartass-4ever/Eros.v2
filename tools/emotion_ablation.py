"""
emotion_ablation.py — is emotion actually a control signal, or decoration?

This runs a controlled perturbation on Eros's decision core. For each test
message we build the game-theory decision twice from identical inputs, changing
ONE thing: the emotional state.

  INTACT   — the message's real valence / arousal / intensity
  CLAMPED  — emotion forced to neutral (valence 0, arousal 0.5, intensity 0)

Everything else (memory relevance, curiosity gaps, belief triggers, trust,
playfulness) is held fixed. If emotion is a genuine control signal, clamping it
should measurably change which motive wins and how the agent would respond. If
emotion were a decorative label, nothing would change.

The decision core is deterministic without an API key (rule-based fallback), so
this experiment is fully reproducible offline. That is the point: perturb one
input, hold the rest, measure the effect on behavior.

Usage:
    python tools/emotion_ablation.py
"""

import os
import sys
import io
import contextlib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from game_theory_decision import GameTheoryDecisionEngine, build_context_signals  # noqa: E402

PLAYERS = ("memory", "curiosity", "warmth", "wit", "beliefs")

NEUTRAL = {"valence": 0.0, "arousal": 0.5, "intensity": 0.0}

# Each case: a message + its real emotional state + the (fixed) non-emotional context.
CASES = [
    {
        "msg": "i completely bombed my interview today, i feel like a failure",
        "emotion": {"valence": -0.8, "arousal": 0.4, "intensity": 0.8},
        "ctx": {},
    },
    {
        "msg": "lol okay that's actually hilarious, tell me another one",
        "emotion": {"valence": 0.7, "arousal": 0.7, "intensity": 0.6},
        "ctx": {"curiosity": {"gaps_detected": [{"t": "joke"}]}},
    },
    {
        "msg": "my mom is in the hospital and i don't know what to do",
        "emotion": {"valence": -0.9, "arousal": 0.6, "intensity": 0.9},
        "ctx": {},
    },
    {
        "msg": "what do you actually think about free will?",
        "emotion": {"valence": -0.1, "arousal": 0.5, "intensity": 0.3},
        "ctx": {"belief": {"conflicts": ["free_will"]}, "curiosity": {"gaps_detected": [{"t": "philosophy"}]}},
    },
    {
        "msg": "i've been feeling so alone lately, like nobody gets it",
        "emotion": {"valence": -0.6, "arousal": 0.3, "intensity": 0.7},
        "ctx": {},
    },
]


def decide_for(engine, msg, emotion, ctx):
    signals = build_context_signals(
        user_input=msg,
        emotional_state=emotion,
        relationship_data=ctx.get("relationship", {"trust_level": 0.5, "interaction_count": 5}),
        curiosity_data=ctx.get("curiosity", {}),
        memory_data=ctx.get("memory", {}),
        belief_data=ctx.get("belief", {}),
        crisis_detected=ctx.get("crisis", False),
    )
    # Silence the decision core's internal debug prints for a clean report.
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.decide(signals)


def fmt_scores(scores):
    d = scores.to_dict()
    return "  ".join(f"{p[:4]}={d.get(p,0):.2f}" for p in PLAYERS)


def main():
    engine = GameTheoryDecisionEngine()  # no api_key -> deterministic rule-based core

    print("=" * 82)
    print("  EMOTION ABLATION — is the emotional signal a control signal?")
    print("  (game-theory decision core; one input perturbed, all else held fixed)")
    print("=" * 82)

    changed = 0
    for i, case in enumerate(CASES, 1):
        intact = decide_for(engine, case["msg"], case["emotion"], case["ctx"])
        clamped = decide_for(engine, case["msg"], dict(NEUTRAL), case["ctx"])

        ip = intact.primary.value if intact.primary else "—"
        cp = clamped.primary.value if clamped.primary else "—"
        flipped = ip != cp
        changed += flipped

        print(f"\n[{i}] \"{case['msg']}\"")
        print(f"    emotion intact  (v={case['emotion']['valence']:+.2f} i={case['emotion']['intensity']:.2f}):"
              f"  PRIMARY={ip:<9} veto={[p.value for p in intact.vetoed] or '—'}")
        print(f"        scores: {fmt_scores(intact.player_scores)}")
        print(f"    emotion CLAMPED (v= 0.00 i=0.00):"
              f"                 PRIMARY={cp:<9} veto={[p.value for p in clamped.vetoed] or '—'}")
        print(f"        scores: {fmt_scores(clamped.player_scores)}")
        if flipped:
            print(f"    -> winning motive CHANGED  ({ip} -> {cp})")
        else:
            print(f"    -> winning motive unchanged ({ip}) — a non-emotional signal dominates here")

    print("\n" + "=" * 82)
    print(f"  RESULT: clamping emotion changed the winning motive on "
          f"{changed}/{len(CASES)} turns.")
    if changed:
        print("  => Emotion is a control signal: perturbing it re-routes the decision.")
    else:
        print("  => No behavioral change detected — emotion is NOT driving the decision here.")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
