"""
Tests for genuine drive-triggered proactivity.

Verify that initiation is driven by INTERNAL pressure crossing a threshold (not
a clock): an unresolved lingering curiosity arc, emotional residue, or silence
each build the right drive and fire the right impulse; fleeting/resolved arcs
don't; and the cooldown, quiet-hours filter, and contact-discharge all work.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from drive_initiation import DriveInitiationEngine  # noqa: E402


def _arc(drive, satisfaction, turns, target="the thing"):
    return {"drive": drive, "satisfaction": satisfaction,
            "turns_active": turns, "target": target}


def test_unresolved_lingering_curiosity_fires():
    eng = DriveInitiationEngine(cooldown=0)
    eng.feed_curiosity([_arc(1.4, 0.1, turns=4, target="your new job")])
    eng.tick(now=1000)
    imp = eng.poll(now=1000)
    assert imp is not None
    assert imp.drive_type == "curiosity"
    assert imp.target == "your new job"


def test_fleeting_or_resolved_curiosity_does_not_fire():
    eng = DriveInitiationEngine(cooldown=0)
    # fleeting (only 1 turn) and resolved (high satisfaction) — neither should count
    eng.feed_curiosity([_arc(1.5, 0.1, turns=1), _arc(1.5, 0.9, turns=5)])
    eng.tick(now=1000)
    assert eng.poll(now=1000) is None


def test_emotional_residue_builds_care_and_fires():
    eng = DriveInitiationEngine(cooldown=0)
    eng.feed_emotion_residue(valence=-0.9, intensity=1.0, target="the fight with your friend")
    eng.tick(now=1000)
    imp = eng.poll(now=1000)
    assert imp is not None and imp.drive_type == "care"
    assert "friend" in (imp.target or "")


def test_positive_emotion_does_not_build_care():
    eng = DriveInitiationEngine(cooldown=0)
    eng.feed_emotion_residue(valence=0.8, intensity=1.0)
    eng.tick(now=1000)
    assert eng.poll(now=1000) is None


def test_silence_builds_connection_and_fires():
    eng = DriveInitiationEngine(cooldown=0, connection_rate_per_hour=0.25)
    start = 10_000.0
    eng.note_contact(now=start)
    # 5 hours of silence -> connection ~1.25 > threshold 1.0
    later = start + 5 * 3600
    eng.tick(now=later)
    imp = eng.poll(now=later)
    assert imp is not None and imp.drive_type == "connection"


def test_note_contact_discharges_connection():
    eng = DriveInitiationEngine(cooldown=0, connection_rate_per_hour=0.25)
    start = 10_000.0
    eng.note_contact(now=start)
    later = start + 5 * 3600
    eng.tick(now=later)
    eng.note_contact(now=later)          # user came back
    eng.tick(now=later + 1)
    assert eng.poll(now=later + 1) is None


def test_cooldown_prevents_rapid_refire():
    eng = DriveInitiationEngine(cooldown=1200)  # 20 min
    eng.feed_curiosity([_arc(1.4, 0.1, turns=4)])
    eng.tick(now=1000)
    first = eng.poll(now=1000)
    assert first is not None
    # immediately after, even with pressure, cooldown blocks
    eng.feed_curiosity([_arc(1.4, 0.1, turns=4)])
    eng.tick(now=1005)
    assert eng.poll(now=1005) is None


def test_quiet_hours_suppresses():
    eng = DriveInitiationEngine(cooldown=0)
    eng.feed_curiosity([_arc(1.4, 0.1, turns=4)])
    eng.tick(now=1000)
    assert eng.poll(now=1000, quiet=True) is None
    # but fires once quiet lifts
    assert eng.poll(now=1000, quiet=False) is not None


def test_strongest_drive_wins():
    eng = DriveInitiationEngine(cooldown=0)
    eng.feed_curiosity([_arc(0.7, 0.2, turns=3)])   # curiosity ~0.56
    eng.feed_emotion_residue(valence=-0.95, intensity=1.0)  # care ~0.95... boost it
    eng.feed_emotion_residue(valence=-0.95, intensity=1.0)  # care accumulates > curiosity
    eng.tick(now=1000)
    imp = eng.poll(now=1000)
    assert imp is not None and imp.drive_type == "care"


def test_proactive_mind_reaches_out_from_curiosity_arc():
    """End-to-end: ProactiveMind reads a fake CNS's live arcs and reaches out."""
    import asyncio
    from proactive_mind import ProactiveMind

    class _Arc:
        drive = 1.4; satisfaction = 0.1; turns_active = 4; target = "your move to Berlin"

    class _DM:  active = {"a": _Arc()}
    class _Cur: dm = _DM()
    class _CNS:
        interaction_count = 3
        _emotional_trajectory = [{"valence": 0.1, "intensity": 0.2}]
        curiosity_system = _Cur()

    sent = []
    async def _push(text): sent.append(text)

    pm = ProactiveMind(user_name="Mahika")
    pm.attach(_CNS(), _push)
    pm._in_quiet_hours = lambda now_dt: False  # deterministic regardless of wall clock

    asyncio.run(pm._tick())
    assert sent, "expected a proactive message"
    assert "Berlin" in sent[0]  # the message names what it was curious about


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
