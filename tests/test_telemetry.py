"""
Tests for the interior telemetry layer.

These verify that a per-turn interior snapshot is captured correctly from the
internal objects (emotion signal + game-theory motive competition), that it
round-trips through JSONL, and that capture never raises on missing data.
"""

import os
import sys
import json
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from interior_telemetry import (  # noqa: E402
    capture, InteriorTrace, TelemetryRecorder, PLAYERS,
)


# --- lightweight stand-ins for the real internal objects -------------------

class _Enum:
    def __init__(self, name):
        self.name = name


class _Scores:
    def __init__(self, d):
        self._d = d
    def to_dict(self):
        return dict(self._d)


class _Decision:
    def __init__(self, confidence, scores):
        self.confidence = confidence
        self.player_scores = _Scores(scores)


class _Synth:
    def __init__(self):
        self.response_mode = _Enum("EMPATHETIC_DEPTH")
        self.mode_intensity = 0.72
        self.attention_priority = _Enum("HIGH")
        self.cognitive_load = 0.61
        self.primary_player = "warmth"
        self.secondary_player = "curiosity"
        self.vetoed_players = ["wit"]
        self.game_decision = _Decision(
            0.83, {p: 0.0 for p in PLAYERS} | {"warmth": 0.9, "curiosity": 0.6}
        )


def test_capture_emotion_signal():
    t = capture(
        turn=1, user_id="u1", user_input="i feel awful today",
        emotion_data={"valence": -0.7, "arousal": 0.3, "intensity": 0.8,
                      "emotion": "sadness", "learning_source": "heuristic"},
        response="I'm here.", processing_time=0.12,
    )
    assert t.valence == -0.7
    assert t.arousal == 0.3
    assert t.emotion == "sadness"
    assert t.emotion_source == "heuristic"   # provenance is recorded
    assert t.response_len == len("I'm here.")


def test_capture_motive_competition():
    t = capture(
        turn=2, user_id="u1", user_input="tell me a joke",
        synthesized_context=_Synth(),
    )
    assert t.primary_player == "warmth"
    assert t.secondary_player == "curiosity"
    assert t.vetoed_players == ["wit"]
    assert t.decision_confidence == 0.83
    assert t.response_mode == "EMPATHETIC_DEPTH"
    assert t.attention_priority == "HIGH"
    assert t.player_scores["warmth"] == 0.9
    assert set(t.player_scores) == set(PLAYERS)


def test_capture_memory_and_curiosity():
    t = capture(
        turn=3, user_id="u1", user_input="remember when",
        memory_results={"episodic": object(), "semantic": object()},
        curiosity_gaps=[{"target": "job"}, {"target": "family"}],
    )
    assert t.memory_hits == {"episodic": 1, "semantic": 1}
    assert t.curiosity_gaps == 2


def test_capture_never_raises_on_empty():
    # Everything missing/None -> defaults, no exception.
    t = capture(turn=0, user_id="", user_input="")
    assert isinstance(t, InteriorTrace)
    assert t.emotion == "neutral"
    assert t.player_scores == {}


def test_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.jsonl")
        rec = TelemetryRecorder(path=path, enabled=True)
        t = capture(turn=1, user_id="u1", user_input="hi",
                    emotion_data={"valence": 0.5, "emotion": "joy"})
        rec.record(t)
        rec.record(capture(turn=2, user_id="u1", user_input="bye"))
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        assert len(rows) == 2
        assert rows[0]["turn"] == 1 and rows[0]["emotion"] == "joy"
        assert rows[1]["turn"] == 2


def test_disabled_recorder_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.jsonl")
        rec = TelemetryRecorder(path=path, enabled=False)
        rec.record(capture(turn=1, user_id="u1", user_input="hi"))
        assert not os.path.exists(path)
        assert rec.last is not None  # still tracked in-memory


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
