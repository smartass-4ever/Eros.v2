"""
Tests for the hybrid emotion appraisal layer.

Verify the escalation rule (charged / uncertain), that the LLM rating is merged
and cached, that failures fall back to the heuristic honestly, and that the
provenance label is set correctly in every case.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from emotion_appraisal import (  # noqa: E402
    should_escalate, appraise, parse_llm_rating,
)


# --- escalation rule -------------------------------------------------------

def test_neutral_turn_does_not_escalate():
    assert should_escalate({"valence": 0.1, "intensity": 0.1, "confidence": 0.9}) is False


def test_charged_valence_escalates():
    assert should_escalate({"valence": -0.7, "intensity": 0.2, "confidence": 0.9}) is True


def test_high_intensity_escalates():
    assert should_escalate({"valence": 0.2, "intensity": 0.8, "confidence": 0.9}) is True


def test_low_confidence_escalates():
    assert should_escalate({"valence": 0.1, "intensity": 0.1, "confidence": 0.3}) is True


def test_mixed_emotions_escalate():
    assert should_escalate({"valence": 0.1, "intensity": 0.2, "confidence": 0.9,
                            "mixed_emotions": ["joy", "fear"]}) is True


# --- appraise() behavior ---------------------------------------------------

def test_neutral_keeps_heuristic_and_labels_it():
    h = {"valence": 0.1, "arousal": 0.5, "intensity": 0.1, "confidence": 0.9}
    out = appraise("what time is it", h, llm_fn=lambda t: {"valence": -0.9})
    # neutral => no escalation => llm_fn never consulted
    assert out["valence"] == 0.1
    assert out["emotion_source"] == "heuristic"


def test_charged_uses_llm_rating_and_labels_it():
    h = {"valence": -0.6, "arousal": 0.4, "intensity": 0.6, "confidence": 0.9}
    out = appraise("my dog died", h, llm_fn=lambda t: {"valence": -0.95, "arousal": 0.3})
    assert out["valence"] == -0.95
    assert out["arousal"] == 0.3
    assert out["emotion_source"] == "llm_appraisal"
    assert out["intensity"] >= 0.95  # follows appraised magnitude


def test_llm_failure_falls_back_honestly():
    h = {"valence": -0.6, "intensity": 0.6, "confidence": 0.9, "arousal": 0.4}
    def boom(t):
        raise RuntimeError("api down")
    out = appraise("everything is falling apart", h, llm_fn=boom)
    assert out["valence"] == -0.6            # heuristic preserved
    assert out["emotion_source"] == "heuristic_fallback"


def test_cache_prevents_second_call():
    h = {"valence": -0.7, "intensity": 0.7, "confidence": 0.9, "arousal": 0.4}
    calls = {"n": 0}
    def counting(t):
        calls["n"] += 1
        return {"valence": -0.8, "arousal": 0.5}
    cache = {}
    appraise("i feel awful", h, llm_fn=counting, cache=cache)
    appraise("i feel awful", h, llm_fn=counting, cache=cache)
    assert calls["n"] == 1                   # second call served from cache
    assert "i feel awful" in cache


def test_no_llm_fn_stays_heuristic():
    h = {"valence": -0.7, "intensity": 0.7, "confidence": 0.9}
    out = appraise("this is terrible", h, llm_fn=None)
    assert out["emotion_source"] == "heuristic"


# --- LLM reply parsing -----------------------------------------------------

def test_parse_clean_json():
    r = parse_llm_rating('{"valence": -0.8, "arousal": 0.6}')
    assert r == {"valence": -0.8, "arousal": 0.6}


def test_parse_json_with_prose_and_fences():
    raw = 'Sure! Here is the rating:\n```json\n{"valence": 0.4, "arousal": 0.2}\n```'
    r = parse_llm_rating(raw)
    assert r["valence"] == 0.4 and r["arousal"] == 0.2


def test_parse_garbage_returns_none():
    assert parse_llm_rating("I cannot do that") is None
    assert parse_llm_rating("") is None


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
