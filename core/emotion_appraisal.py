"""
emotion_appraisal.py — a hybrid valence/arousal (PAD) signal.

The base emotion inference in this codebase is a fast lexical heuristic
(counting affect words). That is cheap and fine for neutral small-talk, but it
is not a defensible emotional *appraisal* on the turns that actually matter.

This module adds a second tier: on turns that are emotionally *charged* or where
the heuristic is *uncertain*, it escalates to an LLM appraisal that rates the
message's valence and arousal directly, then caches the result. Neutral turns
stay on the fast path, so the extra cost is paid only where it changes behavior.

Every result carries its provenance in `emotion_source`:
  - "heuristic"      — fast lexical path only
  - "llm_appraisal"  — escalated and the LLM rating was used
  - "heuristic_fallback" — escalated but the LLM call failed; heuristic kept

The module is deliberately LLM-agnostic: callers inject an `llm_fn(text) -> {valence, arousal}`
callable, so the escalation/merge/caching logic can be tested in isolation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# Escalation thresholds. These are the "why does the system escalate?" answer.
CHARGED_VALENCE = 0.5    # |valence| at/above this => emotionally significant
CHARGED_INTENSITY = 0.5  # intensity at/above this => emotionally significant
UNCERTAIN_CONFIDENCE = 0.5  # heuristic confidence below this => unsure


def should_escalate(heuristic: Dict[str, Any]) -> bool:
    """
    Decide whether a turn warrants an LLM appraisal.

    Escalate when the turn is CHARGED (carries real emotional weight, so getting
    the number right matters) or the heuristic is UNCERTAIN (low confidence or
    conflicting/mixed signals). Otherwise the fast path is good enough.
    """
    valence = _f(heuristic.get("valence"))
    intensity = _f(heuristic.get("intensity"))
    confidence = _f(heuristic.get("confidence"), 1.0)
    mixed = heuristic.get("mixed_emotions") or []

    charged = abs(valence) >= CHARGED_VALENCE or intensity >= CHARGED_INTENSITY
    uncertain = confidence < UNCERTAIN_CONFIDENCE or (isinstance(mixed, (list, tuple)) and len(mixed) > 1)
    return charged or uncertain


def appraise(
    text: str,
    heuristic: Dict[str, Any],
    llm_fn: Optional[Callable[[str], Optional[Dict[str, float]]]] = None,
    cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Return an emotion dict, escalating to `llm_fn` when warranted.

    `llm_fn(text)` should return {"valence": float, "arousal": float} (arousal
    optional) or None on failure. Results are cached by normalized text.
    """
    result = dict(heuristic)  # never mutate the caller's dict

    if not should_escalate(heuristic) or llm_fn is None:
        result["emotion_source"] = heuristic.get("emotion_source", "heuristic")
        return result

    key = (text or "").strip().lower()
    rating: Optional[Dict[str, float]] = None

    if cache is not None and key in cache:
        rating = cache[key]
    else:
        try:
            rating = llm_fn(text)
        except Exception:
            rating = None
        if rating and cache is not None:
            cache[key] = rating

    if not rating or "valence" not in rating:
        # LLM path failed or gave nothing usable — keep the heuristic, say so.
        result["emotion_source"] = "heuristic_fallback"
        return result

    v = _clamp(_f(rating.get("valence")), -1.0, 1.0)
    a = _clamp(_f(rating.get("arousal", heuristic.get("arousal", 0.5)), 0.5), 0.0, 1.0)
    result["valence"] = v
    result["arousal"] = a
    # Intensity follows the appraised magnitude unless the heuristic was higher.
    result["intensity"] = max(_f(heuristic.get("intensity")), abs(v))
    result["emotion_source"] = "llm_appraisal"
    return result


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_llm_rating(raw: str) -> Optional[Dict[str, float]]:
    """
    Parse an LLM appraisal reply into {valence, arousal}.

    Accepts a JSON object anywhere in the text, tolerating surrounding prose or
    code fences. Returns None if no valence can be recovered.
    """
    import json
    import re

    if not raw:
        return None
    # Grab the first {...} block.
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if "valence" not in obj:
        return None
    out = {"valence": _f(obj.get("valence"))}
    if "arousal" in obj:
        out["arousal"] = _f(obj.get("arousal"))
    return out


APPRAISAL_PROMPT = (
    "You are an emotion appraiser. Rate the emotional content of the user's "
    "message on two axes and respond with ONLY a JSON object, no prose:\n"
    '{"valence": <float -1.0 (very negative) to 1.0 (very positive)>, '
    '"arousal": <float 0.0 (calm) to 1.0 (highly activated)>}\n\n'
    "Message: {text}"
)
