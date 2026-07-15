"""
Interior telemetry — a structured, per-turn record of Eros's internal state.

The point of this module is to make the agent's *interior* measurable rather
than something you infer from prose. Every turn, Eros produces a low-dimensional
snapshot of what was going on inside it: the emotional control signal (valence /
arousal / intensity), the competition between motives (the game-theory player
scores, which motive won, what got vetoed), attention/load, memory and curiosity
activity, and how that mapped to the response.

Design goals:
  - Stdlib only, no heavy deps — so it can be unit-tested and read in isolation.
  - Defensive capture — telemetry must NEVER break the response pipeline.
  - Honest provenance — every derived number records HOW it was derived
    (`emotion_source`), so "where does this number come from?" has an answer.

Traces are appended as JSON Lines (one object per turn) to
`data/interior_trace.jsonl` by default. Set EROS_TELEMETRY=0 to disable, or
EROS_TELEMETRY_PATH to redirect.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


PLAYERS = ("memory", "curiosity", "warmth", "wit", "beliefs")


@dataclass
class InteriorTrace:
    """One turn's snapshot of the agent interior."""
    turn: int = 0
    timestamp: float = field(default_factory=time.time)
    user_id: str = "default"
    user_input: str = ""            # truncated for readability

    # -- Emotion: the low-dimensional control signal --
    valence: float = 0.0
    arousal: float = 0.5
    intensity: float = 0.0
    emotion: str = "neutral"
    emotion_source: str = "unknown"  # provenance of the emotion numbers

    # -- Motive competition: the game-theory decision core --
    player_scores: Dict[str, float] = field(default_factory=dict)
    primary_player: Optional[str] = None
    secondary_player: Optional[str] = None
    vetoed_players: List[str] = field(default_factory=list)
    decision_confidence: float = 0.0
    response_mode: Optional[str] = None
    mode_intensity: float = 0.0

    # -- Attention / load --
    attention_priority: Optional[str] = None
    cognitive_load: float = 0.0

    # -- Memory & curiosity activity --
    memory_hits: Dict[str, int] = field(default_factory=dict)
    curiosity_gaps: int = 0

    # -- Output --
    response_len: int = 0
    processing_time: float = 0.0

    # -- Flags --
    safety_intervention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _truncate(text: str, n: int = 240) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def capture(
    *,
    turn: int,
    user_id: str,
    user_input: str,
    emotion_data: Optional[Dict[str, Any]] = None,
    synthesized_context: Any = None,
    memory_results: Optional[Dict[str, Any]] = None,
    curiosity_gaps: Optional[List[Any]] = None,
    priority: Any = None,
    cognitive_load: Any = None,
    response: str = "",
    processing_time: float = 0.0,
    safety_intervention: bool = False,
) -> InteriorTrace:
    """
    Build an InteriorTrace from whatever internal objects are available.

    Everything is optional and read defensively: missing pieces just leave
    their default values. This must never raise into the pipeline.
    """
    t = InteriorTrace(turn=turn, user_id=user_id or "default", user_input=_truncate(user_input))

    emotion_data = emotion_data or {}
    t.valence = _f(emotion_data.get("valence"))
    t.arousal = _f(emotion_data.get("arousal"), 0.5)
    t.intensity = _f(emotion_data.get("intensity"))
    t.emotion = str(emotion_data.get("emotion", "neutral"))
    # Provenance: the hybrid appraisal sets emotion_source to the path actually
    # taken (heuristic / llm_appraisal / heuristic_fallback); prefer it over the
    # heuristic's own learning_source tag.
    t.emotion_source = str(emotion_data.get("emotion_source") or emotion_data.get("learning_source") or "heuristic")

    # Motive competition from the game-theory decision (carried on SynthesizedContext).
    if synthesized_context is not None:
        t.response_mode = _attr(synthesized_context, "response_mode", transform=_enum_name)
        t.mode_intensity = _f(_attr(synthesized_context, "mode_intensity"))
        t.attention_priority = _attr(synthesized_context, "attention_priority", transform=_enum_name)
        t.cognitive_load = _f(_attr(synthesized_context, "cognitive_load"))
        t.primary_player = _attr(synthesized_context, "primary_player", transform=_enum_name)
        t.secondary_player = _attr(synthesized_context, "secondary_player", transform=_enum_name)
        veto = _attr(synthesized_context, "vetoed_players") or []
        t.vetoed_players = [_enum_name(v) for v in veto] if isinstance(veto, (list, tuple)) else []

        decision = _attr(synthesized_context, "game_decision")
        if decision is not None:
            t.decision_confidence = _f(_attr(decision, "confidence"))
            scores = _attr(decision, "player_scores")
            if scores is not None and hasattr(scores, "to_dict"):
                t.player_scores = {k: round(_f(v), 4) for k, v in scores.to_dict().items()}

    # Priority / load can also be passed directly (fallback if not on context).
    if not t.attention_priority and priority is not None:
        t.attention_priority = _enum_name(priority)
    if not t.cognitive_load and cognitive_load is not None:
        t.cognitive_load = _f(_attr(cognitive_load, "total_load", cognitive_load))

    # Memory hits: count each memory type that returned something.
    if memory_results:
        hits: Dict[str, int] = {}
        for k in memory_results:
            key = _enum_name(k).lower() if not isinstance(k, str) else k.lower()
            hits[key] = hits.get(key, 0) + 1
        t.memory_hits = hits

    t.curiosity_gaps = len(curiosity_gaps or [])
    t.response_len = len(response or "")
    t.processing_time = round(_f(processing_time), 4)
    t.safety_intervention = bool(safety_intervention)
    return t


def _attr(obj: Any, name: str, default: Any = None, transform=None):
    val = getattr(obj, name, default)
    if val is None:
        return default
    return transform(val) if transform else val


def _enum_name(v: Any) -> Optional[str]:
    if v is None:
        return None
    # Enum -> its .name or .value; plain str/other -> str
    if hasattr(v, "name"):
        return v.name
    if hasattr(v, "value"):
        return str(v.value)
    return str(v)


class TelemetryRecorder:
    """Appends InteriorTrace rows as JSON Lines. Cheap, append-only, crash-safe."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.environ.get("EROS_TELEMETRY", "1") != "0"
        self.enabled = enabled
        self.path = path or os.environ.get("EROS_TELEMETRY_PATH") or os.path.join("data", "interior_trace.jsonl")
        self._last: Optional[InteriorTrace] = None

    def record(self, trace: InteriorTrace) -> None:
        self._last = trace
        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:  # telemetry must never break the pipeline
            print(f"[TELEMETRY] write failed (continuing): {e}")

    @property
    def last(self) -> Optional[InteriorTrace]:
        return self._last


# Module-level singleton for convenience.
recorder = TelemetryRecorder()


def record_turn(**kwargs) -> InteriorTrace:
    """Capture + record in one call. Returns the trace (also useful for tests/UX)."""
    trace = capture(**kwargs)
    recorder.record(trace)
    return trace
