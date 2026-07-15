"""
drive_initiation.py — genuine drive-triggered proactivity.

The old proactive layer reached out because of the wall clock (a 7am greeting).
This replaces that with initiation driven by INTERNAL state: motivational
pressure accumulates from three drives, and when the strongest crosses a
threshold, Eros reaches out — and *which* drive won decides WHY it reached out
(and therefore what it says). Behavior emerges from internal set-points, not a
timer. (This is the "innate drives" idea — the thing multi-agent work like
Project Sid explicitly flags as missing.)

The three drives:
  - curiosity   : an unresolved, lingering curiosity arc that never got answered
                  (reads the real DopamineArc drive levels from the curiosity system)
  - care        : emotional residue — the user left on a low note and it wants to check in
  - connection  : slow pressure that grows with silence (time since last contact)

Only the connection drive involves time at all, and even that is a *drive that
builds*, not a scheduled event. A cooldown and a quiet-hours filter keep it
humane. Pure/stdlib so the decision logic is unit-testable; the message text is
produced by the caller, keyed on the impulse's drive_type + target.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any


@dataclass
class InitiationImpulse:
    drive_type: str            # "curiosity" | "care" | "connection"
    strength: float            # pressure at the moment it fired
    reason: str                # human-readable why
    target: Optional[str] = None   # what it's about (for curiosity/care)


# Defaults (tunable; tests override).
INITIATE_THRESHOLD = 1.0
COOLDOWN_SECONDS = 20 * 60          # at most ~one initiation per 20 min
CONNECTION_RATE_PER_HOUR = 0.25     # silence pressure: hits 1.0 after ~4h quiet
CARE_HALFLIFE_SECONDS = 90 * 60     # care fades by half every ~90 min


def _get(obj: Any, name: str, default=0.0):
    """Duck-typed read: works for both objects (arc.drive) and dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class DriveInitiationEngine:
    def __init__(self, threshold: float = INITIATE_THRESHOLD,
                 cooldown: float = COOLDOWN_SECONDS,
                 connection_rate_per_hour: float = CONNECTION_RATE_PER_HOUR,
                 care_halflife: float = CARE_HALFLIFE_SECONDS):
        self.threshold = threshold
        self.cooldown = cooldown
        self.connection_rate = connection_rate_per_hour
        self.care_halflife = care_halflife

        self._curiosity = 0.0
        self._curiosity_target: Optional[str] = None
        self._care = 0.0
        self._care_target: Optional[str] = None
        self._connection = 0.0

        now = time.time()
        self._last_contact = now
        self._last_tick = now
        self._last_initiation: Optional[float] = None  # None = never initiated

    # ---- feeding internal state ------------------------------------------

    def note_contact(self, now: Optional[float] = None):
        """The user just interacted — discharge silence, ease care."""
        now = now if now is not None else time.time()
        self._last_contact = now
        self._connection = 0.0
        self._care *= 0.5  # they're engaging now; the worry eases

    def feed_curiosity(self, arcs: Iterable[Any]):
        """
        Recompute curiosity pressure from the live curiosity arcs. Only arcs
        that are unresolved (satisfaction < 0.6) AND have lingered (>= 2 turns)
        count — a fleeting question doesn't make Eros reach out; an unanswered,
        persistent one does.
        """
        best, target = 0.0, None
        for a in arcs or []:
            drive = float(_get(a, "drive", 0.0))
            sat = float(_get(a, "satisfaction", 0.0))
            turns = float(_get(a, "turns_active", 0.0))
            if sat < 0.6 and turns >= 2:
                pressure = drive * (1.0 - sat)
                if pressure > best:
                    best, target = pressure, _get(a, "target", None)
        self._curiosity = best
        self._curiosity_target = target

    def feed_emotion_residue(self, valence: float, intensity: float = 1.0,
                             target: Optional[str] = None):
        """A turn that ended on a low note builds care pressure."""
        if valence < -0.3:
            # One clearly rough message should be enough to want to check in,
            # so care is weighted a bit above raw |valence| * intensity.
            self._care = min(1.6, self._care + (-valence) * max(0.0, min(1.0, intensity)) * 1.2)
            self._care_target = target or "how you're doing"

    def tick(self, now: Optional[float] = None):
        """Advance time-based dynamics: connection grows, care decays."""
        now = now if now is not None else time.time()
        dt = max(0.0, now - self._last_tick)
        self._last_tick = now
        # connection pressure from accumulated silence
        silence_hours = (now - self._last_contact) / 3600.0
        self._connection = min(1.6, silence_hours * self.connection_rate)
        # care decays toward zero
        if self._care > 0 and dt > 0:
            self._care *= math.exp(-dt / self.care_halflife)

    # ---- deciding whether to reach out -----------------------------------

    def pressures(self) -> Dict[str, float]:
        return {"curiosity": self._curiosity, "care": self._care, "connection": self._connection}

    def poll(self, now: Optional[float] = None, quiet: bool = False) -> Optional[InitiationImpulse]:
        """Return an impulse if the dominant drive has crossed threshold."""
        now = now if now is not None else time.time()
        if quiet:
            return None
        if self._last_initiation is not None and now - self._last_initiation < self.cooldown:
            return None

        pressures = self.pressures()
        drive_type = max(pressures, key=pressures.get)
        strength = pressures[drive_type]
        if strength < self.threshold:
            return None

        if drive_type == "curiosity":
            target = self._curiosity_target
            reason = f"unresolved curiosity about '{target}' kept building (drive {strength:.2f})"
        elif drive_type == "care":
            target = self._care_target
            reason = f"emotional residue — you left on a low note (care {strength:.2f})"
        else:
            target = None
            reason = f"connection drive from {((now - self._last_contact)/3600.0):.1f}h of silence"

        self._last_initiation = now
        self._discharge(drive_type)
        return InitiationImpulse(drive_type=drive_type, strength=round(strength, 3),
                                 reason=reason, target=target)

    def _discharge(self, drive_type: str):
        """After acting on a drive, release its pressure so it doesn't re-fire."""
        if drive_type == "curiosity":
            self._curiosity = 0.0
        elif drive_type == "care":
            self._care = 0.0
        else:
            self._connection = 0.0
            self._last_contact = time.time()  # reaching out counts as contact
