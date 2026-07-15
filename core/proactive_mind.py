"""
Proactive Mind — the part of Eros that thinks without being asked.
Runs as a background asyncio task alongside the main conversation loop.

Responsibilities:
  - Time-aware greetings (morning, night)
  - Surfacing reminders
  - Noticing patterns and initiating
  - Ambient awareness (time, day, recent conversation context)
"""
import asyncio
import os
import time
from datetime import datetime, time as dtime
from typing import Callable, Optional, Awaitable

try:
    from drive_initiation import DriveInitiationEngine
except ImportError:
    DriveInitiationEngine = None


class ProactiveMind:
    """
    Background coroutine that periodically checks if Eros has something
    to say unprompted, and calls `push_fn` with the message if so.
    """

    CHECK_INTERVAL = 60  # seconds between proactive checks

    def __init__(self, user_name: str = "Mahika"):
        self.user_name = user_name
        self._running = False
        self._push_fn: Optional[Callable[[str], Awaitable[None]]] = None
        self._cns = None  # set after CNS boot
        # Genuine drive-triggered initiation (replaces clock-based greetings).
        self.drive_engine = DriveInitiationEngine() if DriveInitiationEngine else None
        self._last_seen_interactions = 0

    def attach(self, cns, push_fn: Callable[[str], Awaitable[None]]):
        self._cns = cns
        self._push_fn = push_fn

    async def _push(self, text: str):
        if self._push_fn:
            await self._push_fn(text)

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._tick()
            except Exception:
                pass
            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _tick(self):
        """
        Drive-triggered check: read Eros's internal state, let motivational
        pressure build, and reach out only when a drive crosses threshold.
        This is NOT a clock — the trigger is internal, the dominant drive
        decides what gets said.
        """
        if not self.drive_engine:
            return

        now = time.time()
        now_dt = datetime.now()
        cns = self._cns

        # 1) Did the user interact since last check? Discharge silence, and
        #    absorb the emotional residue of their most recent turn.
        interactions = getattr(cns, "interaction_count", 0) if cns else 0
        if interactions > self._last_seen_interactions:
            self._last_seen_interactions = interactions
            self.drive_engine.note_contact(now)
            traj = getattr(cns, "_emotional_trajectory", None) if cns else None
            if traj:
                last = traj[-1]
                self.drive_engine.feed_emotion_residue(
                    valence=float(last.get("valence", 0.0)),
                    intensity=float(last.get("intensity", 0.0)),
                )

        # 2) Feed the live, unresolved curiosity arcs (real DopamineArc drives).
        arcs = []
        try:
            arcs = list(cns.curiosity_system.dm.active.values())
        except Exception:
            arcs = []
        self.drive_engine.feed_curiosity(arcs)

        # 3) Advance time-based dynamics and see if a drive wants to speak.
        self.drive_engine.tick(now)
        impulse = self.drive_engine.poll(now, quiet=self._in_quiet_hours(now_dt))
        if impulse:
            print(f"[PROACTIVE] drive-triggered: {impulse.reason}")
            await self._push(self._message_for(impulse))

    def _in_quiet_hours(self, now_dt: datetime) -> bool:
        """Don't reach out in the middle of the night, however strong the drive."""
        hour = now_dt.hour
        return hour < 7 or hour >= 23

    def _message_for(self, impulse) -> str:
        name = self.user_name
        if impulse.drive_type == "curiosity":
            if impulse.target:
                return (f"Hey {name} — I keep circling back to {impulse.target}. "
                        "You never quite finished that thought and it's been sitting with me. "
                        "What ended up happening?")
            return f"Hey {name}, something you said earlier has been nagging at me. Can we come back to it?"
        if impulse.drive_type == "care":
            return (f"Been thinking about you since earlier, {name}. "
                    "You didn't sound okay. How are you holding up?")
        # connection
        return (f"It's gone a little quiet, {name}. No agenda — just wanted to check in. "
                "How's your day treating you?")

    def stop(self):
        self._running = False


class ReminderEngine:
    """Simple in-memory reminder system (persists via DB when available)."""

    def __init__(self):
        self._reminders = []

    def add(self, user_id: str, text: str, remind_at: datetime):
        self._reminders.append({
            "user_id": user_id,
            "text": text,
            "remind_at": remind_at,
            "done": False,
        })

    def due(self) -> list:
        now = datetime.now()
        due = [r for r in self._reminders if not r["done"] and r["remind_at"] <= now]
        for r in due:
            r["done"] = True
        return due

    async def watch(self, push_fn: Callable[[str], Awaitable[None]], interval: int = 30):
        while True:
            for r in self.due():
                await push_fn(f"Reminder: {r['text']}")
            await asyncio.sleep(interval)
