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
from datetime import datetime, time as dtime
from typing import Callable, Optional, Awaitable


class ProactiveMind:
    """
    Background coroutine that periodically checks if Eros has something
    to say unprompted, and calls `push_fn` with the message if so.
    """

    CHECK_INTERVAL = 60  # seconds between proactive checks

    def __init__(self, user_name: str = "Mahika"):
        self.user_name = user_name
        self._greeted_morning = False
        self._greeted_night = False
        self._last_date = None
        self._running = False
        self._push_fn: Optional[Callable[[str], Awaitable[None]]] = None
        self._cns = None  # set after CNS boot

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
        now = datetime.now()
        today = now.date()

        # Reset daily flags on new day
        if self._last_date != today:
            self._greeted_morning = False
            self._greeted_night = False
            self._last_date = today

        hour = now.hour

        # Morning greeting (7–9am)
        if 7 <= hour < 9 and not self._greeted_morning:
            self._greeted_morning = True
            await self._morning_brief(now)

        # Evening check-in (21–22)
        elif 21 <= hour < 22 and not self._greeted_night:
            self._greeted_night = True
            await self._push(
                f"It's getting late, {self.user_name}. "
                "How did today go? You can tell me anything."
            )

    async def _morning_brief(self, now: datetime):
        lines = [f"Morning, {self.user_name}."]

        # Day of week context
        day = now.strftime("%A")
        if day == "Monday":
            lines.append("New week. What are we doing with it?")
        elif day == "Friday":
            lines.append("Friday. Whatever's weighing on you from the week — I'm here.")
        elif day in ("Saturday", "Sunday"):
            lines.append("Weekend. No agenda unless you want one.")
        else:
            lines.append("What's the plan today?")

        await self._push(" ".join(lines))

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
