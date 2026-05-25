"""Stub for UserIntentTracker — tracks user goals and background intents."""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class UserIntent:
    intent_type: str = ""
    description: str = ""
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""


class UserIntentTracker:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.intents: List[UserIntent] = []

    def detect_intents(self, user_input: str, user_id: str = "") -> List[UserIntent]:
        return []

    def get_active_intents(self, user_id: str) -> List[UserIntent]:
        return [i for i in self.intents if i.user_id == user_id]

    def clear_intent(self, intent: UserIntent):
        if intent in self.intents:
            self.intents.remove(intent)
