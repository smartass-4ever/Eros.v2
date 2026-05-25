"""
Stub for unified CNS personality system.
Exposes the personality traits that merged_cns_flow reads directly.
"""
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PersonaState:
    wit: float = 0.70
    warmth: float = 0.70
    sharpness: float = 0.65
    playfulness: float = 0.75
    confidence: float = 0.85


@dataclass
class JamesBondPersona:
    charm: float = 0.85
    directness: float = 0.80
    composure: float = 0.90


class PsychologyToPersonaTranslator:
    def translate(self, psychological_state: Dict = None) -> PersonaState:
        return PersonaState()

    def get_expression_modifiers(self, persona: PersonaState = None) -> Dict:
        return {}


class UnifiedCNSPersonality:
    def __init__(self):
        self.playfulness = 0.75
        self.enthusiasm_level = 0.70
        self.empathy = 0.80
        self.traits = {
            "warmth": 0.70,
            "sharpness": 0.65,
            "wit": 0.70,
        }

    def get_personality_context(self, user_id: str = None) -> Dict[str, Any]:
        return {
            "traits": self.traits,
            "playfulness": self.playfulness,
            "enthusiasm_level": self.enthusiasm_level,
            "empathy": self.empathy,
        }

    def learn_from_expression_feedback(self, feedback: Dict, user_id: str = None):
        for trait, delta in (feedback or {}).items():
            if trait in self.traits:
                self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + delta))
        self.playfulness = max(0.0, min(1.0, self.playfulness + feedback.get("playfulness", 0)))
        self.empathy = max(0.0, min(1.0, self.empathy + feedback.get("empathy", 0)))
