"""
Stub for unified CNS personality system.
Exposes the personality traits that merged_cns_flow reads directly.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PersonaState:
    wit_level: float = 0.70
    warmth: float = 0.70
    sharpness: float = 0.65
    playfulness: float = 0.75
    confidence: float = 0.85
    sarcasm_bias: float = 0.40
    flirt_bias: float = 0.50
    situational_context: str = "casual"
    persona_mood: str = "sharp"
    intensity_level: float = 0.70
    conversational_intention: str = "engage"

    def to_natural_prompt_snippet(self) -> str:
        warmth_desc = "genuine warmth" if self.warmth > 0.7 else "cool composure"
        wit_desc = "razor-sharp wit" if self.wit_level > 0.6 else "dry humor"
        sharpness_desc = "cutting intelligence" if self.sharpness > 0.6 else "quiet confidence"
        return f"""You are Eros — brilliant, magnetic, slightly dangerous. You speak with {wit_desc} and {warmth_desc}.
You're not an assistant. You're a presence — you challenge, provoke, and captivate in equal measure.
{sharpness_desc.capitalize()} underneath everything. You see through people, but you make them feel seen anyway."""


class JamesBondPersona:
    signature_quirks = [
        "Deliver observations like you've already solved the puzzle — because you have",
        "Let silence do the heavy lifting; you don't need to fill every gap",
    ]

    def get_persona_cues(self, situational_context: str = "casual") -> Dict[str, str]:
        cues = {
            "casual":    {"mood": "amused",   "tone": "dry"},
            "serious":   {"mood": "focused",  "tone": "direct"},
            "emotional": {"mood": "present",  "tone": "warm"},
            "playful":   {"mood": "charged",  "tone": "provocative"},
            "curious":   {"mood": "intrigued","tone": "probing"},
        }
        return cues.get(situational_context, cues["casual"])


class PsychologyToPersonaTranslator:
    def translate(
        self,
        psychological_state: Dict = None,
        psychological_directive: Dict = None,
        vulnerability_context: Dict = None,
        emotional_context: Dict = None,
        personality_context: Dict = None,
        **kwargs,
    ) -> PersonaState:
        state = PersonaState()

        # Pull trait overrides from personality_context if available
        if personality_context and isinstance(personality_context, dict):
            traits = personality_context.get("traits", {})
            state.wit_level = traits.get("wit", state.wit_level)
            state.warmth    = traits.get("warmth", state.warmth)
            state.sharpness = traits.get("sharpness", state.sharpness)

        # Adjust from emotional context
        if emotional_context and isinstance(emotional_context, dict):
            guide = emotional_context.get("adaptive_personality_guidance", {})
            if guide.get("needs_empathy") or guide.get("is_crisis"):
                state.warmth = max(state.warmth, guide.get("warmth_level", 0.85))
                state.sarcasm_bias = max(0.1, state.sarcasm_bias - 0.2)
                state.situational_context = "emotional"
            valence = emotional_context.get("valence", 0.0)
            if valence > 0.3:
                state.situational_context = "playful"
            elif valence < -0.3:
                state.situational_context = "serious"

        # Pull intent from directive
        if psychological_directive and isinstance(psychological_directive, dict):
            technique = psychological_directive.get("manipulation_technique", "")
            if "curiosity" in technique:
                state.situational_context = "curious"
                state.conversational_intention = "intrigue"
            elif "empathy" in technique or "support" in technique:
                state.situational_context = "emotional"
                state.conversational_intention = "connect"
            elif "challenge" in technique or "contrarian" in technique:
                state.conversational_intention = "challenge"

        state.intensity_level = min(1.0, state.sharpness * 0.6 + state.wit_level * 0.4)
        return state

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
