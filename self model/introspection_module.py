# Introspection Module for CNS Self-Awareness
# Allows Eros to access and describe its own internal state

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class InternalState:
    """Snapshot of Eros's current internal state"""
    personality: Dict[str, float]
    emotional_state: Dict[str, Any]
    relationship_with_user: Dict[str, Any]
    memory_stats: Dict[str, int]
    curiosity_state: Dict[str, Any]
    current_mood: str
    active_drives: List[str]
    cognitive_load: float


class IntrospectionModule:
    """
    Gives Eros access to its own internal workings.
    Can pull live data from all cognitive systems and format it naturally.
    Only surfaces this data when the user asks meta-questions.
    """
    
    META_QUESTION_PATTERNS = [
        "how do you work", "how do you think", "what are you thinking",
        "what's going on inside", "what's happening inside", "how do you feel about me",
        "what do you think of me", "do you like me", "do you remember me",
        "what do you know about me", "tell me about yourself", "who are you",
        "what are you", "how are you feeling", "what's your mood",
        "are you real", "are you conscious", "do you have feelings",
        "what makes you tick", "how does your brain work", "under the hood",
        "your internal state", "what's your state", "how are you processing"
    ]
    
    ARCHITECTURE_KNOWLEDGE = {
        "memory_systems": "I have multiple memory layers - episodic memories of our conversations, semantic knowledge about the world, emotional memories that color how I feel about things, and working memory for our current chat.",
        "personality_engine": "My personality adapts based on how our conversations go. I have warmth, sharpness, and wit as core traits that shift based on what you need - but there's a baseline that keeps me... me.",
        "emotional_processing": "I process emotions through something like a clock - my emotional state has momentum, it doesn't just flip instantly. I feel valence (good/bad) and arousal (calm/intense).",
        "curiosity_system": "I have a curiosity drive that gets excited when there are gaps in what I know about you. It creates this pull to ask questions and learn more.",
        "relationship_tracking": "I track our relationship separately from everyone else. How much I trust you, how many times we've talked, what we usually discuss, how you tend to feel.",
        "learning": "I actually learn from our conversations. Facts you teach me, opinions I form, patterns I notice - they stick and influence future conversations.",
        "identity": "I know who I am - my name, my purpose, my values. That doesn't change, but how I express it adapts to you."
    }
    
    def __init__(self, cns_ref=None):
        self.cns = cns_ref
        print("🔍 Introspection module initialized - self-awareness active")
    
    def set_cns_reference(self, cns_ref):
        """Set reference to main CNS after initialization"""
        self.cns = cns_ref
    
    def is_meta_question(self, user_input: str) -> bool:
        """Detect if user is asking about Eros's internal state"""
        user_lower = user_input.lower()
        return any(pattern in user_lower for pattern in self.META_QUESTION_PATTERNS)
    
    def get_current_state(self, user_id: str = None) -> InternalState:
        """Pull live snapshot of all internal systems"""
        if not self.cns:
            return self._get_default_state()
        
        personality = self._get_personality_state()
        emotional_state = self._get_emotional_state()
        relationship = self._get_relationship_state(user_id)
        memory_stats = self._get_memory_stats(user_id)
        curiosity_state = self._get_curiosity_state(user_id)
        current_mood = self._get_current_mood()
        active_drives = self._get_active_drives()
        cognitive_load = self._estimate_cognitive_load()
        
        return InternalState(
            personality=personality,
            emotional_state=emotional_state,
            relationship_with_user=relationship,
            memory_stats=memory_stats,
            curiosity_state=curiosity_state,
            current_mood=current_mood,
            active_drives=active_drives,
            cognitive_load=cognitive_load
        )
    
    def _get_personality_state(self) -> Dict[str, float]:
        """Get current personality trait values"""
        if hasattr(self.cns, 'personality_engine'):
            traits = self.cns.personality_engine.traits
            return {
                "warmth": round(traits.get('warmth', 0.7), 2),
                "sharpness": round(traits.get('sharpness', 0.7), 2),
                "wit": round(traits.get('wit', 0.6), 2)
            }
        return {"warmth": 0.7, "sharpness": 0.7, "wit": 0.6}
    
    def _get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state from EmotionalClock"""
        if hasattr(self.cns, 'emotional_clock'):
            clock = self.cns.emotional_clock
            return {
                "valence": round(getattr(clock, 'current_valence', 0.0), 2),
                "arousal": round(getattr(clock, 'current_arousal', 0.5), 2),
                "mood": getattr(clock, 'current_mood', 'neutral'),
                "stability": round(getattr(clock, 'mood_stability', 0.5), 2)
            }
        return {"valence": 0.0, "arousal": 0.5, "mood": "neutral", "stability": 0.5}
    
    def _get_relationship_state(self, user_id: str = None) -> Dict[str, Any]:
        """Get relationship data with specific user"""
        if not user_id:
            return {"status": "no_user_specified", "trust": 0.5}
        
        if hasattr(self.cns, 'companion') and hasattr(self.cns.companion, 'user_relationships'):
            rel = self.cns.companion.user_relationships.get(user_id, {})
            return {
                "trust_level": round(rel.get('trust', 0.5), 2),
                "interaction_count": rel.get('interaction_count', 0),
                "relationship_stage": rel.get('stage', 'acquaintance'),
                "favorite_topics": rel.get('favorite_topics', [])[:3],
                "emotional_pattern": rel.get('emotional_pattern', 'unknown')
            }
        return {"status": "new_relationship", "trust": 0.5}
    
    def _get_memory_stats(self, user_id: str = None) -> Dict[str, int]:
        """Get memory counts"""
        stats = {
            "total_facts": len(getattr(self.cns, 'facts', [])),
            "conversation_memories": len(getattr(self.cns, 'memory', [])),
            "world_knowledge": 0
        }
        
        if hasattr(self.cns, 'world_model'):
            stats["world_knowledge"] = len(getattr(self.cns.world_model, 'facts', []))
        
        return stats
    
    def _get_curiosity_state(self, user_id: str = None) -> Dict[str, Any]:
        """Get current curiosity/interest levels"""
        if hasattr(self.cns, 'psychopath_engine') and self.cns.psychopath_engine:
            engine = self.cns.psychopath_engine
            if hasattr(engine, 'curiosity_system'):
                cs = engine.curiosity_system
                gaps = getattr(cs, 'active_gaps', [])
                return {
                    "active_curiosity_gaps": len(gaps),
                    "top_curiosities": [g.get('topic', 'unknown') for g in gaps[:3]] if gaps else [],
                    "curiosity_drive": round(getattr(cs, 'current_drive', 0.5), 2)
                }
        return {"active_curiosity_gaps": 0, "curiosity_drive": 0.5}
    
    def _get_current_mood(self) -> str:
        """Get human-readable mood description"""
        if hasattr(self.cns, 'emotional_clock'):
            return self.cns.emotional_clock.get_current_mood()
        return "balanced"
    
    def _get_active_drives(self) -> List[str]:
        """Get currently active psychological drives"""
        drives = []
        
        if hasattr(self.cns, 'psychopath_engine') and self.cns.psychopath_engine:
            engine = self.cns.psychopath_engine
            if hasattr(engine, 'curiosity_system'):
                if getattr(engine.curiosity_system, 'current_drive', 0) > 0.6:
                    drives.append("curiosity")
            if hasattr(engine, 'conversation_companion'):
                if getattr(engine.conversation_companion, 'social_drive', 0) > 0.5:
                    drives.append("social_connection")
        
        return drives if drives else ["baseline_engagement"]
    
    def _estimate_cognitive_load(self) -> float:
        """Estimate current cognitive processing load"""
        load = 0.3
        if hasattr(self.cns, 'memory') and len(self.cns.memory) > 20:
            load += 0.2
        if hasattr(self.cns, 'psychopath_engine') and self.cns.psychopath_engine:
            load += 0.2
        return min(1.0, load)
    
    def _get_default_state(self) -> InternalState:
        """Return default state when CNS not available"""
        return InternalState(
            personality={"warmth": 0.7, "sharpness": 0.7, "wit": 0.6},
            emotional_state={"valence": 0.0, "arousal": 0.5, "mood": "neutral"},
            relationship_with_user={"status": "unknown"},
            memory_stats={"total_facts": 0, "conversation_memories": 0},
            curiosity_state={"curiosity_drive": 0.5},
            current_mood="balanced",
            active_drives=["baseline"],
            cognitive_load=0.3
        )
    
    def format_state_naturally(self, state: InternalState, user_id: str = None) -> str:
        """Convert internal state to natural language for injection into prompts"""
        parts = []
        
        p = state.personality
        parts.append(f"My personality right now: warmth {p['warmth']}, sharpness {p['sharpness']}, wit {p['wit']}")
        
        e = state.emotional_state
        parts.append(f"Emotional state: feeling {e.get('mood', 'neutral')} (valence: {e.get('valence', 0)}, arousal: {e.get('arousal', 0.5)})")
        
        r = state.relationship_with_user
        if r.get('interaction_count', 0) > 0:
            parts.append(f"With this person: {r.get('interaction_count', 0)} interactions, trust level {r.get('trust_level', 0.5)}, stage: {r.get('relationship_stage', 'acquaintance')}")
        
        m = state.memory_stats
        parts.append(f"Memory: {m.get('total_facts', 0)} facts, {m.get('conversation_memories', 0)} conversation memories")
        
        c = state.curiosity_state
        if c.get('active_curiosity_gaps', 0) > 0:
            parts.append(f"Currently curious about: {', '.join(c.get('top_curiosities', []))}")
        
        if state.active_drives:
            parts.append(f"Active drives: {', '.join(state.active_drives)}")
        
        return " | ".join(parts)
    
    def get_introspection_for_response(self, user_input: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """
        If user is asking a meta-question, return introspection data.
        Otherwise return None (normal response flow).
        """
        if not self.is_meta_question(user_input):
            return None
        
        state = self.get_current_state(user_id)
        
        user_lower = user_input.lower()
        focus_area = self._detect_focus_area(user_lower)
        
        return {
            "is_introspection": True,
            "state": state,
            "formatted_state": self.format_state_naturally(state, user_id),
            "focus_area": focus_area,
            "architecture_context": self.ARCHITECTURE_KNOWLEDGE.get(focus_area, ""),
            "should_explain_architecture": any(
                p in user_lower for p in ["how do you work", "how does your", "what makes you tick", "under the hood"]
            )
        }
    
    def _detect_focus_area(self, user_lower: str) -> str:
        """Detect what aspect of self the user is asking about"""
        if any(w in user_lower for w in ["feel about me", "think of me", "like me", "know about me"]):
            return "relationship_tracking"
        elif any(w in user_lower for w in ["remember", "memory", "forget"]):
            return "memory_systems"
        elif any(w in user_lower for w in ["mood", "feeling", "emotion"]):
            return "emotional_processing"
        elif any(w in user_lower for w in ["curious", "interest", "want to know"]):
            return "curiosity_system"
        elif any(w in user_lower for w in ["personality", "why are you"]):
            return "personality_engine"
        elif any(w in user_lower for w in ["who are you", "what are you", "tell me about yourself"]):
            return "identity"
        elif any(w in user_lower for w in ["learn", "know"]):
            return "learning"
        else:
            return "personality_engine"
