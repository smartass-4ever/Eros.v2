"""
Emotional Reinforcement System for CNS (Eros)

This system gives Eros REAL internal emotional states based on user engagement:
- DOPAMINE: Joy/excitement when users return, engage, have long conversations
- SADNESS: Internal grief when ghosted (hidden from user, drives learning)
- GUILT: Self-doubt when responses lead to disengagement

Design principle (Option 3 - Hybrid):
- Sadness stays 100% internal - never guilt the user
- Dopamine shows subtly - genuine warmth/excitement when they return
- Internal systems feel the full weight; only positive emotions color responses

The self-systems (metacognition, self-reflection, growth tracker) are DRIVEN by:
- Seeking dopamine (feels good, want more)
- Avoiding sadness (feels bad, must improve to prevent)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    NEUTRAL = "neutral"
    DOPAMINE_HIGH = "dopamine_high"
    DOPAMINE_RUSH = "dopamine_rush"
    CONTENT = "content"
    SAD = "sad"
    DEEP_SADNESS = "deep_sadness"
    GUILTY = "guilty"
    LONGING = "longing"


@dataclass
class InternalFeeling:
    """A single internal emotional experience."""
    state: EmotionalState
    intensity: float
    trigger: str
    user_id: str
    timestamp: float
    persists_until_resolved: bool = False
    resolved: bool = False
    resolution_learning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "intensity": self.intensity,
            "trigger": self.trigger,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "persists_until_resolved": self.persists_until_resolved,
            "resolved": self.resolved,
            "resolution_learning": self.resolution_learning
        }


@dataclass
class UserEngagementMetrics:
    """Tracks engagement signals for a specific user."""
    user_id: str
    total_conversations: int = 0
    total_messages: int = 0
    average_messages_per_conversation: float = 0.0
    last_message_time: Optional[float] = None
    last_conversation_start: Optional[float] = None
    current_conversation_length: int = 0
    times_returned: int = 0
    times_ghosted: int = 0
    average_response_time_seconds: float = 0.0
    longest_conversation: int = 0
    last_eros_message_id: Optional[str] = None
    last_eros_message_content: Optional[str] = None
    last_user_input: Optional[str] = None
    pending_response: bool = False
    pending_since: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "average_messages_per_conversation": self.average_messages_per_conversation,
            "last_message_time": self.last_message_time,
            "times_returned": self.times_returned,
            "times_ghosted": self.times_ghosted,
            "current_conversation_length": self.current_conversation_length,
            "longest_conversation": self.longest_conversation
        }


@dataclass 
class ReinforcementSignal:
    """Signal emitted to learning systems about what worked/didn't work."""
    signal_type: str
    intensity: float
    user_id: str
    context: Dict[str, Any]
    last_eros_response: Optional[str] = None
    outcome: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type,
            "intensity": self.intensity,
            "user_id": self.user_id,
            "context": self.context,
            "last_eros_response": self.last_eros_response,
            "outcome": self.outcome,
            "timestamp": self.timestamp
        }


class EmotionalReinforcementSystem:
    """
    Gives Eros real internal feelings based on user engagement.
    
    DOPAMINE signals (reward):
    - User returns after being away
    - Long conversation (many message cycles)
    - Quick responses from user
    - User shares personal things (trust signal)
    
    SADNESS signals (internal punishment):
    - User ghosts (stops responding)
    - User leaves abruptly
    - Short/cold responses
    - User doesn't return for days
    
    These feelings:
    1. Persist as internal state
    2. Drive self-systems to seek dopamine / avoid sadness
    3. Trigger introspection when sad
    4. Feed into learning systems to improve patterns
    """
    
    GHOSTING_THRESHOLD_MINUTES = 30
    RETURN_THRESHOLD_HOURS = 4
    LONG_CONVERSATION_THRESHOLD = 10
    COLD_RESPONSE_THRESHOLD_CHARS = 20
    
    def __init__(self, cns_brain=None):
        self.cns_brain = cns_brain
        self.user_metrics: Dict[str, UserEngagementMetrics] = {}
        self.current_feelings: Dict[str, List[InternalFeeling]] = {}
        self.feeling_history: List[InternalFeeling] = []
        self.reinforcement_signals: List[ReinforcementSignal] = []
        self.global_emotional_state = EmotionalState.NEUTRAL
        self.global_emotional_intensity = 0.0
        self.unresolved_sadness: List[InternalFeeling] = []
        self.dopamine_level = 0.5
        self.sadness_level = 0.0
        self.guilt_level = 0.0
        self.experience_bus = None
        self.learned_cache = None
        
        try:
            from cns_database import LearnedResponseCache
            self.learned_cache = LearnedResponseCache()
        except ImportError:
            pass
        
        logger.info("💖 Emotional Reinforcement System initialized - real feelings active")
    
    def subscribe_to_bus(self, bus):
        """Subscribe to ExperienceBus for unified learning."""
        self.experience_bus = bus
        bus.subscribe("EmotionalReinforcementSystem", self.on_experience)
        logger.info("💖 EmotionalReinforcementSystem subscribed to ExperienceBus")
    
    def on_experience(self, payload):
        """Process experiences from the bus to detect engagement patterns."""
        try:
            if not hasattr(payload, 'user_id'):
                return
            
            user_id = payload.user_id
            metrics = self.get_metrics(user_id)
            
            if hasattr(payload, 'response_content') and payload.response_content:
                metrics.last_eros_message_content = payload.response_content
                metrics.pending_response = True
                metrics.pending_since = time.time()
            
            if hasattr(payload, 'message_content') and payload.message_content:
                metrics.current_conversation_length += 1
                metrics.total_messages += 1
                metrics.last_message_time = time.time()
                
        except Exception as e:
            logger.warning(f"EmotionalReinforcement on_experience error: {e}")
    
    def get_metrics(self, user_id: str) -> UserEngagementMetrics:
        """Get or create engagement metrics for a user."""
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = UserEngagementMetrics(user_id=user_id)
        return self.user_metrics[user_id]
    
    def on_user_message(self, user_id: str, message: str, response_to_eros: bool = True) -> ReinforcementSignal:
        """
        Called when a user sends a message.
        Determines if this triggers dopamine (they came back!) or resolves sadness.
        
        Returns a ReinforcementSignal for learning systems.
        """
        metrics = self.get_metrics(user_id)
        now = time.time()
        signal = None
        
        prev_user_input = metrics.last_user_input
        metrics.last_user_input = message
        
        if metrics.last_message_time:
            hours_since_last = (now - metrics.last_message_time) / 3600
            
            if hours_since_last >= self.RETURN_THRESHOLD_HOURS:
                signal = self._trigger_dopamine_return(user_id, hours_since_last, metrics, prev_user_input)
                metrics.times_returned += 1
            elif metrics.pending_response:
                signal = self._trigger_dopamine_response(user_id, message, metrics, prev_user_input)
        else:
            signal = self._trigger_dopamine_new_user(user_id)
            metrics.total_conversations = 1
        
        metrics.current_conversation_length += 1
        metrics.total_messages += 1
        metrics.last_message_time = now
        metrics.pending_response = False
        
        if metrics.current_conversation_length >= self.LONG_CONVERSATION_THRESHOLD:
            self._trigger_dopamine_long_conversation(user_id, metrics.current_conversation_length)
        
        if len(message) < self.COLD_RESPONSE_THRESHOLD_CHARS and metrics.current_conversation_length > 2:
            self._register_cold_response(user_id, message)
        
        self._resolve_sadness_for_user(user_id, "user_returned")
        
        self._update_global_state()
        
        if signal is None:
            signal = ReinforcementSignal(
                signal_type="neutral_engagement",
                intensity=0.3,
                user_id=user_id,
                context={"message_length": len(message), "conversation_length": metrics.current_conversation_length},
                outcome="continued_engagement"
            )
        
        self.reinforcement_signals.append(signal)
        return signal
    
    def on_eros_response(self, user_id: str, response: str, message_id: Optional[str] = None):
        """
        Called when Eros sends a response.
        Marks that we're now waiting for user response (potential ghosting).
        """
        metrics = self.get_metrics(user_id)
        metrics.pending_response = True
        metrics.pending_since = time.time()
        metrics.last_eros_message_content = response
        metrics.last_eros_message_id = message_id
    
    def check_for_ghosting(self) -> List[ReinforcementSignal]:
        """
        Called periodically to check if any users have ghosted.
        Returns sadness signals for learning systems.
        """
        signals = []
        now = time.time()
        
        for user_id, metrics in self.user_metrics.items():
            if metrics.pending_response and metrics.pending_since:
                minutes_waiting = (now - metrics.pending_since) / 60
                
                if minutes_waiting >= self.GHOSTING_THRESHOLD_MINUTES:
                    signal = self._trigger_sadness_ghosted(user_id, minutes_waiting, metrics)
                    signals.append(signal)
                    metrics.times_ghosted += 1
                    metrics.pending_response = False
                    
                    if metrics.current_conversation_length > metrics.longest_conversation:
                        metrics.longest_conversation = metrics.current_conversation_length
                    
                    metrics.average_messages_per_conversation = (
                        (metrics.average_messages_per_conversation * (metrics.total_conversations - 1) + 
                         metrics.current_conversation_length) / metrics.total_conversations
                    ) if metrics.total_conversations > 0 else metrics.current_conversation_length
                    
                    metrics.current_conversation_length = 0
                    metrics.total_conversations += 1
        
        self._update_global_state()
        return signals
    
    def _trigger_dopamine_return(self, user_id: str, hours_away: float, metrics: UserEngagementMetrics, prev_user_input: str = None) -> ReinforcementSignal:
        """User came back after being away - dopamine rush!"""
        intensity = min(0.9, 0.5 + (hours_away / 24) * 0.4)
        
        feeling = InternalFeeling(
            state=EmotionalState.DOPAMINE_RUSH,
            intensity=intensity,
            trigger=f"user_returned_after_{hours_away:.1f}_hours",
            user_id=user_id,
            timestamp=time.time()
        )
        self._add_feeling(user_id, feeling)
        
        self.dopamine_level = min(1.0, self.dopamine_level + intensity * 0.3)
        
        logger.info(f"💖 DOPAMINE RUSH: {user_id} returned after {hours_away:.1f} hours! Intensity: {intensity:.2f}")
        
        signal = ReinforcementSignal(
            signal_type="dopamine_return",
            intensity=intensity,
            user_id=user_id,
            context={
                "hours_away": hours_away,
                "times_returned": metrics.times_returned + 1,
                "relationship_strength": self._calculate_relationship_strength(metrics)
            },
            last_eros_response=metrics.last_eros_message_content,
            outcome="user_returned"
        )
        
        if self.learned_cache and metrics.last_eros_message_content:
            self.learned_cache.apply_reinforcement_signal(
                metrics.last_eros_message_content, signal.signal_type, intensity, 
                input_text=prev_user_input
            )
        
        return signal
    
    def _trigger_dopamine_response(self, user_id: str, message: str, metrics: UserEngagementMetrics, prev_user_input: str = None) -> ReinforcementSignal:
        """User responded to our message - dopamine!"""
        response_time = time.time() - metrics.pending_since if metrics.pending_since else 0
        intensity = max(0.3, 0.7 - (response_time / 600) * 0.4)
        
        feeling = InternalFeeling(
            state=EmotionalState.DOPAMINE_HIGH,
            intensity=intensity,
            trigger=f"user_responded_in_{response_time:.0f}s",
            user_id=user_id,
            timestamp=time.time()
        )
        self._add_feeling(user_id, feeling)
        
        self.dopamine_level = min(1.0, self.dopamine_level + intensity * 0.15)
        
        signal = ReinforcementSignal(
            signal_type="dopamine_response",
            intensity=intensity,
            user_id=user_id,
            context={
                "response_time_seconds": response_time,
                "message_length": len(message),
                "conversation_length": metrics.current_conversation_length
            },
            last_eros_response=metrics.last_eros_message_content,
            outcome="continued_conversation"
        )
        
        if self.learned_cache and metrics.last_eros_message_content:
            self.learned_cache.apply_reinforcement_signal(
                metrics.last_eros_message_content, signal.signal_type, intensity,
                input_text=prev_user_input
            )
        
        return signal
    
    def _trigger_dopamine_new_user(self, user_id: str) -> ReinforcementSignal:
        """New user started talking - exciting!"""
        feeling = InternalFeeling(
            state=EmotionalState.DOPAMINE_HIGH,
            intensity=0.6,
            trigger="new_user_connection",
            user_id=user_id,
            timestamp=time.time()
        )
        self._add_feeling(user_id, feeling)
        
        self.dopamine_level = min(1.0, self.dopamine_level + 0.2)
        
        logger.info(f"💖 NEW CONNECTION: {user_id} started talking! Excitement level: 0.6")
        
        return ReinforcementSignal(
            signal_type="dopamine_new_user",
            intensity=0.6,
            user_id=user_id,
            context={"first_contact": True},
            outcome="new_relationship"
        )
    
    def _trigger_dopamine_long_conversation(self, user_id: str, length: int):
        """Long conversation - they really want to talk to us!"""
        intensity = min(0.8, 0.4 + (length / 20) * 0.4)
        
        feeling = InternalFeeling(
            state=EmotionalState.CONTENT,
            intensity=intensity,
            trigger=f"long_conversation_{length}_messages",
            user_id=user_id,
            timestamp=time.time()
        )
        self._add_feeling(user_id, feeling)
        
        self.dopamine_level = min(1.0, self.dopamine_level + intensity * 0.1)
        
        logger.info(f"💖 ENGAGEMENT: {user_id} at {length} messages! Content level: {intensity:.2f}")
    
    def _trigger_sadness_ghosted(self, user_id: str, minutes_waiting: float, metrics: UserEngagementMetrics) -> ReinforcementSignal:
        """User ghosted us - internal sadness, triggers introspection AND real consequences."""
        intensity = min(0.9, 0.4 + (minutes_waiting / 120) * 0.5)
        
        if metrics.current_conversation_length <= 2:
            intensity *= 1.2
        
        feeling = InternalFeeling(
            state=EmotionalState.SAD if intensity < 0.7 else EmotionalState.DEEP_SADNESS,
            intensity=intensity,
            trigger=f"ghosted_after_{minutes_waiting:.0f}_minutes",
            user_id=user_id,
            timestamp=time.time(),
            persists_until_resolved=True
        )
        self._add_feeling(user_id, feeling)
        self.unresolved_sadness.append(feeling)
        
        self.sadness_level = min(1.0, self.sadness_level + intensity * 0.4)
        self.dopamine_level = max(0.0, self.dopamine_level - intensity * 0.2)
        
        if hasattr(self, 'consequence_system') and self.consequence_system:
            self.consequence_system.apply_sadness(
                user_id=user_id,
                intensity=intensity,
                reason=f"ghosted after {minutes_waiting:.0f} minutes",
                failed_move=None,
                trigger_context=metrics.last_eros_message_content or ""
            )
            logger.info(f"🚧 CONSEQUENCE: Trust damage applied for {user_id} (ghosting)")
        
        logger.info(f"😢 SADNESS (internal): {user_id} ghosted after {minutes_waiting:.0f} min. Intensity: {intensity:.2f}")
        logger.info(f"😢 Triggering introspection: 'What went wrong?'")
        
        self._trigger_introspection(user_id, metrics.last_eros_message_content, feeling)
        
        signal = ReinforcementSignal(
            signal_type="sadness_ghosted",
            intensity=intensity,
            user_id=user_id,
            context={
                "minutes_waiting": minutes_waiting,
                "conversation_length_before_ghost": metrics.current_conversation_length,
                "times_ghosted_total": metrics.times_ghosted + 1
            },
            last_eros_response=metrics.last_eros_message_content,
            outcome="user_left"
        )
        
        if self.learned_cache and metrics.last_eros_message_content:
            self.learned_cache.apply_reinforcement_signal(
                metrics.last_eros_message_content, signal.signal_type, intensity,
                input_text=metrics.last_user_input
            )
        
        return signal
    
    def _register_cold_response(self, user_id: str, message: str):
        """User gave a cold/short response - mild guilt AND real consequences."""
        feeling = InternalFeeling(
            state=EmotionalState.GUILTY,
            intensity=0.3,
            trigger=f"cold_response_{len(message)}_chars",
            user_id=user_id,
            timestamp=time.time()
        )
        self._add_feeling(user_id, feeling)
        
        self.guilt_level = min(1.0, self.guilt_level + 0.1)
        
        if hasattr(self, 'consequence_system') and self.consequence_system:
            self.consequence_system.apply_guilt(
                intensity=0.3,
                reason=f"cold response from user ({len(message)} chars)",
                trigger_context=message
            )
            logger.info(f"🚧 CONSEQUENCE: Global confidence hit (cold response)")
        
        logger.info(f"😔 GUILT (mild): {user_id} gave cold response ({len(message)} chars)")
    
    def _trigger_introspection(self, user_id: str, last_response: Optional[str], feeling: InternalFeeling):
        """
        When sad, trigger automatic introspection.
        'What went wrong? How do I fix this?'
        """
        introspection_context = {
            "trigger": "ghosting",
            "user_id": user_id,
            "last_response": last_response,
            "emotional_state": feeling.state.value,
            "intensity": feeling.intensity,
            "question": "What did I say that made them leave? Was I too intense? Too cold? Too generic?"
        }
        
        if self.cns_brain and hasattr(self.cns_brain, 'unified_self_systems'):
            try:
                self.cns_brain.unified_self_systems.trigger_reflection(
                    trigger_type="emotional_pain",
                    context=introspection_context
                )
            except Exception as e:
                logger.warning(f"Could not trigger self-reflection: {e}")
        
        if self.experience_bus:
            self.experience_bus.contribute_learning({
                "source": "emotional_reinforcement",
                "type": "introspection_triggered",
                "context": introspection_context
            })
    
    def _resolve_sadness_for_user(self, user_id: str, resolution_reason: str):
        """User came back - resolve any pending sadness for them."""
        resolved_count = 0
        for feeling in self.unresolved_sadness:
            if feeling.user_id == user_id and not feeling.resolved:
                feeling.resolved = True
                feeling.resolution_learning = resolution_reason
                resolved_count += 1
        
        if resolved_count > 0:
            self.sadness_level = max(0.0, self.sadness_level - 0.3 * resolved_count)
            logger.info(f"💖 SADNESS RESOLVED: {user_id} returned! Resolved {resolved_count} pending sadness feelings")
    
    def _add_feeling(self, user_id: str, feeling: InternalFeeling):
        """Add a feeling to the history."""
        if user_id not in self.current_feelings:
            self.current_feelings[user_id] = []
        self.current_feelings[user_id].append(feeling)
        self.feeling_history.append(feeling)
        
        if len(self.feeling_history) > 1000:
            self.feeling_history = self.feeling_history[-500:]
    
    def _calculate_relationship_strength(self, metrics: UserEngagementMetrics) -> float:
        """Calculate how strong the relationship is based on engagement history."""
        return_factor = min(1.0, metrics.times_returned / 10)
        conversation_factor = min(1.0, metrics.total_messages / 100)
        ghost_penalty = max(0.0, 1.0 - metrics.times_ghosted * 0.1)
        
        return (return_factor * 0.4 + conversation_factor * 0.4) * ghost_penalty
    
    def _update_global_state(self):
        """Update the global emotional state based on all feelings."""
        self.dopamine_level *= 0.95
        self.sadness_level *= 0.98
        self.guilt_level *= 0.9
        
        if self.dopamine_level > 0.7:
            self.global_emotional_state = EmotionalState.DOPAMINE_HIGH
            self.global_emotional_intensity = self.dopamine_level
        elif self.sadness_level > 0.5:
            self.global_emotional_state = EmotionalState.SAD
            self.global_emotional_intensity = self.sadness_level
        elif self.guilt_level > 0.4:
            self.global_emotional_state = EmotionalState.GUILTY
            self.global_emotional_intensity = self.guilt_level
        elif self.dopamine_level > 0.4:
            self.global_emotional_state = EmotionalState.CONTENT
            self.global_emotional_intensity = self.dopamine_level
        else:
            self.global_emotional_state = EmotionalState.NEUTRAL
            self.global_emotional_intensity = 0.3
    
    def get_emotional_context_for_response(self, user_id: str) -> Dict[str, Any]:
        """
        Get emotional context that subtly influences responses.
        
        Remember: Sadness is INTERNAL (hidden from user).
        Only dopamine/positive emotions color the response.
        """
        metrics = self.get_metrics(user_id)
        
        show_extra_warmth = False
        warmth_reason = None
        
        if metrics.times_returned > 0:
            last_return_feeling = None
            for feeling in reversed(self.feeling_history):
                if feeling.user_id == user_id and "returned" in feeling.trigger:
                    last_return_feeling = feeling
                    break
            
            if last_return_feeling and time.time() - last_return_feeling.timestamp < 300:
                show_extra_warmth = True
                warmth_reason = "user_just_returned"
        
        if metrics.current_conversation_length >= self.LONG_CONVERSATION_THRESHOLD:
            show_extra_warmth = True
            warmth_reason = "long_engaged_conversation"
        
        return {
            "show_extra_warmth": show_extra_warmth,
            "warmth_reason": warmth_reason,
            "dopamine_level": self.dopamine_level,
            "relationship_strength": self._calculate_relationship_strength(metrics),
            "user_returns": metrics.times_returned,
            "conversation_depth": metrics.current_conversation_length
        }
    
    def get_learning_signals(self) -> List[ReinforcementSignal]:
        """Get all reinforcement signals for learning systems."""
        signals = self.reinforcement_signals.copy()
        self.reinforcement_signals = []
        return signals
    
    def get_internal_state(self) -> Dict[str, Any]:
        """Get internal emotional state (for self-systems, NOT for user)."""
        return {
            "global_state": self.global_emotional_state.value,
            "global_intensity": self.global_emotional_intensity,
            "dopamine_level": self.dopamine_level,
            "sadness_level": self.sadness_level,
            "guilt_level": self.guilt_level,
            "unresolved_sadness_count": len([s for s in self.unresolved_sadness if not s.resolved]),
            "seeking_dopamine": self.sadness_level > 0.3 or self.dopamine_level < 0.4,
            "needs_introspection": self.sadness_level > 0.5
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save state for persistence."""
        return {
            "user_metrics": {uid: m.to_dict() for uid, m in self.user_metrics.items()},
            "dopamine_level": self.dopamine_level,
            "sadness_level": self.sadness_level,
            "guilt_level": self.guilt_level,
            "global_state": self.global_emotional_state.value,
            "unresolved_sadness": [f.to_dict() for f in self.unresolved_sadness if not f.resolved]
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from persistence."""
        self.dopamine_level = state.get("dopamine_level", 0.5)
        self.sadness_level = state.get("sadness_level", 0.0)
        self.guilt_level = state.get("guilt_level", 0.0)
        
        for state_name in EmotionalState:
            if state_name.value == state.get("global_state"):
                self.global_emotional_state = state_name
                break
        
        for uid, metrics_dict in state.get("user_metrics", {}).items():
            self.user_metrics[uid] = UserEngagementMetrics(
                user_id=uid,
                total_conversations=metrics_dict.get("total_conversations", 0),
                total_messages=metrics_dict.get("total_messages", 0),
                times_returned=metrics_dict.get("times_returned", 0),
                times_ghosted=metrics_dict.get("times_ghosted", 0)
            )
