"""
ExperienceBus - Central event system for unified learning across all cognitive systems.

This is the "neural backbone" that connects all of Eros's learning systems.
Every meaningful interaction is broadcast as an ExperiencePayload so all systems
learn from the same shared experience.

Subscribers:
- GrowthTracker: Records growth events, tracks learning milestones
- NeuroplasticOptimizer: Updates neural weights, trait adaptations
- InnerLifeSystem: Queues reflections, processes thoughts
- CNSPersonalityEngine: Adapts personality traits dynamically
- CognitiveOrchestrator: Receives learning feedback for real-time adjustment
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time
import asyncio


class ExperienceType(Enum):
    CONVERSATION = "conversation"
    EMOTIONAL_MOMENT = "emotional_moment"
    BELIEF_CHALLENGE = "belief_challenge"
    CURIOSITY_GAP = "curiosity_gap"
    LEARNING_EVENT = "learning_event"
    RELATIONSHIP_SHIFT = "relationship_shift"
    INSIGHT_GENERATED = "insight_generated"
    PROACTIVE_OUTREACH = "proactive_outreach"
    HARDWARE_SYNC = "hardware_sync"
    # Physical action events (NEW - agentic capabilities)
    PHYSICAL_ACTION = "physical_action"       # An action was attempted
    ACTION_SUCCESS = "action_success"         # Action completed successfully
    ACTION_FAILURE = "action_failure"         # Action failed
    ACTION_PENDING = "action_pending"         # Action awaiting user confirmation (HITL)
    ACTION_REJECTED = "action_rejected"       # User rejected the action (HITL)


@dataclass
class ExperiencePayload:
    """Unified experience data broadcast to all learning systems"""
    experience_type: ExperienceType
    user_id: str
    timestamp: float = field(default_factory=time.time)
    
    message_content: Optional[str] = None
    response_content: Optional[str] = None
    
    emotional_analysis: Dict[str, Any] = field(default_factory=dict)
    cognitive_output: Dict[str, Any] = field(default_factory=dict)
    
    belief_conflict: Optional[Dict[str, Any]] = None
    curiosity_gap: Optional[Dict[str, Any]] = None
    
    personality_state: Dict[str, float] = field(default_factory=dict)
    relationship_stage: Optional[str] = None
    
    learning_signals: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Physical action data (NEW - agentic capabilities)
    action_data: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    
    def get_emotional_intensity(self) -> float:
        return self.emotional_analysis.get('intensity', 0.5)
    
    def get_emotional_valence(self) -> float:
        return self.emotional_analysis.get('valence', 0.0)
    
    def has_learning_opportunity(self) -> bool:
        return bool(
            self.belief_conflict or 
            self.curiosity_gap or
            self.get_emotional_intensity() > 0.7 or
            self.learning_signals.get('is_significant', False)
        )


class ExperienceBus:
    """
    Central event bus that broadcasts experiences to all learning systems.
    
    This creates a unified learning loop where every system observes the same
    experiences and can contribute back their insights.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._async_subscribers: Dict[str, List[Callable]] = {}
        self._experience_history: List[ExperiencePayload] = []
        self._max_history = 100
        self._learning_feedback: Dict[str, Any] = {}
        
        print("🚌 ExperienceBus initialized - unified learning backbone active")
    
    def subscribe(self, system_name: str, callback: Callable[[ExperiencePayload], None]):
        """Subscribe a system to receive experience broadcasts"""
        if system_name not in self._subscribers:
            self._subscribers[system_name] = []
        self._subscribers[system_name].append(callback)
        print(f"🔗 {system_name} subscribed to ExperienceBus")
    
    def subscribe_async(self, system_name: str, callback: Callable[[ExperiencePayload], Any]):
        """Subscribe an async system to receive experience broadcasts"""
        if system_name not in self._async_subscribers:
            self._async_subscribers[system_name] = []
        self._async_subscribers[system_name].append(callback)
        print(f"🔗 {system_name} subscribed to ExperienceBus (async)")
    
    def unsubscribe(self, system_name: str):
        """Unsubscribe a system from experience broadcasts"""
        if system_name in self._subscribers:
            del self._subscribers[system_name]
        if system_name in self._async_subscribers:
            del self._async_subscribers[system_name]
    
    def emit(self, experience: ExperiencePayload):
        """Broadcast an experience to all subscribed systems"""
        self._experience_history.append(experience)
        if len(self._experience_history) > self._max_history:
            self._experience_history.pop(0)
        
        for system_name, callbacks in self._subscribers.items():
            for callback in callbacks:
                try:
                    callback(experience)
                except Exception as e:
                    print(f"⚠️ ExperienceBus: {system_name} failed to process experience: {e}")
    
    async def emit_async(self, experience: ExperiencePayload):
        """Broadcast an experience to all subscribed systems (async-aware)"""
        self._experience_history.append(experience)
        if len(self._experience_history) > self._max_history:
            self._experience_history.pop(0)
        
        for system_name, callbacks in self._subscribers.items():
            for callback in callbacks:
                try:
                    callback(experience)
                except Exception as e:
                    print(f"⚠️ ExperienceBus: {system_name} sync callback failed: {e}")
        
        async_tasks = []
        for system_name, callbacks in self._async_subscribers.items():
            for callback in callbacks:
                async_tasks.append(self._safe_async_call(system_name, callback, experience))
        
        if async_tasks:
            await asyncio.gather(*async_tasks)
    
    async def _safe_async_call(self, system_name: str, callback: Callable, experience: ExperiencePayload):
        """Safely call an async callback"""
        try:
            await callback(experience)
        except Exception as e:
            print(f"⚠️ ExperienceBus: {system_name} async callback failed: {e}")
    
    def contribute_learning(self, system_name: str, learning_data: Dict[str, Any]):
        """Systems can contribute their learning insights back to the bus"""
        if system_name not in self._learning_feedback:
            self._learning_feedback[system_name] = []
        
        self._learning_feedback[system_name].append({
            'timestamp': time.time(),
            'data': learning_data
        })
        
        if len(self._learning_feedback[system_name]) > 50:
            self._learning_feedback[system_name].pop(0)
    
    def get_learning_feedback(self, since: float = 0) -> Dict[str, List[Dict]]:
        """Get learning feedback from all systems since a timestamp"""
        result = {}
        for system_name, feedback_list in self._learning_feedback.items():
            recent = [f for f in feedback_list if f['timestamp'] > since]
            if recent:
                result[system_name] = recent
        return result
    
    def get_recent_experiences(self, count: int = 10, user_id: str = None) -> List[ExperiencePayload]:
        """Get recent experiences, optionally filtered by user"""
        experiences = self._experience_history
        if user_id:
            experiences = [e for e in experiences if e.user_id == user_id]
        return experiences[-count:]
    
    def get_learning_summary(self, user_id: str = None) -> Dict[str, Any]:
        """Get a summary of recent learning across all systems"""
        recent = self.get_recent_experiences(50, user_id)
        
        return {
            'total_experiences': len(recent),
            'belief_challenges': sum(1 for e in recent if e.belief_conflict),
            'curiosity_gaps': sum(1 for e in recent if e.curiosity_gap),
            'high_emotion_moments': sum(1 for e in recent if e.get_emotional_intensity() > 0.7),
            'learning_opportunities': sum(1 for e in recent if e.has_learning_opportunity()),
            'avg_emotional_intensity': sum(e.get_emotional_intensity() for e in recent) / max(len(recent), 1),
            'subscriber_count': len(self._subscribers) + len(self._async_subscribers),
            'feedback_sources': list(self._learning_feedback.keys())
        }


class UnifiedLearningCoordinator:
    """
    Coordinates learning across all systems using the ExperienceBus.
    
    This is the "meta-learner" that:
    1. Collects learning signals from all systems
    2. Synthesizes them into unified insights
    3. Distributes relevant feedback to systems that need it
    """
    
    def __init__(self, experience_bus: ExperienceBus):
        self.bus = experience_bus
        self.learning_state: Dict[str, Any] = {
            'total_experiences': 0,
            'growth_momentum': 0.5,
            'personality_stability': 0.7,
            'relationship_depth_avg': 0.3,
            'insight_generation_rate': 0.0
        }
        
        self.bus.subscribe("LearningCoordinator", self._on_experience)
        print("🧠 UnifiedLearningCoordinator active - meta-learning enabled")
    
    def _on_experience(self, experience: ExperiencePayload):
        """Process every experience for meta-learning"""
        self.learning_state['total_experiences'] += 1
        
        if experience.has_learning_opportunity():
            self.learning_state['growth_momentum'] = min(1.0, 
                self.learning_state['growth_momentum'] + 0.02)
        else:
            self.learning_state['growth_momentum'] = max(0.3,
                self.learning_state['growth_momentum'] - 0.005)
        
        if experience.experience_type == ExperienceType.INSIGHT_GENERATED:
            recent_count = self.learning_state.get('recent_insights', 0) + 1
            self.learning_state['recent_insights'] = recent_count
            self.learning_state['insight_generation_rate'] = min(1.0, recent_count * 0.1)
    
    def get_learning_context_for_orchestrator(self) -> Dict[str, Any]:
        """Provide learning context to the cognitive orchestrator"""
        feedback = self.bus.get_learning_feedback(since=time.time() - 3600)
        
        growth_signals = feedback.get('GrowthTracker', [])
        personality_signals = feedback.get('PersonalityEngine', [])
        neuroplastic_signals = feedback.get('NeuroplasticOptimizer', [])
        
        return {
            'growth_momentum': self.learning_state['growth_momentum'],
            'personality_stability': self.learning_state['personality_stability'],
            'recent_growth_events': len(growth_signals),
            'recent_personality_shifts': len(personality_signals),
            'recent_neuroplastic_updates': len(neuroplastic_signals),
            'insight_rate': self.learning_state['insight_generation_rate'],
            'learning_active': True
        }
    
    def synthesize_user_learning(self, user_id: str) -> Dict[str, Any]:
        """Synthesize learning about a specific user"""
        user_experiences = self.bus.get_recent_experiences(30, user_id)
        
        if not user_experiences:
            return {'known': False}
        
        return {
            'known': True,
            'interaction_count': len(user_experiences),
            'avg_emotion': sum(e.get_emotional_intensity() for e in user_experiences) / len(user_experiences),
            'belief_discussions': sum(1 for e in user_experiences if e.belief_conflict),
            'curiosity_moments': sum(1 for e in user_experiences if e.curiosity_gap),
            'relationship_trajectory': self._calculate_trajectory(user_experiences)
        }
    
    def _calculate_trajectory(self, experiences: List[ExperiencePayload]) -> str:
        """Calculate relationship trajectory from experiences"""
        if len(experiences) < 5:
            return 'new'
        
        early = experiences[:len(experiences)//2]
        late = experiences[len(experiences)//2:]
        
        early_intensity = sum(e.get_emotional_intensity() for e in early) / len(early)
        late_intensity = sum(e.get_emotional_intensity() for e in late) / len(late)
        
        if late_intensity > early_intensity + 0.1:
            return 'deepening'
        elif late_intensity < early_intensity - 0.1:
            return 'cooling'
        else:
            return 'stable'


_global_bus: Optional[ExperienceBus] = None
_global_coordinator: Optional[UnifiedLearningCoordinator] = None


def get_experience_bus() -> ExperienceBus:
    """Get the global experience bus singleton"""
    global _global_bus
    if _global_bus is None:
        _global_bus = ExperienceBus()
    return _global_bus


def get_learning_coordinator() -> UnifiedLearningCoordinator:
    """Get the global learning coordinator singleton"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = UnifiedLearningCoordinator(get_experience_bus())
    return _global_coordinator


def emit_experience(
    experience_type: ExperienceType,
    user_id: str,
    message_content: str = None,
    response_content: str = None,
    emotional_analysis: Dict = None,
    cognitive_output: Dict = None,
    belief_conflict: Dict = None,
    curiosity_gap: Dict = None,
    personality_state: Dict = None,
    relationship_stage: str = None,
    learning_signals: Dict = None,
    metadata: Dict = None,
    action_data: Dict = None
):
    """Convenience function to emit an experience to the global bus"""
    bus = get_experience_bus()
    
    experience = ExperiencePayload(
        experience_type=experience_type,
        user_id=user_id,
        message_content=message_content,
        response_content=response_content,
        emotional_analysis=emotional_analysis or {},
        cognitive_output=cognitive_output or {},
        belief_conflict=belief_conflict,
        curiosity_gap=curiosity_gap,
        personality_state=personality_state or {},
        relationship_stage=relationship_stage,
        learning_signals=learning_signals or {},
        metadata=metadata or {},
        action_data=action_data
    )
    
    bus.emit(experience)
    return experience


def emit_action_experience(
    user_id: str,
    action_type: str,
    action_params: Dict = None,
    success: bool = None,
    result: str = None,
    error: str = None,
    requires_confirmation: bool = False
):
    """Convenience function to emit a physical action experience"""
    if success is None:
        experience_type = ExperienceType.ACTION_PENDING if requires_confirmation else ExperienceType.PHYSICAL_ACTION
    elif success:
        experience_type = ExperienceType.ACTION_SUCCESS
    else:
        experience_type = ExperienceType.ACTION_FAILURE
    
    action_data = {
        "action_type": action_type,
        "params": action_params or {},
        "success": success,
        "result": result,
        "error": error,
        "requires_confirmation": requires_confirmation
    }
    
    return emit_experience(
        experience_type=experience_type,
        user_id=user_id,
        action_data=action_data,
        learning_signals={
            "is_significant": True,
            "action_executed": success is not None,
            "action_type": action_type
        }
    )
