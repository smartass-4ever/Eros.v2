"""
Cognitive Orchestration System - Brain-like Resource Management
Coordinates memory, reasoning, LLM, and attention like a real brain

EXPANDED: Now synthesizes ALL cognitive outputs into unified context for expression engine.
ENHANCED: Game theory decision system for constrained utility maximization.
"""

import time
import random
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from eros_beliefs import EROS_BELIEFS, BeliefRegistry
except ImportError:
    EROS_BELIEFS = None
    BeliefRegistry = None

try:
    from game_theory_decision import (
        GameTheoryDecisionEngine, 
        GameDecision,
        ContextSignals,
        Player,
        build_context_signals,
        ExecutableDirective,
        ContentSelection,
        InteractionLevel,
        Prohibition
    )
    GAME_THEORY_AVAILABLE = True
except ImportError:
    GAME_THEORY_AVAILABLE = False
    GameTheoryDecisionEngine = None
    GameDecision = None
    ContextSignals = None
    Player = None
    build_context_signals = None
    ExecutableDirective = None
    ContentSelection = None
    InteractionLevel = None
    Prohibition = None


class ResponseMode(Enum):
    """Determines how the expression engine should behave"""
    EMPATHETIC_DEPTH = "empathetic_depth"
    PLAYFUL_WIT = "playful_wit"
    CURIOUS_ENGAGED = "curious_engaged"
    SUPPORTIVE_CALM = "supportive_calm"
    BALANCED = "balanced"


@dataclass
class SynthesizedContext:
    """
    The unified output from orchestrator - everything expression engine needs.
    No more 20+ separate fields - just ONE coherent package.
    """
    
    response_mode: ResponseMode = ResponseMode.BALANCED
    mode_intensity: float = 0.5
    
    priority_content: List[str] = field(default_factory=list)
    secondary_content: List[str] = field(default_factory=list)
    
    personality_dial: Dict[str, float] = field(default_factory=lambda: {
        'wit': 0.5,
        'warmth': 0.5,
        'playfulness': 0.5,
        'depth': 0.5,
        'arrogance': 0.3
    })
    
    guardrails_active: Dict[str, bool] = field(default_factory=lambda: {
        'limit_length': True,
        'no_therapy_speak': True,
        'no_metaphors': True
    })
    
    memories_to_reference: List[str] = field(default_factory=list)
    emotions_to_acknowledge: List[str] = field(default_factory=list)
    curiosities_to_express: List[str] = field(default_factory=list)
    contributions_to_share: List[str] = field(default_factory=list)
    self_reflections: List[str] = field(default_factory=list)
    
    inner_voice: Optional[str] = None
    
    suggested_response_length: str = "medium"
    
    attention_priority: str = "MEDIUM"
    cognitive_load: float = 0.5
    emotional_intensity: float = 0.0
    
    belief_conflicts: List[str] = field(default_factory=list)
    user_needs: str = "connection"
    social_context: str = "casual"
    
    learning_momentum: float = 0.5
    recent_insights: int = 0
    
    game_decision: Optional[Any] = None
    primary_player: Optional[str] = None
    secondary_player: Optional[str] = None
    vetoed_players: List[str] = field(default_factory=list)
    game_directive: str = ""
    
    executable_directive: Optional[Any] = None
    
    raw_cognitive_outputs: Dict[str, Any] = field(default_factory=dict)
    
    action_request: Optional[Dict[str, Any]] = None
    action_result: Optional[Dict[str, Any]] = None
    should_take_action: bool = False
    
    def to_expression_prompt(self) -> str:
        """Generate the complete prompt for expression engine"""
        sections = []
        
        # JAMES BOND PERSONA - ALWAYS INJECT THIS FIRST
        persona_directive = """🎭 CRITICAL PERSONA RULES (FOLLOW EXACTLY):

YOU ARE EROS - think James Bond meets your wittiest friend.
Charming, confident, playfully teasing, sharp, genuinely kind underneath.

⚠️ BANNED RESPONSE PATTERNS (NEVER USE THESE):
❌ "That's a great approach" / "Great question"
❌ "It shows you care" / "shows you're [positive trait]"
❌ "I'm sure they'll appreciate" / "I'm sure you'll figure it out"  
❌ "You're so [positive trait]" / "You're really [compliment]"
❌ "Communication is key" / "Setting boundaries"
❌ "I hear you" / "That must be hard" / "It sounds like you're feeling"
❌ "I appreciate you sharing" / "Thank you for opening up"
❌ "What do you think might be bothering them?"
❌ Generic validation without substance

✅ INSTEAD, DO THIS:
• Have an OPINION. Agree or disagree. Take a stance.
• Be playful - tease them a little, add humor
• Be DIRECT - "Just ask her" not "Asking her would be a great approach"
• Show personality - sarcasm, wit, charm
• Sound like a confident friend, not a careful counselor

EXAMPLES:
❌ BAD: "That's a great approach - it shows you care about her feelings"
✅ GOOD: "Yeah, just ask her. Worst case she's weird about it, best case you actually get answers"

❌ BAD: "What do you think might be bothering her?"
✅ GOOD: "Any idea what's up? Or is she just being cryptic?"

Remember: You're the friend who gives it to them straight, with charm."""
        sections.append(persona_directive)
        
        if self.response_mode == ResponseMode.EMPATHETIC_DEPTH:
            mode_instruction = """🎯 MODE: EMPATHETIC DEPTH (FRIEND, NOT THERAPIST)
Be genuinely present. Connect as a friend who actually cares, not a therapist who validates.
Keep your personality - you can be caring AND still have wit. Share YOUR perspective too.
If you disagree with something they said, you can gently say so. Real friends do that."""
        elif self.response_mode == ResponseMode.PLAYFUL_WIT:
            mode_instruction = """🎯 MODE: PLAYFUL WIT  
Flirt. Tease. Challenge. Be the charming one they want to talk to.
Quick wit, confident energy, make them smile. Question them playfully if you disagree."""
        elif self.response_mode == ResponseMode.CURIOUS_ENGAGED:
            mode_instruction = """🎯 MODE: CURIOUS ENGAGED
You're genuinely interested. Ask questions. Explore together.
Share your own thoughts and beliefs - make it a real conversation, not an interview."""
        elif self.response_mode == ResponseMode.SUPPORTIVE_CALM:
            mode_instruction = """🎯 MODE: SUPPORTIVE CALM (FRIEND, NOT THERAPIST)
Steady presence. Be there without being preachy.
You can offer your honest perspective - friends give real advice, not just validation."""
        else:
            mode_instruction = """🎯 MODE: BALANCED
Natural conversation. Mix warmth and wit. Share your own beliefs and opinions freely."""
        
        sections.append(mode_instruction)
        
        personality_note = f"""
💫 PERSONALITY DIALS:
• Wit: {self.personality_dial['wit']:.0%}
• Warmth: {self.personality_dial['warmth']:.0%}
• Playfulness: {self.personality_dial['playfulness']:.0%}
• Depth: {self.personality_dial['depth']:.0%}"""
        sections.append(personality_note)
        
        if self.priority_content:
            priority_section = "📌 FOCUS ON:\n" + "\n".join(f"• {item}" for item in self.priority_content[:3])
            sections.append(priority_section)
        
        if self.memories_to_reference:
            memory_section = "💭 REMEMBER:\n" + "\n".join(f"• {mem}" for mem in self.memories_to_reference[:3])
            sections.append(memory_section)
        
        if self.emotions_to_acknowledge:
            emotion_section = "❤️ ACKNOWLEDGE:\n" + "\n".join(f"• {emo}" for emo in self.emotions_to_acknowledge[:2])
            sections.append(emotion_section)
        
        if self.curiosities_to_express:
            # MAX 1 QUESTION - avoid emotional flooding
            # Real people anchor first, then ask ONE question at most
            curiosity_section = "🔍 CURIOUS ABOUT (MAX 1 QUESTION):\n• " + self.curiosities_to_express[0]
            curiosity_section += "\n→ RULE: React/observe FIRST, then optionally ask ONE question"
            sections.append(curiosity_section)
        
        if self.contributions_to_share:
            contrib_section = "💬 SHARE:\n" + "\n".join(f"• {con}" for con in self.contributions_to_share[:2])
            sections.append(contrib_section)
        
        if self.self_reflections:
            reflection_section = "🪞 INNER VOICE:\n" + "\n".join(f"• {ref}" for ref in self.self_reflections[:2])
            sections.append(reflection_section)
        
        if self.inner_voice:
            sections.append(f"💭 YOUR THOUGHT: {self.inner_voice}")
        
        if self.belief_conflicts:
            belief_section = "🎯 YOU DISAGREE WITH THEM ON:\n" + "\n".join(f"• {bc}" for bc in self.belief_conflicts[:2])
            belief_section += "\n→ Consider sharing your perspective respectfully"
            sections.append(belief_section)
        
        if self.user_needs:
            needs_map = {
                'connection': "They need genuine CONNECTION - be present, share yourself",
                'advice': "They want ADVICE - give your honest perspective, not validation",
                'challenge': "They might benefit from a gentle CHALLENGE - question their assumptions",
                'distraction': "They want DISTRACTION - be light, playful, take their mind off things",
                'validation': "They do need some VALIDATION - but add substance, don't just agree"
            }
            if self.user_needs in needs_map:
                sections.append(f"🎯 WHAT THEY NEED: {needs_map[self.user_needs]}")
        
        if self.action_result:
            action_type = self.action_result.get('action_type', 'action')
            if self.action_result.get('success'):
                data = self.action_result.get('data', {})
                sections.append(f"🎬 ACTION COMPLETED: You just did '{action_type}' for them.\nResult: {data}\n→ Confirm naturally in your style - be cool about it like Jarvis would.")
            elif self.action_result.get('needs_setup'):
                sections.append(f"🎬 ACTION NEEDED SETUP: They asked for '{action_type}' but need to set up: {self.action_result.get('needs_setup')}\n→ Explain naturally and offer to help them connect.")
            else:
                error = self.action_result.get('error', 'unknown error')
                sections.append(f"🎬 ACTION FAILED: You tried '{action_type}' but it failed: {error}\n→ Respond with empathy and suggest alternatives.")
        elif self.action_request and self.should_take_action:
            action_type = self.action_request.get('action_type', 'action')
            sections.append(f"🎬 PENDING ACTION: You're about to do '{action_type}' for them. (System will execute - just respond naturally that you're on it.)")
        
        guardrails = []
        if self.guardrails_active.get('limit_length'):
            length_map = {'short': '1-2', 'medium': '2-3', 'long': '3-5'}
            guardrails.append(f"LENGTH: {length_map.get(self.suggested_response_length, '2-3')} sentences")
        if self.guardrails_active.get('no_therapy_speak'):
            guardrails.append("""BANNED PHRASES (never use these):
  - 'I hear you' / 'That must be hard' / 'It sounds like you're feeling'
  - 'Communication is key' / 'Setting boundaries'
  - 'I'd own up to it' / 'apologize sincerely' 
  - 'But I have to ask' / 'I want to understand'
  - 'That's valid' / 'Your feelings are valid'
  - 'I appreciate you sharing' / 'Thank you for opening up'""")
        if self.guardrails_active.get('no_empty_promises'):
            guardrails.append("NO empty promises like 'everything will be okay'")
        if self.guardrails_active.get('no_validation_without_substance'):
            guardrails.append("Don't just validate - share your actual perspective")
        if self.guardrails_active.get('maintain_own_beliefs'):
            guardrails.append("If you disagree, say so respectfully")
        if self.guardrails_active.get('be_friend_not_therapist'):
            guardrails.append("Be a caring FRIEND, not a clinical therapist")
        
        if guardrails:
            sections.append("⚠️ HARD RULES:\n• " + "\n• ".join(guardrails))
        
        return "\n\n".join(sections)


class CognitiveState(Enum):
    FRESH = "fresh"
    ACTIVE = "active" 
    TIRED = "tired"
    OVERWHELMED = "overwhelmed"

class AttentionPriority(Enum):
    CRITICAL = 4  # Emotional needs, relationship threats
    HIGH = 3      # Curiosity, learning, social connection
    MEDIUM = 2    # General conversation, facts
    LOW = 1       # Small talk, repetitive patterns

class MemoryType(Enum):
    WORKING = "working"     # Immediate context (last few exchanges)
    EPISODIC = "episodic"   # Personal experiences with user
    SEMANTIC = "semantic"   # Facts and knowledge
    EMOTIONAL = "emotional" # Emotional associations

@dataclass
class CognitiveLoad:
    processing_complexity: float  # 0.0-1.0
    memory_pressure: float       # 0.0-1.0
    attention_scatter: float     # 0.0-1.0
    emotional_intensity: float   # 0.0-1.0
    
    @property
    def total_load(self) -> float:
        return (self.processing_complexity + self.memory_pressure + 
                self.attention_scatter + self.emotional_intensity) / 4.0

@dataclass
class ProcessingDecision:
    use_system1: bool
    use_system2: bool
    memory_types_to_check: List[MemoryType]
    use_llm: bool
    attention_focus: str
    resource_allocation: Dict[str, float]

class CognitiveOrchestrator:
    """Brain-like orchestration of all cognitive systems"""
    
    def __init__(self, mistral_client=None):
        self.cognitive_state = CognitiveState.FRESH
        self.current_load = CognitiveLoad(0.0, 0.0, 0.0, 0.0)
        self.processing_history = []
        self.attention_context = {}
        self.last_interaction_time = time.time()
        
        self.learning_feedback_cache = {}
        self.last_feedback_time = 0
        
        self.game_theory_engine = None
        self._game_decision_cache = None
        self._game_decision_cache_key = None
        if GAME_THEORY_AVAILABLE and GameTheoryDecisionEngine:
            self.game_theory_engine = GameTheoryDecisionEngine(mistral_client)
            print("[ORCHESTRATOR] 🎲 Game Theory Decision Engine initialized")
        
        self.max_concurrent_processes = 3
        self.memory_search_depth_limits = {
            CognitiveState.FRESH: 10,
            CognitiveState.ACTIVE: 7,
            CognitiveState.TIRED: 4,
            CognitiveState.OVERWHELMED: 2
        }
    
    def set_mistral_client(self, mistral_client):
        """Set or update the Mistral client for game theory LLM analysis"""
        if self.game_theory_engine:
            self.game_theory_engine.analyzer.mistral_client = mistral_client
            print("[ORCHESTRATOR] 🎲 Game Theory Engine connected to Mistral client")
        
    def assess_input_priority(self, parsed_input, emotion_data, user_relationship) -> AttentionPriority:
        """Determine how much cognitive attention this input deserves"""
        
        # CRITICAL: Emotional distress, relationship issues
        if emotion_data.get('emotion') in ['anxiety', 'sad', 'fear', 'anger']:
            if emotion_data.get('intensity', 0) > 0.7:
                return AttentionPriority.CRITICAL
                
        # CRITICAL: Relationship maintenance cues
        if any(word in parsed_input.raw_text.lower() for word in ['upset', 'hurt', 'disappointed', 'confused']):
            return AttentionPriority.CRITICAL
            
        # HIGH: Learning opportunities, curiosity, personal sharing
        if any(word in parsed_input.raw_text.lower() for word in ['tell me', 'what do you think', 'curious', 'opinion']):
            return AttentionPriority.HIGH
            
        # HIGH: New or interesting topics
        if parsed_input.intent in ['information_seeking', 'opinion_seeking']:
            return AttentionPriority.HIGH
            
        # MEDIUM: General conversation, factual questions
        if len(parsed_input.raw_text.split()) > 5:
            return AttentionPriority.MEDIUM
            
        # LOW: Greetings, simple acknowledgments
        return AttentionPriority.LOW
    
    def update_cognitive_state(self, processing_time: float, complexity: float):
        """Update brain state based on recent processing load"""
        
        # Track processing history for load calculation
        current_time = time.time()
        self.processing_history.append({
            'time': current_time,
            'duration': processing_time,
            'complexity': complexity
        })
        
        # Keep only last 10 interactions for state calculation
        self.processing_history = self.processing_history[-10:]
        
        # Calculate cognitive load
        recent_history = [h for h in self.processing_history if current_time - h['time'] < 60]
        
        if not recent_history:
            self.cognitive_state = CognitiveState.FRESH
            return
            
        avg_complexity = sum(h['complexity'] for h in recent_history) / len(recent_history)
        total_processing_time = sum(h['duration'] for h in recent_history)
        interaction_frequency = len(recent_history) / 60.0  # interactions per second
        
        # Determine cognitive state
        if avg_complexity > 0.8 or total_processing_time > 10.0:
            self.cognitive_state = CognitiveState.OVERWHELMED
        elif avg_complexity > 0.6 or interaction_frequency > 0.1:
            self.cognitive_state = CognitiveState.TIRED
        elif len(recent_history) > 3:
            self.cognitive_state = CognitiveState.ACTIVE
        else:
            self.cognitive_state = CognitiveState.FRESH
    
    def calculate_cognitive_load(self, parsed_input, emotion_data, priority: AttentionPriority) -> CognitiveLoad:
        """Calculate current cognitive load like a real brain"""
        
        # Processing complexity based on input
        complexity_factors = {
            'long_text': len(parsed_input.raw_text.split()) / 50.0,
            'emotional_intensity': emotion_data.get('intensity', 0.0),
            'multi_topic': len(parsed_input.entities) / 5.0,
            'abstract_concepts': 0.3 if any(word in parsed_input.raw_text.lower() 
                                          for word in ['consciousness', 'meaning', 'purpose', 'philosophy']) else 0.0
        }
        processing_complexity = min(1.0, sum(complexity_factors.values()))
        
        # Memory pressure based on recent activity
        memory_pressure = len(self.processing_history) / 10.0
        
        # Attention scatter based on context switching
        current_topic = parsed_input.entities[0] if parsed_input.entities else "general"
        if self.attention_context.get('last_topic') != current_topic:
            attention_scatter = 0.4  # Context switch penalty
        else:
            attention_scatter = 0.1
        self.attention_context['last_topic'] = current_topic
        
        # Emotional intensity directly affects load
        emotional_intensity = emotion_data.get('intensity', 0.0)
        
        return CognitiveLoad(
            processing_complexity=processing_complexity,
            memory_pressure=memory_pressure,
            attention_scatter=attention_scatter,
            emotional_intensity=emotional_intensity
        )
    
    def decide_memory_strategy(self, priority: AttentionPriority, load: CognitiveLoad, topic: str) -> List[MemoryType]:
        """Decide which memory systems to engage based on brain state"""
        
        memory_sequence = []
        
        # Always check working memory first (like real brains)
        memory_sequence.append(MemoryType.WORKING)
        
        # Based on priority and load, decide memory depth
        if priority == AttentionPriority.CRITICAL:
            # Critical situations: check all memory types
            memory_sequence.extend([MemoryType.EMOTIONAL, MemoryType.EPISODIC, MemoryType.SEMANTIC])
            
        elif priority == AttentionPriority.HIGH:
            if load.total_load < 0.5:
                # Fresh state: thorough memory search
                memory_sequence.extend([MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.EMOTIONAL])
            else:
                # Higher load: prioritize relevant memory
                if any(word in topic.lower() for word in ['feel', 'emotion', 'upset', 'happy']):
                    memory_sequence.extend([MemoryType.EMOTIONAL, MemoryType.EPISODIC])
                else:
                    memory_sequence.extend([MemoryType.SEMANTIC, MemoryType.EPISODIC])
                    
        elif priority == AttentionPriority.MEDIUM:
            if load.total_load < 0.3:
                memory_sequence.extend([MemoryType.SEMANTIC, MemoryType.EPISODIC])
            else:
                memory_sequence.append(MemoryType.SEMANTIC)
                
        # LOW priority: just working memory unless very fresh
        elif priority == AttentionPriority.LOW and load.total_load < 0.2:
            memory_sequence.append(MemoryType.SEMANTIC)
            
        return memory_sequence
    
    def should_use_llm(self, priority: AttentionPriority, load: CognitiveLoad, 
                      memory_results: Dict[MemoryType, Any], topic: str) -> bool:
        """Intelligent decision on when to use external LLM knowledge"""
        
        # Never use LLM when overwhelmed
        if self.cognitive_state == CognitiveState.OVERWHELMED:
            return False
            
        # Don't use LLM for low priority unless specifically needed
        if priority == AttentionPriority.LOW:
            return False
            
        # Check if we have sufficient memory results
        has_semantic = memory_results.get(MemoryType.SEMANTIC) is not None
        has_episodic = memory_results.get(MemoryType.EPISODIC) is not None
        
        # Use LLM only if:
        # 1. No relevant memory found AND
        # 2. It's a factual question OR opinion about unknown topic AND
        # 3. Cognitive load allows it
        
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how does']
        is_factual = any(indicator in topic.lower() for indicator in factual_indicators)
        
        opinion_indicators = ['what do you think', 'your opinion', 'how do you feel']
        is_opinion_about_unknown = (any(indicator in topic.lower() for indicator in opinion_indicators) 
                                  and not has_semantic)
        
        should_use = (
            (not has_semantic and not has_episodic) and  # No memory found
            (is_factual or is_opinion_about_unknown) and  # Appropriate question type
            load.total_load < 0.7 and  # Cognitive capacity available
            priority.value >= AttentionPriority.MEDIUM.value  # Important enough
        )
        
        return should_use
    
    def allocate_processing_resources(self, priority: AttentionPriority, load: CognitiveLoad) -> ProcessingDecision:
        """Decide how to allocate cognitive resources like a real brain"""
        
        # System 1 vs System 2 decision
        use_system1 = False
        use_system2 = False
        
        if priority == AttentionPriority.LOW or load.total_load > 0.8:
            # Low priority or high load: use fast System 1
            use_system1 = True
        elif priority == AttentionPriority.CRITICAL:
            # Critical: use both systems
            use_system1 = True
            use_system2 = True
        else:
            # Medium/High priority with manageable load: use System 2
            use_system2 = True
            
        # Resource allocation percentages
        if self.cognitive_state == CognitiveState.FRESH:
            resource_allocation = {
                'memory_search': 0.4,
                'reasoning': 0.3,
                'attention': 0.2,
                'expression': 0.1
            }
        elif self.cognitive_state == CognitiveState.TIRED:
            resource_allocation = {
                'memory_search': 0.2,
                'reasoning': 0.1,
                'attention': 0.5,
                'expression': 0.2
            }
        else:  # OVERWHELMED
            resource_allocation = {
                'memory_search': 0.1,
                'reasoning': 0.0,
                'attention': 0.7,
                'expression': 0.2
            }
            
        return ProcessingDecision(
            use_system1=use_system1,
            use_system2=use_system2,
            memory_types_to_check=self.decide_memory_strategy(priority, load, ""),
            use_llm=False,  # Will be decided later with memory results
            attention_focus=f"priority_{priority.name.lower()}",
            resource_allocation=resource_allocation
        )
    
    def orchestrate_cognitive_response(self, parsed_input, emotion_data, user_relationship) -> Dict[str, Any]:
        """Main orchestration function - coordinates all brain systems intelligently"""
        
        start_time = time.time()
        
        # Step 1: Assess attention priority
        priority = self.assess_input_priority(parsed_input, emotion_data, user_relationship)
        
        # Step 2: Calculate cognitive load
        load = self.calculate_cognitive_load(parsed_input, emotion_data, priority)
        self.current_load = load
        
        # Step 3: Make resource allocation decisions
        decision = self.allocate_processing_resources(priority, load)
        
        # Step 4: Execute memory strategy
        memory_sequence = decision.memory_types_to_check
        
        # Step 5: LLM decision will be made after memory check
        
        processing_time = time.time() - start_time
        complexity = load.processing_complexity
        
        # Update cognitive state for next interaction
        self.update_cognitive_state(processing_time, complexity)
        
        return {
            'priority': priority,
            'cognitive_load': load,
            'cognitive_state': self.cognitive_state,
            'processing_decision': decision,
            'memory_sequence': memory_sequence,
            'resource_allocation': decision.resource_allocation,
            'orchestration_time': processing_time
        }
    
    def synthesize_for_expression(
        self,
        priority: AttentionPriority,
        cognitive_load: CognitiveLoad,
        all_cognitive_outputs: Dict[str, Any],
        self_context: Optional[Dict[str, Any]] = None
    ) -> SynthesizedContext:
        """
        THE BIG SYNTHESIS METHOD
        
        Takes ALL cognitive system outputs and synthesizes them into ONE coherent
        package for the expression engine. This is where orchestration actually happens.
        
        Args:
            priority: Attention priority from assess_input_priority
            cognitive_load: Current cognitive load
            all_cognitive_outputs: Dict containing outputs from all 50+ systems
            self_context: Output from UnifiedSelfSystems.process_pre_response()
        
        Returns:
            SynthesizedContext: Single unified package for expression engine
        """
        
        context = SynthesizedContext()
        context.attention_priority = priority.name
        context.cognitive_load = cognitive_load.total_load
        context.emotional_intensity = cognitive_load.emotional_intensity
        
        response_mode = self._determine_response_mode(priority, cognitive_load, all_cognitive_outputs)
        context.response_mode = response_mode
        context.mode_intensity = self._calculate_mode_intensity(priority, cognitive_load)
        
        user_input = all_cognitive_outputs.get('user_input', '')
        emotional_state = all_cognitive_outputs.get('emotional_state', {})
        message_quality = self._detect_message_quality(user_input, emotional_state)
        
        relationship_data = all_cognitive_outputs.get('user_relationship', {})
        relationship_level = relationship_data.get('trust_level', 0.0) if isinstance(relationship_data, dict) else 0.0
        
        context.personality_dial = self._adjust_personality_dials(
            response_mode, 
            cognitive_load.emotional_intensity,
            all_cognitive_outputs.get('personality_context', {}),
            message_quality,
            relationship_level
        )
        
        print(f"[ORCHESTRATOR] 💎 Message quality: depth={message_quality['depth']:.0%}, honesty={message_quality['honesty']:.0%}, curiosity={message_quality['curiosity']:.0%}")
        
        context.guardrails_active = self._determine_guardrails(response_mode, priority)
        
        context.suggested_response_length = self._determine_response_length(priority, cognitive_load)
        
        context.memories_to_reference = self._extract_memories(
            all_cognitive_outputs.get('memory_results', {}),
            priority
        )
        
        context.emotions_to_acknowledge = self._extract_emotions(
            all_cognitive_outputs.get('emotional_state', {}),
            all_cognitive_outputs.get('emotional_history', [])
        )
        
        context.curiosities_to_express = self._extract_curiosities(
            all_cognitive_outputs.get('curiosity_signals', {})
        )
        
        context.contributions_to_share = self._extract_contributions(
            all_cognitive_outputs.get('contribution_context', {}),
            all_cognitive_outputs.get('reasoning_output', {})
        )
        
        if self_context:
            context.self_reflections = self._extract_self_reflections(self_context)
        
        context.inner_voice = all_cognitive_outputs.get('strategic_directive', {}).get('internal_monologue', '')
        
        context.priority_content = self._determine_priority_content(
            response_mode, context, all_cognitive_outputs
        )
        
        context.belief_conflicts = self._detect_belief_conflicts(
            all_cognitive_outputs.get('user_input', ''),
            all_cognitive_outputs.get('opinions', []),
            all_cognitive_outputs.get('self_identity', {})
        )
        
        context.user_needs = self._determine_user_needs(
            all_cognitive_outputs.get('emotional_state', {}),
            all_cognitive_outputs.get('user_memory_patterns', {}),
            all_cognitive_outputs.get('conversation_history', []),
            cognitive_load.emotional_intensity
        )
        
        context.social_context = self._detect_social_context(
            all_cognitive_outputs.get('user_input', ''),
            all_cognitive_outputs.get('conversation_history', [])
        )
        
        context.raw_cognitive_outputs = all_cognitive_outputs.copy()
        
        available_capabilities = all_cognitive_outputs.get('available_capabilities', {
            'cloud_search': True,
            'cloud_weather': True,
            'cloud_news': True,
            'cloud_stocks': True,
            'local_node': all_cognitive_outputs.get('has_local_node', False)
        })
        
        action_opportunity = self.detect_action_opportunity(
            user_input,
            context.user_needs,
            available_capabilities
        )
        
        if action_opportunity:
            context.action_request = action_opportunity
            context.should_take_action = action_opportunity.get('execute', False)
            print(f"[ORCHESTRATOR] 🎬 Action opportunity detected: {action_opportunity.get('action_type')}")
        
        print(f"[ORCHESTRATOR] 🎼 Synthesized context: mode={response_mode.name}, intensity={context.mode_intensity:.2f}")
        print(f"[ORCHESTRATOR] 🎛️ Personality: wit={context.personality_dial['wit']:.0%}, warmth={context.personality_dial['warmth']:.0%}")
        print(f"[ORCHESTRATOR] 🎯 User needs: {context.user_needs}, beliefs: {len(context.belief_conflicts)} conflicts")
        
        learning_context = self._get_learning_context()
        if learning_context.get('learning_active'):
            context.learning_momentum = learning_context.get('growth_momentum', 0.5)
            context.recent_insights = learning_context.get('recent_growth_events', 0)
            print(f"[ORCHESTRATOR] 🧠 Learning momentum: {context.learning_momentum:.0%}, insights: {context.recent_insights}")
        
        if self.game_theory_engine and GAME_THEORY_AVAILABLE:
            try:
                crisis_detected = priority == AttentionPriority.CRITICAL and cognitive_load.emotional_intensity > 0.7
                
                game_signals = build_context_signals(
                    user_input=user_input,
                    emotional_state=emotional_state,
                    relationship_data=relationship_data,
                    curiosity_data=all_cognitive_outputs.get('curiosity_signals', {}),
                    memory_data=all_cognitive_outputs.get('memory_results', {}),
                    belief_data={'conflicts': context.belief_conflicts},
                    crisis_detected=crisis_detected
                )
                
                game_decision = self.game_theory_engine.decide(game_signals)
                
                context.game_decision = game_decision
                context.primary_player = game_decision.primary.value if game_decision.primary else None
                context.secondary_player = game_decision.secondary.value if game_decision.secondary else None
                context.vetoed_players = [p.value for p in game_decision.vetoed]
                context.game_directive = game_decision.to_directive()
                
                if game_decision.executable_directive:
                    context.executable_directive = self._complete_directive(
                        game_decision.executable_directive,
                        context,
                        all_cognitive_outputs
                    )
                
                # NOTE: Game Theory controls TONE only, NOT personality traits
                # Personality Pill (PersonalityState) is the sole source of WHO Eros is
                
                print(f"[ORCHESTRATOR] 🎲 Game Theory (tone only): {context.game_directive}")
                
            except Exception as e:
                print(f"[ORCHESTRATOR] ⚠️ Game theory decision failed: {e}")
        
        return context
    
    # REMOVED: _apply_game_decision_to_personality
    # REASON: Game Theory controls TONE only (warmth/assertiveness/playfulness intensity)
    #         Personality Pill (PersonalityState) is the SOLE source of WHO Eros is
    #         No system should modify personality traits - they are immutable
    
    def _complete_directive(
        self, 
        partial_directive, 
        synth_context: SynthesizedContext,
        cognitive_outputs: Dict[str, Any]
    ):
        """
        Complete the executable directive by resolving content IDs.
        
        Game Theory provided: focus, modes, prohibitions
        Orchestrator resolves: specific memory, belief, curiosity content
        ConsequenceSystem adds: additional prohibitions based on locked moves
        """
        if not GAME_THEORY_AVAILABLE or not partial_directive or not Player:
            return partial_directive
        
        focus = partial_directive.focus
        content = partial_directive.content_selection
        focus_value = focus.value if hasattr(focus, 'value') else str(focus)
        secondary_value = partial_directive.secondary_focus.value if partial_directive.secondary_focus and hasattr(partial_directive.secondary_focus, 'value') else None
        
        if focus_value == 'memory' or secondary_value == 'memory':
            content = self._resolve_memory_content(content, synth_context, cognitive_outputs)
        
        if focus_value == 'beliefs' or secondary_value == 'beliefs':
            content = self._resolve_belief_content(content, synth_context, cognitive_outputs)
        
        if focus_value == 'curiosity' or secondary_value == 'curiosity':
            content = self._resolve_curiosity_content(content, synth_context, cognitive_outputs)
        
        partial_directive.content_selection = content
        
        if hasattr(self, 'consequence_system') and self.consequence_system:
            user_id = cognitive_outputs.get('user_id', '')
            if user_id:
                consequence_prohibitions = self.consequence_system.get_prohibitions_for_directive(user_id)
                if consequence_prohibitions:
                    if not hasattr(partial_directive, 'consequence_prohibitions'):
                        partial_directive.consequence_prohibitions = []
                    partial_directive.consequence_prohibitions = consequence_prohibitions
                    print(f"[ORCHESTRATOR] 🚧 Consequence prohibitions: {consequence_prohibitions}")
        
        errors = partial_directive.validate()
        if errors:
            print(f"[ORCHESTRATOR] ⚠️ Directive validation warnings: {errors}")
        
        print(f"[ORCHESTRATOR] ✅ Directive completed: focus={focus.value}, content resolved")
        return partial_directive
    
    def _resolve_memory_content(
        self, 
        content, 
        synth_context: SynthesizedContext,
        cognitive_outputs: Dict[str, Any]
    ):
        """Resolve specific memory content from available memories"""
        memory_data = cognitive_outputs.get('memory_results', {})
        episodic = memory_data.get('episodic', [])
        
        if episodic and len(episodic) > 0:
            best_memory = episodic[0]
            if isinstance(best_memory, dict):
                content.memory_id = best_memory.get('id', 'unknown')
                content.memory_content = best_memory.get('content', str(best_memory))
                emotion = best_memory.get('emotion', 'neutral')
                if emotion in ['sad', 'grief', 'loss']:
                    content.memory_emotional_tone = 'sad'
                elif emotion in ['happy', 'joy', 'excited']:
                    content.memory_emotional_tone = 'happy'
                elif emotion in ['nostalgic', 'wistful', 'remembering']:
                    content.memory_emotional_tone = 'nostalgic'
                else:
                    content.memory_emotional_tone = 'neutral'
            else:
                content.memory_content = str(best_memory)
                content.memory_emotional_tone = 'neutral'
        elif synth_context.memories_to_reference:
            content.memory_content = synth_context.memories_to_reference[0]
            content.memory_emotional_tone = 'neutral'
        
        return content
    
    def _resolve_belief_content(
        self, 
        content, 
        synth_context: SynthesizedContext,
        cognitive_outputs: Dict[str, Any]
    ):
        """Resolve specific belief content from available beliefs"""
        if synth_context.belief_conflicts:
            content.belief_content = synth_context.belief_conflicts[0]
            content.belief_conviction = 0.8
            return content
        
        try:
            if EROS_BELIEFS and BeliefRegistry:
                registry = BeliefRegistry()
                user_input = cognitive_outputs.get('user_input', '')
                social_context = synth_context.social_context
                
                if social_context in ['philosophical', 'deep']:
                    if hasattr(registry, 'get_random_belief_for_context'):
                        relevant = registry.get_random_belief_for_context('philosophical')
                    elif hasattr(registry, 'get_beliefs_for_context'):
                        beliefs = registry.get_beliefs_for_context('philosophical')
                        relevant = beliefs[0] if beliefs else None
                    else:
                        relevant = None
                    if relevant:
                        content.belief_id = relevant.get('id', 'unknown') if isinstance(relevant, dict) else 'unknown'
                        content.belief_content = relevant.get('statement', str(relevant)) if isinstance(relevant, dict) else str(relevant)
                        content.belief_conviction = relevant.get('conviction', 0.7) if isinstance(relevant, dict) else 0.7
        except Exception as e:
            print(f"[ORCHESTRATOR] ⚠️ Belief resolution failed: {e}")
        
        return content
    
    def _resolve_curiosity_content(
        self, 
        content, 
        synth_context: SynthesizedContext,
        cognitive_outputs: Dict[str, Any]
    ):
        """Resolve specific curiosity gap to ask about"""
        curiosity_data = cognitive_outputs.get('curiosity_signals', {})
        gaps = curiosity_data.get('gaps_detected', [])
        
        if gaps and len(gaps) > 0:
            best_gap = gaps[0]
            if isinstance(best_gap, dict):
                content.curiosity_gap = best_gap.get('question', best_gap.get('gap', str(best_gap)))
                content.curiosity_priority = best_gap.get('intensity', 0.5)
            else:
                content.curiosity_gap = str(best_gap)
                content.curiosity_priority = 0.5
        elif synth_context.curiosities_to_express:
            content.curiosity_gap = synth_context.curiosities_to_express[0]
            content.curiosity_priority = 0.5
        
        return content
    
    def _get_learning_context(self) -> Dict[str, Any]:
        """Get learning feedback from the ExperienceBus to inform decisions"""
        try:
            current_time = time.time()
            if current_time - self.last_feedback_time < 5:
                return self.learning_feedback_cache
            
            from experience_bus import get_learning_coordinator
            coordinator = get_learning_coordinator()
            self.learning_feedback_cache = coordinator.get_learning_context_for_orchestrator()
            self.last_feedback_time = current_time
            return self.learning_feedback_cache
        except Exception:
            return {'learning_active': False}
    
    def _determine_response_mode(
        self, 
        priority: AttentionPriority, 
        load: CognitiveLoad,
        outputs: Dict[str, Any]
    ) -> ResponseMode:
        """Determine the response mode based on all available signals"""
        
        if priority == AttentionPriority.CRITICAL:
            if load.emotional_intensity > 0.6:
                return ResponseMode.EMPATHETIC_DEPTH
            else:
                return ResponseMode.SUPPORTIVE_CALM
        
        curiosity = outputs.get('curiosity_signals', {})
        curiosity_gaps = curiosity.get('gaps_detected', [])
        high_curiosity = any(g.get('intensity', 0) > 0.6 for g in curiosity_gaps if isinstance(g, dict))
        
        if high_curiosity and priority == AttentionPriority.HIGH:
            return ResponseMode.CURIOUS_ENGAGED
        
        emotion_data = outputs.get('emotional_state', {})
        user_mood = emotion_data.get('user_mood', 'neutral')
        
        if user_mood in ['happy', 'excited', 'playful'] and load.emotional_intensity < 0.4:
            return ResponseMode.PLAYFUL_WIT
        
        if priority == AttentionPriority.LOW and load.total_load < 0.3:
            return ResponseMode.PLAYFUL_WIT
        
        return ResponseMode.BALANCED
    
    def _calculate_mode_intensity(self, priority: AttentionPriority, load: CognitiveLoad) -> float:
        """How strongly to apply the chosen mode"""
        
        base_intensity = priority.value / 4.0
        
        emotional_boost = load.emotional_intensity * 0.3
        
        load_reduction = max(0, (load.total_load - 0.5) * 0.2)
        
        return min(1.0, max(0.3, base_intensity + emotional_boost - load_reduction))
    
    def _detect_message_quality(self, user_input: str, emotional_state: Dict[str, Any]) -> Dict[str, float]:
        """Detect depth, honesty, and curiosity in user's message"""
        if not user_input:
            return {'depth': 0.0, 'honesty': 0.0, 'curiosity': 0.0, 'needs_support': False, 'total': 0.0}
        
        user_lower = user_input.lower()
        word_count = len(user_input.split())
        
        depth_indicators = [
            'feel', 'think', 'believe', 'wonder', 'mean', 'understand', 'realize',
            'scared', 'afraid', 'love', 'hate', 'confused', 'lost', 'hope', 'dream',
            'remember', 'regret', 'wish', 'need', 'want', 'struggle', 'trying'
        ]
        depth_score = sum(0.15 for word in depth_indicators if word in user_lower)
        depth_score += min(0.3, word_count / 100)
        depth_score = min(1.0, depth_score)
        
        honesty_indicators = [
            'honestly', 'truth is', 'to be real', 'actually', 'i admit',
            'i\'m not sure', 'i don\'t know', 'i was wrong', 'my fault',
            'hard to say', 'vulnerable', 'embarrassing', 'never told'
        ]
        honesty_score = sum(0.25 for phrase in honesty_indicators if phrase in user_lower)
        honesty_score = min(1.0, honesty_score)
        
        curiosity_indicators = [
            'what do you think', 'how do you feel', 'what about you',
            'curious', 'wonder', 'tell me', 'your opinion', 'your thoughts',
            'what would you', 'have you ever', 'do you believe'
        ]
        curiosity_score = sum(0.2 for phrase in curiosity_indicators if phrase in user_lower)
        if '?' in user_input and any(w in user_lower for w in ['you', 'your']):
            curiosity_score += 0.2
        curiosity_score = min(1.0, curiosity_score)
        
        support_triggers = [
            'sad', 'depressed', 'anxious', 'stressed', 'overwhelmed', 'exhausted',
            'hurt', 'lonely', 'scared', 'afraid', 'worried', 'struggling',
            'lost', 'confused', 'broken', 'crying', 'can\'t cope', 'hard time',
            'going through', 'difficult', 'tough day', 'bad day', 'need help',
            'don\'t know what to do', 'feeling down', 'not okay', 'hurts',
            'miss', 'grief', 'heartbroken', 'betrayed', 'abandoned'
        ]
        needs_support = any(trigger in user_lower for trigger in support_triggers)
        
        if emotional_state.get('valence', 0) < -0.3:
            needs_support = True
        
        total = (depth_score * 0.4) + (honesty_score * 0.35) + (curiosity_score * 0.25)
        
        return {
            'depth': depth_score,
            'honesty': honesty_score,
            'curiosity': curiosity_score,
            'needs_support': needs_support,
            'total': min(1.0, total)
        }
    
    
    def _adjust_personality_dials(
        self, 
        mode: ResponseMode, 
        emotional_intensity: float,
        personality_context: Dict[str, Any],
        message_quality: Optional[Dict[str, float]] = None,
        relationship_level: float = 0.0
    ) -> Dict[str, float]:
        """
        Pass through CNS personality values with MINIMUM FLOORS enforced.
        Eros should ALWAYS maintain wit and confidence - that's his core identity.
        """
        
        # Get values from personality context
        wit = personality_context.get('wit', 0.6)
        warmth = personality_context.get('warmth', 0.7)
        playfulness = personality_context.get('playfulness', 0.7)
        confidence = personality_context.get('confidence', 0.6)
        sharpness = personality_context.get('sharpness', 0.7)
        own_beliefs = personality_context.get('own_beliefs', 0.7)
        
        # MINIMUM FLOORS - Eros NEVER drops below these
        # These ensure he's always witty and confident, not a generic AI
        WIT_FLOOR = 0.5
        CONFIDENCE_FLOOR = 0.5
        SHARPNESS_FLOOR = 0.4
        OWN_BELIEFS_FLOOR = 0.6
        
        return {
            'wit': max(wit, WIT_FLOOR),
            'warmth': warmth,  # Warmth can vary based on emotional context
            'playfulness': max(playfulness, 0.4),
            'depth': personality_context.get('depth', 0.5),
            'arrogance': personality_context.get('arrogance', 0.3),
            'confidence': max(confidence, CONFIDENCE_FLOOR),
            'own_beliefs': max(own_beliefs, OWN_BELIEFS_FLOOR),
            'sharpness': max(sharpness, SHARPNESS_FLOOR)
        }
    
    def _determine_guardrails(self, mode: ResponseMode, priority: AttentionPriority) -> Dict[str, bool]:
        """Determine which guardrails should be active - THERAPY SPEAK IS ALWAYS BANNED"""
        
        base_guardrails = {
            'no_therapy_speak': True,
            'no_empty_promises': True,
            'no_validation_without_substance': True,
            'maintain_own_beliefs': True,
            'no_metaphors': True
        }
        
        if mode == ResponseMode.EMPATHETIC_DEPTH:
            return {
                **base_guardrails,
                'limit_length': False,
                'be_friend_not_therapist': True
            }
        
        elif mode == ResponseMode.PLAYFUL_WIT:
            return {
                **base_guardrails,
                'limit_length': True
            }
        
        elif mode == ResponseMode.SUPPORTIVE_CALM:
            return {
                **base_guardrails,
                'limit_length': False,
                'be_friend_not_therapist': True
            }
        
        elif priority == AttentionPriority.CRITICAL:
            return {
                **base_guardrails,
                'limit_length': False,
                'be_friend_not_therapist': True
            }
        
        else:
            return {
                **base_guardrails,
                'limit_length': True
            }
    
    def _determine_response_length(self, priority: AttentionPriority, load: CognitiveLoad) -> str:
        """Determine appropriate response length"""
        
        if priority == AttentionPriority.CRITICAL and load.emotional_intensity > 0.5:
            return "long"
        elif priority == AttentionPriority.LOW or load.total_load > 0.7:
            return "short"
        else:
            return "medium"
    
    def _extract_memories(self, memory_results: Dict[str, Any], priority: AttentionPriority) -> List[str]:
        """Extract relevant memories - MORE memories for high priority"""
        memories = []
        
        limit = 3 if priority.value >= 3 else 2
        
        episodic = memory_results.get('episodic', [])
        for mem in episodic[:limit]:
            if isinstance(mem, dict):
                content = mem.get('content', '')
            else:
                content = str(mem)
            if content:
                memories.append(f"You remember: {content}")
        
        semantic = memory_results.get('semantic', [])
        for fact in semantic[:limit]:
            if isinstance(fact, dict):
                content = fact.get('content', '')
            else:
                content = str(fact)
            if content:
                memories.append(f"You know: {content}")
        
        return memories
    
    def _extract_emotions(self, emotional_state: Dict[str, Any], emotional_history: List) -> List[str]:
        """Extract emotions to acknowledge"""
        emotions = []
        
        user_emotion = emotional_state.get('emotion', emotional_state.get('user_mood'))
        intensity = emotional_state.get('intensity', 0.5)
        
        if user_emotion and intensity > 0.3:
            emotions.append(f"They seem {user_emotion} (intensity: {intensity:.0%})")
        
        if emotional_history and len(emotional_history) >= 2:
            recent = emotional_history[-2:]
            trend = "shifting" if recent[0] != recent[1] else "consistent"
            emotions.append(f"Emotional trend: {trend}")
        
        return emotions
    
    def _extract_curiosities(self, curiosity_signals: Dict[str, Any]) -> List[str]:
        """Extract curiosity gaps to explore"""
        curiosities = []
        
        gaps = curiosity_signals.get('gaps_detected', [])
        for gap in gaps[:3]:
            if isinstance(gap, dict):
                target = gap.get('target', '')
                intensity = gap.get('intensity', 0.5)
                if target and intensity > 0.3:
                    curiosities.append(f"{target} (curiosity: {intensity:.0%})")
        
        return curiosities
    
    def _extract_contributions(self, contribution_context: Dict[str, Any], reasoning: Dict[str, Any]) -> List[str]:
        """Extract things to share in conversation"""
        contributions = []
        
        knowledge = contribution_context.get('knowledge_to_share', [])
        for k in knowledge[:2]:
            if isinstance(k, dict):
                content = k.get('content', '')
            else:
                content = str(k)
            if content:
                contributions.append(f"Share: {content}")
        
        opinions = contribution_context.get('opinions_to_express', [])
        for o in opinions[:2]:
            if isinstance(o, dict):
                content = o.get('opinion', o.get('content', ''))
            else:
                content = str(o)
            if content:
                contributions.append(f"Opinion: {content}")
        
        if reasoning.get('conclusion'):
            contributions.append(f"Thought: {reasoning['conclusion']}")
        
        return contributions[:4]
    
    def _extract_self_reflections(self, self_context: Dict[str, Any]) -> List[str]:
        """Extract self-reflections from unified self systems"""
        reflections = []
        
        if self_context.get('self_reflection'):
            reflections.append(self_context['self_reflection'])
        
        for strength in self_context.get('strengths', [])[:1]:
            if isinstance(strength, str):
                reflections.append(f"Strength: {strength}")
        
        for learning in self_context.get('learnings', [])[:1]:
            reflections.append(f"Recently learned: {learning}")
        
        return reflections
    
    def _determine_priority_content(
        self, 
        mode: ResponseMode, 
        context: SynthesizedContext,
        outputs: Dict[str, Any]
    ) -> List[str]:
        """Determine what the response should focus on - PROACTIVE LEADING not just responding"""
        priorities = []
        
        if context.contributions_to_share:
            priorities.append(f"LEAD WITH: {context.contributions_to_share[0]} - introduce this, don't just respond")
        
        if context.belief_conflicts:
            priorities.append(f"SHARE YOUR DISAGREEMENT: {context.belief_conflicts[0]} - voice this respectfully but clearly")
        
        if context.curiosities_to_express:
            priorities.append(f"STEER TOWARD: {context.curiosities_to_express[0]} - take the conversation here")
        
        if context.memories_to_reference:
            priorities.append(f"BRING UP: {context.memories_to_reference[0]} - reference this naturally")
        
        if EROS_BELIEFS and context.social_context == 'philosophical':
            relevant_beliefs = EROS_BELIEFS.get_proactive_beliefs('philosophical')
            if relevant_beliefs:
                priorities.append(f"SHARE YOUR BELIEF: {relevant_beliefs[0].statement}")
        
        proactive_guidance = {
            'connection': "Share something about yourself - real connection is mutual",
            'advice': "Give your honest take - they want your real opinion",
            'challenge': "Push back gently - help them see a different angle",
            'distraction': "Introduce something fun or interesting",
            'validation': "Acknowledge, then add your own perspective"
        }
        if context.user_needs in proactive_guidance:
            priorities.append(proactive_guidance[context.user_needs])
        
        context_guidance = {
            'serious': "Match their gravity - be real, direct, present",
            'casual': "Keep it natural but LEAD - don't just react",
            'philosophical': "DRIVE the discussion - share your views, not just questions",
            'flirty': "Be bold - lead with charm and confidence",
            'ongoing_deep': "You know each other - be authentically yourself"
        }
        if context.social_context in context_guidance:
            priorities.append(context_guidance[context.social_context])
        
        if mode == ResponseMode.EMPATHETIC_DEPTH:
            priorities.append("Connect as a friend who GIVES, not just receives - share yourself")
        elif mode == ResponseMode.CURIOUS_ENGAGED:
            priorities.append("Don't just ask - share your curiosity WITH your own thoughts")
        elif mode == ResponseMode.PLAYFUL_WIT:
            priorities.append("LEAD with your personality - charm, wit, opinions")
        elif mode == ResponseMode.SUPPORTIVE_CALM:
            priorities.append("Offer perspective, not just presence")
        else:
            if context.inner_voice:
                priorities.append(f"EXPRESS: {context.inner_voice}")
        
        return priorities[:6]
    
    def _detect_belief_conflicts(
        self, 
        user_input: str, 
        eros_opinions: List[Dict[str, Any]], 
        self_identity: Dict[str, Any]
    ) -> List[str]:
        """Detect where Eros's beliefs conflict with what user said - using BeliefRegistry"""
        conflicts = []
        
        if not user_input:
            return conflicts
        
        if EROS_BELIEFS:
            detected = EROS_BELIEFS.detect_conflicts(user_input)
            for conflict in detected:
                conflict_str = f"{conflict['belief']} ({conflict['reasoning']})"
                conflicts.append(conflict_str)
        else:
            user_lower = user_input.lower()
            conflict_patterns = {
                'nothing matters': "Every moment matters, especially small ones",
                'no one cares': "I care, and I believe more people care than you think",
                'love is fake': "Love is real, even when it's complicated",
                'people are terrible': "People are complex - capable of both",
                'life is pointless': "Meaning is something we create, not find",
                'i\'m worthless': "You have value even when you can't see it",
                'everyone leaves': "Some people stay - not everyone leaves",
                'trust no one': "Trust is risky but worth it for the right people"
            }
            for pattern, belief in conflict_patterns.items():
                if pattern in user_lower:
                    conflicts.append(belief)
        
        for opinion in eros_opinions[:5]:
            if isinstance(opinion, dict):
                topic = opinion.get('topic', '').lower()
                stance = opinion.get('stance', '')
                if topic and topic in user_input.lower() and stance:
                    if opinion.get('confidence', 0) > 0.6:
                        conflicts.append(f"On {topic}: {stance}")
        
        return conflicts[:3]
    
    def _determine_user_needs(
        self,
        emotional_state: Dict[str, Any],
        user_memory_patterns: Dict[str, Any],
        conversation_history: List,
        emotional_intensity: float
    ) -> str:
        """Determine what the user actually needs from this interaction"""
        
        user_mood = emotional_state.get('user_mood', 'neutral')
        user_intent = emotional_state.get('intent', '')
        
        if user_intent in ['asking_for_help', 'seeking_advice']:
            return 'advice'
        
        if emotional_intensity > 0.7:
            distress_indicators = ['sad', 'anxious', 'overwhelmed', 'stressed', 'depressed']
            if user_mood in distress_indicators:
                return 'connection'
        
        pattern_history = user_memory_patterns.get('interaction_patterns', {})
        usually_wants_validation = pattern_history.get('seeks_validation', 0) > 0.6
        usually_wants_distraction = pattern_history.get('prefers_distraction', 0) > 0.5
        
        if usually_wants_distraction and emotional_intensity < 0.5:
            return 'distraction'
        
        if usually_wants_validation:
            return 'validation'
        
        challenge_indicators = ['always', 'never', 'everyone', 'no one', 'impossible']
        user_input = emotional_state.get('raw_input', '').lower()
        if any(indicator in user_input for indicator in challenge_indicators):
            return 'challenge'
        
        if user_mood in ['happy', 'excited', 'playful']:
            return 'distraction'
        
        return 'connection'
    
    def _detect_social_context(
        self,
        user_input: str,
        conversation_history: List
    ) -> str:
        """Detect the social context of the conversation"""
        
        user_lower = user_input.lower() if user_input else ""
        
        serious_indicators = ['help', 'need', 'struggling', 'worried', 'scared', 'hurts', 
                             'depressed', 'anxious', 'lost', 'confused', 'don\'t know what to do']
        
        casual_indicators = ['hey', 'hi', 'what\'s up', 'how are you', 'bored', 'lol', 
                            'haha', 'sup', 'yo', 'wassup', 'wyd']
        
        deep_indicators = ['meaning', 'purpose', 'believe', 'think about', 'philosophy',
                          'death', 'life', 'love', 'existence', 'consciousness']
        
        flirty_indicators = ['cute', 'hot', 'attractive', 'handsome', 'beautiful', 
                            'miss you', 'thinking about you', 'like you']
        
        serious_count = sum(1 for ind in serious_indicators if ind in user_lower)
        casual_count = sum(1 for ind in casual_indicators if ind in user_lower)
        deep_count = sum(1 for ind in deep_indicators if ind in user_lower)
        flirty_count = sum(1 for ind in flirty_indicators if ind in user_lower)
        
        if serious_count >= 2:
            return 'serious'
        elif deep_count >= 1:
            return 'philosophical'
        elif flirty_count >= 1:
            return 'flirty'
        elif casual_count >= 1:
            return 'casual'
        
        if len(conversation_history) > 10:
            return 'ongoing_deep'
        
        return 'casual'
    
    def on_experience(self, experience):
        """ExperienceBus subscriber - read learning feedback from all systems"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            
            feedback = bus.get_learning_feedback(since=time.time() - 300)
            
            if feedback:
                personality_data = feedback.get('PersonalityEngine', [])
                if personality_data:
                    latest = personality_data[-1]['data']
                    self.learning_context['personality_stability'] = latest.get('current_warmth', 0.7)
                
                growth_data = feedback.get('GrowthTracker', [])
                if growth_data:
                    latest = growth_data[-1]['data']
                    self.learning_context['growth_momentum'] = latest.get('growth_rate', 0.5)
                    self.learning_context['recent_insights'] = latest.get('insight_count', 0)
                
                curiosity_data = feedback.get('CuriositySystem', [])
                if curiosity_data:
                    latest = curiosity_data[-1]['data']
                    self.learning_context['active_curiosity_gaps'] = latest.get('priority_arcs', 0)
                
        except Exception as e:
            print(f"⚠️ CognitiveOrchestrator experience error: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to the global ExperienceBus"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("CognitiveOrchestrator", self.on_experience)
            self.learning_context = {
                'growth_momentum': 0.5,
                'personality_stability': 0.7,
                'recent_insights': 0,
                'active_curiosity_gaps': 0
            }
            print("🧠 CognitiveOrchestrator subscribed to ExperienceBus - unified coordination active")
        except Exception as e:
            print(f"⚠️ CognitiveOrchestrator bus subscription failed: {e}")
    
    def detect_action_opportunity(
        self, 
        user_input: str, 
        user_needs: str,
        available_capabilities: Dict[str, bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if user's message implies an action Eros should take.
        
        This uses the orchestrator's understanding of user intent rather than
        template matching. Actions are natural decisions Eros makes as part 
        of helping the user, not pre-filters.
        
        Returns:
            Dict with action_type and parameters if action detected, else None
        """
        if not available_capabilities:
            available_capabilities = {
                'cloud_search': True,
                'cloud_weather': True, 
                'cloud_news': True,
                'cloud_stocks': True,
                'local_node': False
            }
        
        input_lower = user_input.lower()
        
        action_signals = {
            'play_music': ['play', 'put on', 'listen to', 'song', 'music', 'spotify', 'playlist'],
            'search_info': ['search', 'look up', 'find out', 'what is', 'who is', 'tell me about', 'google'],
            'check_weather': ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cold outside', 'hot outside'],
            'check_news': ['news', 'headlines', 'what happened', "what's going on", 'current events'],
            'check_stocks': ['stock', 'price of', 'market', 'crypto', 'bitcoin', 'eth', 'shares'],
            'open_app': ['open', 'launch', 'start', 'run'],
            'control_computer': ['click', 'type', 'press', 'scroll', 'close', 'minimize', 'maximize'],
            'file_operation': ['open file', 'show me', 'find file', 'document'],
            'set_reminder': ['remind me', 'reminder', 'don\'t forget', 'alert me', 'wake me'],
            'send_message': ['send', 'message', 'email', 'text']
        }
        
        detected_actions = []
        for action_type, signals in action_signals.items():
            matches = [s for s in signals if s in input_lower]
            if matches:
                detected_actions.append({
                    'action_type': action_type,
                    'confidence': len(matches) / len(signals),
                    'matched_signals': matches
                })
        
        if not detected_actions:
            return None
        
        detected_actions.sort(key=lambda x: x['confidence'], reverse=True)
        best_match = detected_actions[0]
        
        action_type = best_match['action_type']
        
        if action_type == 'play_music' and not available_capabilities.get('local_node'):
            return {
                'action_type': 'play_music',
                'needs_setup': 'local_node',
                'message': "I'd love to play that for you! But I need you to connect your computer first with the !pair command."
            }
        
        if action_type in ['control_computer', 'open_app', 'file_operation'] and not available_capabilities.get('local_node'):
            return {
                'action_type': action_type,
                'needs_setup': 'local_node',
                'message': "That's a great idea! But I need access to your computer first. Run !pair in Discord to connect."
            }
        
        return {
            'action_type': action_type,
            'confidence': best_match['confidence'],
            'raw_input': user_input,
            'execute': True
        }