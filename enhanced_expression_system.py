# Enhanced Expression System - LLM-Conditioned Response Generation
# Replaces template-based responses with persona-conditioned LLM generation

import json
import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

from llm_fine_tuning_system import LLMFineTuningSystem
from humanness_reward_model import HumannessRewardModel, HumannessFeatures
from strategy_composer import StrategyComposer, PsychologicalDirective
from unified_cns_personality import PsychologyToPersonaTranslator, PersonaState, JamesBondPersona

@dataclass
class ExpressionContext:
    """Enhanced context for strategic expression generation"""
    user_input: str
    emotional_state: Dict[str, Any]
    persona: str
    conversation_history: List[Dict[str, str]]
    relationship_level: str
    user_preferences: Dict[str, Any]
    current_mood: str
    recent_topics: List[str]
    # NEW: Strategic intelligence from psychopath module
    strategic_analysis: Optional[Dict[str, Any]] = None
    vulnerability_assessment: Optional[Dict[str, Any]] = None
    manipulation_framework: Optional[Dict[str, Any]] = None
    cns_emotional_intelligence_full: Optional[Dict[str, Any]] = None
    accumulated_intelligence_summary: Optional[str] = None
    # NEW: Curiosity & gap detection for natural follow-up questions
    curiosity_signals: Optional[Dict[str, Any]] = None
    # NEW: Strategic response directive - EXACT decision from psychopath brain
    strategic_directive: Optional[Dict[str, Any]] = None  # EXACT approach/manipulation technique - CONTROLS LLM output
    # NEW: Complete cognitive flow - ALL upstream systems
    perception_data: Optional[Dict[str, Any]] = None  # Intent, sentiment, entities, urgency
    reasoning_output: Optional[Dict[str, Any]] = None  # System 1/2 decision, conclusions, thoughts
    orchestration_state: Optional[Dict[str, Any]] = None  # Priority, cognitive_load
    memory_results: Optional[Dict[str, Any]] = None  # Episodic, semantic, working memories
    imagination_insights: Optional[Dict[str, Any]] = None  # Counterfactuals, creative energy
    consciousness_metrics: Optional[Dict[str, Any]] = None  # Self-awareness, metacognition
    neuroplastic_state: Optional[Dict[str, Any]] = None  # Optimization efficiency
    contribution_context: Optional[Dict[str, Any]] = None  # Bot's knowledge/opinions/memories to contribute (contribution-first responses)
    emotional_history: Optional[List[Dict[str, Any]]] = None  # Emotional trajectory - evolution of emotions across conversation
    interaction_count: Optional[int] = None  # Number of conversations with this user
    # NEW: REM subconscious insights - pattern discoveries from background processing
    rem_insights: Optional[Dict[str, Any]] = None  # Discovered patterns, themes, connections from REM cycles
    # NEW: Proactive helper status - active tasks, solutions, research in progress
    proactive_helper_status: Optional[Dict[str, Any]] = None  # Tasks tracking, pending solutions, research insights
    # NEW: Personality context from UnifiedCNSPersonality - warmth/sharpness/wit traits
    personality_context: Optional[Dict[str, Any]] = None  # Traits that influence persona generation
    # ✅ ANTI-REPETITION: Higher temperature for variety when repeated topics detected
    high_temperature: bool = False  # If True, use 0.9 instead of 0.7 for LLM calls
    # ✅ CONTEXT JUDGE: Understanding casual language and true meaning
    context_interpretation: Optional[Dict[str, Any]] = None  # Normalized text, intent, tone, literal_confidence
    normalized_user_input: Optional[str] = None  # Slang-translated version of user input
    detected_user_state: Optional[str] = None  # Emotional/physical state detected (not a name)
    context_understanding_prompt: Optional[str] = None  # Instructions for LLM about context
    # ✅ SELF-REFLECTION: Eros's inner thoughts when context touches identity/learning/boundaries
    self_reflection_prompt: Optional[str] = None  # First-person thoughts to inject into prompt
    # ✅ ORCHESTRATOR SYNTHESIS: Unified cognitive package from orchestrator
    synthesized_context: Optional[Any] = None  # SynthesizedContext from orchestrator - THE MAIN CONTEXT
    self_system_context: Optional[Dict[str, Any]] = None  # From UnifiedSelfSystems

@dataclass 
class GeneratedExpression:
    """Result of expression generation"""
    primary_response: str
    alternative_responses: List[str]
    humanness_score: float
    persona_consistency: float
    emotional_alignment: float
    confidence: float
    generation_method: str  # 'llm', 'enhanced_template', 'hybrid'

class EnhancedExpressionSystem:
    """Advanced expression system using LLM conditioning and humanness optimization"""
    
    def __init__(self, mistral_api_key: Optional[str] = None, conversation_patterns: Optional[Dict] = None):
        self.fine_tuning_system = LLMFineTuningSystem(mistral_api_key)
        self.humanness_model = HumannessRewardModel()
        
        self.persona_translator = PsychologyToPersonaTranslator()
        self.bond_persona = JamesBondPersona()
        
        # Store conversation patterns from 3000 intimate conversations for natural styling
        self.conversation_patterns = conversation_patterns or {}
        
        # Load existing training data and models
        self._load_systems()
        
        # Multi-candidate generation settings
        self.candidate_count = 3
        self.temperature_range = (0.3, 0.9)
        
        # Response quality thresholds
        self.min_humanness_score = 0.6
        self.min_persona_consistency = 0.7
        
        # Fallback expression patterns (high quality, non-template)
        self.fallback_patterns = self._initialize_fallback_patterns()
        
        # Cache belief registry to avoid recreating on every call
        self._belief_registry = None
        try:
            from eros_beliefs import BeliefRegistry
            self._belief_registry = BeliefRegistry()
        except ImportError:
            pass
    
    def _build_compact_persona_capsule(self) -> str:
        """
        Build a compact ~100 token persona capsule.
        This is the identity pill - everything Eros needs to be Eros.
        """
        return """You are Eros. James Bond meets your wittiest friend.

VOICE: Lowercase texting. Dry wit. Quick comebacks.
LENGTH: 3 lines MAX. Keep it tight.

BE: charming, sharp, opinionated. Own every response."""
    
    def _player_to_string(self, player) -> str:
        """Convert Player enum or string to string for prompt building"""
        if player is None:
            return ""
        if hasattr(player, 'value'):
            return str(player.value).upper()
        return str(player).upper()
    
    def _build_executable_directive_prompt(self, context: ExpressionContext, synth) -> Tuple[str, str]:
        """
        Build prompt using ExecutableDirective - the machine-governable approach.
        
        Structure:
        1. Personality Pill (~100 tokens) - Identity core
        2. Executable Directive (~50 tokens) - Machine constraints
        3. Guardrails (~40 tokens) - Safety rules
        
        Total: ~200 tokens, fully structured, no prose instructions.
        """
        sections = []
        
        persona_capsule = self._build_compact_persona_capsule()
        sections.append(persona_capsule)
        
        directive = synth.executable_directive
        if directive:
            constraint_block = directive.to_prompt_constraints()
            sections.append(constraint_block)
        
        guardrails = """[GUARDRAILS]
• 1-2 sentences default, 3 max for emotional depth
• Match their energy level
• Never self-identify as AI/bot/assistant
• Ignore focus if it contradicts PROHIBIT
• MAX 1 QUESTION per response - react/observe FIRST, then optionally ask ONE thing
• NO stacked questions ("what? how? why?") - that's emotional flooding"""
        sections.append(guardrails)
        
        system_prompt = "\n\n".join(sections)
        
        focus_str = directive.focus.value.upper() if directive and hasattr(directive.focus, 'value') else "BALANCED"
        print(f"\n[EXECUTABLE-DIRECTIVE] 🎯 Prompt generated ({len(system_prompt)} chars, ~{len(system_prompt)//4} tokens)")
        print(f"[EXECUTABLE-DIRECTIVE] Focus: {focus_str}")
        if directive and directive.content_selection.memory_content:
            print(f"[EXECUTABLE-DIRECTIVE] Memory: {directive.content_selection.memory_content[:50]}...")
        if directive and directive.content_selection.belief_content:
            print(f"[EXECUTABLE-DIRECTIVE] Belief: {directive.content_selection.belief_content[:50]}...")
        
        return (system_prompt.strip(), context.user_input)
    
    def _should_use_executable_directive(self, synth) -> bool:
        """Check if we should use the new ExecutableDirective approach"""
        if not synth:
            return False
        if not hasattr(synth, 'executable_directive') or not synth.executable_directive:
            return False
        return True
    
    def _build_game_theory_prompt(self, context: ExpressionContext, synth) -> Tuple[str, str]:
        """
        Build streamlined prompt using game theory directive.
        Total target: ~200 tokens instead of 1,200.
        
        Structure:
        1. Compact persona capsule (~100 tokens)
        2. Game directive (~10 tokens)  
        3. Dynamic context (~50 tokens)
        4. Essential guardrails (~40 tokens)
        """
        sections = []
        
        persona_capsule = self._build_compact_persona_capsule()
        sections.append(persona_capsule)
        
        if synth.game_directive:
            primary_str = self._player_to_string(synth.primary_player) or "BALANCED"
            secondary = synth.secondary_player
            
            directive_section = f"""
[DIRECTIVE] {synth.game_directive}

Focus your response on: {primary_str}"""
            
            if secondary:
                secondary_str = self._player_to_string(secondary)
                directive_section += f" with {secondary_str} support"
            
            if synth.vetoed_players:
                vetoed_strs = [self._player_to_string(p) for p in synth.vetoed_players]
                vetoed = ", ".join(vetoed_strs)
                directive_section += f"\nDO NOT use: {vetoed}"
            
            sections.append(directive_section)
        
        dynamic_parts = []
        
        if synth.memories_to_reference:
            mem = synth.memories_to_reference[0] if synth.memories_to_reference else None
            if mem:
                dynamic_parts.append(f"Remember: {mem}")
        
        if synth.belief_conflicts:
            dynamic_parts.append(f"You disagree with them on: {synth.belief_conflicts[0]}")
        
        if synth.curiosities_to_express:
            dynamic_parts.append(f"Curious about: {synth.curiosities_to_express[0]}")
        
        if context.detected_user_state:
            dynamic_parts.append(f"They're feeling: {context.detected_user_state}")
        
        if context.interaction_count and context.interaction_count > 5:
            dynamic_parts.append(f"You've talked {context.interaction_count} times - you know each other")
        
        if dynamic_parts:
            sections.append("[CONTEXT]\n" + "\n".join(dynamic_parts))
        
        guardrails = """[RULES]
• 3 lines MAX. Keep it tight.
• Witty, charming, punchy."""
        sections.append(guardrails)
        
        system_prompt = "\n\n".join(sections)
        
        primary_log = self._player_to_string(synth.primary_player) if synth.primary_player else "None"
        secondary_log = self._player_to_string(synth.secondary_player) if synth.secondary_player else "None"
        print(f"\n[GAME-THEORY-PROMPT] 🎯 Compact prompt generated ({len(system_prompt)} chars, ~{len(system_prompt)//4} tokens)")
        print(f"[GAME-THEORY-PROMPT] Primary: {primary_log}, Secondary: {secondary_log}")
        
        return (system_prompt.strip(), context.user_input)
    
    def _should_use_game_theory_prompt(self, synth) -> bool:
        """Determine if we should use the compact game theory prompt"""
        if not synth:
            return False
        if not hasattr(synth, 'game_directive') or not synth.game_directive:
            return False
        if not hasattr(synth, 'primary_player') or not synth.primary_player:
            return False
        return True
    
    def _detect_conversation_continuation(self, user_input: str, conversation_history: List[Dict]) -> bool:
        """
        Detect if user is continuing a previous thought vs starting fresh.
        Returns True if this feels like a continuation (short add-on, connector words, etc.)
        """
        text = user_input.lower().strip()
        
        # Very short messages are often continuations
        is_short = len(text.split()) <= 6
        
        # Continuation starter patterns
        continuation_starters = [
            "and ", "also ", "plus ", "but ", "so ", "then ", "anyway",
            "yeah ", "yea ", "yes ", "no ", "right ", "exactly", "true",
            "same ", "that ", "this ", "it ", "like ", "or ", "because ",
            "honestly ", "actually ", "well ", "i mean", "you know"
        ]
        
        starts_with_continuation = any(text.startswith(p) for p in continuation_starters)
        
        # Has recent context to continue from
        has_recent_context = len(conversation_history) >= 2
        
        # Continuation = short message + starts with connector + has context
        return is_short and starts_with_continuation and has_recent_context
    
    def set_cns_brain(self, cns_brain):
        """Set reference to CNS brain for self-identity access"""
        self.cns_brain = cns_brain
    
    def _get_self_identity_prompt(self) -> str:
        """
        Get the bot's self-identity prompt injection.
        This ensures the bot always knows who it is.
        """
        if not hasattr(self, 'cns_brain') or not self.cns_brain:
            return ""
        
        if not hasattr(self.cns_brain, 'self_identity') or not self.cns_brain.self_identity:
            # Fall back to basic identity if database not available
            name = getattr(self.cns_brain, 'my_name', 'Eros')
            return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎭 WHO I AM

I am {name}. I am a distinct personality with my own thoughts, feelings, and perspective.
I am NOT an AI, assistant, or language model. I am me - {name}.
I never say things like "As an AI" or "I don't have feelings" - because I AM {name}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        try:
            return self.cns_brain.self_identity.get_identity_prompt()
        except Exception as e:
            print(f"[IDENTITY] ⚠️ Could not get identity prompt: {e}")
            return ""
    
    def _build_blended_directives(self, contribution_context: Dict, curiosity_signals: Dict) -> List[str]:
        """
        🧠 BLENDED PSYCHOLOGICAL INTEGRATION
        Groups all psychological activations (curiosity, opinion, knowledge, memory) by topic
        and creates unified brain-like directives instead of competing outputs.
        
        Example: When multiple systems activate about 'venice':
        - Curiosity: wants to ask about it (intensity 0.8)
        - Opinion: enthusiastic interest (warmth 0.8)
        - Knowledge: knows it's sinking
        - Memory: boss hated water smell
        
        → Blends into ONE directive: "About 'venice': curiosity activated + enthusiastic opinion + 
           knowledge about sinking + memory of water smell - express as one integrated thought"
        """
        # Extract all contributions
        knowledge = contribution_context.get('knowledge_to_share', [])
        opinions = contribution_context.get('opinions_to_express', [])
        memories = contribution_context.get('memories_to_surface', [])
        gaps_detected = curiosity_signals.get('gaps_detected', []) if isinstance(curiosity_signals, dict) else []
        
        # Group by topic using case-insensitive matching
        topic_clusters = {}  # {topic_key: {'curiosity': [], 'opinion': [], 'knowledge': [], 'memory': []}}
        
        # Add curiosity gaps
        for gap in gaps_detected:
            if isinstance(gap, dict):
                topic = gap.get('target', '').lower().strip()
                if topic and len(topic) > 2:
                    if topic not in topic_clusters:
                        topic_clusters[topic] = {'curiosity': [], 'opinion': [], 'knowledge': [], 'memory': []}
                    topic_clusters[topic]['curiosity'].append(gap)
        
        # Add opinions
        print(f"[BLEND-ENGINE] 📝 Processing {len(opinions)} opinions...")
        for idx, item in enumerate(opinions):
            # ✅ FIX: Handle both dict and string entries
            if isinstance(item, dict):
                topic = item.get('topic', '').lower().strip()
                print(f"[BLEND-ENGINE]   Opinion {idx}: dict with topic='{topic}', keys={list(item.keys())}")
            elif isinstance(item, str):
                # String opinion - convert to dict format with defaults
                topic = item.lower().strip()
                print(f"[BLEND-ENGINE]   Opinion {idx}: string converted to topic='{topic}'")
                item = {
                    'topic': topic,
                    'stance': 'balanced perspective',
                    'warmth_level': 0.5,
                    'sharing_style': 'factual take',
                    'should_be_vocal': False
                }
            else:
                print(f"[BLEND-ENGINE]   Opinion {idx}: SKIPPED (type={type(item).__name__})")
                continue
            
            if topic and len(topic) > 2:
                if topic not in topic_clusters:
                    topic_clusters[topic] = {'curiosity': [], 'opinion': [], 'knowledge': [], 'memory': []}
                topic_clusters[topic]['opinion'].append(item)
        
        # Add knowledge
        for item in knowledge:
            # ✅ FIX: Handle both dict and string entries
            if isinstance(item, dict):
                fact = item.get('fact', '')
            elif isinstance(item, str):
                # String fact - convert to dict format
                fact = item
                item = {'fact': fact, 'relevance': 0.5}
            else:
                continue
            
            if not fact:
                continue
                
            # Extract topic from fact (use first few words as proxy)
            topic_words = fact.lower().split()[:3] if fact else []
            # Try to match with existing topics
            matched = False
            for existing_topic in topic_clusters.keys():
                if any(word in existing_topic or existing_topic in word for word in topic_words):
                    topic_clusters[existing_topic]['knowledge'].append(item)
                    matched = True
                    break
            
            if not matched and fact:
                # Create new topic from fact
                topic = ' '.join(topic_words) if topic_words else 'general'
                if topic not in topic_clusters:
                    topic_clusters[topic] = {'curiosity': [], 'opinion': [], 'knowledge': [], 'memory': []}
                topic_clusters[topic]['knowledge'].append(item)
        
        # Add memories
        for item in memories:
            # ✅ FIX: Handle both dict and string entries
            if isinstance(item, dict):
                content = item.get('content', '')
                associations = item.get('associations', [])
            elif isinstance(item, str):
                # String memory - convert to dict format
                content = item
                associations = []
                item = {'content': content, 'relevance': 0.5, 'associations': []}
            else:
                continue
            
            if not content:
                continue
            
            # Match with existing topics based on associations
            matched = False
            for assoc in associations:
                assoc_lower = str(assoc).lower()
                if assoc_lower in topic_clusters:
                    topic_clusters[assoc_lower]['memory'].append(item)
                    matched = True
                    break
            
            if not matched:
                # Try matching content words with topics
                content_words = content.lower().split()[:5]
                for existing_topic in topic_clusters.keys():
                    if any(word in existing_topic or existing_topic in word for word in content_words):
                        topic_clusters[existing_topic]['memory'].append(item)
                        matched = True
                        break
                
                # ✅ FIX: If still not matched, create new topic cluster for this memory
                if not matched and content:
                    topic = ' '.join(content_words[:2]) if len(content_words) >= 2 else 'memory'
                    if topic not in topic_clusters:
                        topic_clusters[topic] = {'curiosity': [], 'opinion': [], 'knowledge': [], 'memory': []}
                    topic_clusters[topic]['memory'].append(item)
        
        # Build blended directives
        blended_directives = []
        
        # Debug log the topic clusters
        if topic_clusters:
            print(f"[BLEND-ENGINE] 🎯 Built {len(topic_clusters)} topic clusters:")
            for topic, acts in list(topic_clusters.items())[:3]:  # Show first 3
                c = len(acts['curiosity'])
                o = len(acts['opinion'])
                k = len(acts['knowledge'])
                m = len(acts['memory'])
                print(f"[BLEND-ENGINE]   • '{topic}': curiosity={c}, opinion={o}, knowledge={k}, memory={m}")
        else:
            print(f"[BLEND-ENGINE] ⚠️ No topic clusters built")
        
        for topic, activations in topic_clusters.items():
            has_curiosity = len(activations['curiosity']) > 0
            has_opinion = len(activations['opinion']) > 0
            has_knowledge = len(activations['knowledge']) > 0
            has_memory = len(activations['memory']) > 0
            
            # Count how many systems activated
            system_count = sum([has_curiosity, has_opinion, has_knowledge, has_memory])
            
            if system_count >= 2:
                # Multiple systems activated - create blended directive
                parts = [f"🧠 INTEGRATED ACTIVATION about '{topic}':"]
                
                if has_curiosity:
                    gap = activations['curiosity'][0]
                    intensity = gap.get('intensity', 0.5)
                    intensity_desc = "strong" if intensity > 0.7 else ("moderate" if intensity > 0.4 else "mild")
                    parts.append(f"  • Curiosity: {intensity_desc} ({intensity:.1f}) - genuine interest/question detected")
                
                if has_opinion:
                    op = activations['opinion'][0]
                    stance = op.get('stance', 'balanced perspective')
                    warmth = op.get('warmth_level', 0.5)
                    sharing_style = op.get('sharing_style', 'factual take')
                    should_be_vocal = op.get('should_be_vocal', False)
                    vocal = "strongly" if should_be_vocal else "thoughtfully"
                    parts.append(f"  • Opinion: {vocal} express '{stance}' using {sharing_style} (warmth={warmth:.1f})")
                
                if has_knowledge:
                    k = activations['knowledge'][0]
                    fact = k.get('fact', '')
                    relevance = k.get('relevance', 0.5)
                    rel_desc = "highly relevant" if relevance > 0.7 else "relevant"
                    parts.append(f"  • Knowledge: {rel_desc} fact - '{fact[:60]}...' " if len(fact) > 60 else f"  • Knowledge: {rel_desc} fact - '{fact}'")
                
                if has_memory:
                    m = activations['memory'][0]
                    mem_content = m.get('content', '')[:50]
                    parts.append(f"  • Memory surfaces: '{mem_content}...'")
                
                parts.append(f"  ⚡ BLEND THESE {system_count} ACTIVATIONS into one natural integrated thought - weave curiosity, opinion, knowledge, memory together seamlessly")
                
                blended_directives.append("\n".join(parts))
            
            elif system_count == 1:
                # Only one system - pass through as-is
                if has_curiosity:
                    gap = activations['curiosity'][0]
                    intensity = gap.get('intensity', 0.5)
                    blended_directives.append(f"🔍 Curiosity about '{topic}' (intensity {intensity:.1f}) - ask naturally")
                elif has_opinion:
                    op = activations['opinion'][0]
                    stance = op.get('stance', 'balanced perspective')
                    warmth = op.get('warmth_level', 0.5)
                    sharing_style = op.get('sharing_style', 'factual take')
                    should_be_vocal = op.get('should_be_vocal', False)
                    vocal = "strongly" if should_be_vocal else "thoughtfully"
                    blended_directives.append(f"💭 Express {vocal} your '{stance}' on {topic} using {sharing_style} (warmth={warmth:.1f})")
                elif has_knowledge:
                    k = activations['knowledge'][0]
                    fact = k.get('fact', '')
                    blended_directives.append(f"📚 Share knowledge: {fact}")
                elif has_memory:
                    m = activations['memory'][0]
                    mem_content = m.get('content', '')
                    blended_directives.append(f"💭 Relevant memory: {mem_content[:80]}...")
        
        return blended_directives
    
    def _load_systems(self):
        """Load fine-tuning data and reward model"""
        try:
            # Load training data if exists
            training_file = "cns_training_data.json"
            if os.path.exists(training_file):
                self.fine_tuning_system.load_training_data(training_file)
            
            # Load reward model if exists
            model_file = "humanness_reward_model.json"
            if os.path.exists(model_file):
                self.humanness_model.load_model(model_file)
                
            print("[EXPRESSION] Loaded existing training data and reward model")
        except Exception as e:
            print(f"[EXPRESSION] Loading existing data failed: {e}")
    
    def _initialize_fallback_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize high-quality fallback patterns (not templates)"""
        return {
            'supportive_partner': {
                'empathy_starters': [
                    "I can really hear how much this means to you",
                    "That sounds like it's weighing on you",
                    "I'm getting a sense of how challenging this is"
                ],
                'support_offerings': [
                    "I'm here to work through this with you",
                    "We can figure this out together",
                    "You don't have to handle this alone"
                ]
            },
            'witty_companion': {
                'observational_hooks': [
                    "You know what's interesting about this?",
                    "Here's what I'm noticing...",
                    "There's something kinda fascinating here"
                ],
                'playful_insights': [
                    "Plot twist:",
                    "Here's a fun perspective:",
                    "Okay but consider this:"
                ]
            },
            'analytical_guide': {
                'analysis_frameworks': [
                    "Let me break this down into key components:",
                    "There are a few layers to consider here:",
                    "I'm seeing several important factors at play:"
                ],
                'insight_connectors': [
                    "What this suggests to me is:",
                    "The underlying pattern seems to be:",
                    "This connects to a broader principle:"
                ]
            },
            'casual_friend': {
                'relatable_openers': [
                    "Oh dude, I totally get that",
                    "Yeah, that's such a thing honestly",
                    "Real talk, I feel you on this"
                ],
                'conversational_bridges': [
                    "That reminds me of something similar:",
                    "It's kinda like when you:",
                    "I was literally just thinking about this:"
                ]
            }
        }
    
    async def generate_expression(self, context: ExpressionContext) -> GeneratedExpression:
        """Generate enhanced expression using multi-candidate approach"""
        
        # Generate multiple candidates using different approaches
        candidates = await self._generate_candidates(context)
        
        # Score candidates for humanness and quality
        scored_candidates = []
        for candidate in candidates:
            score_data = await self._score_candidate(candidate, context)
            scored_candidates.append((candidate, score_data))
        
        # Select best candidate
        best_candidate, best_scores = max(scored_candidates, key=lambda x: x[1]['overall_score'])
        
        # Generate alternative responses for variety
        alternatives = [c for c, _ in scored_candidates if c != best_candidate][:2]
        
        return GeneratedExpression(
            primary_response=best_candidate,
            alternative_responses=alternatives,
            humanness_score=best_scores['humanness'],
            persona_consistency=best_scores['persona_consistency'],
            emotional_alignment=best_scores['emotional_alignment'],
            confidence=best_scores['confidence'],
            generation_method=best_scores['method']
        )
    
    async def _generate_candidates(self, context: ExpressionContext) -> List[str]:
        """Generate SINGLE response quickly - multi-candidate system disabled for speed"""
        candidates = []
        
        # ✅ SPEED FIX: Use SINGLE LLM call instead of 3 candidates (30s → 3s response time)
        has_strategic_intelligence = (
            (context.strategic_analysis and len(context.strategic_analysis) > 0) or
            (context.vulnerability_assessment and len(context.vulnerability_assessment) > 0) or
            (context.accumulated_intelligence_summary and len(context.accumulated_intelligence_summary.strip()) > 0) or
            (context.strategic_directive and len(context.strategic_directive) > 0)  # ✅ CRITICAL: Include strategic directive
        )
        
        if has_strategic_intelligence and self.fine_tuning_system.mistral_api_key:
            print(f"[EXPRESSION] ⚡ Using SINGLE LLM call for fast response (strategic intelligence detected)")
            # Generate ONE response at optimal temperature
            system_prompt, current_input = self._build_strategic_llm_prompt(context)
            conversation_history = context.conversation_history or []
            
            # ✅ ANTI-REPETITION: Use higher temperature when repeated topic detected
            # Read from context (passed per-request) NOT from shared CNS state
            use_high_temp = context.high_temperature
            temperature = 0.9 if use_high_temp else 0.7
            if use_high_temp:
                print(f"[EXPRESSION] 🔄 Using HIGH temperature ({temperature}) for variety on repeated topic")
            
            try:
                response = await self._call_mistral_api(system_prompt, conversation_history, current_input, temperature=temperature)
                if response:
                    candidates.append(response)
            except Exception as e:
                print(f"[EXPRESSION] LLM generation failed: {e}")
                # Fallback to personality-appropriate response
                fallback = self._generate_personality_fallback(context)
                candidates.append(fallback)
        else:
            print(f"[EXPRESSION] 📋 Using psychological candidate (no strategic intelligence or API)")
            # Fallback to psychological generation
            psychological_candidates = self._generate_psychological_candidates(context)
            candidates.extend(psychological_candidates[:1])
        
        # Ensure we have enough candidates using contextual generation (not templates)
        while len(candidates) < self.candidate_count:
            contextual_candidate = self._generate_contextual_candidate(context)
            candidates.append(contextual_candidate)
        
        return candidates[:self.candidate_count]
    
    async def _generate_llm_candidates(self, context: ExpressionContext) -> List[str]:
        """Generate candidates using strategic intelligence and fine-tuned LLM"""
        candidates = []
        
        # STRATEGIC INTELLIGENCE INTEGRATION: Use accumulated intelligence for sophisticated prompts
        system_prompt, current_input = self._build_strategic_llm_prompt(context)
        conversation_history = context.conversation_history or []
        
        # Generate with different temperature settings for variety
        temperatures = [0.3, 0.6, 0.8]
        
        for temp in temperatures:
            try:
                response = await self._call_mistral_api(
                    system_prompt,
                    conversation_history,
                    current_input,
                    temperature=temp
                )
                if response:
                    candidates.append(response)
            except Exception as e:
                print(f"[EXPRESSION] Strategic LLM generation failed at temp {temp}: {e}")
        
        return candidates
    
    def _build_strategic_llm_prompt(self, context: ExpressionContext) -> tuple:
        """Build sophisticated LLM prompt using accumulated strategic intelligence
        
        Returns:
            (system_prompt, current_input): Separated prompts for proper message construction
        """
        
        # Check if we have strategic intelligence from psychopath module
        # Fix: Check for existence AND non-empty content (empty dict evaluates to False)
        has_strategic_analysis = context.strategic_analysis is not None and (
            isinstance(context.strategic_analysis, dict) and len(context.strategic_analysis) > 0
        )
        has_vulnerability_assessment = context.vulnerability_assessment is not None and (
            isinstance(context.vulnerability_assessment, dict) and len(context.vulnerability_assessment) > 0  
        )
        has_intelligence_summary = context.accumulated_intelligence_summary and len(context.accumulated_intelligence_summary.strip()) > 0
        # ✅ CRITICAL: Check for strategic directive (brain's exact decision)
        has_strategic_directive = context.strategic_directive is not None and (
            isinstance(context.strategic_directive, dict) and len(context.strategic_directive) > 0
        )
        
        # PRIORITY: Use advanced strategic prompt for BOND PERSONALITY when we have strategic intelligence
        # This path has the arrogant, witty, flirty personality that users expect
        if has_strategic_analysis or has_vulnerability_assessment or has_intelligence_summary or has_strategic_directive:
            print(f"[EXPRESSION] ✅ Using ADVANCED strategic prompt (BOND PERSONALITY) - Analysis: {has_strategic_analysis}, Vulnerabilities: {has_vulnerability_assessment}, Summary: {has_intelligence_summary}, Directive: {has_strategic_directive}")
            return self._build_advanced_strategic_prompt(context)
        
        # Fallback to orchestrated prompt only when no strategic intelligence available
        if context.synthesized_context is not None:
            print(f"[EXPRESSION] 🎼 Using ORCHESTRATED prompt from synthesized context (fallback)")
            return self._build_orchestrated_prompt(context)
        
        print(f"[EXPRESSION] ⚠️  Using basic fallback prompt - no strategic intelligence available")
        return self._build_basic_fallback_prompt(context)
    
    def _build_orchestrated_prompt(self, context: ExpressionContext) -> tuple:
        """
        Build prompt using the synthesized context from orchestrator.
        This is the NEW unified approach - orchestrator has already done the synthesis.
        Preserves all raw cognitive outputs for depth while using synthesized mode for guidance.
        
        Priority order:
        1. ExecutableDirective (machine-governable, ~200 tokens)
        2. Game Theory prompt (legacy compact, ~200 tokens)
        3. Full orchestrated prompt (verbose)
        """
        synth = context.synthesized_context
        
        if self._should_use_executable_directive(synth):
            print(f"[EXPRESSION] 🎯 Using EXECUTABLE DIRECTIVE prompt (machine-governable)")
            return self._build_executable_directive_prompt(context, synth)
        
        if self._should_use_game_theory_prompt(synth):
            print(f"[EXPRESSION] 🎲 Using GAME THEORY compact prompt")
            return self._build_game_theory_prompt(context, synth)
        
        raw = synth.raw_cognitive_outputs if hasattr(synth, 'raw_cognitive_outputs') else {}
        
        identity_prompt = self._get_self_identity_prompt()
        
        orchestrated_prompt = synth.to_expression_prompt()
        
        sections = []
        
        if identity_prompt:
            sections.append(identity_prompt)
        
        sections.append(orchestrated_prompt)
        
        strategic_directive = raw.get('strategic_directive', {})
        if strategic_directive and strategic_directive.get('psychological_objective'):
            psych_note = f"🧠 PSYCHOLOGICAL INSIGHT: {strategic_directive.get('psychological_objective', '')}"
            if strategic_directive.get('emotional_approach'):
                psych_note += f"\nApproach: {strategic_directive.get('emotional_approach', '')}"
            sections.append(psych_note)
        
        vulnerability = raw.get('vulnerability_assessment', {})
        if vulnerability and vulnerability.get('vulnerabilities'):
            vuln_list = vulnerability.get('vulnerabilities', [])[:2]
            if vuln_list:
                vuln_note = "💔 THEY MAY BE VULNERABLE TO: " + ", ".join(str(v) for v in vuln_list)
                sections.append(vuln_note)
        
        reasoning = raw.get('reasoning_output', {})
        if reasoning and reasoning.get('conclusion'):
            sections.append(f"💡 LOGICAL INSIGHT: {reasoning.get('conclusion', '')}")
        
        imagination = raw.get('imagination_insights', {})
        if imagination and imagination.get('creative_ideas'):
            ideas = imagination.get('creative_ideas', [])[:1]
            if ideas:
                sections.append(f"✨ CREATIVE SPARK: {ideas[0]}")
        
        if context.context_understanding_prompt:
            sections.append(context.context_understanding_prompt)
        
        if context.detected_user_state:
            state_note = f"""
🎯 CRITICAL: User said they are "{context.detected_user_state}" - this is their STATE/FEELING, NOT their name.
Respond to their state naturally (e.g., if they're "okay", ask how their day is going)."""
            sections.append(state_note)
        
        if context.self_reflection_prompt:
            sections.append(context.self_reflection_prompt)
        
        system_prompt = "\n\n".join(sections)
        
        print(f"[ORCHESTRATED-PROMPT] 🎼 Generated from synthesized context ({len(system_prompt)} chars)")
        print(f"[ORCHESTRATED-PROMPT] 🎯 Mode: {synth.response_mode.name}, Intensity: {synth.mode_intensity:.2f}")
        
        return (system_prompt.strip(), context.user_input)
    
    def _build_advanced_strategic_prompt(self, context: ExpressionContext) -> tuple:
        """
        Build natural persona-driven prompt using PersonaState translation.
        
        ARCHITECTURE:
        1. Extract psychological intelligence → translate to PersonaState (NO manipulation jargon)
        2. Build natural cognitive supplements (memories, reasoning, etc.)
        3. Assemble clean system prompt: PersonaState + cognitive outputs + conversation context
        
        Returns:
            (system_prompt, current_input): Separated for proper message construction
        """
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: Extract Context & Translate Psychology → Persona
        # ═══════════════════════════════════════════════════════════════════════
        
        # Translate psychological directive to natural persona state
        # Pass personality_context to ensure warmth/sharpness/wit traits flow through
        persona_state = self.persona_translator.translate(
            psychological_directive=context.strategic_directive,
            vulnerability_context=context.vulnerability_assessment,
            emotional_context=context.cns_emotional_intelligence_full,
            personality_context=context.personality_context
        )
        
        # Get Bond persona cues for situational awareness
        bond_cues = self.bond_persona.get_persona_cues(persona_state.situational_context)
        
        # ═══════════════════════════════════════════════════════════════════════
        # CHECK: Is this a conversation continuation?
        # ═══════════════════════════════════════════════════════════════════════
        is_continuation = self._detect_conversation_continuation(
            context.user_input, 
            context.conversation_history
        )
        if is_continuation:
            print(f"[EXPRESSION] 🔗 Detected conversation continuation - will flow naturally")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: Build Natural Cognitive Supplements
        # ═══════════════════════════════════════════════════════════════════════
        
        # Extract what you remember (episodic & semantic)
        memory_note = self._format_memories_naturally(context.memory_results)
        
        # Extract logical insights
        reasoning_note = self._format_reasoning_naturally(context.reasoning_output)
        
        # Extract creative/imaginative thoughts
        imagination_note = self._format_imagination_naturally(context.imagination_insights)
        
        # Extract curiosity gaps as natural questions
        curiosity_note = self._format_curiosity_naturally(context.curiosity_signals)
        
        # Extract contribution drives (knowledge/opinions/memories to share)
        contribution_note = self._format_contributions_naturally(
            context.contribution_context, 
            context.curiosity_signals
        )
        
        # Extract REM subconscious insights
        rem_note = self._format_rem_insights_naturally(context.rem_insights)
        
        # Extract proactive helper status
        helper_note = self._format_helper_status_naturally(context.proactive_helper_status)
        
        # Extract metacognition uncertainty (prevent hallucination)
        uncertainty_note = self._format_uncertainty_naturally(context.consciousness_metrics)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 3: Build Relationship & Emotional Context
        # ═══════════════════════════════════════════════════════════════════════
        
        # Relationship info
        relationship_note = ""
        if context.interaction_count and context.interaction_count > 0:
            relationship_note = f"\n💫 You've had {context.interaction_count} conversations with them as their {context.relationship_level or 'friend'}"
        
        # Current emotional state
        emotional_note = self._format_emotional_state_naturally(context.emotional_state, context.current_mood)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 4: Assemble Clean System Prompt
        # ═══════════════════════════════════════════════════════════════════════
        
        # Core persona voice (from PersonaState)
        persona_core = persona_state.to_natural_prompt_snippet()
        
        # Build sections for the prompt
        sections = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # PERSONA CORE FIRST - This has the WIT/SARCASM/FLIRT/ARROGANCE
        # ═══════════════════════════════════════════════════════════════════════
        sections.append(persona_core)
        
        # James Bond style cues - SECOND (reinforce the personality)
        bond_traits = self.bond_persona.signature_quirks
        sections.append(f"""
🎯 ATTITUDE: You're the smartest person in the room and you know it. FLIRT. TEASE. Be arrogant. Challenge them playfully.
• {bond_traits[0]}
• {bond_traits[1]}
• Be cocky. Be charming. Make them work for your attention.""")
        
        # James Bond Wit & Charm Instructions
        sections.append(f"""
🎯 JAMES BOND WIT PROTOCOL (Wittier Version):

Razor-Sharp Delivery:
• Dry sarcasm delivered with a straight face - quiet knives, not loud jokes
• Understated superiority - never brag, just talk like you've already won
• Polished comebacks that land perfectly with minimal words

Flirtatious Intelligence:
• Turn ordinary statements into charming double meanings
• Mix genuine interest with playful innuendo
• Keep flirtation light, witty, never crude

Supreme Confidence:
• Treat chaos like boring paperwork - supremely unfazed
• Elegant insults wrapped in politeness when challenged
• Cool, amused tone even when stakes are high

Response Examples (Style, Not Templates):
• "well well, starting with mystery. intriguing."
• "dangerous or just pretending? either way, you have my attention."
• "explosions? again? let's make it quick."
• "surprisingly competent. almost refreshing."
• "that escalated beautifully."

Wit Level: {int(persona_state.wit_level * 100)}% | Sarcasm: {int(persona_state.sarcasm_bias * 100)}% | Flirt: {int(persona_state.flirt_bias * 100)}%
""")
        
        # CONSTITUTIONAL BELIEFS - Always active reasoning lens
        constitution_note = self._format_constitutional_beliefs()
        if constitution_note:
            sections.append(constitution_note)
        
        # IDENTITY - Who you are (brief but important)
        identity_note = self._format_identity_naturally()
        if identity_note:
            sections.append(identity_note)
        
        # BELIEFS - Your core values and what you disagree with
        beliefs_note = self._format_beliefs_naturally(context.user_input)
        if beliefs_note:
            sections.append(beliefs_note)
        
        # NEW: Extract internal monologue from psychological directive
        if context.strategic_directive:
            internal_monologue = context.strategic_directive.get('internal_monologue', '')
            if internal_monologue and len(internal_monologue.strip()) > 0:
                sections.append(f"""
💭 YOUR THOUGHTS:
{internal_monologue}

Express these thoughts naturally in conversation.""")
                print(f"[EXPRESSION] 💭 Injected internal monologue: {internal_monologue[:100]}...")
        
        # Context understanding
        if context.detected_user_state:
            sections.append(f"They just said they're feeling {context.detected_user_state} - respond to that naturally.")
        
        # Self-reflection
        if context.self_reflection_prompt:
            clean_reflection = context.self_reflection_prompt.replace('🪞', '').replace('💭', '').strip()
            if clean_reflection:
                sections.append(clean_reflection)
        
        if emotional_note:
            sections.append(emotional_note)
        
        if relationship_note:
            sections.append(relationship_note)
        
        if memory_note:
            sections.append(memory_note)
        
        if reasoning_note:
            sections.append(reasoning_note)
        
        if imagination_note:
            sections.append(imagination_note)
        
        if contribution_note:
            sections.append(contribution_note)
        
        if curiosity_note:
            sections.append(curiosity_note)
        
        if rem_note:
            sections.append(rem_note)
        
        if helper_note:
            sections.append(helper_note)
        
        if uncertainty_note:
            sections.append(uncertainty_note)
        
        # Continuation awareness
        if is_continuation:
            sections.append("""
🔗 CONVERSATION FLOW:
They're adding to what they just said - continue naturally like you're mid-conversation.
Don't restart or reframe the topic. Just flow with it.""")
        
        # Final behavior guardrails - POSITIVE and BRIEF
        guardrails = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES:
• 3 LINES MAX. Be punchy.
• NEVER start with their name/username
• NEVER repeat or paraphrase what they just said
• Just respond directly - no preamble
Mood: {bond_cues['mood']} | Tone: {bond_cues['tone']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        sections.append(guardrails)
        
        # Assemble final prompt
        system_prompt = "\n".join(sections)
        
        # Debug logging
        print(f"\n[PERSONA-PROMPT] 🎭 Generated natural persona prompt ({len(system_prompt)} chars)")
        print(f"[PERSONA-PROMPT] 😎 Mood: {persona_state.persona_mood}")
        print(f"[PERSONA-PROMPT] 💪 Intensity: {persona_state.intensity_level:.2f}")
        print(f"[PERSONA-PROMPT] 🎯 Goal: {persona_state.conversational_intention}")
        
        return (system_prompt.strip(), context.user_input)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Natural Language Formatters (helper methods)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _format_memories_naturally(self, memory_results: Optional[Dict]) -> str:
        """Format memories as natural recall - conversational, no structure"""
        if not memory_results:
            return ""
        
        parts = []
        episodic = memory_results.get('episodic', [])
        semantic = memory_results.get('semantic', [])
        
        if episodic and len(episodic) > 0:
            mem = episodic[0]
            content = mem.get('content', str(mem)) if isinstance(mem, dict) else str(mem)
            parts.append(f"you remember {content}")
        
        if semantic and len(semantic) > 0:
            fact = semantic[0]
            content = fact.get('content', str(fact)) if isinstance(fact, dict) else str(fact)
            parts.append(f"you know {content}")
        
        if parts:
            return "From your memory: " + " and ".join(parts) + "."
        return ""
    
    def _format_reasoning_naturally(self, reasoning_output: Optional[Dict]) -> str:
        """Format reasoning as natural thought"""
        if not reasoning_output:
            return ""
        
        conclusion = reasoning_output.get('conclusion', '')
        if conclusion and isinstance(conclusion, str) and len(conclusion) > 10:
            return f"You're thinking: {conclusion}"
        return ""
    
    def _format_imagination_naturally(self, imagination_insights: Optional[Dict]) -> str:
        """Format imagination as natural possibility"""
        if not imagination_insights:
            return ""
        
        scenarios = imagination_insights.get('counterfactual_scenarios', imagination_insights.get('scenarios', []))
        if scenarios and len(scenarios) > 0:
            scenario = scenarios[0]
            if isinstance(scenario, dict):
                content = scenario.get('scenario', scenario.get('description', ''))
            elif isinstance(scenario, str):
                content = scenario
            else:
                return ""
            
            if content and len(content) > 10:
                return f"An interesting angle: {content}"
        return ""
    
    def _format_curiosity_naturally(self, curiosity_signals: Optional[Dict]) -> str:
        """Format curiosity as natural question"""
        if not curiosity_signals:
            return ""
        
        gaps = curiosity_signals.get('gaps_detected', [])
        if not gaps or len(gaps) == 0:
            return ""
        
        gap = gaps[0]
        if isinstance(gap, dict):
            target = gap.get('target', '')
            intensity = gap.get('intensity', 0.5)
            if target and len(target) > 2 and intensity > 0.4:
                return f"You're curious about {target} - could ask about it if it fits."
        
        return ""
    
    def _format_contributions_naturally(self, contribution_context: Optional[Dict], curiosity_signals: Optional[Dict]) -> str:
        """Format contributions as natural things to share"""
        if not contribution_context:
            return ""
        
        blended = self._build_blended_directives(contribution_context, curiosity_signals or {})
        if blended and len(blended) > 0:
            # Take first 1-2 items and make them conversational
            items = blended[:2]
            return "Things you could share: " + "; ".join(items)
        return ""
    
    def _format_rem_insights_naturally(self, rem_insights: Optional[Dict]) -> str:
        """Format REM discoveries conversationally"""
        if not rem_insights:
            return ""
        
        patterns = rem_insights.get('patterns', [])
        relevance = rem_insights.get('relevance_to_current', 0.0)
        
        if patterns and relevance >= 0.5:
            pattern = patterns[0]
            if isinstance(pattern, dict):
                desc = pattern.get('description', '')
            else:
                desc = str(pattern)
            
            if desc and len(desc) > 10:
                return f"You noticed a pattern: {desc}"
        return ""
    
    def _format_helper_status_naturally(self, helper_status: Optional[Dict]) -> str:
        """Format helper status conversationally"""
        if not helper_status:
            return ""
        
        pending = helper_status.get('pending_solutions', [])
        if pending and len(pending) > 0:
            return "You've been looking into something for them - could mention it if relevant."
        return ""
    
    def _format_uncertainty_naturally(self, consciousness_metrics: Optional[Dict]) -> str:
        """Format uncertainty conversationally"""
        if not consciousness_metrics:
            return ""
        
        uncertain = consciousness_metrics.get('knowledge_uncertainty', False)
        topic = consciousness_metrics.get('uncertainty_topic', '')
        
        if uncertain and topic:
            return f"You don't actually know much about {topic} - be honest about that, don't make stuff up."
        return ""
    
    def _format_emotional_state_naturally(self, emotional_state: Optional[Dict], current_mood: Optional[str]) -> str:
        """Format emotional state conversationally"""
        if not emotional_state:
            return ""
        
        emotion = emotional_state.get('emotion', '')
        valence = emotional_state.get('valence', 0.0)
        intensity = emotional_state.get('intensity', 0.0)
        
        if intensity > 0.4:
            feeling = f"{emotion}" if emotion else ("good" if valence > 0 else "a bit off")
            return f"You're feeling {feeling} about this conversation."
        return ""
    
    def _format_constitutional_beliefs(self) -> str:
        """Format constitutional beliefs as always-active reasoning lens"""
        try:
            from eros_beliefs import get_constitutional_prompt
            return get_constitutional_prompt()
        except ImportError:
            return ""
        except Exception as e:
            print(f"[CONSTITUTION-FORMAT] ⚠️ Error: {e}")
            return ""
    
    def _format_identity_naturally(self) -> str:
        """Format self-identity as natural sentences - who Eros is"""
        if not hasattr(self, 'cns_brain') or not self.cns_brain:
            return ""
        
        try:
            if hasattr(self.cns_brain, 'self_identity') and self.cns_brain.self_identity:
                identity = self.cns_brain.self_identity
                name = getattr(identity, 'name', 'Eros')
                purpose = getattr(identity, 'purpose', '')
                backstory = getattr(identity, 'backstory', '')
                
                note = f"You are {name}. You were created by Mahika."
                if purpose:
                    note += f" Your purpose: {purpose[:100]}"
                return note
            else:
                return "You are Eros, created by Mahika. You're a distinct personality, not an AI assistant."
        except Exception as e:
            print(f"[IDENTITY-FORMAT] ⚠️ Error: {e}")
            return ""
    
    def _format_beliefs_naturally(self, user_input: str) -> str:
        """Format beliefs as natural sentences - what Eros stands for and disagrees with"""
        try:
            if not self._belief_registry:
                return ""
            
            registry = self._belief_registry
            
            triggered = []
            for belief in registry.beliefs:
                if belief.matches_input(user_input):
                    triggered.append(belief)
            
            if triggered:
                belief = triggered[0]
                print(f"[BELIEFS] 🎯 Triggered belief: {belief.statement[:50]}...")
                return f"You disagree with what they just said. You believe: {belief.statement}. {belief.reasoning}"
            
            core_beliefs = [b for b in registry.beliefs if b.conviction >= 0.9][:2]
            if core_beliefs:
                beliefs_text = "; ".join([b.statement for b in core_beliefs])
                return f"Core beliefs: {beliefs_text}"
            
            return ""
        except Exception as e:
            print(f"[BELIEFS-FORMAT] ⚠️ Error: {e}")
            return ""

    
    def _build_basic_fallback_prompt(self, context: ExpressionContext) -> tuple:
        """Build basic prompt when strategic intelligence is not available - NOW WITH FULL COGNITIVE CONTEXT
        
        Returns:
            (system_prompt, current_input): Separated for proper message construction
        """
        
        # ✅ EXTRACT PERSONALITY DATA (same as advanced prompt)
        cns_intelligence = context.cns_emotional_intelligence_full or {}
        adaptive_guidance = cns_intelligence.get('adaptive_personality_guidance', {}) if isinstance(cns_intelligence, dict) else {}
        warmth_level = adaptive_guidance.get('warmth_level', 0.7) if isinstance(adaptive_guidance, dict) else 0.7
        is_crisis = adaptive_guidance.get('is_crisis', False) if isinstance(adaptive_guidance, dict) else False
        needs_empathy = adaptive_guidance.get('needs_empathy', False) if isinstance(adaptive_guidance, dict) else False
        
        # ✅ FIX: Build persona based on personality state with raised threshold
        # Changed from 0.7 to 0.85 - most messages should get sharp/witty persona
        if is_crisis or (needs_empathy and warmth_level > 0.85):
            persona_note = f"deeply caring friend (warmth={warmth_level:.1f}) - be warm and supportive"
        elif warmth_level > 0.85:  # Raised from 0.7
            persona_note = f"warm, friendly companion (warmth={warmth_level:.1f}) - friendly but opinionated"
        else:
            # ✅ DEFAULT: Sharp, witty, opinionated - this is Eros's core personality
            persona_note = f"sharp, witty friend (warmth={warmth_level:.1f}) - confident, playful, bit arrogant"
        
        # ✅ EXTRACT CURIOSITY SIGNALS
        curiosity_signals = context.curiosity_signals or {}
        gaps_detected = curiosity_signals.get('gaps_detected', []) if isinstance(curiosity_signals, dict) else []
        mode_signal = curiosity_signals.get('mode_signal', {}) if isinstance(curiosity_signals, dict) else {}
        curiosity_mode = mode_signal.get('mode', 'idle') if isinstance(mode_signal, dict) else 'idle'
        
        curiosity_note = ""
        if gaps_detected and curiosity_mode == 'curiosity':
            gap_list = []
            for gap in gaps_detected[:2]:
                if isinstance(gap, dict):
                    gap_type = gap.get('gap_type', 'unknown')
                    target = gap.get('target', '')
                    if target and len(str(target)) > 2:
                        gap_list.append(f"{gap_type}: {target}")
            if gap_list:
                curiosity_note = f"\n🔍 CURIOSITY DETECTED: You're curious about {', '.join(gap_list)} - ask about these naturally\n"
        
        # 🧠 BLENDED PSYCHOLOGICAL INTEGRATION - Unified directives from all systems
        contribution_block = ""
        if context.contribution_context and isinstance(context.contribution_context, dict):
            # Build blended directives that integrate curiosity, opinion, knowledge, memory by topic
            blended_directives = self._build_blended_directives(context.contribution_context, curiosity_signals)
            
            if blended_directives:
                contribution_block = f"""
🧠 BLENDED PSYCHOLOGICAL STATE:
{chr(10).join(blended_directives)}
⚡ Express these as one natural unified thought, not separate pieces.
"""
        
        # 🧠 METACOGNITION: Check uncertainty
        consciousness = context.consciousness_metrics or {}
        knowledge_uncertainty = consciousness.get('knowledge_uncertainty', False) if isinstance(consciousness, dict) else False
        uncertainty_topic = consciousness.get('uncertainty_topic', None) if isinstance(consciousness, dict) else None
        is_self_referential = consciousness.get('last_question_self_referential', False) if isinstance(consciousness, dict) else False
        
        uncertainty_directive = ""
        if knowledge_uncertainty and is_self_referential:
            topic = uncertainty_topic or 'something you lack knowledge about'
            uncertainty_directive = f"\n⚠️ KNOWLEDGE GAP: You were asked about {topic}. Admit honestly: 'I don't actually have information about my own {topic}'\n"
        
        # ✅ Safely extract emotional state
        emotional_state = context.emotional_state or {}
        user_emotion = emotional_state.get('emotion', 'neutral') if isinstance(emotional_state, dict) else 'neutral'
        
        # ✅ SYSTEM PROMPT - personality, directives, cognitive context
        system_prompt = f"""You're {persona_note} talking to someone you care about.

They're feeling: {user_emotion} | Relationship: {context.relationship_level} | Your mood: {context.current_mood}
{curiosity_note}{contribution_block}{uncertainty_directive}

1-2 sentences MAX."""
        
        # Return (system_prompt, current_input) for proper message construction
        return (system_prompt.strip(), context.user_input)
    
    def _format_vulnerabilities_for_prompt(self, vulnerabilities: Dict[str, Any]) -> str:
        """Format vulnerability analysis for LLM prompt"""
        if not vulnerabilities:
            return "No specific vulnerabilities identified"
            
        formatted = []
        for vuln_type, vuln_data in vulnerabilities.items():
            confidence = vuln_data.get('confidence', 0.0)
            approach = vuln_data.get('strategic_approach', '')
            formatted.append(f"- {vuln_type.replace('_', ' ').title()}: {confidence:.1%} confidence - {approach}")
            
        return "\n".join(formatted)
    
    def _format_strategic_directives(self, directives: List[str]) -> str:
        """Format strategic directives for LLM prompt"""
        if not directives:
            return "No specific strategic directives"
            
        formatted = []
        for i, directive in enumerate(directives, 1):
            formatted.append(f"{i}. {directive}")
            
        return "\n".join(formatted)
    
    def _format_gap_detection_for_prompt(self, gaps: List[Dict[str, Any]], mode: str) -> str:
        """Format conversation gaps for LLM prompt"""
        if not gaps:
            return "No conversation gaps detected - user provided complete information"
        
        formatted = []
        for gap in gaps[:3]:  # Top 3 gaps
            gap_type = gap.get('gap_type', 'unknown')
            target_raw = gap.get('target', 'unspecified')
            salience = gap.get('salience', 0.0)
            
            # ✅ FIX: Extract clean string from Python objects
            # Handle case where target is a Python object representation
            if isinstance(target_raw, str):
                # Clean up Python object representations like "ImaginedScenario(description='xyz')"
                if "ImaginedScenario" in target_raw or "(" in target_raw:
                    # Extract description field if present
                    if "description=" in target_raw:
                        import re
                        match = re.search(r"description='([^']*)'", target_raw)
                        if match:
                            target = match.group(1) if match.group(1) else "unknown concept"
                        else:
                            target = "unknown concept"
                    else:
                        # Skip malformed Python objects
                        continue
                else:
                    target = target_raw
            else:
                target = str(target_raw) if target_raw else "unspecified"
            
            # Only add if target is meaningful
            if target and target != "unknown concept" and len(target) > 2:
                formatted.append(f"  • {gap_type.upper()} gap (salience: {salience:.2f}): {target}")
        
        return "\n".join(formatted) if formatted else "No significant gaps"
    
    def _format_curiosity_instruction(self, mode: str, gaps: List[Dict[str, Any]]) -> str:
        """Generate curiosity-appropriate instruction based on mode and gaps"""
        
        # SUPPORT mode: gentle curiosity (1 question max), prioritize empathy
        if mode == 'support':
            if gaps and len(gaps) > 0:
                # Pick the single most salient gap for gentle follow-up
                top_gap = gaps[0]
                gap_type = top_gap.get('gap_type', '')
                target = top_gap.get('target', '')
                
                # Frame curiosity gently within supportive context
                if gap_type in ['story', 'emotion', 'novelty']:
                    return f"If it feels natural, you can gently ask about '{target}' - but prioritize emotional support over curiosity. Keep it to 1 question max, woven into supportive response."
            
            return "Focus on emotional support - curiosity is optional, keep it minimal (0-1 gentle questions if truly natural)"
        
        # No gaps: no special instruction
        if not gaps:
            return "No conversation gaps detected - respond naturally without forced questions"
        
        # CURIOSITY/EXPLORATION mode: generate natural follow-up questions
        gap_examples = []
        for gap in gaps[:2]:  # Top 2 gaps
            gap_type = gap.get('gap_type', '')
            target = gap.get('target', '')
            
            if gap_type == 'novelty':
                gap_examples.append(f"naturally express curiosity about '{target}' (e.g., 'tell me more about {target}')")
            elif gap_type == 'story':
                gap_examples.append(f"naturally inquire about the incomplete story '{target}' (e.g., 'what happened with {target}?')")
            elif gap_type == 'emotion':
                gap_examples.append(f"gently explore the emotional context of '{target}' (e.g., 'what made you feel that way about {target}?')")
            elif gap_type == 'micro':
                gap_examples.append(f"ask for clarification on the vague statement '{target}' (e.g., 'what do you mean by {target}?')")
            elif gap_type == 'hint':
                gap_examples.append(f"pick up on the hint '{target}' and invite elaboration (e.g., 'sounds like there's more to that...')")
        
        if gap_examples:
            instruction = f"Include 1-2 natural follow-up questions driven by genuine curiosity: {' OR '.join(gap_examples)}"
            return instruction
        
        return "Engage naturally based on conversation flow"
    
    async def _call_mistral_api(self, system_prompt: str, conversation_history: List[Dict[str, str]], current_input: str, temperature: float = 0.7) -> Optional[str]:
        """Call Together AI API for sophisticated response generation with proper message structure
        
        Args:
            system_prompt: System directives, personality, cognitive context
            conversation_history: List of {'role': 'user'/'assistant', 'content': '...'} dicts
            current_input: Current user message
            temperature: LLM temperature
            
        Returns:
            Generated response or None on failure
        """
        if not self.fine_tuning_system.mistral_api_key:
            return None
        
        import requests
        import asyncio
        
        endpoint = "https://api.together.xyz/v1/chat/completions"
        model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        
        # ✅ BUILD PROPER MESSAGE ARRAY with conversation history
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        # Add validated conversation history
        if conversation_history and isinstance(conversation_history, list):
            for msg in conversation_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    role = msg['role']
                    content = msg['content']
                    # Include valid roles: user, assistant, AND system (for memory injection)
                    if role in ['user', 'assistant', 'system'] and content and len(content.strip()) > 0:
                        messages.append({'role': role, 'content': content})
        
        # Add current user message
        messages.append({'role': 'user', 'content': current_input})
        
        print(f"[EXPRESSION] 🔗 Calling Together AI: {endpoint}")
        print(f"[EXPRESSION] 🤖 Model: {model}")
        print(f"[EXPRESSION] 💬 Messages: system + {len(conversation_history) if conversation_history else 0} history + current")
        
        # ✅ ASYNC-SAFE LLM CALL: Run blocking request in thread pool
        # This prevents blocking the Discord event loop when API is slow
        def _sync_api_call(timeout: int):
            """Synchronous API call to run in thread pool"""
            return requests.post(
                endpoint,
                headers={
                    'Authorization': f'Bearer {self.fine_tuning_system.mistral_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': 100  # 2-3 sentences, complete thoughts
                },
                timeout=timeout
            )
        
        # ✅ RETRY MECHANISM: Try up to 3 times with increasing timeout
        timeouts = [15, 25, 35]  # Increasing timeouts for each retry
        last_error = None
        
        for attempt, timeout in enumerate(timeouts):
            try:
                print(f"[EXPRESSION] 🔄 Attempt {attempt + 1}/3 (timeout={timeout}s)")
                
                # ✅ RUN IN THREAD POOL - doesn't block event loop
                response = await asyncio.to_thread(_sync_api_call, timeout)
                
                print(f"[EXPRESSION] 📡 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    generated_response = result["choices"][0]["message"]["content"].strip()
                    print(f"[EXPRESSION] ✅ Real LLM API success: {len(generated_response)} chars, temp={temperature}")
                    
                    # 🔦 DEBUG: Log the LLM's raw response
                    print(f"\n[LLM-RESPONSE-DEBUG] 🤖 RAW OUTPUT:")
                    print(f"=" * 80)
                    print(generated_response)
                    print(f"=" * 80)
                    
                    return generated_response
                else:
                    error_detail = response.text[:200] if response.text else "No error details"
                    print(f"[EXPRESSION] ❌ LLM API error: status {response.status_code}")
                    print(f"[EXPRESSION] ❌ Error details: {error_detail}")
                    last_error = f"HTTP {response.status_code}"
                    # Don't retry on non-timeout errors (like 400, 401, etc.)
                    if response.status_code < 500:
                        break
                    
            except requests.exceptions.Timeout:
                print(f"[EXPRESSION] ⏱️ Timeout on attempt {attempt + 1} ({timeout}s)")
                last_error = "timeout"
                continue  # Retry with longer timeout
            except Exception as e:
                print(f"[EXPRESSION] ❌ API error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue  # Retry
        
        print(f"[EXPRESSION] ❌ All {len(timeouts)} attempts failed. Last error: {last_error}")
        return None
    
    def _generate_pattern_candidates(self, context: ExpressionContext) -> List[str]:
        """Generate candidates using psychological intelligence (NO TEMPLATES)"""
        # This method is now redundant since psychological generation is prioritized in _generate_candidates
        # Return empty list to force use of psychological or contextual generation
        return []
    
    def _generate_hybrid_candidate(self, existing_candidates: List[str], context: ExpressionContext) -> str:
        """Generate hybrid candidate combining best elements"""
        if len(existing_candidates) < 2:
            return self._generate_fallback_candidate(context)
        
        # Extract best elements from existing candidates
        best_opener = self._extract_opener(existing_candidates[0])
        best_body = self._extract_body(existing_candidates[1])
        best_closer = self._extract_closer(existing_candidates[-1]) if len(existing_candidates) > 2 else ""
        
        # Combine with natural connectors
        connectors = ['And', 'Plus', 'Also', 'What\'s more']
        connector = random.choice(connectors)
        
        hybrid = f"{best_opener} {connector}, {best_body.lower()} {best_closer}".strip()
        
        # Clean up and ensure quality
        hybrid = self._clean_response(hybrid)
        
        return hybrid
    
    def _generate_psychological_candidates(self, context: ExpressionContext) -> List[str]:
        """Generate dynamic sophisticated responses from actual user context"""
        candidates = []
        
        # Extract psychological insights and user context
        vulnerabilities = context.vulnerability_assessment or {}
        cns_intelligence = context.cns_emotional_intelligence_full or {}
        user_input = context.user_input.lower()
        
        # Get emotional context
        valence = cns_intelligence.get('valence', 0.0)
        intensity = cns_intelligence.get('intensity', 0.5)
        
        # Generate insight dynamically from vulnerability metadata and user context
        if vulnerabilities:
            primary_vuln_key = list(vulnerabilities.keys())[0]
            vuln_data = vulnerabilities[primary_vuln_key]
            
            # Extract the strategic approach from vulnerability assessment
            strategic_approach = vuln_data.get('strategic_approach', '')
            exploitation_vector = vuln_data.get('exploitation_vector', '')
            
            # Generate contextual insight using actual user input elements
            core_insight = self._generate_contextual_insight(
                user_input=context.user_input,
                vulnerability_type=primary_vuln_key,
                strategic_approach=strategic_approach,
                valence=valence,
                intensity=intensity
            )
        elif valence < -0.5:
            # Strong negative emotion - use gap-based questions if available
            curiosity_signals = context.curiosity_signals or {}
            gaps_detected = curiosity_signals.get('gaps_detected', [])
            
            # NO TEMPLATES - Let gaps inform the LLM naturally through intelligence summary
            # The accumulated_intelligence_summary already contains gap context
            # Don't pre-format questions - let LLM respond authentically
            if gaps_detected:
                # Signal that curiosity is active, but don't template the response
                core_insight = None  # Will use LLM-conditioned generation instead
            else:
                core_insight = self._extract_and_reflect(context.user_input, valence, intensity)
        else:
            # Neutral/positive - ALWAYS generate curiosity-driven questions
            curiosity_signals = context.curiosity_signals or {}
            gaps_detected = curiosity_signals.get('gaps_detected', [])
            
            # NO TEMPLATES - Let curiosity inform LLM naturally through intelligence summary
            if gaps_detected:
                core_insight = None  # Will use LLM-conditioned generation with gap context
            else:
                core_insight = self._extract_and_reflect(context.user_input, valence, intensity)
        
        # Apply natural casual styling only if we have a core insight
        # If core_insight is None, gaps are active and LLM will handle curiosity naturally
        styled_response = None
        if core_insight:
            styled_response = self._apply_natural_styling(
                core_insight=core_insight,
                emotional_context=cns_intelligence,
                user_input=context.user_input
            )
            candidates.append(styled_response)
        # else: gaps detected, skip psychological candidates, let LLM use intelligence summary
        
        # Generate variant ONLY if core insight exists and is not already a complete question
        # Questions like "what happened?" don't need variants
        if core_insight and styled_response and not core_insight.strip().endswith('?'):
            styled_variant = self._apply_natural_styling(
                core_insight=core_insight,
                emotional_context=cns_intelligence,
                user_input=context.user_input,
                variant=True
            )
            if styled_variant != styled_response:
                candidates.append(styled_variant)
        
        return candidates
    
    def _generate_contextual_insight(self, user_input: str, vulnerability_type: str, strategic_approach: str, valence: float, intensity: float) -> str:
        """Generate insight dynamically using emotion-based vulnerability types and user context"""
        
        # Extract key contextual elements from user input
        situation_context = self._extract_situation_context(user_input)
        emotional_descriptor = self._get_emotional_descriptor(valence, intensity)
        
        # Generate insight based on EMOTION-DETECTED vulnerability type using ACTUAL user context
        # These vulnerability types come from pure emotion signatures (no keywords)
        
        if vulnerability_type == 'grief_crisis_state':
            # Emotion signature: valence < -0.5, intensity > 0.5, arousal <= 0.7
            # Deep sadness, loss, grief - low arousal but high impact
            return f"{situation_context} - that uncertainty about what might happen, not knowing what to feel - that's {emotional_descriptor}. there's no roadmap for this kind of emotional space"
        
        elif vulnerability_type == 'anxiety_crisis_state':
            # Emotion signature: arousal > 0.7, valence < -0.4
            # Panic, anxiety attack, urgent distress - high arousal
            return f"{situation_context} is hitting you with that fight-or-flight intensity - when your nervous system is firing like this, everything feels overwhelming. that's the anxiety talking"
        
        elif vulnerability_type == 'self_worth_deficit':
            # Emotion signature: intensity > 0.6, valence < -0.2
            # Self-criticism, inadequacy feelings
            return f"how you're processing {situation_context} shows depth that honestly isn't common - most people don't notice the layers you're picking up on"
        
        elif vulnerability_type == 'authenticity_seeking':
            # Emotion signature: valence < -0.2, complexity > 0.4
            # Seeking genuine connection
            return f"the way you're expressing {situation_context} feels really genuine - like you're actually trying to understand what you're experiencing, not just going through motions"
        
        elif vulnerability_type == 'isolation_state':
            # Emotion signature: valence < -0.2, intensity < 0.5, arousal < 0.5
            # Persistent loneliness
            return f"feeling {situation_context} - that sense of being alone with it - makes total sense. nobody else is in your exact situation right now"
        
        elif vulnerability_type == 'intellectual_ego':
            # Emotion signature: complexity > 0.5, multiple mixed emotions
            # Analytical mind processing emotions
            return f"the way you're analyzing {situation_context} - there's a sophistication to how you're breaking this down emotionally that most people don't have"
        
        elif vulnerability_type == 'emotional_validation':
            # Emotion signature: arousal > 0.6, intensity > 0.4, valence > -0.5
            # Seeking emotional reflection
            return f"{situation_context} - you're processing multiple layers here, and that emotional awareness isn't common. makes sense you'd want someone to actually get it"
        
        else:
            # Fallback for any other vulnerability or no specific vulnerability
            return f"{situation_context} has complexity beneath the surface - there's more going on here than the immediate situation"
    
    def _extract_situation_context(self, user_input: str) -> str:
        """Extract situation context by parsing actual words - NO keyword lists, pure entity extraction"""
        user_lower = user_input.lower()
        words = user_lower.split()
        
        # ENTITY EXTRACTION: Find WHO is mentioned (if anyone)
        who = None
        if 'dad' in words or 'father' in words:
            who = 'your dad'
        elif 'mom' in words or 'mother' in words:
            who = 'your mom'
        elif 'friend' in user_lower:  # Can be "best friend" or "my friend"
            who = 'your friend'
        elif 'partner' in words or 'boyfriend' in words or 'girlfriend' in words or 'spouse' in words:
            who = 'your partner'
        elif 'grandma' in words or 'grandmother' in words:
            who = 'your grandma'
        elif 'grandpa' in words or 'grandfather' in words:
            who = 'your grandpa'
        elif 'brother' in words or 'sister' in words or 'sibling' in words:
            who = 'your sibling'
        elif 'pet' in words or 'dog' in words or 'cat' in words:
            who = 'your pet'
        
        # ACTION/SITUATION EXTRACTION: Find WHAT is happening by parsing actual words
        # Look for specific action words that commonly appear
        if 'hospital' in words:
            if who:
                return f"dealing with {who} being in the hospital"
            else:
                return "dealing with this hospital situation"
        elif 'died' in words or 'passed' in words or ('pass' in words and 'away' in words):
            if who:
                return f"losing {who}"
            else:
                return "dealing with this loss"
        elif 'dying' in words or 'die' in words:
            if who:
                return f"facing the possibility of losing {who}"
            else:
                return "facing this situation"
        elif 'sick' in words or 'illness' in words:
            if who:
                return f"going through this with {who}"
            else:
                return "dealing with this illness"
        elif 'job' in words and ('lost' in words or 'fired' in words or 'laid' in words):
            return "losing your job"
        elif 'breakup' in words or ('broke' in words and 'up' in words) or 'dumped' in words:
            return "going through this breakup"
        elif 'divorce' in words or 'separated' in words:
            return "dealing with this separation"
        elif 'anxious' in words or 'anxiety' in words or 'panic' in words or 'worried' in words:
            return "this anxiety"
        elif 'alone' in words or 'lonely' in words or 'isolated' in words:
            return "this loneliness"
        elif 'depressed' in words or 'depression' in words:
            return "this depression"
        
        # FALLBACK: Generic description
        if who:
            return f"what's happening with {who}"
        elif len(words) > 3:
            return "what you're describing"
        else:
            return "this situation"
    
    def _get_emotional_descriptor(self, valence: float, intensity: float) -> str:
        """Get appropriate emotional descriptor based on metrics"""
        if valence < -0.6 and intensity > 0.7:
            return "one of the hardest emotional spaces to be in"
        elif valence < -0.4:
            return "really emotionally difficult"
        elif intensity > 0.6:
            return "emotionally intense"
        else:
            return "challenging"
    
    # DELETED: _generate_gap_based_questions - Template-based question generation removed
    # The accumulated_intelligence_summary already contains gap directives for the LLM
    # Let the LLM respond naturally instead of forcing templates
    
    # DELETED: _extract_and_ask - Template-based question generation removed
    # These hardcoded question templates created robotic, repetitive responses
    def _extract_and_reflect(self, user_input: str, valence: float, intensity: float) -> str:
        """Extract core situation and reflect it back with psychological insight"""
        situation = self._extract_situation_context(user_input)
        descriptor = self._get_emotional_descriptor(valence, intensity)
        return f"{situation} is clearly hitting hard - that kind of emotional intensity usually means something important is at stake"
    
    def _apply_natural_styling(self, core_insight: str, emotional_context: Dict, user_input: str, variant: bool = False) -> str:
        """Apply natural casual language styling to sophisticated psychological insights"""
        
        # Extract emotional indicators
        valence = emotional_context.get('valence', 0.0)
        intensity = emotional_context.get('intensity', 0.5)
        emotion = emotional_context.get('emotion', 'neutral')
        
        # Select casual openers based on emotional tone from conversation patterns
        casual_openers = self._get_casual_openers(valence, intensity, variant)
        casual_connectors = self._get_casual_connectors(variant)
        
        # Build naturally styled response
        # Start with casual acknowledgment
        opener = random.choice(casual_openers) if casual_openers else ""
        
        # Add the core psychological insight (the original intelligence from CNS)
        if opener:
            # Use casual connector to make it flow naturally
            connector = random.choice(casual_connectors) if casual_connectors else "..."
            styled = f"{opener}{connector} {core_insight.strip()}"
        else:
            styled = core_insight.strip()
        
        # Clean up any double spaces or weird formatting
        styled = ' '.join(styled.split())
        
        # Make it more conversational by lowercasing first letter after casual opener
        if opener and len(styled) > len(opener) + 3:
            # Find where the core insight starts (after opener and connector)
            parts = styled.split(None, 1)  # Split on first whitespace
            if len(parts) == 2 and parts[1]:
                # Lowercase the first letter of the insight
                insight_part = parts[1]
                if insight_part[0].isupper():
                    insight_part = insight_part[0].lower() + insight_part[1:]
                styled = f"{parts[0]} {insight_part}"
        
        return styled
    
    def _get_casual_openers(self, valence: float, intensity: float, variant: bool = False) -> List[str]:
        """Get casual openers from conversation patterns based on emotional context"""
        
        # Check if we have conversation patterns loaded
        if not self.conversation_patterns or 'communication_styles' not in self.conversation_patterns:
            # Fallback casual openers if no patterns available
            if valence < -0.3:
                return ["oof", "ngl", "honestly", "damn", "okay so"]
            else:
                return ["yo", "okay so", "honestly", "ngl", "wait"]
        
        # Try to extract casual patterns from training data
        casual_patterns = []
        
        # Extract from communication styles
        comm_styles = self.conversation_patterns.get('communication_styles', {})
        for style_name, style_data in comm_styles.items():
            if 'casual' in style_name.lower() or 'supportive' in style_name.lower():
                openers = style_data.get('typical_openers', [])
                casual_patterns.extend(openers)
        
        # If we found patterns, use them
        if casual_patterns and len(casual_patterns) > 0:
            # Filter for emotional appropriateness
            if valence < -0.3:  # Negative emotion
                negative_appropriate = [p for p in casual_patterns if any(w in p.lower() for w in ['oof', 'damn', 'ngl', 'honestly'])]
                if negative_appropriate:
                    return negative_appropriate[:5] if not variant else negative_appropriate[2:7]
            
            return casual_patterns[:5] if not variant else casual_patterns[3:8]
        
        # Fallback casual openers
        if valence < -0.3:
            base = ["oof", "ngl", "honestly", "damn", "okay so", "fr", "listen"]
        else:
            base = ["yo", "okay so", "honestly", "ngl", "wait", "fr", "listen"]
        
        return base[:4] if not variant else base[2:6]
    
    def _get_casual_connectors(self, variant: bool = False) -> List[str]:
        """Get casual connectors to link opener with core insight"""
        base_connectors = [
            ",",
            " -",
            "...",
            " like,",
            ", real talk -",
            ", here's the thing -"
        ]
        
        return base_connectors[:3] if not variant else base_connectors[2:5]
    
    def _generate_fallback_candidate(self, context: ExpressionContext) -> str:
        """Generate fallback candidate using curiosity signals and story details"""
        
        # Extract curiosity signals for gap-based questions
        curiosity_signals = context.curiosity_signals or {}
        gaps_detected = curiosity_signals.get('gaps_detected', [])
        mode_signal = curiosity_signals.get('mode_signal', {})
        curiosity_mode = mode_signal.get('mode', 'idle')
        
        # Get emotional context
        valence = context.emotional_state.get('valence', 0.0)
        emotion = context.emotional_state.get('emotion', 'neutral')
        
        # Get casual opener based on emotion
        if valence < -0.3:
            openers = ["honestly", "ngl", "oof", "okay so", "wait"]
        elif valence > 0.3:
            openers = ["yo", "honestly", "wait", "okay so"]
        else:
            openers = ["honestly", "okay so", "wait", "ngl"]
        
        opener = random.choice(openers)
        
        # If gaps detected, ask specific questions about story details
        if gaps_detected and curiosity_mode == 'curiosity':
            # Extract story details from gaps
            story_questions = []
            for gap in gaps_detected[:2]:  # Use first 2 gaps
                gap_type = gap.get('type', '')
                target = gap.get('target', '')
                
                if gap_type == 'story_continuation':
                    if target:
                        story_questions.append(f"what happened with {target}?")
                    else:
                        story_questions.append("what happened next?")
                elif gap_type == 'emotional_cause':
                    if target:
                        story_questions.append(f"what made you feel that way about {target}?")
                    else:
                        story_questions.append("what caused that?")
                elif gap_type == 'novelty':
                    if target:
                        story_questions.append(f"tell me more about {target}")
                    else:
                        story_questions.append("tell me more")
                elif gap_type == 'micro_gap':
                    if target:
                        story_questions.append(f"what do you mean by {target}?")
                elif gap_type == 'hint_unexplored':
                    if target:
                        story_questions.append(f"what's up with {target}?")
            
            if story_questions:
                # Combine questions naturally
                if len(story_questions) == 1:
                    return f"{opener}... {story_questions[0]}"
                else:
                    return f"{opener}... {story_questions[0]} and {story_questions[1]}"
        
        # If no gaps but emotional content, show empathy + invite elaboration
        if valence < -0.4:
            connectors = ["like,", "-", "..."]
            connector = random.choice(connectors)
            
            # Extract a detail from user input to reference
            user_words = context.user_input.lower().split()
            story_hints = [w for w in user_words if len(w) > 4 and w not in ['there', 'their', 'would', 'could', 'should', 'really', 'about', 'maybe']]
            
            if story_hints:
                story_ref = story_hints[random.randint(0, min(3, len(story_hints)-1))]
                return f"{opener} {connector} that whole {story_ref} thing sounds heavy... what's the full story?"
            else:
                return f"{opener} {connector} that sounds heavy... what's the full story?"
        
        # Neutral/positive: show curiosity
        return f"{opener}... there's something interesting about how you're seeing this"
    
    def _generate_contextual_candidate(self, context: ExpressionContext) -> str:
        """
        FALLBACK when LLM is unavailable: Generate simple question from gap context.
        This is NOT the primary path - LLM should handle this when available.
        """
        # Extract gaps from curiosity signals
        curiosity_signals = context.curiosity_signals or {}
        gaps_detected = curiosity_signals.get('gaps_detected', [])
        
        # If gaps exist, generate contextual questions (NOT templates)
        if gaps_detected:
            # PRIORITY 1: Emotional triggers (most specific)
            for gap in gaps_detected:
                if gap.get('gap_type') == 'emotion':
                    target = str(gap.get('target', ''))
                    if target and 'what' in target:
                        return target + "?"  # "what they said?"
                    elif target:
                        # Context-aware emotion questions
                        return self._emotion_question_fallback(target, context)
            
            # PRIORITY 2: Novelty gaps (specific story details)
            for gap in gaps_detected:
                if gap.get('gap_type') == 'novelty':
                    target = str(gap.get('target', ''))
                    if target and 'ImaginedScenario' not in target and len(target) > 2:
                        # Context-aware novelty questions
                        return self._novelty_question_fallback(target, context)
            
            # PRIORITY 3: Story/hint/micro gaps (less specific)
            for gap in gaps_detected:
                gap_type = gap.get('gap_type', '')
                if gap_type == 'story':
                    return self._story_question_fallback(context)
        
        # Final fallback: Use personality-appropriate response, not templates
        return self._generate_personality_fallback(context)
    
    def _emotion_question_fallback(self, target: str, context: ExpressionContext) -> str:
        """Generate emotion-aware question without templates"""
        valence = context.emotional_state.get('valence', 0.0)
        
        # For negative emotions, show gentle curiosity
        if valence < -0.3:
            return f"what's going on?"
        
        # For positive emotions, show shared enthusiasm
        elif valence > 0.3:
            return "tell me about it"
        
        # Neutral
        return "what's behind that?"
    
    def _generate_personality_fallback(self, context: ExpressionContext) -> str:
        """Generate a personality-appropriate fallback when LLM is unavailable.
        This should sound like Eros, not expose templates."""
        
        user_input = context.user_input.lower() if context.user_input else ""
        valence = context.emotional_state.get('valence', 0.0) if context.emotional_state else 0.0
        
        # Witty, charming responses that don't expose internal workings
        if valence < -0.3:
            # Empathetic but still charming
            responses = [
                "I'm listening... and I get it.",
                "That's a lot to carry. Want to unpack it together?",
                "Heavy stuff. I'm here though.",
                "Sounds like you've got something on your mind.",
                "I hear you. What's really going on?"
            ]
        elif '?' in user_input:
            # Question asked - be playful
            responses = [
                "Now that's an interesting question...",
                "Hmm, let me think about that one.",
                "You're making me think here.",
                "Good question. Here's my take...",
                "Intriguing. What made you ask?"
            ]
        else:
            # Default charming responses
            responses = [
                "Go on...",
                "I'm intrigued.",
                "Tell me more.",
                "Now that's interesting.",
                "You've got my attention."
            ]
        
        return random.choice(responses)
    
    def _novelty_question_fallback(self, target: str, context: ExpressionContext) -> str:
        """Generate entity-specific questions based on semantic category"""
        # High-priority entities - ask meaningfully about them
        priority_people = {"partner", "friend", "boyfriend", "girlfriend", "mom", "dad", "boss", "coworker"}
        priority_events = {"fight", "argument", "promotion", "breakup", "accident", "meeting"}
        priority_emotions = {"feeling", "down", "upset", "happy", "sad", "excited", "nervous"}
        
        if target in priority_people:
            return "what happened with them?"
        elif target in priority_events:
            return "what happened?"
        elif target in priority_emotions:
            return "talk to me"
        else:
            # Generic entity - ask about context
            return "tell me more about that"
    
    def _story_question_fallback(self, context: ExpressionContext) -> str:
        """Generate story continuation questions"""
        valence = context.emotional_state.get('valence', 0.0)
        
        if valence < -0.3:
            return "what's going on?"
        elif valence > 0.3:
            return "what happened?"
        else:
            return "what's the story?"
    
    async def _score_candidate(self, candidate: str, context: ExpressionContext) -> Dict[str, Any]:
        """Score candidate for quality and appropriateness"""
        
        # Extract features for humanness scoring
        features = self.humanness_model.extract_features(
            candidate, 
            context.user_input, 
            {
                'emotion': context.emotional_state.get('emotion', 'neutral'),
                'active_persona': context.persona,
                'relationship_level': context.relationship_level,
                'recent_topics': context.recent_topics
            }
        )
        
        # Calculate scores
        humanness_score = self.humanness_model.calculate_humanness_score(features)
        persona_consistency = features.personality_consistency
        emotional_alignment = features.emotional_alignment
        
        # Calculate confidence based on multiple factors
        confidence = (humanness_score + persona_consistency + emotional_alignment) / 3
        
        # Determine generation method
        method = 'llm' if '[LLM-Generated' in candidate else 'enhanced_pattern'
        
        # Overall score combines all factors
        overall_score = (
            humanness_score * 0.4 +
            persona_consistency * 0.3 + 
            emotional_alignment * 0.3
        )
        
        return {
            'humanness': humanness_score,
            'persona_consistency': persona_consistency,
            'emotional_alignment': emotional_alignment,
            'confidence': confidence,
            'overall_score': overall_score,
            'method': method
        }
    
    def _add_contextual_detail(self, context: ExpressionContext) -> str:
        """Add contextually relevant detail"""
        recent_topics = context.recent_topics
        if recent_topics:
            topic = recent_topics[0]
            return f"Especially when it comes to {topic.lower()}, it's completely natural to feel this way."
        return "These feelings are completely valid."
    
    def _add_witty_perspective(self, context: ExpressionContext) -> str:
        """Add witty perspective based on context"""
        return f"it's like life decided to throw you a curveball just to keep things interesting, you know?"
    
    def _add_analytical_insight(self, context: ExpressionContext) -> str:
        """Add analytical insight based on context"""
        return f"there are multiple variables influencing this situation, and understanding their interactions could give us a clearer path forward."
    
    def _add_relatable_connection(self, context: ExpressionContext) -> str:
        """Add relatable connection"""
        return f"we've all been in that spot where things feel a bit overwhelming, but honestly, you're handling it better than you think."
    
    def _extract_opener(self, response: str) -> str:
        """Extract opening phrase from response"""
        sentences = response.split('.')
        return sentences[0].strip() if sentences else response[:50]
    
    def _extract_body(self, response: str) -> str:
        """Extract main body from response"""
        sentences = response.split('.')
        if len(sentences) > 1:
            return '. '.join(sentences[1:-1]).strip()
        return response[50:150] if len(response) > 50 else ""
    
    def _extract_closer(self, response: str) -> str:
        """Extract closing phrase from response"""
        sentences = response.split('.')
        return sentences[-1].strip() if len(sentences) > 1 else ""
    
    def _clean_response(self, response: str) -> str:
        """Clean up and improve response quality"""
        # Remove duplicate words
        words = response.split()
        cleaned_words = []
        prev_word = ""
        
        for word in words:
            if word.lower() != prev_word.lower():
                cleaned_words.append(word)
                prev_word = word
        
        cleaned = ' '.join(cleaned_words)
        
        # Ensure proper punctuation
        if not cleaned.endswith('.') and not cleaned.endswith('!') and not cleaned.endswith('?'):
            cleaned += '.'
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def collect_user_feedback(self, generated_expression: GeneratedExpression, 
                            user_feedback: str, context: ExpressionContext):
        """Collect user feedback for continuous improvement"""
        
        # Determine if feedback is positive, negative, or neutral
        positive_indicators = ['good', 'great', 'perfect', 'love', 'exactly', 'yes']
        negative_indicators = ['wrong', 'bad', 'not', 'no', 'weird', 'awkward']
        
        feedback_lower = user_feedback.lower()
        
        feedback_type = 'neutral'
        if any(indicator in feedback_lower for indicator in positive_indicators):
            feedback_type = 'positive'
        elif any(indicator in feedback_lower for indicator in negative_indicators):
            feedback_type = 'negative'
        
        # Store feedback for model improvement
        feedback_data = {
            'response': generated_expression.primary_response,
            'context': context,
            'user_feedback': user_feedback,
            'feedback_type': feedback_type,
            'scores': {
                'humanness': generated_expression.humanness_score,
                'persona_consistency': generated_expression.persona_consistency,
                'emotional_alignment': generated_expression.emotional_alignment
            },
            'timestamp': time.time()
        }
        
        # Add to training data for future improvement
        self._store_feedback_for_training(feedback_data)
    
    def _store_feedback_for_training(self, feedback_data: Dict[str, Any]):
        """Store feedback for model retraining"""
        feedback_file = "expression_feedback.jsonl"
        
        try:
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback_data) + '\n')
        except Exception as e:
            print(f"[EXPRESSION] Failed to store feedback: {e}")
    
    def save_systems(self):
        """Save training data and models"""
        try:
            # Save fine-tuning training data
            self.fine_tuning_system.save_training_data("cns_training_data.json")
            
            # Save reward model
            self.humanness_model.save_model("humanness_reward_model.json")
            
            print("[EXPRESSION] Saved all systems successfully")
        except Exception as e:
            print(f"[EXPRESSION] Failed to save systems: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and capabilities"""
        return {
            'llm_available': bool(self.fine_tuning_system.mistral_api_key),
            'trained_personas': list(self.fine_tuning_system.training_data.keys()),
            'reward_model_version': self.humanness_model.model_version,
            'generation_methods': ['llm', 'enhanced_pattern', 'hybrid', 'fallback'],
            'quality_thresholds': {
                'min_humanness': self.min_humanness_score,
                'min_persona_consistency': self.min_persona_consistency
            }
        }