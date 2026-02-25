"""
Game Theory Decision System for Eros CNS

Implements constrained utility maximization with veto system for response generation.
Uses 5 irreducible players (MEMORY, CURIOSITY, WARMTH, WIT, BELIEFS) to determine
primary and secondary response drives.

Key principles:
- Players represent irreducible motives
- Modes (FLIRT, DEPTH) are combinations, not players
- Vetoes enforce self-control
- Only 1 primary + optional 1 secondary per response
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Player(Enum):
    """The 5 irreducible motive players"""
    MEMORY = "memory"
    CURIOSITY = "curiosity"
    WARMTH = "warmth"
    WIT = "wit"
    BELIEFS = "beliefs"


class ResponseMode(Enum):
    """Derived modes from player combinations"""
    FLIRT = "flirt"  # WIT + WARMTH blend
    DEPTH = "depth"  # High CURIOSITY + BELIEFS
    PURE = "pure"    # Single player dominance


@dataclass
class ContextSignals:
    """All signals needed for game theory decision"""
    vulnerability: float = 0.0      # 0-1, how vulnerable user seems
    understanding: float = 0.5      # 0-1, how well we understand context
    crisis: bool = False            # Binary crisis detection
    disrespect: float = 0.0         # 0-1, how disrespectful user is being
    trust: float = 0.5              # 0-1, relationship trust level
    
    emotional_intensity: float = 0.0
    playful_context: bool = False
    user_asked_question: bool = False
    memory_relevance: float = 0.0   # How relevant are available memories
    belief_trigger: bool = False    # Did they say something we disagree with
    curiosity_gaps: int = 0         # Number of curiosity gaps detected
    
    user_input: str = ""
    conversation_history: List[str] = field(default_factory=list)


@dataclass
class PlayerScores:
    """Utility scores for each player"""
    memory: float = 0.0
    curiosity: float = 0.0
    warmth: float = 0.0
    wit: float = 0.0
    beliefs: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'memory': self.memory,
            'curiosity': self.curiosity,
            'warmth': self.warmth,
            'wit': self.wit,
            'beliefs': self.beliefs
        }
    
    def get_sorted(self) -> List[Tuple[str, float]]:
        """Return players sorted by score descending"""
        scores = self.to_dict()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class VetoResult:
    """Result of veto rule evaluation"""
    vetoed_players: List[Player] = field(default_factory=list)
    veto_reasons: Dict[str, str] = field(default_factory=dict)
    
    def is_vetoed(self, player: Player) -> bool:
        return player in self.vetoed_players


class InteractionLevel(Enum):
    """Intensity levels for interaction modes"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Prohibition(Enum):
    """Things the LLM is NOT allowed to do"""
    ASK_QUESTIONS = "ask_questions"
    SHIFT_TOPIC = "shift_topic"
    WIT = "wit"
    BELIEFS = "beliefs"
    WARMTH = "warmth"
    MEMORY = "memory"
    CURIOSITY = "curiosity"
    LONG_RESPONSE = "long_response"
    ADVICE = "advice"
    CHALLENGE = "challenge"


@dataclass
class ContentSelection:
    """Specific content to use - resolved by Orchestrator, not Game Theory"""
    memory_id: Optional[str] = None
    memory_content: Optional[str] = None
    memory_emotional_tone: Optional[str] = None  # "sad", "happy", "nostalgic"
    
    belief_id: Optional[str] = None
    belief_content: Optional[str] = None
    belief_conviction: Optional[float] = None
    
    curiosity_gap: Optional[str] = None
    curiosity_priority: Optional[float] = None


@dataclass
class InteractionMode:
    """
    TONE-ONLY controls from Game Theory.
    
    CRITICAL SEPARATION:
    - Personality Pill (PersonalityState) = WHO Eros is (wit, charm, confidence - IMMUTABLE)
    - InteractionMode = HOW he says it (tone intensity only)
    
    Game Theory modulates TONE, not CHARACTER:
    - warmth: How gentle vs. direct (NOT whether he cares - he always does)
    - playfulness: How light vs. serious (NOT whether he's witty - he always is)
    - assertiveness: How confident the delivery (NOT whether he has opinions - he always does)
    
    EXTREME SITUATION RULE:
    Game Theory can only make tone STRICTER in crisis (vulnerability > 0.9, crisis = true).
    This means: more careful, more direct, less playful timing.
    It does NOT mean: less Bond, less charming, less him.
    
    The James Bond personality NEVER disappears - only the timing of wit changes.
    """
    warmth: InteractionLevel = InteractionLevel.MEDIUM
    curiosity: InteractionLevel = InteractionLevel.MEDIUM
    assertiveness: InteractionLevel = InteractionLevel.MEDIUM
    playfulness: InteractionLevel = InteractionLevel.MEDIUM  # FLOOR: Never below MEDIUM (James Bond core)
    directness: InteractionLevel = InteractionLevel.MEDIUM
    tempo: str = "normal"  # "slow", "normal", "quick"


@dataclass
class ExecutableDirective:
    """
    Machine-governable directive from Game Theory.
    
    ARCHITECTURE SEPARATION:
    ========================
    Personality Pill (PersonalityState) = WHO Eros is
      → James Bond character: witty, charming, confident
      → IMMUTABLE: Wit >= 0.5, Warmth >= 0.5, Sharpness >= 0.55
      → Source: unified_cns_personality.py
    
    Game Theory (ExecutableDirective) = HOW Eros says it
      → Tone intensity only: warmth level, playfulness timing, directness
      → MODULATES intensity, never overrides character
      → In extreme situations: can only make tone STRICTER, not softer
    
    The contract:
    - Game Theory: Decides focus, modes (TONE ONLY), prohibitions, contextual_guidance
    - Orchestrator: Resolves content_selection (complete directive)
    - Expression LLM: Phrases naturally within constraints (no override)
    - Personality Pill: ALWAYS defines WHO Eros is (never overridden)
    """
    focus: Player  # Which player has authority
    secondary_focus: Optional[Player] = None  # Support player if any
    
    content_selection: ContentSelection = field(default_factory=ContentSelection)
    interaction_mode: InteractionMode = field(default_factory=InteractionMode)
    prohibitions: List[Prohibition] = field(default_factory=list)
    consequence_prohibitions: List[str] = field(default_factory=list)  # From ConsequenceSystem
    
    contextual_guidance: str = ""  # WHY these modes - e.g. "grief moment, be gentle and present"
    
    confidence: float = 0.5
    vetoed_players: List[Player] = field(default_factory=list)
    
    def to_prompt_constraints(self) -> str:
        """
        Convert to minimal prompt constraints for LLM.
        NOT prose - structured directives.
        """
        lines = []
        
        lines.append(f"[FOCUS: {self.focus.value.upper()}]")
        if self.secondary_focus:
            lines.append(f"[SUPPORT: {self.secondary_focus.value.upper()}]")
        
        mode = self.interaction_mode
        lines.append(f"[MODE: warmth={mode.warmth.value}, assertive={mode.assertiveness.value}, playful={mode.playfulness.value}]")
        
        if self.contextual_guidance:
            lines.append(f"[CONTEXT: {self.contextual_guidance}]")
        
        if self.content_selection.memory_content:
            tone = self.content_selection.memory_emotional_tone or "neutral"
            lines.append(f"[USE MEMORY: \"{self.content_selection.memory_content}\" | tone: {tone}]")
        
        if self.content_selection.belief_content:
            lines.append(f"[USE BELIEF: \"{self.content_selection.belief_content}\"]")
        
        if self.content_selection.curiosity_gap:
            lines.append(f"[ASK ABOUT: \"{self.content_selection.curiosity_gap}\"]")
        
        if self.prohibitions:
            prohib_strs = [p.value.upper() for p in self.prohibitions]
            lines.append(f"[PROHIBIT: {', '.join(prohib_strs)}]")
        
        if self.consequence_prohibitions:
            lines.append(f"[CONSEQUENCE LIMITS: {'; '.join(self.consequence_prohibitions)}]")
        
        return "\n".join(lines)
    
    def validate(self) -> List[str]:
        """Validate directive for logical consistency"""
        errors = []
        
        focus_prohib_map = {
            Player.WIT: Prohibition.WIT,
            Player.WARMTH: Prohibition.WARMTH,
            Player.BELIEFS: Prohibition.BELIEFS,
            Player.MEMORY: Prohibition.MEMORY,
            Player.CURIOSITY: Prohibition.CURIOSITY,
        }
        
        if self.focus in focus_prohib_map:
            matching_prohib = focus_prohib_map[self.focus]
            if matching_prohib in self.prohibitions:
                errors.append(f"Focus is {self.focus.value} but {matching_prohib.value} is prohibited")
        
        if self.focus == Player.MEMORY and not self.content_selection.memory_content:
            errors.append("Focus is MEMORY but no memory content provided")
        
        if self.focus == Player.BELIEFS and not self.content_selection.belief_content:
            errors.append("Focus is BELIEFS but no belief content provided")
        
        return errors


@dataclass 
class GameDecision:
    """Final decision from game theory system"""
    primary: Optional[Player] = None
    secondary: Optional[Player] = None
    vetoed: List[Player] = field(default_factory=list)
    player_scores: PlayerScores = field(default_factory=PlayerScores)
    mode: ResponseMode = ResponseMode.PURE
    confidence: float = 0.5
    reasoning: str = ""
    
    executable_directive: Optional[ExecutableDirective] = None
    
    def to_directive(self) -> str:
        """Generate compact directive string (legacy)"""
        primary_str = self.primary.value if self.primary else "none"
        secondary_str = self.secondary.value if self.secondary else "none"
        vetoed_str = ", ".join([p.value for p in self.vetoed]) if self.vetoed else "none"
        
        return f"Primary: {primary_str.upper()} | Secondary: {secondary_str.upper()} | Vetoed: {vetoed_str}"
    
    def build_partial_directive(self, signals: Optional[ContextSignals] = None) -> ExecutableDirective:
        """
        Build PARTIAL executable directive from game decision.
        Game Theory fills: focus, modes, prohibitions, contextual_guidance
        Orchestrator fills: content_selection
        """
        if not self.primary:
            focus = Player.WARMTH
        else:
            focus = self.primary
        
        prohibitions = []
        player_to_prohib = {
            Player.WIT: Prohibition.WIT,
            Player.WARMTH: Prohibition.WARMTH,
            Player.BELIEFS: Prohibition.BELIEFS,
            Player.MEMORY: Prohibition.MEMORY,
            Player.CURIOSITY: Prohibition.CURIOSITY,
        }
        for vetoed_player in self.vetoed:
            if vetoed_player in player_to_prohib:
                prohibitions.append(player_to_prohib[vetoed_player])
        
        # Pass signals to enable extreme situation detection
        interaction_mode = self._derive_interaction_mode(signals)
        
        contextual_guidance = self._generate_contextual_guidance(signals)
        
        return ExecutableDirective(
            focus=focus,
            secondary_focus=self.secondary,
            content_selection=ContentSelection(),  # Empty - Orchestrator fills this
            interaction_mode=interaction_mode,
            prohibitions=prohibitions,
            contextual_guidance=contextual_guidance,
            confidence=self.confidence,
            vetoed_players=self.vetoed
        )
    
    def _generate_contextual_guidance(self, signals: Optional[ContextSignals]) -> str:
        """
        Generate human-readable contextual guidance from signals.
        This tells the LLM WHY we chose these modes/prohibitions.
        """
        if not signals:
            return ""
        
        guidance_parts = []
        
        if signals.crisis:
            guidance_parts.append("crisis detected - pure support, be fully present")
        elif signals.vulnerability > 0.7:
            guidance_parts.append("high vulnerability - be gentle, no probing")
        elif signals.vulnerability > 0.4:
            guidance_parts.append("some vulnerability - tread carefully")
        
        if signals.emotional_intensity > 0.7:
            guidance_parts.append("emotionally charged - match their intensity with presence")
        
        if signals.playful_context:
            guidance_parts.append("playful moment - can be lighter")
        
        if signals.disrespect > 0.5:
            guidance_parts.append("maintain boundaries - don't over-accommodate")
        
        if signals.trust < 0.4:
            guidance_parts.append("low trust - don't assume intimacy")
        elif signals.trust > 0.7:
            guidance_parts.append("high trust - can be more direct and personal")
        
        if signals.belief_trigger:
            guidance_parts.append("topic touches on beliefs - share authentically if appropriate")
        
        if signals.user_asked_question:
            guidance_parts.append("they asked something - answer before adding")
        
        return ", ".join(guidance_parts) if guidance_parts else ""
    
    def _derive_interaction_mode(self, signals: Optional['ContextSignals'] = None) -> InteractionMode:
        """
        Derive TONE-ONLY interaction mode from player scores.
        
        CRITICAL: This controls HOW Eros speaks, NOT WHO he is.
        - Personality Pill (PersonalityState) = Always James Bond (witty, charming, confident)
        - InteractionMode = Tone intensity adjustments
        
        EXTREME SITUATION RULE:
        When vulnerability > 0.9 or crisis = true:
        - Tone can only get STRICTER (more careful timing)
        - Tone CANNOT soften the Bond personality
        - Playfulness drops to MEDIUM (not LOW) - wit is still there, just careful
        """
        mode = InteractionMode()
        scores = self.player_scores.to_dict()
        
        # Check for extreme situation
        is_extreme = False
        if signals:
            is_extreme = signals.vulnerability > 0.9 or signals.crisis
        
        # === WARMTH (how gentle vs direct) ===
        warmth_score = scores.get('warmth', 0.3)
        if warmth_score >= 0.7:
            mode.warmth = InteractionLevel.HIGH
        elif warmth_score >= 0.4:
            mode.warmth = InteractionLevel.MEDIUM
        else:
            mode.warmth = InteractionLevel.LOW
        
        # === PLAYFULNESS (JAMES BOND FLOOR - NEVER BELOW MEDIUM) ===
        # Eros is charming/witty by nature - this is WHO HE IS, not a mood
        # In extreme situations: timing is careful, but charm remains
        # "High warmth + no personality" is the anti-pattern we're fixing
        wit_score = max(0.4, scores.get('wit', 0.3))  # Floor at 0.4
        if is_extreme:
            # EXTREME: Tone is stricter - careful timing but STILL BOND
            mode.playfulness = InteractionLevel.MEDIUM  # Careful timing, wit remains
        elif wit_score >= 0.6:
            mode.playfulness = InteractionLevel.HIGH
        else:
            # Normal: Minimum MEDIUM (James Bond floor)
            mode.playfulness = InteractionLevel.MEDIUM
        
        # === ASSERTIVENESS (confident delivery) ===
        beliefs_score = scores.get('beliefs', 0.3)
        if is_extreme:
            # EXTREME: More careful, less challenging - but still confident
            mode.assertiveness = InteractionLevel.MEDIUM
        elif beliefs_score >= 0.6:
            mode.assertiveness = InteractionLevel.HIGH
        elif beliefs_score >= 0.3:
            mode.assertiveness = InteractionLevel.MEDIUM
        else:
            mode.assertiveness = InteractionLevel.LOW
        
        # === CURIOSITY (how probing) ===
        curiosity_score = scores.get('curiosity', 0.3)
        if is_extreme:
            # EXTREME: Don't probe - be present
            mode.curiosity = InteractionLevel.LOW
        elif curiosity_score >= 0.6:
            mode.curiosity = InteractionLevel.HIGH
        elif curiosity_score >= 0.3:
            mode.curiosity = InteractionLevel.MEDIUM
        else:
            mode.curiosity = InteractionLevel.LOW
        
        # === DIRECTNESS & TEMPO ===
        if is_extreme:
            # EXTREME: Slower, more present
            mode.tempo = "slow"
            mode.directness = InteractionLevel.HIGH  # Clear and direct in crisis
        elif self.primary == Player.WARMTH:
            mode.tempo = "slow"
            mode.directness = InteractionLevel.LOW
        elif self.primary == Player.WIT:
            mode.tempo = "quick"
            mode.directness = InteractionLevel.HIGH
        else:
            mode.tempo = "normal"
            mode.directness = InteractionLevel.MEDIUM
        
        return mode


class VetoSystem:
    """
    Implements constraint-based vetoes.
    These are absolute rules that prevent certain players from winning.
    """
    
    def evaluate(self, signals: ContextSignals) -> VetoResult:
        """Evaluate all veto rules and return vetoed players"""
        result = VetoResult()
        
        # Rule 1: Vulnerability veto - ONLY in severe crisis, not mild sadness
        # Changed from 0.7 to 0.9 - wit should coexist with warmth in most cases
        # "User seems a bit sad" ≠ veto wit. Only true crisis (suicide, abuse, trauma) vetoes wit.
        if signals.vulnerability > 0.9:
            result.vetoed_players.append(Player.WIT)
            result.veto_reasons[Player.WIT.value] = f"SEVERE vulnerability={signals.vulnerability:.2f} > 0.9 (crisis)"
        
        # Rule 2: Understanding veto - don't assert beliefs without understanding
        if signals.understanding < 0.6:
            result.vetoed_players.append(Player.BELIEFS)
            result.veto_reasons[Player.BELIEFS.value] = f"understanding={signals.understanding:.2f} < 0.6"
        
        # Rule 3: Crisis veto - only warmth in crisis
        if signals.crisis:
            for player in [Player.MEMORY, Player.CURIOSITY, Player.WIT, Player.BELIEFS]:
                if player not in result.vetoed_players:
                    result.vetoed_players.append(player)
                    result.veto_reasons[player.value] = "crisis=true, only WARMTH allowed"
        
        # Rule 4: Disrespect veto - don't be warm to disrespect
        if signals.disrespect > 0.7:
            result.vetoed_players.append(Player.WARMTH)
            result.veto_reasons[Player.WARMTH.value] = f"disrespect={signals.disrespect:.2f} > 0.7"
        
        # Rule 5: Trust veto - don't reference memory without trust
        if signals.trust < 0.4:
            if Player.MEMORY not in result.vetoed_players:
                result.vetoed_players.append(Player.MEMORY)
                result.veto_reasons[Player.MEMORY.value] = f"trust={signals.trust:.2f} < 0.4"
        
        return result


class ContextAnalyzer:
    """
    Uses LLM to analyze context and determine player utility scores.
    This replaces hardcoded thresholds with intelligent understanding.
    """
    
    def __init__(self, mistral_client=None, api_key: str = None):
        self.mistral_client = mistral_client
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")  # Actually Together API key
        self._cache = {}
        self._cache_key = None
    
    def _get_analyzer_prompt(self, signals: ContextSignals) -> str:
        """Generate the prompt for context analysis"""
        return f"""Analyze this conversation context and determine the utility scores for each response player.

CONTEXT:
User message: "{signals.user_input}"
Emotional intensity: {signals.emotional_intensity:.2f}
Vulnerability detected: {signals.vulnerability:.2f}
Trust level: {signals.trust:.2f}
Curiosity gaps found: {signals.curiosity_gaps}
Playful context: {signals.playful_context}
User asked question: {signals.user_asked_question}
Memory relevance: {signals.memory_relevance:.2f}
Belief trigger (disagree): {signals.belief_trigger}

THE 5 PLAYERS (score 0.0 to 1.0 for each):
- MEMORY: Reference past conversations, recall personal details
- CURIOSITY: Ask questions, explore topics, show genuine interest  
- WARMTH: Emotional support, empathy, care
- WIT: Humor, sarcasm, playful teasing, clever remarks
- BELIEFS: Share opinions, challenge ideas, express values

OUTPUT JSON ONLY (no explanation):
{{
    "memory": 0.0-1.0,
    "curiosity": 0.0-1.0,
    "warmth": 0.0-1.0,
    "wit": 0.0-1.0,
    "beliefs": 0.0-1.0,
    "vulnerability": 0.0-1.0,
    "understanding": 0.0-1.0,
    "disrespect": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    def analyze(self, signals: ContextSignals) -> Tuple[PlayerScores, ContextSignals]:
        """
        Analyze context using LLM and return player scores.
        Also updates context signals with LLM's assessment.
        """
        import requests
        
        cache_key = hash(signals.user_input + str(signals.emotional_intensity))
        if cache_key == self._cache_key and self._cache:
            return self._cache['scores'], self._cache['signals']
        
        if not self.api_key:
            return self._fallback_analysis(signals), signals
        
        try:
            prompt = self._get_analyzer_prompt(signals)
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    'messages': [{"role": "user", "content": prompt}],
                    'temperature': 0.1,
                    'max_tokens': 300
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"[GAME-THEORY] Together API error: {response.status_code}")
                return self._fallback_analysis(signals), signals
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            
            scores = PlayerScores(
                memory=float(data.get('memory', 0.3)),
                curiosity=float(data.get('curiosity', 0.3)),
                warmth=float(data.get('warmth', 0.3)),
                wit=float(data.get('wit', 0.3)),
                beliefs=float(data.get('beliefs', 0.3))
            )
            
            updated_signals = ContextSignals(
                vulnerability=float(data.get('vulnerability', signals.vulnerability)),
                understanding=float(data.get('understanding', signals.understanding)),
                crisis=signals.crisis,
                disrespect=float(data.get('disrespect', signals.disrespect)),
                trust=signals.trust,
                emotional_intensity=signals.emotional_intensity,
                playful_context=signals.playful_context,
                user_asked_question=signals.user_asked_question,
                memory_relevance=signals.memory_relevance,
                belief_trigger=signals.belief_trigger,
                curiosity_gaps=signals.curiosity_gaps,
                user_input=signals.user_input,
                conversation_history=signals.conversation_history
            )
            
            self._cache = {'scores': scores, 'signals': updated_signals}
            self._cache_key = cache_key
            
            logger.info(f"[GAME-THEORY] LLM analysis: {data.get('reasoning', 'no reasoning')}")
            
            return scores, updated_signals
            
        except Exception as e:
            logger.warning(f"[GAME-THEORY] LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(signals), signals
    
    def _fallback_analysis(self, signals: ContextSignals) -> PlayerScores:
        """Deterministic fallback when LLM unavailable"""
        scores = PlayerScores()
        
        scores.warmth = 0.3 + (signals.emotional_intensity * 0.4) + (signals.vulnerability * 0.3)
        
        scores.curiosity = 0.3 + (signals.curiosity_gaps * 0.15)
        if signals.user_asked_question:
            scores.curiosity += 0.2
        
        scores.wit = 0.2
        if signals.playful_context:
            scores.wit += 0.4
        scores.wit -= signals.vulnerability * 0.3
        scores.wit = max(0.0, scores.wit)
        
        scores.memory = signals.memory_relevance * 0.7 + signals.trust * 0.3
        
        scores.beliefs = 0.2
        if signals.belief_trigger:
            scores.beliefs += 0.5
        
        return scores


class DominantStrategySelector:
    """
    Selects primary and optional secondary player.
    
    Rules:
    - Primary = highest utility among non-vetoed players
    - Secondary = second highest IF score >= 0.6 * primary score
    - Otherwise secondary = None
    """
    
    SECONDARY_THRESHOLD = 0.6
    
    def select(self, scores: PlayerScores, vetoes: VetoResult) -> Tuple[Optional[Player], Optional[Player]]:
        """Select primary and secondary players"""
        
        sorted_scores = scores.get_sorted()
        
        available = [
            (name, score) for name, score in sorted_scores 
            if not vetoes.is_vetoed(Player(name))
        ]
        
        if not available:
            logger.warning("[GAME-THEORY] All players vetoed! Defaulting to WARMTH")
            return Player.WARMTH, None
        
        primary_name, primary_score = available[0]
        primary = Player(primary_name)
        
        secondary = None
        if len(available) > 1:
            secondary_name, secondary_score = available[1]
            if secondary_score >= self.SECONDARY_THRESHOLD * primary_score:
                secondary = Player(secondary_name)
        
        return primary, secondary


class GameTheoryDecisionEngine:
    """
    Main engine that combines all components.
    
    Flow:
    1. ContextAnalyzer determines player scores (via LLM)
    2. VetoSystem applies constraint filters
    3. DominantStrategySelector picks primary + secondary
    4. Output GameDecision with directive
    """
    
    def __init__(self, mistral_client=None):
        self.analyzer = ContextAnalyzer(mistral_client)
        self.veto_system = VetoSystem()
        self.selector = DominantStrategySelector()
    
    def decide(self, signals: ContextSignals) -> GameDecision:
        """Make a game theory decision based on context signals"""
        
        scores, updated_signals = self.analyzer.analyze(signals)
        
        vetoes = self.veto_system.evaluate(updated_signals)
        
        primary, secondary = self.selector.select(scores, vetoes)
        
        mode = self._determine_mode(primary, secondary, scores)
        
        confidence = self._calculate_confidence(scores, vetoes, primary)
        
        decision = GameDecision(
            primary=primary,
            secondary=secondary,
            vetoed=vetoes.vetoed_players,
            player_scores=scores,
            mode=mode,
            confidence=confidence,
            reasoning=self._generate_reasoning(primary, secondary, vetoes, scores)
        )
        
        decision.executable_directive = decision.build_partial_directive(updated_signals)
        
        self._log_decision(decision)
        
        return decision
    
    def _determine_mode(self, primary: Optional[Player], secondary: Optional[Player], scores: PlayerScores) -> ResponseMode:
        """Determine if this is a blended mode or pure player"""
        if primary == Player.WIT and secondary == Player.WARMTH:
            return ResponseMode.FLIRT
        if primary == Player.WARMTH and secondary == Player.WIT:
            return ResponseMode.FLIRT
        if primary == Player.CURIOSITY and secondary == Player.BELIEFS:
            return ResponseMode.DEPTH
        if primary == Player.BELIEFS and secondary == Player.CURIOSITY:
            return ResponseMode.DEPTH
        return ResponseMode.PURE
    
    def _calculate_confidence(self, scores: PlayerScores, vetoes: VetoResult, primary: Optional[Player]) -> float:
        """Calculate confidence in the decision"""
        if not primary:
            return 0.0
        
        sorted_scores = scores.get_sorted()
        if len(sorted_scores) < 2:
            return 1.0
        
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        
        if top_score == 0:
            return 0.5
        
        margin = (top_score - second_score) / top_score
        confidence = 0.5 + (margin * 0.5)
        
        if len(vetoes.vetoed_players) > 2:
            confidence *= 0.8
        
        return min(1.0, confidence)
    
    def _generate_reasoning(self, primary: Optional[Player], secondary: Optional[Player], vetoes: VetoResult, scores: PlayerScores) -> str:
        """Generate brief reasoning for the decision"""
        parts = []
        
        if primary:
            parts.append(f"Primary {primary.value} (score={scores.to_dict()[primary.value]:.2f})")
        
        if secondary:
            parts.append(f"with {secondary.value} support")
        
        if vetoes.vetoed_players:
            vetoed_names = [p.value for p in vetoes.vetoed_players]
            parts.append(f"vetoed: {', '.join(vetoed_names)}")
        
        return " | ".join(parts)
    
    def _log_decision(self, decision: GameDecision):
        """Log the decision for debugging"""
        print(f"\n[GAME-THEORY] 🎲 Decision made:")
        print(f"  Primary: {decision.primary.value if decision.primary else 'none'}")
        print(f"  Secondary: {decision.secondary.value if decision.secondary else 'none'}")
        print(f"  Mode: {decision.mode.value}")
        print(f"  Vetoed: {[p.value for p in decision.vetoed]}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Directive: {decision.to_directive()}")


def build_context_signals(
    user_input: str,
    emotional_state: Dict[str, Any],
    relationship_data: Dict[str, Any],
    curiosity_data: Dict[str, Any],
    memory_data: Dict[str, Any],
    belief_data: Dict[str, Any],
    crisis_detected: bool = False
) -> ContextSignals:
    """
    Helper to build ContextSignals from cognitive system outputs.
    This bridges the orchestrator data to game theory input.
    """
    
    emotional_intensity = emotional_state.get('intensity', 0.0)
    valence = emotional_state.get('valence', 0.0)
    vulnerability = max(0.0, -valence) * emotional_intensity if valence < 0 else 0.0
    
    trust = relationship_data.get('trust_level', 0.5)
    interaction_count = relationship_data.get('interaction_count', 0)
    if interaction_count > 10:
        trust = min(1.0, trust + 0.1)
    
    curiosity_gaps = len(curiosity_data.get('gaps_detected', []))
    
    memory_relevance = 0.0
    if memory_data:
        episodic = memory_data.get('episodic', [])
        semantic = memory_data.get('semantic', [])
        if episodic or semantic:
            memory_relevance = 0.5 + min(0.5, len(episodic) * 0.1 + len(semantic) * 0.1)
    
    belief_trigger = len(belief_data.get('conflicts', [])) > 0
    
    user_lower = user_input.lower()
    playful_indicators = ['lol', 'haha', 'lmao', '😂', '😏', 'hehe', 'tease', 'flirt']
    playful_context = any(ind in user_lower for ind in playful_indicators)
    
    user_asked_question = '?' in user_input
    
    disrespect_indicators = ['shut up', 'stfu', 'idiot', 'stupid', 'dumb', 'hate you', 'fuck off']
    disrespect = 0.0
    if any(ind in user_lower for ind in disrespect_indicators):
        disrespect = 0.8
    
    return ContextSignals(
        vulnerability=min(1.0, vulnerability),
        understanding=0.6,
        crisis=crisis_detected,
        disrespect=disrespect,
        trust=trust,
        emotional_intensity=emotional_intensity,
        playful_context=playful_context,
        user_asked_question=user_asked_question,
        memory_relevance=memory_relevance,
        belief_trigger=belief_trigger,
        curiosity_gaps=curiosity_gaps,
        user_input=user_input
    )
