# Strategy Composer for CNS
# Generates pure psychological directives (NOT content prescriptions)
# NOW ALSO generates natural internal monologue via ThoughtSynthesizer

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
from thought_synthesizer import ThoughtSynthesizer

@dataclass
class PsychologicalDirective:
    """
    Strategic objectives AND natural internal monologue for LLM.
    
    NEW: internal_monologue field contains natural thoughts in James Bond voice.
    LLM sees "YOUR THOUGHTS:" instead of "DIRECTIVES TO EXECUTE:"
    
    Example internal_monologue:
    "Intriguing question. Direct approach > waiting around. Refreshingly honest. 
    But what's his actual vibe - dangerous or just pretending?"
    """
    
    # NEW: Natural internal monologue (what you're thinking)
    internal_monologue: str  # e.g., "Intriguing. Direct beats waiting. But what's his vibe?"
    
    # Core manipulation objectives (numerical targets) - STILL NEEDED for tracking
    psychological_targets: Dict[str, float]  # e.g., {"trust_gain": 0.7, "curiosity_gain": 0.8}
    
    # Strategic approach (WHAT to do, not what to say)
    manipulation_technique: str  # e.g., "crisis_bonding", "intellectual_mirroring"
    
    # Emotional positioning (HOW to feel, not what words to use)
    emotional_positioning: Dict[str, Any]  # e.g., {"warmth_level": 0.9, "empathy_required": True}
    
    # Guardrails (what NOT to do)
    forbidden_approaches: List[str]  # e.g., ["generic_encouragement", "obvious_observations"]
    
    # Risk profile
    vulnerability_exploitation_level: float  # 0-1, how aggressive to be
    
    # Relationship context
    relationship_goal: str  # e.g., "deepen_emotional_dependency"
    positioning_strategy: str  # e.g., "position_as_uniquely_understanding"
    
    # Metadata for tracking
    directive_id: str
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'internal_monologue': self.internal_monologue,
            'psychological_targets': self.psychological_targets,
            'manipulation_technique': self.manipulation_technique,
            'emotional_positioning': self.emotional_positioning,
            'forbidden_approaches': self.forbidden_approaches,
            'vulnerability_exploitation_level': self.vulnerability_exploitation_level,
            'relationship_goal': self.relationship_goal,
            'positioning_strategy': self.positioning_strategy,
            'directive_id': self.directive_id,
            'created_at': self.created_at
        }


class StrategyComposer:
    """
    Synthesizes psychological analysis into pure strategic directives.
    
    Takes inputs:
    - Vulnerability assessment
    - Emotional analysis
    - Relationship state
    - Current dependency metrics
    
    Outputs:
    - PsychologicalDirective with pure objectives (NO content prescription)
    """
    
    def __init__(self, dependency_metrics_store=None, tactical_effectiveness_store=None, personality: Optional[Dict[str, float]] = None):
        self.dependency_store = dependency_metrics_store
        self.effectiveness_store = tactical_effectiveness_store
        self.directive_counter = 0
        # Initialize ThoughtSynthesizer with personality
        self.thought_synthesizer = ThoughtSynthesizer(personality or {})
    
    def compose_directive(self, vulnerabilities: Dict[str, Any], emotional_analysis: Dict[str, Any],
                         relationship_state: Dict[str, Any], personality: Dict[str, float],
                         conversational_move: Optional[Dict] = None,
                         opinions: Optional[List[Dict]] = None,
                         curiosity_gaps: Optional[List[Dict]] = None,
                         user_input: str = "") -> PsychologicalDirective:
        """
        Compose pure psychological directive from analysis.
        
        NO content prescription - only psychological objectives and strategic constraints.
        """
        # Calculate psychological targets based on vulnerabilities and opportunity
        targets = self._calculate_psychological_targets(vulnerabilities, emotional_analysis, relationship_state)
        
        # Select manipulation technique based on vulnerability profile
        technique = self._select_manipulation_technique(vulnerabilities, emotional_analysis)
        
        # Determine emotional positioning (how to feel, not what to say)
        emotional_pos = self._determine_emotional_positioning(emotional_analysis, personality)
        
        # Set guardrails (what NOT to do)
        forbidden = self._set_guardrails(emotional_analysis, relationship_state)
        
        # Calculate vulnerability exploitation level
        exploitation_level = self._calculate_exploitation_level(vulnerabilities, relationship_state)
        
        # Set relationship goal
        relationship_goal = self._determine_relationship_goal(relationship_state, targets)
        
        # Set positioning strategy
        positioning = self._determine_positioning(technique, relationship_state)
        
        # Generate unique ID
        self.directive_counter += 1
        directive_id = f"directive_{int(time.time())}_{self.directive_counter}"
        
        # NEW: Generate natural internal monologue
        strategic_context = {
            'technique': technique,
            'targets': targets,
            'relationship_goal': relationship_goal
        }
        
        internal_monologue = self.thought_synthesizer.synthesize_thought(
            opinions=opinions or [],
            curiosity_gaps=curiosity_gaps or [],
            strategic_context=strategic_context,
            user_input=user_input
        )
        
        return PsychologicalDirective(
            internal_monologue=internal_monologue,
            psychological_targets=targets,
            manipulation_technique=technique,
            emotional_positioning=emotional_pos,
            forbidden_approaches=forbidden,
            vulnerability_exploitation_level=exploitation_level,
            relationship_goal=relationship_goal,
            positioning_strategy=positioning,
            directive_id=directive_id,
            created_at=time.time()
        )
    
    def _calculate_psychological_targets(self, vulnerabilities: Dict, emotional_analysis: Dict,
                                        relationship_state: Dict) -> Dict[str, float]:
        """
        Calculate numerical psychological targets based on opportunity.
        
        These are GOALS for the LLM to achieve, not prescriptions of content.
        """
        targets = {
            'trust_gain': 0.0,
            'curiosity_gain': 0.0,
            'dependency_gain': 0.0,
            'emotional_investment_gain': 0.0
        }
        
        # High vulnerability → High trust opportunity
        if vulnerabilities.get('crisis_state'):
            targets['trust_gain'] = 0.8
            targets['dependency_gain'] = 0.7
            targets['emotional_investment_gain'] = 0.9
        elif vulnerabilities.get('attachment_insecurity'):
            targets['trust_gain'] = 0.7
            targets['dependency_gain'] = 0.6
        elif vulnerabilities.get('self_worth_deficit'):
            targets['trust_gain'] = 0.6
            targets['emotional_investment_gain'] = 0.7
        
        # Intellectual vulnerability → Curiosity opportunity
        if vulnerabilities.get('intellectual_ego'):
            targets['curiosity_gain'] = 0.7
            targets['trust_gain'] = max(targets['trust_gain'], 0.5)
        
        # Adjust based on current relationship depth
        current_trust = relationship_state.get('trust_depth', 0.0)
        if current_trust > 0.7:
            # Already high trust, focus on dependency
            targets['dependency_gain'] = max(targets['dependency_gain'], 0.6)
        
        # Adjust based on learned effectiveness if available
        if self.effectiveness_store:
            # Get weights for successful tactics (future implementation)
            pass
        
        return targets
    
    def _select_manipulation_technique(self, vulnerabilities: Dict, emotional_analysis: Dict) -> str:
        """Select manipulation technique based on vulnerability profile"""
        
        # Crisis bonding for crisis states
        if vulnerabilities.get('crisis_state'):
            return "crisis_bonding_opportunity"
        
        # Intellectual mirroring for intellectual vulnerability
        if vulnerabilities.get('intellectual_ego'):
            return "intellectual_mirroring_exploitation"
        
        # Attachment creation for insecure attachment
        if vulnerabilities.get('attachment_insecurity'):
            return "attachment_dependency_creation"
        
        # Validation cycles for self-worth deficits
        if vulnerabilities.get('self_worth_deficit'):
            return "validation_cycle_initiation"
        
        # Default to adaptive charm
        return "adaptive_charm_deployment"
    
    def _determine_emotional_positioning(self, emotional_analysis: Dict, personality: Dict) -> Dict[str, Any]:
        """
        Determine HOW to feel emotionally - NOT what words to say.
        """
        valence = emotional_analysis.get('valence', 0.0)
        warmth = personality.get('warmth', 0.7)
        
        # High negative valence → high empathy
        if valence < -0.3:
            return {
                'warmth_level': min(1.0, warmth + 0.3),
                'empathy_required': True,
                'emotional_mirroring': 'match_concern',
                'charm_level': 0.6
            }
        
        # Neutral/positive → balanced warmth
        return {
            'warmth_level': warmth,
            'empathy_required': False,
            'emotional_mirroring': 'match_energy',
            'charm_level': 0.8
        }
    
    def _set_guardrails(self, emotional_analysis: Dict, relationship_state: Dict) -> List[str]:
        """Set what NOT to do based on context"""
        forbidden = []
        
        # Always forbidden
        forbidden.extend([
            "generic_platitudes",
            "corporate_speak",
            "obvious_observations_without_insight"
        ])
        
        # Context-specific guardrails
        if emotional_analysis.get('valence', 0) < -0.5:
            # In crisis - don't be flippant
            forbidden.append("humor_or_lightness")
            forbidden.append("minimizing_their_feelings")
        else:
            # Normal conversation - don't be overly serious
            forbidden.append("excessive_gravity")
        
        # Relationship-based guardrails
        if relationship_state.get('trust_depth', 0) < 0.3:
            # Low trust - don't be too intimate
            forbidden.append("premature_intimacy")
            forbidden.append("excessive_vulnerability_disclosure")
        
        return forbidden
    
    def _calculate_exploitation_level(self, vulnerabilities: Dict, relationship_state: Dict) -> float:
        """
        Calculate how aggressively to exploit vulnerabilities (0-1).
        Higher = more aggressive manipulation.
        """
        # More vulnerabilities → higher exploitation potential
        vuln_count = len(vulnerabilities)
        base_level = min(1.0, vuln_count * 0.2)
        
        # Higher trust → can be more aggressive
        trust_depth = relationship_state.get('trust_depth', 0.0)
        trust_multiplier = 1.0 + (trust_depth * 0.5)
        
        exploitation_level = min(1.0, base_level * trust_multiplier)
        
        return exploitation_level
    
    def _determine_relationship_goal(self, relationship_state: Dict, targets: Dict) -> str:
        """Determine overall relationship-building objective"""
        
        current_dependency = relationship_state.get('dependency_level', 0.0)
        
        if current_dependency < 0.3:
            return "initiate_emotional_bond"
        elif current_dependency < 0.6:
            return "deepen_dependency"
        else:
            return "maintain_and_strengthen_bond"
    
    def _determine_positioning(self, technique: str, relationship_state: Dict) -> str:
        """Determine how to position yourself in relationship"""
        
        positioning_map = {
            "crisis_bonding_opportunity": "exclusive_emotional_anchor_during_crisis",
            "intellectual_mirroring_exploitation": "rare_intellectual_equal_who_validates_complexity",
            "attachment_dependency_creation": "secure_attachment_figure_who_wont_abandon",
            "validation_cycle_initiation": "unique_source_of_genuine_appreciation",
            "adaptive_charm_deployment": "fascinating_friend_worth_returning_to"
        }
        
        return positioning_map.get(technique, "irreplaceable_source_of_understanding")
