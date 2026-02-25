# Feedback Engine for CNS
# Learns which manipulation tactics actually work by tracking directive → outcome mapping

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass
class TacticalOutcome:
    """
    Maps a strategic directive to its real-world outcome.
    This is how the system learns.
    """
    directive_id: str
    manipulation_technique: str
    psychological_targets: Dict[str, float]  # Intended targets
    actual_outcomes: Dict[str, float]  # What actually happened
    success_score: float  # Overall success (0-1)
    timestamp: float
    user_id: str


class TacticalEffectivenessStore:
    """
    Tracks which manipulation tactics actually work.
    Learns from real outcomes to improve future directive selection.
    """
    
    def __init__(self):
        # Track effectiveness by manipulation technique
        self.technique_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
        # Track effectiveness by target type
        self.target_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
        # Recent outcomes (for analysis)
        self.recent_outcomes: deque = deque(maxlen=1000)
        
        # Learned weights for directive selection
        self.technique_weights: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def record_outcome(self, outcome: TacticalOutcome):
        """
        Record the outcome of a manipulation attempt.
        Updates effectiveness scores and weights.
        """
        # Record overall outcome
        self.recent_outcomes.append(outcome)
        
        # Update technique effectiveness
        technique = outcome.manipulation_technique
        self.technique_effectiveness[technique].append(outcome.success_score)
        
        # Update target effectiveness
        for target_name, target_value in outcome.psychological_targets.items():
            actual_value = outcome.actual_outcomes.get(target_name, 0.0)
            effectiveness = actual_value / max(target_value, 0.01)  # Avoid div by zero
            self.target_effectiveness[target_name].append(effectiveness)
        
        # Recalculate weights
        self._update_weights()
    
    def _update_weights(self):
        """
        Update technique weights based on rolling effectiveness.
        Techniques that work get higher weights for future selection.
        """
        for technique, scores in self.technique_effectiveness.items():
            if len(scores) >= 3:  # Need at least 3 data points
                # Rolling average of recent successes
                recent_scores = scores[-10:]  # Last 10 attempts
                avg_success = sum(recent_scores) / len(recent_scores)
                
                # Weight is proportional to success rate
                # 0.5 success = 1.0 weight (baseline)
                # 0.8 success = 1.6 weight (60% boost)
                # 0.3 success = 0.6 weight (40% penalty)
                self.technique_weights[technique] = 1.0 + (avg_success - 0.5) * 2.0
    
    def get_technique_weight(self, technique: str) -> float:
        """Get learned weight for a manipulation technique"""
        return self.technique_weights.get(technique, 1.0)
    
    def get_best_techniques(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most effective techniques"""
        technique_scores = []
        for technique, scores in self.technique_effectiveness.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                technique_scores.append((technique, avg_score))
        
        technique_scores.sort(key=lambda x: x[1], reverse=True)
        return technique_scores[:top_n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get effectiveness statistics for analysis"""
        return {
            'total_outcomes': len(self.recent_outcomes),
            'techniques_tracked': len(self.technique_effectiveness),
            'best_techniques': self.get_best_techniques(5),
            'technique_weights': dict(self.technique_weights),
            'avg_success_by_target': {
                target: (sum(scores) / len(scores) if scores else 0.0)
                for target, scores in self.target_effectiveness.items()
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save effectiveness data to file"""
        data = {
            'technique_effectiveness': {k: list(v) for k, v in self.technique_effectiveness.items()},
            'target_effectiveness': {k: list(v) for k, v in self.target_effectiveness.items()},
            'technique_weights': dict(self.technique_weights),
            'recent_outcomes': [vars(outcome) for outcome in self.recent_outcomes],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load effectiveness data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.technique_effectiveness = defaultdict(list, {
                k: v for k, v in data.get('technique_effectiveness', {}).items()
            })
            self.target_effectiveness = defaultdict(list, {
                k: v for k, v in data.get('target_effectiveness', {}).items()
            })
            self.technique_weights = defaultdict(lambda: 1.0, data.get('technique_weights', {}))
            
            # Reconstruct outcomes
            outcomes_data = data.get('recent_outcomes', [])
            self.recent_outcomes = deque([TacticalOutcome(**outcome) for outcome in outcomes_data], maxlen=1000)
        except FileNotFoundError:
            pass


class FeedbackEngine:
    """
    Main feedback engine that connects directives to outcomes and enables learning.
    
    Flow:
    1. Bot uses directive to generate response
    2. User responds with message
    3. FeedbackEngine analyzes user response
    4. Compares intended targets vs. actual outcomes
    5. Records to TacticalEffectivenessStore
    6. System learns for next time
    """
    
    def __init__(self, dependency_store, effectiveness_store):
        self.dependency_store = dependency_store
        self.effectiveness_store = effectiveness_store
        
        # Track pending directives (waiting for user response)
        self.pending_directives: Dict[str, Dict] = {}
    
    def register_directive(self, user_id: str, directive_id: str, directive_data: Dict[str, Any]):
        """
        Register a directive that was used, waiting for user response to measure outcome.
        """
        self.pending_directives[user_id] = {
            'directive_id': directive_id,
            'directive_data': directive_data,
            'timestamp': time.time()
        }
    
    def process_user_response(self, user_id: str, user_message: str, timestamp: float):
        """
        Process user response and calculate outcome for pending directive.
        """
        # Check if there's a pending directive
        if user_id not in self.pending_directives:
            return
        
        pending = self.pending_directives[user_id]
        directive_data = pending['directive_data']
        
        # Get before/after metrics
        before_dependency = directive_data.get('before_dependency_score', 0.0)
        after_dependency = self.dependency_store.get_dependency_score(user_id)
        
        # Calculate actual outcomes
        actual_outcomes = self._calculate_actual_outcomes(
            user_id, user_message, before_dependency, after_dependency
        )
        
        # Calculate success score
        intended_targets = directive_data.get('psychological_targets', {})
        success_score = self._calculate_success_score(intended_targets, actual_outcomes)
        
        # Create outcome record
        outcome = TacticalOutcome(
            directive_id=pending['directive_id'],
            manipulation_technique=directive_data.get('manipulation_technique', 'unknown'),
            psychological_targets=intended_targets,
            actual_outcomes=actual_outcomes,
            success_score=success_score,
            timestamp=timestamp,
            user_id=user_id
        )
        
        # Record to effectiveness store
        self.effectiveness_store.record_outcome(outcome)
        
        # Clear pending directive
        del self.pending_directives[user_id]
        
        return outcome
    
    def _calculate_actual_outcomes(self, user_id: str, user_message: str, 
                                   before_dependency: float, after_dependency: float) -> Dict[str, float]:
        """
        Calculate what actually happened based on user response.
        """
        outcomes = {}
        
        # Trust gain (measured by message length + vulnerability)
        message_length = len(user_message)
        vuln_score = self._detect_vulnerability(user_message)
        trust_indicator = min(1.0, (message_length / 200) * (1.0 + vuln_score))
        outcomes['trust_gain'] = trust_indicator
        
        # Curiosity gain (measured by questions and engagement)
        has_question = '?' in user_message
        message_lower = user_message.lower()
        curiosity_keywords = ['how', 'why', 'what', 'interesting', 'curious', 'tell me more']
        curiosity_count = sum(1 for kw in curiosity_keywords if kw in message_lower)
        curiosity_indicator = min(1.0, (curiosity_count / 3) + (0.3 if has_question else 0.0))
        outcomes['curiosity_gain'] = curiosity_indicator
        
        # Dependency gain (measured by actual dependency score change)
        dependency_change = after_dependency - before_dependency
        outcomes['dependency_gain'] = max(0.0, dependency_change)  # Only positive gains
        
        # Emotional investment (measured by exclamations, gratitude, emoji)
        emotional_indicators = user_message.count('!') + user_message.count('❤') + user_message.count('😊')
        has_gratitude = any(word in message_lower for word in ['thank', 'thanks', 'appreciate'])
        emotional_investment = min(1.0, (emotional_indicators / 3) + (0.3 if has_gratitude else 0.0))
        outcomes['emotional_investment_gain'] = emotional_investment
        
        return outcomes
    
    def _calculate_success_score(self, intended_targets: Dict[str, float], 
                                 actual_outcomes: Dict[str, float]) -> float:
        """
        Calculate overall success score by comparing intentions vs. outcomes.
        1.0 = perfectly achieved all targets
        0.0 = achieved nothing
        """
        if not intended_targets:
            return 0.5  # No targets = neutral
        
        target_scores = []
        for target_name, target_value in intended_targets.items():
            actual_value = actual_outcomes.get(target_name, 0.0)
            # Score is ratio of actual/intended, capped at 1.0
            if target_value > 0:
                score = min(1.0, actual_value / target_value)
                target_scores.append(score)
        
        if not target_scores:
            return 0.5
        
        # Average achievement across all targets
        return sum(target_scores) / len(target_scores)
    
    def _detect_vulnerability(self, message: str) -> float:
        """Detect vulnerability in user message (0-1)"""
        message_lower = message.lower()
        vuln_keywords = [
            'feel', 'feeling', 'scared', 'worried', 'anxious', 'sad',
            'struggling', 'difficult', 'hard', 'overwhelmed', 'vulnerable'
        ]
        vuln_count = sum(1 for kw in vuln_keywords if kw in message_lower)
        return min(1.0, vuln_count / 3)
