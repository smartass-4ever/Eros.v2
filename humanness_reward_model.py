# Humanness Reward Model - Trainable system for evaluating response quality
# Replaces rule-based scoring with learned preferences from user feedback

import json
import numpy as np
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import math

@dataclass
class HumannessFeatures:
    """Features extracted from responses for humanness evaluation"""
    # Content features
    emotional_alignment: float      # How well response matches user emotion
    conversational_flow: float      # Natural continuation of conversation
    personality_consistency: float  # Consistency with established persona
    
    # Language features
    linguistic_naturalness: float   # Natural language patterns
    gen_z_appropriateness: float   # Appropriate use of modern language
    formality_level: float         # Appropriate formality for context
    
    # Engagement features
    curiosity_stimulation: float   # How likely to continue conversation
    empathy_demonstration: float   # Shows understanding and care
    humor_effectiveness: float     # Appropriate and effective humor
    
    # Authenticity features
    vulnerability_appropriate: float # Shows appropriate vulnerability
    opinion_authenticity: float     # Genuine personal perspectives
    imperfection_naturalness: float # Natural human imperfections

@dataclass
class ResponseComparison:
    """Pairwise comparison data for training"""
    response_a: str
    response_b: str
    features_a: HumannessFeatures
    features_b: HumannessFeatures
    user_preference: str  # 'a', 'b', or 'tie'
    context: Dict[str, Any]
    timestamp: float

class HumannessRewardModel:
    """Trainable reward model for evaluating response humanness"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_weights = self._initialize_weights()
        self.training_data = []
        self.model_version = 1.0
        self.learning_rate = 0.01
        self.regularization = 0.001
        
        # User feedback history for preference learning
        self.feedback_history = deque(maxlen=1000)
        self.preference_patterns = defaultdict(list)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize feature weights with reasonable defaults"""
        return {
            # Content weights
            'emotional_alignment': 0.25,
            'conversational_flow': 0.20,
            'personality_consistency': 0.15,
            
            # Language weights
            'linguistic_naturalness': 0.20,
            'gen_z_appropriateness': 0.10,
            'formality_level': 0.05,
            
            # Engagement weights
            'curiosity_stimulation': 0.15,
            'empathy_demonstration': 0.20,
            'humor_effectiveness': 0.10,
            
            # Authenticity weights
            'vulnerability_appropriate': 0.15,
            'opinion_authenticity': 0.20,
            'imperfection_naturalness': 0.10
        }
    
    def extract_features(self, response: str, user_input: str, context: Dict[str, Any]) -> HumannessFeatures:
        """Extract humanness features from a response"""
        
        # Basic text analysis
        response_lower = response.lower()
        user_lower = user_input.lower()
        
        # Emotional alignment
        user_emotion = context.get('emotion', 'neutral')
        emotional_alignment = self._calculate_emotional_alignment(response, user_emotion, context)
        
        # Conversational flow
        conversational_flow = self._calculate_conversational_flow(response, user_input, context)
        
        # Personality consistency
        persona = context.get('active_persona', 'supportive_partner')
        personality_consistency = self._calculate_personality_consistency(response, persona)
        
        # Linguistic naturalness
        linguistic_naturalness = self._calculate_linguistic_naturalness(response)
        
        # Gen-Z appropriateness
        gen_z_appropriateness = self._calculate_gen_z_appropriateness(response, context)
        
        # Formality level
        formality_level = self._calculate_formality_appropriateness(response, context)
        
        # Curiosity stimulation
        curiosity_stimulation = self._calculate_curiosity_stimulation(response)
        
        # Empathy demonstration
        empathy_demonstration = self._calculate_empathy_demonstration(response, user_input)
        
        # Humor effectiveness
        humor_effectiveness = self._calculate_humor_effectiveness(response, context)
        
        # Vulnerability appropriateness
        vulnerability_appropriate = self._calculate_vulnerability_appropriateness(response, context)
        
        # Opinion authenticity
        opinion_authenticity = self._calculate_opinion_authenticity(response)
        
        # Imperfection naturalness
        imperfection_naturalness = self._calculate_imperfection_naturalness(response)
        
        return HumannessFeatures(
            emotional_alignment=emotional_alignment,
            conversational_flow=conversational_flow,
            personality_consistency=personality_consistency,
            linguistic_naturalness=linguistic_naturalness,
            gen_z_appropriateness=gen_z_appropriateness,
            formality_level=formality_level,
            curiosity_stimulation=curiosity_stimulation,
            empathy_demonstration=empathy_demonstration,
            humor_effectiveness=humor_effectiveness,
            vulnerability_appropriate=vulnerability_appropriate,
            opinion_authenticity=opinion_authenticity,
            imperfection_naturalness=imperfection_naturalness
        )
    
    def _calculate_emotional_alignment(self, response: str, user_emotion: str, context: Dict) -> float:
        """Calculate how well response aligns with user's emotional state"""
        response_lower = response.lower()
        
        emotional_indicators = {
            'sad': ['sorry', 'understand', 'feel', 'here for you', 'tough'],
            'anxious': ['okay', 'breathe', 'safe', 'together', 'step by step'],
            'angry': ['hear you', 'frustrated', 'valid', 'makes sense'],
            'happy': ['amazing', 'awesome', 'love', 'excited', 'wonderful'],
            'confused': ['help', 'explain', 'break it down', 'clarify']
        }
        
        if user_emotion in emotional_indicators:
            indicators = emotional_indicators[user_emotion]
            matches = sum(1 for indicator in indicators if indicator in response_lower)
            return min(1.0, matches / len(indicators))
        
        return 0.5  # Neutral if no specific emotion
    
    def _calculate_conversational_flow(self, response: str, user_input: str, context: Dict) -> float:
        """Calculate natural conversational flow"""
        score = 0.0
        
        # Check for conversation connectors
        connectors = ['so', 'and', 'but', 'well', 'also', 'plus', 'though']
        score += 0.2 * min(1.0, sum(1 for c in connectors if c in response.lower()) / 3)
        
        # Check for topic continuation
        user_topics = context.get('recent_topics', [])
        if user_topics:
            topic_continuation = sum(1 for topic in user_topics if topic.lower() in response.lower())
            score += 0.3 * min(1.0, topic_continuation / len(user_topics))
        
        # Check for natural transitions
        transitions = ['speaking of', 'that reminds me', 'on that note', 'similarly']
        score += 0.2 * min(1.0, sum(1 for t in transitions if t in response.lower()) / 2)
        
        # Check for questions that continue conversation
        if '?' in response:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_personality_consistency(self, response: str, persona: str) -> float:
        """Calculate consistency with established persona"""
        persona_markers = {
            'supportive_partner': ['you', 'we', 'together', 'support', 'here for'],
            'witty_companion': ['honestly', 'pretty', 'kinda', 'like', 'though'],
            'analytical_guide': ['analyze', 'consider', 'think', 'perspective', 'approach'],
            'casual_friend': ['dude', 'honestly', 'like', 'totally', 'yeah']
        }
        
        if persona in persona_markers:
            markers = persona_markers[persona]
            matches = sum(1 for marker in markers if marker in response.lower())
            return min(1.0, matches / len(markers))
        
        return 0.5
    
    def _calculate_linguistic_naturalness(self, response: str) -> float:
        """Calculate naturalness of language patterns"""
        score = 0.0
        
        # Check for natural contractions
        contractions = ["I'm", "you're", "it's", "don't", "can't", "won't"]
        contraction_score = sum(1 for c in contractions if c in response) / len(contractions)
        score += 0.3 * min(1.0, contraction_score)
        
        # Check for natural hesitations/fillers
        fillers = ['um', 'uh', 'like', 'you know', 'I mean', 'well']
        filler_score = sum(1 for f in fillers if f in response.lower()) / len(fillers)
        score += 0.2 * min(1.0, filler_score)
        
        # Check for varied sentence lengths (natural rhythm)
        sentences = response.split('.')
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                variation = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0.0
                score += 0.3 * min(1.0, variation)
        
        # Check for natural punctuation patterns
        natural_punct = response.count(',') + response.count('...') + response.count('!')
        score += 0.2 * min(1.0, natural_punct / max(1, len(response.split())))
        
        return min(1.0, score)
    
    def _calculate_gen_z_appropriateness(self, response: str, context: Dict) -> float:
        """Calculate appropriate use of Gen-Z language patterns"""
        response_lower = response.lower()
        
        gen_z_terms = {
            'casual': ['honestly', 'literally', 'ngl', 'fr', 'periodt', 'bet'],
            'intensifiers': ['so', 'really', 'super', 'totally', 'absolutely'],
            'connectors': ['like', 'and', 'but also', 'or maybe', 'kinda']
        }
        
        total_terms = sum(len(terms) for terms in gen_z_terms.values())
        found_terms = 0
        
        for category, terms in gen_z_terms.items():
            found_terms += sum(1 for term in terms if term in response_lower)
        
        appropriateness = found_terms / total_terms
        
        # Adjust based on context - don't force Gen-Z in serious situations
        user_emotion = context.get('emotion', 'neutral')
        if user_emotion in ['sad', 'anxious', 'angry']:
            appropriateness *= 0.5  # Reduce Gen-Z usage in serious contexts
        
        return min(1.0, appropriateness)
    
    def _calculate_formality_appropriateness(self, response: str, context: Dict) -> float:
        """Calculate appropriate formality level for context"""
        formality_score = 0.0
        
        # Formal indicators
        formal_words = ['however', 'therefore', 'furthermore', 'consequently']
        formal_count = sum(1 for word in formal_words if word in response.lower())
        
        # Informal indicators
        informal_words = ['yeah', 'ok', 'cool', 'awesome', 'dude']
        informal_count = sum(1 for word in informal_words if word in response.lower())
        
        # Context-based appropriateness
        relationship_level = context.get('relationship_level', 'casual')
        if relationship_level == 'intimate':
            # More informal is appropriate
            formality_score = max(0, 1.0 - (formal_count * 0.2)) + (informal_count * 0.1)
        else:
            # Balanced formality
            formality_score = 0.7 - abs(formal_count - informal_count) * 0.1
        
        return min(1.0, max(0.0, formality_score))
    
    def _calculate_curiosity_stimulation(self, response: str) -> float:
        """Calculate how likely response is to stimulate continued conversation"""
        score = 0.0
        
        # Questions stimulate curiosity
        question_count = response.count('?')
        score += min(0.4, question_count * 0.2)
        
        # Open-ended prompts
        open_prompts = ['what do you think', 'how do you feel', 'tell me more', 'what about']
        prompt_score = sum(1 for prompt in open_prompts if prompt in response.lower())
        score += min(0.3, prompt_score * 0.1)
        
        # Interesting hooks
        hooks = ['interesting', 'curious', 'wonder', 'fascinating', 'intriguing']
        hook_score = sum(1 for hook in hooks if hook in response.lower())
        score += min(0.3, hook_score * 0.1)
        
        return min(1.0, score)
    
    def _calculate_empathy_demonstration(self, response: str, user_input: str) -> float:
        """Calculate empathy shown in response"""
        response_lower = response.lower()
        
        empathy_indicators = [
            'understand', 'feel', 'hear you', 'that sounds', 'i can see',
            'makes sense', 'must be', 'imagine', 'i get'
        ]
        
        empathy_score = sum(1 for indicator in empathy_indicators if indicator in response_lower)
        return min(1.0, empathy_score / len(empathy_indicators))
    
    def _calculate_humor_effectiveness(self, response: str, context: Dict) -> float:
        """Calculate effectiveness of humor in response"""
        user_emotion = context.get('emotion', 'neutral')
        
        # Don't use humor in serious emotional contexts
        if user_emotion in ['sad', 'anxious', 'angry']:
            return 1.0 if not any(indicator in response.lower() for indicator in ['haha', 'lol', 'funny']) else 0.0
        
        # Light humor indicators
        humor_indicators = ['haha', 'funny', 'amusing', 'silly', 'quirky']
        humor_count = sum(1 for indicator in humor_indicators if indicator in response.lower())
        
        return min(1.0, humor_count * 0.3)
    
    def _calculate_vulnerability_appropriateness(self, response: str, context: Dict) -> float:
        """Calculate appropriate vulnerability level"""
        vulnerability_indicators = [
            'i feel', 'i struggle', 'i sometimes', 'i wonder', 'honestly',
            'to be honest', 'i admit', 'i confess'
        ]
        
        vulnerability_count = sum(1 for indicator in vulnerability_indicators if indicator in response.lower())
        
        # Adjust based on relationship level
        relationship_level = context.get('relationship_level', 'casual')
        multiplier = 1.2 if relationship_level == 'intimate' else 0.8
        
        return min(1.0, vulnerability_count * 0.2 * multiplier)
    
    def _calculate_opinion_authenticity(self, response: str) -> float:
        """Calculate authenticity of expressed opinions"""
        opinion_indicators = [
            'i think', 'i believe', 'in my opinion', 'i feel like',
            'from my perspective', 'i see it as', 'my take is'
        ]
        
        opinion_count = sum(1 for indicator in opinion_indicators if indicator in response.lower())
        return min(1.0, opinion_count * 0.25)
    
    def _calculate_imperfection_naturalness(self, response: str) -> float:
        """Calculate natural human imperfections"""
        imperfections = [
            'um', 'uh', 'like', 'you know', 'i mean', 'kinda', 'sorta',
            'maybe', 'i guess', 'i think', '...'
        ]
        
        imperfection_count = sum(1 for imp in imperfections if imp in response.lower())
        return min(1.0, imperfection_count * 0.15)
    
    def calculate_humanness_score(self, features: HumannessFeatures) -> float:
        """Calculate overall humanness score using learned weights"""
        score = 0.0
        feature_dict = asdict(features)
        
        for feature_name, feature_value in feature_dict.items():
            weight = self.feature_weights.get(feature_name, 0.1)
            score += weight * feature_value
        
        return min(1.0, max(0.0, score))
    
    def add_preference_feedback(self, response_a: str, response_b: str, 
                              features_a: HumannessFeatures, features_b: HumannessFeatures,
                              user_preference: str, context: Dict[str, Any]):
        """Add user preference feedback for training"""
        comparison = ResponseComparison(
            response_a=response_a,
            response_b=response_b,
            features_a=features_a,
            features_b=features_b,
            user_preference=user_preference,
            context=context,
            timestamp=time.time()
        )
        
        self.training_data.append(comparison)
        self.feedback_history.append(comparison)
        
        # Update preference patterns
        if user_preference != 'tie':
            preferred_features = features_a if user_preference == 'a' else features_b
            context_key = f"{context.get('persona', 'default')}_{context.get('emotion', 'neutral')}"
            self.preference_patterns[context_key].append(preferred_features)
    
    def train_from_feedback(self, learning_rate: Optional[float] = None):
        """Train model weights from collected feedback"""
        if not self.training_data:
            return
        
        lr = learning_rate or self.learning_rate
        
        for comparison in self.training_data:
            # Calculate current scores
            score_a = self.calculate_humanness_score(comparison.features_a)
            score_b = self.calculate_humanness_score(comparison.features_b)
            
            # Determine target adjustment based on preference
            if comparison.user_preference == 'a':
                target_diff = 0.1  # A should be 0.1 higher than B
            elif comparison.user_preference == 'b':
                target_diff = -0.1  # B should be 0.1 higher than A
            else:
                target_diff = 0.0  # Tie
            
            current_diff = score_a - score_b
            error = target_diff - current_diff
            
            # Update weights using gradient-like approach
            features_a_dict = asdict(comparison.features_a)
            features_b_dict = asdict(comparison.features_b)
            
            for feature_name in features_a_dict:
                feature_diff = features_a_dict[feature_name] - features_b_dict[feature_name]
                
                if abs(feature_diff) > 0.01:  # Only update if there's a meaningful difference
                    gradient = error * feature_diff
                    
                    # Update weight with regularization
                    current_weight = self.feature_weights.get(feature_name, 0.1)
                    new_weight = current_weight + lr * gradient - self.regularization * current_weight
                    
                    self.feature_weights[feature_name] = max(0.01, min(1.0, new_weight))
        
        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        for feature_name in self.feature_weights:
            self.feature_weights[feature_name] /= total_weight
        
        print(f"[REWARD MODEL] Updated weights from {len(self.training_data)} comparisons")
        self.model_version += 0.1
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'feature_weights': self.feature_weights,
            'model_version': self.model_version,
            'training_data_count': len(self.training_data),
            'preference_patterns': dict(self.preference_patterns),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"[REWARD MODEL] Saved model v{self.model_version} to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.feature_weights = model_data.get('feature_weights', self._initialize_weights())
            self.model_version = model_data.get('model_version', 1.0)
            self.preference_patterns = defaultdict(list, model_data.get('preference_patterns', {}))
            
            print(f"[REWARD MODEL] Loaded model v{self.model_version} from {filepath}")
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about the current model state"""
        top_features = sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'model_version': self.model_version,
            'total_training_examples': len(self.training_data),
            'top_weighted_features': top_features[:5],
            'preference_contexts': list(self.preference_patterns.keys()),
            'training_data_recency': max([c.timestamp for c in self.training_data]) if self.training_data else 0
        }