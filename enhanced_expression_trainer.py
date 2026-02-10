"""
Enhanced Expression Module Trainer - Full Dataset Integration
Trains the CNS expression module with comprehensive conversation patterns
"""

import json
import random
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from neuroplastic_optimizer import NeuroplasticInsight

class ConversationPattern:
    """Represents a learned conversation pattern"""
    
    def __init__(self, pattern_data: Dict[str, Any]):
        self.persona = pattern_data.get('persona', 'default')
        self.style = pattern_data.get('style', 'neutral')
        self.emotion = pattern_data.get('emotion', 'neutral')
        self.response_text = pattern_data.get('text', '')
        self.context = pattern_data.get('context', {})
        self.engagement_hooks = pattern_data.get('engagement_hooks', [])
        self.usage_count = 0
        self.success_rate = 0.0
        
    def matches_context(self, current_emotion: str, desired_style: str, persona_hint: str = None) -> float:
        """Calculate how well this pattern matches current context"""
        match_score = 0.0
        
        # Emotion matching
        if self.emotion == current_emotion:
            match_score += 0.4
        elif self._emotions_compatible(self.emotion, current_emotion):
            match_score += 0.2
            
        # Style matching
        if self.style == desired_style:
            match_score += 0.3
        elif self._styles_compatible(self.style, desired_style):
            match_score += 0.15
            
        # Persona matching
        if persona_hint and self.persona == persona_hint:
            match_score += 0.3
        
        return min(1.0, match_score)
        
    def _emotions_compatible(self, emotion1: str, emotion2: str) -> bool:
        """Check if emotions are compatible"""
        positive_emotions = {'happy', 'excited', 'neutral'}
        negative_emotions = {'sad', 'stressed', 'angry'}
        
        return (emotion1 in positive_emotions and emotion2 in positive_emotions) or \
               (emotion1 in negative_emotions and emotion2 in negative_emotions)
               
    def _styles_compatible(self, style1: str, style2: str) -> bool:
        """Check if styles are compatible"""
        casual_styles = {'genz_slang', 'millennial_casual'}
        formal_styles = {'professional_tone', 'trainer_voice'}
        intimate_styles = {'romantic_chat'}
        
        return (style1 in casual_styles and style2 in casual_styles) or \
               (style1 in formal_styles and style2 in formal_styles) or \
               (style1 in intimate_styles and style2 in intimate_styles)

class EnhancedExpressionTrainer:
    """Enhanced training system for expression module with full dataset integration"""
    
    def __init__(self):
        self.conversation_patterns = []
        self.persona_patterns = defaultdict(list)
        self.style_patterns = defaultdict(list)
        self.emotion_patterns = defaultdict(list)
        
        # Pattern effectiveness tracking
        self.pattern_effectiveness = {}
        self.contextual_success_rates = defaultdict(float)
        
        # Neuroplastic integration
        self.neuroplastic_enhancements = {
            'pattern_diversity': 0.0,
            'contextual_accuracy': 0.0,
            'emotional_resonance': 0.0,
            'style_consistency': 0.0
        }
        
    def load_conversation_dataset(self, dataset_path: str):
        """Load and process the comprehensive conversation dataset"""
        print(f"🚀 Loading conversation dataset from {dataset_path}")
        
        patterns_loaded = 0
        personas_found = set()
        styles_found = set()
        emotions_found = set()
        
        try:
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            conv_data = json.loads(line)
                            patterns_loaded += self._process_conversation(conv_data)
                            
                            # Track diversity
                            personas_found.add(conv_data.get('persona', 'default'))
                            styles_found.add(conv_data.get('style', 'neutral'))
                            
                            for turn in conv_data.get('turns', []):
                                if turn.get('speaker') == 'ai':
                                    emotions_found.add(turn.get('emotion', 'neutral'))
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {dataset_path}")
            return False
            
        # Calculate neuroplastic enhancements
        self.neuroplastic_enhancements['pattern_diversity'] = len(self.conversation_patterns) / 1000.0
        self.neuroplastic_enhancements['contextual_accuracy'] = len(personas_found) * len(styles_found) / 50.0
        self.neuroplastic_enhancements['emotional_resonance'] = len(emotions_found) / 10.0
        
        print(f"✅ Loaded {patterns_loaded} conversation patterns")
        print(f"🎭 Personas: {len(personas_found)} | Styles: {len(styles_found)} | Emotions: {len(emotions_found)}")
        print(f"🚀 Neuroplastic enhancement calculated: {sum(self.neuroplastic_enhancements.values()):.2f}")
        
        return True
        
    def _process_conversation(self, conv_data: Dict[str, Any]) -> int:
        """Process a single conversation and extract patterns"""
        patterns_extracted = 0
        
        persona = conv_data.get('persona', 'default')
        style = conv_data.get('style', 'neutral')
        tags = conv_data.get('tags', [])
        
        # Process AI turns to learn response patterns
        for i, turn in enumerate(conv_data.get('turns', [])):
            if turn.get('speaker') == 'ai':
                # Get context from previous user turn
                context = {}
                if i > 0:
                    prev_turn = conv_data['turns'][i-1]
                    context = {
                        'user_emotion': prev_turn.get('emotion', 'neutral'),
                        'user_text': prev_turn.get('text', ''),
                        'conversation_flow': i // 2  # Turn pair number
                    }
                
                pattern = ConversationPattern({
                    'persona': persona,
                    'style': style,
                    'emotion': turn.get('emotion', 'neutral'),
                    'text': turn.get('text', ''),
                    'context': context,
                    'engagement_hooks': tags
                })
                
                self.conversation_patterns.append(pattern)
                self.persona_patterns[persona].append(pattern)
                self.style_patterns[style].append(pattern)
                self.emotion_patterns[turn.get('emotion', 'neutral')].append(pattern)
                
                patterns_extracted += 1
                
        return patterns_extracted
        
    def get_enhanced_response_pattern(self, context: Dict[str, Any]) -> Optional[ConversationPattern]:
        """Get enhanced response pattern based on neuroplastic optimization"""
        current_emotion = context.get('emotion', 'neutral')
        desired_style = context.get('style', 'millennial_casual')
        persona_hint = context.get('persona', None)
        cognitive_load = context.get('cognitive_load', 0.5)
        
        # Filter relevant patterns
        candidate_patterns = []
        
        # Start with emotion-based filtering
        emotion_patterns = self.emotion_patterns.get(current_emotion, [])
        if not emotion_patterns:
            # Fallback to compatible emotions
            for emotion, patterns in self.emotion_patterns.items():
                if self._emotions_compatible(current_emotion, emotion):
                    emotion_patterns.extend(patterns)
        
        # Score and rank patterns
        for pattern in emotion_patterns[:50]:  # Limit for efficiency
            match_score = pattern.matches_context(current_emotion, desired_style, persona_hint)
            
            # Apply neuroplastic enhancement multipliers
            if pattern.style in ['genz_slang', 'millennial_casual']:
                match_score *= (1.0 + self.neuroplastic_enhancements['pattern_diversity'])
            
            # Cognitive load adaptation
            if cognitive_load > 0.7:  # High load - prefer simpler patterns
                if len(pattern.response_text.split()) <= 5:
                    match_score *= 1.2
            else:  # Low load - can use more complex patterns
                if len(pattern.response_text.split()) > 5:
                    match_score *= 1.1
                    
            if match_score > 0.3:  # Minimum threshold
                candidate_patterns.append((pattern, match_score))
        
        if not candidate_patterns:
            return None
            
        # Select best pattern with some randomness for variety
        candidate_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 patterns with weighted selection
        top_patterns = candidate_patterns[:3]
        weights = [score for _, score in top_patterns]
        
        if weights:
            selected_pattern = random.choices([p for p, _ in top_patterns], weights=weights)[0]
            selected_pattern.usage_count += 1
            return selected_pattern
            
        return None
        
    def _emotions_compatible(self, emotion1: str, emotion2: str) -> bool:
        """Check if emotions are compatible for pattern matching"""
        positive_emotions = {'happy', 'excited', 'neutral'}
        negative_emotions = {'sad', 'stressed', 'angry'}
        
        return (emotion1 in positive_emotions and emotion2 in positive_emotions) or \
               (emotion1 in negative_emotions and emotion2 in negative_emotions)
               
    def generate_neuroplastic_insight(self) -> NeuroplasticInsight:
        """Generate neuroplastic insight from expression training"""
        total_enhancement = sum(self.neuroplastic_enhancements.values())
        
        return NeuroplasticInsight(
            source='enhanced_expression_trainer',
            content={
                'total_patterns': len(self.conversation_patterns),
                'personas_available': len(self.persona_patterns),
                'styles_available': len(self.style_patterns),
                'emotions_covered': len(self.emotion_patterns),
                'enhancement_multiplier': min(1.5, 1.0 + total_enhancement * 0.1)
            },
            confidence=min(1.0, total_enhancement / 3.0),
            relevance_score=0.8,
            timestamp=time.time(),
            cognitive_enhancement={
                'expression': total_enhancement * 0.2,
                'personality_adaptation': total_enhancement * 0.15,
                'contextual_response': total_enhancement * 0.25
            }
        )
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
            'total_patterns': len(self.conversation_patterns),
            'personas': list(self.persona_patterns.keys()),
            'styles': list(self.style_patterns.keys()),
            'emotions': list(self.emotion_patterns.keys()),
            'neuroplastic_enhancements': self.neuroplastic_enhancements.copy(),
            'pattern_distribution': {
                'by_persona': {k: len(v) for k, v in self.persona_patterns.items()},
                'by_style': {k: len(v) for k, v in self.style_patterns.items()},
                'by_emotion': {k: len(v) for k, v in self.emotion_patterns.items()}
            }
        }
        
    def update_pattern_effectiveness(self, pattern: ConversationPattern, effectiveness_score: float):
        """Update pattern effectiveness based on usage outcomes"""
        pattern_id = f"{pattern.persona}_{pattern.style}_{pattern.emotion}"
        
        if pattern_id not in self.pattern_effectiveness:
            self.pattern_effectiveness[pattern_id] = []
            
        self.pattern_effectiveness[pattern_id].append(effectiveness_score)
        
        # Update pattern success rate
        pattern.success_rate = sum(self.pattern_effectiveness[pattern_id]) / len(self.pattern_effectiveness[pattern_id])
        
        # Update contextual success rates
        context_key = f"{pattern.emotion}_{pattern.style}"
        current_rate = self.contextual_success_rates[context_key]
        self.contextual_success_rates[context_key] = (current_rate + effectiveness_score) / 2.0
        
        # CRITICAL: Generate personality feedback based on pattern success
        return self._generate_personality_feedback(pattern, effectiveness_score)
        
    def _generate_personality_feedback(self, pattern: ConversationPattern, effectiveness_score: float) -> Dict[str, float]:
        """Generate personality trait adjustments based on successful patterns"""
        trait_adjustments = {}
        
        # Strong feedback for very successful patterns
        if effectiveness_score > 0.8:
            adjustment_strength = 0.15
        elif effectiveness_score > 0.6:
            adjustment_strength = 0.1
        elif effectiveness_score > 0.4:
            adjustment_strength = 0.05
        else:
            adjustment_strength = -0.05  # Negative feedback for poor patterns
            
        # Map personas to personality traits
        persona_trait_mapping = {
            'playful_friend': {'humor': adjustment_strength, 'extraversion': adjustment_strength * 0.8},
            'motivational_trainer': {'assertiveness': adjustment_strength, 'enthusiasm_level': adjustment_strength * 0.7},
            'deep_listener': {'empathy': adjustment_strength, 'introspection_depth': adjustment_strength * 0.6},
            'casual_professional': {'conscientiousness': adjustment_strength * 0.5, 'formality': adjustment_strength * 0.3},
            'supportive_partner': {'empathy': adjustment_strength, 'warmth': adjustment_strength * 0.8}
        }
        
        # Map styles to personality traits
        style_trait_mapping = {
            'genz_slang': {'playfulness': adjustment_strength, 'creativity': adjustment_strength * 0.6},
            'millennial_casual': {'openness': adjustment_strength * 0.5, 'playfulness': adjustment_strength * 0.4},
            'romantic_chat': {'empathy': adjustment_strength, 'warmth': adjustment_strength * 0.9},
            'trainer_voice': {'assertiveness': adjustment_strength, 'logical_thinking': adjustment_strength * 0.5},
            'professional_tone': {'conscientiousness': adjustment_strength, 'formality': adjustment_strength * 0.7}
        }
        
        # Apply persona-based adjustments
        if pattern.persona in persona_trait_mapping:
            for trait, adjustment in persona_trait_mapping[pattern.persona].items():
                trait_adjustments[trait] = trait_adjustments.get(trait, 0) + adjustment
                
        # Apply style-based adjustments
        if pattern.style in style_trait_mapping:
            for trait, adjustment in style_trait_mapping[pattern.style].items():
                trait_adjustments[trait] = trait_adjustments.get(trait, 0) + adjustment
                
        # Emotional state reinforcement
        emotion_trait_mapping = {
            'happy': {'enthusiasm_level': adjustment_strength * 0.3, 'humor': adjustment_strength * 0.2},
            'excited': {'extraversion': adjustment_strength * 0.4, 'enthusiasm_level': adjustment_strength * 0.5},
            'sad': {'empathy': adjustment_strength * 0.4, 'introspection_depth': adjustment_strength * 0.3},
            'stressed': {'logical_thinking': adjustment_strength * 0.2, 'assertiveness': -adjustment_strength * 0.1},
            'neutral': {'conscientiousness': adjustment_strength * 0.1}
        }
        
        if pattern.emotion in emotion_trait_mapping:
            for trait, adjustment in emotion_trait_mapping[pattern.emotion].items():
                trait_adjustments[trait] = trait_adjustments.get(trait, 0) + adjustment
                
        return trait_adjustments
        
    def optimize_for_neuroplasticity(self) -> Dict[str, float]:
        """Optimize the training system for maximum neuroplastic integration"""
        optimizations = {}
        
        # Pattern diversity optimization
        persona_distribution = [len(patterns) for patterns in self.persona_patterns.values()]
        style_distribution = [len(patterns) for patterns in self.style_patterns.values()]
        
        if persona_distribution:
            persona_balance = min(persona_distribution) / max(persona_distribution)
            optimizations['persona_balance'] = persona_balance
            
        if style_distribution:
            style_balance = min(style_distribution) / max(style_distribution)
            optimizations['style_balance'] = style_balance
            
        # Effectiveness optimization
        if self.pattern_effectiveness:
            avg_effectiveness = sum(sum(scores) for scores in self.pattern_effectiveness.values()) / \
                              sum(len(scores) for scores in self.pattern_effectiveness.values())
            optimizations['avg_effectiveness'] = avg_effectiveness
            
        # Calculate overall optimization score
        optimizations['overall_score'] = sum(optimizations.values()) / len(optimizations) if optimizations else 0.0
        
        return optimizations