#!/usr/bin/env python3

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class EmotionalContext:
    """Rich emotional context understanding"""
    primary_emotion: str
    emotion_intensity: float
    emotional_subtext: str
    empathy_requirements: List[str]
    emotional_memory_triggers: List[str]
    appropriate_response_tone: str

@dataclass
class ConversationContext:
    """Deep conversation context tracking"""
    topic_thread: str
    conversation_depth: int
    relationship_context: str
    shared_experiences: List[str]
    unspoken_implications: List[str]
    context_continuity: float

class EnhancedContextualEmotionalSystem:
    """Advanced context understanding and emotional intelligence"""
    
    def __init__(self):
        self.conversation_history = deque(maxlen=50)
        self.emotional_patterns = {}
        self.context_layers = {}
        self.relationship_memory = {}
        
    def analyze_deep_context(self, human_input: str, user_id: str, conversation_history: List[str] = None) -> ConversationContext:
        """Understand deep contextual layers in conversation"""
        
        # Extract topic threading
        topic_thread = self._extract_topic_continuity(human_input, conversation_history)
        
        # Assess conversation depth
        depth = self._assess_conversation_depth(human_input, conversation_history)
        
        # Understand relationship context
        relationship_context = self._understand_relationship_context(user_id, human_input)
        
        # Identify shared experiences
        shared_experiences = self._identify_shared_experiences(user_id, human_input)
        
        # Detect unspoken implications
        implications = self._detect_unspoken_implications(human_input, conversation_history)
        
        # Calculate context continuity
        continuity = self._calculate_context_continuity(human_input, conversation_history)
        
        return ConversationContext(
            topic_thread=topic_thread,
            conversation_depth=depth,
            relationship_context=relationship_context,
            shared_experiences=shared_experiences,
            unspoken_implications=implications,
            context_continuity=continuity
        )
    
    def analyze_emotional_intelligence(self, human_input: str, user_id: str) -> EmotionalContext:
        """Deep emotional intelligence analysis"""
        
        # Identify primary emotion with nuance
        primary_emotion = self._identify_primary_emotion(human_input)
        
        # Measure emotional intensity
        intensity = self._measure_emotional_intensity(human_input)
        
        # Understand emotional subtext
        subtext = self._understand_emotional_subtext(human_input, user_id)
        
        # Determine empathy requirements
        empathy_needs = self._determine_empathy_requirements(human_input, primary_emotion)
        
        # Identify emotional memory triggers
        memory_triggers = self._identify_emotional_memory_triggers(human_input, user_id)
        
        # Set appropriate response tone
        response_tone = self._determine_response_tone(primary_emotion, intensity, subtext)
        
        return EmotionalContext(
            primary_emotion=primary_emotion,
            emotion_intensity=intensity,
            emotional_subtext=subtext,
            empathy_requirements=empathy_needs,
            emotional_memory_triggers=memory_triggers,
            appropriate_response_tone=response_tone
        )
    
    def _extract_topic_continuity(self, input_text: str, history: List[str]) -> str:
        """Extract the continuous topic thread"""
        
        if not history:
            return "new_conversation"
            
        # Analyze topic keywords and themes
        current_keywords = self._extract_keywords(input_text)
        
        # Look for topic continuity in recent history
        recent_context = ' '.join(history[-5:]).lower() if history else ""
        
        # Topic categories with contextual depth
        topic_categories = {
            'personal_growth': ['learn', 'grow', 'improve', 'develop', 'change', 'goal', 'aspiration', 'dream'],
            'relationships': ['friend', 'family', 'partner', 'relationship', 'connection', 'love', 'trust', 'conflict'],
            'work_career': ['job', 'work', 'career', 'boss', 'colleague', 'project', 'stress', 'promotion', 'interview'],
            'emotional_wellbeing': ['feel', 'emotion', 'stress', 'anxiety', 'happy', 'sad', 'worry', 'peace', 'mindfulness'],
            'life_philosophy': ['meaning', 'purpose', 'existence', 'belief', 'value', 'morality', 'spirituality', 'truth'],
            'creativity_passion': ['create', 'art', 'music', 'write', 'passion', 'inspiration', 'imagination', 'express'],
            'technology_future': ['ai', 'technology', 'future', 'innovation', 'digital', 'internet', 'automation'],
            'health_lifestyle': ['health', 'fitness', 'exercise', 'diet', 'sleep', 'wellness', 'lifestyle', 'habits']
        }
        
        # Find matching topic thread
        for topic, keywords in topic_categories.items():
            if any(keyword in recent_context for keyword in keywords):
                if any(keyword in input_text.lower() for keyword in keywords):
                    return f"{topic}_continuation"
        
        return "topic_shift"
    
    def _assess_conversation_depth(self, input_text: str, history: List[str]) -> int:
        """Assess the depth level of conversation"""
        
        depth_indicators = {
            1: ['what', 'how', 'when', 'where'],  # Surface questions
            2: ['why', 'because', 'think', 'feel', 'believe'],  # Personal thoughts
            3: ['meaningful', 'purpose', 'value', 'important', 'matter'],  # Values level
            4: ['identity', 'who am i', 'existence', 'consciousness', 'soul'],  # Identity level
            5: ['transcendent', 'spiritual', 'divine', 'ultimate', 'infinite']  # Transcendent level
        }
        
        input_lower = input_text.lower()
        max_depth = 1
        
        for depth, indicators in depth_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                max_depth = max(max_depth, depth)
                
        return max_depth
    
    def _understand_relationship_context(self, user_id: str, input_text: str) -> str:
        """Understand the relationship context and dynamic"""
        
        relationship_patterns = {
            'seeking_support': ['help', 'support', 'advice', 'guidance', 'struggling', 'difficult'],
            'sharing_joy': ['excited', 'happy', 'amazing', 'wonderful', 'celebration', 'achievement'],
            'intellectual_exploration': ['think', 'consider', 'analyze', 'understand', 'explore', 'discover'],
            'emotional_processing': ['feel', 'emotion', 'processing', 'working through', 'dealing with'],
            'casual_connection': ['how are you', 'what\'s up', 'checking in', 'just wanted to', 'by the way'],
            'deep_intimacy': ['vulnerable', 'trust', 'personal', 'private', 'intimate', 'close', 'secret'],
            'creative_collaboration': ['create', 'imagine', 'brainstorm', 'idea', 'creative', 'innovative']
        }
        
        input_lower = input_text.lower()
        
        for context, patterns in relationship_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                return context
                
        return "neutral_interaction"
    
    def _identify_shared_experiences(self, user_id: str, input_text: str) -> List[str]:
        """Identify references to shared experiences"""
        
        # Look for shared experience indicators
        shared_indicators = [
            'remember when', 'like we discussed', 'as you mentioned', 'from our conversation',
            'you helped me with', 'we talked about', 'you said', 'when you told me'
        ]
        
        input_lower = input_text.lower()
        shared_experiences = []
        
        for indicator in shared_indicators:
            if indicator in input_lower:
                shared_experiences.append(f"referenced_shared_memory: {indicator}")
                
        return shared_experiences
    
    def _detect_unspoken_implications(self, input_text: str, history: List[str]) -> List[str]:
        """Detect what's not being said directly"""
        
        implications = []
        input_lower = input_text.lower()
        
        # Detect hesitation patterns
        if any(phrase in input_lower for phrase in ['i guess', 'maybe', 'sort of', 'kind of']):
            implications.append("hesitation_uncertainty")
            
        # Detect minimization
        if any(phrase in input_lower for phrase in ['just', 'only', 'not a big deal', 'whatever']):
            implications.append("emotional_minimization")
            
        # Detect seeking validation
        if any(phrase in input_lower for phrase in ['right?', 'don\'t you think', 'am i wrong', 'what do you think']):
            implications.append("seeking_validation")
            
        # Detect deeper concern
        if any(phrase in input_lower for phrase in ['but what if', 'i worry', 'concerned', 'afraid']):
            implications.append("underlying_concern")
            
        return implications
    
    def _calculate_context_continuity(self, input_text: str, history: List[str]) -> float:
        """Calculate how well the input continues the conversation context"""
        
        if not history:
            return 0.5
            
        # Simple continuity calculation
        current_words = set(input_text.lower().split())
        recent_context = ' '.join(history[-3:]).lower() if history else ""
        context_words = set(recent_context.split())
        
        if not context_words:
            return 0.5
            
        overlap = len(current_words.intersection(context_words))
        continuity = min(1.0, overlap / max(1, len(context_words) * 0.3))
        
        return continuity
    
    def _identify_primary_emotion(self, input_text: str) -> str:
        """Identify the primary emotion with nuanced understanding"""
        
        emotion_patterns = {
            'joy_excitement': ['excited', 'happy', 'thrilled', 'amazing', 'wonderful', 'love', 'fantastic'],
            'sadness_grief': ['sad', 'depressed', 'grief', 'loss', 'heartbroken', 'devastated', 'mourning'],
            'anxiety_worry': ['anxious', 'worried', 'nervous', 'stressed', 'overwhelmed', 'panic', 'fear'],
            'anger_frustration': ['angry', 'frustrated', 'furious', 'annoyed', 'irritated', 'mad', 'upset'],
            'love_affection': ['love', 'adore', 'cherish', 'treasure', 'affection', 'care', 'devoted'],
            'guilt_shame': ['guilty', 'ashamed', 'regret', 'sorry', 'fault', 'blame', 'terrible'],
            'hope_optimism': ['hope', 'optimistic', 'positive', 'confident', 'believe', 'faith', 'trust'],
            'loneliness_isolation': ['lonely', 'alone', 'isolated', 'disconnected', 'empty', 'abandoned'],
            'gratitude_appreciation': ['grateful', 'thankful', 'appreciate', 'blessed', 'fortunate', 'lucky'],
            'confusion_uncertainty': ['confused', 'uncertain', 'lost', 'unclear', 'don\'t understand', 'puzzled']
        }
        
        input_lower = input_text.lower()
        
        for emotion, patterns in emotion_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                return emotion
                
        return "neutral_contemplative"
    
    def _measure_emotional_intensity(self, input_text: str) -> float:
        """Measure the intensity of emotional expression"""
        
        intensity_indicators = {
            'very_high': ['extremely', 'incredibly', 'absolutely', 'completely', '!!!', 'totally'],
            'high': ['really', 'very', 'so', 'quite', '!!', 'definitely', 'absolutely'],
            'medium': ['pretty', 'somewhat', 'kind of', 'sort of', '!', 'fairly'],
            'low': ['a little', 'slightly', 'maybe', 'perhaps', 'might']
        }
        
        input_lower = input_text.lower()
        intensity = 0.5  # baseline
        
        if any(indicator in input_lower for indicator in intensity_indicators['very_high']):
            intensity = 0.9
        elif any(indicator in input_lower for indicator in intensity_indicators['high']):
            intensity = 0.7
        elif any(indicator in input_lower for indicator in intensity_indicators['medium']):
            intensity = 0.6
        elif any(indicator in input_lower for indicator in intensity_indicators['low']):
            intensity = 0.3
            
        return intensity
    
    def _understand_emotional_subtext(self, input_text: str, user_id: str) -> str:
        """Understand the emotional subtext beneath the words"""
        
        subtext_patterns = {
            'crying_for_help': ['fine', 'okay', 'whatever', 'doesn\'t matter', 'i guess'],
            'seeking_connection': ['anyone else', 'does this happen to you', 'am i the only one'],
            'testing_boundaries': ['you probably think', 'i know this sounds', 'you might not understand'],
            'expressing_vulnerability': ['i\'ve never told anyone', 'this is hard to say', 'i feel exposed'],
            'seeking_permission': ['is it okay if', 'would it be weird', 'i hope you don\'t mind'],
            'processing_trauma': ['keeps happening', 'can\'t get over', 'haunts me', 'stuck in my head'],
            'celebrating_growth': ['first time', 'never thought i could', 'proud of myself', 'breakthrough']
        }
        
        input_lower = input_text.lower()
        
        for subtext, patterns in subtext_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                return subtext
                
        return "direct_expression"
    
    def _determine_empathy_requirements(self, input_text: str, primary_emotion: str) -> List[str]:
        """Determine what kind of empathetic response is needed"""
        
        empathy_map = {
            'sadness_grief': ['acknowledgment', 'gentle_presence', 'validation', 'emotional_holding'],
            'anxiety_worry': ['reassurance', 'grounding', 'perspective', 'calming_presence'],
            'anger_frustration': ['validation', 'understanding', 'non_judgmental_listening'],
            'joy_excitement': ['celebration', 'enthusiasm_matching', 'shared_joy'],
            'guilt_shame': ['compassion', 'self_forgiveness_guidance', 'perspective_reframing'],
            'loneliness_isolation': ['connection', 'belonging_affirmation', 'warm_presence'],
            'confusion_uncertainty': ['clarity', 'patient_guidance', 'supportive_exploration']
        }
        
        return empathy_map.get(primary_emotion, ['basic_empathy', 'active_listening'])
    
    def _identify_emotional_memory_triggers(self, input_text: str, user_id: str) -> List[str]:
        """Identify what might trigger emotional memories"""
        
        # This would connect to user's emotional history
        # For now, identify potential trigger categories
        trigger_categories = {
            'relationship_triggers': ['breakup', 'argument', 'rejection', 'abandonment'],
            'achievement_triggers': ['success', 'failure', 'competition', 'recognition'],
            'family_triggers': ['parent', 'childhood', 'sibling', 'family dynamics'],
            'loss_triggers': ['death', 'goodbye', 'ending', 'change', 'moving'],
            'identity_triggers': ['who am i', 'purpose', 'belonging', 'acceptance']
        }
        
        input_lower = input_text.lower()
        triggers = []
        
        for category, patterns in trigger_categories.items():
            if any(pattern in input_lower for pattern in patterns):
                triggers.append(category)
                
        return triggers
    
    def _determine_response_tone(self, emotion: str, intensity: float, subtext: str) -> str:
        """Determine the most appropriate response tone"""
        
        tone_matrix = {
            ('sadness_grief', 'high'): 'gentle_compassionate',
            ('sadness_grief', 'low'): 'warm_understanding',
            ('anxiety_worry', 'high'): 'calm_reassuring',
            ('anxiety_worry', 'low'): 'supportive_grounding',
            ('joy_excitement', 'high'): 'enthusiastic_celebratory',
            ('joy_excitement', 'low'): 'warm_appreciative',
            ('anger_frustration', 'high'): 'validating_non_reactive',
            ('anger_frustration', 'low'): 'understanding_supportive',
            ('confusion_uncertainty', 'any'): 'patient_clarifying'
        }
        
        intensity_level = 'high' if intensity > 0.7 else 'low'
        tone_key = (emotion, intensity_level)
        
        return tone_matrix.get(tone_key, tone_matrix.get((emotion, 'any'), 'warm_authentic'))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Return top 10 keywords

# Integration function for CNS
def create_enhanced_contextual_emotional_system():
    """Create enhanced contextual emotional system for CNS"""
    return EnhancedContextualEmotionalSystem()

if __name__ == "__main__":
    # Test the system
    system = EnhancedContextualEmotionalSystem()
    
    test_inputs = [
        "I'm really struggling with this decision about my career",
        "That's so exciting! I can't wait to hear more about it",
        "I feel like nobody understands what I'm going through",
        "You know what, I think I'm finally starting to figure things out",
        "This is really hard to talk about, but I trust you"
    ]
    
    print("🧠 Testing Enhanced Contextual Emotional System")
    print("="*60)
    
    for i, input_text in enumerate(test_inputs, 1):
        context = system.analyze_deep_context(input_text, "test_user", [])
        emotion = system.analyze_emotional_intelligence(input_text, "test_user")
        
        print(f"\n{i}. Input: \"{input_text}\"")
        print(f"   Context: {context.topic_thread} (depth: {context.conversation_depth})")
        print(f"   Emotion: {emotion.primary_emotion} ({emotion.emotion_intensity:.2f})")
        print(f"   Subtext: {emotion.emotional_subtext}")
        print(f"   Response Tone: {emotion.appropriate_response_tone}")
        print(f"   Empathy Needs: {', '.join(emotion.empathy_requirements)}")