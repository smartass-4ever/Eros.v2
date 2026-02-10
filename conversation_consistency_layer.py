"""
Conversation Consistency Layer - Makes CNS feel alive across long chats like Xiaoice
"""

import time
import random
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict

class ConversationConsistencyLayer:
    """Maintains personality and emotional consistency across long conversations"""
    
    def __init__(self):
        self.conversation_state = {}
        self.personality_momentum = {}
        self.topic_coherence = {}
        self.emotional_continuity = deque(maxlen=10)
        self.response_patterns = defaultdict(int)
        
    def maintain_consistency(self, user_id: str, user_input: str, base_response: str, 
                           emotional_context: Dict, personality_state: Dict) -> str:
        """Apply consistency checks and adjustments to maintain natural flow"""
        
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {
                "established_tone": None,
                "conversation_energy": 0.5,
                "topic_thread": None,
                "emotional_baseline": {"valence": 0.0, "arousal": 0.5},
                "personality_established": False
            }
        
        state = self.conversation_state[user_id]
        
        # 1. EMOTIONAL CONTINUITY - Smooth transitions like Xiaoice
        consistent_response = self._apply_emotional_continuity(
            base_response, emotional_context, state, user_input
        )
        
        # 2. PERSONALITY MOMENTUM - Consistent character across chat
        consistent_response = self._apply_personality_momentum(
            consistent_response, personality_state, state
        )
        
        # 3. CONVERSATIONAL FLOW - Natural topic transitions
        consistent_response = self._maintain_conversational_flow(
            consistent_response, user_input, state
        )
        
        # 4. ANTI-REPETITION - Vary responses naturally
        consistent_response = self._avoid_repetitive_patterns(
            consistent_response, user_id
        )
        
        # Update state for next interaction
        self._update_conversation_state(user_id, user_input, consistent_response, emotional_context)
        
        return consistent_response
    
    def _apply_emotional_continuity(self, response: str, emotional_context: Dict, 
                                  state: Dict, user_input: str) -> str:
        """Smooth emotional transitions like a real person"""
        
        current_valence = emotional_context.get('valence', 0.0)
        current_arousal = emotional_context.get('arousal', 0.5)
        
        # Check for dramatic emotional shifts that need smoothing
        baseline = state["emotional_baseline"]
        valence_shift = abs(current_valence - baseline["valence"])
        arousal_shift = abs(current_arousal - baseline["arousal"])
        
        # If big emotional shift, acknowledge the transition naturally
        if valence_shift > 0.4 or arousal_shift > 0.3:
            # Add transitional phrases for emotional shifts
            if current_valence > baseline["valence"] + 0.3:
                # Shifting more positive
                transition_phrases = [
                    "That's actually really encouraging to think about. ",
                    "You know what, that does lift my mood. ",
                    "I'm feeling more optimistic about this now. "
                ]
            elif current_valence < baseline["valence"] - 0.3:
                # Shifting more negative  
                transition_phrases = [
                    "That does make me pause and think more seriously. ",
                    "Hmm, that brings up some deeper concerns for me. ",
                    "I'm feeling the weight of what you're sharing. "
                ]
            else:
                transition_phrases = [""]
            
            if transition_phrases[0]:  # If we have a transition
                transition = random.choice(transition_phrases)
                response = f"{transition}{response}"
        
        return response
    
    def _apply_personality_momentum(self, response: str, personality_state: Dict, state: Dict) -> str:
        """Maintain consistent personality traits across conversation"""
        
        if not state["personality_established"]:
            # First few interactions - establish personality clearly
            if personality_state.get('playfulness', 0) > 0.7:
                # Add playful energy consistently
                if random.random() < 0.3 and not any(p in response.lower() for p in ['!', '?', 'haha', 'hehe']):
                    response += " 😊"
            
            if personality_state.get('empathy', 0) > 0.8:
                # Add empathetic warmth
                empathy_indicators = ['understand', 'feel', 'hear', 'sense']
                if not any(indicator in response.lower() for indicator in empathy_indicators):
                    if random.random() < 0.4:
                        response = f"I can sense the importance of this to you. {response}"
            
            state["personality_established"] = True
        
        else:
            # Maintain established personality momentum
            if state.get("established_tone") == "playful":
                # Keep some lightness in responses
                if random.random() < 0.2 and '.' in response:
                    response = response.replace('.', ' :)', 1)  # Replace first period
            
            elif state.get("established_tone") == "thoughtful":
                # Maintain reflective quality
                if not response.startswith(("I think", "That makes me", "I wonder", "It seems")):
                    if random.random() < 0.3:
                        response = f"I find myself thinking that {response.lower()}"
        
        return response
    
    def _maintain_conversational_flow(self, response: str, user_input: str, state: Dict) -> str:
        """Keep natural conversation flow and topic coherence"""
        
        # Track topic continuity
        current_topic = self._extract_topic(user_input)
        
        if state.get("topic_thread") and current_topic != state["topic_thread"]:
            # Topic shift detected - add natural transition
            if random.random() < 0.25:  # 25% chance of topic bridge
                bridge_phrases = [
                    "That's an interesting shift. ",
                    "Speaking of that, ",
                    "That reminds me - "
                ]
                bridge = random.choice(bridge_phrases)
                response = f"{bridge}{response.lower()}"
        
        state["topic_thread"] = current_topic
        return response
    
    def _avoid_repetitive_patterns(self, response: str, user_id: str) -> str:
        """Prevent robotic repetition that breaks immersion"""
        
        # Track response patterns
        pattern_key = response[:30]  # First 30 chars as pattern
        self.response_patterns[f"{user_id}:{pattern_key}"] += 1
        
        # If we've used this pattern recently, vary it
        if self.response_patterns[f"{user_id}:{pattern_key}"] > 1:
            
            # Apply variation strategies
            if response.startswith("That's"):
                variations = ["It's", "This is", "I find this", "This seems"]
                response = f"{random.choice(variations)}{response[6:]}"
            
            elif response.startswith("I think"):
                variations = ["I believe", "It seems to me", "I feel like", "My sense is"]
                response = f"{random.choice(variations)}{response[7:]}"
            
            elif response.startswith("I understand"):
                variations = ["I can see", "I hear what you're saying", "That makes sense", "I get that"]
                response = f"{random.choice(variations)}{response[12:]}"
        
        return response
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from input for coherence tracking"""
        words = text.lower().split()
        
        # Simple topic extraction - first meaningful noun/concept
        topic_indicators = []
        for word in words:
            if len(word) > 4 and word.isalpha():
                topic_indicators.append(word)
                if len(topic_indicators) >= 2:
                    break
        
        return " ".join(topic_indicators) if topic_indicators else "general"
    
    def _update_conversation_state(self, user_id: str, user_input: str, response: str, emotional_context: Dict):
        """Update state for next interaction"""
        
        state = self.conversation_state[user_id]
        
        # Update emotional baseline with momentum
        current_valence = emotional_context.get('valence', 0.0)
        current_arousal = emotional_context.get('arousal', 0.5)
        
        momentum = 0.7  # Emotional momentum factor
        state["emotional_baseline"]["valence"] = (
            state["emotional_baseline"]["valence"] * momentum + current_valence * (1 - momentum)
        )
        state["emotional_baseline"]["arousal"] = (
            state["emotional_baseline"]["arousal"] * momentum + current_arousal * (1 - momentum)
        )
        
        # Update conversation energy
        energy_indicators = len([w for w in user_input.split() if w.endswith('!') or w.isupper()])
        if energy_indicators > 0:
            state["conversation_energy"] = min(1.0, state["conversation_energy"] + 0.2)
        else:
            state["conversation_energy"] = max(0.2, state["conversation_energy"] - 0.1)
        
        # Establish tone for future consistency
        if not state.get("established_tone"):
            if emotional_context.get('arousal', 0) > 0.7:
                state["established_tone"] = "energetic"
            elif '?' in response or response.startswith(("I think", "I wonder")):
                state["established_tone"] = "thoughtful"
            elif any(indicator in response.lower() for indicator in ['haha', '😊', 'fun', 'enjoy']):
                state["established_tone"] = "playful"
    
    def reset_conversation(self, user_id: str):
        """Reset conversation state for new session"""
        if user_id in self.conversation_state:
            del self.conversation_state[user_id]
        
        # Clear response patterns for this user
        keys_to_remove = [k for k in self.response_patterns.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.response_patterns[key]