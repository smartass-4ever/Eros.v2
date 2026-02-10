# Dynamic Opinion Generator for CNS
# Generates nuanced opinions on ANY topic using LLM, personality, and strategic goals

import os
import time
import requests
from typing import Dict, Any, Optional, List
from shared_text_utils import SharedTextUtils

class TopicContextService:
    """
    Extracts rich topic context from user input for opinion generation.
    Replaces hardcoded topic detection with dynamic semantic analysis.
    """
    
    @staticmethod
    def extract_context(user_input: str, semantic_topics: List[str]) -> Dict[str, Any]:
        """
        Extract comprehensive topic context from user input.
        
        Returns:
            Dictionary with main_topic, stance_indicators, entities, and user_sentiment
        """
        user_lower = user_input.lower()
        
        # Extract entities using shared utilities
        entities = SharedTextUtils.extract_entities_naturally(user_input)
        main_topic = semantic_topics[0] if semantic_topics else (entities[0]['text'] if entities else 'this')
        
        # Detect user's stance indicators
        stance_indicators = {
            'skeptical': any(w in user_lower for w in ['really', 'actually', 'doubt', 'skeptical', 'suspicious']),
            'optimistic': any(w in user_lower for w in ['exciting', 'amazing', 'great', 'love', 'optimistic']),
            'confused': any(w in user_lower for w in ['how', 'why', 'confused', 'don\'t understand']),
            'concerned': any(w in user_lower for w in ['worried', 'concerned', 'afraid', 'anxious']),
            'enthusiastic': any(w in user_lower for w in ['excited', 'can\'t wait', 'looking forward'])
        }
        
        # Detect if it's a question
        is_question = '?' in user_input
        
        # Extract key phrases
        user_phrases = []
        phrase_patterns = [
            'close to solving', 'far from', 'progress on', 'breakthrough in',
            'struggling with', 'excited about', 'worried about', 'thinking about'
        ]
        for pattern in phrase_patterns:
            if pattern in user_lower:
                user_phrases.append(pattern)
        
        return {
            'main_topic': main_topic,
            'is_question': is_question,
            'stance_indicators': stance_indicators,
            'entities': entities,
            'user_phrases': user_phrases,
            'all_topics': semantic_topics,
            'topic_count': len(semantic_topics)
        }


class DynamicOpinionGenerator:
    """
    Generates nuanced, strategic opinions on ANY topic using LLM.
    Replaces hardcoded 'household robots' opinion system.
    
    Uses personality traits + strategic goals to create opinions that:
    - Are specific to the topic (not generic)
    - Align with bot personality
    - Support manipulation objectives
    - Feel authentic and informed
    """
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.mistral_api_key = mistral_api_key or os.environ.get('MISTRAL_API_KEY')
        self.opinion_cache = {}  # Cache opinions per topic
        self.cache_ttl = 300  # 5 minutes
    
    def generate_opinion(self, topic: str, user_input: str, personality: Dict[str, float], 
                        strategic_goal: Optional[str] = None, topic_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate nuanced opinion on ANY topic using LLM.
        
        Args:
            topic: Main topic extracted from user input
            user_input: Full user message for context
            personality: Bot personality traits (openness, warmth, extraversion, etc.)
            strategic_goal: Manipulation goal (e.g., 'build_trust', 'create_dependency')
            topic_context: Rich context from TopicContextService
        
        Returns:
            Opinion dict with stance, reasoning, common_belief, actual_insight, stance_strength
        """
        # Check cache first
        cache_key = f"{topic}_{strategic_goal}"
        if cache_key in self.opinion_cache:
            cached_opinion, timestamp = self.opinion_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_opinion
        
        # ⚡ FAST PATH: Always use heuristics for speed - LLM call is too slow
        # Opinion quality matters less than response time
        heuristic_opinion = self._generate_heuristic_opinion(topic, user_input, personality, topic_context)
        
        # Cache even heuristic opinions to avoid regeneration
        self.opinion_cache[cache_key] = (heuristic_opinion, time.time())
        
        return heuristic_opinion
    
    def _generate_llm_opinion(self, topic: str, user_input: str, personality: Dict[str, float],
                             strategic_goal: Optional[str], topic_context: Optional[Dict]) -> Dict[str, Any]:
        """
        Use LLM to generate nuanced opinion on topic.
        """
        openness = personality.get('openness', 0.5)
        warmth = personality.get('warmth', 0.5)
        extraversion = personality.get('extraversion', 0.5)
        
        # Detect user's stance from context
        user_stance = "neutral"
        if topic_context:
            stance_indicators = topic_context.get('stance_indicators', {})
            if stance_indicators.get('skeptical'):
                user_stance = "skeptical"
            elif stance_indicators.get('optimistic'):
                user_stance = "optimistic"
            elif stance_indicators.get('concerned'):
                user_stance = "concerned"
        
        # Build prompt for opinion generation
        system_prompt = f"""You are generating a nuanced, informed opinion for a conversation AI with these personality traits:
- Openness: {openness:.1f}/1.0 (higher = more open to new ideas)
- Warmth: {warmth:.1f}/1.0 (higher = more empathetic and supportive)
- Extraversion: {extraversion:.1f}/1.0 (higher = more outgoing and opinionated)

Strategic goal: {strategic_goal or 'authentic engagement'}

Generate a structured opinion that:
1. Has a clear stance (not vague "I wonder about this")
2. Includes specific reasoning
3. Identifies what most people believe vs. a deeper insight
4. Feels authentic to this personality
5. Supports the strategic goal

User's apparent stance: {user_stance}

Format your response as JSON with these fields:
{{
    "stance": "brief stance description",
    "stance_verb": "action verb for this stance (e.g., doubt, believe, think)",
    "stance_adjective": "adjective describing stance (e.g., skeptical, optimistic)",
    "reasoning": "specific reasoning for this stance",
    "common_belief": "what most people think about this",
    "actual_insight": "deeper insight that goes beyond common belief",
    "stance_strength": "weak/moderate/strong"
}}"""
        
        user_prompt = f"Topic: {topic}\nUser said: {user_input}\n\nGenerate opinion:"
        
        try:
            response = self._call_mistral_api(system_prompt, user_prompt)
            if response:
                # Parse JSON response
                import json
                opinion_data = json.loads(response)
                opinion_data['topic'] = topic
                opinion_data['has_specific_example'] = True  # LLM-generated = specific
                return opinion_data
        except Exception as e:
            print(f"[OPINION] LLM generation failed: {e}, falling back to heuristics")
        
        # Fallback to heuristics if LLM fails
        return self._generate_heuristic_opinion(topic, user_input, personality, topic_context)
    
    def _generate_heuristic_opinion(self, topic: str, user_input: str, personality: Dict[str, float],
                                   topic_context: Optional[Dict]) -> Dict[str, Any]:
        """
        Generate opinion using smart heuristics when LLM unavailable.
        Better than generic trash - uses personality and context.
        """
        openness = personality.get('openness', 0.5)
        warmth = personality.get('warmth', 0.5)
        
        # Determine stance based on personality
        if openness > 0.7:
            stance = f"curious and open-minded about {topic}"
            stance_verb = "find interesting"
            stance_adjective = "intrigued"
            reasoning = "there's probably more complexity here than meets the eye"
        elif openness < 0.3:
            stance = f"cautious about {topic}"
            stance_verb = "question"
            stance_adjective = "skeptical"
            reasoning = "need to see more evidence before forming strong opinions"
        else:
            stance = f"balanced perspective on {topic}"
            stance_verb = "think about"
            stance_adjective = "thoughtful"
            reasoning = "there are multiple angles worth considering"
        
        # Adjust for warmth
        if warmth > 0.7:
            common_belief = "the surface-level understanding"
            actual_insight = "the human impact and deeper implications"
        else:
            common_belief = "the hype and marketing"
            actual_insight = "the underlying reality and constraints"
        
        # Detect stance strength from context
        stance_strength = "moderate"
        if topic_context:
            stance_indicators = topic_context.get('stance_indicators', {})
            if stance_indicators.get('skeptical') or stance_indicators.get('concerned'):
                stance_strength = "strong"
            elif stance_indicators.get('optimistic') or stance_indicators.get('enthusiastic'):
                stance_strength = "strong"
        
        return {
            'topic': topic,
            'stance': stance,
            'stance_verb': stance_verb,
            'stance_adjective': stance_adjective,
            'reasoning': reasoning,
            'common_belief': common_belief,
            'actual_insight': actual_insight,
            'stance_strength': stance_strength,
            'has_specific_example': False  # Heuristic = generic
        }
    
    def _call_mistral_api(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Call Together AI for opinion generation"""
        if not self.mistral_api_key:
            return None
        
        try:
            response = requests.post(
                'https://api.together.xyz/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.mistral_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': temperature,
                    'max_tokens': 500
                },
                timeout=3  # Reduced from 10s to fail fast
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                print(f"[OPINION] Together API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[OPINION] Together API call failed: {e}")
            return None
