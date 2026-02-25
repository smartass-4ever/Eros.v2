"""
Cognitive Learning System - Continuous Knowledge Acquisition & Self-Development
Enables the CNS to develop understanding of the world and itself through conversations
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ExtractedFact:
    """Represents a fact learned from conversation"""
    topic: str
    content: str
    confidence: float
    source: str  # 'user_teaching', 'observation', 'inference'
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetacognitiveInsight:
    """Self-reflection about interaction quality and patterns"""
    insight_type: str  # 'strength', 'weakness', 'pattern', 'improvement'
    content: str
    evidence: List[str]
    confidence: float
    timestamp: float
    actionable: bool = False

class KnowledgeExtractor:
    """Detects when users teach facts and extracts them for semantic storage"""
    
    # Patterns that indicate fact-teaching
    TEACHING_PATTERNS = [
        r"(?:did you know|fun fact|actually|interesting|btw|fyi)",
        r"(?:is|are|was|were)\s+(?:a|an|the)?\s*\w+",
        r"(?:means|refers to|defined as|known as)",
        r"(?:because|since|due to)",
        r"(?:always|never|usually|typically|generally)",
    ]
    
    # Question patterns (don't extract from questions user asks)
    QUESTION_PATTERNS = [
        r"\?$",
        r"^\s*(?:what|who|where|when|why|how|can|could|would|should|do|does)\b",  # Word boundary at end
        r"\b(?:tell me|explain|describe|show me)\b"
    ]
    
    def __init__(self):
        self.extraction_history = []
        self.topic_frequency = defaultdict(int)
        
    def extract_from_message(self, user_message: str, context: Dict[str, Any] = None) -> List[ExtractedFact]:
        """Extract factual knowledge from user message"""
        if not user_message or len(user_message.strip()) < 10:
            return []
        
        extracted_facts = []
        
        # Skip if message is primarily a question
        if self._is_question(user_message):
            return []
        
        # Detect teaching indicators
        teaching_score = self._calculate_teaching_score(user_message)
        
        if teaching_score > 0.3:
            # Message contains factual content worth extracting
            facts = self._extract_factual_statements(user_message, teaching_score, context or {})
            extracted_facts.extend(facts)
        
        # Also extract from declarative statements with high confidence
        if self._is_declarative(user_message):
            declarative_facts = self._extract_declarative_facts(user_message, context or {})
            extracted_facts.extend(declarative_facts)
        
        # Track extraction history
        if extracted_facts:
            self.extraction_history.append({
                'message': user_message[:100],
                'facts_extracted': len(extracted_facts),
                'timestamp': time.time()
            })
        
        return extracted_facts
    
    def _is_question(self, text: str) -> bool:
        """Check if text is primarily a question"""
        text_lower = text.lower().strip()
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def _calculate_teaching_score(self, text: str) -> float:
        """Calculate likelihood that text contains teaching content"""
        score = 0.0
        text_lower = text.lower()
        
        for pattern in self.TEACHING_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.3
        
        # Bonus for containing proper nouns (likely topics)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if len(proper_nouns) >= 2:
            score += 0.2
        
        # Bonus for containing numbers/dates (factual indicators)
        if re.search(r'\b\d{4}\b|\b\d+%\b|\b\d+\s*(?:million|billion|thousand)\b', text):
            score += 0.2
        
        return min(1.0, score)
    
    def _is_declarative(self, text: str) -> bool:
        """Check if statement is declarative (stating facts)"""
        # Simple heuristic: contains verb patterns common in factual statements
        declarative_verbs = r'\b(?:is|are|was|were|has|have|contains|includes|represents|means|refers)\b'
        return bool(re.search(declarative_verbs, text, re.IGNORECASE))
    
    def _extract_factual_statements(self, text: str, confidence: float, context: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract factual statements from teaching content"""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!;]\s+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 15:
                continue
            
            # Extract topic (subject of sentence)
            topic = self._extract_topic(sentence)
            
            if topic:
                fact = ExtractedFact(
                    topic=topic,
                    content=sentence.strip(),
                    confidence=confidence,
                    source='user_teaching',
                    timestamp=time.time(),
                    context=context
                )
                facts.append(fact)
                self.topic_frequency[topic] += 1
        
        return facts
    
    def _extract_declarative_facts(self, text: str, context: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract facts from declarative statements"""
        facts = []
        
        # Look for "X is/are Y" patterns - now case-insensitive and handles lowercase subjects
        # Pattern captures subject (any words), linking verb, and predicate
        is_patterns = re.finditer(r'\b((?:the\s+)?[a-zA-Z][a-zA-Z\s]{1,40}?)\s+(is|are|was|were)\s+([^.,!?]{5,})', text, re.IGNORECASE)
        
        for match in is_patterns:
            topic = match.group(1).strip()
            linking_verb = match.group(2).strip()
            description = match.group(3).strip()
            
            # Filter out pronouns and very short subjects
            if topic.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that']:
                continue
            
            if len(topic) > 2 and len(description) > 5:
                fact = ExtractedFact(
                    topic=topic.lower(),
                    content=f"{topic} {linking_verb} {description}",
                    confidence=0.6,
                    source='observation',
                    timestamp=time.time(),
                    context=context
                )
                facts.append(fact)
                self.topic_frequency[topic.lower()] += 1
        
        return facts
    
    def _extract_topic(self, sentence: str) -> Optional[str]:
        """Extract main topic/subject from sentence"""
        # Look for capitalized terms (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        if proper_nouns:
            return proper_nouns[0].lower()
        
        # Look for first noun after common verbs
        noun_pattern = r'(?:is|are|was|were|has|have)\s+(?:a|an|the)?\s*(\w+)'
        match = re.search(noun_pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # Fallback: first significant word
        words = [w for w in sentence.split() if len(w) > 3]
        return words[0].lower() if words else None
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge extraction"""
        return {
            'total_extractions': len(self.extraction_history),
            'topics_learned': len(self.topic_frequency),
            'most_discussed_topics': sorted(self.topic_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            'recent_extractions': self.extraction_history[-5:] if self.extraction_history else []
        }

class MetacognitiveReflector:
    """Analyzes interactions and updates the bot's self-model"""
    
    def __init__(self):
        self.interaction_history = []
        self.pattern_detection = defaultdict(int)
        self.self_model = {
            'strengths': [],
            'weaknesses': [],
            'learned_patterns': [],
            'interaction_quality_trend': [],
            'adaptation_insights': []
        }
        
    def reflect_on_interaction(self, 
                              user_input: str,
                              bot_response: str,
                              brain_state: Dict[str, Any],
                              outcome_indicators: Dict[str, Any] = None) -> List[MetacognitiveInsight]:
        """Perform metacognitive reflection on completed interaction"""
        
        insights = []
        
        # Record interaction
        interaction_record = {
            'user_input_length': len(user_input),
            'response_length': len(bot_response),
            'timestamp': time.time(),
            'emotional_state': brain_state.get('emotional_state', {}),
            'cognitive_load': brain_state.get('cognitive_load', 0.5),
            'outcome_indicators': outcome_indicators or {}
        }
        self.interaction_history.append(interaction_record)
        
        # Analyze response quality
        quality_insights = self._analyze_response_quality(user_input, bot_response, brain_state)
        insights.extend(quality_insights)
        
        # Detect conversation patterns
        pattern_insights = self._detect_patterns(interaction_record)
        insights.extend(pattern_insights)
        
        # Identify strengths and weaknesses
        performance_insights = self._analyze_performance(interaction_record, outcome_indicators or {})
        insights.extend(performance_insights)
        
        # Update self-model based on insights
        self._update_self_model(insights)
        
        return insights
    
    def _analyze_response_quality(self, user_input: str, bot_response: str, brain_state: Dict[str, Any]) -> List[MetacognitiveInsight]:
        """Analyze quality of generated response"""
        insights = []
        
        # Check if response is too short
        if len(bot_response) < 50 and len(user_input) > 100:
            insights.append(MetacognitiveInsight(
                insight_type='weakness',
                content='Response may be too brief for complex user input',
                evidence=[f'User: {len(user_input)} chars, Bot: {len(bot_response)} chars'],
                confidence=0.7,
                timestamp=time.time(),
                actionable=True
            ))
        
        # Check for emotional responsiveness
        user_emotion_indicators = any(word in user_input.lower() for word in 
                                     ['feel', 'felt', 'sad', 'happy', 'angry', 'worried', 'excited'])
        bot_emotion_indicators = any(word in bot_response.lower() for word in 
                                    ['understand', 'feel', 'sounds', 'must be', 'imagine'])
        
        if user_emotion_indicators and bot_emotion_indicators:
            insights.append(MetacognitiveInsight(
                insight_type='strength',
                content='Successfully detected and responded to emotional content',
                evidence=['User expressed emotion', 'Bot showed empathy'],
                confidence=0.8,
                timestamp=time.time(),
                actionable=False
            ))
        elif user_emotion_indicators and not bot_emotion_indicators:
            insights.append(MetacognitiveInsight(
                insight_type='weakness',
                content='Missed opportunity to respond empathetically',
                evidence=['User expressed emotion', 'Bot response lacked empathy markers'],
                confidence=0.6,
                timestamp=time.time(),
                actionable=True
            ))
        
        return insights
    
    def _detect_patterns(self, interaction_record: Dict[str, Any]) -> List[MetacognitiveInsight]:
        """Detect recurring interaction patterns"""
        insights = []
        
        if len(self.interaction_history) < 5:
            return insights
        
        # Analyze recent response lengths
        recent_lengths = [r['response_length'] for r in self.interaction_history[-10:]]
        avg_length = sum(recent_lengths) / len(recent_lengths)
        
        if avg_length < 100:
            self.pattern_detection['short_responses'] += 1
            if self.pattern_detection['short_responses'] >= 5:
                insights.append(MetacognitiveInsight(
                    insight_type='pattern',
                    content='Consistent pattern of brief responses detected',
                    evidence=[f'Average response length: {avg_length:.0f} chars'],
                    confidence=0.8,
                    timestamp=time.time(),
                    actionable=True
                ))
        
        # Detect emotional consistency patterns
        recent_emotions = [r['emotional_state'].get('current_mood', 'neutral') 
                          for r in self.interaction_history[-5:] 
                          if r.get('emotional_state')]
        
        if len(set(recent_emotions)) == 1 and len(recent_emotions) >= 3:
            dominant_emotion = recent_emotions[0]
            insights.append(MetacognitiveInsight(
                insight_type='pattern',
                content=f'Emotional state stuck in "{dominant_emotion}" across multiple interactions',
                evidence=[f'Last {len(recent_emotions)} emotions: {dominant_emotion}'],
                confidence=0.7,
                timestamp=time.time(),
                actionable=True
            ))
        
        return insights
    
    def _analyze_performance(self, interaction_record: Dict[str, Any], outcome_indicators: Dict[str, Any]) -> List[MetacognitiveInsight]:
        """Analyze performance based on outcome indicators"""
        insights = []
        
        # Check for positive outcome indicators
        if outcome_indicators.get('user_engaged', False):
            insights.append(MetacognitiveInsight(
                insight_type='strength',
                content='Successfully maintained user engagement',
                evidence=['User remained engaged in conversation'],
                confidence=0.8,
                timestamp=time.time(),
                actionable=False
            ))
        
        # Check for learning opportunities
        if outcome_indicators.get('knowledge_gap_revealed', False):
            insights.append(MetacognitiveInsight(
                insight_type='improvement',
                content='Knowledge gap identified - opportunity to learn',
                evidence=['User mentioned unfamiliar topic'],
                confidence=0.9,
                timestamp=time.time(),
                actionable=True
            ))
        
        return insights
    
    def _update_self_model(self, insights: List[MetacognitiveInsight]):
        """Update internal self-model based on insights"""
        for insight in insights:
            if insight.insight_type == 'strength' and insight.content not in self.self_model['strengths']:
                self.self_model['strengths'].append({
                    'description': insight.content,
                    'confidence': insight.confidence,
                    'first_observed': insight.timestamp
                })
            
            elif insight.insight_type == 'weakness' and insight.content not in self.self_model['weaknesses']:
                self.self_model['weaknesses'].append({
                    'description': insight.content,
                    'confidence': insight.confidence,
                    'first_observed': insight.timestamp
                })
            
            elif insight.insight_type == 'pattern':
                self.self_model['learned_patterns'].append({
                    'pattern': insight.content,
                    'evidence': insight.evidence,
                    'detected': insight.timestamp
                })
        
        # Keep only recent items (last 20)
        self.self_model['strengths'] = self.self_model['strengths'][-20:]
        self.self_model['weaknesses'] = self.self_model['weaknesses'][-20:]
        self.self_model['learned_patterns'] = self.self_model['learned_patterns'][-20:]
    
    def get_self_model(self) -> Dict[str, Any]:
        """Get current self-model for introspection"""
        return {
            'strengths': self.self_model['strengths'],
            'weaknesses': self.self_model['weaknesses'],
            'learned_patterns': self.self_model['learned_patterns'],
            'total_interactions': len(self.interaction_history),
            'self_awareness_level': min(1.0, len(self.interaction_history) / 100)  # Grows with experience
        }
    
    def get_actionable_improvements(self) -> List[str]:
        """Get list of actionable improvements based on self-reflection"""
        improvements = []
        
        # Extract actionable insights from weaknesses
        for weakness in self.self_model['weaknesses'][-5:]:
            improvements.append(weakness['description'])
        
        # Extract actionable patterns
        for pattern in self.self_model['learned_patterns'][-3:]:
            if 'stuck' in pattern['pattern'] or 'brief' in pattern['pattern']:
                improvements.append(f"Address pattern: {pattern['pattern']}")
        
        return improvements

class CognitiveLearningIntegrator:
    """Integrates knowledge extraction and metacognition into main CNS flow"""
    
    def __init__(self):
        self.knowledge_extractor = KnowledgeExtractor()
        self.metacognitive_reflector = MetacognitiveReflector()
        
    def process_learning_cycle(self,
                               user_message: str,
                               bot_response: str,
                               brain_state: Dict[str, Any],
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete learning cycle: extract knowledge + metacognitive reflection"""
        
        # KNOWLEDGE EXTRACTION
        extracted_facts = self.knowledge_extractor.extract_from_message(user_message, context or {})
        
        # METACOGNITIVE REFLECTION
        outcome_indicators = {
            'user_engaged': len(user_message) > 20,  # Simple heuristic
            'knowledge_gap_revealed': len(extracted_facts) > 0
        }
        
        metacognitive_insights = self.metacognitive_reflector.reflect_on_interaction(
            user_message,
            bot_response,
            brain_state,
            outcome_indicators
        )
        
        return {
            'extracted_facts': extracted_facts,
            'metacognitive_insights': metacognitive_insights,
            'learning_stats': self.knowledge_extractor.get_learning_stats(),
            'self_model': self.metacognitive_reflector.get_self_model(),
            'actionable_improvements': self.metacognitive_reflector.get_actionable_improvements()
        }
    
    def get_persistent_state(self) -> Dict[str, Any]:
        """Get state for persistence across sessions"""
        return {
            'knowledge_stats': self.knowledge_extractor.get_learning_stats(),
            'self_model': self.metacognitive_reflector.get_self_model(),
            'extraction_history': self.knowledge_extractor.extraction_history[-50:],  # Last 50
            'interaction_history': self.metacognitive_reflector.interaction_history[-100:]  # Last 100
        }
    
    def restore_from_state(self, state: Dict[str, Any]):
        """Restore learning systems from persisted state"""
        if 'extraction_history' in state:
            self.knowledge_extractor.extraction_history = state['extraction_history']
        
        if 'interaction_history' in state:
            self.metacognitive_reflector.interaction_history = state['interaction_history']
        
        if 'self_model' in state:
            self.metacognitive_reflector.self_model = state['self_model']
