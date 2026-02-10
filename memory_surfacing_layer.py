"""
Memory Surfacing Layer - Naturally weave personal history into conversations like Xiaoice
Now learns which memory retrievals lead to engagement via ExperienceBus
"""

import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class MemorySurfacingLayer:
    """Naturally injects relevant memories and relationship context into conversations"""
    
    def __init__(self):
        self.conversation_memory = []
        self.relationship_callbacks = defaultdict(list)
        self.topic_threads = {}
        
        self.last_surfaced_memories: Dict[str, List[Dict]] = {}
        self.memory_effectiveness: Dict[str, Dict] = {}
        self.memory_pattern_scores: Dict[str, float] = {}
        
    def surface_relevant_memories(self, user_input: str, user_id: str, companion_data: Dict, cns_memory: List) -> Dict:
        """Find and surface relevant memories naturally"""
        
        current_time = time.time()
        surfaced_elements = {
            "memory_callbacks": [],
            "relationship_context": None,
            "conversation_threads": [],
            "integration_style": "natural"
        }
        
        # 1. RELATIONSHIP MEMORY CALLBACKS
        if user_id in companion_data.get('users', {}):
            user_profile = companion_data['users'][user_id]
            relationship_context = self._extract_relationship_callbacks(user_profile, user_input)
            if relationship_context:
                surfaced_elements["relationship_context"] = relationship_context
        
        # 2. CONVERSATION THREAD CONTINUITY  
        thread_context = self._find_conversation_threads(user_input, user_id)
        if thread_context:
            surfaced_elements["conversation_threads"] = thread_context
        
        # 3. RELEVANT MEMORY INJECTION
        relevant_memories = self._find_contextual_memories(user_input, cns_memory, limit=2)
        if relevant_memories:
            surfaced_elements["memory_callbacks"] = relevant_memories
        
        self.last_surfaced_memories[user_id] = {
            'memories': relevant_memories,
            'relationship_context': surfaced_elements.get("relationship_context"),
            'threads': surfaced_elements.get("conversation_threads"),
            'timestamp': time.time(),
            'input_context': user_input[:100]
        }
        
        return surfaced_elements
    
    def _extract_relationship_callbacks(self, user_profile: Dict, current_input: str) -> Optional[Dict]:
        """Extract relevant relationship context like Xiaoice does"""
        
        # Check relationship stage and interaction history
        relationship_stage = user_profile.get('relationship_stage', 'stranger')
        total_interactions = user_profile.get('total_interactions', 0)
        first_interaction = user_profile.get('first_interaction', 0)
        
        # Calculate relationship age
        if first_interaction:
            days_known = (time.time() - first_interaction) / (24 * 3600)
            
            # XIAOICE-STYLE RELATIONSHIP CALLBACKS
            if relationship_stage == 'friend' and total_interactions > 20:
                return {
                    "type": "established_friendship",
                    "context": f"We've been talking for {days_known:.0f} days now",
                    "warmth_boost": 0.3,
                    "integration_phrases": [
                        "You know, after all our conversations...",
                        "I've been thinking about what you mentioned before...",
                        "It's nice how we've gotten to know each other..."
                    ]
                }
            elif total_interactions > 5:
                return {
                    "type": "growing_familiarity", 
                    "context": f"Getting to know you over {total_interactions} chats",
                    "warmth_boost": 0.2,
                    "integration_phrases": [
                        "I'm starting to understand how you think...",
                        "You seem like the kind of person who...",
                        "I notice you often..."
                    ]
                }
        
        return None
    
    def _find_conversation_threads(self, current_input: str, user_id: str) -> List[Dict]:
        """Find ongoing conversation threads like topic continuity"""
        
        threads = []
        current_time = time.time()
        
        # Check if this continues a previous topic
        if user_id in self.topic_threads:
            user_threads = self.topic_threads[user_id]
            
            for topic, thread_data in user_threads.items():
                if (current_time - thread_data['last_activity']) < 3600:  # 1 hour recency
                    # Simple topic continuity check
                    overlap_words = set(current_input.lower().split()) & set(topic.split())
                    if len(overlap_words) >= 2:
                        threads.append({
                            "type": "topic_continuation",
                            "topic": topic,
                            "context": thread_data['context'],
                            "integration_phrases": [
                                "Going back to what we were discussing...",
                                "That reminds me of when you mentioned...",
                                "This connects to what you said about..."
                            ]
                        })
        
        return threads
    
    def _find_contextual_memories(self, user_input: str, cns_memory: List, limit: int = 2) -> List[Dict]:
        """Find memories that naturally relate to current conversation"""
        
        if not cns_memory:
            return []
        
        input_words = set(user_input.lower().split())
        relevant_memories = []
        
        for fact in cns_memory[:100]:  # Check recent memories
            if hasattr(fact, 'content'):
                fact_words = set(fact.content.lower().split())
                relevance_score = len(input_words & fact_words)
                
                if relevance_score >= 2:  # At least 2 word overlap
                    memory_age = (time.time() - getattr(fact, 'timestamp', time.time())) / (24 * 3600)
                    
                    relevant_memories.append({
                        "content": fact.content,
                        "relevance": relevance_score,
                        "age_days": memory_age,
                        "integration_style": "casual_reference" if memory_age < 7 else "deeper_reflection"
                    })
        
        # Sort by relevance and recency
        relevant_memories.sort(key=lambda x: (x['relevance'], -x['age_days']), reverse=True)
        
        return relevant_memories[:limit]
    
    def integrate_memories_into_response(self, base_response: str, surfaced_elements: Dict) -> str:
        """Naturally integrate surfaced memories into the response like Xiaoice"""
        
        if not any(surfaced_elements.values()):
            return base_response
        
        integrated_response = base_response
        integration_added = False
        
        # 1. ADD RELATIONSHIP CONTEXT
        if surfaced_elements.get("relationship_context"):
            rel_context = surfaced_elements["relationship_context"]
            if not integration_added and random.random() < 0.4:  # 40% chance
                phrase = random.choice(rel_context["integration_phrases"])
                integrated_response = f"{phrase} {base_response}"
                integration_added = True
        
        # 2. ADD MEMORY CALLBACKS
        if surfaced_elements.get("memory_callbacks") and not integration_added:
            memory = surfaced_elements["memory_callbacks"][0]  # Use most relevant
            
            if memory["integration_style"] == "casual_reference" and random.random() < 0.3:
                memory_snippet = memory["content"][:50] + "..." if len(memory["content"]) > 50 else memory["content"]
                integrated_response = f"That reminds me of {memory_snippet}. {base_response}"
                integration_added = True
        
        # 3. ADD CONVERSATION THREAD CONTINUITY
        if surfaced_elements.get("conversation_threads") and not integration_added:
            thread = surfaced_elements["conversation_threads"][0]
            if random.random() < 0.35:  # 35% chance
                phrase = random.choice(thread["integration_phrases"])
                integrated_response = f"{phrase} {base_response}"
        
        return integrated_response
    
    def update_conversation_threads(self, user_input: str, user_id: str, response: str):
        """Update conversation threads for future memory surfacing"""
        
        if user_id not in self.topic_threads:
            self.topic_threads[user_id] = {}
        
        # Extract key topics from current conversation
        words = user_input.lower().split()
        if len(words) > 3:  # Only track substantial topics
            topic_key = " ".join(words[:3])  # First 3 words as topic key
            
            self.topic_threads[user_id][topic_key] = {
                "context": user_input[:100],
                "response": response[:100],
                "last_activity": time.time()
            }
        
        # Clean old threads (older than 24 hours)
        current_time = time.time()
        for topic in list(self.topic_threads[user_id].keys()):
            if (current_time - self.topic_threads[user_id][topic]['last_activity']) > 86400:
                del self.topic_threads[user_id][topic]
    
    def subscribe_to_bus(self, bus):
        """Subscribe to ExperienceBus for memory retrieval learning"""
        bus.subscribe("MemorySurfacingLayer", self.on_experience)
        logger.info("🔗 MemorySurfacingLayer subscribed to ExperienceBus - memory effectiveness learning active")
    
    def on_experience(self, payload):
        """
        Learn which memory retrievals lead to engagement.
        When dopamine signals indicate success, boost those retrieval patterns.
        When sadness signals indicate ghosting, demote those patterns.
        """
        try:
            user_id = payload.user_id
            if not user_id or user_id not in self.last_surfaced_memories:
                return
            
            learning_signals = getattr(payload, 'learning_signals', {}) or {}
            emotional = getattr(payload, 'emotional_analysis', {}) or {}
            
            dopamine = learning_signals.get('dopamine_level', 0)
            sadness = learning_signals.get('sadness_level', 0)
            engagement_positive = dopamine > 0.4 or emotional.get('valence', 0) > 0.3
            engagement_negative = sadness > 0.4 or emotional.get('valence', 0) < -0.3
            
            surfaced = self.last_surfaced_memories.get(user_id, {})
            memories = surfaced.get('memories', [])
            relationship_ctx = surfaced.get('relationship_context')
            
            if not memories and not relationship_ctx:
                return
            
            for memory in memories:
                memory_key = self._get_memory_pattern_key(memory)
                if not memory_key:
                    continue
                
                current_score = self.memory_pattern_scores.get(memory_key, 0.5)
                
                if engagement_positive:
                    new_score = min(1.0, current_score + 0.1 * dopamine)
                    self.memory_pattern_scores[memory_key] = new_score
                    logger.debug(f"📈 Memory pattern boosted: {memory_key[:30]}... ({current_score:.2f} → {new_score:.2f})")
                elif engagement_negative:
                    new_score = max(0.1, current_score - 0.15 * sadness)
                    self.memory_pattern_scores[memory_key] = new_score
                    logger.debug(f"📉 Memory pattern demoted: {memory_key[:30]}... ({current_score:.2f} → {new_score:.2f})")
            
            if relationship_ctx:
                rel_type = relationship_ctx.get('type', 'unknown')
                if rel_type not in self.memory_effectiveness:
                    self.memory_effectiveness[rel_type] = {'success': 0, 'failure': 0}
                
                if engagement_positive:
                    self.memory_effectiveness[rel_type]['success'] += 1
                elif engagement_negative:
                    self.memory_effectiveness[rel_type]['failure'] += 1
            
            if engagement_positive or engagement_negative:
                del self.last_surfaced_memories[user_id]
                
        except Exception as e:
            logger.error(f"MemorySurfacingLayer on_experience error: {e}")
    
    def _get_memory_pattern_key(self, memory: Dict) -> Optional[str]:
        """Generate a key for memory pattern tracking"""
        content = memory.get('content', '')
        if not content:
            return None
        words = content.lower().split()[:5]
        return '_'.join(words) if words else None
    
    def get_boosted_memories(self, memories: List[Dict], boost_threshold: float = 0.6) -> List[Dict]:
        """
        Filter/prioritize memories based on learned effectiveness.
        Memories with high pattern scores get prioritized.
        """
        if not memories:
            return []
        
        scored_memories = []
        for memory in memories:
            pattern_key = self._get_memory_pattern_key(memory)
            score = self.memory_pattern_scores.get(pattern_key, 0.5) if pattern_key else 0.5
            memory_copy = dict(memory)
            memory_copy['effectiveness_score'] = score
            scored_memories.append(memory_copy)
        
        scored_memories.sort(key=lambda x: x.get('effectiveness_score', 0.5), reverse=True)
        
        return scored_memories
    
    def get_relationship_effectiveness_stats(self) -> Dict[str, float]:
        """Get effectiveness rates for different relationship context types"""
        stats = {}
        for rel_type, counts in self.memory_effectiveness.items():
            total = counts['success'] + counts['failure']
            if total > 0:
                stats[rel_type] = counts['success'] / total
        return stats