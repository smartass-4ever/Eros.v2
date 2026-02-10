"""
Intelligent Memory System - Brain-like Memory Coordination
Coordinates working, episodic, semantic, and emotional memory like a real brain

ENHANCED with:
- Temporal phrase detection ("last week", "yesterday", "that time when...")
- Semantic similarity beyond keyword matching
- Tertiary mention support with lower priority weighting
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from cognitive_orchestrator import MemoryType, CognitiveState
import time
import re
import copy
from datetime import datetime, timedelta


class TemporalResolver:
    """Resolves temporal phrases to approximate time ranges"""
    
    TEMPORAL_PATTERNS = {
        r'\byesterday\b': timedelta(days=1),
        r'\blast\s*night\b': timedelta(days=1),
        r'\btoday\b': timedelta(hours=12),
        r'\bearlier\b': timedelta(hours=6),
        r'\blast\s*week\b': timedelta(weeks=1),
        r'\bfew\s*days\s*ago\b': timedelta(days=3),
        r'\bcouple\s*days\s*ago\b': timedelta(days=2),
        r'\blast\s*month\b': timedelta(weeks=4),
        r'\brecently\b': timedelta(days=7),
        r'\bthe\s*other\s*day\b': timedelta(days=3),
        r'\ba\s*while\s*ago\b': timedelta(weeks=2),
        r'\blast\s*time\b': timedelta(days=14),
        r'\bthat\s*time\s*when\b': timedelta(days=30),
        r'\bremember\s*when\b': timedelta(days=30),
    }
    
    @classmethod
    def extract_temporal_context(cls, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal hints from query, return time range context"""
        query_lower = query.lower()
        
        for pattern, delta in cls.TEMPORAL_PATTERNS.items():
            if re.search(pattern, query_lower):
                now = time.time()
                target_time = now - delta.total_seconds()
                return {
                    'has_temporal_reference': True,
                    'target_timestamp': target_time,
                    'time_window': delta.total_seconds(),
                    'pattern_matched': pattern
                }
        
        return {'has_temporal_reference': False}
    
    @classmethod
    def score_temporal_match(cls, memory_timestamp: float, temporal_context: Dict) -> float:
        """Score how well a memory's timestamp matches the temporal query"""
        if not temporal_context.get('has_temporal_reference'):
            return 0.0
        
        target = temporal_context['target_timestamp']
        window = temporal_context['time_window']
        
        time_diff = abs(memory_timestamp - target)
        
        if time_diff < window * 0.5:
            return 1.0
        elif time_diff < window:
            return 0.7
        elif time_diff < window * 2:
            return 0.4
        elif time_diff < window * 4:
            return 0.2
        else:
            return 0.0


class SemanticMatcher:
    """Enhanced semantic matching beyond keyword overlap"""
    
    CONCEPT_SYNONYMS = {
        'talked': ['discussed', 'mentioned', 'said', 'conversation', 'chat'],
        'remember': ['recall', 'mentioned', 'told', 'said', 'brought up'],
        'work': ['job', 'career', 'office', 'boss', 'project', 'meeting'],
        'relationship': ['dating', 'boyfriend', 'girlfriend', 'partner', 'love', 'crush'],
        'family': ['mom', 'dad', 'parent', 'sister', 'brother', 'kid', 'child'],
        'problem': ['issue', 'trouble', 'struggle', 'challenge', 'difficulty'],
        'feeling': ['emotion', 'mood', 'felt', 'feel', 'emotional'],
        'happy': ['excited', 'joy', 'glad', 'pleased', 'good'],
        'sad': ['upset', 'down', 'depressed', 'unhappy', 'low'],
        'angry': ['mad', 'frustrated', 'annoyed', 'irritated', 'pissed'],
        'scared': ['afraid', 'worried', 'anxious', 'nervous', 'fear'],
        'help': ['assist', 'support', 'advice', 'guidance', 'suggestion'],
        'friend': ['buddy', 'pal', 'bestie', 'mate'],
        'money': ['cash', 'finances', 'budget', 'debt', 'pay', 'salary'],
        'school': ['college', 'university', 'class', 'study', 'exam', 'test'],
    }
    
    @classmethod
    def _sanitize_token(cls, word: str) -> str:
        """Strip punctuation from a word for clean matching"""
        return re.sub(r'[^\w\s]', '', word).strip().lower()
    
    @classmethod
    def expand_query(cls, query: str) -> set:
        """Expand query words with synonyms for broader matching"""
        raw_words = query.lower().split()
        words = set(cls._sanitize_token(w) for w in raw_words if cls._sanitize_token(w))
        expanded = set(words)
        
        for word in words:
            for concept, synonyms in cls.CONCEPT_SYNONYMS.items():
                if word == concept or word in synonyms:
                    expanded.add(concept)
                    expanded.update(synonyms)
        
        return expanded
    
    @classmethod
    def calculate_semantic_similarity(cls, query: str, content: str) -> float:
        """Calculate semantic similarity with synonym expansion"""
        expanded_query = cls.expand_query(query)
        content_words = set(content.lower().split())
        
        direct_overlap = len(expanded_query & content_words)
        
        expanded_content = cls.expand_query(content)
        concept_overlap = len(expanded_query & expanded_content)
        
        query_len = max(len(expanded_query), 1)
        
        direct_score = min(1.0, direct_overlap / query_len)
        concept_score = min(1.0, concept_overlap / (query_len * 2))
        
        return direct_score * 0.7 + concept_score * 0.3
    
    @classmethod
    def detect_tertiary_mentions(cls, query: str, content: str) -> Tuple[bool, float]:
        """
        Detect indirect/tertiary references with lower confidence.
        Returns (is_tertiary_match, confidence_weight capped at 0.6)
        """
        if not query or not query.strip():
            return False, 0.0
            
        query_lower = query.lower()
        content_lower = content.lower()
        
        indirect_phrases = [
            r'\bthat\s+thing\b',
            r'\bwhat\s+we\s+(?:talked|discussed)\b',
            r'\byou\s+know\b',
            r'\blike\s+before\b',
            r'\bsame\s+(?:thing|situation)\b',
            r'\bremember\b',
        ]
        
        has_indirect_reference = any(re.search(p, query_lower) for p in indirect_phrases)
        
        if has_indirect_reference:
            partial_match = cls.calculate_semantic_similarity(query, content) > 0.2
            if partial_match:
                return True, 0.4
        
        words = query_lower.split()
        if len(words) < 2:
            return False, 0.0
            
        two_word_phrases = []
        for i in range(len(words) - 1):
            two_word_phrases.append(f"{words[i]} {words[i+1]}")
        
        phrase_matches = sum(1 for phrase in two_word_phrases if phrase in content_lower)
        if phrase_matches > 0:
            weight = min(0.6, 0.3 + (phrase_matches * 0.1))
            return True, weight
        
        return False, 0.0

# Import database persistence layer
try:
    from cns_database import MemoryPersistence
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[MEMORY] ⚠️ Database persistence not available - using in-memory only")

@dataclass
class MemoryResult:
    content: Any
    confidence: float
    retrieval_time: float
    memory_type: MemoryType
    associations: List[str] = None

class WorkingMemory:
    """Immediate context memory - last few exchanges"""
    
    def __init__(self, capacity: int = 7):  # Miller's magic number ±2
        self.capacity = capacity
        self.buffer = []
        self.current_context = {}
        
    def store(self, item: Dict[str, Any]):
        """Store item in working memory"""
        self.buffer.append({
            'content': item,
            'timestamp': time.time(),
            'access_count': 0
        })
        
        # 🔦 DEBUG: Log what we're storing
        print(f"[MEMORY-STORE] 💾 Working memory stored: keys={list(item.keys())}, raw_input={item.get('raw_input', 'MISSING')[:50]}...")
        
        # Maintain capacity
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            
    def recall(self, query: str) -> Optional[MemoryResult]:
        """Quick recall from working memory"""
        start_time = time.time()
        
        # 🔦 DEBUG: Log what we're searching for
        print(f"[MEMORY-SEARCH] 🔍 Working memory searching for: '{query[:50]}...' in {len(self.buffer)} items")
        
        for item in reversed(self.buffer):  # Most recent first
            content_str = str(item['content']).lower()
            # 🔦 DEBUG: Show what we're comparing against
            print(f"[MEMORY-SEARCH] 🔎 Checking item: content_preview='{content_str[:80]}...'")
            
            if query.lower() in content_str:
                print(f"[MEMORY-SEARCH] ✅ MATCH FOUND!")
                item['access_count'] += 1
                return MemoryResult(
                    content=item['content'],
                    confidence=0.9,  # High confidence for recent items
                    retrieval_time=time.time() - start_time,
                    memory_type=MemoryType.WORKING
                )
        
        print(f"[MEMORY-SEARCH] ❌ No matches found in working memory")
        return None
        
    def get_context(self) -> Dict[str, Any]:
        """Get current conversational context"""
        if not self.buffer:
            return {}
            
        recent_items = self.buffer[-3:]  # Last 3 exchanges
        return {
            'recent_topics': [item['content'].get('topic', '') for item in recent_items],
            'recent_emotions': [item['content'].get('emotion', '') for item in recent_items],
            'conversation_flow': len(self.buffer)
        }

class EpisodicMemory:
    """Personal experiences and interactions memory"""
    
    def __init__(self):
        self.episodes = []
        self.user_interactions = {}
        
    def store_episode(self, user_id: str, content: Dict[str, Any]):
        """Store a personal interaction episode"""
        episode = {
            'user_id': user_id,
            'content': content,
            'timestamp': time.time(),
            'emotional_context': content.get('emotion', {}),
            'importance': self._calculate_importance(content)
        }
        
        self.episodes.append(episode)
        
        # Update user interaction history
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        self.user_interactions[user_id].append(episode)
        
    def recall_episode(self, user_id: str, query: str) -> Optional[MemoryResult]:
        """Recall similar past episodes with this user - ENHANCED with semantic + temporal matching"""
        start_time = time.time()
        
        if user_id not in self.user_interactions:
            return None
        
        temporal_context = TemporalResolver.extract_temporal_context(query)
        
        best_match = None
        best_score = 0.0
        
        best_match_type = None
        for episode in reversed(self.user_interactions[user_id]):
            match_result = self._episode_matches_enhanced(episode, query, temporal_context)
            if match_result['is_match'] and match_result['score'] > best_score:
                best_score = match_result['score']
                best_match = episode
                best_match_type = match_result['match_type']
        
        if best_match:
            confidence = min(1.0, best_match['importance'] * best_score)
            result_content = copy.deepcopy(best_match)
            result_content['match_type'] = best_match_type
            return MemoryResult(
                content=result_content,
                confidence=confidence,
                retrieval_time=time.time() - start_time,
                memory_type=MemoryType.EPISODIC
            )
        return None
        
    def _calculate_importance(self, content: Dict[str, Any]) -> float:
        """Calculate episode importance based on emotional intensity and novelty"""
        emotional_weight = content.get('emotion', {}).get('intensity', 0.0)
        topic_novelty = 0.5
        return min(1.0, emotional_weight * 0.7 + topic_novelty * 0.3)
    
    def _episode_matches_enhanced(self, episode: Dict[str, Any], query: str, temporal_context: Dict) -> Dict:
        """
        Enhanced episode matching with:
        - Direct keyword matching (highest confidence)
        - Semantic similarity with synonym expansion (medium confidence)
        - Temporal matching for time-based queries (context boost)
        - Tertiary/indirect mentions (lower confidence)
        """
        episode_text = str(episode['content']).lower()
        episode_timestamp = episode.get('timestamp', 0)
        
        query_words = query.lower().split()
        direct_matches = sum(1 for word in query_words if word in episode_text and len(word) > 2)
        if direct_matches >= 2:
            temporal_boost = TemporalResolver.score_temporal_match(episode_timestamp, temporal_context) * 0.2
            return {
                'is_match': True,
                'score': 0.9 + temporal_boost,
                'match_type': 'direct'
            }
        
        semantic_score = SemanticMatcher.calculate_semantic_similarity(query, episode_text)
        if semantic_score > 0.35:
            temporal_boost = TemporalResolver.score_temporal_match(episode_timestamp, temporal_context) * 0.2
            return {
                'is_match': True,
                'score': 0.6 + (semantic_score * 0.3) + temporal_boost,
                'match_type': 'semantic'
            }
        
        if temporal_context.get('has_temporal_reference'):
            temporal_score = TemporalResolver.score_temporal_match(episode_timestamp, temporal_context)
            if temporal_score > 0.5:
                return {
                    'is_match': True,
                    'score': 0.4 + temporal_score * 0.4,
                    'match_type': 'temporal'
                }
        
        is_tertiary, tertiary_weight = SemanticMatcher.detect_tertiary_mentions(query, episode_text)
        if is_tertiary:
            return {
                'is_match': True,
                'score': tertiary_weight,
                'match_type': 'tertiary'
            }
        
        return {'is_match': False, 'score': 0.0, 'match_type': None}
        
    def _episode_matches(self, episode: Dict[str, Any], query: str) -> bool:
        """Legacy simple matching - kept for backward compatibility"""
        episode_text = str(episode['content']).lower()
        return any(word in episode_text for word in query.lower().split())

class EmotionalMemory:
    """Emotional associations and patterns"""
    
    def __init__(self):
        self.emotional_associations = {}
        self.pattern_history = []
        
    def store_emotional_context(self, trigger: str, emotion_data: Dict[str, Any]):
        """Store emotional association"""
        if trigger not in self.emotional_associations:
            self.emotional_associations[trigger] = []
            
        self.emotional_associations[trigger].append({
            'emotion': emotion_data,
            'timestamp': time.time(),
            'strength': emotion_data.get('intensity', 0.0)
        })
        
    def recall_emotional_context(self, query: str) -> Optional[MemoryResult]:
        """Recall emotional context for similar situations"""
        start_time = time.time()
        
        for trigger, associations in self.emotional_associations.items():
            if trigger.lower() in query.lower():
                # Get most recent and strongest association
                best_association = max(associations, key=lambda x: x['strength'])
                return MemoryResult(
                    content=best_association,
                    confidence=best_association['strength'],
                    retrieval_time=time.time() - start_time,
                    memory_type=MemoryType.EMOTIONAL
                )
        return None

class IntelligentMemorySystem:
    """Coordinates all memory types like a real brain"""
    
    def __init__(self, world_model):
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.emotional_memory = EmotionalMemory()
        self.semantic_memory = world_model  # Use existing WorldModelMemory
        
        # Initialize database persistence for long-term memories
        self.db_persistence = None
        self.db_available = False  # Runtime availability flag
        if DB_AVAILABLE:
            try:
                import os
                if os.environ.get('DATABASE_URL'):
                    self.db_persistence = MemoryPersistence()
                    # Test connection with a simple query
                    session = self.db_persistence.db.get_session()
                    session.close()
                    self.db_available = True
                    print("[MEMORY] ✅ Database persistence connected for long-term memories")
                else:
                    print("[MEMORY] ⚠️ DATABASE_URL not set - using in-memory only")
            except Exception as e:
                print(f"[MEMORY] ⚠️ Database connection failed: {e} - using in-memory only")
                self.db_persistence = None
                self.db_available = False
        
        # Memory search limits based on cognitive state
        self.search_depth_limits = {
            CognitiveState.FRESH: 10,
            CognitiveState.ACTIVE: 7,
            CognitiveState.TIRED: 4,
            CognitiveState.OVERWHELMED: 2
        }
        
    def coordinated_memory_search(self, query: str, memory_sequence: List[MemoryType], 
                                cognitive_state: CognitiveState, user_id: str = None) -> Dict[MemoryType, MemoryResult]:
        """Coordinate memory search with advanced salience filtering and relevance scoring"""
        
        results = {}
        all_candidates = []
        search_limit = self.search_depth_limits.get(cognitive_state, 5)
        
        # Phase 1: Collect all memory candidates from different systems
        for memory_type in memory_sequence:
            candidates = self._get_memory_candidates(memory_type, query, user_id)
            all_candidates.extend(candidates)
            
        # Phase 2: Advanced salience filtering
        filtered_candidates = self._apply_salience_filtering(all_candidates, query, cognitive_state)
        
        # Phase 3: Select best candidates up to search limit
        best_candidates = sorted(filtered_candidates, key=lambda x: x['salience_score'], reverse=True)[:search_limit]
        
        # Phase 4: Create MemoryResult objects for selected candidates
        for candidate in best_candidates:
            memory_type = candidate['memory_type']
            if memory_type not in results:  # Take only the best result per memory type
                results[memory_type] = MemoryResult(
                    content=candidate['content'],
                    confidence=candidate['confidence'] * candidate['salience_score'],  # Boost confidence with salience
                    retrieval_time=candidate['retrieval_time'],
                    memory_type=memory_type,
                    associations=candidate.get('associations', [])
                )
                
        return results
    
    def _get_memory_candidates(self, memory_type: MemoryType, query: str, user_id: str = None) -> List[Dict]:
        """Get all candidate memories from a specific memory system"""
        candidates = []
        start_time = time.time()
        
        if memory_type == MemoryType.WORKING:
            result = self.working_memory.recall(query)
            if result:
                candidates.append({
                    'memory_type': memory_type,
                    'content': result.content,
                    'confidence': result.confidence,
                    'retrieval_time': result.retrieval_time,
                    'raw_result': result
                })
                
        elif memory_type == MemoryType.EPISODIC and user_id:
            # 1) Try database first for persistent memories
            if self.db_available and self.db_persistence:
                try:
                    db_memories = self.db_persistence.search_memories(user_id, query, limit=5)
                    print(f"[MEMORY-DB] 🔍 Episodic search for '{query[:30]}...' returned {len(db_memories)} memories")
                    for mem in db_memories:
                        if mem.get('memory_type') == 'episodic':
                            candidates.append({
                                'memory_type': memory_type,
                                'content': mem.get('content', ''),
                                'confidence': mem.get('importance', 0.7),
                                'retrieval_time': time.time() - start_time,
                                'raw_result': mem,
                                'associations': []
                            })
                except Exception as e:
                    print(f"[MEMORY-DB] ❌ Database episodic search failed: {e}")
            
            # 2) Also check in-memory for very recent episodes
            result = self.episodic_memory.recall_episode(user_id, query)
            if result:
                candidates.append({
                    'memory_type': memory_type,
                    'content': result.content,
                    'confidence': result.confidence,
                    'retrieval_time': result.retrieval_time,
                    'raw_result': result
                })
                
        elif memory_type == MemoryType.SEMANTIC:
            # 1) Try database first for stored semantic facts
            if self.db_available and self.db_persistence and user_id:
                try:
                    db_memories = self.db_persistence.search_memories(user_id, query, limit=5)
                    print(f"[MEMORY-DB] 🔍 Semantic search for '{query[:30]}...' returned {len(db_memories)} memories")
                    for mem in db_memories:
                        if mem.get('memory_type') == 'semantic':
                            candidates.append({
                                'memory_type': memory_type,
                                'content': mem.get('content', ''),
                                'confidence': mem.get('importance', 0.8),
                                'retrieval_time': time.time() - start_time,
                                'raw_result': mem
                            })
                except Exception as e:
                    print(f"[MEMORY-DB] ❌ Database semantic search failed: {e}")
            
            # 2) Also check WorldModelMemory
            semantic_content = self.semantic_memory.recall(query)
            if semantic_content:
                candidates.append({
                    'memory_type': memory_type,
                    'content': semantic_content,
                    'confidence': 0.8,
                    'retrieval_time': time.time() - start_time,
                    'raw_result': None
                })
                
        elif memory_type == MemoryType.EMOTIONAL:
            result = self.emotional_memory.recall_emotional_context(query)
            if result:
                candidates.append({
                    'memory_type': memory_type,
                    'content': result.content,
                    'confidence': result.confidence,
                    'retrieval_time': result.retrieval_time,
                    'raw_result': result,
                    'associations': result.associations
                })
                
        return candidates
    
    def _apply_salience_filtering(self, candidates: List[Dict], query: str, cognitive_state: CognitiveState) -> List[Dict]:
        """
        Apply sophisticated salience filtering to memory candidates.
        ENHANCED with semantic similarity, temporal awareness, and tertiary mentions.
        """
        temporal_context = TemporalResolver.extract_temporal_context(query)
        
        for candidate in candidates:
            salience_score = 0.0
            content_str = str(candidate['content']).lower()
            
            semantic_similarity = SemanticMatcher.calculate_semantic_similarity(query, content_str)
            salience_score += semantic_similarity * 0.35
            
            type_priority = {
                MemoryType.WORKING: 0.4,
                MemoryType.EPISODIC: 0.3,
                MemoryType.EMOTIONAL: 0.2,
                MemoryType.SEMANTIC: 0.1
            }
            salience_score += type_priority.get(candidate['memory_type'], 0.1)
            
            salience_score += candidate['confidence'] * 0.2
            
            if candidate['memory_type'] == MemoryType.EPISODIC:
                raw_result = candidate.get('raw_result', {})
                if isinstance(raw_result, dict):
                    timestamp = raw_result.get('timestamp', 0)
                elif hasattr(raw_result, 'content') and isinstance(raw_result.content, dict):
                    timestamp = raw_result.content.get('timestamp', 0)
                else:
                    timestamp = 0
                
                if timestamp:
                    temporal_score = TemporalResolver.score_temporal_match(timestamp, temporal_context)
                    if temporal_score > 0:
                        salience_score += temporal_score * 0.25
                        candidate['match_type'] = 'temporal'
                    
                    age_hours = (time.time() - timestamp) / 3600
                    if age_hours < 24:
                        salience_score += 0.15
                    elif age_hours < 168:
                        salience_score += 0.08
            
            if candidate['memory_type'] == MemoryType.EMOTIONAL:
                emotion_words = ['feel', 'emotion', 'mood', 'happy', 'sad', 'anxious', 'worried', 'excited']
                if any(word in query.lower() for word in emotion_words):
                    salience_score += 0.2
            
            is_tertiary, tertiary_weight = SemanticMatcher.detect_tertiary_mentions(query, content_str)
            if is_tertiary:
                salience_score += tertiary_weight * 0.15
                candidate['match_type'] = 'tertiary'
            
            if cognitive_state in [CognitiveState.TIRED, CognitiveState.OVERWHELMED]:
                if candidate['memory_type'] == MemoryType.WORKING:
                    salience_score += 0.15
                else:
                    salience_score -= 0.1
            
            candidate['salience_score'] = min(1.0, max(0.0, salience_score))
            
        salience_threshold = 0.15
        filtered = [c for c in candidates if c['salience_score'] >= salience_threshold]
        
        if filtered:
            print(f"[MEMORY-SALIENCE] 🎯 Filtered {len(candidates)} → {len(filtered)} candidates (threshold {salience_threshold})")
            for c in filtered[:3]:
                print(f"  - {c['memory_type'].name}: score={c['salience_score']:.2f}, type={c.get('match_type', 'direct')}")
        
        return filtered
        
    def store_interaction(self, user_id: str, content: Dict[str, Any]):
        """Store interaction across appropriate memory systems"""
        
        # Always store in working memory
        self.working_memory.store(content)
        
        # Get emotion data for importance calculation
        emotion_data = content.get('emotion', {})
        emotional_intensity = emotion_data.get('intensity', 0) if isinstance(emotion_data, dict) else 0
        
        # Store in episodic if it has personal significance
        if emotional_intensity > 0.3:
            self.episodic_memory.store_episode(user_id, content)
            
            # Also persist to database for long-term memory
            if self.db_available and self.db_persistence:
                try:
                    # Create a summary of the interaction for storage
                    raw_input = content.get('raw_input', '')
                    topic = content.get('topic', '')
                    memory_content = f"[{topic}] {raw_input}" if topic else raw_input
                    
                    if memory_content:
                        self.db_persistence.store_memory(
                            user_id=user_id,
                            memory_type='episodic',
                            content=memory_content,
                            context={'topic': topic, 'emotion': emotion_data},
                            emotional_valence=emotion_data.get('valence', 0.0) if isinstance(emotion_data, dict) else 0.0,
                            emotional_arousal=emotion_data.get('arousal', 0.0) if isinstance(emotion_data, dict) else 0.0,
                            importance=min(1.0, emotional_intensity + 0.3)  # Boost importance for emotional content
                        )
                        print(f"[MEMORY-DB] 💾 Stored episodic memory for user {user_id[:8]}...")
                except Exception as e:
                    print(f"[MEMORY-DB] ❌ Failed to persist episodic memory: {e}")
            
        # Store emotional associations
        topic = content.get('topic', '')
        if topic and emotion_data:
            self.emotional_memory.store_emotional_context(topic, emotion_data)
            
    def get_memory_context(self) -> Dict[str, Any]:
        """Get current memory context for reasoning"""
        return {
            'working_context': self.working_memory.get_context(),
            'episodic_count': len(self.episodic_memory.episodes),
            'emotional_associations': len(self.emotional_memory.emotional_associations)
        }
    
    def store_action_memory(self, user_id: str, action_data: Dict[str, Any]):
        """Store a significant action as an episodic memory"""
        action_type = action_data.get('action_type', 'action')
        params = action_data.get('params', {})
        result = action_data.get('result', {})
        success = action_data.get('success', False)
        
        action_desc = f"{action_type}: {params.get('query', params.get('path', params.get('app', 'unknown')))}"
        
        episode = {
            'raw_input': f"Action executed: {action_desc}",
            'topic': 'action',
            'action_type': action_type,
            'action_params': params,
            'success': success,
            'emotion': {'intensity': 0.5, 'valence': 0.6 if success else -0.3}
        }
        
        self.episodic_memory.store_episode(user_id, episode)
        
        if self.db_available and self.db_persistence:
            try:
                result_text = result.get('display_text', str(result)[:200]) if isinstance(result, dict) else str(result)[:200]
                memory_content = f"[Action:{action_type}] {action_desc} -> {result_text}"
                
                self.db_persistence.store_memory(
                    user_id=user_id,
                    memory_type='action',
                    content=memory_content,
                    context={'action_type': action_type, 'params': params, 'success': success},
                    emotional_valence=0.3 if success else -0.2,
                    emotional_arousal=0.4,
                    importance=0.6
                )
                print(f"[MEMORY-DB] 🦾 Stored action memory: {action_type}")
            except Exception as e:
                print(f"[MEMORY-DB] ❌ Failed to persist action memory: {e}")
    
    def on_experience(self, payload):
        """Handle experiences from ExperienceBus - automatically store action memories"""
        try:
            from experience_bus import ExperienceType
            
            exp_type = payload.experience_type
            
            if exp_type in [ExperienceType.ACTION_SUCCESS, ExperienceType.ACTION_FAILURE]:
                user_id = payload.user_id
                action_data = payload.action_data
                
                if user_id and action_data:
                    self.store_action_memory(user_id, action_data)
                    print(f"[MEMORY] 🧠 Action memory captured from ExperienceBus: {action_data.get('action_type')}")
        except Exception as e:
            print(f"[MEMORY] ⚠️ Error processing experience: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to ExperienceBus for unified learning"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("IntelligentMemorySystem", self.on_experience)
            print("🧠 IntelligentMemorySystem subscribed to ExperienceBus - action memories will be captured")
        except Exception as e:
            print(f"⚠️ IntelligentMemorySystem bus subscription failed: {e}")