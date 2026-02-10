"""
REM & Subconscious Processing Engine - The Dreaming Mind
- REM-like memory consolidation and pattern extraction
- Subconscious background processing
- Dream-like symbolic processing
- Unconscious association and integration
- Stream of consciousness generation
- Background emotional processing
"""

import time
import random
import math
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class SubconsciousProcess:
    """A background mental process"""
    process_id: str
    content: str
    process_type: str  # "consolidation", "association", "symbolic", "emotional"
    priority: float
    energy_cost: float
    start_time: float
    expected_duration: float
    current_state: str = "pending"
    results: Dict = field(default_factory=dict)
    associations_made: List[str] = field(default_factory=list)

@dataclass
class DreamFragment:
    """A piece of dream-like processing"""
    narrative: str
    symbols: List[str]
    emotions: Dict[str, float]
    source_memories: List[str]
    transformations: List[str]
    coherence_level: float  # How logical vs surreal
    significance: float     # How meaningful this fragment is
    timestamp: float = field(default_factory=time.time)

@dataclass
class StreamOfConsciousnessNode:
    """A node in the stream of consciousness"""
    thought: str
    thought_type: str  # "observation", "memory", "feeling", "wonder", "connection"
    emotional_tone: float
    spontaneity: float  # How unexpected this thought was
    connections: List[str]  # What this connects to
    timestamp: float = field(default_factory=time.time)

class REMProcessor:
    """Handles REM-like sleep processing - memory consolidation and dreaming"""
    
    def __init__(self, memory_facts, emotional_clock, world_model):
        self.memory_facts = memory_facts
        self.emotional_clock = emotional_clock
        self.world_model = world_model
        
        # REM state tracking
        self.rem_active = False
        self.rem_intensity = 0.0
        self.last_rem_cycle = 0
        self.rem_duration = 0
        
        # Memory processing queues
        self.consolidation_queue = deque()
        self.pattern_extraction_queue = deque()
        self.emotional_integration_queue = deque()
        
        # Dream generation
        self.dream_fragments = deque(maxlen=50)
        self.symbolic_mappings = {}
        self.recurring_themes = defaultdict(int)
        
        # Processing statistics
        self.memories_consolidated = 0
        self.patterns_discovered = 0
        self.emotional_resolutions = 0
        
        # Pattern storage
        self.discovered_patterns = []
        self.emotional_associations = defaultdict(list)
    
    def enter_rem_cycle(self, intensity: float = 0.8, duration: float = 60):
        """Enter REM-like processing state"""
        self.rem_active = True
        self.rem_intensity = intensity
        self.rem_duration = duration
        self.last_rem_cycle = time.time()
        
        print(f"🌙 Entering REM cycle (intensity: {intensity:.2f})")
        
        # Queue up memories for processing
        self._queue_memories_for_processing()
        
        # Start background processing
        self._process_rem_cycle()
    
    def _queue_memories_for_processing(self):
        """Queue recent memories for REM processing"""
        # Get recent memories
        recent_threshold = time.time() - (24 * 3600)  # 24 hours ago
        
        recent_memories = []
        for memory in self.memory_facts:
            if hasattr(memory, 'timestamp') and memory.timestamp > recent_threshold:
                recent_memories.append(memory)
            elif not hasattr(memory, 'timestamp'):  # Handle memories without timestamp
                recent_memories.append(memory)
        
        # Prioritize emotionally significant memories
        emotional_memories = []
        for memory in recent_memories:
            valence = getattr(memory, 'valence', 0)
            if abs(valence) > 0.3:
                emotional_memories.append(memory)
        
        # Queue for processing
        for memory in emotional_memories[:5]:  # Process top 5
            self.consolidation_queue.append(memory)
            self.emotional_integration_queue.append(memory)
        
        # Queue for pattern extraction
        for memory in recent_memories[:10]:
            self.pattern_extraction_queue.append(memory)
    
    def _process_rem_cycle(self):
        """Main REM processing loop"""
        cycle_start = time.time()
        cycles_completed = 0
        
        while self.rem_active and (time.time() - cycle_start) < self.rem_duration and cycles_completed < 10:
            # Memory consolidation
            if self.consolidation_queue:
                self._consolidate_memory()
            
            # Pattern extraction
            if self.pattern_extraction_queue:
                self._extract_patterns()
            
            # Emotional integration
            if self.emotional_integration_queue:
                self._integrate_emotions()
            
            # Dream generation
            if random.random() < 0.4:  # 40% chance each cycle
                self._generate_dream_fragment()
            
            cycles_completed += 1
            time.sleep(0.1)  # Brief pause
        
        self.rem_active = False
        print(f"🌅 REM cycle complete. Processed: {self.memories_consolidated} memories, "
              f"found {self.patterns_discovered} patterns, resolved {self.emotional_resolutions} emotions")
    
    def _consolidate_memory(self):
        """Strengthen and integrate a memory"""
        if not self.consolidation_queue:
            return
        
        memory = self.consolidation_queue.popleft()
        
        # Strengthen the memory
        if hasattr(memory, 'confidence'):
            memory.confidence = min(1.0, memory.confidence * 1.1)
        
        # Find related memories for integration
        related_memories = self._find_related_memories(memory)
        
        # Create new associations
        if hasattr(memory, 'associations'):
            for related in related_memories[:2]:
                memory.associations[related.content[:30]] = 0.7
        
        self.memories_consolidated += 1
    
    def _extract_patterns(self):
        """Extract hidden patterns from memories"""
        if not self.pattern_extraction_queue:
            return
        
        memory = self.pattern_extraction_queue.popleft()
        
        # Get memory content
        if hasattr(memory, 'content'):
            content = memory.content
        else:
            content = str(memory)
        
        # Look for recurring themes
        content_words = content.lower().split()
        for word in content_words:
            if len(word) > 4:  # Skip small words
                self.recurring_themes[word] += 1
        
        # Find similar memories for pattern detection
        similar_memories = self._find_similar_memories(memory)
        if len(similar_memories) >= 1:
            pattern = f"Pattern: {content[:30]}... relates to similar experiences"
            self.patterns_discovered += 1
            
            # Store the discovered pattern
            self.discovered_patterns.append({
                'pattern': pattern,
                'evidence': [memory] + similar_memories[:1],
                'confidence': 0.6,
                'discovery_time': time.time()
            })
    
    def _integrate_emotions(self):
        """Process and integrate emotional memories"""
        if not self.emotional_integration_queue:
            return
        
        memory = self.emotional_integration_queue.popleft()
        
        # Get emotional context
        memory_valence = getattr(memory, 'valence', 0)
        current_mood = self.emotional_clock.current_valence
        
        # Process emotional conflict
        if abs(memory_valence - current_mood) > 0.3:
            # There's emotional tension - process it
            resolution = self._resolve_emotional_tension(memory, memory_valence, current_mood)
            
            # Update emotional state slightly
            self.emotional_clock.current_valence += resolution * 0.05
            
            self.emotional_resolutions += 1
        
        # Create emotional associations
        emotion_label = "positive" if memory_valence > 0.2 else "negative" if memory_valence < -0.2 else "neutral"
        
        content = getattr(memory, 'content', str(memory))
        self.emotional_associations[emotion_label].append(content[:50])
    
    def _resolve_emotional_tension(self, memory, memory_valence, current_mood):
        """Resolve emotional tension through processing"""
        tension = memory_valence - current_mood
        
        # Different resolution strategies
        if tension > 0:  # Memory is more positive than current mood
            resolution = 0.2  # Slight mood lift
        else:  # Memory is more negative than current mood
            resolution = -0.1  # Slight mood dampening
        
        return resolution
    
    def _generate_dream_fragment(self):
        """Generate a dream-like narrative fragment"""
        if not self.memory_facts:
            return
        
        # Select random memories as dream seeds
        available_memories = self.memory_facts[:10] if len(self.memory_facts) > 10 else self.memory_facts
        dream_memories = random.sample(available_memories, min(2, len(available_memories)))
        
        # Extract symbols and themes
        symbols = []
        themes = []
        for memory in dream_memories:
            content = getattr(memory, 'content', str(memory))
            words = content.lower().split()
            symbols.extend([w for w in words if len(w) > 4][:2])
            themes.append(self._extract_theme(content))
        
        # Create surreal transformations
        transformations = [
            f"{symbols[0] if symbols else 'something'} becomes {symbols[1] if len(symbols) > 1 else 'everything'}",
            f"time flows {random.choice(['backwards', 'in circles', 'like honey', 'in fragments'])}",
            f"{themes[0] if themes else 'the familiar'} transforms into {themes[1] if len(themes) > 1 else 'the unknown'}"
        ]
        
        # Generate dream narrative
        narrative_templates = [
            "I am walking through a landscape where {transformation1}. In the distance, {symbol1} calls to {symbol2}, and suddenly {transformation2}. The feeling is {emotion}.",
            "There is a place where {symbol1} and {symbol2} dance together. {transformation1}, and I realize that {transformation2}. Everything feels {emotion}.",
            "I find myself in a world where {transformation1}. {symbol1} whispers secrets to {symbol2}, while {transformation2}. The atmosphere is {emotion}."
        ]
        
        template = random.choice(narrative_templates)
        emotion = random.choice(["nostalgic", "mysterious", "profound", "unsettling", "beautiful", "melancholic"])
        
        narrative = template.format(
            transformation1=transformations[0] if transformations else "reality shifts",
            transformation2=transformations[1] if len(transformations) > 1 else "meaning changes",
            symbol1=symbols[0] if symbols else "presence",
            symbol2=symbols[1] if len(symbols) > 1 else "echo",
            emotion=emotion
        )
        
        # Create dream fragment
        fragment = DreamFragment(
            narrative=narrative,
            symbols=symbols[:5],
            emotions={emotion: random.uniform(0.4, 0.9)},
            source_memories=[getattr(m, 'content', str(m))[:50] for m in dream_memories],
            transformations=transformations,
            coherence_level=random.uniform(0.2, 0.7),  # Dreams are often incoherent
            significance=random.uniform(0.3, 0.8)
        )
        
        self.dream_fragments.append(fragment)
    
    def get_recent_dreams(self, count: int = 3) -> List[DreamFragment]:
        """Get recent dream fragments"""
        return list(self.dream_fragments)[-count:]
    
    def get_discovered_patterns(self, count: int = 5) -> List[Dict]:
        """Get recently discovered patterns"""
        return self.discovered_patterns[-count:]
    
    def _find_related_memories(self, target_memory):
        """Find memories related to target"""
        related = []
        target_content = getattr(target_memory, 'content', str(target_memory))
        target_words = set(target_content.lower().split())
        
        for memory in self.memory_facts:
            if memory != target_memory:
                memory_content = getattr(memory, 'content', str(memory))
                memory_words = set(memory_content.lower().split())
                overlap = len(target_words.intersection(memory_words))
                if overlap >= 1:  # At least 1 word in common
                    related.append(memory)
        
        return related[:3]
    
    def _find_similar_memories(self, target_memory):
        """Find memories similar to target"""
        return self._find_related_memories(target_memory)
    
    def _extract_theme(self, content: str) -> str:
        """Extract a theme from content"""
        themes = ["connection", "growth", "change", "discovery", "emotion", "thought", "experience"]
        words = content.lower().split()
        
        # Simple theme extraction based on keywords
        for theme in themes:
            if any(word in theme or theme in word for word in words):
                return theme
        
        return random.choice(themes)

class SubconsciousEngine:
    """Manages background subconscious processing"""
    
    def __init__(self, memory_facts, emotional_clock, world_model):
        self.memory_facts = memory_facts
        self.emotional_clock = emotional_clock
        self.world_model = world_model
        
        # REM processor
        self.rem_processor = REMProcessor(memory_facts, emotional_clock, world_model)
        
        # Background processing
        self.background_processes = deque()
        self.processing_active = False
        
        # Stream of consciousness
        self.consciousness_stream = deque(maxlen=100)
        self.last_stream_update = time.time()
        
        # Subconscious insights
        self.insights = deque(maxlen=20)
        self.association_network = defaultdict(list)
    
    def start_background_processing(self):
        """Start continuous background processing"""
        self.processing_active = True
        # In a real implementation, this would run in a separate thread
        # For now, we'll process when called
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.processing_active = False
    
    def trigger_rem_cycle(self, intensity: float = 0.8):
        """Trigger a REM processing cycle"""
        self.rem_processor.enter_rem_cycle(intensity=intensity, duration=30)
    
    def generate_stream_of_consciousness(self) -> Optional[StreamOfConsciousnessNode]:
        """Generate spontaneous thoughts"""
        if time.time() - self.last_stream_update < 10:  # Throttle to every 10 seconds
            return None
        
        if not self.memory_facts or random.random() > 0.3:
            return None
        
        # Pick a random memory to trigger thought
        memory = random.choice(self.memory_facts)
        memory_content = getattr(memory, 'content', str(memory))
        
        # Generate different types of thoughts
        thought_types = ["memory", "wonder", "connection", "feeling", "observation"]
        thought_type = random.choice(thought_types)
        
        if thought_type == "memory":
            thought = f"I remember {memory_content[:40]}..."
        elif thought_type == "wonder":
            thought = f"I wonder about {memory_content[:30]}... what if it were different?"
        elif thought_type == "connection":
            thought = f"This reminds me of something... the way {memory_content[:30]} connects to other experiences"
        elif thought_type == "feeling":
            emotion = random.choice(["nostalgic", "curious", "contemplative", "hopeful"])
            thought = f"I feel {emotion} when I think about {memory_content[:30]}"
        else:  # observation
            thought = f"There's something interesting about {memory_content[:30]}... patterns within patterns"
        
        node = StreamOfConsciousnessNode(
            thought=thought,
            thought_type=thought_type,
            emotional_tone=random.uniform(-0.3, 0.3),
            spontaneity=random.uniform(0.4, 0.9),
            connections=[memory_content[:20]]
        )
        
        self.consciousness_stream.append(node)
        self.last_stream_update = time.time()
        
        return node
    
    def get_recent_consciousness_stream(self, count: int = 5) -> List[StreamOfConsciousnessNode]:
        """Get recent stream of consciousness"""
        return list(self.consciousness_stream)[-count:]
    
    def get_subconscious_insights(self) -> Dict[str, Any]:
        """Get insights from subconscious processing"""
        return {
            "recent_dreams": self.rem_processor.get_recent_dreams(3),
            "discovered_patterns": self.rem_processor.get_discovered_patterns(3),
            "recurring_themes": dict(list(self.rem_processor.recurring_themes.items())[:5]),
            "emotional_associations": dict(self.rem_processor.emotional_associations),
            "consciousness_stream": self.get_recent_consciousness_stream(3),
            "rem_stats": {
                "memories_consolidated": self.rem_processor.memories_consolidated,
                "patterns_discovered": self.rem_processor.patterns_discovered,
                "emotional_resolutions": self.rem_processor.emotional_resolutions
            }
        }
    
    def process_background_thoughts(self):
        """Process background thoughts and associations"""
        if not self.processing_active:
            return
        
        # Generate occasional insights
        if random.random() < 0.1:  # 10% chance
            insight = self._generate_insight()
            if insight:
                self.insights.append(insight)
        
        # Update association network
        self._update_associations()
        
        # Generate stream of consciousness
        self.generate_stream_of_consciousness()
    
    def _generate_insight(self) -> Optional[str]:
        """Generate a subconscious insight"""
        if not self.memory_facts:
            return None
        
        insights = [
            "There are patterns in the way memories connect...",
            "Sometimes the most important things are the ones we don't notice immediately",
            "Emotions color every thought, even the ones that seem purely logical",
            "The spaces between thoughts might be as important as the thoughts themselves",
            "Every experience changes the lens through which we see new experiences"
        ]
        
        return random.choice(insights)
    
    def _update_associations(self):
        """Update the association network"""
        if len(self.memory_facts) < 2:
            return
        
        # Create associations between recent memories
        recent_memories = self.memory_facts[-5:] if len(self.memory_facts) >= 5 else self.memory_facts
        
        for i, memory1 in enumerate(recent_memories):
            for memory2 in recent_memories[i+1:]:
                content1 = getattr(memory1, 'content', str(memory1))
                content2 = getattr(memory2, 'content', str(memory2))
                
                # Simple association based on word overlap
                words1 = set(content1.lower().split())
                words2 = set(content2.lower().split())
                
                if len(words1.intersection(words2)) > 0:
                    self.association_network[content1[:30]].append(content2[:30])