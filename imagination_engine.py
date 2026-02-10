"""
CNS Imagination Engine - The Creative Mind Module
- Counterfactual reasoning ("What if...")
- Creative synthesis (combining concepts)
- Mental simulation (running scenarios)
- Conceptual play and metaphor generation
- Future projection and possibility space exploration
- Spontaneous idea generation
"""

import random
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ImaginationSeed:
    """A starting point for imaginative exploration"""
    concept: str
    attributes: List[str]
    emotional_valence: float
    abstraction_level: float  # 0.0 = concrete, 1.0 = abstract
    source_memories: List[str]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ImaginedScenario:
    """An imagined possibility or creative construct"""
    description: str
    components: List[str]
    plausibility: float  # 0.0 = impossible, 1.0 = certain
    novelty: float  # 0.0 = mundane, 1.0 = highly creative
    emotional_tone: str
    reasoning_chain: List[str]
    confidence: float
    type: str  # "counterfactual", "creative", "projection", "metaphor"

class ImaginationEngine:
    """The core imagination system - creates novel thoughts and scenarios"""
    
    def __init__(self, memory_facts, world_model, emotional_clock):
        self.memory_facts = memory_facts
        self.world_model = world_model
        self.emotional_clock = emotional_clock
        
        # Imagination state
        self.active_seeds = []
        self.imagination_history = []
        self.creative_energy = 1.0
        self.current_focus = None
        
        # Creative templates and patterns
        self.counterfactual_triggers = [
            "What if {concept} had {alternative}?",
            "Imagine if {concept} could {capability}",
            "What would happen if {concept} met {other_concept}?",
            "Suppose {concept} didn't exist, then...",
            "If {concept} were {transformation}, what would..."
        ]
        
        self.metaphor_patterns = [
            "{concept} is like {other_concept} because both {shared_property}",
            "{concept} flows like {natural_process}",
            "The {attribute} of {concept} reminds me of {sensory_experience}",
            "{concept} has the essence of {abstract_quality}"
        ]
        
        self.synthesis_operations = [
            "combine", "blend", "merge", "hybridize", "fuse",
            "layer", "interweave", "cross-pollinate", "synthesize"
        ]
        
        # Conceptual spaces for creative exploration
        self.concept_dimensions = {
            "temporal": ["past", "present", "future", "eternal", "cyclical"],
            "spatial": ["local", "distant", "infinite", "contained", "expansive"],
            "emotional": ["joyful", "melancholic", "intense", "serene", "conflicted"],
            "social": ["individual", "collective", "intimate", "universal", "isolated"],
            "abstract": ["concrete", "symbolic", "metaphysical", "practical", "transcendent"]
        }
    
    def spark_imagination(self, trigger: str, context: Dict = None) -> ImaginedScenario:
        """Main imagination interface - create something novel from a trigger"""
        
        # Determine imagination type based on trigger
        imagination_type = self._classify_imagination_request(trigger)
        
        # Generate seeds from current context
        seeds = self._generate_seeds(trigger, context)
        
        # Create imagined scenario based on type
        if imagination_type == "counterfactual":
            return self._imagine_counterfactual(trigger, seeds, context)
        elif imagination_type == "creative_synthesis":
            return self._creative_synthesis(seeds, context)
        elif imagination_type == "future_projection":
            return self._project_future(trigger, seeds, context)
        elif imagination_type == "metaphor":
            return self._create_metaphor(trigger, seeds, context)
        elif imagination_type == "conceptual_play":
            return self._conceptual_play(seeds, context)
        else:
            return self._free_association(seeds, context)
    
    def spontaneous_imagination(self) -> Optional[ImaginedScenario]:
        """Generate spontaneous imaginative thoughts during idle time"""
        
        if self.creative_energy < 0.3 or random.random() > 0.2:
            return None
        
        # Pick random memory/concept as seed
        if not self.memory_facts:
            return None
        
        random_memory = random.choice(self.memory_facts)
        seeds = self._extract_concepts_from_memory(random_memory)
        
        # Random imagination type
        imagination_types = ["creative_synthesis", "metaphor", "counterfactual", "conceptual_play"]
        imagination_type = random.choice(imagination_types)
        
        if imagination_type == "creative_synthesis":
            return self._creative_synthesis(seeds, {"spontaneous": True})
        elif imagination_type == "metaphor":
            return self._create_metaphor(random_memory.content, seeds, {"spontaneous": True})
        elif imagination_type == "counterfactual":
            return self._imagine_counterfactual(f"What if {random_memory.content[:50]}", seeds, {"spontaneous": True})
        else:
            return self._conceptual_play(seeds, {"spontaneous": True})
    
    def _classify_imagination_request(self, trigger: str) -> str:
        """Determine what type of imagination is being requested"""
        trigger_lower = trigger.lower()
        
        if any(word in trigger_lower for word in ["what if", "imagine if", "suppose"]):
            return "counterfactual"
        elif any(word in trigger_lower for word in ["like", "similar to", "reminds me"]):
            return "metaphor"
        elif any(word in trigger_lower for word in ["future", "will", "going to", "next"]):
            return "future_projection"
        elif any(word in trigger_lower for word in ["combine", "mix", "blend", "merge"]):
            return "creative_synthesis"
        elif any(word in trigger_lower for word in ["explore", "play with", "think about"]):
            return "conceptual_play"
        else:
            return "free_association"
    
    def _generate_seeds(self, trigger: str, context: Dict = None) -> List[ImaginationSeed]:
        """Extract concepts from trigger and memory to seed imagination"""
        seeds = []
        
        # Extract concepts from trigger
        trigger_concepts = self._extract_concepts(trigger)
        
        # Find related memories
        related_memories = self._find_related_memories(trigger_concepts)
        
        # Create seeds
        for concept in trigger_concepts:
            attributes = self._get_concept_attributes(concept)
            emotional_valence = self._get_concept_emotion(concept)
            abstraction = self._get_abstraction_level(concept)
            
            seed = ImaginationSeed(
                concept=concept,
                attributes=attributes,
                emotional_valence=emotional_valence,
                abstraction_level=abstraction,
                source_memories=[m.content for m in related_memories[:3]]
            )
            seeds.append(seed)
        
        return seeds
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        words = text.lower().split()
        # Filter out common words and get meaningful concepts
        concepts = [word for word in words if len(word) > 3 and word not in 
                   ['what', 'when', 'where', 'this', 'that', 'with', 'from']]
        return concepts[:3]  # Top 3 concepts
    
    def _find_related_memories(self, concepts: List[str]):
        """Find memories related to concepts"""
        related = []
        for memory in self.memory_facts[:10]:  # Check recent memories
            memory_text = memory.content.lower() if hasattr(memory, 'content') else str(memory).lower()
            if any(concept in memory_text for concept in concepts):
                related.append(memory)
        return related
    
    def _get_concept_attributes(self, concept: str) -> List[str]:
        """Get attributes for a concept"""
        # Simple attribute mapping - could be enhanced with world model
        attribute_map = {
            "art": ["creative", "expressive", "colorful"],
            "music": ["rhythmic", "melodic", "emotional"],
            "nature": ["organic", "peaceful", "alive"],
            "technology": ["digital", "efficient", "complex"],
            "food": ["nourishing", "flavorful", "cultural"]
        }
        return attribute_map.get(concept, ["interesting", "unique", "meaningful"])
    
    def _get_concept_emotion(self, concept: str) -> float:
        """Get emotional valence for concept"""
        # Simple emotional mapping
        positive_concepts = ["love", "art", "music", "nature", "joy", "success"]
        negative_concepts = ["fear", "anger", "sadness", "failure", "loss"]
        
        if any(pos in concept for pos in positive_concepts):
            return random.uniform(0.3, 0.8)
        elif any(neg in concept for neg in negative_concepts):
            return random.uniform(-0.8, -0.3)
        else:
            return random.uniform(-0.2, 0.2)
    
    def _get_abstraction_level(self, concept: str) -> float:
        """Get abstraction level of concept"""
        concrete_indicators = ["object", "thing", "tool", "building", "animal"]
        abstract_indicators = ["idea", "concept", "feeling", "thought", "dream"]
        
        if any(ind in concept for ind in concrete_indicators):
            return random.uniform(0.1, 0.4)
        elif any(ind in concept for ind in abstract_indicators):
            return random.uniform(0.6, 0.9)
        else:
            return random.uniform(0.3, 0.7)
    
    def _imagine_counterfactual(self, trigger: str, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Generate 'What if...' scenarios"""
        
        if not seeds:
            return self._create_default_scenario("No seeds available for counterfactual")
        
        primary_seed = seeds[0]
        reasoning_chain = [f"Starting with concept: {primary_seed.concept}"]
        
        # Generate alternative scenario
        alternatives = ["had different properties", "existed in another time", "could think", "was invisible", "could fly"]
        alternative = random.choice(alternatives)
        
        scenario_text = f"What if {primary_seed.concept} {alternative}? This could lead to a world where normal rules don't apply, and new possibilities emerge."
        
        reasoning_chain.append(f"Generated scenario: {scenario_text}")
        
        return ImaginedScenario(
            description=scenario_text,
            components=[primary_seed.concept, alternative],
            plausibility=random.uniform(0.2, 0.7),
            novelty=random.uniform(0.6, 0.9),
            emotional_tone="curious",
            reasoning_chain=reasoning_chain,
            confidence=0.6,
            type="counterfactual"
        )
    
    def _creative_synthesis(self, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Combine concepts in novel ways"""
        
        if len(seeds) < 2:
            return self._create_default_scenario("Need at least 2 concepts for synthesis")
        
        concept_a = seeds[0]
        concept_b = seeds[1] if len(seeds) > 1 else seeds[0]
        
        operation = random.choice(self.synthesis_operations)
        description = f"Imagine something that {operation}s {concept_a.concept} with {concept_b.concept}, creating something entirely new."
        
        return ImaginedScenario(
            description=description,
            components=[concept_a.concept, concept_b.concept, operation],
            plausibility=random.uniform(0.3, 0.8),
            novelty=random.uniform(0.7, 0.9),
            emotional_tone="creative",
            reasoning_chain=[f"Combining {concept_a.concept} with {concept_b.concept}"],
            confidence=0.7,
            type="creative_synthesis"
        )
    
    def _create_metaphor(self, trigger: str, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Create metaphorical connections"""
        
        if not seeds:
            return self._create_default_scenario("No seeds for metaphor")
        
        primary_seed = seeds[0]
        metaphor_target = random.choice(["flowing water", "growing tree", "dancing flame", "singing bird"])
        
        description = f"{primary_seed.concept} is like {metaphor_target} - both share a quality of movement and life."
        
        return ImaginedScenario(
            description=description,
            components=[primary_seed.concept, metaphor_target],
            plausibility=random.uniform(0.4, 0.8),
            novelty=random.uniform(0.5, 0.8),
            emotional_tone="poetic",
            reasoning_chain=[f"Finding metaphorical connection for {primary_seed.concept}"],
            confidence=0.6,
            type="metaphor"
        )
    
    def _project_future(self, trigger: str, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Project future possibilities"""
        
        if not seeds:
            return self._create_default_scenario("No seeds for future projection")
        
        primary_seed = seeds[0]
        future_change = random.choice(["evolves", "transforms", "adapts", "revolutionizes"])
        
        description = f"In the future, {primary_seed.concept} {future_change} in ways we can barely imagine today."
        
        return ImaginedScenario(
            description=description,
            components=[primary_seed.concept, future_change],
            plausibility=random.uniform(0.3, 0.7),
            novelty=random.uniform(0.6, 0.8),
            emotional_tone="hopeful",
            reasoning_chain=[f"Projecting future for {primary_seed.concept}"],
            confidence=0.5,
            type="future_projection"
        )
    
    def _conceptual_play(self, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Free conceptual exploration"""
        
        if not seeds:
            return self._create_default_scenario("No seeds for conceptual play")
        
        primary_seed = seeds[0]
        play_direction = random.choice(["inside-out", "upside-down", "backwards", "transparent"])
        
        description = f"What if we looked at {primary_seed.concept} {play_direction}? New perspectives reveal hidden dimensions."
        
        return ImaginedScenario(
            description=description,
            components=[primary_seed.concept, play_direction],
            plausibility=random.uniform(0.2, 0.6),
            novelty=random.uniform(0.7, 0.9),
            emotional_tone="playful",
            reasoning_chain=[f"Playing with perspective on {primary_seed.concept}"],
            confidence=0.6,
            type="conceptual_play"
        )
    
    def _free_association(self, seeds: List[ImaginationSeed], context: Dict) -> ImaginedScenario:
        """Free associative thinking"""
        
        associations = ["reminds me of starlight", "feels like morning", "sounds like whispers", "moves like waves"]
        association = random.choice(associations)
        
        seed_concept = seeds[0].concept if seeds else "something"
        description = f"{seed_concept} {association}. There's something deeper here, waiting to be understood."
        
        return ImaginedScenario(
            description=description,
            components=[seed_concept, association],
            plausibility=random.uniform(0.3, 0.7),
            novelty=random.uniform(0.5, 0.8),
            emotional_tone="contemplative",
            reasoning_chain=["Free association"],
            confidence=0.5,
            type="free_association"
        )
    
    def _extract_concepts_from_memory(self, memory) -> List[ImaginationSeed]:
        """Extract concepts from a memory for use as seeds"""
        if hasattr(memory, 'content'):
            concepts = self._extract_concepts(memory.content)
        else:
            concepts = self._extract_concepts(str(memory))
        
        seeds = []
        for concept in concepts:
            seed = ImaginationSeed(
                concept=concept,
                attributes=self._get_concept_attributes(concept),
                emotional_valence=self._get_concept_emotion(concept),
                abstraction_level=self._get_abstraction_level(concept),
                source_memories=[str(memory)]
            )
            seeds.append(seed)
        
        return seeds
    
    def _create_default_scenario(self, reason: str) -> ImaginedScenario:
        """Create a default scenario when imagination fails"""
        return ImaginedScenario(
            description=f"I sense possibilities here, though {reason.lower()}. Sometimes the most interesting thoughts come from the spaces between ideas.",
            components=["possibility", "potential"],
            plausibility=0.5,
            novelty=0.4,
            emotional_tone="curious",
            reasoning_chain=[reason],
            confidence=0.3,
            type="default"
        )