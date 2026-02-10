# Advanced Curiosity + Dopamine System for CNS
# Integrated conversation gap detection and dopamine-driven curiosity arcs
# Works with CNS memory and emotion processing systems

import time, math, random, uuid, re
from typing import List, Dict, Any, Optional

# --------- Advanced Gap Detector ---------
class AdvancedGapDetector:
    """
    Produces gap hypotheses from user input using heuristics:
    - novelty: new entities not in short-term memory
    - story: incomplete events (trailing ellipsis, narrative verbs without result)
    - emotion: emotion words without explanatory clause
    - micro-gap: vague adjectives/phrases, hedges, "it was weird", "kinda"
    - contradiction: conflicts with recent memory
    Each hypothesis is a dict: {gap_type, target, salience, confidence, emotional_context}
    """
    vague_markers = ["weird","odd","kinda","sorta","weirdly","strange","vague","somehow"]
    hedges = ["maybe","probably","might","could","I guess","I suppose"]
    narrative_verbs = ["went","left","came","happened","did","said","told","met","saw","got","started","stopped"]
    
    # Stopwords to filter from novelty detection
    STOPWORDS = {
        # Common filler words
        "like", "just", "so", "really", "very", "kinda", "sorta", "maybe", "probably",
        "too", "also", "still", "even", "always", "never", "only", "much", "more", "most",
        # Conversation starters & fillers (CRITICAL FIX)
        "hey", "hi", "hello", "well", "okay", "ok", "yeah", "yep", "nope", "hmm",
        "um", "uh", "oh", "ah", "guess", "say", "telling", "big", "little", "small",
        # Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their", "mine", "yours",
        # Articles & conjunctions
        "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
        # Prepositions
        "in", "on", "at", "to", "from", "with", "about", "as", "by", "of",
        # Common verbs (expanded)
        "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could", "can", "may", "might",
        "get", "got", "getting", "go", "going", "went", "think", "know",
        "wait", "waited", "waiting", "try", "tried", "trying", "want", "wanted", "wanting",
        "make", "made", "making", "take", "took", "taking", "give", "gave", "giving",
        "feel", "felt", "feeling", "see", "saw", "seeing", "look", "looked", "looking",
        "come", "came", "coming", "mean", "meant", "meaning", "keep", "kept", "keeping",
        # Question words
        "what", "when", "where", "who", "why", "how", "which",
        # Other common words
        "this", "that", "these", "those", "there", "here", "now", "then",
        "one", "two", "first", "last", "next", "some", "any", "all", "each", "every",
        "thing", "things", "something", "anything", "everything", "nothing",
        # Meta-words - words ABOUT entities, not actual entities
        "name", "person", "people", "place", "situation", "moment", "time", "day",
        "called", "named", "someone", "anyone", "everyone", "nobody"
    }
    
    # Prepositions that signal important context
    CONTEXT_PREPOSITIONS = {"at", "to", "with", "from", "in", "on", "during", "after", "before"}
    
    # HIGH-PRIORITY NOUNS: Words that represent story significance
    # These should be extracted FIRST as they're meaningful entities
    PRIORITY_NOUNS = {
        # People & relationships
        "partner", "friend", "boyfriend", "girlfriend", "husband", "wife", "spouse",
        "mom", "dad", "mother", "father", "brother", "sister", "family",
        "boss", "coworker", "colleague", "team", "manager", "client",
        "stranger", "neighbor", "roommate", "therapist", "doctor",
        # Events & actions (past tense verbs count as event nouns)
        "fight", "argument", "conversation", "meeting", "date", "party", "dinner",
        "interview", "presentation", "exam", "test", "appointment", "trip", "vacation",
        "breakup", "wedding", "funeral", "birthday", "promotion", "fired", "hired",
        "accident", "incident", "surgery", "diagnosis", "treatment",
        # Emotions & states
        "feeling", "feelings", "down", "upset", "happy", "sad", "angry", "anxious",
        "excited", "nervous", "stressed", "depressed", "lonely", "hurt", "betrayed",
        "confused", "lost", "hopeful", "relieved", "proud", "ashamed", "guilty",
        # Important objects/places
        "home", "house", "apartment", "office", "school", "hospital", "restaurant",
        "car", "phone", "email", "message", "letter", "photo", "gift",
        # Life domains
        "job", "work", "career", "school", "college", "relationship", "health",
        "money", "finance", "future", "past", "childhood", "dream", "goal"
    }
    
    def __init__(self, cns_brain):
        """Initialize with CNS brain for memory and emotion access"""
        self.cns_brain = cns_brain
    
    def detect(self, text: str, tone_label: Optional[str]=None) -> List[Dict[str,Any]]:
        gaps = []
        text_clean = text.strip()
        ent_candidates = self._extract_entities(text)
        recent = set(self._get_recent_entities())
        emotion_label, emotion_int = self._extract_emotion(text)
        tone = tone_label or emotion_label or "neutral"
        
        # 1. Novelty gaps for new entities
        for e in ent_candidates:
            if e not in recent:
                gaps.append(self._make_gap("novelty", target=e, salience=0.6, emotion_context=tone, confidence=0.8))
        
        # 2. Story gaps: incomplete narratives
        if self._is_incomplete_narrative(text):
            gaps.append(self._make_gap("story", target=self._shorten(text), salience=0.7, emotion_context=tone, confidence=0.9))
        
        # 3. Emotion gaps: emotion detected but lacks reason clause
        # Enhanced: Extract the TRIGGER not just the emotion
        has_exp = self._has_explanation(text)
        if emotion_label and not has_exp:
            emotional_trigger = self._extract_emotional_trigger(text, emotion_label)
            gaps.append(self._make_gap("emotion", target=emotional_trigger, salience=0.95*emotion_int, emotion_context=tone, confidence=0.95))
        
        # 4. Micro-gaps: vague adjectives, hedges, implicit content
        if self._has_vague_tokens(text):
            gaps.append(self._make_gap("micro", target=self._shorten(text), salience=0.5, emotion_context=tone, confidence=0.6))
        
        # 5. Contradiction gaps
        contradictions = self._find_contradictions(text)
        for c in contradictions:
            gaps.append(self._make_gap("contradiction", target=c, salience=0.9, emotion_context=tone, confidence=0.9))
        
        # 6. Implicit question detection (user hints but not explicit)
        if self._is_hinting(text):
            gaps.append(self._make_gap("hint", target=self._shorten(text), salience=0.65, emotion_context=tone, confidence=0.85))
        
        # sort by confidence*salience descending
        gaps.sort(key=lambda g: g["confidence"]*g["salience"], reverse=True)
        return gaps
    
    # ------ CNS integration methods ------
    def _get_recent_entities(self, window=20):
        """Extract recent entities from CNS semantic and episodic memory"""
        entities = []
        
        if hasattr(self.cns_brain, 'semantic_memory'):
            # Get recent facts from semantic memory
            facts = getattr(self.cns_brain.semantic_memory, 'facts', {})
            for fact_key in list(facts.keys())[-window:]:
                # Extract all meaningful words (lowercase, no stopwords)
                words = re.findall(r'\b[a-z]+\b', fact_key.lower())
                for word in words:
                    if word not in self.STOPWORDS and len(word) > 2:
                        entities.append(word)
        
        # Also check episodic memory for entities
        if hasattr(self.cns_brain, 'episodic_memory'):
            episodes = getattr(self.cns_brain.episodic_memory, 'memories', [])
            for episode in episodes[-window:]:
                content = episode.get('content', '') if isinstance(episode, dict) else str(episode)
                words = re.findall(r'\b[a-z]+\b', content.lower())
                for word in words:
                    if word not in self.STOPWORDS and len(word) > 2:
                        entities.append(word)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_emotion(self, text):
        """Use CNS emotion processor to extract emotion and intensity"""
        if hasattr(self.cns_brain, 'emotion_processor'):
            # Get emotion data from CNS
            emotion_data = self.cns_brain.emotion_processor.process_emotion(text)
            
            # Extract dominant emotion and intensity
            emotion_label = emotion_data.get('emotion', None)
            
            # Calculate intensity from valence and arousal
            valence = abs(emotion_data.get('valence', 0.0))
            arousal = emotion_data.get('arousal', 0.5)
            intensity = (valence + arousal) / 2.0
            
            return (emotion_label, intensity)
        
        # Fallback to simple keyword matching if CNS not available
        text_l = text.lower()
        if any(w in text_l for w in ["angry","furious","mad"]):
            return ("anger", 0.9)
        if any(w in text_l for w in ["sad","depressed","lonely","devastated","heartbroken"]):
            return ("sadness", 0.85)
        if any(w in text_l for w in ["terrified","scared","afraid","fear","anxious","worried","nervous"]):
            return ("fear", 0.9)
        if any(w in text_l for w in ["happy","joy","glad","excited"]):
            return ("joy", 0.8)
        if any(w in text_l for w in ["frustrat","annoyed"]):
            return ("frustration", 0.75)
        return (None, 0.0)
    
    def _extract_emotional_trigger(self, text: str, emotion_label: str) -> str:
        """
        Extract what CAUSED the emotion, not just the emotion itself.
        
        Examples:
        - "I felt terrified when she said that" → "what she said"
        - "I'm scared after the party" → "what happened at the party"
        - "I got angry when he did that" → "what he did"
        """
        text_lower = text.lower()
        
        # Pattern 1: "emotion when X said/did..."
        when_patterns = [
            (r"when\s+(he|she|they)\s+said", "what they said"),
            (r"when\s+(he|she|they)\s+did", "what they did"),
            (r"when\s+\w+\s+told", "what they told me"),
            (r"after\s+(he|she|they)\s+said", "what they said"),
            (r"after\s+(he|she|they)\s+did", "what they did")
        ]
        
        for pattern, description in when_patterns:
            if re.search(pattern, text_lower):
                return description
        
        # Pattern 2: "emotion after/at/during EVENT"
        # Extract the event/place
        event_patterns = [
            r"after\s+the\s+(\w+)",     # "after the party"
            r"at\s+the\s+(\w+)",        # "at the meeting"
            r"during\s+the\s+(\w+)",    # "during the conversation"
            r"from\s+the\s+(\w+)"       # "from the argument"
        ]
        
        for pattern in event_patterns:
            match = re.search(pattern, text_lower)
            if match:
                event = match.group(1)
                if event not in self.STOPWORDS:
                    return f"what happened at the {event}"
        
        # Pattern 3: "emotion about X" or "emotion over X"
        about_pattern = r"(?:about|over)\s+(.{5,30})"
        match = re.search(about_pattern, text_lower)
        if match:
            return match.group(1).strip()
        
        # Fallback: return the emotion itself
        return emotion_label
    
    def _find_contradictions(self, text):
        """Find contradictions with CNS memory"""
        contradictions = []
        
        if hasattr(self.cns_brain, 'semantic_memory'):
            facts = getattr(self.cns_brain.semantic_memory, 'facts', {})
            
            # Check for love/hate contradictions
            if "love" in text.lower():
                for fact in facts.keys():
                    if "hate" in fact.lower():
                        contradictions.append("love vs hate")
                        break
            
            if "hate" in text.lower():
                for fact in facts.keys():
                    if "love" in fact.lower():
                        contradictions.append("hate vs love")
                        break
        
        return contradictions
    
    def _add_statement_to_memory(self, text):
        """Add statement to CNS memory for future contradiction detection"""
        if hasattr(self.cns_brain, 'semantic_memory'):
            # Store as a fact in semantic memory
            self.cns_brain.semantic_memory.add_fact(
                f"user_statement_{time.time()}", 
                text, 
                confidence=0.8
            )
    
    # ------ heuristics helpers ------
    def _extract_entities(self, text):
        """
        Extract story-relevant entities from casual text (no capitalization required).
        Prioritizes:
        1. Name patterns: "name is X", "called X" → extract X
        2. Contextual nouns (after prepositions: "at the beach", "with maya")
        3. All non-stopword nouns
        4. Filters out common filler words and meta-words
        """
        entities = []
        text_lower = text.lower()
        
        # PRIORITY 1: Extract actual names from name patterns
        # "Her name is Maya" → extract "Maya"
        # "called Ryan" → extract "Ryan"
        # "named Sarah" → extract "Sarah"
        name_patterns = [
            r"name\s+is\s+(\w+)",          # "name is Maya"
            r"called\s+(\w+)",              # "called Ryan"
            r"named\s+(\w+)",               # "named Sarah"
            r"name's\s+(\w+)",              # "name's Alex"
            r"name\s+was\s+(\w+)"           # "name was Jordan"
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in self.STOPWORDS and len(match) > 2:
                    entities.append(match)
        
        # PRIORITY 2: Extract HIGH-PRIORITY semantic nouns
        # These are words that indicate story significance (people, events, emotions)
        words_all = re.findall(r'\b[a-z]+\b', text_lower)
        priority_entities = []
        for word in words_all:
            if word in self.PRIORITY_NOUNS and word not in entities:
                priority_entities.append(word)
        
        # Add priority entities FIRST
        entities.extend(priority_entities)
        
        # PRIORITY 3: Extract contextual patterns: preposition + (the)? + NOUN
        # Examples: "at the beach", "with maya", "to the party", "from college"
        for prep in self.CONTEXT_PREPOSITIONS:
            # Pattern: prep + optional "the/a/an" + word
            pattern = rf"\b{prep}\s+(?:the\s+|a\s+|an\s+)?(\w+)"
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in self.STOPWORDS and len(match) > 2 and match not in entities:
                    entities.append(match)
        
        # PRIORITY 4: Extract remaining meaningful words (longer = more likely to be significant)
        # Only add non-stopword, non-common-verb words that aren't already in entities
        for word in words_all:
            if (word not in self.STOPWORDS and 
                word not in self.narrative_verbs and
                len(word) >= 4 and  # Prefer longer words (more specific)
                word not in entities):
                entities.append(word)
        
        # Limit to top candidates (avoid overwhelming with too many entities)
        return entities[:8]
    
    def _is_incomplete_narrative(self, text):
        t = text.strip()
        if t.endswith(("...", "but", "so", "then", "-")): 
            return True
        tokens = re.findall(r"\b\w+\b", t.lower())
        if any(v in tokens for v in self.narrative_verbs) and len(tokens) < 10:
            return True
        return False
    
    def _has_explanation(self, text):
        """
        Check if text has a meaningful explanation, not just pronouns.
        "I'm scared because of that" → NOT a real explanation (pronoun-heavy)
        "I'm scared because of the accident" → Real explanation
        """
        # Check for explicit explanation words
        has_connector = any(word in text.lower() for word in ["because","since","so that","as a result","due to"])
        
        if not has_connector:
            return False
        
        # Even with connector, vague pronouns don't count as real explanation
        # "when she said that" is NOT a real explanation - we still need to know WHAT she said
        pronoun_heavy = any(phrase in text.lower() for phrase in ["said that", "did that", "told me that", "about that", "about it", "when that"])
        
        if pronoun_heavy:
            return False  # Treat as incomplete explanation
        
        return True
    
    def _has_vague_tokens(self, text):
        tl = text.lower()
        return any(v in tl for v in self.vague_markers) or any(h in tl for h in self.hedges)
    
    def _is_hinting(self, text):
        t = text.strip()
        if t.endswith("...") or len(t.split()) <= 3 and t.endswith("!"):
            return True
        if re.search(r"\byou know\b|\bright\b|\bsee\b", t.lower()):
            return True
        return False
    
    def _make_gap(self, gtype, target, salience, emotion_context, confidence):
        return {
            "gap_type": gtype,
            "target": target,
            "salience": salience,
            "emotional_context": emotion_context,
            "confidence": confidence,
            "created_at": time.time()
        }
    
    def _shorten(self, text, n=6):
        return " ".join(text.split()[:n])

# --------- Advanced Dopamine Arc & Manager ---------
class DopamineArc:
    """
    An arc that tracks drive, satisfaction, salience, anticipation spikes, and decay.
    """
    def __init__(self, gap):
        self.arc_id = str(uuid.uuid4())
        self.gap_type = gap["gap_type"]
        self.target = gap["target"]
        self.salience = float(gap.get("salience", 0.6))
        self.emotion = gap.get("emotional_context", "neutral")
        self.state = "OPEN"   # OPEN -> ACTIVE -> SUSTAIN -> CLOSING -> CLOSED
        self.drive = 1.0 * self.salience
        self.satisfaction = 0.0
        self.anticipation = 0.0   # reward-prediction signal: spikes when hints present
        self.hint_count = 0
        self.created_at = time.time()
        self.last_update = time.time()
        self.turns_active = 0
    
    def register_hint(self):
        self.hint_count += 1
        self.anticipation = min(1.0, self.anticipation + 0.2)
        self.drive = min(1.5, self.drive + 0.1 * (1 + self.anticipation))
    
    def update(self, clarity: float, tone_label: Optional[str]=None):
        """
        clarity: 0..1 how much user info reduces the gap this turn
        tone_label: if user is upset, suppress curiosity
        """
        self.turns_active += 1
        now = time.time()
        # emotion override: if user upset/sad -> reduce drive and increase salience (linger)
        if tone_label in ("sad","angry","upset","fear","grief"):
            self.drive *= 0.4
            self.salience = min(1.0, self.salience * 1.2)
            self.last_update = now
            return
        
        if self.state == "OPEN":
            self.state = "ACTIVE"
        
        boost = 1.0 + 0.5 * self.anticipation
        decay_factor = (1 - clarity * 0.35)
        self.drive = max(0.05, self.drive * decay_factor * (1/boost))
        self.satisfaction = min(1.0, self.satisfaction + clarity * 0.55 * (1 - 0.2*self.salience))
        
        if self.satisfaction > 0.6 and self.state in ("ACTIVE","SUSTAIN"):
            self.state = "CLOSING"
        if self.satisfaction >= 0.95:
            self.state = "CLOSED"
            self.drive = 0.0
        
        self.anticipation = max(0.0, self.anticipation - 0.15 * clarity)
        self.last_update = now
    
    def decay(self):
        elapsed = time.time() - self.last_update
        rate = 0.0006 * (2 - self.salience)
        self.drive *= math.exp(-rate * elapsed)
        if self.state in ("OPEN","ACTIVE","SUSTAIN"):
            self.satisfaction = max(0.0, self.satisfaction - 0.0001 * elapsed)
    
    def to_signal(self):
        return {
            "arc_id": self.arc_id,
            "gap_type": self.gap_type,
            "target": self.target,
            "drive": round(self.drive,3),
            "satisfaction": round(self.satisfaction,3),
            "state": self.state,
            "salience": round(self.salience,3),
            "anticipation": round(self.anticipation,3),
            "hint_count": self.hint_count
        }
    
    def to_dict(self):
        """Serialize for brain state persistence"""
        return {
            "arc_id": self.arc_id,
            "gap_type": self.gap_type,
            "target": self.target,
            "salience": self.salience,
            "emotion": self.emotion,
            "state": self.state,
            "drive": self.drive,
            "satisfaction": self.satisfaction,
            "anticipation": self.anticipation,
            "hint_count": self.hint_count,
            "created_at": self.created_at,
            "last_update": self.last_update,
            "turns_active": self.turns_active
        }
    
    @staticmethod
    def from_dict(data):
        """Deserialize from brain state"""
        gap = {
            "gap_type": data["gap_type"],
            "target": data["target"],
            "salience": data["salience"],
            "emotional_context": data.get("emotion", "neutral")
        }
        arc = DopamineArc(gap)
        arc.arc_id = data["arc_id"]
        arc.state = data.get("state", "OPEN")
        arc.drive = data.get("drive", arc.drive)
        arc.satisfaction = data.get("satisfaction", 0.0)
        arc.anticipation = data.get("anticipation", 0.0)
        arc.hint_count = data.get("hint_count", 0)
        arc.created_at = data.get("created_at", time.time())
        arc.last_update = data.get("last_update", time.time())
        arc.turns_active = data.get("turns_active", 0)
        return arc

class AdvancedDopamineManager:
    """
    Manages multiple arcs, prioritization, recall, persistence, storing to long-term memory.
    """
    def __init__(self, cns_brain):
        self.cns_brain = cns_brain
        self.active: Dict[str, DopamineArc] = {}
        self.history: List[DopamineArc] = []
        self.recall_prob = 0.25
    
    def add_gaps(self, gaps: List[Dict[str,Any]]):
        for g in gaps:
            merged = self._merge_with_existing(g)
            if not merged:
                arc = DopamineArc(g)
                self.active[arc.arc_id] = arc
    
    def _merge_with_existing(self, gap):
        for arc in list(self.active.values()):
            if arc.gap_type == gap["gap_type"] and self._target_similar(arc.target, gap["target"]):
                arc.salience = max(arc.salience, gap.get("salience", 0.5))
                arc.drive = min(1.5, arc.drive + 0.15 * gap.get("confidence", 0.6))
                return True
        return False
    
    def _target_similar(self, a, b):
        return a.lower() in b.lower() or b.lower() in a.lower()
    
    def register_hint_for_target(self, text):
        for arc in self.active.values():
            if arc.target.lower() in text.lower() or arc.gap_type in text.lower():
                arc.register_hint()
    
    def update_arc_clarity(self, arc_id: str, clarity: float, tone: Optional[str]=None):
        if arc_id in self.active:
            arc = self.active[arc_id]
            arc.update(clarity, tone)
            if arc.state == "CLOSED":
                self._save_arc_to_memory(arc)
                self.history.append(arc)
                del self.active[arc_id]
    
    def _save_arc_to_memory(self, arc):
        """Save closed arc to CNS semantic memory"""
        if hasattr(self.cns_brain, 'semantic_memory'):
            self.cns_brain.semantic_memory.add_fact(
                f"curiosity_arc_{arc.arc_id}",
                f"Explored {arc.gap_type} about {arc.target}",
                confidence=arc.satisfaction
            )
    
    def decay_all(self):
        for arc in self.active.values():
            arc.decay()
    
    def get_priority_arcs(self, top_n=2) -> List[DopamineArc]:
        if not self.active:
            return []
        self.decay_all()
        scored = []
        for arc in self.active.values():
            score = arc.drive * (1 + 0.5*arc.salience) + 0.2 * arc.anticipation
            score *= random.uniform(0.95, 1.05)
            scored.append((score, arc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [arc for _,arc in scored[:top_n]]
    
    def recall_arc(self) -> Optional[DopamineArc]:
        candidates = [a for a in self.history if a.satisfaction < 0.8]
        if candidates and random.random() < self.recall_prob:
            return random.choice(candidates)
        return None
    
    def save_state(self):
        """Save arcs to dict for brain state persistence"""
        return {
            "active_arcs": [arc.to_dict() for arc in self.active.values()],
            "history_arcs": [arc.to_dict() for arc in self.history[-20:]]  # Keep last 20
        }
    
    def load_state(self, state_dict):
        """Load arcs from brain state"""
        if "active_arcs" in state_dict:
            self.active = {}
            for arc_data in state_dict["active_arcs"]:
                arc = DopamineArc.from_dict(arc_data)
                self.active[arc.arc_id] = arc
        
        if "history_arcs" in state_dict:
            self.history = [DopamineArc.from_dict(arc_data) for arc_data in state_dict["history_arcs"]]

# --------- Mode Manager: decides exploration/support/idle/playful/recall ---------
class ModeManager:
    def __init__(self, dm: AdvancedDopamineManager):
        self.dm = dm
        self.locked_exploration: Optional[str] = None
    
    def decide(self, tone_label: Optional[str]=None) -> Dict[str,Any]:
        # Support mode override for emotional states
        if tone_label in ("sad","angry","upset","fear","grief"):
            return {"mode":"support", "reason":"user_emotional", "arc":None}
        
        # Continue locked exploration
        if self.locked_exploration and self.locked_exploration in self.dm.active:
            arc = self.dm.active[self.locked_exploration]
            return {"mode":"exploration","arc":arc.to_signal()}
        
        arcs = self.dm.get_priority_arcs(top_n=2)
        if not arcs:
            rec = self.dm.recall_arc()
            if rec:
                return {"mode":"recall", "arc": rec.to_signal()}
            return {"mode":"idle", "arc": None}
        
        top = arcs[0]
        if top.drive > 0.8 and (top.turns_active >= 1 or random.random() < 0.35):
            self.locked_exploration = top.arc_id
            return {"mode":"exploration", "arc": top.to_signal()}
        
        if tone_label in ("joy","neutral") and random.random() < 0.18:
            return {"mode":"playful", "arc": top.to_signal()}
        
        return {"mode":"curiosity","arc": top.to_signal()}

# --------- Unified Curiosity System ---------
class CuriositySystem:
    """
    Unified curiosity and conversation gap detection system for CNS.
    Integrates with CNS memory and emotion processing.
    
    CONSTITUTIONAL PRINCIPLE: Understanding before opinion
    - Detect gaps to seek understanding BEFORE forming judgments
    - Curiosity precedes opinion in all internal processing
    - No knee-jerk reactions, no reflexive opposition
    - This prevents shallow takes and enables depth
    """
    def __init__(self, cns_brain):
        self.cns_brain = cns_brain
        self.detector = AdvancedGapDetector(cns_brain)
        self.dm = AdvancedDopamineManager(cns_brain)
        self.mode = ModeManager(self.dm)
        
        # Load constitutional beliefs for reference
        self.constitutional_beliefs = []
        try:
            from eros_beliefs import get_constitutional_beliefs
            self.constitutional_beliefs = get_constitutional_beliefs()
        except ImportError:
            pass
    
    def process_turn(self, user_input: str, tone_hint: Optional[str]=None, imagination_insights: Optional[Dict]=None):
        """
        Main entry point for processing a conversation turn.
        Enhanced with imagination engine creative insights.
        
        Args:
            user_input: User's message
            tone_hint: Override emotional tone detection
            imagination_insights: Creative insights from imagination engine (metaphors, counterfactuals, scenarios)
        
        Returns gaps detected and mode signal for strategic response.
        """
        # 1) Gap detection BEFORE memory update
        gaps = self.detector.detect(user_input, tone_hint)
        
        # 1.5) ENHANCE WITH IMAGINATION ENGINE INSIGHTS
        # Imagination engine provides creative perspectives that reveal deeper gaps
        if imagination_insights:
            creative_gaps = self._enhance_with_imagination(gaps, imagination_insights, user_input)
            gaps.extend(creative_gaps)
            # Re-sort by salience
            gaps.sort(key=lambda g: g["confidence"]*g["salience"], reverse=True)
        
        # 2) Register hinting if present
        if any(g["gap_type"]=="hint" for g in gaps):
            self.dm.register_hint_for_target(user_input)
        
        # 3) Add gaps into dopamine manager
        self.dm.add_gaps(gaps)
        
        # 4) Create mode signal based on tone & arcs
        # CRITICAL: Use provided tone_hint if available (for crisis override)
        if tone_hint:
            emotion_label = tone_hint
        else:
            emotion_label, intensity = self.detector._extract_emotion(user_input)
        
        mode_signal = self.mode.decide(tone_label=emotion_label)
        
        # 5) Update memory after detection
        self.detector._add_statement_to_memory(user_input)
        
        # 6) Return structured signal
        return {
            "gaps": gaps, 
            "mode_signal": mode_signal,
            "priority_arcs": self.dm.get_priority_arcs(top_n=3),
            "imagination_enhanced": bool(imagination_insights)
        }
    
    def provide_clarification(self, arc_id: str, clarity: float, tone_hint: Optional[str]=None):
        """Called when user provides follow-up that might fill a gap"""
        self.dm.update_arc_clarity(arc_id, clarity, tone_hint)
    
    def simulate_recall_and_merge(self):
        """Utility to pull historical arcs and merge into active"""
        recalled = self.dm.recall_arc()
        if recalled:
            gap = {
                "gap_type": recalled.gap_type, 
                "target": recalled.target, 
                "salience": recalled.salience,
                "emotional_context": recalled.emotion, 
                "confidence": 0.7
            }
            self.dm.add_gaps([gap])
            return gap
        return None
    
    def save_state(self):
        """Save curiosity system state for brain persistence"""
        return self.dm.save_state()
    
    def load_state(self, state_dict):
        """Load curiosity system state from brain"""
        self.dm.load_state(state_dict)
    
    def _enhance_with_imagination(self, existing_gaps: List[Dict], imagination_insights: Dict, user_input: str) -> List[Dict]:
        """
        Use imagination engine insights to detect deeper conversation gaps.
        
        Imagination provides:
        - metaphors: Reveal abstract concepts that could be explored
        - counterfactuals: "What if..." scenarios that suggest missing context
        - scenarios: Simulated futures that expose unexpressed concerns
        - creative_synthesis: Novel connections between concepts
        """
        creative_gaps = []
        
        # Extract metaphors - reveal abstract thinking gaps
        metaphors = imagination_insights.get('metaphors', [])
        for metaphor in metaphors[:2]:  # Top 2 metaphors
            if isinstance(metaphor, dict):
                target = metaphor.get('concept', metaphor.get('metaphor', ''))
            else:
                target = str(metaphor)
            
            if target and len(target) > 3:
                creative_gaps.append({
                    "gap_type": "creative_metaphor",
                    "target": target,
                    "salience": 0.65,
                    "confidence": 0.75,
                    "emotional_context": imagination_insights.get('creative_mood', 'neutral'),
                    "context": f"metaphorical thinking reveals: {target}"
                })
        
        # Extract counterfactuals - reveal unexplored possibilities
        counterfactuals = imagination_insights.get('counterfactuals', [])
        for cf in counterfactuals[:1]:  # Top counterfactual
            if isinstance(cf, dict):
                scenario = cf.get('scenario', cf.get('what_if', ''))
            else:
                scenario = str(cf)
            
            if scenario and len(scenario) > 10:
                creative_gaps.append({
                    "gap_type": "unexplored_possibility",
                    "target": scenario[:50],  # Shorten
                    "salience": 0.70,
                    "confidence": 0.80,
                    "emotional_context": "curious",
                    "context": f"alternative perspective: {scenario}"
                })
        
        # Extract future projections - reveal hidden concerns
        future_projections = imagination_insights.get('future_projection', [])
        if future_projections:
            projection = future_projections[0] if isinstance(future_projections, list) else future_projections
            if isinstance(projection, dict):
                concern = projection.get('projection', projection.get('concern', ''))
            else:
                concern = str(projection)
            
            if concern and len(concern) > 5:
                creative_gaps.append({
                    "gap_type": "hidden_concern",
                    "target": concern[:40],
                    "salience": 0.75,
                    "confidence": 0.85,
                    "emotional_context": "anticipatory",
                    "context": f"deeper concern: {concern}"
                })
        
        # Extract creative synthesis - novel connections suggest missing links
        synthesis = imagination_insights.get('creative_synthesis', {})
        if synthesis and isinstance(synthesis, dict):
            combined_concepts = synthesis.get('combined_concepts', [])
            if combined_concepts and len(combined_concepts) >= 2:
                creative_gaps.append({
                    "gap_type": "conceptual_bridge",
                    "target": f"{combined_concepts[0]} + {combined_concepts[1]}",
                    "salience": 0.60,
                    "confidence": 0.70,
                    "emotional_context": "integrative",
                    "context": "connecting unrelated concepts"
                })
        
        return creative_gaps
    
    def on_experience(self, experience):
        """ExperienceBus subscriber - detect curiosity gaps from conversation experiences"""
        try:
            if experience.message_content:
                emotional_analysis = experience.emotional_analysis
                tone_hint = emotional_analysis.get('emotion', 'neutral') if emotional_analysis else None
                
                gaps = self.detector.detect(experience.message_content, tone_hint)
                if gaps:
                    self.dm.add_gaps(gaps)
                    
                    try:
                        from experience_bus import get_experience_bus
                        bus = get_experience_bus()
                        bus.contribute_learning("CuriositySystem", {
                            'gaps_detected': len(gaps),
                            'priority_arcs': len(self.dm.get_priority_arcs(top_n=3)),
                            'top_gap_type': gaps[0]['gap_type'] if gaps else None
                        })
                    except Exception:
                        pass
        except Exception as e:
            print(f"⚠️ CuriositySystem experience error: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to the global ExperienceBus"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("CuriositySystem", self.on_experience)
            print("🔍 CuriositySystem subscribed to ExperienceBus - curiosity gap detection active")
        except Exception as e:
            print(f"⚠️ CuriositySystem bus subscription failed: {e}")
