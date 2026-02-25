# Conversation Companion System for CNS
# Tracks user interests and drives natural conversation reach-outs
# Mirrors curiosity_dopamine_system.py architecture

import time, math, random, uuid, re
from typing import List, Dict, Any, Optional

# --------- Conversation Opportunity Detector ---------
class ConversationOpportunityDetector:
    """
    ✅ NATURAL SEMANTIC DETECTION - NO KEYWORD TEMPLATES
    
    Detects conversation opportunities from user input using semantic analysis:
    - interests: Extracted naturally from nouns/entities user mentions
    - life_events: Detected from temporal context + activity patterns
    - ongoing_topics: Incomplete narratives/questions to follow up on
    - opinions: Strong emotional triggers + context
    
    Mirrors curiosity system's natural gap detection approach.
    Each opportunity is a dict: {opportunity_type, target, salience, confidence, emotional_context}
    """
    
    # Stopwords for filtering (same as curiosity system + contractions)
    STOPWORDS = {
        # Common filler words
        "like", "just", "so", "really", "very", "kinda", "sorta", "maybe", "probably",
        "too", "also", "still", "even", "always", "never", "only", "much", "more", "most",
        # Conversation starters
        "hey", "hi", "hello", "well", "okay", "ok", "yeah", "yep", "nope", "hmm",
        "um", "uh", "oh", "ah", "guess", "say", "telling", "big", "little", "small",
        # Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their", "mine", "yours",
        # Contractions (critical for filtering fragments like "i'm researching")
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "that's", "there's",
        "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd",
        "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't",
        "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mustn't", "let's",
        # Articles & conjunctions
        "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
        # Prepositions
        "in", "on", "at", "to", "from", "with", "about", "as", "by", "of",
        # Common verbs
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
        # Meta-words
        "name", "person", "people", "place", "situation", "moment", "time", "day",
        "called", "named", "someone", "anyone", "everyone", "nobody"
    }
    
    # Temporal indicators (detect naturally from context, not keyword match)
    TEMPORAL_WORDS = ["weekend", "tomorrow", "tonight", "today", "week", "monday", 
                      "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                      "morning", "afternoon", "evening", "later", "soon", "upcoming"]
    
    # Emotional intensity markers (detect from context)
    EMOTIONAL_MARKERS = ["love", "hate", "favorite", "favourite", "obsessed", 
                         "amazing", "terrible", "brilliant", "awful", "best", "worst"]
    
    def __init__(self, cns_brain):
        """Initialize with CNS brain for memory and emotion access"""
        self.cns_brain = cns_brain
    
    def detect(self, text: str, tone_label: Optional[str]=None) -> List[Dict[str,Any]]:
        """✅ NATURAL SEMANTIC DETECTION - Extract conversation opportunities from actual content"""
        opportunities = []
        text_lower = text.lower()
        
        emotion_label, emotion_int = self._extract_emotion(text)
        tone = tone_label or emotion_label or "neutral"
        
        # 1. ✅ Extract entities/topics NATURALLY (like curiosity system's novelty detection)
        entities = self._extract_entities_naturally(text, text_lower)
        for entity in entities:
            opportunities.append(self._make_opportunity(
                "interest", target=entity['text'], salience=entity['salience'], 
                emotion_context=tone, confidence=entity['confidence']
            ))
        
        # 2. ✅ Detect temporal context + activities (life events)
        life_events = self._detect_temporal_activities(text, text_lower)
        for event in life_events:
            opportunities.append(self._make_opportunity(
                "life_event", target=event['text'], salience=0.75,
                emotion_context=tone, confidence=event['confidence']
            ))
        
        # 3. ✅ Extract opinions from emotional context (not keyword matching)
        opinions = self._extract_opinions_from_emotion(text, text_lower, emotion_int)
        for opinion in opinions:
            opportunities.append(self._make_opportunity(
                "opinion", target=opinion['text'], salience=0.7,
                emotion_context=tone, confidence=opinion['confidence']
            ))
        
        # 4. ✅ Detect ongoing narrative threads (incomplete stories)
        if self._is_incomplete_narrative(text):
            opportunities.append(self._make_opportunity(
                "ongoing_topic", target=self._shorten(text, 8), salience=0.6,
                emotion_context=tone, confidence=0.75
            ))
        
        # Sort by confidence * salience
        opportunities.sort(key=lambda o: o["confidence"]*o["salience"], reverse=True)
        return opportunities
    
    def _extract_entities_naturally(self, text: str, text_lower: str) -> List[Dict[str, Any]]:
        """✅ Extract entities/topics NATURALLY from text (noun-phrase extraction, not random word pairs)"""
        entities = []
        
        # Common verbs to filter out (prevent "researching china's" fragments)
        VERB_FORMS = {
            "researching", "studying", "learning", "reading", "writing", "working", "playing",
            "watching", "listening", "building", "creating", "developing", "exploring",
            "thinking", "wondering", "considering", "planning", "hoping", "trying"
        }
        
        # Normalize: split, strip punctuation, remove possessives
        words = []
        for w in text_lower.split():
            # Strip punctuation
            clean = w.strip('''.,!?;:()"' ''')
            # Remove possessive (china's → china, user's → user)
            if clean.endswith("'s"):
                clean = clean[:-2]
            words.append(clean)
        
        # Extract meaningful nouns (filter stopwords + verbs)
        for i, word in enumerate(words):
            # Skip stopwords and verb forms (but allow short words in phrases)
            if word in self.STOPWORDS or word in VERB_FORMS:
                continue
            
            # Skip single-char noise (but allow 2-char like "AI", "ML", "VR")
            if len(word) < 2:
                continue
            
            # Multi-word noun phrases (prefer these over single words)
            if i < len(words) - 1:
                next_word = words[i+1]
                
                # Both words must be noun-like (not stopwords, not verbs)
                # Allow 2-char tokens in phrases (e.g., "china AI landscape")
                if (next_word not in self.STOPWORDS and 
                    next_word not in VERB_FORMS and 
                    len(next_word) >= 2):
                    
                    two_word = f"{word} {next_word}"
                    entities.append({
                        'text': two_word,
                        'salience': 0.85,  # Multi-word noun phrases = most specific
                        'confidence': 0.9
                    })
                    
                    # Try 3-word noun phrase
                    if i < len(words) - 2:
                        third_word = words[i+2]
                        if (third_word not in self.STOPWORDS and 
                            third_word not in VERB_FORMS and 
                            len(third_word) >= 2):
                            
                            three_word = f"{word} {next_word} {third_word}"
                            entities.append({
                                'text': three_word,
                                'salience': 0.95,  # 3-word phrases = highest specificity
                                'confidence': 0.95
                            })
            
            # Single word entities (require >3 chars to filter noise)
            if len(word) > 3:
                entities.append({
                    'text': word,
                    'salience': 0.6,  # Single words less specific
                    'confidence': 0.65
                })
        
        # Deduplicate (remove exact duplicates only, keep distinct shorter phrases)
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen:
                seen.add(entity['text'])
                unique_entities.append(entity)
        
        # Sort by salience * confidence and return top 5
        unique_entities.sort(key=lambda e: e['salience'] * e['confidence'], reverse=True)
        return unique_entities[:5]
    
    def _detect_temporal_activities(self, text: str, text_lower: str) -> List[Dict[str, Any]]:
        """✅ Detect life events from temporal context + activity patterns"""
        events = []
        
        # Pattern: "going to X", "planning to X", "have X tomorrow"
        activity_patterns = [
            r"going to (\w+(?:\s+\w+)?)",
            r"planning (?:to |on )?(\w+(?:\s+\w+)?)",
            r"have (?:a |an )?(\w+) (?:today|tomorrow|tonight|this week)",
            r"(?:tomorrow|tonight|this week) (?:i'm |i |we're |we )?(\w+(?:\s+\w+)?)"
        ]
        
        for pattern in activity_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                clean = match.strip()
                if clean not in self.STOPWORDS and len(clean) > 3:
                    events.append({
                        'text': clean,
                        'confidence': 0.8
                    })
        
        # Temporal context detection
        for temporal in self.TEMPORAL_WORDS:
            if temporal in text_lower:
                # Extract surrounding context
                idx = text_lower.index(temporal)
                context_start = max(0, idx - 30)
                context_end = min(len(text_lower), idx + 40)
                context = text_lower[context_start:context_end]
                
                # Extract the activity from context
                context_words = [w.strip('''.,!?;:()"' ''') for w in context.split() 
                                if len(w) > 3 and w not in self.STOPWORDS]
                
                if context_words:
                    activity = ' '.join(context_words[:3])  # First 3 meaningful words
                    events.append({
                        'text': f"{temporal} - {activity}",
                        'confidence': 0.75
                    })
        
        # Deduplicate
        seen = set()
        unique_events = []
        for event in events:
            if event['text'] not in seen:
                seen.add(event['text'])
                unique_events.append(event)
        
        return unique_events[:3]
    
    def _extract_opinions_from_emotion(self, text: str, text_lower: str, emotion_intensity: float) -> List[Dict[str, Any]]:
        """✅ Extract opinions from emotional triggers + context (not keyword matching)"""
        opinions = []
        
        # Only extract opinions if there's emotional intensity
        if emotion_intensity < 0.3:
            return opinions
        
        # Find emotionally charged words in context
        for marker in self.EMOTIONAL_MARKERS:
            if marker in text_lower:
                # Extract the subject of the emotion
                pattern = rf"{marker}\s+(\w+(?:\s+\w+)?(?:\s+\w+)?)"
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    clean = match.strip()
                    if clean not in self.STOPWORDS and len(clean) > 3:
                        opinions.append({
                            'text': f"{marker}s {clean}",
                            'confidence': min(0.9, 0.6 + emotion_intensity)  # Higher emotion = higher confidence
                        })
        
        # Also detect strong statements (exclamation marks + intensity)
        if '!' in text and emotion_intensity > 0.5:
            # Extract the statement
            sentences = text.split('!')
            for sentence in sentences[:2]:  # First 2 sentences with !
                words = [w.strip('''.,!?;:()"' ''') for w in sentence.lower().split() 
                        if len(w) > 3 and w not in self.STOPWORDS]
                if words:
                    statement = ' '.join(words[:4])  # First 4 meaningful words
                    opinions.append({
                        'text': statement,
                        'confidence': 0.7 + (emotion_intensity * 0.2)
                    })
        
        return opinions[:3]
    
    def _is_incomplete_narrative(self, text: str) -> bool:
        """✅ Detect incomplete narratives/questions (ongoing conversation threads)"""
        text_stripped = text.strip()
        
        # Trailing ellipsis or unfinished thoughts
        if text_stripped.endswith(("...", "..", "...")):
            return True
        
        # Questions (natural follow-up opportunity)
        if text_stripped.endswith("?"):
            return True
        
        # Incomplete statements (but, so, and then)
        if text_stripped.endswith(("but", "so", "and then", "though", "however")):
            return True
        
        # Narrative verbs without conclusion (like curiosity system)
        narrative_verbs = ["went", "came", "started", "happened", "told", "said", "met", "saw"]
        text_lower = text.lower()
        for verb in narrative_verbs:
            if verb in text_lower and not any(end in text_lower for end in ["then", "after", "later", "finally"]):
                # Has narrative verb but no conclusion = incomplete
                return True
        
        return False
    
    def _extract_emotion(self, text):
        """Use CNS emotion processor to extract emotion and intensity"""
        if hasattr(self.cns_brain, 'emotion_processor'):
            emotion_data = self.cns_brain.emotion_processor.process_emotion(text)
            emotion_label = emotion_data.get('emotion', None)
            valence = abs(emotion_data.get('valence', 0.0))
            arousal = emotion_data.get('arousal', 0.5)
            intensity = (valence + arousal) / 2.0
            return (emotion_label, intensity)
        
        return (None, 0.0)
    
    def _make_opportunity(self, otype, target, salience, emotion_context, confidence):
        return {
            "opportunity_type": otype,
            "target": target,
            "salience": salience,
            "emotional_context": emotion_context,
            "confidence": confidence,
            "created_at": time.time()
        }
    
    def _shorten(self, text, n=6):
        return " ".join(text.split()[:n])

# --------- Conversation Drive ---------
class ConversationDrive:
    """
    Tracks drive/intensity for a conversation opportunity
    Mirrors DopamineArc structure but for conversation urges
    """
    def __init__(self, opportunity):
        self.drive_id = str(uuid.uuid4())
        self.opportunity_type = opportunity["opportunity_type"]
        self.target = opportunity["target"]
        self.salience = float(opportunity.get("salience", 0.6))
        self.emotion = opportunity.get("emotional_context", "neutral")
        self.state = "OPEN"  # OPEN -> ACTIVE -> SUSTAIN -> SATISFIED
        self.drive = 1.0 * self.salience
        self.satisfaction = 0.0
        self.created_at = time.time()
        self.last_update = time.time()
        self.mentions = 0  # How many times user mentioned this
    
    def boost(self, intensity: float = 0.2):
        """Boost drive when relevant event happens (F1 race, weekend)"""
        self.drive = min(1.5, self.drive + intensity)
        self.last_update = time.time()
    
    def register_mention(self):
        """User mentioned this topic again - boost drive"""
        self.mentions += 1
        self.drive = min(1.5, self.drive + 0.1 * (1 + self.mentions * 0.1))
        self.last_update = time.time()
    
    def update(self, relevance: float = 0.0):
        """
        Update drive based on relevance
        relevance: 0..1 how much we addressed this conversation opportunity
        """
        now = time.time()
        
        if self.state == "OPEN":
            self.state = "ACTIVE"
        
        # Natural decay with satisfaction
        self.drive = max(0.05, self.drive * (1 - relevance * 0.3))
        self.satisfaction = min(1.0, self.satisfaction + relevance * 0.5)
        
        if self.satisfaction > 0.7:
            self.state = "SATISFIED"
            self.drive *= 0.5
        
        self.last_update = now
    
    def decay(self):
        """Natural decay over time"""
        elapsed = time.time() - self.last_update
        rate = 0.0005 * (2 - self.salience)  # Higher salience = slower decay
        self.drive *= math.exp(-rate * elapsed)
    
    def to_signal(self):
        return {
            "drive_id": self.drive_id,
            "opportunity_type": self.opportunity_type,
            "target": self.target,
            "drive": round(self.drive, 3),
            "satisfaction": round(self.satisfaction, 3),
            "state": self.state,
            "salience": round(self.salience, 3),
            "mentions": self.mentions
        }
    
    def to_dict(self):
        """Serialize for brain state persistence"""
        return {
            "drive_id": self.drive_id,
            "opportunity_type": self.opportunity_type,
            "target": self.target,
            "salience": self.salience,
            "emotion": self.emotion,
            "state": self.state,
            "drive": self.drive,
            "satisfaction": self.satisfaction,
            "mentions": self.mentions,
            "created_at": self.created_at,
            "last_update": self.last_update
        }
    
    @staticmethod
    def from_dict(data):
        """Deserialize from brain state"""
        opportunity = {
            "opportunity_type": data["opportunity_type"],
            "target": data["target"],
            "salience": data["salience"],
            "emotional_context": data.get("emotion", "neutral")
        }
        drive = ConversationDrive(opportunity)
        drive.drive_id = data["drive_id"]
        drive.state = data.get("state", "OPEN")
        drive.drive = data.get("drive", drive.drive)
        drive.satisfaction = data.get("satisfaction", 0.0)
        drive.mentions = data.get("mentions", 0)
        drive.created_at = data.get("created_at", time.time())
        drive.last_update = data.get("last_update", time.time())
        return drive

# --------- Conversation Companion Manager ---------
class ConversationCompanionManager:
    """
    Manages conversation drives, prioritization, temporal events
    Mirrors AdvancedDopamineManager structure
    """
    def __init__(self, cns_brain):
        self.cns_brain = cns_brain
        self.active: Dict[str, ConversationDrive] = {}
        self.history: List[ConversationDrive] = []
        self.detector = ConversationOpportunityDetector(cns_brain)
    
    def process_input(self, text: str, tone_label: Optional[str]=None):
        """Process user input and update conversation drives"""
        opportunities = self.detector.detect(text, tone_label)
        self.add_opportunities(opportunities)
        
        # Check if existing drives were mentioned
        text_lower = text.lower()
        for drive in list(self.active.values()):
            if drive.target.lower() in text_lower:
                drive.register_mention()
    
    def add_opportunities(self, opportunities: List[Dict[str,Any]]):
        """Add new conversation opportunities"""
        for opp in opportunities:
            merged = self._merge_with_existing(opp)
            if not merged:
                drive = ConversationDrive(opp)
                self.active[drive.drive_id] = drive
    
    def _merge_with_existing(self, opportunity):
        """Merge with existing drive if target is similar"""
        for drive in list(self.active.values()):
            if drive.opportunity_type == opportunity["opportunity_type"] and \
               self._target_similar(drive.target, opportunity["target"]):
                drive.salience = max(drive.salience, opportunity.get("salience", 0.5))
                drive.register_mention()
                return True
        return False
    
    def _target_similar(self, t1, t2):
        """Check if two targets are similar"""
        t1_clean = set(t1.lower().split())
        t2_clean = set(t2.lower().split())
        overlap = len(t1_clean & t2_clean)
        return overlap > 0
    
    def boost_drives_by_event(self, event_type: str, intensity: float = 0.3):
        """
        Boost conversation drives when external events happen
        Example: "f1_race" event boosts all F1-related drives
        """
        event_lower = event_type.lower()
        for drive in list(self.active.values()):
            if event_lower in drive.target.lower():
                drive.boost(intensity)
                print(f"[CONVERSATION] 💬 Boosted drive for {drive.target} due to {event_type}")
    
    def update_all(self):
        """Update all active drives (decay, cleanup)"""
        now = time.time()
        to_archive = []
        
        for drive_id, drive in list(self.active.items()):
            drive.decay()
            
            # Archive drives that are satisfied or too old
            age_days = (now - drive.created_at) / 86400
            if drive.state == "SATISFIED" or drive.drive < 0.1 or age_days > 7:
                to_archive.append(drive_id)
        
        for drive_id in to_archive:
            self.history.append(self.active[drive_id])
            del self.active[drive_id]
        
        # Keep history manageable
        if len(self.history) > 100:
            self.history = self.history[-50:]
    
    def get_top_drives(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N conversation drives by intensity"""
        signals = [drive.to_signal() for drive in self.active.values()]
        signals.sort(key=lambda s: s["drive"], reverse=True)
        return signals[:n]
    
    def get_conversation_urgency(self) -> float:
        """Get overall conversation urgency (0-1)"""
        if not self.active:
            return 0.0
        
        top_drives = sorted([d.drive for d in self.active.values()], reverse=True)[:3]
        return min(1.0, sum(top_drives) / 2.0)
    
    def to_dict(self):
        """Serialize for persistence"""
        return {
            "active": {did: drive.to_dict() for did, drive in self.active.items()},
            "history": [drive.to_dict() for drive in self.history[-20:]]  # Keep last 20
        }
    
    def from_dict(self, data):
        """Deserialize from persistence"""
        self.active = {
            did: ConversationDrive.from_dict(d) 
            for did, d in data.get("active", {}).items()
        }
        self.history = [
            ConversationDrive.from_dict(d) 
            for d in data.get("history", [])
        ]
