# Shared Text Utilities for CNS
# Deduplicates stopword lists and entity extraction logic used across modules

import re
from typing import List, Set, Dict, Any

class SharedTextUtils:
    """
    Centralized text processing utilities used across curiosity, conversation, and manipulation systems.
    Eliminates duplicate code for stopwords, entity extraction, and text normalization.
    """
    
    # Unified stopword list - used by curiosity_dopamine_system and conversation_companion_system
    STOPWORDS: Set[str] = {
        # Common filler words
        "like", "just", "so", "really", "very", "kinda", "sorta", "maybe", "probably",
        "too", "also", "still", "even", "always", "never", "only", "much", "more", "most",
        # Conversation starters
        "hey", "hi", "hello", "well", "okay", "ok", "yeah", "yep", "nope", "hmm",
        "um", "uh", "oh", "ah", "guess", "say", "telling", "big", "little", "small",
        # Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their", "mine", "yours",
        # Contractions
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
        "called", "named", "someone", "anyone", "everyone", "nobody",
        # Verb forms to filter
        "researching", "studying", "learning", "reading", "writing", "working", "playing",
        "watching", "listening", "building", "creating", "developing", "exploring",
        "thinking", "wondering", "considering", "planning", "hoping", "trying"
    }
    
    # Context prepositions (signal important entities)
    CONTEXT_PREPOSITIONS: Set[str] = {"at", "to", "with", "from", "in", "on", "during", "after", "before"}
    
    # High-priority nouns (story significance markers)
    PRIORITY_NOUNS: Set[str] = {
        # People & relationships
        "partner", "friend", "boyfriend", "girlfriend", "husband", "wife", "spouse",
        "mom", "dad", "mother", "father", "brother", "sister", "family",
        "boss", "coworker", "colleague", "team", "manager", "client",
        # Events & actions
        "fight", "argument", "conversation", "meeting", "date", "party", "dinner",
        "interview", "presentation", "exam", "test", "appointment", "trip", "vacation",
        # Emotions & states
        "feeling", "feelings", "down", "upset", "happy", "sad", "angry", "anxious",
        "excited", "nervous", "stressed", "depressed", "lonely", "hurt", "betrayed",
        # Important objects/places
        "home", "house", "apartment", "office", "school", "hospital", "restaurant",
        # Life domains
        "job", "work", "career", "school", "college", "relationship", "health",
        "money", "finance", "future", "past", "childhood", "dream", "goal"
    }
    
    @classmethod
    def extract_entities_naturally(cls, text: str) -> List[Dict[str, Any]]:
        """
        Extract story-relevant entities using natural language processing.
        
        Prioritizes:
        1. Name patterns ("name is X", "called X")
        2. High-priority semantic nouns (people, events, emotions)
        3. Contextual nouns (after prepositions: "at the beach", "with maya")
        4. Meaningful multi-word phrases
        
        Returns:
            List of entities with salience and confidence scores
        """
        entities = []
        text_lower = text.lower()
        
        # Extract names from patterns
        name_patterns = [
            r"name\s+is\s+(\w+)",
            r"called\s+(\w+)",
            r"named\s+(\w+)",
            r"name's\s+(\w+)",
            r"name\s+was\s+(\w+)"
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in cls.STOPWORDS and len(match) > 2:
                    entities.append({
                        'text': match,
                        'salience': 0.95,  # Names are highly salient
                        'confidence': 0.95,
                        'type': 'name'
                    })
        
        # Extract priority nouns
        words_all = re.findall(r'\b[a-z]+\b', text_lower)
        for word in words_all:
            if word in cls.PRIORITY_NOUNS and word not in [e['text'] for e in entities]:
                entities.append({
                    'text': word,
                    'salience': 0.85,
                    'confidence': 0.9,
                    'type': 'priority_noun'
                })
        
        # Extract contextual patterns (preposition + noun)
        for prep in cls.CONTEXT_PREPOSITIONS:
            pattern = rf"\b{prep}\s+(?:the\s+|a\s+|an\s+)?(\w+)"
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in cls.STOPWORDS and len(match) > 2:
                    if match not in [e['text'] for e in entities]:
                        entities.append({
                            'text': match,
                            'salience': 0.75,
                            'confidence': 0.8,
                            'type': 'contextual'
                        })
        
        # Extract multi-word noun phrases
        words = cls._normalize_words(text_lower)
        for i, word in enumerate(words):
            if word in cls.STOPWORDS or len(word) < 2:
                continue
            
            # 3-word phrases (highest specificity)
            if i < len(words) - 2:
                next_word = words[i+1]
                third_word = words[i+2]
                if (next_word not in cls.STOPWORDS and len(next_word) >= 2 and
                    third_word not in cls.STOPWORDS and len(third_word) >= 2):
                    three_word = f"{word} {next_word} {third_word}"
                    entities.append({
                        'text': three_word,
                        'salience': 0.95,
                        'confidence': 0.95,
                        'type': 'multi_word'
                    })
            
            # 2-word phrases
            elif i < len(words) - 1:
                next_word = words[i+1]
                if next_word not in cls.STOPWORDS and len(next_word) >= 2:
                    two_word = f"{word} {next_word}"
                    entities.append({
                        'text': two_word,
                        'salience': 0.85,
                        'confidence': 0.9,
                        'type': 'multi_word'
                    })
            
            # Single words (require >3 chars)
            if len(word) > 3 and word not in [e['text'] for e in entities]:
                entities.append({
                    'text': word,
                    'salience': 0.6,
                    'confidence': 0.65,
                    'type': 'single_word'
                })
        
        # Deduplicate and sort by relevance
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen:
                seen.add(entity['text'])
                unique_entities.append(entity)
        
        unique_entities.sort(key=lambda e: e['salience'] * e['confidence'], reverse=True)
        return unique_entities[:8]  # Top 8 entities
    
    @classmethod
    def _normalize_words(cls, text_lower: str) -> List[str]:
        """Normalize text: remove punctuation and possessives"""
        words = []
        for w in text_lower.split():
            clean = w.strip('''.,!?;:()"' ''')
            # Remove possessives (china's → china)
            if clean.endswith("'s"):
                clean = clean[:-2]
            if clean:
                words.append(clean)
        return words
    
    @classmethod
    def is_stopword(cls, word: str) -> bool:
        """Check if word is a stopword"""
        return word.lower() in cls.STOPWORDS
    
    @classmethod
    def filter_stopwords(cls, words: List[str]) -> List[str]:
        """Remove stopwords from word list"""
        return [w for w in words if w.lower() not in cls.STOPWORDS]
