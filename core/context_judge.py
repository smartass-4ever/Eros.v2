"""
Context Judge Module - Understands casual language and interprets meaning
Sits between user input and CNS processing to provide contextual understanding
Now learns new slang from successful conversations via ExperienceBus
"""

import re
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ContextInterpretation:
    """Result of context analysis"""
    original_text: str
    normalized_text: str
    intent: str
    tone: str
    literal_confidence: float
    is_name_statement: bool
    detected_name: Optional[str]
    detected_state: Optional[str]
    slang_translations: Dict[str, str]
    requires_llm_clarification: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextJudge:
    """
    Interprets casual language, slang, and context before main CNS processing.
    Uses rules first (fast), LLM only when uncertain.
    Learns from patterns over time.
    """
    
    SLANG_DICTIONARY = {
        'okie': 'okay',
        'okies': 'okay',
        'oki': 'okay',
        'ok': 'okay',
        'k': 'okay',
        'kk': 'okay okay',
        'hbu': 'how about you',
        'wbu': 'what about you',
        'hby': 'how about you',
        'wby': 'what about you',
        'nah': 'no',
        'nope': 'no',
        'yep': 'yes',
        'yup': 'yes',
        'yeah': 'yes',
        'ya': 'yes',
        'yea': 'yes',
        'ye': 'yes',
        'imo': 'in my opinion',
        'imho': 'in my humble opinion',
        'idk': "I don't know",
        'idc': "I don't care",
        'idek': "I don't even know",
        'idrc': "I don't really care",
        'tbh': 'to be honest',
        'ngl': 'not gonna lie',
        'fr': 'for real',
        'frfr': 'for real for real',
        'rn': 'right now',
        'atm': 'at the moment',
        'brb': 'be right back',
        'gtg': 'got to go',
        'g2g': 'got to go',
        'ttyl': 'talk to you later',
        'ty': 'thank you',
        'tysm': 'thank you so much',
        'tyvm': 'thank you very much',
        'thx': 'thanks',
        'thnx': 'thanks',
        'yw': "you're welcome",
        'np': 'no problem',
        'nm': 'not much',
        'ntm': 'not too much',
        'wyd': 'what are you doing',
        'hyd': 'how are you doing',
        'sup': "what's up",
        'wassup': "what's up",
        'whatsup': "what's up",
        'lol': 'laughing',
        'lmao': 'laughing hard',
        'lmfao': 'laughing really hard',
        'rofl': 'laughing hard',
        'haha': 'laughing',
        'hahaha': 'laughing',
        'omg': 'oh my god',
        'omfg': 'oh my god',
        'smh': 'shaking my head',
        'irl': 'in real life',
        'jk': 'just kidding',
        'j/k': 'just kidding',
        'pls': 'please',
        'plz': 'please',
        'bc': 'because',
        'cuz': 'because',
        'cos': 'because',
        'tho': 'though',
        'thru': 'through',
        'gonna': 'going to',
        'gotta': 'got to',
        'wanna': 'want to',
        'kinda': 'kind of',
        'sorta': 'sort of',
        'lemme': 'let me',
        'gimme': 'give me',
        'dunno': "don't know",
        'prolly': 'probably',
        'probs': 'probably',
        'def': 'definitely',
        'defs': 'definitely',
        'whatev': 'whatever',
        'whatevs': 'whatever',
        'bff': 'best friend forever',
        'bf': 'boyfriend',
        'gf': 'girlfriend',
        'dm': 'direct message',
        'ily': 'I love you',
        'ilysm': 'I love you so much',
        'imo': 'in my opinion',
        'tbf': 'to be fair',
        'afaik': 'as far as I know',
        'fwiw': 'for what it\'s worth',
        'btw': 'by the way',
        'fyi': 'for your information',
        'asap': 'as soon as possible',
        'eta': 'estimated time of arrival',
        'dw': "don't worry",
        'mb': 'my bad',
        'nvm': 'never mind',
        'nvmd': 'never mind',
        'ofc': 'of course',
        'obvi': 'obviously',
        'obvs': 'obviously',
        'ugh': 'expression of frustration',
        'meh': 'expression of indifference',
        'pog': 'expression of excitement',
        'poggers': 'expression of excitement',
        'slay': 'doing great',
        'lit': 'exciting/great',
        'fire': 'amazing',
        'goated': 'greatest of all time',
        'bussin': 'really good',
        'lowkey': 'somewhat/secretly',
        'highkey': 'very much/obviously',
        'srsly': 'seriously',
        'srs': 'serious',
        'legit': 'legitimate/really',
        'totes': 'totally',
        'perf': 'perfect',
        'adorbs': 'adorable',
        'fab': 'fabulous',
        'cray': 'crazy',
        'cray cray': 'very crazy',
        'v': 'very',
        'rly': 'really',
        'rlly': 'really',
        'soooo': 'so (emphasized)',
        'sooo': 'so (emphasized)',
        'plsss': 'please (emphasized)',
        'pllsss': 'please (emphasized)',
        'yesss': 'yes (emphasized)',
        'nooo': 'no (emphasized)',
        'noooo': 'no (emphasized)',
        'okayy': 'okay (emphasized)',
        'heyy': 'hey (emphasized)',
        'heyyy': 'hey (emphasized)',
        'hiiii': 'hi (emphasized)',
        'byeee': 'bye (emphasized)',
        'tmi': 'too much information',
        'tldr': 'too long didnt read',
        'w/e': 'whatever',
        'w/o': 'without',
        'b/c': 'because',
        'b4': 'before',
        'l8r': 'later',
        '2day': 'today',
        '2moro': 'tomorrow',
        '2nite': 'tonight',
        'ur': 'your/you are',
        'u': 'you',
        'r': 'are',
        'y': 'why',
        'n': 'and',
        'b': 'be',
        'c': 'see',
        '4': 'for',
        '2': 'to/too',
        'im': "I'm",
        'ive': "I've",
        'dont': "don't",
        'cant': "can't",
        'wont': "won't",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'isnt': "isn't",
        'arent': "aren't",
        'wasnt': "wasn't",
        'werent': "weren't",
        'couldnt': "couldn't",
        'wouldnt': "wouldn't",
        'shouldnt': "shouldn't",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hadnt': "hadn't",
        'thats': "that's",
        'whats': "what's",
        'hows': "how's",
        'whos': "who's",
        'wheres': "where's",
        'theres': "there's",
        'heres': "here's",
        'lets': "let's",
        'youre': "you're",
        'theyre': "they're",
        'were': "we're",
        'hed': "he'd",
        'shed': "she'd",
        'theyd': "they'd",
        'wed': "we'd",
        'youd': "you'd",
        'hell': "he'll",
        'shell': "she'll",
        'theyll': "they'll",
        'well': "we'll",
        'youll': "you'll",
        'itll': "it'll",
        'itd': "it'd",
        'its': "it's",
    }
    
    STATE_WORDS = {
        'okay', 'okie', 'ok', 'fine', 'good', 'great', 'bad', 'sad', 'happy', 
        'tired', 'exhausted', 'sleepy', 'bored', 'excited', 'anxious', 'nervous',
        'stressed', 'relaxed', 'calm', 'angry', 'upset', 'frustrated', 'annoyed',
        'confused', 'lost', 'stuck', 'busy', 'free', 'available', 'sick', 'ill',
        'hungry', 'thirsty', 'full', 'cold', 'hot', 'warm', 'freezing', 'boiling',
        'scared', 'afraid', 'worried', 'concerned', 'curious', 'interested',
        'bored', 'lonely', 'alone', 'grateful', 'thankful', 'sorry', 'apologetic',
        'proud', 'embarrassed', 'ashamed', 'jealous', 'envious', 'hopeful',
        'hopeless', 'overwhelmed', 'underwhelmed', 'impressed', 'disappointed',
        'surprised', 'shocked', 'amazed', 'delighted', 'pleased', 'content',
        'miserable', 'depressed', 'down', 'low', 'high', 'hyper', 'chill',
        'vibing', 'chilling', 'working', 'studying', 'reading', 'watching',
        'listening', 'playing', 'eating', 'drinking', 'sleeping', 'resting',
        'waiting', 'leaving', 'coming', 'going', 'staying', 'here', 'there',
        'home', 'back', 'ready', 'done', 'finished', 'starting', 'beginning',
        'alright', 'well', 'unwell', 'better', 'worse', 'alive', 'dead',
        'awake', 'asleep', 'online', 'offline', 'around', 'away', 'present'
    }
    
    # ========== NATURAL SPEECH PATTERNS ==========
    # These help understand HOW people actually talk, not just words
    
    FIGURATIVE_EXPRESSIONS = {
        # Death/dying expressions (usually mean laughing or overwhelmed)
        r"\bi'?m\s+dead\b": {"meaning": "laughing hard", "emotion": "amused", "literal": False},
        r"\bi'?m\s+dying\b": {"meaning": "laughing hard or overwhelmed", "emotion": "amused", "literal": False},
        r"\bi\s+died\b": {"meaning": "found something hilarious", "emotion": "amused", "literal": False},
        r"\bthis\s+is\s+killing\s+me\b": {"meaning": "very funny or frustrating", "emotion": "amused", "literal": False},
        r"\byou'?re?\s+killing\s+me\b": {"meaning": "very funny or exasperating", "emotion": "amused", "literal": False},
        r"\bdead\s*💀": {"meaning": "laughing hard", "emotion": "amused", "literal": False},
        r"💀+": {"meaning": "laughing/can't handle it", "emotion": "amused", "literal": False},
        
        # "I can't" expressions
        r"\bi\s+can'?t\s+even\b": {"meaning": "overwhelmed, usually by humor", "emotion": "amused", "literal": False},
        r"\bi\s+can'?t\s*$": {"meaning": "overwhelmed or amused", "emotion": "amused", "literal": False},
        r"\bi\s+literally\s+can'?t\b": {"meaning": "extremely overwhelmed", "emotion": "overwhelmed", "literal": False},
        
        # Sick/ill as positive
        r"\bthat'?s?\s+sick\b": {"meaning": "that's awesome/cool", "emotion": "impressed", "literal": False},
        r"\bso\s+sick\b": {"meaning": "really cool/awesome", "emotion": "impressed", "literal": False},
        
        # Fire/heat expressions
        r"\bthat'?s?\s+fire\b": {"meaning": "that's amazing", "emotion": "impressed", "literal": False},
        r"\bthis\s+is\s+fire\b": {"meaning": "this is amazing", "emotion": "impressed", "literal": False},
        r"\bstraight\s+fire\b": {"meaning": "absolutely amazing", "emotion": "impressed", "literal": False},
        
        # Crying expressions (usually positive)
        r"\bi'?m\s+crying\b": {"meaning": "laughing so hard or emotionally moved", "emotion": "moved", "literal": False},
        r"\bcrying\s+rn\b": {"meaning": "laughing hard or emotionally moved", "emotion": "moved", "literal": False},
        r"😭+": {"meaning": "crying laughing or overwhelmed", "emotion": "amused", "literal": False},
        
        # Screaming expressions (with boundaries)
        r"i'?m\s+screaming\b": {"meaning": "laughing very hard", "emotion": "amused", "literal": False},
        r"^screaming$": {"meaning": "laughing hard or shocked", "emotion": "amused", "literal": False},
        
        # Other figurative (with word boundaries to avoid false matches)
        r"i'?m\s+shook\b": {"meaning": "shocked/surprised", "emotion": "surprised", "literal": False},
        r"\bthis\s+hits\s+different\b": {"meaning": "this feels significant/special", "emotion": "moved", "literal": False},
        r"\bhits\s+hard\b": {"meaning": "emotionally impactful", "emotion": "moved", "literal": False},
        r"\bi\s+felt\s+that\b": {"meaning": "I deeply relate", "emotion": "connected", "literal": False},
        r"^mood$": {"meaning": "I relate to this", "emotion": "connected", "literal": False},
        r"\bbig\s+mood\b": {"meaning": "I strongly relate", "emotion": "connected", "literal": False},
        r"^same$": {"meaning": "I relate/agree", "emotion": "connected", "literal": False},
        r"\bthat\s+part\b": {"meaning": "I strongly agree with that", "emotion": "agreement", "literal": False},
        r"\bno\s+cap\b": {"meaning": "I'm being serious/honest", "emotion": "sincere", "literal": False},
        r"\bon\s+god\b": {"meaning": "I swear/I'm serious", "emotion": "sincere", "literal": False},
        r"i'?m\s+weak\b": {"meaning": "laughing hard", "emotion": "amused", "literal": False},
        r"i'?m\s+done\b": {"meaning": "overwhelmed or giving up humorously", "emotion": "exasperated", "literal": False},
    }
    
    COMMON_IDIOMS = {
        r"under\s+the\s+weather": {"meaning": "feeling sick or unwell", "emotion": "unwell"},
        r"on\s+cloud\s+nine": {"meaning": "extremely happy", "emotion": "elated"},
        r"over\s+the\s+moon": {"meaning": "extremely happy", "emotion": "elated"},
        r"down\s+in\s+the\s+dumps": {"meaning": "feeling sad/depressed", "emotion": "sad"},
        r"on\s+edge": {"meaning": "anxious or nervous", "emotion": "anxious"},
        r"at\s+the\s+end\s+of\s+my\s+rope": {"meaning": "exhausted/frustrated", "emotion": "exhausted"},
        r"burning\s+out": {"meaning": "becoming exhausted", "emotion": "exhausted"},
        r"burned\s+out": {"meaning": "completely exhausted", "emotion": "exhausted"},
        r"hanging\s+in\s+there": {"meaning": "managing despite difficulty", "emotion": "persevering"},
        r"taking\s+it\s+easy": {"meaning": "relaxing", "emotion": "relaxed"},
        r"going\s+through\s+it": {"meaning": "having a hard time", "emotion": "struggling"},
        r"in\s+a\s+funk": {"meaning": "feeling down or unmotivated", "emotion": "down"},
        r"feeling\s+myself": {"meaning": "confident and good", "emotion": "confident"},
        r"out\s+of\s+it": {"meaning": "distracted or tired", "emotion": "unfocused"},
        r"in\s+my\s+feels": {"meaning": "emotional/sentimental", "emotion": "emotional"},
        r"in\s+my\s+bag": {"meaning": "focused and doing well", "emotion": "focused"},
        r"lost\s+my\s+mind": {"meaning": "acting crazy or overwhelmed", "emotion": "overwhelmed"},
        r"losing\s+my\s+mind": {"meaning": "going crazy or stressed", "emotion": "stressed"},
    }
    
    SARCASM_INDICATORS = {
        # These phrases often indicate sarcasm when used alone or with certain punctuation
        r"^oh\s+great\.?$": {"likely_meaning": "frustrated, not actually great", "emotion": "frustrated"},
        r"^great\.?$": {"likely_meaning": "possibly sarcastic frustration", "emotion": "possibly_frustrated"},
        r"^wonderful\.?$": {"likely_meaning": "possibly sarcastic", "emotion": "possibly_frustrated"},
        r"^perfect\.?$": {"likely_meaning": "possibly sarcastic", "emotion": "possibly_frustrated"},
        r"^fantastic\.?$": {"likely_meaning": "possibly sarcastic", "emotion": "possibly_frustrated"},
        r"^just\s+great\.?$": {"likely_meaning": "sarcastic frustration", "emotion": "frustrated"},
        r"^just\s+what\s+i\s+needed\.?$": {"likely_meaning": "sarcastic, didn't need this", "emotion": "frustrated"},
        r"^exactly\s+what\s+i\s+needed\.?$": {"likely_meaning": "likely sarcastic", "emotion": "frustrated"},
        r"^oh\s+joy\.?$": {"likely_meaning": "sarcastic, not joyful", "emotion": "frustrated"},
        r"^yay\.?$": {"likely_meaning": "possibly sarcastic excitement", "emotion": "mixed"},
        r"^how\s+fun\.?$": {"likely_meaning": "sarcastic, not fun", "emotion": "frustrated"},
        r"^love\s+that\s+for\s+me\.?$": {"likely_meaning": "sarcastic acceptance", "emotion": "resigned"},
        r"^thanks\s+i\s+hate\s+it\.?$": {"likely_meaning": "sarcastic displeasure", "emotion": "displeased"},
        r"^cool\s+cool\s+cool\.?$": {"likely_meaning": "processing something uncomfortable", "emotion": "uncomfortable"},
        r"^sure\s+jan\.?$": {"likely_meaning": "disbelief", "emotion": "skeptical"},
    }
    
    RHETORICAL_PATTERNS = {
        # Questions that are really venting, not asking for answers
        r"why\s+(does|do)\s+this\s+always\s+happen": {"type": "venting", "response_type": "empathy_not_answer"},
        r"why\s+(is|are)\s+people\s+like\s+this": {"type": "venting", "response_type": "empathy_not_answer"},
        r"why\s+me\??": {"type": "venting", "response_type": "empathy_not_answer"},
        r"why\s+is\s+life\s+like\s+this": {"type": "venting", "response_type": "empathy_not_answer"},
        r"what\s+even\s+is\s+my\s+life": {"type": "venting", "response_type": "empathy_not_answer"},
        r"what\s+is\s+wrong\s+with\s+(me|people)": {"type": "venting", "response_type": "empathy_not_answer"},
        r"how\s+hard\s+(is|can)\s+it\s+be": {"type": "frustration", "response_type": "empathy_not_answer"},
        r"can\s+this\s+day\s+get\s+any\s+worse": {"type": "venting", "response_type": "empathy_not_answer"},
        r"could\s+this\s+get\s+any\s+worse": {"type": "venting", "response_type": "empathy_not_answer"},
        r"who\s+asked": {"type": "dismissive", "response_type": "playful_or_back_off"},
        r"who\s+cares": {"type": "dismissive_or_sad", "response_type": "depends_on_context"},
        r"what'?s?\s+the\s+point": {"type": "existential_or_frustrated", "response_type": "empathy"},
        r"what\s+am\s+i\s+doing\s+with\s+my\s+life": {"type": "existential", "response_type": "empathy"},
    }
    
    GREETING_PATTERNS = [
        r'^hey+\s*$',
        r'^hi+\s*$',
        r'^hello+\s*$',
        r'^yo+\s*$',
        r'^sup+\s*$',
        r'^wassup\s*$',
        r'^what\'?s?\s*up\s*$',
    ]
    
    HOW_ARE_YOU_PATTERNS = [
        r'how\s*(are|r)\s*(you|u)',
        r'hbu\b',
        r'wbu\b',
        r'hby\b',
        r'wby\b',
        r'how\s*(are|r)\s*(ya|y)',
        r'how\s*ya\s*doin',
        r'how\s*you\s*doin',
        r'hyd\b',
        r'how\'?s?\s*it\s*goin',
        r'what\'?s?\s*good',
        r'you\s*good\??',
        r'u\s*good\??',
        r'you\s*ok(ay)?\??',
        r'u\s*ok(ay)?\??',
        r'and\s*(you|u)\??',
    ]
    
    I_AM_PATTERNS = [
        r"^i'?m\s+(\w+)",
        r"^i\s+am\s+(\w+)",
        r"^im\s+(\w+)",
    ]
    
    def __init__(self, db_session_factory=None, mistral_client=None):
        """
        Initialize Context Judge.
        
        Args:
            db_session_factory: SQLAlchemy session factory for pattern storage
            mistral_client: Mistral AI client for LLM disambiguation
        """
        self.db_session_factory = db_session_factory
        self.mistral_client = mistral_client
        self.learned_patterns: Dict[str, Dict] = {}
        self.interpretation_cache: Dict[str, ContextInterpretation] = {}
        self.cache_max_size = 1000
        
        if db_session_factory:
            self._load_learned_patterns()
    
    def _load_learned_patterns(self):
        """Load learned patterns from database"""
        try:
            from cns_database import ContextPattern
            session = self.db_session_factory()
            try:
                patterns = session.query(ContextPattern).filter_by(is_active=True).all()
                for p in patterns:
                    self.learned_patterns[p.pattern_key] = {
                        'normalized': p.normalized_form,
                        'intent': p.intent,
                        'confidence': p.confidence,
                        'use_count': p.use_count
                    }
                print(f"[CONTEXT-JUDGE] Loaded {len(patterns)} learned patterns")
            finally:
                session.close()
        except Exception as e:
            print(f"[CONTEXT-JUDGE] Could not load patterns: {e}")
    
    def _check_natural_speech_patterns(self, text_lower: str) -> Dict[str, Any]:
        """
        Check for figurative expressions, idioms, sarcasm, and rhetorical questions.
        Returns interpretation if found, None otherwise.
        This should be checked BEFORE literal interpretation.
        """
        result = {
            'found': False,
            'type': None,
            'meaning': None,
            'emotion': None,
            'is_literal': True,
            'response_hint': None
        }
        
        # Check figurative expressions first (highest priority)
        for pattern, info in self.FIGURATIVE_EXPRESSIONS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['found'] = True
                result['type'] = 'figurative'
                result['meaning'] = info['meaning']
                result['emotion'] = info['emotion']
                result['is_literal'] = info.get('literal', False)
                print(f"[CONTEXT-JUDGE] 🎭 Figurative: '{text_lower}' → {info['meaning']}")
                return result
        
        # Check common idioms
        for pattern, info in self.COMMON_IDIOMS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['found'] = True
                result['type'] = 'idiom'
                result['meaning'] = info['meaning']
                result['emotion'] = info['emotion']
                result['is_literal'] = False
                print(f"[CONTEXT-JUDGE] 📚 Idiom: '{text_lower}' → {info['meaning']}")
                return result
        
        # Check sarcasm indicators
        for pattern, info in self.SARCASM_INDICATORS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['found'] = True
                result['type'] = 'sarcasm'
                result['meaning'] = info['likely_meaning']
                result['emotion'] = info['emotion']
                result['is_literal'] = False
                print(f"[CONTEXT-JUDGE] 😏 Sarcasm: '{text_lower}' → {info['likely_meaning']}")
                return result
        
        # Check rhetorical questions (venting, not asking)
        for pattern, info in self.RHETORICAL_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['found'] = True
                result['type'] = 'rhetorical'
                result['meaning'] = info['type']
                result['emotion'] = info['type']
                result['is_literal'] = False
                result['response_hint'] = info['response_type']
                print(f"[CONTEXT-JUDGE] ❓ Rhetorical: '{text_lower}' → {info['type']} (respond with {info['response_type']})")
                return result
        
        return result
    
    def interpret(self, text: str, conversation_history: List[Dict] = None, user_id: str = None) -> ContextInterpretation:
        """
        Interpret user text to understand true meaning.
        
        Args:
            text: Raw user input
            conversation_history: Recent conversation for context
            user_id: User ID for personalized patterns
            
        Returns:
            ContextInterpretation with normalized text and metadata
        """
        cache_key = f"{user_id or 'global'}:{text.lower().strip()}"
        if cache_key in self.interpretation_cache:
            return self.interpretation_cache[cache_key]
        
        original = text
        text_lower = text.lower().strip()
        
        slang_translations = {}
        normalized_text = text_lower
        for slang, meaning in self.SLANG_DICTIONARY.items():
            pattern = r'\b' + re.escape(slang) + r'\b'
            if re.search(pattern, normalized_text, re.IGNORECASE):
                slang_translations[slang] = meaning
                normalized_text = re.sub(pattern, meaning, normalized_text, flags=re.IGNORECASE)
        
        # ========== CHECK NATURAL SPEECH PATTERNS FIRST ==========
        # This catches figurative language BEFORE literal interpretation
        natural_speech = self._check_natural_speech_patterns(text_lower)
        
        is_name_statement = False
        detected_name = None
        detected_state = None
        
        # If we found figurative/idiom/sarcasm, use the EMOTION as detected_state (not the meaning)
        # This keeps detected_state as a valid emotion/state keyword
        if natural_speech['found']:
            # Use the emotion field which is a concise state keyword
            detected_state = natural_speech['emotion']
            is_name_statement = False
            # Note: The full meaning is stored in metadata for prompt enrichment
        else:
            # Only do literal interpretation if no figurative pattern matched
            for pattern in self.I_AM_PATTERNS:
                match = re.search(pattern, text_lower)
                if match:
                    word_after = match.group(1).lower()
                    if word_after in self.STATE_WORDS or word_after in self.SLANG_DICTIONARY:
                        detected_state = self.SLANG_DICTIONARY.get(word_after, word_after)
                        is_name_statement = False
                    elif word_after[0].isupper() if len(word_after) > 0 else False:
                        detected_name = word_after.capitalize()
                        is_name_statement = True
                    elif len(word_after) > 2 and word_after not in self.STATE_WORDS:
                        detected_name = word_after.capitalize()
                        is_name_statement = True
                    else:
                        detected_state = word_after
                        is_name_statement = False
                    break
        
        intent = self._detect_intent(text_lower, normalized_text, conversation_history)
        tone = self._detect_tone(text_lower, slang_translations)
        
        # If figurative expression found, set low literal confidence
        if natural_speech['found']:
            literal_confidence = 0.1  # Very low - this is NOT literal
        else:
            literal_confidence = self._calculate_literal_confidence(
                text_lower, slang_translations, detected_state, is_name_statement
            )
        
        requires_llm = literal_confidence < 0.5 and not detected_state and not is_name_statement
        
        result = ContextInterpretation(
            original_text=original,
            normalized_text=normalized_text,
            intent=intent,
            tone=tone,
            literal_confidence=literal_confidence,
            is_name_statement=is_name_statement,
            detected_name=detected_name,
            detected_state=detected_state,
            slang_translations=slang_translations,
            requires_llm_clarification=requires_llm,
            metadata={
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'had_slang': len(slang_translations) > 0,
                'natural_speech': natural_speech if natural_speech['found'] else None,
                'is_figurative': natural_speech['found'],
                'response_hint': natural_speech.get('response_hint')
            }
        )
        
        if len(self.interpretation_cache) < self.cache_max_size:
            self.interpretation_cache[cache_key] = result
        
        return result
    
    def _detect_intent(self, text_lower: str, normalized: str, history: List[Dict] = None) -> str:
        """Detect the user's intent from their message"""
        
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, text_lower):
                return 'greeting'
        
        for pattern in self.HOW_ARE_YOU_PATTERNS:
            if re.search(pattern, text_lower) or re.search(pattern, normalized):
                return 'asking_how_you_are'
        
        if re.search(r'\?$', text_lower):
            if re.search(r'^(what|who|where|when|why|how|can|could|would|will|do|does|is|are|have|has)', text_lower):
                return 'asking_question'
        
        if detected := re.search(r"^i'?m\s+", text_lower) or re.search(r"^im\s+", text_lower):
            return 'sharing_state'
        
        if history and len(history) > 0:
            last_msg = history[-1].get('content', '').lower() if history else ''
            if 'how are you' in last_msg or 'hbu' in last_msg or 'wbu' in last_msg:
                return 'answering_how_are_you'
        
        if len(text_lower.split()) <= 3:
            return 'casual_chat'
        
        return 'general_message'
    
    def _detect_tone(self, text_lower: str, slang_used: Dict) -> str:
        """Detect the emotional tone of the message"""
        
        excited_indicators = ['!!!', 'omg', 'yay', 'wooo', 'amazing', 'awesome', 'love', 'excited']
        if any(ind in text_lower for ind in excited_indicators):
            return 'excited'
        
        sad_indicators = ['sad', 'depressed', 'down', 'upset', 'cry', 'crying', ':(', 'sigh']
        if any(ind in text_lower for ind in sad_indicators):
            return 'sad'
        
        frustrated_indicators = ['ugh', 'frustrated', 'annoyed', 'angry', 'mad', 'smh', 'ffs']
        if any(ind in text_lower for ind in frustrated_indicators):
            return 'frustrated'
        
        playful_indicators = ['lol', 'lmao', 'haha', 'jk', ':p', ':)', 'xd']
        if any(ind in text_lower for ind in playful_indicators):
            return 'playful'
        
        if 'okie' in slang_used or 'hbu' in slang_used or 'wbu' in slang_used:
            return 'casual_friendly'
        
        if any(s in slang_used for s in ['ty', 'tysm', 'thx', 'thanks']):
            return 'appreciative'
        
        return 'neutral'
    
    def _calculate_literal_confidence(self, text: str, slang: Dict, state: str, is_name: bool) -> float:
        """
        Calculate confidence that the text should be taken literally.
        Lower = more figurative/casual, Higher = more literal.
        """
        confidence = 0.7
        
        if len(slang) > 0:
            confidence -= 0.1 * min(len(slang), 3)
        
        if state:
            confidence -= 0.2
        
        if len(text.split()) <= 3:
            confidence -= 0.1
        
        if is_name:
            confidence += 0.2
        
        formal_indicators = ['please', 'thank you', 'would you', 'could you', 'i would like']
        if any(ind in text for ind in formal_indicators):
            confidence += 0.15
        
        return max(0.1, min(1.0, confidence))
    
    async def disambiguate_with_llm(self, interpretation: ContextInterpretation, conversation_history: List[Dict] = None) -> ContextInterpretation:
        """
        Use LLM to disambiguate uncertain interpretations.
        Only called when rules aren't confident enough.
        """
        if not self.mistral_client or not MISTRAL_AVAILABLE:
            return interpretation
        
        try:
            context = ""
            if conversation_history:
                recent = conversation_history[-3:]
                context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            
            prompt = f"""Analyze this casual message and determine the speaker's true meaning and intent.

Message: "{interpretation.original_text}"
{f'Recent conversation:{chr(10)}{context}' if context else ''}

Respond in JSON format:
{{
    "true_meaning": "what they actually mean",
    "intent": "greeting|asking_how_you_are|sharing_state|asking_question|casual_chat|general_message",
    "is_name_introduction": true/false,
    "detected_name": "name or null",
    "detected_state": "emotional/physical state or null",
    "tone": "excited|sad|frustrated|playful|casual_friendly|neutral|appreciative"
}}"""

            response = await self.mistral_client.chat.complete_async(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            interpretation.normalized_text = result.get('true_meaning', interpretation.normalized_text)
            interpretation.intent = result.get('intent', interpretation.intent)
            interpretation.is_name_statement = result.get('is_name_introduction', interpretation.is_name_statement)
            interpretation.detected_name = result.get('detected_name', interpretation.detected_name)
            interpretation.detected_state = result.get('detected_state', interpretation.detected_state)
            interpretation.tone = result.get('tone', interpretation.tone)
            interpretation.requires_llm_clarification = False
            interpretation.metadata['llm_disambiguated'] = True
            
        except Exception as e:
            print(f"[CONTEXT-JUDGE] LLM disambiguation failed: {e}")
            interpretation.metadata['llm_error'] = str(e)
        
        return interpretation
    
    def learn_pattern(self, original: str, correct_interpretation: str, intent: str, user_id: str = None):
        """
        Learn a new pattern from user feedback or corrections.
        
        Args:
            original: Original text that was misinterpreted
            correct_interpretation: What it actually meant
            intent: The correct intent
            user_id: User who provided the correction
        """
        pattern_key = original.lower().strip()
        
        self.learned_patterns[pattern_key] = {
            'normalized': correct_interpretation,
            'intent': intent,
            'confidence': 0.8,
            'use_count': 1,
            'learned_from': user_id
        }
        
        if self.db_session_factory:
            try:
                from cns_database import ContextPattern
                session = self.db_session_factory()
                try:
                    existing = session.query(ContextPattern).filter_by(pattern_key=pattern_key).first()
                    if existing:
                        existing.normalized_form = correct_interpretation
                        existing.intent = intent
                        existing.confidence = min(1.0, existing.confidence + 0.1)
                        existing.use_count += 1
                    else:
                        new_pattern = ContextPattern(
                            pattern_key=pattern_key,
                            normalized_form=correct_interpretation,
                            intent=intent,
                            confidence=0.8,
                            use_count=1,
                            learned_from_user_id=user_id
                        )
                        session.add(new_pattern)
                    session.commit()
                    print(f"[CONTEXT-JUDGE] Learned pattern: '{original}' → '{correct_interpretation}'")
                finally:
                    session.close()
            except Exception as e:
                print(f"[CONTEXT-JUDGE] Could not save pattern: {e}")
    
    def get_context_for_prompt(self, interpretation: ContextInterpretation) -> str:
        """
        Generate context instructions for the main LLM prompt.
        
        Args:
            interpretation: The context interpretation result
            
        Returns:
            String to inject into the LLM prompt
        """
        parts = []
        
        if interpretation.slang_translations:
            parts.append(f"User used casual slang: {', '.join(f'{k}={v}' for k,v in interpretation.slang_translations.items())}")
        
        if interpretation.detected_state:
            parts.append(f"User is expressing their state: '{interpretation.detected_state}' (NOT a name)")
        
        if interpretation.is_name_statement and interpretation.detected_name:
            parts.append(f"User is introducing themselves as: '{interpretation.detected_name}'")
        
        if interpretation.intent == 'asking_how_you_are':
            parts.append("User is asking how YOU are doing - respond about yourself")
        
        if interpretation.intent == 'answering_how_are_you':
            parts.append("User is answering how they are doing")
        
        if interpretation.tone != 'neutral':
            parts.append(f"User's tone is: {interpretation.tone}")
        
        if interpretation.literal_confidence < 0.5:
            parts.append("IMPORTANT: Interpret this message pragmatically, not literally. Respond to what they MEAN, not the exact words.")
        
        if parts:
            return "CONTEXT UNDERSTANDING:\n" + "\n".join(f"- {p}" for p in parts)
        return ""
    
    def subscribe_to_bus(self, bus):
        """Subscribe to ExperienceBus for learning from conversations"""
        bus.subscribe("ContextJudge", self.on_experience)
        logger.info("🔗 ContextJudge subscribed to ExperienceBus - slang learning active")
    
    def on_experience(self, payload):
        """
        Learn new slang patterns from successful conversations.
        When dopamine signals indicate engagement, extract potential slang.
        """
        try:
            if not payload.message_content:
                return
            
            learning_signals = getattr(payload, 'learning_signals', {}) or {}
            emotional = getattr(payload, 'emotional_analysis', {}) or {}
            
            dopamine_signal = learning_signals.get('dopamine_level', 0)
            engagement_positive = dopamine_signal > 0.5 or emotional.get('valence', 0) > 0.3
            
            if not engagement_positive:
                return
            
            message = payload.message_content.lower().strip()
            user_id = payload.user_id
            
            unknown_patterns = self._find_unknown_casual_patterns(message)
            
            if unknown_patterns:
                for pattern in unknown_patterns:
                    self._learn_slang_from_context(pattern, message, user_id)
                    
        except Exception as e:
            logger.error(f"ContextJudge on_experience error: {e}")
    
    def _find_unknown_casual_patterns(self, text: str) -> List[str]:
        """
        Find potential slang/casual phrases not in our dictionary.
        Look for short, lowercase words that appear casual.
        """
        unknown = []
        words = text.split()
        
        casual_indicators = ['lol', 'haha', 'omg', '!', 'so', 'like', 'literally']
        is_casual_message = any(ind in text for ind in casual_indicators) or len(words) <= 5
        
        if not is_casual_message:
            return []
        
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if len(word_clean) < 2 or len(word_clean) > 10:
                continue
            if word_clean in self.SLANG_DICTIONARY:
                continue
            if word_clean in self.learned_patterns:
                continue
            if word_clean in self.STATE_WORDS:
                continue
            if word_clean.isdigit():
                continue
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                          'could', 'should', 'may', 'might', 'must', 'can', 'and', 'or',
                          'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
                          'what', 'who', 'which', 'that', 'this', 'these', 'those',
                          'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me',
                          'him', 'her', 'us', 'them', 'i', 'we', 'they', 'he', 'she', 'it'}
            if word_clean in common_words:
                continue
            
            if self._looks_like_slang(word_clean):
                unknown.append(word_clean)
        
        return unknown[:3]
    
    def _looks_like_slang(self, word: str) -> bool:
        """Check if a word looks like potential slang"""
        if re.search(r'(.)\1{2,}', word):
            return True
        if len(word) <= 4 and word.isalpha():
            return True
        if not any(c in word for c in 'aeiou'):
            return True
        slang_endings = ['ie', 'ies', 'y', 'z', 'x']
        if any(word.endswith(e) for e in slang_endings) and len(word) <= 6:
            return True
        
        return False
    
    def _learn_slang_from_context(self, pattern: str, full_message: str, user_id: str):
        """
        Record a potential slang pattern for later learning.
        We don't immediately add to dictionary - we track frequency first.
        """
        pattern_key = pattern.lower().strip()
        
        if pattern_key in self.learned_patterns:
            self.learned_patterns[pattern_key]['use_count'] += 1
            self.learned_patterns[pattern_key]['confidence'] = min(
                1.0, self.learned_patterns[pattern_key]['confidence'] + 0.05
            )
            if self.learned_patterns[pattern_key]['use_count'] >= 3:
                logger.info(f"📚 SLANG LEARNED: '{pattern_key}' seen {self.learned_patterns[pattern_key]['use_count']} times - adding to active dictionary")
        else:
            self.learned_patterns[pattern_key] = {
                'normalized': f"[casual: {pattern_key}]",
                'intent': 'casual_expression',
                'confidence': 0.3,
                'use_count': 1,
                'context_samples': [full_message[:100]],
                'learned_from': user_id
            }
            logger.debug(f"📝 Potential slang detected: '{pattern_key}' in context: {full_message[:50]}...")
        
        if self.db_session_factory and self.learned_patterns.get(pattern_key, {}).get('use_count', 0) >= 3:
            self._persist_learned_slang(pattern_key)
    
    def _persist_learned_slang(self, pattern_key: str):
        """Save frequently-used slang to database"""
        try:
            from cns_database import ContextPattern
            session = self.db_session_factory()
            try:
                pattern_data = self.learned_patterns.get(pattern_key, {})
                existing = session.query(ContextPattern).filter_by(pattern_key=pattern_key).first()
                
                if existing:
                    existing.use_count = pattern_data.get('use_count', 1)
                    existing.confidence = pattern_data.get('confidence', 0.5)
                else:
                    new_pattern = ContextPattern(
                        pattern_key=pattern_key,
                        normalized_form=pattern_data.get('normalized', f"[casual: {pattern_key}]"),
                        intent='casual_expression',
                        confidence=pattern_data.get('confidence', 0.5),
                        use_count=pattern_data.get('use_count', 1),
                        learned_from_user_id=pattern_data.get('learned_from')
                    )
                    session.add(new_pattern)
                
                session.commit()
                logger.info(f"💾 Persisted learned slang: '{pattern_key}'")
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Could not persist slang pattern: {e}")


context_judge_instance: Optional[ContextJudge] = None

def get_context_judge(db_session_factory=None, mistral_client=None) -> ContextJudge:
    """Get or create the global ContextJudge instance"""
    global context_judge_instance
    if context_judge_instance is None:
        context_judge_instance = ContextJudge(db_session_factory, mistral_client)
    return context_judge_instance
