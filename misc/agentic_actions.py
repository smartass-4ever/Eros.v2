"""
Agentic Actions System for Eros
Enables Eros to take real actions, not just chat.
"""

import re
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class ActionType(Enum):
    CHAT = "chat"
    REMINDER = "reminder"
    SCHEDULE = "schedule"
    TWITTER_POST = "twitter_post"
    MESSAGE = "message"
    SEARCH = "search"
    UNKNOWN_ACTION = "unknown_action"
    WEB_SEARCH = "web_search"
    CHECK_WEATHER = "check_weather"
    CHECK_NEWS = "check_news"
    CHECK_STOCKS = "check_stocks"
    PLAY_SPOTIFY = "play_spotify"
    CALENDAR_EVENT = "calendar_event"
    SEND_EMAIL = "send_email"
    LOCAL_FILE_OPEN = "local_file_open"
    LOCAL_FILE_MOVE = "local_file_move"
    LOCAL_APP_LAUNCH = "local_app_launch"
    LOCAL_CLIPBOARD = "local_clipboard"
    LOCAL_SCREENSHOT = "local_screenshot"
    LOCAL_SYSTEM_INFO = "local_system_info"
    LOCAL_RUN_SCRIPT = "local_run_script"
    EXPLORE_WINDOW = "explore_window"
    CLICK_CONTROL = "click_control"
    TYPE_IN_CONTROL = "type_in_control"
    SCROLL_CONTROL = "scroll_control"
    GET_WINDOW_LIST = "get_window_list"
    FOCUS_WINDOW = "focus_window"


class ConfirmationLevel(Enum):
    NONE = 0
    QUICK = 1
    EXPLICIT = 2


@dataclass
class ActionIntent:
    action_type: ActionType
    confidence: float
    raw_input: str
    extracted_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    requires_confirmation: bool = False


@dataclass 
class ToolSchema:
    name: str
    description: str
    required_slots: List[str]
    optional_slots: List[str] = field(default_factory=list)
    confirmation_level: ConfirmationLevel = ConfirmationLevel.NONE
    irreversible: bool = False
    

class ActionIntentClassifier:
    """Detects when user wants Eros to DO something vs just chat"""
    
    def __init__(self):
        self.action_patterns = {
            ActionType.REMINDER: [
                r"(?:hey |yo |oh |btw |please |pls |could you |can you |would you )?remind me (?:to |about )?(.+)",
                r"(?:hey |yo )?(?:set|create) (?:a )?reminder (?:for |to |about )?(.+)",
                r"(?:hey |yo )?don'?t (?:let me )?forget (?:to |about )?(.+)",
                r"(?:hey |yo )?(?:ping|alert|notify|tell) me (?:to |about |when |if )?(.+)",
                r"(?:hey |yo )?(?:make sure|ensure) (?:i |to )?(?:remember|don'?t forget) (?:to |about )?(.+)",
                r"(?:hey |yo )?i need (?:a )?reminder (?:to |about |for )?(.+)",
                r"(?:hey |yo )?(?:give me a |set a )?heads up (?:about |for |when )?(.+)",
                r"remind me (.+)",
            ],
            ActionType.SCHEDULE: [
                r"schedule (.+?) (?:for|at|on) (.+)",
                r"add (.+?) to (?:my )?calendar",
                r"book (?:a )?(.+?) (?:for|at|on) (.+)",
                r"put (.+?) on (?:my )?calendar",
            ],
            ActionType.TWITTER_POST: [
                r"(?:tweet|post) ['\"]?(.+?)['\"]?(?:\s+on twitter)?",
                r"post (?:this )?(?:to|on) (?:twitter|x)[:\s]*(.+)?",
                r"tweet (?:this|that|out)[:\s]*(.+)?",
                r"send (?:a )?tweet[:\s]*(.+)?",
            ],
            ActionType.MESSAGE: [
                r"(?:send|text|message) (.+?) (?:to|that|saying) (.+)",
                r"tell (.+?) that (.+)",
                r"let (.+?) know (.+)",
            ],
            ActionType.WEB_SEARCH: [
                r"(?:search|look up|find|google) (?:for |about )?(.+)",
                r"(?:search the web|web search) (?:for )?(.+)",
                r"(?:can you |could you )?(?:search|look up|find|google) (.+)",
            ],
            ActionType.CHECK_WEATHER: [
                r"(?:what(?:'s| is) the )?weather (?:in |for |at )?(.+)?",
                r"(?:is it |will it )(?:rain|snow|cold|hot|sunny|cloudy)",
                r"(?:how(?:'s| is) the )?weather(?: today| tomorrow)?",
            ],
            ActionType.CHECK_NEWS: [
                r"(?:what(?:'s| is) )?(?:in the )?news(?: about | on )?(.+)?",
                r"(?:any |latest |recent )?news(?: about | on )?(.+)?",
                r"what(?:'s| is) happening (?:in |with )?(.+)?",
            ],
            ActionType.CHECK_STOCKS: [
                r"(?:check |what(?:'s| is) )?(?:the )?(?:stock |price (?:of )?)?([a-zA-Z]{1,5})(?: stock| price)?",
                r"(?:how(?:'s| is) )?([a-zA-Z]{1,5})(?: doing| stock)?",
                r"(?:bitcoin|btc|ethereum|eth|crypto)(?: price)?",
                r"(?:stock|price)(?: of | for )?([a-zA-Z]{1,5})",
            ],
            ActionType.SEND_EMAIL: [
                r"(?:send|shoot|write|compose|draft) (?:an? )?(?:email|mail|message) (?:to )?(.+)",
                r"(?:email|mail) (.+?)(?:\s+(?:about|saying|that|to say|with|:)\s+(.+))?",
                r"(?:can you |could you |please )?(?:send|email|mail) (.+)",
                r"(?:write|compose|draft) (?:an? )?(?:email|mail) (?:to )?(.+)",
            ],
            ActionType.EXPLORE_WINDOW: [
                r"(?:what(?:'s| is) |show me |list |get )?(?:the )?(?:buttons?|controls?|elements?|ui|interface)(?: in | on | of )?(?:this |the |current )?(?:window|app|screen)?",
                r"(?:what can (?:i|you) )?(?:click|interact with|see)(?: (?:in|on) (?:this |the )?(?:window|app|screen))?",
                r"(?:scan|explore|analyze|map)(?: (?:this |the |current ))?(?:window|app|screen|ui)",
                r"what(?:'s| is) (?:on |in )?(?:this |the |my )?screen",
                r"(?:show me |what are )?(?:the )?(?:available )?(?:buttons?|controls?|options?)",
            ],
            ActionType.CLICK_CONTROL: [
                r"(?:click|press|tap|hit|select)(?: (?:the|on))? ['\"]?(.+?)['\"]?(?: button| control| element)?$",
                r"(?:click|press|tap)(?: (?:the|on))? (.+)",
            ],
            ActionType.TYPE_IN_CONTROL: [
                r"(?:type|enter|input|write) ['\"]?(.+?)['\"]? (?:in|into|in the) ['\"]?(.+?)['\"]?",
                r"(?:fill|put) ['\"]?(.+?)['\"]? (?:in|into) (?:the )?['\"]?(.+?)['\"]?(?: field| box| input)?",
            ],
            ActionType.GET_WINDOW_LIST: [
                r"(?:what |which |show me )?(?:windows?|apps?)(?: are)? (?:open|running|active)",
                r"(?:list|show)(?: (?:all|my))? (?:open )?(?:windows?|apps?|applications?)",
            ],
            ActionType.FOCUS_WINDOW: [
                r"(?:switch|go|change) to (.+)",
                r"(?:focus|activate|bring up)(?: on)? (.+)",
                r"(?:open|show)(?: the)? (.+) (?:window|app)",
            ],
            ActionType.SCROLL_CONTROL: [
                r"scroll (?:down|up)(?: (?:in|on) (.+))?",
                r"(?:page |scroll )?(?:down|up)(?: (?:in|on) (.+))?",
            ],
        }
        
        self.time_patterns = {
            'seconds': r'(\d+)\s*(?:sec(?:ond)?s?)',
            'minutes': r'(\d+)\s*(?:min(?:ute)?s?)',
            'a_minute': r'\b(?:a|one)\s+min(?:ute)?\b',
            'a_few_minutes': r'\b(?:a few|couple(?:\s+of)?)\s+min(?:ute)?s?\b',
            'hours': r'(\d+)\s*(?:hours?|hrs?)',
            'an_hour': r'\b(?:an?|one)\s+hour\b',
            'days': r'(\d+)\s*(?:days?)',
            'tomorrow': r'\btomorrow\b',
            'tonight': r'\btonight\b',
            'at_time': r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
        }
    
    def classify(self, user_input: str) -> ActionIntent:
        """Classify user input as action or chat"""
        input_lower = user_input.lower().strip()
        print(f"[REGEX-CLASSIFIER] Checking: '{input_lower[:50]}...'")
        
        if ('email' in input_lower or 'mail' in input_lower) and '@' in user_input:
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
            if email_match:
                print(f"[REGEX-CLASSIFIER] ✅ Fast-track EMAIL detected: {email_match.group(0)}")
                slots = {'recipient': email_match.group(0)}
                remaining = user_input.replace(email_match.group(0), '').strip()
                for sep in ['about', 'saying', 'that', 'to say', 'with', ':']:
                    if sep in remaining.lower():
                        parts = remaining.lower().split(sep, 1)
                        if len(parts) > 1:
                            slots['body'] = parts[1].strip()
                            break
                if 'body' not in slots:
                    words = remaining.split()
                    content_start = False
                    content_parts = []
                    for w in words:
                        if content_start:
                            content_parts.append(w)
                        elif w.lower() not in ['send', 'an', 'email', 'to', 'mail', 'message', 'please', 'can', 'you', 'could']:
                            content_start = True
                            content_parts.append(w)
                    if content_parts:
                        slots['body'] = ' '.join(content_parts)
                return ActionIntent(
                    action_type=ActionType.SEND_EMAIL,
                    confidence=0.90,
                    raw_input=user_input,
                    extracted_slots=slots,
                    missing_slots=[],
                    requires_confirmation=False
                )
        
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, input_lower, re.IGNORECASE)
                if match:
                    print(f"[REGEX-CLASSIFIER] ✅ MATCHED {action_type.value} with pattern: {pattern[:40]}...")
                    slots = self._extract_slots(action_type, user_input, match)
                    missing = self._get_missing_slots(action_type, slots)
                    
                    return ActionIntent(
                        action_type=action_type,
                        confidence=0.85,
                        raw_input=user_input,
                        extracted_slots=slots,
                        missing_slots=missing,
                        requires_confirmation=action_type in [
                            ActionType.TWITTER_POST, 
                            ActionType.MESSAGE,
                            ActionType.SCHEDULE
                        ]
                    )
        
        return ActionIntent(
            action_type=ActionType.CHAT,
            confidence=1.0,
            raw_input=user_input
        )
    
    def _extract_slots(self, action_type: ActionType, text: str, match: re.Match) -> Dict[str, Any]:
        """Extract relevant slots based on action type"""
        slots = {}
        
        if action_type == ActionType.REMINDER:
            slots['reminder_text'] = match.group(1) if match.groups() else text
            slots['time_delta'] = self._parse_time(text)
            
        elif action_type == ActionType.TWITTER_POST:
            slots['tweet_content'] = match.group(1) if match.groups() else None
            
        elif action_type == ActionType.SCHEDULE:
            if len(match.groups()) >= 2:
                slots['event_name'] = match.group(1)
                slots['event_time'] = match.group(2)
                
        elif action_type == ActionType.MESSAGE:
            if len(match.groups()) >= 2:
                slots['recipient'] = match.group(1)
                slots['message_content'] = match.group(2)
        
        elif action_type == ActionType.SEND_EMAIL:
            if match.groups():
                full_text = match.group(1) if match.group(1) else text
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
                if email_match:
                    slots['recipient'] = email_match.group(0)
                    remaining = full_text.replace(email_match.group(0), '').strip()
                    if remaining:
                        for sep in ['about', 'saying', 'that', 'to say', 'with', ':']:
                            if sep in remaining.lower():
                                parts = remaining.lower().split(sep, 1)
                                if len(parts) > 1:
                                    slots['body'] = parts[1].strip()
                                    break
                        if 'body' not in slots and remaining:
                            slots['body'] = remaining
                else:
                    slots['recipient'] = full_text.split()[0] if full_text.split() else None
                if len(match.groups()) >= 2 and match.group(2):
                    slots['body'] = match.group(2)
        
        elif action_type == ActionType.WEB_SEARCH:
            slots['query'] = match.group(1) if match.groups() else text
        
        elif action_type == ActionType.CHECK_WEATHER:
            if match.groups() and match.group(1):
                slots['location'] = match.group(1)
        
        elif action_type == ActionType.CHECK_NEWS:
            if match.groups() and match.group(1):
                slots['topic'] = match.group(1)
        
        elif action_type == ActionType.CHECK_STOCKS:
            if match.groups() and match.group(1):
                slots['symbol'] = match.group(1).upper()
        
        elif action_type == ActionType.CLICK_CONTROL:
            if match.groups() and match.group(1):
                slots['control_name'] = match.group(1).strip()
        
        elif action_type == ActionType.TYPE_IN_CONTROL:
            if match.groups():
                slots['text'] = match.group(1) if match.group(1) else ""
                if len(match.groups()) >= 2 and match.group(2):
                    slots['control_name'] = match.group(2).strip()
        
        elif action_type == ActionType.FOCUS_WINDOW:
            if match.groups() and match.group(1):
                slots['window_name'] = match.group(1).strip()
        
        elif action_type == ActionType.SCROLL_CONTROL:
            direction = "down" if "down" in text.lower() else "up"
            slots['direction'] = direction
            if match.groups() and match.group(1):
                slots['target'] = match.group(1).strip()
        
        return slots
    
    def _parse_time(self, text: str) -> Optional[timedelta]:
        """Parse time expressions into timedelta"""
        text_lower = text.lower()
        
        print(f"[TIME-PARSE] Parsing: '{text_lower}'")
        
        seconds_match = re.search(self.time_patterns['seconds'], text_lower)
        if seconds_match:
            secs = int(seconds_match.group(1))
            print(f"[TIME-PARSE] ✅ Matched {secs} seconds")
            return timedelta(seconds=secs)
        
        if re.search(self.time_patterns['a_few_minutes'], text_lower):
            print(f"[TIME-PARSE] ✅ Matched 'a few minutes' -> 3 min")
            return timedelta(minutes=3)
        
        if re.search(self.time_patterns['a_minute'], text_lower):
            print(f"[TIME-PARSE] ✅ Matched 'a minute' -> 1 min")
            return timedelta(minutes=1)
        
        minutes_match = re.search(self.time_patterns['minutes'], text_lower)
        if minutes_match:
            mins = int(minutes_match.group(1))
            print(f"[TIME-PARSE] ✅ Matched {mins} minutes")
            return timedelta(minutes=mins)
        
        if re.search(self.time_patterns['an_hour'], text_lower):
            print(f"[TIME-PARSE] ✅ Matched 'an hour' -> 1 hour")
            return timedelta(hours=1)
        
        hours_match = re.search(self.time_patterns['hours'], text_lower)
        if hours_match:
            hrs = int(hours_match.group(1))
            print(f"[TIME-PARSE] ✅ Matched {hrs} hours")
            return timedelta(hours=hrs)
        
        days_match = re.search(self.time_patterns['days'], text_lower)
        if days_match:
            d = int(days_match.group(1))
            print(f"[TIME-PARSE] ✅ Matched {d} days")
            return timedelta(days=d)
        
        if re.search(self.time_patterns['tomorrow'], text_lower):
            print(f"[TIME-PARSE] ✅ Matched 'tomorrow'")
            return timedelta(days=1)
        
        if re.search(self.time_patterns['tonight'], text_lower):
            now = datetime.now()
            tonight = now.replace(hour=20, minute=0, second=0)
            if tonight < now:
                tonight += timedelta(days=1)
            print(f"[TIME-PARSE] ✅ Matched 'tonight'")
            return tonight - now
        
        print(f"[TIME-PARSE] ⚠️ No time match - defaulting to 5 minutes")
        return timedelta(minutes=5)
    
    def _get_missing_slots(self, action_type: ActionType, slots: Dict) -> List[str]:
        """Determine which required slots are missing"""
        required = {
            ActionType.REMINDER: ['reminder_text'],
            ActionType.TWITTER_POST: ['tweet_content'],
            ActionType.SCHEDULE: ['event_name', 'event_time'],
            ActionType.MESSAGE: ['recipient', 'message_content'],
        }
        
        missing = []
        for slot in required.get(action_type, []):
            if slot not in slots or slots[slot] is None:
                missing.append(slot)
        
        return missing


class LLMIntentDetector:
    """LLM-powered intent detection for natural language understanding.
    
    Uses Mistral to understand user intent and extract structured action data.
    Falls back to regex-based classifier for reliability.
    """
    
    SYSTEM_PROMPT = """You are an intent detector for Eros, a personal AI assistant.
Analyze the user's message and determine if they want Eros to EXECUTE an action or just CHAT.

Available actions:
- chat: Just conversation, no action needed
- reminder: Set a reminder (e.g., "remind me to...", "don't let me forget...")
- web_search: Search the internet (e.g., "search for...", "look up...", "find info about...")
- check_weather: Get weather info (e.g., "what's the weather...", "is it gonna rain...")
- check_news: Get news headlines (e.g., "what's in the news...", "any news about...")
- check_stocks: Stock/crypto prices (e.g., "how's AAPL doing...", "bitcoin price...")
- play_spotify: Control music (e.g., "play...", "put on some music...", "play that song...")
- calendar_event: Calendar actions (e.g., "add to calendar...", "schedule...", "what's on my calendar...")
- send_email: Send email (e.g., "email John...", "send an email to...")
- local_file_open: Open a file (e.g., "open my resume", "show me the report")
- local_app_launch: Launch app (e.g., "open Chrome", "launch Spotify", "start VSCode")
- local_screenshot: Take screenshot (e.g., "take a screenshot", "capture my screen")
- local_system_info: System info (e.g., "how much disk space...", "what's my battery...")

Respond with JSON only:
{
  "action": "action_name",
  "confidence": 0.0-1.0,
  "params": {extracted parameters},
  "reasoning": "brief explanation"
}

Extract relevant params like:
- reminder: {"text": "...", "time": "..."}
- web_search: {"query": "..."}
- check_weather: {"location": "..."}
- check_stocks: {"symbol": "..."}
- play_spotify: {"query": "...", "type": "track/artist/playlist"}
- calendar_event: {"title": "...", "time": "..."}
- send_email: {"recipient": "...", "subject": "...", "body": "..."}
- local_file_open: {"filename": "..."}
- local_app_launch: {"app_name": "..."}

If unsure, default to chat with low confidence."""

    def __init__(self, fallback_classifier: ActionIntentClassifier = None):
        self.fallback = fallback_classifier or ActionIntentClassifier()
        self._api_key = os.environ.get("MISTRAL_API_KEY")
    
    async def detect_intent(self, user_message: str, context: Dict[str, Any] = None) -> ActionIntent:
        """Detect user intent using LLM with fallback to regex"""
        if not self._api_key:
            print("[LLM-INTENT] No API key, using fallback")
            return self.fallback.classify(user_message)
        
        try:
            result = await self._call_llm(user_message, context)
            if result:
                return result
        except Exception as e:
            print(f"[LLM-INTENT] LLM failed: {e}, using fallback")
        
        return self.fallback.classify(user_message)
    
    async def _call_llm(self, user_message: str, context: Dict = None) -> Optional[ActionIntent]:
        """Call Together API for intent detection"""
        import aiohttp
        
        context_str = ""
        if context:
            if context.get("has_local_node"):
                context_str = "\nNote: User has local node connected, so local actions are available."
            if context.get("connected_services"):
                context_str += f"\nConnected services: {', '.join(context['connected_services'])}"
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT + context_str},
            {"role": "user", "content": user_message}
        ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 200,
                    "temperature": 0.1
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                data = await response.json()
        
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        
        action_name = parsed.get("action", "chat")
        confidence = parsed.get("confidence", 0.5)
        params = parsed.get("params", {})
        
        action_type = self._map_action_name(action_name)
        
        if action_type == ActionType.CHAT or confidence < 0.6:
            return None
        
        requires_confirmation = action_type in [
            ActionType.CALENDAR_EVENT,
            ActionType.LOCAL_FILE_MOVE,
            ActionType.LOCAL_RUN_SCRIPT,
        ]
        
        return ActionIntent(
            action_type=action_type,
            confidence=confidence,
            raw_input=user_message,
            extracted_slots=params,
            missing_slots=self._get_missing_for_action(action_type, params),
            requires_confirmation=requires_confirmation
        )
    
    def _map_action_name(self, name: str) -> ActionType:
        """Map LLM action name to ActionType enum"""
        mapping = {
            "chat": ActionType.CHAT,
            "reminder": ActionType.REMINDER,
            "web_search": ActionType.WEB_SEARCH,
            "search": ActionType.WEB_SEARCH,
            "check_weather": ActionType.CHECK_WEATHER,
            "weather": ActionType.CHECK_WEATHER,
            "check_news": ActionType.CHECK_NEWS,
            "news": ActionType.CHECK_NEWS,
            "check_stocks": ActionType.CHECK_STOCKS,
            "stocks": ActionType.CHECK_STOCKS,
            "play_spotify": ActionType.PLAY_SPOTIFY,
            "spotify": ActionType.PLAY_SPOTIFY,
            "music": ActionType.PLAY_SPOTIFY,
            "calendar_event": ActionType.CALENDAR_EVENT,
            "calendar": ActionType.CALENDAR_EVENT,
            "schedule": ActionType.SCHEDULE,
            "send_email": ActionType.SEND_EMAIL,
            "email": ActionType.SEND_EMAIL,
            "local_file_open": ActionType.LOCAL_FILE_OPEN,
            "open_file": ActionType.LOCAL_FILE_OPEN,
            "local_app_launch": ActionType.LOCAL_APP_LAUNCH,
            "launch_app": ActionType.LOCAL_APP_LAUNCH,
            "open_app": ActionType.LOCAL_APP_LAUNCH,
            "local_screenshot": ActionType.LOCAL_SCREENSHOT,
            "screenshot": ActionType.LOCAL_SCREENSHOT,
            "local_system_info": ActionType.LOCAL_SYSTEM_INFO,
            "system_info": ActionType.LOCAL_SYSTEM_INFO,
            "local_clipboard": ActionType.LOCAL_CLIPBOARD,
            "clipboard": ActionType.LOCAL_CLIPBOARD,
            "local_file_move": ActionType.LOCAL_FILE_MOVE,
            "move_file": ActionType.LOCAL_FILE_MOVE,
            "local_run_script": ActionType.LOCAL_RUN_SCRIPT,
            "run_script": ActionType.LOCAL_RUN_SCRIPT,
        }
        return mapping.get(name.lower(), ActionType.CHAT)
    
    def _get_missing_for_action(self, action_type: ActionType, params: Dict) -> List[str]:
        """Determine missing required slots for an action"""
        required = {
            ActionType.REMINDER: ["text"],
            ActionType.WEB_SEARCH: ["query"],
            ActionType.CHECK_WEATHER: [],
            ActionType.CHECK_NEWS: [],
            ActionType.CHECK_STOCKS: ["symbol"],
            ActionType.PLAY_SPOTIFY: ["query"],
            ActionType.CALENDAR_EVENT: ["title"],
            ActionType.SEND_EMAIL: ["recipient"],
            ActionType.LOCAL_FILE_OPEN: ["filename"],
            ActionType.LOCAL_APP_LAUNCH: ["app_name"],
            ActionType.LOCAL_SCREENSHOT: [],
            ActionType.LOCAL_SYSTEM_INFO: [],
        }
        
        missing = []
        for slot in required.get(action_type, []):
            if slot not in params or not params[slot]:
                missing.append(slot)
        
        return missing


class ToolRegistry:
    """Registry of available tools/actions"""
    
    def __init__(self):
        self.tools: Dict[ActionType, ToolSchema] = {}
        self.executors: Dict[ActionType, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        self.register(
            ActionType.REMINDER,
            ToolSchema(
                name="reminder",
                description="Set a reminder to be sent later",
                required_slots=["reminder_text"],
                optional_slots=["time_delta"],
                confirmation_level=ConfirmationLevel.NONE,
                irreversible=False
            )
        )
        
        self.register(
            ActionType.TWITTER_POST,
            ToolSchema(
                name="twitter_post", 
                description="Post a tweet to Twitter/X",
                required_slots=["tweet_content"],
                confirmation_level=ConfirmationLevel.EXPLICIT,
                irreversible=True
            )
        )
        
        self.register(
            ActionType.SCHEDULE,
            ToolSchema(
                name="schedule",
                description="Add an event to calendar",
                required_slots=["event_name", "event_time"],
                confirmation_level=ConfirmationLevel.QUICK,
                irreversible=False
            )
        )
    
    def register(self, action_type: ActionType, schema: ToolSchema):
        self.tools[action_type] = schema
    
    def register_executor(self, action_type: ActionType, executor: Callable):
        self.executors[action_type] = executor
    
    def get_tool(self, action_type: ActionType) -> Optional[ToolSchema]:
        return self.tools.get(action_type)
    
    def get_executor(self, action_type: ActionType) -> Optional[Callable]:
        return self.executors.get(action_type)


@dataclass
class Reminder:
    user_id: str
    channel_id: str
    reminder_text: str
    remind_at: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False
    id: Optional[int] = None


class ReminderManager:
    """Manages reminder storage and execution"""
    
    def __init__(self, db_session_factory=None):
        self.reminders: List[Reminder] = []
        self.db_session_factory = db_session_factory
        self._load_reminders()
    
    def _load_reminders(self):
        """Load pending reminders from database"""
        if self.db_session_factory:
            try:
                from cns_database import CNSDatabase
                db = CNSDatabase()
                session = db.get_session()
                from sqlalchemy import text
                result = session.execute(text("""
                    SELECT id, user_id, channel_id, reminder_text, remind_at, created_at 
                    FROM reminders WHERE completed = false AND remind_at > NOW()
                """))
                for row in result:
                    self.reminders.append(Reminder(
                        id=row[0],
                        user_id=row[1],
                        channel_id=row[2],
                        reminder_text=row[3],
                        remind_at=row[4],
                        created_at=row[5]
                    ))
                session.close()
                print(f"[REMINDER] Loaded {len(self.reminders)} pending reminders")
            except Exception as e:
                print(f"[REMINDER] Could not load reminders: {e}")
    
    def add_reminder(self, user_id: str, channel_id: str, reminder_content: str, 
                     time_delta: timedelta) -> Reminder:
        """Add a new reminder"""
        remind_at = datetime.utcnow() + time_delta
        
        reminder = Reminder(
            user_id=user_id,
            channel_id=channel_id,
            reminder_text=reminder_content,
            remind_at=remind_at
        )
        
        if self.db_session_factory:
            try:
                from cns_database import CNSDatabase
                db = CNSDatabase()
                session = db.get_session()
                from sqlalchemy import text as sql_text
                result = session.execute(sql_text("""
                    INSERT INTO reminders (user_id, channel_id, reminder_text, remind_at, created_at, completed)
                    VALUES (:user_id, :channel_id, :reminder_text, :remind_at, :created_at, false)
                    RETURNING id
                """), {
                    'user_id': user_id,
                    'channel_id': channel_id,
                    'reminder_text': reminder_content,
                    'remind_at': remind_at,
                    'created_at': reminder.created_at
                })
                reminder.id = result.fetchone()[0]
                session.commit()
                session.close()
                print(f"[REMINDER] Saved reminder {reminder.id} to database")
            except Exception as e:
                print(f"[REMINDER] Could not save reminder: {e}")
        
        self.reminders.append(reminder)
        return reminder
    
    def get_due_reminders(self) -> List[Reminder]:
        """Get reminders that are due now"""
        now = datetime.utcnow()
        due = [r for r in self.reminders if r.remind_at <= now and not r.completed]
        return due
    
    def mark_completed(self, reminder: Reminder):
        """Mark a reminder as completed"""
        reminder.completed = True
        
        if reminder.id and self.db_session_factory:
            try:
                from cns_database import CNSDatabase
                db = CNSDatabase()
                session = db.get_session()
                from sqlalchemy import text
                session.execute(text("""
                    UPDATE reminders SET completed = true WHERE id = :id
                """), {'id': reminder.id})
                session.commit()
                session.close()
            except Exception as e:
                print(f"[REMINDER] Could not mark completed: {e}")
        
        self.reminders.remove(reminder)


class AgenticActionSystem:
    """Main coordinator for agentic actions"""
    
    def __init__(self):
        self.classifier = ActionIntentClassifier()
        self.registry = ToolRegistry()
        self.reminder_manager = ReminderManager(db_session_factory=True)
        
        self.registry.register_executor(ActionType.REMINDER, self._execute_reminder)
    
    def process_input(self, user_input: str, user_id: str, channel_id: str) -> Optional[Dict]:
        """Process user input and detect if it's an action request"""
        intent = self.classifier.classify(user_input)
        
        if intent.action_type == ActionType.CHAT:
            return None
        
        print(f"[AGENTIC] Detected action: {intent.action_type.value} (confidence: {intent.confidence})")
        print(f"[AGENTIC] Slots: {intent.extracted_slots}")
        
        if intent.missing_slots:
            return {
                'type': 'missing_slots',
                'action': intent.action_type.value,
                'missing': intent.missing_slots,
                'message': self._generate_slot_request(intent)
            }
        
        tool = self.registry.get_tool(intent.action_type)
        if tool and tool.confirmation_level == ConfirmationLevel.EXPLICIT:
            return {
                'type': 'needs_confirmation',
                'action': intent.action_type.value,
                'slots': intent.extracted_slots,
                'message': self._generate_confirmation_request(intent)
            }
        
        return self.execute_action(intent, user_id, channel_id)
    
    def execute_action(self, intent: ActionIntent, user_id: str, channel_id: str) -> Dict:
        """Execute the detected action"""
        executor = self.registry.get_executor(intent.action_type)
        
        if not executor:
            return {
                'type': 'error',
                'message': f"I don't have the ability to do that yet. {intent.action_type.value} isn't set up."
            }
        
        try:
            result = executor(intent, user_id, channel_id)
            return {
                'type': 'success',
                'action': intent.action_type.value,
                'result': result
            }
        except Exception as e:
            print(f"[AGENTIC] Execution error: {e}")
            return {
                'type': 'error', 
                'message': f"Something went wrong: {str(e)}"
            }
    
    def _execute_reminder(self, intent: ActionIntent, user_id: str, channel_id: str) -> Dict:
        """Execute reminder action"""
        reminder_text = intent.extracted_slots.get('reminder_text', intent.raw_input)
        time_delta = intent.extracted_slots.get('time_delta', timedelta(hours=1))
        
        reminder = self.reminder_manager.add_reminder(
            user_id=user_id,
            channel_id=channel_id,
            reminder_content=reminder_text,
            time_delta=time_delta
        )
        
        remind_time = reminder.remind_at
        time_str = self._format_time_until(time_delta)
        
        return {
            'reminder_id': reminder.id,
            'remind_at': remind_time.isoformat(),
            'time_until': time_str,
            'text': reminder_text,
            'response': f"Got it. I'll remind you {time_str} about: {reminder_text}"
        }
    
    def _format_time_until(self, delta: timedelta) -> str:
        """Format timedelta into human readable string"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"in {total_seconds} second{'s' if total_seconds != 1 else ''}"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"in {minutes} minute{'s' if minutes != 1 else ''}"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"in {hours} hour{'s' if hours != 1 else ''}"
        else:
            days = total_seconds // 86400
            return f"in {days} day{'s' if days != 1 else ''}"
    
    def _generate_slot_request(self, intent: ActionIntent) -> str:
        """Generate a natural request for missing information"""
        if ActionType.TWITTER_POST == intent.action_type:
            return "What would you like me to tweet?"
        elif ActionType.REMINDER == intent.action_type:
            return "What should I remind you about?"
        elif ActionType.SCHEDULE == intent.action_type:
            missing = intent.missing_slots
            if 'event_name' in missing:
                return "What event should I add to your calendar?"
            if 'event_time' in missing:
                return "When should I schedule that for?"
        return "I need a bit more info to do that."
    
    def _generate_confirmation_request(self, intent: ActionIntent) -> str:
        """Generate confirmation request for irreversible actions"""
        if intent.action_type == ActionType.TWITTER_POST:
            content = intent.extracted_slots.get('tweet_content', '')
            return f"Ready to post this tweet:\n\n\"{content}\"\n\nShould I send it? (yes/no)"
        return "Should I go ahead with that? (yes/no)"


agentic_system = AgenticActionSystem()
