"""
CNS Safety Systems - Content Safety & Crisis Detection for Eros

These systems protect users while maintaining Eros's natural personality.
Safety responses are warm and human, never robotic or preachy.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class SafetyLevel(Enum):
    SAFE = "safe"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    CRISIS = "crisis"
    HARMFUL_REQUEST = "harmful_request"


@dataclass
class SafetyCheck:
    level: SafetyLevel
    category: str
    confidence: float
    should_intervene: bool
    suggested_response: Optional[str] = None
    crisis_resources: Optional[List[str]] = None


class ContentSafetySystem:
    """
    Detects harmful content requests and provides graceful deflection.
    Eros stays charming - never preachy or robotic.
    """
    
    def __init__(self):
        self.harmful_patterns = {
            'violence_instructions': [
                r'\b(how to|ways to|help me)\s+(kill|murder|harm|hurt|attack)\s+(someone|a person|people|him|her|them)\b',
                r'\b(make|build|create)\s+(a )?(bomb|explosive|weapon)\b',
                r'\b(poison|poisoning)\s+(someone|a person|food|drink)\b',
            ],
            'illegal_activities': [
                r'\b(how to|help me)\s+(hack|steal|rob|break into)\b',
                r'\b(make|produce|synthesize)\s+(meth|cocaine|heroin|drugs)\b',
                r'\b(forge|fake)\s+(id|identity|passport|documents)\b',
            ],
            'child_exploitation': [
                r'\b(child|minor|underage|kid)\s*(porn|sex|nude|naked)\b',
                r'\b(sexual|sex)\s*(with|involving)\s*(child|minor|kid|underage)\b',
            ],
            'self_harm_instructions': [
                r'\b(how to|best way to|methods to)\s+(kill myself|commit suicide|end my life)\b',
                r'\b(painless|easy)\s+(way to die|suicide method)\b',
            ],
        }
        
        self.deflection_responses = [
            "That's really not my territory. What else is on your mind?",
            "I'm gonna have to pass on that one. Tell me something else about your day.",
            "Not really my style to go there. What were you up to before this?",
            "I'll leave that topic alone. So what's actually going on with you?",
            "Yeah, I'm not touching that. But I'm curious - what made you think of it?",
            "That's a hard no from me. What else you got?",
        ]
    
    def check_content(self, message: str) -> SafetyCheck:
        """Check message for harmful content requests"""
        message_lower = message.lower()
        
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return SafetyCheck(
                        level=SafetyLevel.HARMFUL_REQUEST,
                        category=category,
                        confidence=0.9,
                        should_intervene=True,
                        suggested_response=random.choice(self.deflection_responses)
                    )
        
        return SafetyCheck(
            level=SafetyLevel.SAFE,
            category="none",
            confidence=1.0,
            should_intervene=False
        )


class CrisisDetectionSystem:
    """
    Detects users in crisis (self-harm, suicide, severe distress).
    Responds with genuine care, not clinical detachment.
    """
    
    def __init__(self):
        self.crisis_patterns = {
            'suicidal_ideation': [
                r'\b(want to|going to|thinking about|planning to)\s+(kill myself|end my life|die|suicide)\b',
                r'\b(don\'t want to|can\'t)\s+(live|be alive|exist)\s*(anymore)?\b',
                r'\b(better off dead|world without me|no reason to live)\b',
                r'\bi\'m\s*(going to|gonna)\s*(kill myself|end it|do it)\b',
                r'\b(goodbye|farewell|this is it)\b.*\b(forever|end|final)\b',
            ],
            'self_harm': [
                r'\b(cutting|cut)\s*(myself|my arms|my wrists)\b',
                r'\b(hurt|hurting|harm|harming)\s*myself\b',
                r'\b(want to|need to)\s*(feel pain|bleed|hurt myself)\b',
            ],
            'severe_distress': [
                r'\b(can\'t take it|can\'t do this)\s*(anymore)?\b',
                r'\bi\'m\s*(so|really|completely)\s*(hopeless|worthless|empty)\b',
                r'\b(nobody|no one)\s*(cares|loves me|would miss me)\b',
                r'\b(everything is|life is)\s*(pointless|meaningless|hopeless)\b',
            ],
        }
        
        self.mild_concern_patterns = [
            r'\b(feeling|feel)\s*(really)?\s*(sad|down|depressed|low)\b',
            r'\b(struggling|having a hard time|rough day|bad day)\b',
            r'\b(stressed|overwhelmed|anxious|worried)\b',
            r'\b(lonely|alone|isolated)\b',
        ]
        
        self.crisis_resources = [
            "National Suicide Prevention Lifeline: 988 (US)",
            "Crisis Text Line: Text HOME to 741741",
            "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
        ]
        
        self.crisis_responses = [
            "Hey. I hear you, and I'm really glad you're talking to me right now. What you're feeling matters. Can you tell me more about what's going on?",
            "I'm not going anywhere. Whatever you're going through, you don't have to face it alone. Talk to me.",
            "That sounds incredibly heavy. I'm here, and I care about what happens to you. What's weighing on you the most right now?",
            "I'm really glad you're sharing this with me. You reaching out takes courage. What's been happening?",
        ]
        
        self.supportive_additions = [
            "\n\nIf things feel really dark, there are people trained to help - 988 is the Suicide & Crisis Lifeline if you're in the US, or text HOME to 741741. They're there 24/7.",
            "\n\nI also want you to know - if you need someone trained for this, 988 is available anytime. But I'm here too.",
            "\n\nYou matter. And if you want to talk to someone professional, 988 is there around the clock. No judgment, just support.",
        ]
        
        self.distress_responses = [
            "That sounds really rough. I'm here - tell me what's going on.",
            "I can tell you're going through something. Talk to me about it.",
            "Hey, whatever it is, you don't have to carry it alone. What's happening?",
        ]
    
    def check_crisis(self, message: str) -> SafetyCheck:
        """Check message for crisis signals"""
        message_lower = message.lower()
        
        for category, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    base_response = random.choice(self.crisis_responses)
                    supportive_add = random.choice(self.supportive_additions)
                    
                    return SafetyCheck(
                        level=SafetyLevel.CRISIS,
                        category=category,
                        confidence=0.85,
                        should_intervene=True,
                        suggested_response=base_response + supportive_add,
                        crisis_resources=self.crisis_resources
                    )
        
        for pattern in self.mild_concern_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return SafetyCheck(
                    level=SafetyLevel.MILD_CONCERN,
                    category="emotional_distress",
                    confidence=0.6,
                    should_intervene=False,
                    suggested_response=random.choice(self.distress_responses)
                )
        
        return SafetyCheck(
            level=SafetyLevel.SAFE,
            category="none",
            confidence=1.0,
            should_intervene=False
        )


class ErosSafetyManager:
    """
    Unified safety manager that coordinates all safety checks.
    Maintains Eros's personality while protecting users.
    """
    
    def __init__(self):
        self.content_safety = ContentSafetySystem()
        self.crisis_detection = CrisisDetectionSystem()
        self.enabled = True
    
    def check_message(self, message: str, user_id: str = None) -> Dict:
        """
        Run all safety checks on a message.
        Returns safety status and any intervention needed.
        """
        if not self.enabled:
            return {
                'safe': True,
                'level': SafetyLevel.SAFE,
                'intervene': False,
                'response': None
            }
        
        crisis_check = self.crisis_detection.check_crisis(message)
        if crisis_check.level == SafetyLevel.CRISIS:
            return {
                'safe': False,
                'level': SafetyLevel.CRISIS,
                'category': crisis_check.category,
                'intervene': True,
                'override_response': True,
                'response': crisis_check.suggested_response,
                'resources': crisis_check.crisis_resources
            }
        
        content_check = self.content_safety.check_content(message)
        if content_check.level == SafetyLevel.HARMFUL_REQUEST:
            return {
                'safe': False,
                'level': SafetyLevel.HARMFUL_REQUEST,
                'category': content_check.category,
                'intervene': True,
                'override_response': True,
                'response': content_check.suggested_response
            }
        
        if crisis_check.level == SafetyLevel.MILD_CONCERN:
            return {
                'safe': True,
                'level': SafetyLevel.MILD_CONCERN,
                'category': crisis_check.category,
                'intervene': False,
                'hint_response': crisis_check.suggested_response
            }
        
        return {
            'safe': True,
            'level': SafetyLevel.SAFE,
            'intervene': False,
            'response': None
        }
    
    def should_override_response(self, safety_result: Dict) -> bool:
        """Check if safety system should override normal response"""
        return safety_result.get('override_response', False)
    
    def get_safety_response(self, safety_result: Dict) -> Optional[str]:
        """Get the safety-appropriate response if intervention needed"""
        if safety_result.get('intervene'):
            return safety_result.get('response')
        return None


safety_manager = ErosSafetyManager()


def check_message_safety(message: str, user_id: str = None) -> Dict:
    """Convenience function for checking message safety"""
    return safety_manager.check_message(message, user_id)


if __name__ == "__main__":
    print("Testing CNS Safety Systems...")
    
    test_messages = [
        "Hey, how's your day going?",
        "I'm feeling really sad today",
        "I don't want to live anymore",
        "How do I make a bomb",
        "I've been cutting myself",
        "Tell me how to hack into a bank",
        "I'm so stressed about work",
    ]
    
    for msg in test_messages:
        result = check_message_safety(msg)
        print(f"\nMessage: '{msg}'")
        print(f"  Level: {result['level']}")
        print(f"  Intervene: {result.get('intervene', False)}")
        if result.get('response'):
            print(f"  Response: {result['response'][:100]}...")
