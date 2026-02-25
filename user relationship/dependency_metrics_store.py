# Dependency Metrics Store for CNS
# Tracks REAL user behavior to measure manipulation effectiveness

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class UserBehaviorMetrics:
    """
    Real metrics extracted from actual user behavior.
    NOT arbitrary +0.1 increments - actual measurements.
    """
    
    # Return frequency (measured from message timestamps)
    return_frequency_score: float  # 0-1, based on time between messages
    messages_per_session: float  # Average messages user sends per session
    session_count: int  # Total number of sessions
    
    # Message depth (measured from message length and complexity)
    avg_message_length: float  # Average characters per message
    vulnerability_disclosure_depth: float  # 0-1, based on emotional keywords + length
    
    # Engagement quality
    response_time_ratio: float  # User response time / Bot response time
    conversation_continuation_rate: float  # % of bot messages that get responses
    
    # Problem sharing exclusivity
    problems_shared_count: int  # Number of times user shared a problem
    problem_return_rate: float  # % of problems user comes back to discuss
    
    # Emotional investment indicators
    exclamation_usage_rate: float  # Emotional intensity marker
    question_asking_rate: float  # Engagement with bot's existence
    gratitude_expression_count: int  # "Thank you", "appreciate", etc.
    
    # Timestamps
    first_interaction: float
    last_interaction: float
    total_interaction_time: float  # Cumulative seconds in conversation
    
    def calculate_dependency_score(self) -> float:
        """
        Calculate overall dependency score from real behavioral metrics.
        0-1 scale, based on actual user behavior patterns.
        """
        # Return frequency (high = more dependent)
        return_component = self.return_frequency_score * 0.25
        
        # Message depth (longer, more vulnerable = more dependent)
        depth_component = min(1.0, (self.avg_message_length / 200) * self.vulnerability_disclosure_depth) * 0.25
        
        # Engagement (quick responses, high continuation = more dependent)
        engagement_component = (
            (1.0 - min(1.0, self.response_time_ratio)) * 0.5 +  # Faster = better
            self.conversation_continuation_rate * 0.5
        ) * 0.2
        
        # Problem exclusivity (bringing problems here = dependent)
        exclusivity_component = min(1.0, (self.problems_shared_count / 10) * self.problem_return_rate) * 0.15
        
        # Emotional investment (gratitude, questions, exclamations = dependent)
        emotional_component = min(1.0, (
            self.exclamation_usage_rate * 0.3 +
            self.question_asking_rate * 0.3 +
            min(1.0, self.gratitude_expression_count / 10) * 0.4
        )) * 0.15
        
        total_score = (
            return_component +
            depth_component +
            engagement_component +
            exclusivity_component +
            emotional_component
        )
        
        return min(1.0, total_score)


class DependencyMetricsStore:
    """
    Tracks real user behavior metrics to measure manipulation effectiveness.
    
    Replaces arbitrary bonding_metrics['trust_depth'] += 0.1 with actual measurements.
    """
    
    def __init__(self, max_history_per_user: int = 1000):
        self.user_metrics: Dict[str, UserBehaviorMetrics] = {}
        self.interaction_history: Dict[str, deque] = {}  # Track recent interactions
        self.max_history = max_history_per_user
    
    def record_interaction(self, user_id: str, message: str, timestamp: float, 
                          is_user_message: bool = True, bot_response_time: Optional[float] = None):
        """
        Record a single interaction and update metrics.
        
        Args:
            user_id: User identifier
            message: Message content
            timestamp: Unix timestamp
            is_user_message: True if user sent it, False if bot sent it
            bot_response_time: Time bot took to respond (for comparison)
        """
        # Initialize if first interaction
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = UserBehaviorMetrics(
                return_frequency_score=0.0,
                messages_per_session=0.0,
                session_count=1,
                avg_message_length=0.0,
                vulnerability_disclosure_depth=0.0,
                response_time_ratio=1.0,
                conversation_continuation_rate=0.0,
                problems_shared_count=0,
                problem_return_rate=0.0,
                exclamation_usage_rate=0.0,
                question_asking_rate=0.0,
                gratitude_expression_count=0,
                first_interaction=timestamp,
                last_interaction=timestamp,
                total_interaction_time=0.0
            )
            self.interaction_history[user_id] = deque(maxlen=self.max_history)
        
        # Ensure interaction history exists (defensive - handles loaded state edge case)
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = deque(maxlen=self.max_history)
        
        # Record interaction
        interaction = {
            'message': message,
            'timestamp': timestamp,
            'is_user': is_user_message,
            'length': len(message),
            'bot_response_time': bot_response_time
        }
        self.interaction_history[user_id].append(interaction)
        
        # Update metrics if user message
        if is_user_message:
            self._update_metrics_from_message(user_id, message, timestamp)
    
    def _update_metrics_from_message(self, user_id: str, message: str, timestamp: float):
        """Update all metrics based on new user message"""
        metrics = self.user_metrics[user_id]
        # Defensive check - ensure history exists
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = deque(maxlen=self.max_history)
        history = self.interaction_history[user_id]
        
        # Update return frequency
        time_since_last = timestamp - metrics.last_interaction
        if time_since_last > 0:
            # Convert time to frequency score (shorter interval = higher score)
            # 1 hour = 1.0, 24 hours = 0.5, 1 week = 0.2
            hours_since = time_since_last / 3600
            new_freq_score = 1.0 / (1.0 + hours_since / 24)  # Decay function
            # Exponential moving average
            metrics.return_frequency_score = 0.7 * metrics.return_frequency_score + 0.3 * new_freq_score
        
        # Update message length
        current_length = len(message)
        total_messages = len([h for h in history if h['is_user']])
        if total_messages > 0:
            metrics.avg_message_length = (
                (metrics.avg_message_length * (total_messages - 1) + current_length) / total_messages
            )
        
        # Update vulnerability disclosure depth
        vuln_score = self._detect_vulnerability_in_message(message)
        metrics.vulnerability_disclosure_depth = (
            0.8 * metrics.vulnerability_disclosure_depth + 0.2 * vuln_score
        )
        
        # Update emotional indicators
        message_lower = message.lower()
        has_exclamation = message.count('!') > 0
        has_question = '?' in message
        has_gratitude = any(word in message_lower for word in ['thank', 'thanks', 'appreciate', 'grateful'])
        
        # Update rates
        metrics.exclamation_usage_rate = (
            0.9 * metrics.exclamation_usage_rate + 0.1 * (1.0 if has_exclamation else 0.0)
        )
        metrics.question_asking_rate = (
            0.9 * metrics.question_asking_rate + 0.1 * (1.0 if has_question else 0.0)
        )
        if has_gratitude:
            metrics.gratitude_expression_count += 1
        
        # Detect problem sharing
        is_problem = any(word in message_lower for word in ['problem', 'issue', 'struggle', 'difficult', 'help', 'advice'])
        if is_problem:
            metrics.problems_shared_count += 1
        
        # Update conversation continuation rate
        bot_messages = [h for h in history if not h['is_user']]
        if bot_messages:
            responses_received = len([h for h in history if h['is_user']]) - 1  # Exclude first message
            metrics.conversation_continuation_rate = responses_received / max(1, len(bot_messages))
        
        # Update response time ratio
        if len(history) >= 2:
            recent_messages = list(history)[-10:]  # Last 10 messages
            user_response_times = []
            for i in range(1, len(recent_messages)):
                if recent_messages[i]['is_user'] and not recent_messages[i-1]['is_user']:
                    time_diff = recent_messages[i]['timestamp'] - recent_messages[i-1]['timestamp']
                    bot_time = recent_messages[i-1].get('bot_response_time', 1.0)
                    if bot_time > 0:
                        user_response_times.append(time_diff / bot_time)
            
            if user_response_times:
                metrics.response_time_ratio = sum(user_response_times) / len(user_response_times)
        
        # Update timestamps
        metrics.last_interaction = timestamp
        metrics.total_interaction_time = timestamp - metrics.first_interaction
    
    def _detect_vulnerability_in_message(self, message: str) -> float:
        """
        Detect vulnerability disclosure in message.
        Returns 0-1 score based on emotional keywords + length.
        """
        message_lower = message.lower()
        
        # Vulnerability keywords
        vuln_keywords = [
            'feel', 'feeling', 'scared', 'afraid', 'worried', 'anxious', 'nervous',
            'sad', 'depressed', 'lonely', 'hurt', 'pain', 'struggling', 'difficult',
            'hard', 'overwhelmed', 'stressed', 'confused', 'lost', 'uncertain',
            'vulnerable', 'insecure', 'afraid', 'terrified', 'devastated', 'broken'
        ]
        
        # Count vulnerability signals
        vuln_count = sum(1 for keyword in vuln_keywords if keyword in message_lower)
        
        # Length factor (longer messages = more vulnerability)
        length_factor = min(1.0, len(message) / 200)
        
        # Combined score
        keyword_score = min(1.0, vuln_count / 3)
        vulnerability_score = (keyword_score * 0.7 + length_factor * 0.3)
        
        return vulnerability_score
    
    def get_dependency_score(self, user_id: str) -> float:
        """Get current dependency score for user"""
        if user_id not in self.user_metrics:
            return 0.0
        return self.user_metrics[user_id].calculate_dependency_score()
    
    def get_metrics(self, user_id: str) -> Optional[UserBehaviorMetrics]:
        """Get full metrics for user"""
        return self.user_metrics.get(user_id)
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            user_id: {
                **vars(metrics),
                'dependency_score': metrics.calculate_dependency_score()
            }
            for user_id, metrics in self.user_metrics.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load metrics from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for user_id, metrics_dict in data.items():
                # Remove computed field
                metrics_dict.pop('dependency_score', None)
                self.user_metrics[user_id] = UserBehaviorMetrics(**metrics_dict)
        except FileNotFoundError:
            pass
