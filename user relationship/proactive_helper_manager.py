"""
Proactive Helper Manager - Integrates all proactive intelligence systems
Manages user profiles with proactive help state
"""

import asyncio
from typing import Dict, List, Any, Optional
from user_intent_tracker import UserIntentTracker, UserIntent
from autonomous_research_agent import AutonomousResearchAgent, ResearchResult
from task_memory_system import TaskMemorySystem, TemporalTask


class ProactiveHelperManager:
    """
    Coordinates all proactive helper systems
    Manages state in user profiles
    """
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        # Initialize subsystems
        self.intent_tracker = UserIntentTracker(mistral_api_key)
        self.research_agent = AutonomousResearchAgent(mistral_api_key)
        self.task_memory = TaskMemorySystem()
        
        print("🤖 Proactive helper manager initialized - ultra-helpful mode active")
    
    def initialize_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize proactive help fields in user profile"""
        if 'proactive_help' not in user_profile:
            user_profile['proactive_help'] = {
                'help_acceptance_rate': 0.5,  # Neutral starting point
                'preferred_help_style': 'balanced',  # 'aggressive', 'balanced', 'gentle'
                'last_help_offered': 0.0,
                'total_help_offered': 0,
                'total_help_accepted': 0,
                'dismissed_topics': []  # Topics user doesn't want help with
            }
        
        return user_profile
    
    async def process_message(self, user_id: str, message: str, 
                             conversation_history: List[Dict[str, str]] = None,
                             user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process message for proactive help opportunities
        Returns dict with extracted intents, research triggers, tasks
        """
        
        results = {
            'intents_extracted': [],
            'research_triggered': False,
            'tasks_added': [],
            'should_research': []
        }
        
        # Extract intents from message
        intents = await self.intent_tracker.extract_intents(
            user_id, message, conversation_history
        )
        
        results['intents_extracted'] = [i.to_dict() for i in intents]
        
        # Extract temporal tasks
        temporal_info = self.task_memory.extract_temporal_info(message)
        for description, date_info, task_type in temporal_info:
            self.task_memory.add_task(user_id, description, date_info, task_type)
            results['tasks_added'].append({
                'description': description,
                'type': task_type,
                'date': date_info
            })
        
        # Identify high-priority problems for research
        high_priority = [i for i in intents if i.priority == 'high' and i.intent_type in ['problem', 'task']]
        
        if high_priority and user_profile:
            # Check user's help acceptance rate
            help_prefs = user_profile.get('proactive_help', {})
            acceptance_rate = help_prefs.get('help_acceptance_rate', 0.5)
            
            # Research if user is receptive (acceptance rate > 0.3)
            if acceptance_rate > 0.3:
                results['should_research'] = high_priority
                results['research_triggered'] = True
        
        return results
    
    async def trigger_research(self, user_id: str, intents: List[UserIntent],
                             user_context: str = "") -> List[ResearchResult]:
        """Trigger autonomous research for intents"""
        return await self.research_agent.bulk_research(user_id, intents)
    
    def get_proactive_context(self, user_id: str, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get complete proactive help context for user
        Used when generating autonomous messages
        """
        
        context = {
            'pending_solutions': [],
            'upcoming_tasks': [],
            'today_tasks': [],
            'active_problems': [],
            'should_welcome_back': False,
            'help_style': 'balanced'
        }
        
        # Get pending research solutions
        pending = self.research_agent.get_pending_solutions(user_id)
        context['pending_solutions'] = [r.to_dict() for r in pending]
        
        # Get upcoming tasks
        upcoming = self.task_memory.get_upcoming_tasks(user_id, days_ahead=3)
        context['upcoming_tasks'] = [t.to_dict() for t in upcoming]
        
        # Get today's tasks
        today = self.task_memory.get_today_tasks(user_id)
        context['today_tasks'] = [t.to_dict() for t in today]
        
        # Get active problems
        problems = self.intent_tracker.get_high_priority_problems(user_id)
        context['active_problems'] = [p.to_dict() for p in problems]
        
        # Check if should welcome back (user returning after time away)
        if user_profile:
            help_prefs = user_profile.get('proactive_help', {})
            context['help_style'] = help_prefs.get('preferred_help_style', 'balanced')
            
            # Welcome back if there's something helpful to offer
            has_help_to_offer = (
                len(pending) > 0 or 
                len(today) > 0 or
                len(upcoming) > 0
            )
            context['should_welcome_back'] = has_help_to_offer
        
        return context
    
    def update_help_response(self, user_id: str, user_profile: Dict[str, Any],
                            intent_description: str, response: str):
        """
        Update user profile based on response to offered help
        Learns preferences over time
        """
        if 'proactive_help' not in user_profile:
            self.initialize_user_profile(user_profile)
        
        help_prefs = user_profile['proactive_help']
        
        # Record response with research agent
        self.research_agent.record_user_response(user_id, intent_description, response)
        
        # Update acceptance tracking
        if response == 'accepted':
            help_prefs['total_help_accepted'] += 1
            help_prefs['total_help_offered'] += 1
        elif response in ['rejected', 'ignored']:
            help_prefs['total_help_offered'] += 1
        
        # Calculate new acceptance rate
        if help_prefs['total_help_offered'] > 0:
            help_prefs['help_acceptance_rate'] = (
                help_prefs['total_help_accepted'] / help_prefs['total_help_offered']
            )
        
        # Adjust help style based on acceptance rate
        rate = help_prefs['help_acceptance_rate']
        if rate > 0.7:
            help_prefs['preferred_help_style'] = 'aggressive'  # They love help
        elif rate > 0.4:
            help_prefs['preferred_help_style'] = 'balanced'
        else:
            help_prefs['preferred_help_style'] = 'gentle'  # Less pushy
        
        print(f"📊 Help acceptance rate: {rate:.2f} → style: {help_prefs['preferred_help_style']}")
    
    def mark_solution_offered(self, user_id: str, intent_description: str, channel: str = 'conversation'):
        """Mark that a solution was offered to user via specific channel"""
        self.research_agent.mark_offered(user_id, intent_description, channel)
    
    def is_conversation_active(self, user_profile: Dict[str, Any], activity_window_seconds: int = 300) -> bool:
        """
        Check if conversation is still active (< 5 minutes since last message)
        """
        last_message_time = user_profile.get('last_message_time', 0)
        if last_message_time == 0:
            return False
        
        import time
        time_since_message = time.time() - last_message_time
        return time_since_message < activity_window_seconds
    
    def get_fresh_solutions_for_conversation(self, user_id: str, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get fresh solutions for active conversation injection
        Returns solutions completed since user started chatting that haven't been offered yet
        """
        # Get last check time (or start of current session)
        last_check = user_profile.get('last_solution_check', 0)
        
        # Update check time
        import time
        user_profile['last_solution_check'] = time.time()
        
        # Get fresh solutions
        fresh = self.research_agent.get_fresh_solutions(user_id, last_check)
        
        return [r.to_dict() for r in fresh]
    
    def get_help_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of proactive help state"""
        return {
            'intents': self.intent_tracker.get_user_summary(user_id),
            'pending_solutions': len(self.research_agent.get_pending_solutions(user_id)),
            'upcoming_tasks': len(self.task_memory.get_upcoming_tasks(user_id)),
            'today_tasks': len(self.task_memory.get_today_tasks(user_id))
        }
