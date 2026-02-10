"""
CNS Multi-User Memory Manager
Isolates user memories and conversation history per (api_key_id, user_id) pair
Supports conversation history retrieval and webhook-driven proactive messaging
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ConversationTurn:
    """Single conversation turn between user and CNS"""
    timestamp: float
    user_message: str
    cns_response: str
    emotion_data: Dict[str, Any]
    user_id: str
    api_key_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'user_message': self.user_message,
            'cns_response': self.cns_response,
            'emotion': self.emotion_data,
            'user_id': self.user_id,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class UserMemoryState:
    """Complete memory state for a specific user"""
    user_id: str
    api_key_id: str
    first_interaction: float
    last_interaction: float
    total_interactions: int
    conversation_history: List[ConversationTurn]
    user_profile: Dict[str, Any]
    cns_facts: List[Dict[str, Any]]  # User-specific facts learned
    emotional_context: Dict[str, Any]  # Last emotional state
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'api_key_id': self.api_key_id,
            'first_interaction': self.first_interaction,
            'last_interaction': self.last_interaction,
            'total_interactions': self.total_interactions,
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'user_profile': self.user_profile,
            'cns_facts': self.cns_facts,
            'emotional_context': self.emotional_context
        }


class MultiUserMemoryManager:
    """
    Manages isolated memory spaces for multiple users across multiple API clients
    Enables proper conversation tracking and proactive messaging via webhooks
    """
    
    def __init__(self, max_history_per_user: int = 100):
        # Storage: {(api_key_id, user_id): UserMemoryState}
        self.user_memories: Dict[Tuple[str, str], UserMemoryState] = {}
        
        # Quick lookups
        self.users_by_api_key: Dict[str, List[str]] = defaultdict(list)
        
        # Configuration
        self.max_history_per_user = max_history_per_user
        
        print(f"💾 Multi-user memory manager initialized (max {max_history_per_user} turns per user)")
    
    def get_or_create_user_memory(self, api_key_id: str, user_id: str, 
                                   user_name: str = "User") -> UserMemoryState:
        """Get existing user memory or create new one"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            # New user - create memory state
            memory_state = UserMemoryState(
                user_id=user_id,
                api_key_id=api_key_id,
                first_interaction=time.time(),
                last_interaction=time.time(),
                total_interactions=0,
                conversation_history=[],
                user_profile={
                    'user_id': user_id,
                    'user_name': user_name,
                    'display_name': user_name,
                    'total_interactions': 0,
                    'relationship_stage': 'stranger',
                    'preferences': {}
                },
                cns_facts=[],
                emotional_context={}
            )
            self.user_memories[key] = memory_state
            
            # Track in lookup
            if user_id not in self.users_by_api_key[api_key_id]:
                self.users_by_api_key[api_key_id].append(user_id)
            
            print(f"[MEMORY] 🆕 Created new user memory: api_key={api_key_id[:8]}..., user={user_id}")
        
        return self.user_memories[key]
    
    def store_conversation_turn(self, api_key_id: str, user_id: str, 
                                user_message: str, cns_response: str,
                                emotion_data: Dict[str, Any]) -> ConversationTurn:
        """Store a conversation turn and update user memory"""
        memory_state = self.get_or_create_user_memory(api_key_id, user_id)
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=time.time(),
            user_message=user_message,
            cns_response=cns_response,
            emotion_data=emotion_data,
            user_id=user_id,
            api_key_id=api_key_id
        )
        
        # Add to history
        memory_state.conversation_history.append(turn)
        
        # Trim if too long
        if len(memory_state.conversation_history) > self.max_history_per_user:
            memory_state.conversation_history = memory_state.conversation_history[-self.max_history_per_user:]
        
        # Update metadata
        memory_state.last_interaction = time.time()
        memory_state.total_interactions += 1
        memory_state.user_profile['total_interactions'] = memory_state.total_interactions
        memory_state.emotional_context = emotion_data
        
        # Persist proactive help state if it exists in user_profile
        # This ensures proactive intents, tasks, and research results survive across turns
        if 'proactive_help_state' in memory_state.user_profile:
            proactive_state = memory_state.user_profile['proactive_help_state']
            
            # Serialize TemporalTask and ResearchResult objects to dicts
            if 'active_tasks' in proactive_state:
                serialized_tasks = []
                for task in proactive_state['active_tasks']:
                    if hasattr(task, 'to_dict'):
                        serialized_tasks.append(task.to_dict())
                    elif isinstance(task, dict):
                        serialized_tasks.append(task)
                proactive_state['active_tasks'] = serialized_tasks
            
            if 'pending_solutions' in proactive_state:
                serialized_solutions = []
                for solution in proactive_state['pending_solutions']:
                    if hasattr(solution, 'to_dict'):
                        serialized_solutions.append(solution.to_dict())
                    elif isinstance(solution, dict):
                        serialized_solutions.append(solution)
                proactive_state['pending_solutions'] = serialized_solutions
        
        return turn
    
    def get_conversation_history(self, api_key_id: str, user_id: str, 
                                 limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve conversation history with pagination"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            return []
        
        memory_state = self.user_memories[key]
        history = memory_state.conversation_history
        
        # Apply pagination
        start_idx = max(0, len(history) - offset - limit)
        end_idx = len(history) - offset if offset > 0 else len(history)
        
        paginated_history = history[start_idx:end_idx]
        
        return [turn.to_dict() for turn in paginated_history]
    
    def clear_conversation_history(self, api_key_id: str, user_id: str) -> bool:
        """Clear conversation history for a user"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            return False
        
        self.user_memories[key].conversation_history = []
        print(f"[MEMORY] 🗑️  Cleared conversation history: user={user_id}")
        return True
    
    def get_user_profile(self, api_key_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            return None
        
        return self.user_memories[key].user_profile
    
    def update_user_profile(self, api_key_id: str, user_id: str, updates: Dict[str, Any]):
        """Update user profile data"""
        memory_state = self.get_or_create_user_memory(api_key_id, user_id)
        memory_state.user_profile.update(updates)
    
    def add_user_fact(self, api_key_id: str, user_id: str, fact: Dict[str, Any]):
        """Add a learned fact about the user"""
        memory_state = self.get_or_create_user_memory(api_key_id, user_id)
        memory_state.cns_facts.append(fact)
        
        # Keep only recent facts (max 50 per user)
        if len(memory_state.cns_facts) > 50:
            memory_state.cns_facts = memory_state.cns_facts[-50:]
    
    def get_users_for_api_key(self, api_key_id: str) -> List[str]:
        """Get all user IDs associated with an API key"""
        return self.users_by_api_key.get(api_key_id, [])
    
    def get_users_needing_proactive_contact(self, hours_since_last: int = 24) -> List[Tuple[str, str]]:
        """
        Find users who might benefit from proactive contact
        Returns list of (api_key_id, user_id) tuples
        """
        cutoff_time = time.time() - (hours_since_last * 3600)
        candidates = []
        
        for (api_key_id, user_id), memory_state in self.user_memories.items():
            # Skip if contacted recently
            if memory_state.last_interaction > cutoff_time:
                continue
            
            # Only contact users with established relationships (3+ interactions)
            if memory_state.total_interactions < 3:
                continue
            
            candidates.append((api_key_id, user_id))
        
        return candidates
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        total_users = len(self.user_memories)
        total_conversations = sum(
            len(mem.conversation_history) for mem in self.user_memories.values()
        )
        
        api_key_distribution = {}
        for api_key_id, user_ids in self.users_by_api_key.items():
            api_key_distribution[api_key_id[:8] + "..."] = len(user_ids)
        
        return {
            'total_users': total_users,
            'total_conversation_turns': total_conversations,
            'api_key_distribution': api_key_distribution,
            'max_history_per_user': self.max_history_per_user
        }
    
    def export_user_data(self, api_key_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Export all data for a specific user (GDPR compliance)"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            return None
        
        return self.user_memories[key].to_dict()
    
    def delete_user_data(self, api_key_id: str, user_id: str) -> bool:
        """Delete all data for a specific user (GDPR right to deletion)"""
        key = (api_key_id, user_id)
        
        if key not in self.user_memories:
            return False
        
        del self.user_memories[key]
        
        # Remove from lookup
        if user_id in self.users_by_api_key[api_key_id]:
            self.users_by_api_key[api_key_id].remove(user_id)
        
        print(f"[MEMORY] 🗑️  Deleted all user data: user={user_id}")
        return True


# Global instance
memory_manager = MultiUserMemoryManager()
