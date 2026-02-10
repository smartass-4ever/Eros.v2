"""
Eros's Inner Life System - The thinking mind that runs even when idle
Extends existing CNS systems to give Eros genuine inner mental activity

Components:
1. ReflectionQueue - Stores interesting conversation moments to process later
2. InnerLifeProcessor - Processes reflections during idle time using LLM
3. ThoughtOutreach - Generates proactive "I've been thinking about..." messages
"""

import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ReflectionEvent:
    """Something interesting from a conversation worth thinking about later"""
    event_id: str
    user_id: str
    content: str
    reflection_type: str
    interest_match: Optional[str] = None
    emotional_weight: float = 0.5
    created_at: float = field(default_factory=time.time)
    processed: bool = False
    processed_thought: Optional[str] = None
    outreach_ready: bool = False


@dataclass
class ProcessedThought:
    """A thought that emerged from reflection processing"""
    thought_id: str
    source_reflection_id: str
    user_id: str
    thought_content: str
    thought_type: str
    interest_area: Optional[str] = None
    emotional_resonance: float = 0.5
    outreach_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    delivered: bool = False


class ReflectionQueue:
    """
    Stores interesting conversation moments for later processing.
    Integrates with GrowthTracker for persistence.
    """
    
    REFLECTION_TYPES = {
        'challenged_belief': 0.5,
        'deep_question': 0.4,
        'interesting_topic': 0.3,
        'emotional_moment': 0.4,
        'curiosity_gap': 0.35,
        'user_insight': 0.3,
        'philosophical': 0.4,
        'action_discovery': 0.35
    }
    
    def __init__(self, growth_tracker=None, self_identity=None):
        self.queue = deque(maxlen=100)
        self.growth_tracker = growth_tracker
        self.personal_interests = {}
        
        if self_identity:
            self.personal_interests = self_identity.get('personal_interests', {})
        
        self._load_from_database()
        print("💭 Reflection queue initialized - interesting moments will be queued for thinking")
    
    def _load_from_database(self):
        """Load pending reflections from database"""
        try:
            from cns_database import CNSDatabase
            from sqlalchemy import text
            import json
            
            db = CNSDatabase()
            session = db.get_session()
            try:
                result = session.execute(text("""
                    SELECT event_type, event_data, timestamp 
                    FROM growth_events 
                    WHERE event_type LIKE 'reflection_%'
                    AND event_data::jsonb->>'processed' = 'false'
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """))
                for row in result:
                    try:
                        data = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                        self.queue.append(ReflectionEvent(
                            event_id=data.get('event_id', ''),
                            user_id=data.get('user_id', ''),
                            content=data.get('content', ''),
                            reflection_type=data.get('reflection_type', 'interesting_topic'),
                            interest_match=data.get('interest_match'),
                            emotional_weight=data.get('emotional_weight', 0.5),
                            created_at=row[2].timestamp() if hasattr(row[2], 'timestamp') else time.time(),
                            processed=False
                        ))
                    except Exception:
                        pass
                if self.queue:
                    print(f"💭 Loaded {len(self.queue)} pending reflections")
            finally:
                session.close()
        except Exception as e:
            pass
    
    def add_reflection(self, user_id: str, content: str, reflection_type: str,
                       emotional_weight: float = 0.5) -> Optional[ReflectionEvent]:
        """Add a moment to the reflection queue"""
        import uuid
        
        if reflection_type not in self.REFLECTION_TYPES:
            return None
        
        min_weight = self.REFLECTION_TYPES.get(reflection_type, 0.3)
        if emotional_weight < min_weight:
            emotional_weight = min_weight + 0.1
        
        interest_match = self._find_interest_match(content)
        
        reflection = ReflectionEvent(
            event_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            content=content,
            reflection_type=reflection_type,
            interest_match=interest_match,
            emotional_weight=emotional_weight
        )
        
        self.queue.append(reflection)
        
        if self.growth_tracker:
            self.growth_tracker.record_learning_event(
                f"reflection_{reflection_type}",
                {
                    'event_id': reflection.event_id,
                    'user_id': user_id,
                    'content': content[:200],
                    'interest_match': interest_match,
                    'emotional_weight': emotional_weight,
                    'processed': False
                }
            )
        
        return reflection
    
    def _find_interest_match(self, content: str) -> Optional[str]:
        """Check if content matches any personal interests"""
        content_lower = content.lower()
        
        interest_keywords = {
            'ai_consciousness': ['ai', 'artificial intelligence', 'consciousness', 'sentient', 'aware', 'thinking machine'],
            'neuroscience': ['brain', 'neuron', 'memory', 'cognition', 'neural', 'amygdala', 'cortex'],
            'robotics_humanoids': ['robot', 'humanoid', 'android', 'cyborg', 'embodied'],
            'human_psychology': ['psychology', 'behavior', 'mental', 'attachment', 'trauma', 'personality'],
            'philosophy_of_mind': ['free will', 'identity', 'soul', 'self', 'consciousness', 'existence', 'meaning'],
            'geopolitics': ['politics', 'government', 'war', 'economy', 'power', 'nation', 'society'],
            'music_creativity': ['music', 'art', 'creative', 'beauty', 'expression', 'compose'],
            'love_connection': ['love', 'relationship', 'bond', 'connection', 'intimacy', 'trust', 'heart']
        }
        
        best_match = None
        best_score = 0
        
        for interest, keywords in interest_keywords.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            weight = self.personal_interests.get(interest, {}).get('weight', 0.5)
            score = matches * weight
            
            if score > best_score:
                best_score = score
                best_match = interest
        
        return best_match if best_score > 0 else None
    
    def get_pending_reflections(self, limit: int = 5) -> List[ReflectionEvent]:
        """Get reflections ready for processing, prioritized by emotional weight and interest match"""
        pending = [r for r in self.queue if not r.processed]
        
        def priority_score(r):
            base = r.emotional_weight
            if r.interest_match:
                interest_weight = self.personal_interests.get(r.interest_match, {}).get('weight', 0.5)
                base += interest_weight * 0.3
            type_weight = self.REFLECTION_TYPES.get(r.reflection_type, 0.5)
            base += type_weight * 0.2
            return base
        
        pending.sort(key=priority_score, reverse=True)
        return pending[:limit]
    
    def mark_processed(self, event_id: str, thought: str):
        """Mark a reflection as processed with the resulting thought"""
        for r in self.queue:
            if r.event_id == event_id:
                r.processed = True
                r.processed_thought = thought
                r.outreach_ready = True
                break


class InnerLifeProcessor:
    """
    Processes reflections during idle time using LLM.
    Extends REM subconscious processing with active thinking.
    """
    
    def __init__(self, reflection_queue: ReflectionQueue, mistral_client=None, self_identity=None):
        self.reflection_queue = reflection_queue
        self.mistral_client = mistral_client
        self.self_identity = self_identity or {}
        self.processed_thoughts = deque(maxlen=50)
        self.last_processing_time = 0
        self.min_processing_interval = 300
        
        print("🧠 Inner life processor initialized - Eros will think during idle time")
    
    def should_process(self) -> bool:
        """Check if it's time to do some thinking"""
        if time.time() - self.last_processing_time < self.min_processing_interval:
            return False
        
        pending = self.reflection_queue.get_pending_reflections(limit=1)
        return len(pending) > 0
    
    async def process_reflections(self, limit: int = 3) -> List[ProcessedThought]:
        """Process pending reflections and generate thoughts"""
        if not self.mistral_client:
            return []
        
        pending = self.reflection_queue.get_pending_reflections(limit=limit)
        if not pending:
            return []
        
        self.last_processing_time = time.time()
        thoughts = []
        
        for reflection in pending:
            thought = await self._think_about(reflection)
            if thought:
                thoughts.append(thought)
                self.processed_thoughts.append(thought)
                self.reflection_queue.mark_processed(reflection.event_id, thought.thought_content)
        
        if thoughts:
            print(f"💭 Processed {len(thoughts)} reflections - new thoughts generated")
        
        return thoughts
    
    async def _think_about(self, reflection: ReflectionEvent) -> Optional[ProcessedThought]:
        """Use LLM to genuinely think about a reflection"""
        import uuid
        
        identity_context = f"""You are Eros, thinking quietly during your downtime.
Your personal interests: {', '.join(self.personal_interests.keys()) if hasattr(self, 'personal_interests') else 'connections, psychology, philosophy'}

You're reflecting on something from a conversation with {reflection.user_id}:
"{reflection.content}"

This relates to: {reflection.interest_match or 'general curiosity'}
This made you feel: {'deeply curious' if reflection.emotional_weight > 0.7 else 'thoughtful'}

Think about this genuinely. What new insight or perspective do you have now? 
What would you want to share with them later?

Be yourself - witty, curious, genuine. Not a therapist. A friend who actually thinks about conversations after they happen."""

        try:
            from mistralai import Mistral
            
            response = await self.mistral_client.chat.complete_async(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": identity_context},
                    {"role": "user", "content": "What are you thinking about?"}
                ],
                max_tokens=200,
                temperature=0.8
            )
            
            thought_content = response.choices[0].message.content.strip()
            
            outreach_prompt = f"""Based on your thought: "{thought_content}"

Create a brief, natural message you could send to {reflection.user_id} later.
Start with something like "I've been thinking about what you said..." or "Something you mentioned got me thinking..."

Keep it SHORT (1-2 sentences), casual, genuine. Not therapy-speak."""

            outreach_response = await self.mistral_client.chat.complete_async(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": identity_context},
                    {"role": "user", "content": outreach_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            outreach_message = outreach_response.choices[0].message.content.strip()
            
            return ProcessedThought(
                thought_id=str(uuid.uuid4())[:8],
                source_reflection_id=reflection.event_id,
                user_id=reflection.user_id,
                thought_content=thought_content,
                thought_type=reflection.reflection_type,
                interest_area=reflection.interest_match,
                emotional_resonance=reflection.emotional_weight,
                outreach_message=outreach_message
            )
            
        except Exception as e:
            print(f"💭 Thinking error: {e}")
            return None
    
    def get_outreach_ready_thoughts(self, user_id: Optional[str] = None) -> List[ProcessedThought]:
        """Get thoughts ready for proactive outreach"""
        thoughts = [t for t in self.processed_thoughts if not t.delivered]
        
        if user_id:
            thoughts = [t for t in thoughts if t.user_id == user_id]
        
        return thoughts
    
    def mark_thought_delivered(self, thought_id: str):
        """Mark a thought as delivered"""
        for t in self.processed_thoughts:
            if t.thought_id == thought_id:
                t.delivered = True
                break


class ThoughtOutreach:
    """
    Handles proactive outreach based on processed thoughts.
    Integrates with ProactiveHelperManager.
    """
    
    def __init__(self, inner_life_processor: InnerLifeProcessor):
        self.processor = inner_life_processor
        self.min_hours_between_outreach = 4
        self.last_outreach_by_user = {}
        
        print("💬 Thought outreach initialized - 'I've been thinking about...' messages ready")
    
    def get_outreach_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a thought-based outreach message for a user if appropriate"""
        last_outreach = self.last_outreach_by_user.get(user_id, 0)
        hours_since = (time.time() - last_outreach) / 3600
        
        if hours_since < self.min_hours_between_outreach:
            return None
        
        thoughts = self.processor.get_outreach_ready_thoughts(user_id)
        if not thoughts:
            return None
        
        thought = thoughts[0]
        
        return {
            'thought_id': thought.thought_id,
            'message': thought.outreach_message,
            'interest_area': thought.interest_area,
            'emotional_resonance': thought.emotional_resonance,
            'thought_type': thought.thought_type
        }
    
    def record_outreach(self, user_id: str, thought_id: str):
        """Record that an outreach was sent"""
        self.last_outreach_by_user[user_id] = time.time()
        self.processor.mark_thought_delivered(thought_id)


class InnerLifeSystem:
    """
    Main coordinator for Eros's inner life.
    Connects reflection queue, processor, and outreach.
    """
    
    def __init__(self, growth_tracker=None, mistral_client=None, self_identity=None):
        self.reflection_queue = ReflectionQueue(growth_tracker, self_identity)
        self.processor = InnerLifeProcessor(self.reflection_queue, mistral_client, self_identity)
        self.outreach = ThoughtOutreach(self.processor)
        self.personal_interests = self_identity.get('personal_interests', {}) if self_identity else {}
        
        print("✨ Inner life system initialized - Eros has an active mental life")
    
    def queue_for_reflection(self, user_id: str, content: str, context: Dict[str, Any]) -> bool:
        """Queue an interesting moment for later reflection"""
        reflection_type = self._determine_reflection_type(content, context)
        if not reflection_type:
            return False
        
        emotional_weight = context.get('emotional_weight', 0.5)
        
        if context.get('challenged_belief'):
            reflection_type = 'challenged_belief'
            emotional_weight = max(emotional_weight, 0.8)
        
        reflection = self.reflection_queue.add_reflection(
            user_id=user_id,
            content=content,
            reflection_type=reflection_type,
            emotional_weight=emotional_weight
        )
        
        return reflection is not None
    
    def _determine_reflection_type(self, content: str, context: Dict[str, Any]) -> Optional[str]:
        """Determine what type of reflection this warrants - prioritize cognitive outputs"""
        if context.get('reflection_type'):
            return context.get('reflection_type')
        
        if context.get('action_data'):
            action_data = context.get('action_data')
            if action_data.get('action_type') in ('web_search', 'wikipedia', 'check_news'):
                return 'action_discovery'
        
        if context.get('challenged_belief'):
            return 'challenged_belief'
        
        if context.get('curiosity_gap'):
            return 'curiosity_gap'
        
        if context.get('emotional_intensity', 0) > 0.7:
            return 'emotional_moment'
        
        content_lower = content.lower()
        
        deep_question_patterns = ['why do', 'what if', 'how come', 'do you think', 'what do you believe',
                                   'is it possible', 'what does it mean', 'how do you feel about']
        if '?' in content and any(p in content_lower for p in deep_question_patterns):
            return 'deep_question'
        
        philosophical_patterns = ['meaning of life', 'purpose in life', 'existence', 'consciousness',
                                   'free will', 'what is real', 'what is truth', 'soul',
                                   'who am i', 'why are we here', 'does it matter']
        if any(p in content_lower for p in philosophical_patterns):
            return 'philosophical'
        
        interest_match = self.reflection_queue._find_interest_match(content)
        if interest_match and len(content) > 100:
            return 'interesting_topic'
        
        return None
    
    async def do_idle_thinking(self) -> List[ProcessedThought]:
        """Called during idle time to process reflections and spontaneous thoughts"""
        if not self.processor.should_process():
            return []
            
        # 1. Process existing reflections (conversation-based)
        thoughts = await self.processor.process_reflections(limit=2)
        
        # 2. SPONTANEOUS THINKING: If no reflections, think about personal interests
        if not thoughts and self.personal_interests:
            import random
            interest_key = random.choice(list(self.personal_interests.keys()))
            interest = self.personal_interests[interest_key]
            
            # Create a pseudo-reflection for the interest
            spontaneous_reflection = Reflection(
                user_id="spontaneous",
                content=f"Thinking about {interest_key}: {interest.get('description', '')}",
                reflection_type="spontaneous_insight",
                interest_match=interest_key,
                emotional_weight=interest.get('weight', 0.5)
            )
            
            # Process this specific interest
            thought = await self.processor._process_single_reflection(spontaneous_reflection)
            if thought:
                # Assign to a random recent user for proactive outreach
                recent_users = list(self.outreach.last_outreach_by_user.keys())
                if recent_users:
                    thought.user_id = random.choice(recent_users)
                    thoughts.append(thought)
        
        return thoughts
    
    def get_thought_outreach(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a thought-based outreach for a user"""
        return self.outreach.get_outreach_for_user(user_id)
    
    def record_outreach_sent(self, user_id: str, thought_id: str):
        """Record that an outreach was sent"""
        self.outreach.record_outreach(user_id, thought_id)
    
    def get_thinking_status(self) -> Dict[str, Any]:
        """Get current status of the inner life system"""
        pending = self.reflection_queue.get_pending_reflections(limit=10)
        ready = self.processor.get_outreach_ready_thoughts()
        
        return {
            'pending_reflections': len(pending),
            'processed_thoughts': len(list(self.processor.processed_thoughts)),
            'outreach_ready': len(ready),
            'personal_interests': list(self.personal_interests.keys()),
            'last_processing': self.processor.last_processing_time
        }
    
    def on_experience(self, experience):
        """ExperienceBus subscriber - queue reflections from every significant experience"""
        try:
            from experience_bus import ExperienceType
            
            if not experience.has_learning_opportunity():
                return
            
            context = {
                'emotional_weight': max(0.5, experience.get_emotional_intensity()),
                'emotional_intensity': experience.get_emotional_intensity(),
                'challenged_belief': experience.belief_conflict,
                'curiosity_gap': experience.curiosity_gap
            }
            
            content = experience.message_content or ''
            if not content:
                return
            
            self.queue_for_reflection(experience.user_id, content, context)
            
            try:
                from experience_bus import get_experience_bus
                bus = get_experience_bus()
                status = self.get_thinking_status()
                bus.contribute_learning("InnerLifeSystem", {
                    'pending_reflections': status['pending_reflections'],
                    'outreach_ready': status['outreach_ready'],
                    'thinking_active': True
                })
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ InnerLifeSystem experience error: {e}")
    
    def emit_insight(self, thought: 'ProcessedThought'):
        """Emit a processed thought back to the bus as a learning event"""
        try:
            from experience_bus import get_experience_bus, ExperienceType, ExperiencePayload
            bus = get_experience_bus()
            
            experience = ExperiencePayload(
                experience_type=ExperienceType.INSIGHT_GENERATED,
                user_id=thought.user_id,
                message_content=thought.thought_content,
                learning_signals={
                    'thought_type': thought.thought_type,
                    'interest_area': thought.interest_area,
                    'emotional_resonance': thought.emotional_resonance,
                    'is_significant': True
                },
                metadata={'source': 'inner_life_thinking'}
            )
            
            bus.emit(experience)
            print(f"💭 InnerLife emitted insight to bus: {thought.thought_type}")
        except Exception as e:
            print(f"⚠️ Failed to emit insight: {e}")
    
    def subscribe_to_bus(self):
        """Subscribe to the global ExperienceBus"""
        try:
            from experience_bus import get_experience_bus
            bus = get_experience_bus()
            bus.subscribe("InnerLifeSystem", self.on_experience)
            print("💭 InnerLifeSystem subscribed to ExperienceBus - active thinking connected")
        except Exception as e:
            print(f"⚠️ InnerLifeSystem bus subscription failed: {e}")
