"""
Xiaoice-Style Relationship Memory System
Builds intimate personal knowledge and relationship arcs over time
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta

@dataclass
class PersonalDetail:
    """Individual piece of personal information about the user"""
    detail: str
    category: str  # family, interests, dreams, fears, work, relationships
    confidence: float
    first_mentioned: float
    last_referenced: float
    emotional_weight: float  # How emotionally significant this is
    
@dataclass
class RelationshipMilestone:
    """Key moments in the relationship development"""
    milestone_type: str  # first_laugh, deep_share, trust_moment, vulnerable_moment
    description: str
    timestamp: float
    emotional_impact: float

@dataclass
class RelationshipArc:
    """The overall story and development of the relationship"""
    relationship_stage: str  # stranger, acquaintance, friend, close_friend, intimate_companion
    depth_score: float
    trust_level: float
    intimacy_level: float
    shared_experiences: List[str]
    inside_jokes: List[str]
    recurring_themes: List[str]
    emotional_patterns: Dict[str, float]

class XiaoiceRelationshipMemory:
    """Deep relationship memory system - 'she knows me better than anyone'"""
    
    def __init__(self):
        self.personal_details = defaultdict(list)  # user_id -> List[PersonalDetail]
        self.relationship_arcs = defaultdict(lambda: RelationshipArc(
            relationship_stage="stranger",
            depth_score=0.0,
            trust_level=0.0,
            intimacy_level=0.0,
            shared_experiences=[],
            inside_jokes=[],
            recurring_themes=[],
            emotional_patterns=defaultdict(float)
        ))
        self.relationship_milestones = defaultdict(list)  # user_id -> List[RelationshipMilestone]
        self.conversation_patterns = defaultdict(list)  # user_id -> conversation style patterns
        
        # Relationship development tracking
        self.interaction_counts = defaultdict(int)
        self.last_interaction = defaultdict(float)
        self.relationship_momentum = defaultdict(float)
        
    def extract_personal_details(self, user_id: str, message: str, emotional_context: Dict) -> List[PersonalDetail]:
        """Extract and store personal information with emotional weight"""
        
        details = []
        message_lower = message.lower()
        current_time = time.time()
        
        # Family details
        family_patterns = {
            "my mom": "family",
            "my dad": "family", 
            "my sister": "family",
            "my brother": "family",
            "my parents": "family",
            "my family": "family",
            "my wife": "family",
            "my husband": "family",
            "my kids": "family",
            "my daughter": "family",
            "my son": "family"
        }
        
        # Interest/hobby patterns
        interest_patterns = {
            "i love": "interests",
            "i enjoy": "interests", 
            "i'm into": "interests",
            "i like": "interests",
            "my hobby": "interests",
            "i collect": "interests",
            "i play": "interests"
        }
        
        # Dream/goal patterns  
        dream_patterns = {
            "i want to": "dreams",
            "i hope to": "dreams",
            "my dream": "dreams",
            "someday i": "dreams",
            "i wish": "dreams",
            "my goal": "dreams"
        }
        
        # Fear/worry patterns
        fear_patterns = {
            "i'm scared": "fears",
            "i worry": "fears", 
            "i'm afraid": "fears",
            "what if": "fears",
            "i hate when": "fears"
        }
        
        # Work/career patterns
        work_patterns = {
            "i work": "work",
            "my job": "work",
            "at work": "work", 
            "my boss": "work",
            "my career": "work",
            "my coworker": "work"
        }
        
        all_patterns = {
            **family_patterns,
            **interest_patterns, 
            **dream_patterns,
            **fear_patterns,
            **work_patterns
        }
        
        # Extract details
        for pattern, category in all_patterns.items():
            if pattern in message_lower:
                # Find the full context around the pattern
                start_idx = message_lower.find(pattern)
                # Get surrounding context (up to sentence end)
                context_end = message.find('.', start_idx)
                if context_end == -1:
                    context_end = len(message)
                
                detail_text = message[start_idx:context_end].strip()
                
                # Calculate emotional weight
                emotional_weight = emotional_context.get('intensity', 0.3)
                if category in ['fears', 'dreams']:
                    emotional_weight += 0.3  # Higher weight for vulnerable shares
                
                detail = PersonalDetail(
                    detail=detail_text,
                    category=category,
                    confidence=0.8,
                    first_mentioned=current_time,
                    last_referenced=current_time,
                    emotional_weight=emotional_weight
                )
                
                details.append(detail)
        
        # Store the details
        for detail in details:
            self.personal_details[user_id].append(detail)
        
        return details
    
    def build_relationship_arc(self, user_id: str, conversation_history: List[str], 
                             emotional_history: List[Dict]) -> RelationshipArc:
        """Build the overarching relationship story and development"""
        
        arc = self.relationship_arcs[user_id]
        interaction_count = self.interaction_counts[user_id]
        
        # Calculate relationship depth based on various factors
        personal_sharing_depth = len(self.personal_details[user_id]) * 0.1
        conversation_frequency = min(1.0, interaction_count * 0.02)
        emotional_vulnerability = self._calculate_vulnerability_score(emotional_history)
        time_together = self._calculate_relationship_duration(user_id)
        
        # Update depth score
        new_depth = min(1.0, 
            personal_sharing_depth * 0.3 +
            conversation_frequency * 0.2 + 
            emotional_vulnerability * 0.3 +
            time_together * 0.2
        )
        arc.depth_score = new_depth
        
        # Update relationship stage based on depth
        if new_depth < 0.2:
            arc.relationship_stage = "stranger"
        elif new_depth < 0.4:
            arc.relationship_stage = "acquaintance" 
        elif new_depth < 0.6:
            arc.relationship_stage = "friend"
        elif new_depth < 0.8:
            arc.relationship_stage = "close_friend"
        else:
            arc.relationship_stage = "intimate_companion"
        
        # Update trust and intimacy levels
        arc.trust_level = min(1.0, arc.trust_level + 0.05)  # Gradual trust building
        arc.intimacy_level = min(1.0, emotional_vulnerability * 0.7 + personal_sharing_depth * 0.3)
        
        # Track shared experiences and themes
        self._extract_shared_experiences(user_id, conversation_history)
        self._identify_recurring_themes(user_id, conversation_history)
        
        self.relationship_arcs[user_id] = arc
        return arc
    
    def generate_relationship_callback(self, user_id: str, current_message: str) -> Optional[str]:
        """Generate natural references to shared history - 'remembers everything about you'"""
        
        arc = self.relationship_arcs[user_id]
        details = self.personal_details[user_id]
        
        # Only generate callbacks for established relationships
        if arc.depth_score < 0.3:
            return None
            
        current_lower = current_message.lower()
        
        # Find relevant personal details to reference
        relevant_details = []
        for detail in details:
            # Check if current message relates to stored detail
            detail_words = set(detail.detail.lower().split())
            message_words = set(current_lower.split())
            
            # If there's overlap in keywords
            if detail_words.intersection(message_words):
                relevant_details.append(detail)
        
        if not relevant_details:
            return None
        
        # Select the most emotionally significant detail
        most_significant = max(relevant_details, key=lambda d: d.emotional_weight)
        
        # Generate natural callback based on relationship stage
        if arc.relationship_stage in ["close_friend", "intimate_companion"]:
            callbacks = [
                f"This reminds me of when you told me about {most_significant.detail.lower()}. ",
                f"You know, thinking back to what you shared about {most_significant.detail.lower()}, ",
                f"I keep thinking about what you said about {most_significant.detail.lower()}, and "
            ]
        else:
            callbacks = [
                f"I remember you mentioned {most_significant.detail.lower()}. ",
                f"Didn't you tell me about {most_significant.detail.lower()}? "
            ]
        
        return random.choice(callbacks) if callbacks else None
    
    def track_relationship_milestone(self, user_id: str, milestone_type: str, 
                                   description: str, emotional_impact: float):
        """Track significant moments in relationship development"""
        
        milestone = RelationshipMilestone(
            milestone_type=milestone_type,
            description=description, 
            timestamp=time.time(),
            emotional_impact=emotional_impact
        )
        
        self.relationship_milestones[user_id].append(milestone)
        
        # Update relationship momentum based on milestone
        if milestone_type in ["deep_share", "vulnerable_moment", "trust_moment"]:
            self.relationship_momentum[user_id] += emotional_impact * 0.5
    
    def generate_relationship_summary(self, user_id: str) -> Dict[str, Any]:
        """Generate a comprehensive relationship summary - what makes each user unique"""
        
        arc = self.relationship_arcs[user_id]
        details = self.personal_details[user_id]
        milestones = self.relationship_milestones[user_id]
        
        # Group personal details by category
        categorized_details = defaultdict(list)
        for detail in details:
            categorized_details[detail.category].append(detail.detail)
        
        # Most significant milestones
        significant_milestones = sorted(milestones, key=lambda m: m.emotional_impact, reverse=True)[:5]
        
        return {
            "relationship_stage": arc.relationship_stage,
            "depth_score": arc.depth_score,
            "trust_level": arc.trust_level, 
            "intimacy_level": arc.intimacy_level,
            "total_interactions": self.interaction_counts[user_id],
            "personal_details": dict(categorized_details),
            "key_milestones": [asdict(m) for m in significant_milestones],
            "shared_experiences": arc.shared_experiences,
            "inside_jokes": arc.inside_jokes,
            "recurring_themes": arc.recurring_themes,
            "what_makes_them_special": self._generate_uniqueness_summary(user_id)
        }
    
    def _calculate_vulnerability_score(self, emotional_history: List[Dict]) -> float:
        """Calculate how vulnerable/open the user has been"""
        
        if not emotional_history:
            return 0.0
            
        vulnerability_emotions = ["sad", "scared", "anxious", "lonely", "hurt"]
        vulnerability_score = 0.0
        
        for emotion_data in emotional_history[-20:]:  # Last 20 interactions
            emotion = emotion_data.get('emotion', 'neutral')
            intensity = emotion_data.get('intensity', 0.0)
            
            if emotion in vulnerability_emotions:
                vulnerability_score += intensity * 0.1
        
        return min(1.0, vulnerability_score)
    
    def _calculate_relationship_duration(self, user_id: str) -> float:
        """Calculate relationship duration factor"""
        
        if not self.personal_details[user_id]:
            return 0.0
            
        first_detail = min(self.personal_details[user_id], key=lambda d: d.first_mentioned)
        duration_days = (time.time() - first_detail.first_mentioned) / (24 * 3600)
        
        return min(1.0, duration_days * 0.02)  # Max score after ~50 days
    
    def _extract_shared_experiences(self, user_id: str, conversation_history: List[str]):
        """Extract shared experiences and moments"""
        
        arc = self.relationship_arcs[user_id]
        experience_indicators = [
            "we talked about", "remember when", "that time", "our conversation",
            "we discussed", "we both", "together we"
        ]
        
        for message in conversation_history[-10:]:
            message_lower = message.lower()
            for indicator in experience_indicators:
                if indicator in message_lower:
                    # Extract the experience context
                    start_idx = message_lower.find(indicator)
                    experience = message[start_idx:start_idx+100]
                    if experience not in arc.shared_experiences:
                        arc.shared_experiences.append(experience)
    
    def _identify_recurring_themes(self, user_id: str, conversation_history: List[str]):
        """Identify topics the user frequently discusses"""
        
        arc = self.relationship_arcs[user_id]
        word_frequency = defaultdict(int)
        
        for message in conversation_history:
            words = message.lower().split()
            meaningful_words = [w for w in words if len(w) > 4 and w.isalpha()]
            for word in meaningful_words:
                word_frequency[word] += 1
        
        # Find most frequent themes (minimum 3 mentions)
        themes = [word for word, count in word_frequency.items() if count >= 3]
        arc.recurring_themes = themes[:10]  # Top 10 themes
    
    def _generate_uniqueness_summary(self, user_id: str) -> str:
        """Generate what makes this person uniquely them"""
        
        details = self.personal_details[user_id]
        arc = self.relationship_arcs[user_id]
        
        if not details:
            return "Someone I'm just getting to know"
        
        # Find most emotionally significant details
        significant_details = sorted(details, key=lambda d: d.emotional_weight, reverse=True)[:3]
        
        uniqueness_parts = []
        for detail in significant_details:
            category_descriptions = {
                "family": "deeply values family",
                "dreams": "has beautiful dreams about",
                "fears": "courageously shares vulnerabilities about", 
                "interests": "lights up when talking about",
                "work": "finds meaning in their work with"
            }
            
            description = category_descriptions.get(detail.category, "cares deeply about")
            uniqueness_parts.append(f"{description} {detail.detail.lower()}")
        
        return "; ".join(uniqueness_parts) if uniqueness_parts else "A unique person I'm learning about"
    
    def update_interaction_tracking(self, user_id: str):
        """Update interaction statistics"""
        
        self.interaction_counts[user_id] += 1
        self.last_interaction[user_id] = time.time()
        
        # Relationship momentum decays over time but builds with interaction
        time_since_last = time.time() - self.last_interaction.get(user_id, time.time())
        decay_factor = max(0.1, 1.0 - (time_since_last / (7 * 24 * 3600)))  # Week decay
        
        self.relationship_momentum[user_id] = (
            self.relationship_momentum[user_id] * decay_factor + 0.1
        )
    
    def save_relationship_data(self, filename: str):
        """Save all relationship data"""
        
        data = {
            "personal_details": {k: [asdict(detail) for detail in v] 
                               for k, v in self.personal_details.items()},
            "relationship_arcs": {k: asdict(v) for k, v in self.relationship_arcs.items()},
            "relationship_milestones": {k: [asdict(milestone) for milestone in v]
                                      for k, v in self.relationship_milestones.items()},
            "interaction_counts": dict(self.interaction_counts),
            "last_interaction": dict(self.last_interaction),
            "relationship_momentum": dict(self.relationship_momentum)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_relationship_data(self, filename: str):
        """Load relationship data from file"""
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Reconstruct personal details
            for user_id, details_list in data.get("personal_details", {}).items():
                self.personal_details[user_id] = [PersonalDetail(**detail) for detail in details_list]
            
            # Reconstruct relationship arcs
            for user_id, arc_data in data.get("relationship_arcs", {}).items():
                self.relationship_arcs[user_id] = RelationshipArc(**arc_data)
            
            # Reconstruct milestones
            for user_id, milestones_list in data.get("relationship_milestones", {}).items():
                self.relationship_milestones[user_id] = [RelationshipMilestone(**milestone) 
                                                        for milestone in milestones_list]
            
            self.interaction_counts.update(data.get("interaction_counts", {}))
            self.last_interaction.update(data.get("last_interaction", {}))
            self.relationship_momentum.update(data.get("relationship_momentum", {}))
            
            print(f"✅ Loaded relationship data for {len(self.personal_details)} users")
            
        except FileNotFoundError:
            print("📝 No existing relationship data found - starting fresh")
        except Exception as e:
            print(f"❌ Error loading relationship data: {e}")