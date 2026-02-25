"""
Consequence System for CNS (Eros)

Makes emotional states have REAL lasting effects:
- Guilt hits global confidence (Eros becomes less bold everywhere)
- Sadness hits per-user trust (relationship-specific damage)
- Move failures create per-user per-move cooldowns (avoid what burned you)

Design principle: Humans feel real because mistakes close doors.
Eros should compartmentalize like humans do.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import logging
import json

logger = logging.getLogger(__name__)


class MoveType(Enum):
    """Specific moves that can be locked after failure."""
    UNSOLICITED_ADVICE = "unsolicited_advice"
    BOUNDARY_PUSHING_QUESTION = "boundary_pushing_question"
    FLIRT_ESCALATION = "flirt_escalation"
    PROACTIVE_OUTREACH = "proactive_outreach"
    DIRECT_CHALLENGE = "direct_challenge"
    VULNERABILITY_PROMPT = "vulnerability_prompt"
    BOLD_OPINION = "bold_opinion"
    BELIEF_CONFRONTATION = "belief_confrontation"
    DEEP_PERSONAL_QUESTION = "deep_personal_question"
    HUMOR_RISK = "humor_risk"
    COMMITMENT_OFFER = "commitment_offer"
    EMOTIONAL_ESCALATION = "emotional_escalation"


class MoveTier(Enum):
    """Move tiers by risk level."""
    BASE = "base"
    TRUST_GATED = "trust_gated"
    CONFIDENCE_GATED = "confidence_gated"
    SPECIALTY = "specialty"


MOVE_TIER_MAPPING = {
    MoveType.UNSOLICITED_ADVICE: MoveTier.CONFIDENCE_GATED,
    MoveType.BOUNDARY_PUSHING_QUESTION: MoveTier.TRUST_GATED,
    MoveType.FLIRT_ESCALATION: MoveTier.CONFIDENCE_GATED,
    MoveType.PROACTIVE_OUTREACH: MoveTier.TRUST_GATED,
    MoveType.DIRECT_CHALLENGE: MoveTier.TRUST_GATED,
    MoveType.VULNERABILITY_PROMPT: MoveTier.TRUST_GATED,
    MoveType.BOLD_OPINION: MoveTier.CONFIDENCE_GATED,
    MoveType.BELIEF_CONFRONTATION: MoveTier.SPECIALTY,
    MoveType.DEEP_PERSONAL_QUESTION: MoveTier.SPECIALTY,
    MoveType.HUMOR_RISK: MoveTier.CONFIDENCE_GATED,
    MoveType.COMMITMENT_OFFER: MoveTier.SPECIALTY,
    MoveType.EMOTIONAL_ESCALATION: MoveTier.TRUST_GATED,
}

TIER_REQUIREMENTS = {
    MoveTier.BASE: {"trust": 0.0, "confidence": 0.0},
    MoveTier.TRUST_GATED: {"trust": 0.4, "confidence": 0.0},
    MoveTier.CONFIDENCE_GATED: {"trust": 0.0, "confidence": 0.5},
    MoveTier.SPECIALTY: {"trust": 0.7, "confidence": 0.6},
}

BELIEF_ALIGNED_REPAIR_MOVES = {
    "grounded_observation": {
        "description": "Simple observation without pressure",
        "belief": "curiosity",
        "example": "That sounds like a lot to carry."
    },
    "reflective_mirroring": {
        "description": "Reflect back what they said",
        "belief": "empathy", 
        "example": "Sounds like you're feeling [x]."
    },
    "quiet_presence": {
        "description": "Acknowledge without demanding",
        "belief": "patience",
        "example": "I'm here if you want to talk."
    },
    "principled_restraint": {
        "description": "Honor boundaries explicitly",
        "belief": "respect",
        "example": "No pressure - just wanted to check in."
    },
    "gentle_curiosity": {
        "description": "Soft open question",
        "belief": "interest",
        "example": "How are you feeling about it?"
    }
}


@dataclass
class MoveCooldown:
    """A specific move locked for a specific user."""
    move_type: MoveType
    locked_until: float
    reason: str
    original_trigger: str
    
    def is_expired(self) -> bool:
        return time.time() > self.locked_until
    
    def time_remaining(self) -> float:
        return max(0, self.locked_until - time.time())
    
    def to_dict(self) -> Dict:
        return {
            "move_type": self.move_type.value,
            "locked_until": self.locked_until,
            "reason": self.reason,
            "original_trigger": self.original_trigger
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MoveCooldown':
        return cls(
            move_type=MoveType(data["move_type"]),
            locked_until=data["locked_until"],
            reason=data["reason"],
            original_trigger=data.get("original_trigger", "")
        )


@dataclass
class RecoveryProgress:
    """Tracks varied signals needed for recovery."""
    reply_count: int = 0
    engagement_depth_count: int = 0
    time_spaced_interactions: int = 0
    emotional_positive_count: int = 0
    last_interaction_time: Optional[float] = None
    
    def add_interaction(self, depth: str, emotional_tone: str):
        """Record an interaction with its qualities."""
        now = time.time()
        self.reply_count += 1
        
        if depth in ['substantive', 'deep', 'vulnerable']:
            self.engagement_depth_count += 1
        
        if self.last_interaction_time:
            hours_since = (now - self.last_interaction_time) / 3600
            if hours_since > 1:
                self.time_spaced_interactions += 1
        
        if emotional_tone in ['warm', 'grateful', 'happy', 'excited', 'appreciative']:
            self.emotional_positive_count += 1
        
        self.last_interaction_time = now
    
    def meets_recovery_threshold(self, threshold: Dict[str, int]) -> bool:
        """Check if varied signals meet recovery requirements."""
        return (
            self.reply_count >= threshold.get('reply_count', 3) and
            self.engagement_depth_count >= threshold.get('engagement_depth', 2) and
            self.time_spaced_interactions >= threshold.get('time_spacing', 1) and
            self.emotional_positive_count >= threshold.get('emotional_positive', 1)
        )
    
    def reset(self):
        self.reply_count = 0
        self.engagement_depth_count = 0
        self.time_spaced_interactions = 0
        self.emotional_positive_count = 0
    
    def to_dict(self) -> Dict:
        return {
            "reply_count": self.reply_count,
            "engagement_depth_count": self.engagement_depth_count,
            "time_spaced_interactions": self.time_spaced_interactions,
            "emotional_positive_count": self.emotional_positive_count,
            "last_interaction_time": self.last_interaction_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecoveryProgress':
        rp = cls()
        rp.reply_count = data.get("reply_count", 0)
        rp.engagement_depth_count = data.get("engagement_depth_count", 0)
        rp.time_spaced_interactions = data.get("time_spaced_interactions", 0)
        rp.emotional_positive_count = data.get("emotional_positive_count", 0)
        rp.last_interaction_time = data.get("last_interaction_time")
        return rp


@dataclass
class UserTrustState:
    """Per-user trust and cooldowns."""
    user_id: str
    trust_level: float = 0.5
    cooldowns: Dict[str, MoveCooldown] = field(default_factory=dict)
    recovery_progress: RecoveryProgress = field(default_factory=RecoveryProgress)
    trust_damage_history: List[Dict] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    TRUST_FLOOR = 0.2
    TRUST_CEILING = 1.0
    
    def apply_trust_damage(self, amount: float, reason: str):
        """Sadness event damages trust with this user."""
        old_trust = self.trust_level
        self.trust_level = max(self.TRUST_FLOOR, self.trust_level - amount)
        self.trust_damage_history.append({
            "timestamp": time.time(),
            "old_trust": old_trust,
            "new_trust": self.trust_level,
            "damage": amount,
            "reason": reason
        })
        self.recovery_progress.reset()
        self.last_updated = time.time()
        logger.info(f"[CONSEQUENCE] Trust damage for {self.user_id}: {old_trust:.2f} → {self.trust_level:.2f} ({reason})")
    
    def apply_trust_recovery(self, amount: float, reason: str):
        """Positive signals rebuild trust."""
        old_trust = self.trust_level
        self.trust_level = min(self.TRUST_CEILING, self.trust_level + amount)
        self.last_updated = time.time()
        logger.info(f"[CONSEQUENCE] Trust recovery for {self.user_id}: {old_trust:.2f} → {self.trust_level:.2f} ({reason})")
    
    def add_cooldown(self, move_type: MoveType, duration_hours: float, reason: str, trigger: str):
        """Lock a specific move for this user."""
        cooldown = MoveCooldown(
            move_type=move_type,
            locked_until=time.time() + (duration_hours * 3600),
            reason=reason,
            original_trigger=trigger
        )
        self.cooldowns[move_type.value] = cooldown
        self.last_updated = time.time()
        logger.info(f"[CONSEQUENCE] Cooldown added for {self.user_id}: {move_type.value} locked for {duration_hours}h ({reason})")
    
    def is_move_locked(self, move_type: MoveType) -> bool:
        """Check if a specific move is on cooldown."""
        if move_type.value not in self.cooldowns:
            return False
        cooldown = self.cooldowns[move_type.value]
        if cooldown.is_expired():
            del self.cooldowns[move_type.value]
            return False
        return True
    
    def get_locked_moves(self) -> List[MoveType]:
        """Get all currently locked moves."""
        self._clean_expired_cooldowns()
        return [MoveType(k) for k in self.cooldowns.keys()]
    
    def _clean_expired_cooldowns(self):
        """Remove expired cooldowns."""
        expired = [k for k, v in self.cooldowns.items() if v.is_expired()]
        for k in expired:
            del self.cooldowns[k]
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "trust_level": self.trust_level,
            "cooldowns": {k: v.to_dict() for k, v in self.cooldowns.items()},
            "recovery_progress": self.recovery_progress.to_dict(),
            "trust_damage_history": self.trust_damage_history[-10:],
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserTrustState':
        state = cls(user_id=data["user_id"])
        state.trust_level = data.get("trust_level", 0.5)
        state.cooldowns = {
            k: MoveCooldown.from_dict(v) 
            for k, v in data.get("cooldowns", {}).items()
        }
        state.recovery_progress = RecoveryProgress.from_dict(
            data.get("recovery_progress", {})
        )
        state.trust_damage_history = data.get("trust_damage_history", [])
        state.last_updated = data.get("last_updated", time.time())
        return state


class ConsequenceSystem:
    """
    Central consequence manager.
    
    - confidence_global: Eros's overall boldness (guilt hits this)
    - trust[user_id]: Per-user relationship state (sadness hits this)
    - cooldowns[user_id][move_type]: Specific moves locked per user
    """
    
    CONFIDENCE_FLOOR = 0.2
    CONFIDENCE_CEILING = 1.0
    
    CONFIDENCE_RECOVERY_THRESHOLD = {
        'reply_count': 5,
        'engagement_depth': 3,
        'time_spacing': 2,
        'emotional_positive': 2
    }
    
    TRUST_RECOVERY_THRESHOLD = {
        'reply_count': 3,
        'engagement_depth': 2,
        'time_spacing': 1,
        'emotional_positive': 1
    }
    
    def __init__(self, db_connection=None):
        self.confidence_global: float = 0.7
        self.user_states: Dict[str, UserTrustState] = {}
        self.global_recovery_progress = RecoveryProgress()
        self.confidence_damage_history: List[Dict] = []
        self.db = db_connection
        
        self._load_from_database()
        print("🚧 Consequence System initialized - emotions now have teeth")
        print(f"   Global confidence: {self.confidence_global:.2f}")
        print(f"   Tracked users: {len(self.user_states)}")
    
    def get_user_state(self, user_id: str) -> UserTrustState:
        """Get or create user trust state."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserTrustState(user_id=user_id)
        return self.user_states[user_id]
    
    def apply_guilt(self, intensity: float, reason: str, trigger_context: str = ""):
        """
        Guilt hits GLOBAL confidence.
        Eros becomes less bold everywhere temporarily.
        """
        damage = intensity * 0.15
        old_confidence = self.confidence_global
        self.confidence_global = max(
            self.CONFIDENCE_FLOOR, 
            self.confidence_global - damage
        )
        self.confidence_damage_history.append({
            "timestamp": time.time(),
            "old_confidence": old_confidence,
            "new_confidence": self.confidence_global,
            "damage": damage,
            "reason": reason,
            "trigger": trigger_context
        })
        self.global_recovery_progress.reset()
        self._save_to_database()
        logger.info(f"[CONSEQUENCE] Global confidence hit: {old_confidence:.2f} → {self.confidence_global:.2f} (guilt: {reason})")
    
    def apply_sadness(self, user_id: str, intensity: float, reason: str, 
                      failed_move: Optional[MoveType] = None, trigger_context: str = ""):
        """
        Sadness hits PER-USER trust.
        Also locks the specific move that failed if provided.
        """
        state = self.get_user_state(user_id)
        damage = intensity * 0.2
        state.apply_trust_damage(damage, reason)
        
        if failed_move:
            cooldown_hours = 2 + (intensity * 10)
            state.add_cooldown(
                failed_move, 
                cooldown_hours, 
                reason, 
                trigger_context
            )
        
        self._save_to_database()
    
    def process_positive_interaction(self, user_id: str, depth: str, emotional_tone: str):
        """
        Process a positive interaction for recovery.
        Both global confidence and per-user trust can recover.
        """
        state = self.get_user_state(user_id)
        state.recovery_progress.add_interaction(depth, emotional_tone)
        self.global_recovery_progress.add_interaction(depth, emotional_tone)
        
        if state.recovery_progress.meets_recovery_threshold(self.TRUST_RECOVERY_THRESHOLD):
            recovery_amount = 0.1
            state.apply_trust_recovery(recovery_amount, "sustained positive engagement")
            state.recovery_progress.reset()
        
        if self.global_recovery_progress.meets_recovery_threshold(self.CONFIDENCE_RECOVERY_THRESHOLD):
            old_conf = self.confidence_global
            self.confidence_global = min(
                self.CONFIDENCE_CEILING,
                self.confidence_global + 0.1
            )
            if self.confidence_global > old_conf:
                logger.info(f"[CONSEQUENCE] Global confidence recovery: {old_conf:.2f} → {self.confidence_global:.2f}")
            self.global_recovery_progress.reset()
        
        self._save_to_database()
    
    def get_available_moves(self, user_id: str) -> Dict[str, Any]:
        """
        ConsequenceGate: Check what moves are available.
        Returns dict with available moves and locked moves with reasons.
        """
        state = self.get_user_state(user_id)
        trust = state.trust_level
        confidence = self.confidence_global
        
        available = []
        locked = []
        locked_reasons = {}
        
        for move_type in MoveType:
            tier = MOVE_TIER_MAPPING.get(move_type, MoveTier.BASE)
            requirements = TIER_REQUIREMENTS[tier]
            
            if state.is_move_locked(move_type):
                cooldown = state.cooldowns.get(move_type.value)
                locked.append(move_type.value)
                locked_reasons[move_type.value] = {
                    "reason": "on_cooldown",
                    "detail": cooldown.reason if cooldown else "specific move locked",
                    "time_remaining_hours": cooldown.time_remaining() / 3600 if cooldown else 0
                }
            elif trust < requirements["trust"]:
                locked.append(move_type.value)
                locked_reasons[move_type.value] = {
                    "reason": "trust_too_low",
                    "detail": f"Requires trust {requirements['trust']}, have {trust:.2f}",
                    "needed": requirements["trust"] - trust
                }
            elif confidence < requirements["confidence"]:
                locked.append(move_type.value)
                locked_reasons[move_type.value] = {
                    "reason": "confidence_too_low",
                    "detail": f"Requires confidence {requirements['confidence']}, have {confidence:.2f}",
                    "needed": requirements["confidence"] - confidence
                }
            else:
                available.append(move_type.value)
        
        return {
            "available_moves": available,
            "locked_moves": locked,
            "locked_reasons": locked_reasons,
            "trust_level": trust,
            "confidence_global": confidence,
            "repair_moves_available": list(BELIEF_ALIGNED_REPAIR_MOVES.keys())
        }
    
    def get_prohibitions_for_directive(self, user_id: str) -> List[str]:
        """
        Get prohibitions to add to ExecutableDirective.
        Returns list of natural language prohibitions.
        """
        gate = self.get_available_moves(user_id)
        prohibitions = []
        
        for move, info in gate["locked_reasons"].items():
            if info["reason"] == "on_cooldown":
                prohibitions.append(f"NO {move.replace('_', ' ')} - recently backfired")
            elif info["reason"] == "trust_too_low":
                prohibitions.append(f"NO {move.replace('_', ' ')} - trust not yet established")
            elif info["reason"] == "confidence_too_low":
                prohibitions.append(f"NO {move.replace('_', ' ')} - confidence shaken")
        
        return prohibitions
    
    def get_repair_move_suggestions(self, user_id: str) -> List[Dict]:
        """
        Get belief-aligned repair moves when other moves locked.
        """
        gate = self.get_available_moves(user_id)
        if len(gate["locked_moves"]) == 0:
            return []
        
        return [
            {
                "move": name,
                **details
            }
            for name, details in BELIEF_ALIGNED_REPAIR_MOVES.items()
        ]
    
    def subscribe_to_bus(self, bus):
        """Subscribe to ExperienceBus for learning events."""
        bus.subscribe("ConsequenceSystem", self._process_experience)
        print("🔗 ConsequenceSystem subscribed to ExperienceBus")
    
    def _process_experience(self, payload):
        """Process experience payloads from ExperienceBus."""
        from experience_bus import ExperienceType
        
        user_id = payload.user_id
        
        if payload.experience_type == ExperienceType.ACTION_SUCCESS:
            self._process_action_success(user_id, payload.action_data)
        elif payload.experience_type == ExperienceType.ACTION_FAILURE:
            self._process_action_failure(user_id, payload.action_data)
        
        if payload.emotional_outcome:
            tone = payload.emotional_outcome.get('user_emotional_shift', 'neutral')
            if tone in ['positive', 'warm', 'grateful', 'engaged']:
                depth = 'substantive' if payload.response_worked else 'casual'
                self.process_positive_interaction(user_id, depth, tone)
        
        if hasattr(payload, 'consequence_event'):
            event = payload.consequence_event
            if event.get('type') == 'guilt':
                self.apply_guilt(
                    event.get('intensity', 0.5),
                    event.get('reason', 'unknown'),
                    event.get('trigger', '')
                )
            elif event.get('type') == 'sadness':
                move_type = None
                if event.get('failed_move'):
                    try:
                        move_type = MoveType(event['failed_move'])
                    except ValueError:
                        pass
                self.apply_sadness(
                    user_id,
                    event.get('intensity', 0.5),
                    event.get('reason', 'unknown'),
                    move_type,
                    event.get('trigger', '')
                )
    
    def _process_action_success(self, user_id: str, action_data: dict):
        """Process a successful action - builds trust"""
        if not action_data:
            return
        
        state = self.get_user_state(user_id)
        action_type = action_data.get('action_type', 'unknown')
        
        state.action_success_count = getattr(state, 'action_success_count', 0) + 1
        
        if state.action_success_count % 3 == 0:
            state.apply_trust_recovery(0.05, f"reliable action execution ({action_type})")
            logger.info(f"[CONSEQUENCE] Action trust boost for {user_id}: +0.05 (success streak)")
    
    def _process_action_failure(self, user_id: str, action_data: dict):
        """Process a failed action - may damage trust"""
        if not action_data:
            return
        
        state = self.get_user_state(user_id)
        action_type = action_data.get('action_type', 'unknown')
        error = action_data.get('error', 'unknown error')
        
        state.action_failure_count = getattr(state, 'action_failure_count', 0) + 1
        
        if state.action_failure_count >= 2:
            damage = 0.1
            state.apply_trust_damage(damage, f"repeated action failures ({action_type}: {error})")
            logger.info(f"[CONSEQUENCE] Action trust damage for {user_id}: -{damage} (failures)")
            state.action_failure_count = 0
    
    def _load_from_database(self):
        """Load state from database using SQLAlchemy ORM."""
        if not self.db:
            return
        
        try:
            from cns_database import ConsequenceState as ConsequenceStateModel, UserConsequenceState as UserConsequenceStateModel
            session = self.db.get_session()
            
            global_state = session.query(ConsequenceStateModel).filter_by(id=1).first()
            if global_state and global_state.state_data:
                data = global_state.state_data if isinstance(global_state.state_data, dict) else json.loads(global_state.state_data)
                self.confidence_global = data.get('confidence_global', 0.7)
                self.confidence_damage_history = data.get('confidence_damage_history', [])
                self.global_recovery_progress = RecoveryProgress.from_dict(
                    data.get('global_recovery_progress', {})
                )
            
            user_states = session.query(UserConsequenceStateModel).all()
            for row in user_states:
                data = row.state_data if isinstance(row.state_data, dict) else json.loads(row.state_data)
                self.user_states[row.user_id] = UserTrustState.from_dict(data)
            
            session.close()
        except Exception as e:
            logger.debug(f"Could not load consequence state: {e}")
    
    def _save_to_database(self):
        """Save state to database using SQLAlchemy ORM."""
        if not self.db:
            return
        
        try:
            from cns_database import ConsequenceState as ConsequenceStateModel, UserConsequenceState as UserConsequenceStateModel
            session = self.db.get_session()
            
            global_data = {
                'confidence_global': self.confidence_global,
                'confidence_damage_history': self.confidence_damage_history[-20:],
                'global_recovery_progress': self.global_recovery_progress.to_dict()
            }
            
            global_state = session.query(ConsequenceStateModel).filter_by(id=1).first()
            if global_state:
                global_state.state_data = global_data
            else:
                global_state = ConsequenceStateModel(id=1, state_data=global_data)
                session.add(global_state)
            
            for user_id, state in self.user_states.items():
                user_state = session.query(UserConsequenceStateModel).filter_by(user_id=user_id).first()
                state_data = state.to_dict()
                if user_state:
                    user_state.state_data = state_data
                else:
                    user_state = UserConsequenceStateModel(user_id=user_id, state_data=state_data)
                    session.add(user_state)
            
            session.commit()
            session.close()
        except Exception as e:
            logger.debug(f"Could not save consequence state: {e}")
    
    def get_status(self) -> Dict:
        """Get current consequence system status."""
        return {
            "confidence_global": self.confidence_global,
            "confidence_floor": self.CONFIDENCE_FLOOR,
            "user_count": len(self.user_states),
            "users": {
                uid: {
                    "trust": state.trust_level,
                    "locked_moves": state.get_locked_moves(),
                    "recovery_progress": state.recovery_progress.to_dict()
                }
                for uid, state in self.user_states.items()
            },
            "global_recovery_progress": self.global_recovery_progress.to_dict()
        }
