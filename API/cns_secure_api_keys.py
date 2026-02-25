"""
Secure API Key Management for CNS Enterprise
Database-backed with proper hashing and security
Replaces the insecure JSON file storage
"""

import secrets
import hashlib
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from cns_database import CNSDatabase, APIKey, UsageLog


class APITier(Enum):
    """API access tiers with different limits"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class APIKeyInfo:
    """API key information (without the actual key)"""
    customer_id: str
    tier: str
    rate_limit: int
    monthly_quota: int
    usage_this_month: int
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime]


class SecureAPIKeyManager:
    """
    Secure database-backed API key management
    - Keys are hashed before storage (never stored in plaintext)
    - All operations go through PostgreSQL
    - Proper rate limiting and usage tracking
    """
    
    TIER_CONFIGS = {
        'free': {
            'rate_limit': 100,        # requests per hour
            'monthly_quota': 1000,     # requests per month
            'features': ['basic_emotion', 'simple_responses']
        },
        'pro': {
            'rate_limit': 1000,
            'monthly_quota': 50000,
            'features': ['basic_emotion', 'simple_responses', 'memory', 'personality_adaptation']
        },
        'enterprise': {
            'rate_limit': 10000,
            'monthly_quota': 1000000,
            'features': ['all', 'custom_config', 'dedicated_instance', 'priority_support']
        }
    }
    
    def __init__(self):
        self.db = CNSDatabase()
        self._rate_limit_cache = {}  # In-memory cache for rate limiting
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        random_part = secrets.token_hex(32)
        return f"cns_live_{random_part}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for secure storage (one-way)"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_api_key(
        self,
        customer_id: str,
        tier: str = 'free',
        expires_in_days: Optional[int] = None
    ) -> Optional[str]:
        """
        Create a new API key for a customer
        
        Returns the plaintext API key (only shown once!)
        The key is hashed before storage.
        """
        tier_config = self.TIER_CONFIGS.get(tier, self.TIER_CONFIGS['free'])
        
        api_key = self.generate_api_key()
        key_hash = self.hash_api_key(api_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        session = self.db.get_session()
        try:
            new_key = APIKey(
                key_hash=key_hash,
                customer_id=customer_id,
                tier=tier,
                rate_limit=tier_config['rate_limit'],
                monthly_quota=tier_config['monthly_quota'],
                usage_this_month=0,
                is_active=True,
                expires_at=expires_at
            )
            session.add(new_key)
            session.commit()
            
            print(f"✅ API key created for customer: {customer_id} (tier: {tier})")
            return api_key
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error creating API key: {e}")
            return None
        finally:
            session.close()
    
    def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """
        Validate an API key and return its info
        
        Returns None if invalid, expired, or inactive
        """
        key_hash = self.hash_api_key(api_key)
        
        session = self.db.get_session()
        try:
            key_record = session.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()
            
            if not key_record:
                return None
            
            # Check expiration
            if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                return None
            
            # Update last used
            key_record.last_used = datetime.utcnow()
            session.commit()
            
            return APIKeyInfo(
                customer_id=key_record.customer_id,
                tier=key_record.tier,
                rate_limit=key_record.rate_limit,
                monthly_quota=key_record.monthly_quota,
                usage_this_month=key_record.usage_this_month,
                is_active=key_record.is_active,
                created_at=key_record.created_at,
                last_used=key_record.last_used
            )
            
        finally:
            session.close()
    
    def check_rate_limit(self, api_key: str) -> tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Returns: (is_allowed, requests_remaining)
        """
        key_hash = self.hash_api_key(api_key)
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        cache_key = f"{key_hash}_{current_hour.isoformat()}"
        
        session = self.db.get_session()
        try:
            key_record = session.query(APIKey).filter(
                APIKey.key_hash == key_hash
            ).first()
            
            if not key_record:
                return (False, 0)
            
            # Check in-memory cache for current hour
            if cache_key not in self._rate_limit_cache:
                self._rate_limit_cache[cache_key] = 0
            
            current_usage = self._rate_limit_cache[cache_key]
            rate_limit = key_record.rate_limit
            
            if current_usage >= rate_limit:
                return (False, 0)
            
            # Increment usage
            self._rate_limit_cache[cache_key] = current_usage + 1
            
            return (True, rate_limit - current_usage - 1)
            
        finally:
            session.close()
    
    def record_usage(self, api_key: str, endpoint: str, 
                     tokens_used: int = 0, response_time_ms: float = 0,
                     status_code: int = 200):
        """Record API usage for billing and analytics"""
        key_hash = self.hash_api_key(api_key)
        
        session = self.db.get_session()
        try:
            key_record = session.query(APIKey).filter(
                APIKey.key_hash == key_hash
            ).first()
            
            if not key_record:
                return
            
            # Increment monthly usage
            key_record.usage_this_month += 1
            
            # Log the usage
            usage_log = UsageLog(
                customer_id=key_record.customer_id,
                endpoint=endpoint,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                status_code=status_code
            )
            session.add(usage_log)
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error recording usage: {e}")
        finally:
            session.close()
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke (deactivate) an API key"""
        key_hash = self.hash_api_key(api_key)
        
        session = self.db.get_session()
        try:
            key_record = session.query(APIKey).filter(
                APIKey.key_hash == key_hash
            ).first()
            
            if key_record:
                key_record.is_active = False
                session.commit()
                print(f"✅ API key revoked for customer: {key_record.customer_id}")
                return True
            
            return False
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error revoking API key: {e}")
            return False
        finally:
            session.close()
    
    def get_customer_usage(self, customer_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a customer"""
        session = self.db.get_session()
        try:
            since = datetime.utcnow() - timedelta(days=days)
            
            logs = session.query(UsageLog).filter(
                UsageLog.customer_id == customer_id,
                UsageLog.created_at >= since
            ).all()
            
            total_requests = len(logs)
            total_tokens = sum(log.tokens_used for log in logs)
            avg_response_time = (
                sum(log.response_time_ms for log in logs) / total_requests
                if total_requests > 0 else 0
            )
            
            return {
                'customer_id': customer_id,
                'period_days': days,
                'total_requests': total_requests,
                'total_tokens': total_tokens,
                'avg_response_time_ms': avg_response_time,
                'endpoints': self._count_endpoints(logs)
            }
            
        finally:
            session.close()
    
    def _count_endpoints(self, logs: List[UsageLog]) -> Dict[str, int]:
        """Count requests per endpoint"""
        counts = {}
        for log in logs:
            counts[log.endpoint] = counts.get(log.endpoint, 0) + 1
        return counts
    
    def reset_monthly_usage(self):
        """Reset monthly usage counters (call at start of month)"""
        session = self.db.get_session()
        try:
            session.query(APIKey).update({APIKey.usage_this_month: 0})
            session.commit()
            print("✅ Monthly usage counters reset")
        except Exception as e:
            session.rollback()
            print(f"❌ Error resetting usage: {e}")
        finally:
            session.close()


# Singleton instance
_manager_instance = None

def get_secure_key_manager() -> SecureAPIKeyManager:
    """Get the singleton SecureAPIKeyManager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SecureAPIKeyManager()
    return _manager_instance


if __name__ == "__main__":
    # Test the secure key manager
    manager = get_secure_key_manager()
    
    # Create a test key
    test_key = manager.create_api_key(
        customer_id="test_customer_001",
        tier="pro"
    )
    
    if test_key:
        print(f"🔑 Test API key (save this!): {test_key}")
        
        # Validate it
        info = manager.validate_api_key(test_key)
        if info:
            print(f"✅ Key validated: {info}")
        
        # Check rate limit
        allowed, remaining = manager.check_rate_limit(test_key)
        print(f"Rate limit check: allowed={allowed}, remaining={remaining}")
