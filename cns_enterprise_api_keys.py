"""
Enterprise API Key Management for CNS
Handles secure API key generation, validation, and management for B2B clients
"""

import secrets
import hashlib
import json
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class APITier(Enum):
    """API access tiers with different limits"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class APIKeyMetadata:
    """Metadata for an API key"""
    key_id: str
    client_name: str
    client_email: str
    tier: APITier
    created_at: float
    last_used: Optional[float]
    is_active: bool
    rate_limit_hour: int
    rate_limit_day: int
    custom_config: Dict[str, Any]
    notes: str
    webhook_url: Optional[str] = None  # URL for proactive message callbacks


@dataclass
class UsageRecord:
    """Single usage record for analytics"""
    key_id: str
    timestamp: float
    endpoint: str
    tokens_used: int
    processing_time: float
    success: bool


class CNSAPIKeyManager:
    """
    Manages API keys for enterprise clients
    Provides secure key generation, validation, and usage tracking
    """
    
    def __init__(self, storage_path: str = "cns_api_keys.json"):
        self.storage_path = storage_path
        self.keys: Dict[str, APIKeyMetadata] = {}
        self.key_hashes: Dict[str, str] = {}  # hash -> key_id mapping
        self.usage_records: List[UsageRecord] = []
        
        # Tier configurations
        self.tier_configs = {
            APITier.FREE: {
                'rate_limit_hour': 1000,
                'rate_limit_day': 10000,
                'max_warmth': 0.7,
                'features': ['basic_emotion', 'simple_responses']
            },
            APITier.PRO: {
                'rate_limit_hour': 10000,
                'rate_limit_day': 100000,
                'max_warmth': 1.0,
                'features': ['basic_emotion', 'simple_responses', 'memory', 'personality_adaptation']
            },
            APITier.ENTERPRISE: {
                'rate_limit_hour': -1,  # Unlimited
                'rate_limit_day': -1,
                'max_warmth': 1.0,
                'features': ['all', 'custom_config', 'dedicated_instance', 'priority_support']
            }
        }
        
        self.load_keys()
    
    def generate_api_key(self) -> str:
        """
        Generate a secure API key
        Format: cns_live_<32 random hex characters>
        """
        random_part = secrets.token_hex(32)
        return f"cns_live_{random_part}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_api_key(
        self,
        client_name: str,
        client_email: str,
        tier: APITier = APITier.FREE,
        custom_config: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new API key for a client
        
        Returns:
            Dict with 'api_key' (show once) and 'metadata'
        """
        # Generate key
        api_key = self.generate_api_key()
        key_hash = self.hash_api_key(api_key)
        key_id = f"key_{secrets.token_hex(8)}"
        
        # Get tier config
        tier_config = self.tier_configs[tier]
        
        # Create metadata
        metadata = APIKeyMetadata(
            key_id=key_id,
            client_name=client_name,
            client_email=client_email,
            tier=tier,
            created_at=time.time(),
            last_used=None,
            is_active=True,
            rate_limit_hour=tier_config['rate_limit_hour'],
            rate_limit_day=tier_config['rate_limit_day'],
            custom_config=custom_config or {},
            notes=notes
        )
        
        # Store
        self.keys[key_id] = metadata
        self.key_hashes[key_hash] = key_id
        self.save_keys()
        
        return {
            'api_key': api_key,  # SHOW ONLY ONCE
            'key_id': key_id,
            'client_name': client_name,
            'tier': tier.value,
            'rate_limits': {
                'hour': tier_config['rate_limit_hour'],
                'day': tier_config['rate_limit_day']
            },
            'features': tier_config['features'],
            'created_at': datetime.fromtimestamp(metadata.created_at).isoformat(),
            'warning': 'Store this API key securely. It will not be shown again.'
        }
    
    def validate_api_key(self, api_key: str) -> Optional[APIKeyMetadata]:
        """
        Validate an API key and return metadata if valid
        
        Returns:
            APIKeyMetadata if valid, None if invalid
        """
        key_hash = self.hash_api_key(api_key)
        
        # Check if key exists
        if key_hash not in self.key_hashes:
            return None
        
        key_id = self.key_hashes[key_hash]
        metadata = self.keys.get(key_id)
        
        # Check if key is active
        if not metadata or not metadata.is_active:
            return None
        
        # Update last used
        metadata.last_used = time.time()
        self.save_keys()
        
        return metadata
    
    def revoke_api_key(self, key_id: str, reason: str = "") -> bool:
        """
        Revoke an API key
        
        Returns:
            True if revoked, False if not found
        """
        if key_id not in self.keys:
            return False
        
        self.keys[key_id].is_active = False
        self.keys[key_id].notes += f"\nRevoked at {datetime.now().isoformat()}: {reason}"
        self.save_keys()
        
        return True
    
    def upgrade_tier(self, key_id: str, new_tier: APITier) -> bool:
        """
        Upgrade a key to a higher tier
        
        Returns:
            True if upgraded, False if not found
        """
        if key_id not in self.keys:
            return False
        
        metadata = self.keys[key_id]
        old_tier = metadata.tier
        
        metadata.tier = new_tier
        tier_config = self.tier_configs[new_tier]
        metadata.rate_limit_hour = tier_config['rate_limit_hour']
        metadata.rate_limit_day = tier_config['rate_limit_day']
        metadata.notes += f"\nUpgraded from {old_tier.value} to {new_tier.value} at {datetime.now().isoformat()}"
        
        self.save_keys()
        return True
    
    def update_webhook_url(self, key_id: str, webhook_url: str) -> bool:
        """
        Update webhook URL for proactive messaging callbacks
        
        Args:
            key_id: API key ID
            webhook_url: URL to receive webhook POSTs (or empty string to disable)
        
        Returns:
            True if updated, False if not found
        """
        if key_id not in self.keys:
            return False
        
        metadata = self.keys[key_id]
        
        # Validate URL format if provided
        if webhook_url and not webhook_url.startswith(('http://', 'https://')):
            raise ValueError("Webhook URL must start with http:// or https://")
        
        metadata.webhook_url = webhook_url if webhook_url else None
        metadata.notes += f"\nWebhook URL updated at {datetime.now().isoformat()}"
        
        self.save_keys()
        return True
    
    def record_usage(
        self,
        key_id: str,
        endpoint: str,
        tokens_used: int,
        processing_time: float,
        success: bool
    ):
        """Record API usage for analytics"""
        usage = UsageRecord(
            key_id=key_id,
            timestamp=time.time(),
            endpoint=endpoint,
            tokens_used=tokens_used,
            processing_time=processing_time,
            success=success
        )
        
        self.usage_records.append(usage)
        
        # Keep only recent records (last 30 days)
        cutoff = time.time() - (30 * 24 * 60 * 60)
        self.usage_records = [r for r in self.usage_records if r.timestamp > cutoff]
        
        self.save_keys()
    
    def get_usage_stats(self, key_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get usage statistics for a key
        
        Args:
            key_id: API key ID
            hours: Number of hours to look back
            
        Returns:
            Usage statistics
        """
        if key_id not in self.keys:
            return {'error': 'Key not found'}
        
        metadata = self.keys[key_id]
        cutoff = time.time() - (hours * 60 * 60)
        
        # Filter records for this key
        key_records = [r for r in self.usage_records if r.key_id == key_id and r.timestamp > cutoff]
        
        if not key_records:
            return {
                'key_id': key_id,
                'client_name': metadata.client_name,
                'tier': metadata.tier.value,
                'period_hours': hours,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_tokens': 0,
                'avg_processing_time': 0
            }
        
        total_requests = len(key_records)
        successful_requests = sum(1 for r in key_records if r.success)
        failed_requests = total_requests - successful_requests
        total_tokens = sum(r.tokens_used for r in key_records)
        avg_processing_time = sum(r.processing_time for r in key_records) / total_requests
        
        # Endpoint breakdown
        endpoint_counts = {}
        for record in key_records:
            endpoint_counts[record.endpoint] = endpoint_counts.get(record.endpoint, 0) + 1
        
        return {
            'key_id': key_id,
            'client_name': metadata.client_name,
            'tier': metadata.tier.value,
            'period_hours': hours,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'total_tokens': total_tokens,
            'avg_processing_time': avg_processing_time,
            'endpoints': endpoint_counts,
            'rate_limit_status': self.check_rate_limit(key_id)
        }
    
    def check_rate_limit(self, key_id: str) -> Dict[str, Any]:
        """
        Check if key is within rate limits
        
        Returns:
            Status and usage counts
        """
        if key_id not in self.keys:
            return {'error': 'Key not found'}
        
        metadata = self.keys[key_id]
        
        # Count requests in last hour and day
        now = time.time()
        hour_cutoff = now - 3600
        day_cutoff = now - (24 * 3600)
        
        key_records = [r for r in self.usage_records if r.key_id == key_id]
        
        hour_requests = sum(1 for r in key_records if r.timestamp > hour_cutoff)
        day_requests = sum(1 for r in key_records if r.timestamp > day_cutoff)
        
        # Check limits
        hour_limit = metadata.rate_limit_hour
        day_limit = metadata.rate_limit_day
        
        hour_exceeded = hour_limit > 0 and hour_requests >= hour_limit
        day_exceeded = day_limit > 0 and day_requests >= day_limit
        
        return {
            'key_id': key_id,
            'tier': metadata.tier.value,
            'hour': {
                'used': hour_requests,
                'limit': hour_limit if hour_limit > 0 else 'unlimited',
                'remaining': max(0, hour_limit - hour_requests) if hour_limit > 0 else 'unlimited',
                'exceeded': hour_exceeded
            },
            'day': {
                'used': day_requests,
                'limit': day_limit if day_limit > 0 else 'unlimited',
                'remaining': max(0, day_limit - day_requests) if day_limit > 0 else 'unlimited',
                'exceeded': day_exceeded
            },
            'is_blocked': hour_exceeded or day_exceeded
        }
    
    def list_keys(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all API keys"""
        keys_list = []
        
        for key_id, metadata in self.keys.items():
            if active_only and not metadata.is_active:
                continue
            
            keys_list.append({
                'key_id': key_id,
                'client_name': metadata.client_name,
                'client_email': metadata.client_email,
                'tier': metadata.tier.value,
                'created_at': datetime.fromtimestamp(metadata.created_at).isoformat(),
                'last_used': datetime.fromtimestamp(metadata.last_used).isoformat() if metadata.last_used else 'Never',
                'is_active': metadata.is_active,
                'rate_limits': {
                    'hour': metadata.rate_limit_hour if metadata.rate_limit_hour > 0 else 'unlimited',
                    'day': metadata.rate_limit_day if metadata.rate_limit_day > 0 else 'unlimited'
                }
            })
        
        return keys_list
    
    def save_keys(self):
        """Save keys to persistent storage"""
        try:
            data = {
                'keys': {},
                'key_hashes': self.key_hashes,
                'usage_records': []
            }
            
            # Serialize keys
            for key_id, metadata in self.keys.items():
                metadata_dict = asdict(metadata)
                metadata_dict['tier'] = metadata.tier.value
                data['keys'][key_id] = metadata_dict
            
            # Serialize usage records
            data['usage_records'] = [asdict(r) for r in self.usage_records]
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save API keys: {e}")
    
    def load_keys(self):
        """Load keys from persistent storage"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load keys
            self.keys = {}
            for key_id, metadata_dict in data.get('keys', {}).items():
                metadata_dict['tier'] = APITier(metadata_dict['tier'])
                self.keys[key_id] = APIKeyMetadata(**metadata_dict)
            
            # Load key hashes
            self.key_hashes = data.get('key_hashes', {})
            
            # Load usage records
            self.usage_records = [UsageRecord(**r) for r in data.get('usage_records', [])]
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Failed to load API keys: {e}")


# Global API key manager instance
api_key_manager = CNSAPIKeyManager()


# Helper functions for easy access
def create_api_key(client_name: str, client_email: str, tier: APITier = APITier.FREE, **kwargs) -> Dict[str, Any]:
    """Create a new API key"""
    return api_key_manager.create_api_key(client_name, client_email, tier, **kwargs)


def validate_api_key(api_key: str) -> Optional[APIKeyMetadata]:
    """Validate an API key"""
    return api_key_manager.validate_api_key(api_key)


def check_rate_limit(key_id: str) -> Dict[str, Any]:
    """Check rate limit status"""
    return api_key_manager.check_rate_limit(key_id)
