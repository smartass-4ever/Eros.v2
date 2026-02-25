"""
Enterprise Authentication & Rate Limiting Middleware for CNS API
Protects endpoints and enforces rate limits
Updated to use secure database-backed API key management
"""

from aiohttp import web
from typing import Callable, Awaitable
import time
import os
import secrets

try:
    from cns_secure_api_keys import get_secure_key_manager, APIKeyInfo
    USE_SECURE_KEYS = True
except ImportError:
    from cns_enterprise_api_keys import api_key_manager, APIKeyMetadata
    USE_SECURE_KEYS = False

from cns_logging_monitoring import cns_logger


class CNSAuthMiddleware:
    """Authentication middleware for CNS Enterprise API"""
    
    def __init__(self, admin_api_key: str = None):
        if USE_SECURE_KEYS:
            self.key_manager = get_secure_key_manager()
        else:
            from cns_enterprise_api_keys import api_key_manager
            self.key_manager = api_key_manager
        
        if admin_api_key is None:
            admin_api_key = os.environ.get('CNS_ADMIN_API_KEY', '')
        
        if not admin_api_key or admin_api_key == "cns_admin_change_me_in_production":
            raise ValueError(
                "SECURITY ERROR: CNS_ADMIN_API_KEY environment variable must be set to a secure value. "
                "Never use the default placeholder in production. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        
        self.admin_api_key = admin_api_key
    
    @web.middleware
    async def authenticate(self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.Response]]):
        """Middleware to authenticate API requests"""
        if request.path in ['/health', '/status', '/', '/admin', '/admin/dashboard']:
            return await handler(request)
        
        if request.path.startswith('/admin'):
            return await self.authenticate_admin(request, handler)
        
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header:
            return web.json_response({
                'error': 'Missing Authorization header',
                'message': 'Include your API key in the Authorization header: "Authorization: Bearer YOUR_API_KEY"',
                'docs': 'https://docs.cns-api.com/authentication'
            }, status=401)
        
        if not auth_header.startswith('Bearer '):
            return web.json_response({
                'error': 'Invalid Authorization format',
                'message': 'Use format: "Authorization: Bearer YOUR_API_KEY"'
            }, status=401)
        
        api_key = auth_header[7:].strip()
        
        if USE_SECURE_KEYS:
            return await self._authenticate_secure(request, handler, api_key)
        else:
            return await self._authenticate_legacy(request, handler, api_key)
    
    async def _authenticate_secure(self, request: web.Request, handler, api_key: str):
        """Authentication using new secure database-backed key manager"""
        key_info = self.key_manager.validate_api_key(api_key)
        
        if not key_info:
            return web.json_response({
                'error': 'Invalid API key',
                'message': 'The provided API key is invalid or has been revoked',
                'docs': 'https://docs.cns-api.com/authentication'
            }, status=401)
        
        is_allowed, remaining = self.key_manager.check_rate_limit(api_key)
        
        if not is_allowed:
            return web.json_response({
                'error': 'Rate limit exceeded',
                'message': f'You have exceeded your rate limit of {key_info.rate_limit} requests per hour',
                'tier': key_info.tier,
                'retry_after': 3600,
                'upgrade_info': 'Upgrade to Pro or Enterprise for higher limits: https://cns-api.com/pricing'
            }, status=429, headers={
                'Retry-After': '3600',
                'X-RateLimit-Limit': str(key_info.rate_limit),
                'X-RateLimit-Remaining': str(remaining)
            })
        
        request_id = f"req_{secrets.token_hex(8)}"
        request['request_id'] = request_id
        request['api_key'] = api_key
        request['api_key_info'] = key_info
        request['api_key_metadata'] = self._convert_to_legacy_metadata(key_info)
        
        start_time = time.time()
        
        try:
            response = await handler(request)
            processing_time = (time.time() - start_time) * 1000
            
            self.key_manager.record_usage(
                api_key=api_key,
                endpoint=request.path,
                tokens_used=0,
                response_time_ms=processing_time,
                status_code=response.status
            )
            
            cns_logger.log_request(
                request_id=request_id,
                client_id=key_info.customer_id,
                tier=key_info.tier,
                endpoint=request.path,
                method=request.method,
                status_code=response.status,
                latency_ms=processing_time,
                user_agent=request.headers.get('User-Agent'),
                ip_address=request.remote
            )
            
            response.headers['X-RateLimit-Limit'] = str(key_info.rate_limit)
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-API-Customer'] = key_info.customer_id
            response.headers['X-API-Tier'] = key_info.tier
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            self.key_manager.record_usage(
                api_key=api_key,
                endpoint=request.path,
                tokens_used=0,
                response_time_ms=processing_time,
                status_code=500
            )
            
            cns_logger.log_error(
                request_id=request_id,
                client_id=key_info.customer_id,
                error_type=type(e).__name__,
                error_message=str(e),
                endpoint=request.path,
                stack_trace=None,
                context={'method': request.method}
            )
            
            raise
    
    def _convert_to_legacy_metadata(self, key_info):
        """Convert new APIKeyInfo to legacy-compatible format for handlers"""
        class LegacyMetadata:
            def __init__(self, info):
                self.key_id = info.customer_id
                self.tier = info.tier
                self.rate_limit_hour = info.rate_limit
                self.rate_limit_day = info.monthly_quota
        
        return LegacyMetadata(key_info)
    
    async def _authenticate_legacy(self, request: web.Request, handler, api_key: str):
        """Legacy authentication for backward compatibility"""
        metadata = self.key_manager.validate_api_key(api_key)
        
        if not metadata:
            return web.json_response({
                'error': 'Invalid API key',
                'message': 'The provided API key is invalid or has been revoked',
                'docs': 'https://docs.cns-api.com/authentication'
            }, status=401)
        
        rate_limit_status = self.key_manager.check_rate_limit(metadata.key_id)
        
        if rate_limit_status['is_blocked']:
            if rate_limit_status['hour']['exceeded']:
                reset_time = 3600
                limit_type = 'hourly'
                limit = rate_limit_status['hour']['limit']
            else:
                reset_time = 24 * 3600
                limit_type = 'daily'
                limit = rate_limit_status['day']['limit']
            
            from cns_enterprise_api_keys import APITier
            tier_value = metadata.tier.value if hasattr(metadata.tier, 'value') else metadata.tier
            return web.json_response({
                'error': 'Rate limit exceeded',
                'message': f'You have exceeded your {limit_type} rate limit of {limit} requests',
                'tier': tier_value,
                'rate_limits': rate_limit_status,
                'retry_after': reset_time,
                'upgrade_info': 'Upgrade to Pro or Enterprise for higher limits: https://cns-api.com/pricing'
            }, status=429, headers={
                'Retry-After': str(reset_time),
                'X-RateLimit-Limit-Hour': str(rate_limit_status['hour']['limit']),
                'X-RateLimit-Remaining-Hour': str(rate_limit_status['hour']['remaining']),
                'X-RateLimit-Limit-Day': str(rate_limit_status['day']['limit']),
                'X-RateLimit-Remaining-Day': str(rate_limit_status['day']['remaining'])
            })
        
        request_id = f"req_{secrets.token_hex(8)}"
        request['request_id'] = request_id
        request['api_key_metadata'] = metadata
        request['rate_limit_status'] = rate_limit_status
        
        start_time = time.time()
        
        try:
            response = await handler(request)
            processing_time = (time.time() - start_time) * 1000
            success = response.status < 400
            
            self.key_manager.record_usage(
                key_id=metadata.key_id,
                endpoint=request.path,
                tokens_used=0,
                processing_time=processing_time / 1000,
                success=success
            )
            
            from cns_enterprise_api_keys import APITier
            tier_value = metadata.tier.value if hasattr(metadata.tier, 'value') else metadata.tier
            cns_logger.log_request(
                request_id=request_id,
                client_id=metadata.key_id,
                tier=tier_value,
                endpoint=request.path,
                method=request.method,
                status_code=response.status,
                latency_ms=processing_time,
                user_agent=request.headers.get('User-Agent'),
                ip_address=request.remote
            )
            
            response.headers['X-RateLimit-Limit-Hour'] = str(rate_limit_status['hour']['limit'])
            response.headers['X-RateLimit-Remaining-Hour'] = str(rate_limit_status['hour']['remaining'])
            response.headers['X-RateLimit-Limit-Day'] = str(rate_limit_status['day']['limit'])
            response.headers['X-RateLimit-Remaining-Day'] = str(rate_limit_status['day']['remaining'])
            response.headers['X-API-Key-ID'] = metadata.key_id
            response.headers['X-API-Tier'] = tier_value
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            self.key_manager.record_usage(
                key_id=metadata.key_id,
                endpoint=request.path,
                tokens_used=0,
                processing_time=processing_time / 1000,
                success=False
            )
            
            cns_logger.log_error(
                request_id=request_id,
                client_id=metadata.key_id,
                error_type=type(e).__name__,
                error_message=str(e),
                endpoint=request.path,
                stack_trace=None,
                context={'method': request.method}
            )
            
            raise
    
    async def authenticate_admin(self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.Response]]):
        """Authenticate admin requests"""
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header:
            return web.json_response({
                'error': 'Admin authorization required',
                'message': 'Include admin API key in Authorization header'
            }, status=401)
        
        if not auth_header.startswith('Bearer '):
            return web.json_response({
                'error': 'Invalid Authorization format',
                'message': 'Use format: "Authorization: Bearer ADMIN_API_KEY"'
            }, status=401)
        
        provided_key = auth_header[7:].strip()
        
        if provided_key != self.admin_api_key:
            return web.json_response({
                'error': 'Invalid admin API key',
                'message': 'The provided admin API key is invalid'
            }, status=403)
        
        return await handler(request)


class CNSCORSMiddleware:
    """CORS middleware for API access from web applications"""
    
    @web.middleware
    async def cors_handler(self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.Response]]):
        """Handle CORS preflight and add headers"""
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            response = await handler(request)
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        
        return response


def create_middleware_stack():
    """Create the complete middleware stack for CNS Enterprise API"""
    auth = CNSAuthMiddleware()
    cors = CNSCORSMiddleware()
    
    return [
        cors.cors_handler,
        auth.authenticate
    ]
