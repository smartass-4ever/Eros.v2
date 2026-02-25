"""
CNS Enterprise API Server
Production-ready API for B2B integration of CNS emotional intelligence
"""

import asyncio
import os
from aiohttp import web
import json
import time
from typing import Dict
from updated_cns_discord_bot import DiscordCNSCompanion
from cns_enterprise_api_keys import api_key_manager, APITier, create_api_key
from cns_enterprise_middleware import create_middleware_stack
from cns_logging_monitoring import cns_logger
from cns_stripe_billing import billing_manager
from cns_multiuser_memory_manager import memory_manager
from cns_webhook_delivery import webhook_service


class CNSEnterpriseServer:
    """Enterprise API server with multi-tenant support"""
    
    def __init__(self):
        # Multi-tenant CNS instances (isolated per client)
        self.cns_instances: Dict[str, DiscordCNSCompanion] = {}
        
        # Shared instance for free tier (cost optimization)
        self.shared_cns = DiscordCNSCompanion()
    
    def get_cns_instance(self, key_id: str, tier: APITier) -> DiscordCNSCompanion:
        """Get CNS instance for client (dedicated for Pro/Enterprise, shared for Free)"""
        
        if tier == APITier.ENTERPRISE:
            # Dedicated instance for enterprise clients
            if key_id not in self.cns_instances:
                self.cns_instances[key_id] = DiscordCNSCompanion()
            return self.cns_instances[key_id]
        
        elif tier == APITier.PRO:
            # Dedicated instance for pro clients
            if key_id not in self.cns_instances:
                self.cns_instances[key_id] = DiscordCNSCompanion()
            return self.cns_instances[key_id]
        
        else:
            # Shared instance for free tier
            return self.shared_cns
    
    async def root_endpoint(self, request: web.Request):
        """Root endpoint - API welcome message"""
        return web.json_response({
            'service': 'CNS Enterprise API',
            'version': '1.0.0',
            'status': 'running',
            'message': 'Welcome to the CNS (Cognitive Neural System) Enterprise API',
            'endpoints': {
                'health': '/health',
                'api_docs': '/docs',
                'admin_dashboard': '/admin',
                'api_message': 'POST /api/v1/message',
                'hardware_sync': 'POST /api/v1/hardware/sync'
            },
            'documentation': 'See CNS_ENTERPRISE_API_DOCS.md for full documentation',
            'quick_start': 'See QUICK_START.md to get started in 5 minutes'
        })
    
    async def health_check(self, request: web.Request):
        """Health check endpoint (no auth required)"""
        return web.json_response({
            'status': 'healthy',
            'service': 'CNS Enterprise API',
            'version': '1.0.0',
            'timestamp': time.time()
        })
    
    async def serve_admin_dashboard(self, request: web.Request):
        """Serve admin dashboard HTML"""
        try:
            with open('admin_dashboard.html', 'r') as f:
                html_content = f.read()
            return web.Response(text=html_content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text='Admin dashboard not found', status=404)
    
    async def process_hardware_data(self, request: web.Request):
        """
        Endpoint for processing hardware sensor data from smartwatches/wearables
        
        POST /api/v1/hardware/sync
        """
        try:
            metadata = request['api_key_metadata']
            data = await request.json()
            user_id = data.get('user_id')
            biometrics = data.get('biometrics', {})
            
            # Broadcast to ExperienceBus for real-time brain adjustment
            from experience_bus import ExperienceBus, ExperiencePayload
            bus = ExperienceBus()
            payload = ExperiencePayload(
                experience_type=ExperienceType.HARDWARE_SYNC,
                user_id=user_id,
                metadata={
                    "biometrics": biometrics,
                    "source": "smartwatch"
                },
                timestamp=time.time()
            )
            bus.broadcast(payload)
            
            return web.json_response({
                'status': 'synced',
                'biometrics_received': list(biometrics.keys())
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def process_message(self, request: web.Request):
        """
        Main endpoint for processing messages with CNS emotional intelligence
        
        POST /api/v1/message
        Headers:
            Authorization: Bearer YOUR_API_KEY
        Body:
            {
                "message": "I'm feeling anxious about the presentation",
                "user_id": "user_123",
                "user_name": "John",
                "context": {
                    "conversation_history": [],
                    "warmth_override": 0.8
                }
            }
        """
        try:
            # Get authenticated metadata
            metadata = request['api_key_metadata']
            
            # Parse request
            data = await request.json()
            message = data.get('message', '')
            user_id = data.get('user_id', 'anonymous')
            user_name = data.get('user_name', 'User')
            context = data.get('context', {})
            
            if not message:
                return web.json_response({
                    'error': 'Missing message',
                    'message': 'Request body must include "message" field'
                }, status=400)
            
            # Get appropriate CNS instance
            cns = self.get_cns_instance(metadata.key_id, metadata.tier)
            
            # Load user memory context BEFORE processing
            user_memory = memory_manager.get_or_create_user_memory(
                metadata.key_id, user_id, user_name
            )
            
            # Get conversation history for context continuity
            recent_turns = memory_manager.get_conversation_history(
                metadata.key_id, user_id, limit=6
            )
            
            # Convert conversation turns to format expected by CNS
            conversation_history = []
            for turn in recent_turns:
                conversation_history.append({
                    'role': 'user',
                    'content': turn['user_message']
                })
                conversation_history.append({
                    'role': 'assistant',
                    'content': turn['cns_response']
                })
            
            # Set active user for CNS
            cns.active_user_id = user_id
            cns.current_user_id = user_id
            
            # Process with CNS - pass conversation history for context
            start_time = time.time()
            result = await cns.process_input(message, conversation_history=conversation_history)
            processing_time = time.time() - start_time
            
            # Extract emotion data for storage
            emotion_data = {
                'emotion': result['cns_data'].get('emotion', {}).get('emotion', 'neutral'),
                'valence': result['cns_data'].get('emotion', {}).get('valence', 0),
                'arousal': result['cns_data'].get('emotion', {}).get('arousal', 0.5),
                'dominance': result['cns_data'].get('emotion', {}).get('dominance', 0.5),
                'confidence': result['cns_data'].get('confidence', 0)
            }
            
            # Store conversation turn in memory manager
            memory_manager.store_conversation_turn(
                metadata.key_id,
                user_id,
                message,
                result['response'],
                emotion_data
            )
            
            # Build response
            tier_value = metadata.tier.value if isinstance(metadata.tier, APITier) else metadata.tier
            response_data = {
                'response': result['response'],
                'processing_time': processing_time,
                'emotional_state': result['cns_data'].get('emotion', {}),
                'emotional_analysis': emotion_data,
                'metadata': {
                    'user_id': user_id,
                    'tier': tier_value,
                    'instance_type': 'dedicated' if metadata.tier in [APITier.PRO, APITier.ENTERPRISE] else 'shared',
                    'neuroplastic_multiplier': result['cns_data'].get('neuroplastic_multiplier', 1.0),
                    'consciousness_metrics': result['cns_data'].get('consciousness_metrics', {}),
                    'total_interactions': user_memory.total_interactions
                }
            }
            
            return web.json_response(response_data)
            
        except json.JSONDecodeError:
            return web.json_response({
                'error': 'Invalid JSON',
                'message': 'Request body must be valid JSON'
            }, status=400)
        except Exception as e:
            return web.json_response({
                'error': 'Internal server error',
                'message': str(e)
            }, status=500)
    
    async def get_usage_stats(self, request: web.Request):
        """
        Get usage statistics for authenticated API key
        
        GET /api/v1/usage?hours=24
        """
        try:
            metadata = request['api_key_metadata']
            hours = int(request.query.get('hours', '24'))
            
            stats = api_key_manager.get_usage_stats(metadata.key_id, hours)
            
            return web.json_response(stats)
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get usage stats',
                'message': str(e)
            }, status=500)
    
    async def get_rate_limits(self, request: web.Request):
        """
        Get current rate limit status
        
        GET /api/v1/limits
        """
        try:
            rate_limit_status = request['rate_limit_status']
            return web.json_response(rate_limit_status)
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get rate limits',
                'message': str(e)
            }, status=500)
    
    async def get_api_info(self, request: web.Request):
        """
        Get API key information and capabilities
        
        GET /api/v1/info
        """
        try:
            metadata = request['api_key_metadata']
            
            tier_config = api_key_manager.tier_configs[metadata.tier]
            tier_value = metadata.tier.value if isinstance(metadata.tier, APITier) else metadata.tier
            
            return web.json_response({
                'key_id': metadata.key_id,
                'client_name': metadata.client_name,
                'tier': tier_value,
                'features': tier_config['features'],
                'rate_limits': {
                    'hour': metadata.rate_limit_hour if metadata.rate_limit_hour > 0 else 'unlimited',
                    'day': metadata.rate_limit_day if metadata.rate_limit_day > 0 else 'unlimited'
                },
                'instance_type': 'dedicated' if metadata.tier in [APITier.PRO, APITier.ENTERPRISE] else 'shared',
                'created_at': metadata.created_at,
                'last_used': metadata.last_used,
                'custom_config': metadata.custom_config
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get API info',
                'message': str(e)
            }, status=500)
    
    async def get_conversation_history(self, request: web.Request):
        """
        Get conversation history for a specific user
        
        GET /api/v1/conversations/{user_id}?limit=20&offset=0
        """
        try:
            metadata = request['api_key_metadata']
            user_id = request.match_info['user_id']
            limit = int(request.query.get('limit', '20'))
            offset = int(request.query.get('offset', '0'))
            
            # Validate limits
            limit = min(max(1, limit), 100)  # Between 1-100
            offset = max(0, offset)
            
            # Get conversation history from memory manager
            history = memory_manager.get_conversation_history(
                metadata.key_id, user_id, limit, offset
            )
            
            # Get user profile
            user_profile = memory_manager.get_user_profile(metadata.key_id, user_id)
            
            return web.json_response({
                'user_id': user_id,
                'total_interactions': user_profile.get('total_interactions', 0) if user_profile else 0,
                'history': history,
                'limit': limit,
                'offset': offset,
                'count': len(history)
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to retrieve conversation history',
                'message': str(e)
            }, status=500)
    
    async def clear_conversation_history(self, request: web.Request):
        """
        Clear conversation history for a specific user
        
        DELETE /api/v1/conversations/{user_id}/clear
        """
        try:
            metadata = request['api_key_metadata']
            user_id = request.match_info['user_id']
            
            # Clear history
            success = memory_manager.clear_conversation_history(metadata.key_id, user_id)
            
            if not success:
                return web.json_response({
                    'error': 'User not found',
                    'message': f'No conversation history found for user {user_id}'
                }, status=404)
            
            return web.json_response({
                'success': True,
                'user_id': user_id,
                'message': 'Conversation history cleared successfully'
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to clear conversation history',
                'message': str(e)
            }, status=500)
    
    async def register_webhook(self, request: web.Request):
        """
        Register a webhook URL for proactive messaging callbacks
        
        POST /api/v1/webhooks/register
        Body:
            {
                "webhook_url": "https://yourdomain.com/cns/webhook"
            }
        """
        try:
            metadata = request['api_key_metadata']
            data = await request.json()
            
            webhook_url = data.get('webhook_url', '').strip()
            
            if not webhook_url:
                return web.json_response({
                    'error': 'Missing webhook_url',
                    'message': 'Request body must include "webhook_url" field'
                }, status=400)
            
            # Update webhook URL
            try:
                success = api_key_manager.update_webhook_url(metadata.key_id, webhook_url)
            except ValueError as e:
                return web.json_response({
                    'error': 'Invalid webhook URL',
                    'message': str(e)
                }, status=400)
            
            if not success:
                return web.json_response({
                    'error': 'Failed to update webhook',
                    'message': 'API key not found'
                }, status=404)
            
            return web.json_response({
                'success': True,
                'webhook_url': webhook_url,
                'message': 'Webhook URL registered successfully',
                'info': 'CNS will now send proactive messages to this URL when autonomous drives are strong enough'
            })
            
        except json.JSONDecodeError:
            return web.json_response({
                'error': 'Invalid JSON',
                'message': 'Request body must be valid JSON'
            }, status=400)
        except Exception as e:
            return web.json_response({
                'error': 'Failed to register webhook',
                'message': str(e)
            }, status=500)
    
    async def get_webhook_status(self, request: web.Request):
        """
        Get current webhook configuration
        
        GET /api/v1/webhooks/status
        """
        try:
            metadata = request['api_key_metadata']
            
            webhook_url = metadata.webhook_url if hasattr(metadata, 'webhook_url') else None
            
            return web.json_response({
                'webhook_enabled': bool(webhook_url),
                'webhook_url': webhook_url if webhook_url else None,
                'message': 'Webhook is active' if webhook_url else 'No webhook configured'
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get webhook status',
                'message': str(e)
            }, status=500)
    
    async def get_billing_info(self, request: web.Request):
        """
        Get billing information for current API key
        
        GET /api/v1/billing/info
        """
        try:
            metadata = request['api_key_metadata']
            tier_value = metadata.tier.value if isinstance(metadata.tier, APITier) else metadata.tier
            
            billing_info = billing_manager.get_customer_info(metadata.key_id)
            
            if not billing_info:
                return web.json_response({
                    'tier': tier_value,
                    'billing_enabled': False,
                    'message': 'Billing not configured for this account'
                })
            
            return web.json_response({
                'tier': tier_value,
                'billing_enabled': True,
                'customer': billing_info['customer'],
                'subscription': billing_info.get('subscription'),
                'pricing': billing_info['pricing']
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get billing info',
                'message': str(e)
            }, status=500)
    
    async def setup_billing(self, request: web.Request):
        """
        Set up billing for Pro/Enterprise tier
        
        POST /api/v1/billing/setup
        Body:
            {
                "payment_method_id": "pm_xxx",
                "email": "billing@company.com",
                "company_name": "Acme Corp"
            }
        """
        try:
            metadata = request['api_key_metadata']
            data = await request.json()
            
            if metadata.tier == APITier.FREE:
                return web.json_response({
                    'error': 'Billing not required for Free tier',
                    'message': 'Upgrade to Pro or Enterprise to set up billing'
                }, status=400)
            
            email = data.get('email')
            company_name = data.get('company_name')
            payment_method_id = data.get('payment_method_id')
            
            if not email or not company_name:
                return web.json_response({
                    'error': 'Missing required fields',
                    'message': 'email and company_name are required'
                }, status=400)
            
            customer_id = billing_manager.create_customer(
                key_id=metadata.key_id,
                email=email,
                company_name=company_name,
                tier=metadata.tier
            )
            
            if not customer_id:
                return web.json_response({
                    'error': 'Failed to create customer',
                    'message': 'Stripe billing is not enabled'
                }, status=500)
            
            subscription = billing_manager.create_subscription(
                key_id=metadata.key_id,
                tier=metadata.tier,
                payment_method_id=payment_method_id
            )
            
            if not subscription:
                return web.json_response({
                    'error': 'Failed to create subscription',
                    'message': 'Could not set up subscription'
                }, status=500)
            
            return web.json_response({
                'success': True,
                'customer_id': customer_id,
                'subscription': subscription,
                'message': 'Billing set up successfully'
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to set up billing',
                'message': str(e)
            }, status=500)
    
    async def admin_create_key(self, request: web.Request):
        """
        ADMIN ONLY: Create a new API key
        
        POST /admin/keys/create
        Body:
            {
                "client_name": "Company Name",
                "client_email": "contact@company.com",
                "tier": "pro",
                "notes": "Special request from sales"
            }
        """
        try:
            data = await request.json()
            
            client_name = data.get('client_name')
            client_email = data.get('client_email')
            tier_str = data.get('tier', 'free')
            notes = data.get('notes', '')
            
            if not client_name or not client_email:
                return web.json_response({
                    'error': 'Missing required fields',
                    'message': 'client_name and client_email are required'
                }, status=400)
            
            tier = APITier(tier_str.lower())
            
            result = create_api_key(
                client_name=client_name,
                client_email=client_email,
                tier=tier,
                notes=notes
            )
            
            return web.json_response(result, status=201)
            
        except ValueError as e:
            return web.json_response({
                'error': 'Invalid tier',
                'message': 'Tier must be one of: free, pro, enterprise'
            }, status=400)
        except Exception as e:
            return web.json_response({
                'error': 'Failed to create API key',
                'message': str(e)
            }, status=500)
    
    async def admin_list_keys(self, request: web.Request):
        """
        ADMIN ONLY: List all API keys
        
        GET /admin/keys?active_only=true
        """
        try:
            active_only = request.query.get('active_only', 'true').lower() == 'true'
            keys = api_key_manager.list_keys(active_only=active_only)
            
            return web.json_response({
                'total': len(keys),
                'keys': keys
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to list keys',
                'message': str(e)
            }, status=500)
    
    async def admin_revoke_key(self, request: web.Request):
        """
        ADMIN ONLY: Revoke an API key
        
        POST /admin/keys/{key_id}/revoke
        """
        try:
            key_id = request.match_info['key_id']
            data = await request.json()
            reason = data.get('reason', 'No reason provided')
            
            success = api_key_manager.revoke_api_key(key_id, reason)
            
            if success:
                return web.json_response({
                    'success': True,
                    'message': f'API key {key_id} revoked',
                    'reason': reason
                })
            else:
                return web.json_response({
                    'error': 'Key not found',
                    'message': f'API key {key_id} does not exist'
                }, status=404)
                
        except Exception as e:
            return web.json_response({
                'error': 'Failed to revoke key',
                'message': str(e)
            }, status=500)
    
    async def get_system_metrics(self, request: web.Request):
        """
        Get system performance metrics
        
        GET /api/v1/metrics?hours=1
        """
        try:
            hours = int(request.query.get('hours', '1'))
            metrics = cns_logger.get_metrics(hours)
            
            return web.json_response({
                'timestamp': metrics.timestamp,
                'period_hours': hours,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'success_rate': (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0,
                'avg_latency_ms': metrics.avg_latency_ms,
                'p95_latency_ms': metrics.p95_latency_ms,
                'p99_latency_ms': metrics.p99_latency_ms,
                'requests_per_second': metrics.requests_per_second,
                'active_clients': metrics.active_clients,
                'tier_breakdown': metrics.tier_breakdown
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get metrics',
                'message': str(e)
            }, status=500)
    
    async def admin_get_metrics(self, request: web.Request):
        """
        ADMIN ONLY: Get detailed system metrics
        
        GET /admin/monitoring/metrics?hours=24
        """
        try:
            hours = int(request.query.get('hours', '24'))
            metrics = cns_logger.get_metrics(hours)
            
            return web.json_response({
                'timestamp': metrics.timestamp,
                'period_hours': hours,
                'metrics': {
                    'total_requests': metrics.total_requests,
                    'successful_requests': metrics.successful_requests,
                    'failed_requests': metrics.failed_requests,
                    'success_rate': (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0,
                    'latency': {
                        'avg_ms': metrics.avg_latency_ms,
                        'p95_ms': metrics.p95_latency_ms,
                        'p99_ms': metrics.p99_latency_ms
                    },
                    'throughput': {
                        'requests_per_second': metrics.requests_per_second,
                        'active_clients': metrics.active_clients
                    },
                    'tier_breakdown': metrics.tier_breakdown
                }
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get admin metrics',
                'message': str(e)
            }, status=500)
    
    async def admin_get_errors(self, request: web.Request):
        """
        ADMIN ONLY: Get error summary
        
        GET /admin/monitoring/errors?hours=24
        """
        try:
            hours = int(request.query.get('hours', '24'))
            error_summary = cns_logger.get_error_summary(hours)
            
            return web.json_response(error_summary)
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get error summary',
                'message': str(e)
            }, status=500)
    
    async def admin_get_client_stats(self, request: web.Request):
        """
        ADMIN ONLY: Get stats for a specific client
        
        GET /admin/monitoring/client/{key_id}?hours=24
        """
        try:
            key_id = request.match_info['key_id']
            hours = int(request.query.get('hours', '24'))
            
            stats = cns_logger.get_client_stats(key_id, hours)
            
            return web.json_response(stats)
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get client stats',
                'message': str(e)
            }, status=500)
    
    async def admin_create_invoice(self, request: web.Request):
        """
        ADMIN ONLY: Create invoice for a client
        
        POST /admin/billing/invoice/{key_id}
        Body:
            {
                "total_requests": 15000,
                "tier": "pro"
            }
        """
        try:
            key_id = request.match_info['key_id']
            data = await request.json()
            
            total_requests = data.get('total_requests', 0)
            tier_str = data.get('tier', 'free')
            tier = APITier[tier_str.upper()]
            
            invoice = billing_manager.create_invoice(key_id, total_requests, tier)
            
            if not invoice:
                return web.json_response({
                    'error': 'Failed to create invoice',
                    'message': 'Billing not enabled or customer not found'
                }, status=400)
            
            return web.json_response({
                'success': True,
                'invoice': invoice
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to create invoice',
                'message': str(e)
            }, status=500)
    
    async def admin_get_billing_overview(self, request: web.Request):
        """
        ADMIN ONLY: Get billing overview for all customers
        
        GET /admin/billing/overview
        """
        try:
            billing_data = billing_manager.billing_data
            
            total_customers = len(billing_data['customers'])
            active_subscriptions = len([s for s in billing_data['subscriptions'].values() 
                                       if s.get('status') == 'active'])
            total_invoices = len(billing_data['invoices'])
            
            return web.json_response({
                'total_customers': total_customers,
                'active_subscriptions': active_subscriptions,
                'total_invoices': total_invoices,
                'customers': list(billing_data['customers'].values()),
                'subscriptions': list(billing_data['subscriptions'].values()),
                'recent_invoices': billing_data['invoices'][-10:]
            })
            
        except Exception as e:
            return web.json_response({
                'error': 'Failed to get billing overview',
                'message': str(e)
            }, status=500)


async def startup_background_tasks(app):
    """Start background tasks when the server starts"""
    # Start webhook delivery background loop
    app['webhook_task'] = asyncio.create_task(webhook_service.run_background_loop())
    print("[SERVER] 🚀 Webhook delivery background loop started")

async def cleanup_background_tasks(app):
    """Cleanup background tasks when server shuts down"""
    webhook_service.stop()
    if 'webhook_task' in app:
        app['webhook_task'].cancel()
        try:
            await app['webhook_task']
        except asyncio.CancelledError:
            pass
    print("[SERVER] 🛑 Background tasks stopped")

async def create_app():
    """Create and configure the application"""
    server = CNSEnterpriseServer()
    app = web.Application(middlewares=create_middleware_stack())
    
    # Register startup and cleanup handlers
    app.on_startup.append(startup_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    # Public endpoints (authenticated via middleware)
    app.router.add_get('/', server.root_endpoint)
    app.router.add_get('/health', server.health_check)
    app.router.add_post('/api/v1/message', server.process_message)
    app.router.add_post('/api/v1/hardware/sync', server.process_hardware_data)
    app.router.add_get('/api/v1/usage', server.get_usage_stats)
    app.router.add_get('/api/v1/limits', server.get_rate_limits)
    app.router.add_get('/api/v1/info', server.get_api_info)
    app.router.add_get('/api/v1/metrics', server.get_system_metrics)
    app.router.add_get('/api/v1/billing/info', server.get_billing_info)
    app.router.add_post('/api/v1/billing/setup', server.setup_billing)
    
    # Conversation history endpoints
    app.router.add_get('/api/v1/conversations/{user_id}', server.get_conversation_history)
    app.router.add_delete('/api/v1/conversations/{user_id}/clear', server.clear_conversation_history)
    
    # Webhook endpoints
    app.router.add_post('/api/v1/webhooks/register', server.register_webhook)
    app.router.add_get('/api/v1/webhooks/status', server.get_webhook_status)
    
    # Admin dashboard UI (no auth - uses admin key in browser)
    app.router.add_get('/admin', server.serve_admin_dashboard)
    app.router.add_get('/admin/dashboard', server.serve_admin_dashboard)
    
    # Admin API endpoints (admin auth required)
    app.router.add_post('/admin/keys/create', server.admin_create_key)
    app.router.add_get('/admin/keys', server.admin_list_keys)
    app.router.add_post('/admin/keys/{key_id}/revoke', server.admin_revoke_key)
    app.router.add_get('/admin/monitoring/metrics', server.admin_get_metrics)
    app.router.add_get('/admin/monitoring/errors', server.admin_get_errors)
    app.router.add_get('/admin/monitoring/client/{key_id}', server.admin_get_client_stats)
    app.router.add_post('/admin/billing/invoice/{key_id}', server.admin_create_invoice)
    app.router.add_get('/admin/billing/overview', server.admin_get_billing_overview)
    
    return app


if __name__ == '__main__':
    print("🚀 Starting CNS Enterprise API Server...")
    print("📊 Multi-tenant emotional intelligence API")
    print("🔒 Secure API key authentication")
    print("⚡ Rate limiting enabled")
    print("")
    print("Endpoints:")
    print("  - POST /api/v1/message - Process messages with CNS")
    print("  - POST /api/v1/hardware/sync - Sync smartwatch sensor data")
    print("  - GET  /api/v1/usage - Get usage statistics")
    print("  - GET  /api/v1/limits - Check rate limits")
    print("  - GET  /api/v1/info - Get API key info")
    print("")
    print("Admin Endpoints:")
    print("  - POST /admin/keys/create - Create new API key")
    print("  - GET  /admin/keys - List all keys")
    print("  - POST /admin/keys/{id}/revoke - Revoke key")
    print("")
    
    app = asyncio.run(create_app())
    port = int(os.environ.get('PORT', 8000))
    web.run_app(app, host='0.0.0.0', port=port)
