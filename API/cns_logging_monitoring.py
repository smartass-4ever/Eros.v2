"""
Logging & Monitoring System for CNS Enterprise API
Tracks requests, errors, latency, and customer usage patterns
"""

import json
import time
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cns_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('cns_enterprise')


@dataclass
class RequestLog:
    """Log entry for an API request"""
    timestamp: float
    request_id: str
    client_id: str
    tier: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    tokens_used: int
    error: Optional[str]
    user_agent: Optional[str]
    ip_address: Optional[str]


@dataclass
class ErrorLog:
    """Log entry for an error"""
    timestamp: float
    request_id: str
    client_id: str
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    endpoint: str
    context: Dict[str, Any]


@dataclass
class MetricSnapshot:
    """Performance metrics snapshot"""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    active_clients: int
    tier_breakdown: Dict[str, int]


class CNSLogger:
    """Centralized logging and monitoring for CNS Enterprise API"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # In-memory request logs for metrics (last 1 hour)
        self.request_logs: List[RequestLog] = []
        self.error_logs: List[ErrorLog] = []
        
        # Metrics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Request log file
        self.request_log_file = self.log_dir / f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.error_log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def log_request(
        self,
        request_id: str,
        client_id: str,
        tier: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        tokens_used: int = 0,
        error: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log an API request"""
        
        log_entry = RequestLog(
            timestamp=time.time(),
            request_id=request_id,
            client_id=client_id,
            tier=tier,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            error=error,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        # Add to in-memory logs
        self.request_logs.append(log_entry)
        
        # Update metrics
        self.total_requests += 1
        if status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Write to file (JSONL format)
        with open(self.request_log_file, 'a') as f:
            f.write(json.dumps(asdict(log_entry)) + '\n')
        
        # Structured logging
        logger.info(
            f"Request | {request_id} | {client_id} | {tier} | "
            f"{method} {endpoint} | {status_code} | {latency_ms:.2f}ms"
        )
        
        # Clean old logs (keep last hour)
        self._cleanup_old_logs()
    
    def log_error(
        self,
        request_id: str,
        client_id: str,
        error_type: str,
        error_message: str,
        endpoint: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log an error"""
        
        error_entry = ErrorLog(
            timestamp=time.time(),
            request_id=request_id,
            client_id=client_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            endpoint=endpoint,
            context=context or {}
        )
        
        # Add to in-memory logs
        self.error_logs.append(error_entry)
        
        # Write to file
        with open(self.error_log_file, 'a') as f:
            f.write(json.dumps(asdict(error_entry)) + '\n')
        
        # Structured logging
        logger.error(
            f"Error | {request_id} | {client_id} | {error_type} | "
            f"{endpoint} | {error_message}"
        )
    
    def get_metrics(self, hours: int = 1) -> MetricSnapshot:
        """Get performance metrics for the last N hours"""
        
        cutoff = time.time() - (hours * 3600)
        recent_logs = [log for log in self.request_logs if log.timestamp > cutoff]
        
        if not recent_logs:
            return MetricSnapshot(
                timestamp=time.time(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                requests_per_second=0,
                active_clients=0,
                tier_breakdown={}
            )
        
        # Calculate metrics
        successful = [log for log in recent_logs if log.status_code < 400]
        failed = [log for log in recent_logs if log.status_code >= 400]
        
        latencies = sorted([log.latency_ms for log in recent_logs])
        
        p95_index = int(len(latencies) * 0.95)
        p99_index = int(len(latencies) * 0.99)
        
        # Time window
        time_window = time.time() - min(log.timestamp for log in recent_logs)
        rps = len(recent_logs) / time_window if time_window > 0 else 0
        
        # Tier breakdown
        tier_breakdown = {}
        for log in recent_logs:
            tier_breakdown[log.tier] = tier_breakdown.get(log.tier, 0) + 1
        
        # Unique clients
        active_clients = len(set(log.client_id for log in recent_logs))
        
        return MetricSnapshot(
            timestamp=time.time(),
            total_requests=len(recent_logs),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=latencies[p95_index] if p95_index < len(latencies) else latencies[-1],
            p99_latency_ms=latencies[p99_index] if p99_index < len(latencies) else latencies[-1],
            requests_per_second=rps,
            active_clients=active_clients,
            tier_breakdown=tier_breakdown
        )
    
    def get_client_stats(self, client_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific client"""
        
        cutoff = time.time() - (hours * 3600)
        client_logs = [
            log for log in self.request_logs 
            if log.client_id == client_id and log.timestamp > cutoff
        ]
        
        if not client_logs:
            return {
                'client_id': client_id,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_latency_ms': 0,
                'endpoints_used': []
            }
        
        successful = [log for log in client_logs if log.status_code < 400]
        failed = [log for log in client_logs if log.status_code >= 400]
        
        # Endpoint usage
        endpoint_counts = {}
        for log in client_logs:
            endpoint_counts[log.endpoint] = endpoint_counts.get(log.endpoint, 0) + 1
        
        # Error analysis
        errors = [log for log in client_logs if log.error]
        error_types = {}
        for log in errors:
            error_types[log.error] = error_types.get(log.error, 0) + 1
        
        return {
            'client_id': client_id,
            'tier': client_logs[0].tier if client_logs else 'unknown',
            'total_requests': len(client_logs),
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'success_rate': (len(successful) / len(client_logs) * 100) if client_logs else 0,
            'avg_latency_ms': sum(log.latency_ms for log in client_logs) / len(client_logs),
            'total_tokens': sum(log.tokens_used for log in client_logs),
            'endpoints_used': endpoint_counts,
            'error_types': error_types
        }
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        
        cutoff = time.time() - (hours * 3600)
        recent_errors = [log for log in self.error_logs if log.timestamp > cutoff]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_types': {},
                'endpoints_affected': {},
                'clients_affected': []
            }
        
        # Group by error type
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        # Group by endpoint
        endpoints = {}
        for error in recent_errors:
            endpoints[error.endpoint] = endpoints.get(error.endpoint, 0) + 1
        
        # Affected clients
        clients = list(set(error.client_id for error in recent_errors))
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_types,
            'endpoints_affected': endpoints,
            'clients_affected': clients,
            'error_rate': (len(recent_errors) / self.total_requests * 100) if self.total_requests > 0 else 0
        }
    
    def _cleanup_old_logs(self):
        """Remove logs older than 1 hour from memory"""
        cutoff = time.time() - 3600
        self.request_logs = [log for log in self.request_logs if log.timestamp > cutoff]
        self.error_logs = [log for log in self.error_logs if log.timestamp > cutoff]
    
    def export_logs(self, hours: int = 24, format: str = 'json') -> str:
        """Export logs for analysis"""
        
        cutoff = time.time() - (hours * 3600)
        recent_logs = [log for log in self.request_logs if log.timestamp > cutoff]
        
        if format == 'json':
            return json.dumps([asdict(log) for log in recent_logs], indent=2)
        elif format == 'csv':
            # CSV export
            import csv
            import io
            
            output = io.StringIO()
            if recent_logs:
                writer = csv.DictWriter(output, fieldnames=asdict(recent_logs[0]).keys())
                writer.writeheader()
                for log in recent_logs:
                    writer.writerow(asdict(log))
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global logger instance
cns_logger = CNSLogger()


# Helper functions
def log_request(request_id: str, client_id: str, tier: str, endpoint: str,
                method: str, status_code: int, latency_ms: float, **kwargs):
    """Log an API request"""
    cns_logger.log_request(request_id, client_id, tier, endpoint, method,
                          status_code, latency_ms, **kwargs)


def log_error(request_id: str, client_id: str, error_type: str,
             error_message: str, endpoint: str, **kwargs):
    """Log an error"""
    cns_logger.log_error(request_id, client_id, error_type, error_message,
                        endpoint, **kwargs)


def get_metrics(hours: int = 1) -> MetricSnapshot:
    """Get performance metrics"""
    return cns_logger.get_metrics(hours)
