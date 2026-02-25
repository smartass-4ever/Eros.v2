"""
Node Bridge Server - WebSocket server for Local Node connections
Runs alongside the Discord bot to enable local action execution
"""

import asyncio
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass, field
from aiohttp import web, WSMsgType
import aiohttp

try:
    from cns_database import CNSDatabase, LocalNodeConnection
except ImportError:
    CNSDatabase = None
    LocalNodeConnection = None


@dataclass
class PendingPairing:
    """A pending pairing request"""
    code: str
    user_id: str
    discord_channel_id: str
    created_at: datetime
    expires_at: datetime


@dataclass
class ConnectedNode:
    """A connected local node"""
    node_id: str
    user_id: str
    ws: web.WebSocketResponse
    connected_at: datetime
    os_type: str = "unknown"
    os_version: str = "unknown"
    capabilities: Set[str] = field(default_factory=set)
    last_ping: datetime = None


@dataclass
class PendingRequest:
    """A pending action request waiting for node response"""
    request_id: str
    user_id: str
    operation: str
    params: Dict
    created_at: datetime
    future: asyncio.Future


class NodeBridgeServer:
    """WebSocket server for managing Local Node connections"""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = web.Application()
        self.runner = None
        
        self.pending_pairings: Dict[str, PendingPairing] = {}
        self.connected_nodes: Dict[str, ConnectedNode] = {}
        self.user_to_node: Dict[str, str] = {}
        self.pending_requests: Dict[str, PendingRequest] = {}
        
        self._db = None
        self._setup_routes()
    
    @property
    def db(self):
        if self._db is None and CNSDatabase:
            try:
                self._db = CNSDatabase()
            except Exception as e:
                print(f"[NodeBridge] DB init error: {e}")
        return self._db
    
    def _setup_routes(self):
        self.app.router.add_get('/health', self._health_check)
        self.app.router.add_get('/node', self._handle_node_connection)
    
    async def _health_check(self, request):
        return web.json_response({
            "status": "ok",
            "connected_nodes": len(self.connected_nodes),
            "pending_pairings": len(self.pending_pairings)
        })
    
    def generate_pairing_code(self, user_id: str, channel_id: str) -> str:
        """Generate a 6-character pairing code"""
        code = secrets.token_hex(3).upper()
        
        self.pending_pairings[code] = PendingPairing(
            code=code,
            user_id=user_id,
            discord_channel_id=channel_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        
        print(f"[NodeBridge] Generated pairing code {code} for user {user_id}")
        return code
    
    def verify_pairing_code(self, code: str, node_id: str) -> Optional[str]:
        """Verify a pairing code and return user_id if valid"""
        code = code.upper()
        pairing = self.pending_pairings.get(code)
        
        if not pairing:
            return None
        
        if datetime.now() > pairing.expires_at:
            del self.pending_pairings[code]
            return None
        
        user_id = pairing.user_id
        del self.pending_pairings[code]
        
        self._save_pairing_to_db(user_id, node_id)
        
        print(f"[NodeBridge] Node {node_id[:8]}... paired with user {user_id}")
        return user_id
    
    def _save_pairing_to_db(self, user_id: str, node_id: str):
        """Save pairing to database"""
        if not self.db or not LocalNodeConnection:
            return
        
        try:
            session = self.db.Session()
            
            existing = session.query(LocalNodeConnection).filter(
                LocalNodeConnection.node_id == node_id
            ).first()
            
            if existing:
                existing.user_id = user_id
                existing.is_active = True
                existing.last_connected = datetime.now()
            else:
                connection = LocalNodeConnection(
                    node_id=node_id,
                    user_id=user_id,
                    node_name=f"node-{node_id[:8]}",
                    is_active=True,
                    last_connected=datetime.now()
                )
                session.add(connection)
            
            session.commit()
            session.close()
        except Exception as e:
            print(f"[NodeBridge] Error saving pairing: {e}")
    
    async def _handle_node_connection(self, request):
        """Handle WebSocket connection from a Local Node"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        node_id = None
        user_id = None
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    
                    if msg_type == "auth":
                        node_id = data.get("node_id")
                        pairing_code = data.get("pairing_code")
                        stored_user_id = data.get("user_id")
                        
                        if pairing_code:
                            user_id = self.verify_pairing_code(pairing_code, node_id)
                            if not user_id:
                                await ws.send_json({"type": "auth_failed", "error": "Invalid or expired pairing code"})
                                break
                        elif stored_user_id:
                            if self._verify_stored_pairing(node_id, stored_user_id):
                                user_id = stored_user_id
                            else:
                                await ws.send_json({"type": "auth_failed", "error": "Node not paired to this user"})
                                break
                        else:
                            await ws.send_json({"type": "auth_failed", "error": "Pairing code or user_id required"})
                            break
                        
                        self.connected_nodes[node_id] = ConnectedNode(
                            node_id=node_id,
                            user_id=user_id,
                            ws=ws,
                            connected_at=datetime.now()
                        )
                        self.user_to_node[user_id] = node_id
                        
                        await ws.send_json({"type": "auth_success", "user_id": user_id})
                        print(f"[NodeBridge] Node {node_id[:8]}... authenticated for user {user_id}")
                    
                    elif msg_type == "status":
                        if node_id and node_id in self.connected_nodes:
                            node = self.connected_nodes[node_id]
                            node.os_type = data.get("os", "unknown")
                            node.os_version = data.get("os_version", "unknown")
                            node.capabilities = set(data.get("capabilities", []))
                    
                    elif msg_type == "pong":
                        if node_id and node_id in self.connected_nodes:
                            self.connected_nodes[node_id].last_ping = datetime.now()
                    
                    elif msg_type == "result":
                        request_id = data.get("request_id")
                        if request_id in self.pending_requests:
                            pending = self.pending_requests.pop(request_id)
                            if not pending.future.done():
                                pending.future.set_result({
                                    "success": data.get("success", False),
                                    "data": data.get("data"),
                                    "error": data.get("error")
                                })
                
                elif msg.type == WSMsgType.ERROR:
                    print(f"[NodeBridge] WebSocket error: {ws.exception()}")
                    break
        
        finally:
            if node_id and node_id in self.connected_nodes:
                del self.connected_nodes[node_id]
            if user_id and user_id in self.user_to_node:
                del self.user_to_node[user_id]
            print(f"[NodeBridge] Node {node_id[:8] if node_id else 'unknown'}... disconnected")
        
        return ws
    
    def _verify_stored_pairing(self, node_id: str, user_id: str) -> bool:
        """Verify that a node is paired to a user in the database"""
        if not self.db or not LocalNodeConnection:
            return False
        
        try:
            session = self.db.Session()
            connection = session.query(LocalNodeConnection).filter(
                LocalNodeConnection.node_id == node_id,
                LocalNodeConnection.user_id == user_id,
                LocalNodeConnection.is_active == True
            ).first()
            session.close()
            return connection is not None
        except Exception as e:
            print(f"[NodeBridge] Error verifying pairing: {e}")
            return False
    
    def user_has_node(self, user_id: str) -> bool:
        """Check if a user has a connected node"""
        return user_id in self.user_to_node
    
    async def execute_on_node(self, user_id: str, operation: str, params: Dict, 
                               require_confirmation: bool = False, timeout: float = 30.0) -> Dict:
        """Execute an operation on a user's local node"""
        node_id = self.user_to_node.get(user_id)
        if not node_id:
            return {"success": False, "error": "No connected node for this user"}
        
        node = self.connected_nodes.get(node_id)
        if not node:
            return {"success": False, "error": "Node disconnected"}
        
        request_id = secrets.token_hex(8)
        future = asyncio.get_event_loop().create_future()
        
        self.pending_requests[request_id] = PendingRequest(
            request_id=request_id,
            user_id=user_id,
            operation=operation,
            params=params,
            created_at=datetime.now(),
            future=future
        )
        
        msg_type = "confirm" if require_confirmation else "execute"
        await node.ws.send_json({
            "type": msg_type,
            "request_id": request_id,
            "operation": operation,
            "params": params
        })
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            return {"success": False, "error": "Request timed out"}
    
    async def start(self):
        """Start the WebSocket server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()
        print(f"[NodeBridge] WebSocket server started on port {self.port}")
        
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._ping_loop())
    
    async def stop(self):
        """Stop the WebSocket server"""
        if self.runner:
            await self.runner.cleanup()
    
    async def _cleanup_loop(self):
        """Periodically clean up expired pairings and requests"""
        while True:
            await asyncio.sleep(60)
            
            now = datetime.now()
            expired_pairings = [
                code for code, p in self.pending_pairings.items()
                if now > p.expires_at
            ]
            for code in expired_pairings:
                del self.pending_pairings[code]
            
            expired_requests = [
                rid for rid, r in self.pending_requests.items()
                if (now - r.created_at).total_seconds() > 60
            ]
            for rid in expired_requests:
                req = self.pending_requests.pop(rid)
                if not req.future.done():
                    req.future.set_exception(asyncio.TimeoutError())
    
    async def _ping_loop(self):
        """Periodically ping connected nodes"""
        while True:
            await asyncio.sleep(30)
            
            for node_id, node in list(self.connected_nodes.items()):
                try:
                    await node.ws.send_json({"type": "ping"})
                except Exception:
                    pass


_bridge_server: Optional[NodeBridgeServer] = None


def get_node_bridge() -> NodeBridgeServer:
    """Get the global node bridge server"""
    global _bridge_server
    if _bridge_server is None:
        _bridge_server = NodeBridgeServer(port=5000)
    return _bridge_server


async def start_node_bridge():
    """Start the node bridge server"""
    bridge = get_node_bridge()
    await bridge.start()
    return bridge
