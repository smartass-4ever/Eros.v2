"""
Action Orchestrator for Eros Agentic System
Coordinates intent detection, safety checks, and tool execution
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from agentic_actions import LLMIntentDetector, ActionIntent, ActionType
from cloud_tools import execute_cloud_tool, ToolResult
from experience_bus import emit_action_experience, ExperienceType
from cns_database import ActionAuditLog, UserToolPermission, LocalNodeConnection, CNSDatabase


@dataclass
class ActionContext:
    """Context for action execution"""
    user_id: str
    has_local_node: bool = False
    connected_services: list = None
    user_permissions: Dict[str, bool] = None
    
    def __post_init__(self):
        self.connected_services = self.connected_services or []
        self.user_permissions = self.user_permissions or {}


@dataclass
class ActionOutcome:
    """Rich outcome for natural language integration with Eros personality"""
    is_action: bool
    action_type: str = None
    success: bool = None
    result_data: Any = None
    error: str = None
    intent: Any = None
    needs_setup: str = None
    needs_confirmation: bool = False
    confirmation_id: str = None
    
    def to_context_for_llm(self) -> str:
        """Generate context that can be injected into Eros's response generation"""
        if not self.is_action:
            return ""
        
        if self.needs_setup:
            return f"[ACTION CONTEXT: User requested {self.action_type}, but {self.needs_setup}. Explain this naturally and offer to help set it up.]"
        
        if self.needs_confirmation:
            return f"[ACTION CONTEXT: User requested {self.action_type}. This action needs confirmation before executing. Ask naturally if they want you to proceed.]"
        
        if not self.success:
            return f"[ACTION CONTEXT: User asked you to {self.action_type}. It failed: {self.error}. Respond naturally with empathy about the failure and offer alternatives.]"
        
        if self.action_type == "send_email":
            recipient = self.result_data.get("recipient", "them") if self.result_data else "them"
            return f"[ACTION CONTEXT: You just successfully sent an email to {recipient}. Confirm this naturally in your characteristic style - be cool about it like you do this all the time.]"
        
        if self.action_type == "web_search":
            return f"[ACTION CONTEXT: You searched the web for them. Share these results naturally, integrating the information into your response: {self.result_data}]"
        
        if self.action_type == "check_weather":
            return f"[ACTION CONTEXT: You checked the weather for them. Here's the data - share it naturally: {self.result_data}]"
        
        if self.action_type == "check_news":
            return f"[ACTION CONTEXT: You fetched the news for them. Share the headlines naturally: {self.result_data}]"
        
        if self.action_type == "check_stocks":
            return f"[ACTION CONTEXT: You checked stock/crypto prices. Share this data naturally: {self.result_data}]"
        
        if "local_" in self.action_type:
            return f"[ACTION CONTEXT: You executed {self.action_type} on their computer. Result: {self.result_data}. Confirm naturally like a capable assistant would.]"
        
        return f"[ACTION CONTEXT: You completed {self.action_type} for them. Result: {self.result_data}. Confirm naturally.]"


class ActionOrchestrator:
    """Orchestrates the full agentic action flow from intent to execution"""
    
    def __init__(self):
        self.intent_detector = LLMIntentDetector()
        self._pending_actions: Dict[str, ActionIntent] = {}
        self._db = None
    
    @property
    def db(self):
        if self._db is None:
            try:
                self._db = CNSDatabase()
            except Exception:
                pass
        return self._db
    
    async def process_message(self, user_id: str, message: str, 
                             has_local_node: bool = False) -> Tuple[bool, Optional[str], Optional[ActionIntent]]:
        """Process a user message and detect if it's an action request
        
        Returns:
            (is_action, action_response, intent)
            - is_action: True if this is an action request
            - action_response: Text response about the action (or None if just chat)
            - intent: The detected intent (if any)
        """
        context = await self._build_context(user_id, has_local_node)
        
        intent = await self.intent_detector.detect_intent(message, {
            "has_local_node": context.has_local_node,
            "connected_services": context.connected_services
        })
        
        if intent.action_type == ActionType.CHAT or intent.confidence < 0.6:
            return False, None, None
        
        if not self._check_permission(context, intent.action_type):
            return True, f"I'd love to help with that, but you haven't enabled {intent.action_type.value} actions yet. Want me to set that up?", intent
        
        if self._requires_local_node(intent.action_type) and not context.has_local_node:
            return True, "I can do that, but it requires the Eros Local Node on your computer. Want me to help you set it up?", intent
        
        if intent.requires_confirmation:
            action_id = f"{user_id}_{datetime.now().timestamp()}"
            self._pending_actions[action_id] = intent
            return True, self._format_confirmation_request(intent, action_id), intent
        
        result = await self._execute_action(user_id, intent)
        return True, self._format_result(intent, result), intent
    
    async def confirm_action(self, action_id: str, approved: bool) -> Tuple[bool, str]:
        """Handle user confirmation for a pending action"""
        intent = self._pending_actions.pop(action_id, None)
        if not intent:
            return False, "That action has expired. Please try again."
        
        if not approved:
            emit_action_experience(
                user_id=action_id.split("_")[0],
                action_type=intent.action_type.value,
                action_params=intent.extracted_slots,
                success=False,
                error="User rejected action"
            )
            return True, "Got it, I won't do that."
        
        user_id = action_id.split("_")[0]
        result = await self._execute_action(user_id, intent)
        return True, self._format_result(intent, result)
    
    async def _execute_action(self, user_id: str, intent: ActionIntent) -> ToolResult:
        """Execute an action and log it"""
        action_type = intent.action_type.value
        params = intent.extracted_slots
        
        await self._log_action_start(user_id, action_type, params, intent.requires_confirmation)
        
        if self._is_cloud_action(intent.action_type):
            result = await execute_cloud_tool(action_type, params, user_id)
        elif self._is_local_action(intent.action_type):
            result = await self._execute_local_action(user_id, intent)
        else:
            result = ToolResult(success=False, error="Unknown action type")
        
        await self._log_action_result(user_id, action_type, result)
        
        return result
    
    async def _execute_local_action(self, user_id: str, intent: ActionIntent) -> ToolResult:
        """Execute a local action via the Local Node"""
        try:
            from node_bridge_server import get_node_bridge
            bridge = get_node_bridge()
            
            if not bridge.user_has_node(user_id):
                return ToolResult(
                    success=False,
                    error="No local node connected",
                    display_text="I'd love to help with that, but your Eros Local Node isn't connected. Run `!pair` in Discord to connect your computer!"
                )
            
            operation_map = {
                ActionType.LOCAL_FILE_OPEN: "open_file",
                ActionType.LOCAL_APP_LAUNCH: "launch_app",
                ActionType.LOCAL_CLIPBOARD: "get_clipboard",
                ActionType.LOCAL_SCREENSHOT: "take_screenshot",
                ActionType.LOCAL_SYSTEM_INFO: "get_system_info",
                ActionType.EXPLORE_WINDOW: "explore_window",
                ActionType.CLICK_CONTROL: "click_control",
                ActionType.TYPE_IN_CONTROL: "type_in_control",
                ActionType.SCROLL_CONTROL: "scroll_control",
                ActionType.GET_WINDOW_LIST: "get_window_list",
                ActionType.FOCUS_WINDOW: "focus_window",
            }
            
            operation = operation_map.get(intent.action_type)
            if not operation:
                return ToolResult(success=False, error=f"Unknown local action: {intent.action_type}")
            
            params = intent.extracted_slots.copy()
            if intent.action_type == ActionType.LOCAL_FILE_OPEN:
                params["path"] = params.get("file_path") or params.get("query", "")
            elif intent.action_type == ActionType.LOCAL_APP_LAUNCH:
                params["app_name"] = params.get("app") or params.get("query", "")
            elif intent.action_type == ActionType.FOCUS_WINDOW:
                params["window_name"] = params.get("window_name") or params.get("query", "")
            
            result = await bridge.execute_on_node(
                user_id=user_id,
                operation=operation,
                params=params,
                require_confirmation=intent.requires_confirmation
            )
            
            if result.get("success"):
                emit_action_experience(
                    user_id=user_id,
                    action_type=intent.action_type.value,
                    action_params=params,
                    success=True,
                    result_data=result.get("data")
                )
                return ToolResult(
                    success=True,
                    data=result.get("data"),
                    display_text=result.get("data", {}).get("display_text", "Done!")
                )
            else:
                emit_action_experience(
                    user_id=user_id,
                    action_type=intent.action_type.value,
                    action_params=params,
                    success=False,
                    error=result.get("error")
                )
                return ToolResult(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
                
        except ImportError:
            return ToolResult(
                success=False,
                error="Node bridge not available",
                display_text="Local actions aren't set up yet. Coming soon!"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _build_context(self, user_id: str, has_local_node: bool) -> ActionContext:
        """Build context for action execution"""
        connected_services = []
        permissions = {}
        
        if self.db:
            try:
                session = self.db.Session()
                
                from cns_database import ConnectedService
                services = session.query(ConnectedService).filter(
                    ConnectedService.user_id == user_id,
                    ConnectedService.is_connected == True
                ).all()
                connected_services = [s.service_name for s in services]
                
                perms = session.query(UserToolPermission).filter(
                    UserToolPermission.user_id == user_id
                ).all()
                permissions = {p.tool_name: p.is_enabled for p in perms}
                
                if not has_local_node:
                    node = session.query(LocalNodeConnection).filter(
                        LocalNodeConnection.user_id == user_id,
                        LocalNodeConnection.is_active == True
                    ).first()
                    if node:
                        has_local_node = True
                
                session.close()
            except Exception as e:
                print(f"[ActionOrchestrator] Error building context: {e}")
        
        return ActionContext(
            user_id=user_id,
            has_local_node=has_local_node,
            connected_services=connected_services,
            user_permissions=permissions
        )
    
    def _check_permission(self, context: ActionContext, action_type: ActionType) -> bool:
        """Check if user has permission for this action"""
        tool_name = action_type.value
        if tool_name not in context.user_permissions:
            return True
        return context.user_permissions.get(tool_name, True)
    
    def _requires_local_node(self, action_type: ActionType) -> bool:
        """Check if action requires local node"""
        local_actions = {
            ActionType.LOCAL_FILE_OPEN,
            ActionType.LOCAL_FILE_MOVE,
            ActionType.LOCAL_APP_LAUNCH,
            ActionType.LOCAL_CLIPBOARD,
            ActionType.LOCAL_SCREENSHOT,
            ActionType.LOCAL_SYSTEM_INFO,
            ActionType.LOCAL_RUN_SCRIPT,
            ActionType.EXPLORE_WINDOW,
            ActionType.CLICK_CONTROL,
            ActionType.TYPE_IN_CONTROL,
            ActionType.SCROLL_CONTROL,
            ActionType.GET_WINDOW_LIST,
            ActionType.FOCUS_WINDOW,
        }
        return action_type in local_actions
    
    def _is_cloud_action(self, action_type: ActionType) -> bool:
        """Check if action is a cloud action"""
        cloud_actions = {
            ActionType.WEB_SEARCH,
            ActionType.CHECK_WEATHER,
            ActionType.CHECK_NEWS,
            ActionType.CHECK_STOCKS,
            ActionType.SEND_EMAIL,
        }
        return action_type in cloud_actions
    
    def _is_local_action(self, action_type: ActionType) -> bool:
        """Check if action is a local action"""
        return self._requires_local_node(action_type)
    
    def _format_confirmation_request(self, intent: ActionIntent, action_id: str) -> str:
        """Format a confirmation request for the user"""
        action_descriptions = {
            ActionType.SEND_EMAIL: "send an email",
            ActionType.CALENDAR_EVENT: "create a calendar event",
            ActionType.LOCAL_FILE_MOVE: "move files on your computer",
            ActionType.LOCAL_RUN_SCRIPT: "run a script on your computer",
        }
        
        desc = action_descriptions.get(intent.action_type, intent.action_type.value)
        params_str = ", ".join(f"{k}: {v}" for k, v in intent.extracted_slots.items() if v)
        
        return f"I'll {desc} with these details:\n{params_str}\n\nReact with ✅ to confirm or ❌ to cancel. (ID: {action_id})"
    
    def _format_result(self, intent: ActionIntent, result: ToolResult) -> str:
        """Format action result for user"""
        if result.success:
            return result.display_text
        else:
            return f"I tried to {intent.action_type.value} but hit a snag: {result.error}"
    
    async def _log_action_start(self, user_id: str, action_type: str, params: Dict, requires_confirmation: bool):
        """Log action start to database"""
        if not self.db:
            return
        
        try:
            session = self.db.Session()
            log = ActionAuditLog(
                user_id=user_id,
                action_type=action_type,
                action_params=params,
                status="pending" if requires_confirmation else "executing",
                required_confirmation=requires_confirmation
            )
            session.add(log)
            session.commit()
            session.close()
        except Exception as e:
            print(f"[ActionOrchestrator] Error logging action: {e}")
    
    async def _log_action_result(self, user_id: str, action_type: str, result: ToolResult):
        """Log action result to database"""
        if not self.db:
            return
        
        try:
            session = self.db.Session()
            log = session.query(ActionAuditLog).filter(
                ActionAuditLog.user_id == user_id,
                ActionAuditLog.action_type == action_type
            ).order_by(ActionAuditLog.created_at.desc()).first()
            
            if log:
                log.status = "success" if result.success else "failure"
                log.result = result.display_text if result.success else None
                log.error_message = result.error
                log.executed_at = datetime.utcnow()
                session.commit()
            
            session.close()
        except Exception as e:
            print(f"[ActionOrchestrator] Error logging result: {e}")


_orchestrator: Optional[ActionOrchestrator] = None


def get_action_orchestrator() -> ActionOrchestrator:
    """Get the global action orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ActionOrchestrator()
    return _orchestrator


async def process_agentic_message(user_id: str, message: str, 
                                  has_local_node: bool = False) -> Tuple[bool, Optional[str], Optional[ActionIntent]]:
    """Convenience function to process a message for agentic intent"""
    orchestrator = get_action_orchestrator()
    return await orchestrator.process_message(user_id, message, has_local_node)


async def process_action_naturally(user_id: str, message: str, 
                                    has_local_node: bool = False) -> ActionOutcome:
    """Process a message and return rich ActionOutcome for natural LLM integration.
    
    This is the preferred method for integrating actions with Eros's personality.
    Instead of returning templated responses, it returns context that can be
    injected into the main CNS response generation.
    """
    orchestrator = get_action_orchestrator()
    context = await orchestrator._build_context(user_id, has_local_node)
    
    intent = await orchestrator.intent_detector.detect_intent(message, {
        "has_local_node": context.has_local_node,
        "connected_services": context.connected_services
    })
    
    if intent.action_type == ActionType.CHAT or intent.confidence < 0.6:
        return ActionOutcome(is_action=False)
    
    if not orchestrator._check_permission(context, intent.action_type):
        return ActionOutcome(
            is_action=True,
            action_type=intent.action_type.value,
            intent=intent,
            needs_setup=f"you haven't enabled {intent.action_type.value} actions yet"
        )
    
    if orchestrator._requires_local_node(intent.action_type) and not context.has_local_node:
        return ActionOutcome(
            is_action=True,
            action_type=intent.action_type.value,
            intent=intent,
            needs_setup="the Eros Local Node isn't connected on their computer"
        )
    
    if intent.requires_confirmation:
        action_id = f"{user_id}_{datetime.now().timestamp()}"
        orchestrator._pending_actions[action_id] = intent
        return ActionOutcome(
            is_action=True,
            action_type=intent.action_type.value,
            intent=intent,
            needs_confirmation=True,
            confirmation_id=action_id
        )
    
    result = await orchestrator._execute_action(user_id, intent)
    
    return ActionOutcome(
        is_action=True,
        action_type=intent.action_type.value,
        success=result.success,
        result_data=result.data or result.display_text,
        error=result.error,
        intent=intent
    )
