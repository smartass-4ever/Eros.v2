"""
Tool-calling layer for Eros.
Defines tool schemas for the Together API and executes them via local_actions.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple

# ── Tool schemas (OpenAI-compatible function format) ──────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Open an application on the computer",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "App name e.g. chrome, spotify, discord, vscode, terminal"}
                },
                "required": ["app"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the default browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to open"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, e.g. ~/Documents, ~/Desktop"},
                    "pattern": {"type": "string", "description": "Glob pattern, default *"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full file path"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file with given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full file path"},
                    "content": {"type": "string", "description": "Text content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "Open a file with its default application",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full file path to open"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Get the current clipboard content",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Copy text to the clipboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text to copy"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot and save it to the Desktop",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Get CPU, RAM, disk usage and OS info",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ── Web search ────────────────────────────────────────────────────────────────

def _web_search(query: str) -> Dict:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return {"ok": True, "results": "No results found."}
        snippets = []
        for r in results:
            snippets.append(f"{r.get('title', '')}: {r.get('body', '')[:200]}")
        return {"ok": True, "results": "\n\n".join(snippets)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Execute a single tool call ────────────────────────────────────────────────

def execute_tool(name: str, args: Dict) -> str:
    from local_actions import execute_local_action

    if name == "web_search":
        result = _web_search(args.get("query", ""))
    else:
        result = execute_local_action(name, args)

    if result.get("ok"):
        result.pop("ok", None)
        if not result:
            return "Done."
        return json.dumps(result, indent=None)
    else:
        return f"Failed: {result.get('error', 'unknown error')}"


# ── Main: run tool-calling pass against Together API ─────────────────────────

def run_tool_pass(
    user_message: str,
    conversation_history: List[Dict],
    system_prompt: str = "",
    api_key: str = None,
) -> Tuple[Optional[str], List[Dict]]:
    """
    Sends the user message to Together with tool definitions.
    If the model calls a tool, executes it and returns (tool_summary, updated_history).
    If no tool call, returns (None, history_unchanged).

    Returns:
        (action_summary, updated_history)
        action_summary is None if no tool was called.
    """
    key = api_key or os.environ.get("TOGETHER_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not key:
        return None, conversation_history

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Include last 6 turns for context (tool pass stays cheap)
    for turn in conversation_history[-6:]:
        messages.append(turn)

    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 300,
                "temperature": 0.2,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None, conversation_history

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        # No tool call — model just responded normally, skip
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return None, conversation_history

        # Execute all tool calls (usually just one)
        action_parts = []
        tool_results = []

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"].get("arguments", "{}"))
            except Exception:
                fn_args = {}

            print(f"[ACTION] {fn_name}({json.dumps(fn_args)})")
            result_str = execute_tool(fn_name, fn_args)
            action_parts.append(f"{fn_name}: {result_str}")

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": fn_name,
                "content": result_str,
            })

        # Build updated history with assistant tool_calls + tool results
        updated = list(conversation_history)
        updated.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        updated.extend(tool_results)

        summary = "\n".join(action_parts)
        return summary, updated

    except Exception as e:
        print(f"[TOOLS] Tool pass error: {e}")
        return None, conversation_history
