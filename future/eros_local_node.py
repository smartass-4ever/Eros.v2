#!/usr/bin/env python3
"""
Eros Local Node - Runs on user's computer to enable local actions

This daemon connects to the Eros brain server and executes approved local commands.
It uses a secure WebSocket connection and maintains a strict allowlist of operations.

Usage:
    python eros_local_node.py --pair             # Generate pairing code
    python eros_local_node.py --connect CODE     # Connect with Discord pairing code
    python eros_local_node.py                    # Run (after paired)
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import argparse

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)

NODE_VERSION = "1.0.0"
CONFIG_DIR = Path.home() / ".eros_node"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class NodeConfig:
    """Local node configuration"""
    node_id: str
    node_name: str
    paired_user_id: Optional[str] = None
    server_url: str = "wss://65b5afa0-c9d5-42ef-992e-431c839a35fb-00-1dnptrecv18se.worf.replit.dev/node"
    allowed_directories: List[str] = None
    allowed_apps: List[str] = None
    require_confirmation: bool = True

    def __post_init__(self):
        if self.allowed_directories is None:
            home = str(Path.home())
            self.allowed_directories = [
                home,
                str(Path.home() / "Documents"),
                str(Path.home() / "Downloads"),
                str(Path.home() / "Desktop"),
            ]
        if self.allowed_apps is None:
            self.allowed_apps = [
                "chrome",
                "firefox",
                "safari",
                "edge",
                "code",
                "vscode",
                "sublime",
                "spotify",
                "discord",
                "slack",
                "finder",
                "explorer",
                "terminal",
                "iterm",
                "cmd",
                "powershell",
            ]


class LocalCommandExecutor:
    """Executes local commands with safety checks"""

    ALLOWED_OPERATIONS = {
        "open_file",
        "launch_app",
        "get_clipboard",
        "set_clipboard",
        "take_screenshot",
        "get_system_info",
        "list_files",
        "explore_window",
        "click_control",
        "type_in_control",
        "scroll_control",
        "get_window_list",
        "focus_window",
    }

    def __init__(self, config: NodeConfig):
        self.config = config
        self.os_type = platform.system().lower()

    def execute(self, operation: str, params: Dict[str,
                                                   Any]) -> Dict[str, Any]:
        """Execute a local operation safely"""
        if operation not in self.ALLOWED_OPERATIONS:
            return {
                "success": False,
                "error": f"Operation not allowed: {operation}"
            }

        try:
            handler = getattr(self, f"_do_{operation}", None)
            if handler:
                return handler(params)
            return {"success": False, "error": f"No handler for {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_open_file(self, params: Dict) -> Dict:
        """Open a file or folder"""
        filepath = params.get("path", "")
        if not filepath:
            return {"success": False, "error": "No file path provided"}

        path = Path(filepath).expanduser().resolve()

        if not self._is_path_allowed(path):
            return {
                "success": False,
                "error": f"Path not in allowed directories: {path}"
            }

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            if self.os_type == "darwin":
                subprocess.run(["open", str(path)], check=True)
            elif self.os_type == "windows":
                os.startfile(str(path))
            else:
                subprocess.run(["xdg-open", str(path)], check=True)

            return {"success": True, "message": f"Opened: {path.name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_launch_app(self, params: Dict) -> Dict:
        """Launch an application"""
        app_name = params.get("app", "").lower()
        if not app_name:
            return {"success": False, "error": "No app name provided"}

        if not any(allowed.lower() in app_name or app_name in allowed.lower()
                   for allowed in self.config.allowed_apps):
            return {
                "success": False,
                "error": f"App not in allowed list: {app_name}"
            }

        try:
            if self.os_type == "darwin":
                subprocess.run(["open", "-a", app_name], check=True)
            elif self.os_type == "windows":
                subprocess.run(["start", app_name], shell=True, check=True)
            else:
                subprocess.run([app_name], start_new_session=True)

            return {"success": True, "message": f"Launched: {app_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_get_clipboard(self, params: Dict) -> Dict:
        """Get clipboard contents"""
        try:
            if self.os_type == "darwin":
                result = subprocess.run(["pbpaste"],
                                        capture_output=True,
                                        text=True)
                content = result.stdout
            elif self.os_type == "windows":
                import subprocess
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True,
                    text=True)
                content = result.stdout.strip()
            else:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True,
                    text=True)
                content = result.stdout

            return {"success": True, "content": content[:1000]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_set_clipboard(self, params: Dict) -> Dict:
        """Set clipboard contents"""
        content = params.get("content", "")
        if not content:
            return {"success": False, "error": "No content provided"}

        try:
            if self.os_type == "darwin":
                subprocess.run(["pbcopy"], input=content.encode(), check=True)
            elif self.os_type == "windows":
                subprocess.run(["clip"], input=content.encode(), check=True)
            else:
                subprocess.run(["xclip", "-selection", "clipboard"],
                               input=content.encode(),
                               check=True)

            return {"success": True, "message": "Copied to clipboard"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_take_screenshot(self, params: Dict) -> Dict:
        """Take a screenshot"""
        output_path = Path.home(
        ) / "Desktop" / f"eros_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        try:
            if self.os_type == "darwin":
                subprocess.run(["screencapture", str(output_path)], check=True)
            elif self.os_type == "windows":
                try:
                    import PIL.ImageGrab
                    img = PIL.ImageGrab.grab()
                    img.save(str(output_path))
                except ImportError:
                    return {
                        "success": False,
                        "error": "PIL not installed for screenshots"
                    }
            else:
                subprocess.run(["gnome-screenshot", "-f",
                                str(output_path)],
                               check=True)

            return {
                "success": True,
                "path": str(output_path),
                "message": f"Screenshot saved to {output_path.name}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_get_system_info(self, params: Dict) -> Dict:
        """Get basic system information"""
        import shutil

        try:
            total, used, free = shutil.disk_usage("/")

            info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "hostname": platform.node(),
                "cpu": platform.processor(),
                "disk_total_gb": round(total / (1024**3), 1),
                "disk_free_gb": round(free / (1024**3), 1),
                "home_dir": str(Path.home()),
            }

            try:
                import psutil
                info["cpu_percent"] = psutil.cpu_percent()
                info["memory_percent"] = psutil.virtual_memory().percent
                info["battery_percent"] = getattr(psutil.sensors_battery(),
                                                  'percent', None)
            except ImportError:
                pass

            return {"success": True, "info": info}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_list_files(self, params: Dict) -> Dict:
        """List files in a directory"""
        dirpath = params.get("path", str(Path.home()))
        path = Path(dirpath).expanduser().resolve()

        if not self._is_path_allowed(path):
            return {
                "success": False,
                "error": f"Path not in allowed directories: {path}"
            }

        if not path.exists() or not path.is_dir():
            return {"success": False, "error": f"Directory not found: {path}"}

        try:
            files = []
            for item in sorted(path.iterdir())[:50]:
                files.append({
                    "name":
                    item.name,
                    "is_dir":
                    item.is_dir(),
                    "size":
                    item.stat().st_size if item.is_file() else None
                })

            return {"success": True, "path": str(path), "files": files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories"""
        path_str = str(path.resolve())
        for allowed in self.config.allowed_directories:
            if path_str.startswith(str(Path(allowed).resolve())):
                return True
        return False

    def _do_explore_window(self, params: Dict) -> Dict:
        """Scan the active window and return a map of UI elements (buttons, fields, etc.)"""
        try:
            depth = params.get("depth", 2)
            
            if self.os_type == "windows":
                return self._explore_window_windows(depth)
            elif self.os_type == "darwin":
                return self._explore_window_macos(depth)
            else:
                return {"success": False, "error": "UI automation not supported on Linux yet"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _explore_window_windows(self, depth: int = 2) -> Dict:
        """Windows: Use pywinauto to scan the active window's UI tree"""
        try:
            from pywinauto import Desktop
            import io
            from contextlib import redirect_stdout
            
            desktop = Desktop(backend="uia")
            top_window = desktop.active_window()
            window_title = top_window.window_text()
            
            f = io.StringIO()
            with redirect_stdout(f):
                top_window.print_control_identifiers(depth=depth)
            
            ui_map_full = f.getvalue()
            
            controls = []
            for line in ui_map_full.split('\n'):
                line = line.strip()
                if not line:
                    continue
                control_type = None
                control_name = None
                if ' - ' in line:
                    parts = line.split(' - ', 1)
                    control_type = parts[0].strip().strip('|').strip()
                    if len(parts) > 1:
                        name_part = parts[1].strip()
                        if name_part.startswith("'") and "'" in name_part[1:]:
                            control_name = name_part.split("'")[1]
                        else:
                            control_name = name_part
                    if control_type and control_name:
                        controls.append({
                            "type": control_type,
                            "name": control_name,
                            "clickable": control_type in ["Button", "MenuItem", "Link", "CheckBox", "RadioButton", "Tab"]
                        })
            
            buttons = [c for c in controls if c.get("clickable")]
            text_fields = [c for c in controls if c.get("type") in ["Edit", "Text", "ComboBox", "TextBox"]]
            
            return {
                "success": True,
                "window_title": window_title,
                "buttons": buttons[:30],
                "text_fields": text_fields[:20],
                "all_controls_count": len(controls),
                "summary": f"Found {len(buttons)} clickable elements and {len(text_fields)} text fields in '{window_title}'"
            }
        except ImportError:
            return {
                "success": False,
                "error": "pywinauto not installed. Run: pip install pywinauto"
            }
        except Exception as e:
            return {"success": False, "error": f"Windows UI scan failed: {str(e)}"}

    def _explore_window_macos(self, depth: int = 2) -> Dict:
        """macOS: Use AppleScript/System Events to scan the frontmost app"""
        try:
            script = '''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set appName to name of frontApp
                set windowName to ""
                try
                    set windowName to name of front window of frontApp
                end try
                
                set buttonList to {}
                set textFieldList to {}
                
                try
                    set allButtons to buttons of front window of frontApp
                    repeat with btn in allButtons
                        set end of buttonList to name of btn
                    end repeat
                end try
                
                try
                    set allTextFields to text fields of front window of frontApp
                    repeat with tf in allTextFields
                        set tfValue to ""
                        try
                            set tfValue to value of tf
                        end try
                        set end of textFieldList to tfValue
                    end repeat
                end try
                
                return {appName, windowName, buttonList, textFieldList}
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            
            output = result.stdout.strip()
            
            buttons = []
            text_fields = []
            app_name = "Unknown"
            window_name = ""
            
            if output:
                parts = output.split(", ")
                if len(parts) >= 1:
                    app_name = parts[0].strip()
                if len(parts) >= 2:
                    window_name = parts[1].strip()
                
                import re
                button_match = re.search(r'\{([^}]*)\}', output)
                if button_match:
                    buttons = [b.strip() for b in button_match.group(1).split(',') if b.strip()]
            
            return {
                "success": True,
                "window_title": f"{app_name} - {window_name}",
                "buttons": [{"type": "Button", "name": b, "clickable": True} for b in buttons[:30]],
                "text_fields": [{"type": "TextField", "name": tf} for tf in text_fields[:20]],
                "summary": f"Found {len(buttons)} buttons in '{app_name}'"
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "AppleScript timed out"}
        except Exception as e:
            return {"success": False, "error": f"macOS UI scan failed: {str(e)}"}

    def _do_click_control(self, params: Dict) -> Dict:
        """Click a UI control by name"""
        control_name = params.get("control_name", "")
        if not control_name:
            return {"success": False, "error": "No control name provided"}
        
        try:
            if self.os_type == "windows":
                return self._click_control_windows(control_name)
            elif self.os_type == "darwin":
                return self._click_control_macos(control_name)
            else:
                return {"success": False, "error": "UI automation not supported on Linux"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _click_control_windows(self, control_name: str) -> Dict:
        """Windows: Click a control using pywinauto"""
        try:
            from pywinauto import Desktop
            
            desktop = Desktop(backend="uia")
            top_window = desktop.active_window()
            
            try:
                control = top_window.child_window(title=control_name, control_type="Button")
                control.click()
                return {"success": True, "message": f"Clicked button '{control_name}'"}
            except:
                pass
            
            try:
                control = top_window.child_window(title_re=f".*{control_name}.*")
                control.click()
                return {"success": True, "message": f"Clicked control '{control_name}'"}
            except:
                pass
            
            return {"success": False, "error": f"Control '{control_name}' not found in active window"}
        except ImportError:
            return {"success": False, "error": "pywinauto not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _click_control_macos(self, control_name: str) -> Dict:
        """macOS: Click a control using AppleScript"""
        try:
            script = f'''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                try
                    click button "{control_name}" of front window of frontApp
                    return "success"
                end try
                try
                    click menu item "{control_name}" of menu bar 1 of frontApp
                    return "success"
                end try
            end tell
            return "not_found"
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "success" in result.stdout:
                return {"success": True, "message": f"Clicked '{control_name}'"}
            else:
                return {"success": False, "error": f"Control '{control_name}' not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_type_in_control(self, params: Dict) -> Dict:
        """Type text into a UI control"""
        text = params.get("text", "")
        control_name = params.get("control_name", "")
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        try:
            if self.os_type == "windows":
                return self._type_in_control_windows(text, control_name)
            elif self.os_type == "darwin":
                return self._type_in_control_macos(text, control_name)
            else:
                return {"success": False, "error": "UI automation not supported on Linux"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _type_in_control_windows(self, text: str, control_name: str = "") -> Dict:
        """Windows: Type into a control"""
        try:
            from pywinauto import Desktop
            
            desktop = Desktop(backend="uia")
            top_window = desktop.active_window()
            
            if control_name:
                try:
                    control = top_window.child_window(title_re=f".*{control_name}.*", control_type="Edit")
                    control.set_edit_text(text)
                    return {"success": True, "message": f"Typed into '{control_name}'"}
                except:
                    pass
            
            top_window.type_keys(text, with_spaces=True)
            return {"success": True, "message": f"Typed text into active window"}
        except ImportError:
            return {"success": False, "error": "pywinauto not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _type_in_control_macos(self, text: str, control_name: str = "") -> Dict:
        """macOS: Type into the focused control"""
        try:
            escaped_text = text.replace('"', '\\"')
            script = f'''
            tell application "System Events"
                keystroke "{escaped_text}"
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return {"success": True, "message": f"Typed text"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_scroll_control(self, params: Dict) -> Dict:
        """Scroll in the active window"""
        direction = params.get("direction", "down")
        amount = params.get("amount", 3)
        
        try:
            if self.os_type == "windows":
                from pywinauto import Desktop
                desktop = Desktop(backend="uia")
                top_window = desktop.active_window()
                scroll_amount = amount if direction == "down" else -amount
                top_window.scroll(direction=direction, amount=scroll_amount)
                return {"success": True, "message": f"Scrolled {direction}"}
            elif self.os_type == "darwin":
                scroll_dir = "down" if direction == "down" else "up"
                script = f'''
                tell application "System Events"
                    repeat {amount} times
                        key code 125 using {{}} -- down arrow
                    end repeat
                end tell
                '''
                if direction == "up":
                    script = script.replace("125", "126")
                    
                subprocess.run(["osascript", "-e", script], timeout=5)
                return {"success": True, "message": f"Scrolled {direction}"}
            else:
                return {"success": False, "error": "Not supported on Linux"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_get_window_list(self, params: Dict) -> Dict:
        """Get list of open windows"""
        try:
            if self.os_type == "windows":
                from pywinauto import Desktop
                desktop = Desktop(backend="uia")
                windows = []
                for win in desktop.windows():
                    if win.is_visible() and win.window_text():
                        windows.append({
                            "title": win.window_text(),
                            "class": win.class_name()
                        })
                return {"success": True, "windows": windows[:20]}
            elif self.os_type == "darwin":
                script = '''
                tell application "System Events"
                    set windowList to {}
                    repeat with proc in (application processes whose visible is true)
                        set appName to name of proc
                        try
                            repeat with w in windows of proc
                                set end of windowList to appName & ": " & name of w
                            end repeat
                        end try
                    end repeat
                    return windowList
                end tell
                '''
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
                windows = [w.strip() for w in result.stdout.split(",") if w.strip()]
                return {"success": True, "windows": [{"title": w} for w in windows[:20]]}
            else:
                return {"success": False, "error": "Not supported on Linux"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _do_focus_window(self, params: Dict) -> Dict:
        """Focus/activate a window by name"""
        window_name = params.get("window_name", "")
        if not window_name:
            return {"success": False, "error": "No window name provided"}
        
        try:
            if self.os_type == "windows":
                from pywinauto import Desktop
                desktop = Desktop(backend="uia")
                for win in desktop.windows():
                    if window_name.lower() in win.window_text().lower():
                        win.set_focus()
                        return {"success": True, "message": f"Focused '{win.window_text()}'"}
                return {"success": False, "error": f"Window '{window_name}' not found"}
            elif self.os_type == "darwin":
                script = f'''
                tell application "System Events"
                    set frontmost of first application process whose name contains "{window_name}" to true
                end tell
                '''
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return {"success": True, "message": f"Focused '{window_name}'"}
                else:
                    return {"success": False, "error": f"Could not focus '{window_name}'"}
            else:
                return {"success": False, "error": "Not supported on Linux"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ErosLocalNode:
    """Main Local Node daemon"""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.executor = LocalCommandExecutor(config)
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.running = False
        self._session: Optional[aiohttp.ClientSession] = None

    def save_config(self):
        """Save configuration to disk"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(
                {
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "paired_user_id": self.config.paired_user_id,
                    "server_url": self.config.server_url,
                    "allowed_directories": self.config.allowed_directories,
                    "allowed_apps": self.config.allowed_apps,
                    "require_confirmation": self.config.require_confirmation,
                },
                f,
                indent=2)
        print(f"[Eros Node] Config saved to {CONFIG_FILE}")

    @classmethod
    def load_config(cls) -> Optional['ErosLocalNode']:
        """Load configuration from disk"""
        if not CONFIG_FILE.exists():
            return None

        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
            config = NodeConfig(**data)
            return cls(config)
        except Exception as e:
            print(f"[Eros Node] Error loading config: {e}")
            return None

    def generate_pairing_code(self) -> str:
        """Generate a pairing code for Discord"""
        import secrets
        code = secrets.token_hex(4).upper()
        print(f"\n🔗 PAIRING CODE: {code}")
        print(f"\nIn Discord, tell Eros: 'pair my computer with code {code}'")
        return code

    async def connect_and_run(self):
        """Connect to server and run main loop"""
        self.running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print(f"[Eros Node] Starting local node: {self.config.node_name}")
        print(f"[Eros Node] OS: {platform.system()} {platform.release()}")
        print(
            f"[Eros Node] Allowed directories: {len(self.config.allowed_directories)}"
        )
        print(f"[Eros Node] Allowed apps: {len(self.config.allowed_apps)}")

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    self._session = session
                    print(f"[Eros Node] Connecting to server...")

                    headers = {
                        "X-Node-ID": self.config.node_id,
                        "X-User-ID": self.config.paired_user_id or "",
                        "X-Node-Version": NODE_VERSION,
                    }

                    async with session.ws_connect(self.config.server_url,
                                                  headers=headers,
                                                  heartbeat=30) as ws:
                        self.ws = ws
                        print("[Eros Node] Connected! Authenticating...")

                        auth_msg = {
                            "type": "auth",
                            "node_id": self.config.node_id,
                            "user_id": self.config.paired_user_id
                        }
                        if hasattr(self,
                                   '_pairing_code') and self._pairing_code:
                            auth_msg["pairing_code"] = self._pairing_code
                            self._pairing_code = None

                        await ws.send_json(auth_msg)

                        auth_response = await ws.receive_json()
                        if auth_response.get("type") == "auth_success":
                            new_user_id = auth_response.get("user_id")
                            if new_user_id and new_user_id != self.config.paired_user_id:
                                self.config.paired_user_id = new_user_id
                                self.save_config()
                                print(
                                    f"[Eros Node] Paired to user: {new_user_id}"
                                )
                            print("[Eros Node] Authenticated!")
                        elif auth_response.get("type") == "auth_failed":
                            print(
                                f"[Eros Node] Auth failed: {auth_response.get('error')}"
                            )
                            break

                        await self._send_status("connected")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(json.loads(msg.data)
                                                           )
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(
                                    f"[Eros Node] WebSocket error: {ws.exception()}"
                                )
                                break

            except aiohttp.ClientConnectorError:
                print("[Eros Node] Connection failed. Retrying in 10s...")
                await asyncio.sleep(10)
            except Exception as e:
                print(f"[Eros Node] Error: {e}. Retrying in 10s...")
                await asyncio.sleep(10)

        print("[Eros Node] Shutting down...")

    async def _handle_message(self, message: Dict):
        """Handle incoming message from server"""
        msg_type = message.get("type")

        if msg_type == "ping":
            await self._send({"type": "pong"})

        elif msg_type == "execute":
            operation = message.get("operation")
            params = message.get("params", {})
            request_id = message.get("request_id")

            print(f"[Eros Node] Executing: {operation}")
            result = self.executor.execute(operation, params)

            await self._send({
                "type": "result",
                "request_id": request_id,
                **result
            })

        elif msg_type == "confirm":
            operation = message.get("operation")
            params = message.get("params", {})
            request_id = message.get("request_id")

            if self.config.require_confirmation:
                print(f"\n⚠️  CONFIRMATION REQUIRED")
                print(f"Operation: {operation}")
                print(f"Params: {json.dumps(params, indent=2)}")
                response = input("Allow? (y/n): ").strip().lower()

                if response == 'y':
                    result = self.executor.execute(operation, params)
                    await self._send({
                        "type": "result",
                        "request_id": request_id,
                        **result
                    })
                else:
                    await self._send({
                        "type": "result",
                        "request_id": request_id,
                        "success": False,
                        "error": "User denied the operation"
                    })
            else:
                result = self.executor.execute(operation, params)
                await self._send({
                    "type": "result",
                    "request_id": request_id,
                    **result
                })

    async def _send(self, data: Dict):
        """Send message to server"""
        if self.ws:
            await self.ws.send_json(data)

    async def _send_status(self, status: str):
        """Send status update to server"""
        await self._send({
            "type":
            "status",
            "status":
            status,
            "node_id":
            self.config.node_id,
            "os":
            platform.system(),
            "os_version":
            platform.release(),
            "capabilities":
            list(LocalCommandExecutor.ALLOWED_OPERATIONS)
        })

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal"""
        self.running = False


def create_new_node() -> ErosLocalNode:
    """Create a new node configuration"""
    import secrets
    import socket

    node_id = secrets.token_hex(16)
    hostname = socket.gethostname()

    config = NodeConfig(node_id=node_id, node_name=f"{hostname}-eros-node")

    node = ErosLocalNode(config)
    node.save_config()

    print(f"[Eros Node] Created new node: {config.node_name}")
    print(f"[Eros Node] Node ID: {config.node_id[:8]}...")

    return node


def main():
    parser = argparse.ArgumentParser(
        description="Eros Local Node - Connect your computer to Eros")
    parser.add_argument("--pair",
                        action="store_true",
                        help="Generate a pairing code")
    parser.add_argument("--connect",
                        metavar="CODE",
                        help="Connect with a Discord pairing code")
    parser.add_argument("--reset",
                        action="store_true",
                        help="Reset node configuration")
    args = parser.parse_args()

    if args.reset:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
            print("[Eros Node] Configuration reset")
        return

    node = ErosLocalNode.load_config()

    if node is None:
        print(
            "[Eros Node] No existing configuration found. Creating new node..."
        )
        node = create_new_node()

    if args.pair:
        node.generate_pairing_code()
        return

    if args.connect:
        print(f"[Eros Node] Pairing with code: {args.connect}")
        node._pairing_code = args.connect.upper()
        asyncio.run(node.connect_and_run())
        return

    print(f"\n🤖 Eros Local Node v{NODE_VERSION}")
    print("=" * 40)

    if not node.config.paired_user_id:
        print("\n⚠️  This node is not paired yet!")
        print("Run with --pair to generate a pairing code")
        return

    asyncio.run(node.connect_and_run())


if __name__ == "__main__":
    main()
