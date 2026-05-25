"""
Screen awareness — Eros can see what's on your monitor.

Provides:
  - Active window (app name + title)
  - Screenshot capture
  - Screen text extraction (if pytesseract available)
  - Vision LLM description (if API key available)
  - Periodic background monitoring
"""
import os
import sys
import time
import base64
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# ── Active window ─────────────────────────────────────────────────────────────

def get_active_window() -> Dict[str, str]:
    """Get the currently focused window title and process name."""
    try:
        import win32gui
        import win32process
        import psutil

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(pid)
        return {
            "title": title,
            "app": proc.name().replace(".exe", ""),
            "pid": pid,
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: PowerShell
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-command",
             "Add-Type -AssemblyName Microsoft.VisualBasic; "
             "[Microsoft.VisualBasic.Interaction]::AppActivate((Get-Process | "
             "Where-Object {$_.MainWindowHandle -eq [System.IntPtr]([System.Runtime.InteropServices.Marshal]::GetActiveObject('Shell.Application').Windows() | Select-Object -ExpandProperty HWND -First 1)}).Id)"],
            capture_output=True, text=True, timeout=3
        )
    except Exception:
        pass

    # Simpler PowerShell fallback
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-command",
             "Get-Process | Where-Object {$_.MainWindowTitle} | "
             "Sort-Object CPU -Descending | Select-Object -First 1 | "
             "Select-Object ProcessName, MainWindowTitle | ConvertTo-Json"],
            capture_output=True, text=True, timeout=3
        )
        if result.stdout.strip():
            import json
            data = json.loads(result.stdout.strip())
            return {
                "title": data.get("MainWindowTitle", ""),
                "app": data.get("ProcessName", "").replace(".exe", ""),
                "pid": None,
            }
    except Exception:
        pass

    return {"title": "unknown", "app": "unknown", "pid": None}


# ── Screenshot ─────────────────────────────────────────────────────────────────

def capture_screenshot(save_path: Optional[str] = None) -> Optional[str]:
    """Capture the full screen. Returns path to saved image."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()

        if save_path is None:
            data_dir = Path(__file__).parent.parent / "data" / "screenshots"
            data_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(data_dir / f"screen_{ts}.png")

        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"[SCREEN] Screenshot failed: {e}")
        return None


def screenshot_to_base64(max_width: int = 1280) -> Optional[str]:
    """Capture screen and return as base64 string for vision API."""
    try:
        from PIL import ImageGrab, Image
        import io

        img = ImageGrab.grab()

        # Resize if too large (saves tokens)
        if img.width > max_width:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_width, new_h), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[SCREEN] Base64 capture failed: {e}")
        return None


# ── OCR text extraction ────────────────────────────────────────────────────────

def extract_screen_text() -> str:
    """Extract visible text from screen using OCR if available."""
    try:
        import pytesseract
        from PIL import ImageGrab
        img = ImageGrab.grab()
        text = pytesseract.image_to_string(img)
        return text[:3000]
    except ImportError:
        return ""
    except Exception:
        return ""


# ── Vision LLM description ────────────────────────────────────────────────────

async def describe_screen_with_llm(api_key: str = None) -> str:
    """
    Send screenshot to a vision-capable LLM and get a description.
    Uses OpenAI gpt-4o-mini vision if available, else Together.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")
    if not key:
        return ""

    img_b64 = screenshot_to_base64()
    if not img_b64:
        return ""

    # Try OpenAI vision first (cleaner vision support)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": "Describe what's on this screen concisely — app, content, what the person is doing. 2-3 sentences max."}
                        ]
                    }],
                    "max_tokens": 150
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # Try Together vision (Llama Vision)
    together_key = os.environ.get("TOGETHER_API_KEY")
    if together_key:
        try:
            import requests
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-Vision-Free",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": "What's on this screen? App, content, what the person is doing. 2-3 sentences."}
                        ]
                    }],
                    "max_tokens": 150
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    return ""


# ── Full context snapshot ─────────────────────────────────────────────────────

async def get_screen_context(use_vision: bool = True) -> Dict[str, Any]:
    """
    Single call to get everything Eros knows about the current screen.
    Returns a dict ready to inject into conversation context.
    """
    window = get_active_window()
    ocr_text = extract_screen_text()  # empty if no tesseract
    vision_desc = ""

    if use_vision:
        vision_desc = await describe_screen_with_llm()

    return {
        "active_app": window.get("app", ""),
        "window_title": window.get("title", ""),
        "screen_text": ocr_text,
        "vision_description": vision_desc,
        "captured_at": datetime.now().isoformat(),
    }


def get_screen_context_sync(use_vision: bool = False) -> Dict[str, Any]:
    """Synchronous version — just window + OCR, no async vision call."""
    window = get_active_window()
    ocr_text = extract_screen_text()
    return {
        "active_app": window.get("app", ""),
        "window_title": window.get("title", ""),
        "screen_text": ocr_text,
        "captured_at": datetime.now().isoformat(),
    }


# ── Background monitor ────────────────────────────────────────────────────────

class ScreenMonitor:
    """
    Runs in background, tracks what app/window is active.
    Eros reads `current` at any time to know what you're doing.
    """

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self.current: Dict[str, Any] = {}
        self._last_app = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_change = None  # optional callback(old, new)

    def on_app_change(self, callback):
        """Register callback called when focused app changes."""
        self._on_change = callback

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[SCREEN] Monitor started")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                window = get_active_window()
                self.current = {
                    "active_app": window.get("app", ""),
                    "window_title": window.get("title", ""),
                    "updated_at": datetime.now().isoformat(),
                }
                new_app = window.get("app", "")
                if new_app and new_app != self._last_app:
                    if self._on_change and self._last_app:
                        try:
                            self._on_change(self._last_app, new_app)
                        except Exception:
                            pass
                    self._last_app = new_app
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def describe(self) -> str:
        """Human-readable current screen state."""
        if not self.current:
            return "Screen state unknown."
        app = self.current.get("active_app", "")
        title = self.current.get("window_title", "")
        if title and app and title.lower() != app.lower():
            return f"You're in {app} — {title}"
        elif app:
            return f"You're in {app}"
        return "Screen state unknown."


# Singleton monitor
_monitor: Optional[ScreenMonitor] = None

def get_monitor() -> ScreenMonitor:
    global _monitor
    if _monitor is None:
        _monitor = ScreenMonitor()
    return _monitor
