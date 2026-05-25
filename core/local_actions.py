"""
Local action execution — direct OS control without WebSocket.
Eros calls this directly when running locally.
Covers: files, apps, clipboard, screenshot, browser, system info.
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

OS = platform.system().lower()  # "windows", "darwin", "linux"

HOME = Path.home()
ALLOWED_DIRS = [
    str(HOME),
    str(HOME / "Documents"),
    str(HOME / "Downloads"),
    str(HOME / "Desktop"),
    str(HOME / "Pictures"),
    str(HOME / "Music"),
    str(HOME / "Videos"),
]

ALLOWED_APPS = {
    "chrome", "firefox", "edge", "brave",
    "code", "vscode", "notepad", "notepad++",
    "spotify", "discord", "slack", "teams",
    "explorer", "cmd", "powershell", "terminal",
    "calculator", "paint", "word", "excel",
    "obs", "vlc", "steam",
}


def _allowed_path(path: Path) -> bool:
    path_str = str(path.resolve())
    return any(path_str.startswith(d) for d in ALLOWED_DIRS)


# ── File & folder operations ──────────────────────────────────────────────────

def open_file(path: str) -> Dict:
    p = Path(path).expanduser().resolve()
    if not _allowed_path(p):
        return {"ok": False, "error": f"Path not allowed: {p}"}
    if not p.exists():
        return {"ok": False, "error": f"Not found: {p}"}
    try:
        if OS == "windows":
            os.startfile(str(p))
        elif OS == "darwin":
            subprocess.run(["open", str(p)])
        else:
            subprocess.run(["xdg-open", str(p)])
        return {"ok": True, "message": f"Opened {p.name}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_files(path: str = "~", pattern: str = "*") -> Dict:
    p = Path(path).expanduser().resolve()
    if not _allowed_path(p):
        return {"ok": False, "error": f"Path not allowed: {p}"}
    try:
        items = sorted(p.glob(pattern))[:50]
        return {"ok": True, "files": [str(i.name) for i in items], "count": len(items)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def read_file(path: str, max_chars: int = 4000) -> Dict:
    p = Path(path).expanduser().resolve()
    if not _allowed_path(p):
        return {"ok": False, "error": f"Path not allowed: {p}"}
    if not p.exists():
        return {"ok": False, "error": f"Not found: {p}"}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "content": text[:max_chars], "truncated": len(text) > max_chars}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def write_file(path: str, content: str) -> Dict:
    p = Path(path).expanduser().resolve()
    if not _allowed_path(p):
        return {"ok": False, "error": f"Path not allowed: {p}"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "message": f"Written to {p.name}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── App launch ────────────────────────────────────────────────────────────────

def launch_app(app_name: str) -> Dict:
    name = app_name.lower().strip()
    if not any(a in name or name in a for a in ALLOWED_APPS):
        return {"ok": False, "error": f"App not in allowlist: {app_name}"}
    try:
        if OS == "windows":
            subprocess.Popen(["start", "", app_name], shell=True)
        elif OS == "darwin":
            subprocess.Popen(["open", "-a", app_name])
        else:
            subprocess.Popen([app_name])
        return {"ok": True, "message": f"Launched {app_name}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def open_url(url: str) -> Dict:
    try:
        import webbrowser
        webbrowser.open(url)
        return {"ok": True, "message": f"Opened {url}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Clipboard ─────────────────────────────────────────────────────────────────

def get_clipboard() -> Dict:
    try:
        if OS == "windows":
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True
            )
            return {"ok": True, "content": result.stdout.strip()[:2000]}
        elif OS == "darwin":
            result = subprocess.run(["pbpaste"], capture_output=True, text=True)
            return {"ok": True, "content": result.stdout.strip()[:2000]}
        else:
            result = subprocess.run(["xclip", "-o"], capture_output=True, text=True)
            return {"ok": True, "content": result.stdout.strip()[:2000]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_clipboard(content: str) -> Dict:
    try:
        if OS == "windows":
            subprocess.run(["clip"], input=content.encode(), check=True)
        elif OS == "darwin":
            subprocess.run(["pbcopy"], input=content.encode(), check=True)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=content.encode(), check=True)
        return {"ok": True, "message": "Copied to clipboard"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Screenshot ────────────────────────────────────────────────────────────────

def take_screenshot(save_to: str = None) -> Dict:
    if save_to is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to = str(HOME / "Desktop" / f"eros_{ts}.png")
    try:
        try:
            import PIL.ImageGrab as ImageGrab
            img = ImageGrab.grab()
            img.save(save_to)
            return {"ok": True, "path": save_to}
        except ImportError:
            pass
        if OS == "windows":
            ps = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen | Out-Null; $bmp = [System.Drawing.Bitmap]::new([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen(0,0,0,0,$bmp.Size); $bmp.Save("{save_to}")'
            subprocess.run(["powershell", "-command", ps], capture_output=True)
            return {"ok": True, "path": save_to}
        elif OS == "darwin":
            subprocess.run(["screencapture", save_to], check=True)
            return {"ok": True, "path": save_to}
        else:
            subprocess.run(["scrot", save_to], check=True)
            return {"ok": True, "path": save_to}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── System info ───────────────────────────────────────────────────────────────

def get_system_info() -> Dict:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "ok": True,
            "os": f"{platform.system()} {platform.release()}",
            "cpu_percent": cpu,
            "ram_used_gb": round(ram.used / 1e9, 1),
            "ram_total_gb": round(ram.total / 1e9, 1),
            "disk_free_gb": round(disk.free / 1e9, 1),
        }
    except ImportError:
        return {
            "ok": True,
            "os": f"{platform.system()} {platform.release()}",
            "python": sys.version.split()[0],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Run shell command (restricted) ────────────────────────────────────────────

SAFE_SHELL_PREFIXES = [
    "echo ", "dir ", "ls ", "pwd", "whoami", "date",
    "ipconfig", "ifconfig", "ping ", "curl ", "python ",
]


def run_command(cmd: str) -> Dict:
    cmd_lower = cmd.strip().lower()
    if not any(cmd_lower.startswith(p) for p in SAFE_SHELL_PREFIXES):
        return {"ok": False, "error": f"Command not in safe list. Blocked: {cmd[:50]}"}
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return {
            "ok": True,
            "stdout": result.stdout.strip()[:2000],
            "stderr": result.stderr.strip()[:500],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Dispatcher (called by action orchestrator) ────────────────────────────────

ACTION_MAP = {
    "open_file":      lambda p: open_file(p.get("path", "")),
    "list_files":     lambda p: list_files(p.get("path", "~"), p.get("pattern", "*")),
    "read_file":      lambda p: read_file(p.get("path", "")),
    "write_file":     lambda p: write_file(p.get("path", ""), p.get("content", "")),
    "launch_app":     lambda p: launch_app(p.get("app", "")),
    "open_url":       lambda p: open_url(p.get("url", "")),
    "get_clipboard":  lambda p: get_clipboard(),
    "set_clipboard":  lambda p: set_clipboard(p.get("content", "")),
    "take_screenshot":lambda p: take_screenshot(p.get("save_to")),
    "system_info":    lambda p: get_system_info(),
    "run_command":    lambda p: run_command(p.get("cmd", "")),
}


def execute_local_action(action_type: str, params: Dict = None) -> Dict:
    params = params or {}
    handler = ACTION_MAP.get(action_type)
    if not handler:
        return {"ok": False, "error": f"Unknown local action: {action_type}"}
    try:
        return handler(params)
    except Exception as e:
        return {"ok": False, "error": str(e)}
