"""
System tray daemon for Eros.
Sits in the notification area with a menu to control the session.
Run via: pythonw run.py --tray  (no console window)
     or: python  run.py --tray  (with console for debugging)
"""
import threading
import os
import sys
from typing import Optional, Callable

_tray = None


def _make_icon():
    """Generate a simple black circle icon with 'E' — no image file needed."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = 64
        img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([2, 2, size - 2, size - 2], fill=(15, 15, 15))
        draw.text((size // 2, size // 2), "E", fill=(220, 180, 100), anchor="mm")
        return img
    except Exception:
        from PIL import Image
        return Image.new("RGB", (32, 32), color=(20, 20, 20))


class ErosTray:
    """
    System tray icon for Eros.
    Exposes mute, open console, and exit controls.
    """

    def __init__(
        self,
        on_exit: Optional[Callable] = None,
        on_mute_toggle: Optional[Callable] = None,
        wake_detector=None,
    ):
        self._on_exit       = on_exit
        self._on_mute_toggle = on_mute_toggle
        self._wake_detector  = wake_detector
        self._muted          = False
        self._icon           = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="ErosTray")
        self._thread.start()

    def _run(self):
        try:
            import pystray
            from pystray import MenuItem as Item, Menu

            def toggle_mute(icon, item):
                self._muted = not self._muted
                if self._wake_detector:
                    if self._muted:
                        self._wake_detector.mute()
                    else:
                        self._wake_detector.unmute()
                if self._on_mute_toggle:
                    self._on_mute_toggle(self._muted)
                label = "Unmute" if self._muted else "Mute"
                print(f"[TRAY] {'Muted' if self._muted else 'Unmuted'}")
                icon.update_menu()

            def mute_label(item):
                return "Unmute Eros" if self._muted else "Mute Eros"

            def do_exit(icon, item):
                print("[TRAY] Exit requested")
                icon.stop()
                if self._on_exit:
                    self._on_exit()
                else:
                    os._exit(0)

            menu = Menu(
                Item("Eros — online", None, enabled=False),
                Menu.SEPARATOR,
                Item(mute_label, toggle_mute),
                Menu.SEPARATOR,
                Item("Exit", do_exit),
            )

            self._icon = pystray.Icon(
                name="Eros",
                icon=_make_icon(),
                title="Eros",
                menu=menu,
            )
            self._icon.run()
        except Exception as e:
            print(f"[TRAY] Failed to start system tray: {e}")

    def stop(self):
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[ErosTray] = None


def get_tray(**kwargs) -> ErosTray:
    global _instance
    if _instance is None:
        _instance = ErosTray(**kwargs)
    return _instance
