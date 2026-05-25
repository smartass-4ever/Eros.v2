"""
Wake word detection for Eros.

Priority chain:
  1. Porcupine + custom 'hey_eros.ppn'  — best accuracy, needs PICOVOICE_API_KEY + keyword file
  2. Porcupine + built-in 'jarvis'       — still excellent, just needs PICOVOICE_API_KEY
  3. openWakeWord + Whisper verify       — fully local, no keys, lower accuracy
  4. (old) Whisper-only loop             — fallback of last resort

Setup for option 1 (recommended):
  a) Free account at console.picovoice.ai
  b) Create "hey eros" keyword → download hey_eros_windows.ppn
  c) Set in .env:
       PICOVOICE_API_KEY=your_key
       PORCUPINE_KEYWORD_PATH=C:/path/to/hey_eros_windows.ppn
  d) Say "Hey Eros" — that's it.

Setup for option 2 (instant):
  a) Free account at console.picovoice.ai — grab the API key
  b) Set PICOVOICE_API_KEY in .env
  c) Say "Jarvis" to wake Eros.
"""

import os
import threading
import time
import numpy as np
from typing import Callable, Optional

SAMPLE_RATE = 16000

# ── Porcupine backend ─────────────────────────────────────────────────────────

class _PorcupineDetector:
    def __init__(self, on_wake: Callable, access_key: str, keyword_path: str = None):
        import pvporcupine
        import pvrecorder

        self._on_wake = on_wake

        if keyword_path and os.path.exists(keyword_path):
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path],
                sensitivities=[0.6],
            )
            kw_label = os.path.basename(keyword_path)
        else:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["jarvis"],
                sensitivities=[0.6],
            )
            kw_label = "jarvis (say 'Jarvis' to wake Eros)"

        self._recorder = pvrecorder.PvRecorder(
            frame_length=self._porcupine.frame_length
        )
        self._running  = False
        self._muted    = False
        self._thread: Optional[threading.Thread] = None
        self._kw_label = kw_label

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="WakeWord-Porcupine")
        self._thread.start()
        print(f"[WAKE] Porcupine started — wake phrase: {self._kw_label}")

    def stop(self):
        self._running = False
        try:
            self._recorder.stop()
        except Exception:
            pass

    def mute(self):   self._muted = True
    def unmute(self): self._muted = False

    def _loop(self):
        self._recorder.start()
        try:
            while self._running:
                pcm = self._recorder.read()
                if self._muted:
                    continue
                idx = self._porcupine.process(pcm)
                if idx >= 0:
                    print(f"[WAKE] Wake word detected (Porcupine)")
                    try:
                        self._on_wake()
                    except Exception as e:
                        print(f"[WAKE] on_wake error: {e}")
        finally:
            self._recorder.stop()
            self._recorder.delete()
            self._porcupine.delete()


# ── openWakeWord + Whisper verification backend ───────────────────────────────

class _OWWDetector:
    """
    Stage 1: openWakeWord (hey_jarvis model) — always on, very low CPU.
    Stage 2: When OWW fires, capture 2s and run Whisper tiny.
             Accept if 'eros' appears in the transcript.
    """

    OWW_THRESHOLD  = 0.3   # low — we're using it as a coarse gate, Whisper verifies
    WINDOW_FRAMES  = 1280  # openWakeWord expects 80ms chunks at 16kHz
    CAPTURE_SECS   = 2.0
    ENERGY_GATE    = 0.002
    COOLDOWN       = 3.0

    def __init__(self, on_wake: Callable):
        self._on_wake = on_wake
        self._running = False
        self._muted   = False
        self._thread: Optional[threading.Thread] = None
        self._oww     = None
        self._whisper = None

    def _load_models(self):
        try:
            from openwakeword import Model
            self._oww = Model(wakeword_models=["hey_jarvis_v0.1.tflite"], inference_framework="tflite")
            print("[WAKE] openWakeWord loaded (hey_jarvis base)")
        except Exception as e:
            print(f"[WAKE] openWakeWord load failed: {e}")

        try:
            import whisper
            self._whisper = whisper.load_model("tiny")
            print("[WAKE] Whisper tiny loaded for verification")
        except Exception as e:
            print(f"[WAKE] Whisper tiny load failed: {e}")

    def start(self):
        self._load_models()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="WakeWord-OWW")
        self._thread.start()
        print("[WAKE] OWW+Whisper detector started — say 'Eros' or 'Hey Eros'")

    def stop(self):  self._running = False
    def mute(self):  self._muted = True
    def unmute(self): self._muted = False

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._whisper is None:
            return ""
        try:
            f32 = audio.astype(np.float32) / 32768.0
            result = self._whisper.transcribe(
                f32, language="en", fp16=False,
                condition_on_previous_text=False,
            )
            return result.get("text", "").strip().lower()
        except Exception:
            return ""

    def _rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def _loop(self):
        import sounddevice as sd
        buf = np.zeros(0, dtype=np.int16)

        while self._running:
            try:
                chunk = sd.rec(
                    self.WINDOW_FRAMES, samplerate=SAMPLE_RATE,
                    channels=1, dtype="int16"
                )
                sd.wait()
                chunk = chunk.flatten()

                if self._muted:
                    continue

                buf = np.concatenate([buf, chunk])

                # Keep rolling buffer of ~3s for OWW
                max_buf = SAMPLE_RATE * 3
                if len(buf) > max_buf:
                    buf = buf[-max_buf:]

                if self._oww is None:
                    # No OWW — fall back to pure energy + Whisper
                    if self._rms(chunk) > self.ENERGY_GATE:
                        text = self._transcribe(chunk)
                        if text and "eros" in text:
                            print(f"[WAKE] Whisper detected: '{text}'")
                            self._fire()
                    continue

                # Feed to OWW
                f32_buf = buf.astype(np.float32) / 32768.0
                scores  = self._oww.predict(f32_buf)

                best = max(scores.values()) if scores else 0.0
                if best >= self.OWW_THRESHOLD and self._rms(chunk) > self.ENERGY_GATE:
                    # Capture a full window and verify with Whisper
                    full = sd.rec(
                        int(SAMPLE_RATE * self.CAPTURE_SECS),
                        samplerate=SAMPLE_RATE, channels=1, dtype="int16"
                    )
                    sd.wait()
                    text = self._transcribe(full.flatten())
                    if text and "eros" in text:
                        print(f"[WAKE] OWW+Whisper verified: '{text}'")
                        self._fire()
                        buf = np.zeros(0, dtype=np.int16)
                        time.sleep(self.COOLDOWN)

            except Exception as e:
                print(f"[WAKE] OWW loop error: {e}")
                time.sleep(0.5)

    def _fire(self):
        try:
            self._on_wake()
        except Exception as e:
            print(f"[WAKE] on_wake error: {e}")


# ── Public interface ──────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Unified wrapper. Picks the best available backend automatically.
    Exposes: start(), stop(), mute(), unmute()
    """

    def __init__(self, on_wake: Callable, cooldown: float = 3.0):
        self._backend = None
        self._on_wake = on_wake
        self._cooldown = cooldown

    def _pick_backend(self):
        access_key   = os.environ.get("PICOVOICE_API_KEY", "")
        keyword_path = os.environ.get("PORCUPINE_KEYWORD_PATH", "")

        if access_key:
            try:
                import pvporcupine
                import pvrecorder
                return _PorcupineDetector(
                    on_wake=self._on_wake,
                    access_key=access_key,
                    keyword_path=keyword_path or None,
                )
            except ImportError:
                print("[WAKE] pvporcupine/pvrecorder not installed — falling back to OWW")
            except Exception as e:
                print(f"[WAKE] Porcupine init failed: {e} — falling back to OWW")

        return _OWWDetector(on_wake=self._on_wake)

    def start(self):
        self._backend = self._pick_backend()
        self._backend.start()

    def stop(self):
        if self._backend:
            self._backend.stop()

    def mute(self):
        if self._backend:
            self._backend.mute()

    def unmute(self):
        if self._backend:
            self._backend.unmute()


# ── Singleton ─────────────────────────────────────────────────────────────────

_detector: Optional[WakeWordDetector] = None


def get_detector(on_wake: Callable = None, cooldown: float = 3.0) -> WakeWordDetector:
    global _detector
    if _detector is None:
        if on_wake is None:
            raise ValueError("on_wake required for first call")
        _detector = WakeWordDetector(on_wake=on_wake, cooldown=cooldown)
    return _detector
