"""
Voice input — Whisper STT + sounddevice microphone capture.
Supports push-to-talk (spacebar) and continuous listening.
"""
import os
import sys
import time
import threading
import numpy as np
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
WHISPER_MODEL = None
_model_lock = threading.Lock()


def _load_whisper():
    global WHISPER_MODEL
    with _model_lock:
        if WHISPER_MODEL is None:
            try:
                import whisper
                print("[VOICE] Loading Whisper base model...")
                WHISPER_MODEL = whisper.load_model("base")
                print("[VOICE] Whisper ready.")
            except Exception as e:
                print(f"[VOICE] Whisper load failed: {e}")
    return WHISPER_MODEL


def transcribe(audio_np: np.ndarray) -> str:
    model = _load_whisper()
    if model is None:
        return ""
    try:
        import whisper
        audio = audio_np.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0
        result = model.transcribe(audio, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        print(f"[VOICE] Transcription error: {e}")
        return ""


class VoiceInputHandler:
    def __init__(self, device_id=None):
        self.device_id = device_id
        _load_whisper()  # pre-load in background

    def listen(self, duration: float = 5.0) -> str:
        """Record for `duration` seconds and transcribe."""
        try:
            print(f"[VOICE] Listening for {duration}s...")
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=self.device_id,
            )
            sd.wait()
            audio_np = audio.flatten()
            text = transcribe(audio_np)
            return text
        except Exception as e:
            print(f"[VOICE] Listen error: {e}")
            return ""

    def listen_until_silence(self, silence_sec: float = 1.5, max_sec: float = 15.0) -> str:
        """Record until user stops talking (VAD-lite via energy threshold)."""
        chunk = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        threshold = 300
        frames = []
        silent_chunks = 0
        max_chunks = int(max_sec / 0.1)
        silence_trigger = int(silence_sec / 0.1)

        print("[VOICE] Listening (speak now)...")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=self.device_id, blocksize=chunk) as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk)
                frames.append(data.copy())
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
                if rms < threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_trigger and len(frames) > 10:
                        break
                else:
                    silent_chunks = 0

        if not frames:
            return ""
        audio_np = np.concatenate(frames).flatten()
        return transcribe(audio_np)

    @staticmethod
    def list_microphones():
        devices = sd.query_devices()
        mics = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                mics.append((i, d["name"]))
        return mics

    @staticmethod
    def select_microphone():
        mics = VoiceInputHandler.list_microphones()
        if not mics:
            print("[VOICE] No microphones found.")
            return None
        print("\nAvailable microphones:")
        for idx, name in mics:
            print(f"  [{idx}] {name}")
        try:
            choice = input("Select mic number (Enter for default): ").strip()
            return int(choice) if choice else None
        except (ValueError, KeyboardInterrupt):
            return None

    def test_microphone(self, duration: float = 2.0) -> bool:
        try:
            audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="int16", device=self.device_id)
            sd.wait()
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            return rms > 50
        except Exception:
            return False
