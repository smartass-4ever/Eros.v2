"""
Eros — entry point.

Usage:
    python run.py              # text chat
    python run.py --voice      # voice (push spacebar to talk)
    python run.py --voice --continuous   # voice activity detection, no push-to-talk
"""

import sys
import os
import asyncio
import argparse

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
for subdir in ["core", "memory", "self model", "user relationship", "misc", "saftey"]:
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
USER_NAME = os.environ.get("USER_NAME", "Mahika")
USER_ID   = os.environ.get("USER_ID", "mahika")
GROQ_KEY  = os.environ.get("GROQ_API_KEY", "")

if not GROQ_KEY:
    print("\n[!] GROQ_API_KEY not set in .env — LLM calls will use fallback.\n")

# Alias GROQ_API_KEY into every env var the codebase checks for the LLM key.
if GROQ_KEY:
    os.environ.setdefault("TOGETHER_API_KEY", GROQ_KEY)
    os.environ.setdefault("MISTRAL_API_KEY",  GROQ_KEY)

# ── Fix memory persistence ────────────────────────────────────────────────────
# IntelligentMemorySystem checks os.environ['DATABASE_URL'] directly.
# Set it now so memory survives restarts.
if not os.environ.get("DATABASE_URL"):
    _data_dir = os.path.join(ROOT, "data")
    os.makedirs(_data_dir, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_data_dir, 'eros.db')}"


# ── Voice output helper ───────────────────────────────────────────────────────
def _speak_sync(text: str):
    """Run TTS synchronously in its own thread (fresh engine each call — avoids pyttsx3 Windows hang)."""
    print(f"\nEros: {text}\n")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            if any(x in v.name.lower() for x in ("david", "mark", "george", "male")):
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 168)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass


def speak(text: str):
    """Print and speak text. Fire-and-forget in a daemon thread so it never blocks the event loop."""
    import threading
    t = threading.Thread(target=_speak_sync, args=(text,), daemon=True)
    t.start()


# ── Boot ──────────────────────────────────────────────────────────────────────
def boot():
    print("=" * 58)
    print("  Eros  —  initializing...")
    print("=" * 58)

    try:
        from cns_database import initialize_database
        initialize_database()
    except Exception as e:
        print(f"[DB] {e}")

    from merged_cns_flow import CNS
    cns = CNS()

    # Start screen monitor
    try:
        from screen_awareness import get_monitor
        monitor = get_monitor()
        monitor.start()
        cns._screen_monitor = monitor
    except Exception as e:
        print(f"[SCREEN] Monitor unavailable: {e}")

    print("=" * 58)
    print(f"  Online.  Talking to: {USER_NAME}")
    print("=" * 58)
    return cns


# ── Proactive mind push ───────────────────────────────────────────────────────
async def _proactive_push(text: str):
    """Called by ProactiveMind when Eros has something to say unprompted."""
    speak(text)


# ── Text loop ─────────────────────────────────────────────────────────────────
async def text_loop(cns, proactive):
    history = []
    loop = asyncio.get_event_loop()

    if proactive:
        asyncio.ensure_future(proactive.run())

    while True:
        try:
            # Run blocking input() in executor so proactive tasks can run
            user_input = await loop.run_in_executor(
                None, lambda: input(f"{USER_NAME}: ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            speak("Later.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "stop"):
            speak("Later.")
            break

        response = await _process(cns, user_input, history)
        speak(response)
        history = _update_history(history, user_input, response)


# ── Wake-word voice loop ──────────────────────────────────────────────────────
async def wake_word_loop(cns, proactive):
    """Always-on loop: wait for 'Hey Eros', then capture and respond."""
    from voice_input_module import VoiceInputHandler
    voice_in = VoiceInputHandler()
    loop     = asyncio.get_event_loop()
    history  = []
    _wake_event = asyncio.Event()

    def _on_wake():
        loop.call_soon_threadsafe(_wake_event.set)

    try:
        from wake_word import get_detector
        detector = get_detector(on_wake=_on_wake)
        detector.start()
    except Exception as e:
        print(f"[WAKE] Wake word unavailable: {e} — falling back to push-to-talk")
        await voice_loop(cns, proactive, continuous=False)
        return

    if proactive:
        asyncio.ensure_future(proactive.run())

    speak(f"Online. Say 'Hey Eros' when you need me.")
    print("  [waiting for wake word...]")

    while True:
        try:
            _wake_event.clear()
            await _wake_event.wait()

            # Mute detector while we're responding so it doesn't self-trigger
            detector.mute()
            speak("Yeah?")

            user_input = await loop.run_in_executor(
                None, lambda: voice_in.listen_until_silence()
            )

            detector.unmute()

            if not user_input or not user_input.strip():
                print("  [nothing heard — back to listening]")
                continue

            user_input = user_input.strip()
            print(f"\n{USER_NAME}: {user_input}")

            if user_input.lower() in ("quit", "exit", "goodbye", "stop", "bye"):
                speak("Later.")
                break

            response = await _process(cns, user_input, history)
            speak(response)
            history = _update_history(history, user_input, response)
            print("  [waiting for wake word...]")

        except KeyboardInterrupt:
            speak("Later.")
            break
        except Exception as e:
            print(f"[wake loop] {e}")
            detector.unmute()
            continue


# ── Voice loop ────────────────────────────────────────────────────────────────
async def voice_loop(cns, proactive, continuous: bool = False):
    from voice_input_module import VoiceInputHandler
    voice_in = VoiceInputHandler()

    history = []
    loop = asyncio.get_event_loop()

    if proactive:
        asyncio.ensure_future(proactive.run())

    speak(f"Online. Talk to me, {USER_NAME}.")

    while True:
        try:
            if continuous:
                print("  [listening...]")
                user_input = await loop.run_in_executor(
                    None, lambda: voice_in.listen_until_silence()
                )
            else:
                print("  [press ENTER to talk]")
                await loop.run_in_executor(None, input)
                user_input = await loop.run_in_executor(
                    None, lambda: voice_in.listen(duration=7)
                )

            if not user_input:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            print(f"\n{USER_NAME}: {user_input}")

            if user_input.lower() in ("quit", "exit", "goodbye", "stop", "bye"):
                speak("Later.")
                break

            response = await _process(cns, user_input, history)
            speak(response)
            history = _update_history(history, user_input, response)

        except KeyboardInterrupt:
            speak("Later.")
            break
        except Exception as e:
            print(f"[voice] {e}")
            continue


# ── Shared helpers ────────────────────────────────────────────────────────────
async def _process(cns, user_input: str, history: list) -> str:
    # Handle screen-awareness shortcuts
    lower = user_input.lower()
    if any(p in lower for p in ("what am i", "what's on my screen", "what are you seeing",
                                  "what do you see", "look at my screen", "see my screen")):
        try:
            from screen_awareness import get_screen_context
            ctx = await get_screen_context(use_vision=True)
            parts = []
            if ctx.get("active_app"):
                parts.append(f"You're in {ctx['active_app']}")
            if ctx.get("window_title") and ctx["window_title"] != ctx.get("active_app"):
                parts.append(f'— "{ctx["window_title"]}"')
            if ctx.get("vision_description"):
                parts.append(f"\n{ctx['vision_description']}")
            elif ctx.get("screen_text"):
                parts.append(f"\nScreen text: {ctx['screen_text'][:300]}")
            if parts:
                return " ".join(parts)
        except Exception:
            pass

    # Build screen context — injected as a system note before the user message
    screen_note = ""
    try:
        monitor = getattr(cns, "_screen_monitor", None)
        if monitor and monitor.current:
            screen_note = monitor.context_for_llm()
    except Exception:
        pass

    try:
        result = await cns.process_input(
            user_input=user_input,
            conversation_history=history,
            user_id=USER_ID,
            context={"screen": screen_note},
        )
        response = result.get("response") or result.get("text") or str(result)
        return response
    except Exception as e:
        return f"Something went wrong: {e}"


def _update_history(history: list, user_input: str, response: str) -> list:
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history[-40:]  # keep last 20 turns


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Eros — local AI companion")
    parser.add_argument("--voice",      action="store_true", help="Enable voice I/O")
    parser.add_argument("--continuous", action="store_true", help="Continuous listening (no push-to-talk)")
    parser.add_argument("--wake",       action="store_true", help="Always-on wake word mode ('Hey Eros')")
    parser.add_argument("--tray",       action="store_true", help="Run with system tray icon")
    args = parser.parse_args()

    cns = boot()

    # Boot proactive mind
    try:
        from proactive_mind import ProactiveMind
        proactive = ProactiveMind(user_name=USER_NAME)
        proactive.attach(cns, _proactive_push)
    except Exception:
        proactive = None

    # System tray (optional — non-blocking, runs in background thread)
    if args.tray:
        try:
            from tray_daemon import get_tray
            tray = get_tray()
            tray.start()
            print("[TRAY] System tray started")
        except Exception as e:
            print(f"[TRAY] {e}")

    if args.wake:
        asyncio.run(wake_word_loop(cns, proactive))
    elif args.voice:
        asyncio.run(voice_loop(cns, proactive, continuous=args.continuous))
    else:
        print("  Type 'quit' to exit. Use --voice or --wake for voice mode.\n")
        asyncio.run(text_loop(cns, proactive))


if __name__ == "__main__":
    main()
