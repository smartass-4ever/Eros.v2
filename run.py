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
USER_NAME    = os.environ.get("USER_NAME", "Mahika")
USER_ID      = os.environ.get("USER_ID", "mahika")
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")

if not TOGETHER_KEY:
    print("\n[!] TOGETHER_API_KEY not set in .env — LLM calls will use fallback.\n")


# ── Voice output helper ───────────────────────────────────────────────────────
_tts_engine = None

def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
            voices = _tts_engine.getProperty("voices")
            # prefer a deeper / male voice on Windows
            for v in voices:
                if any(x in v.name.lower() for x in ("david", "mark", "george", "male")):
                    _tts_engine.setProperty("voice", v.id)
                    break
            _tts_engine.setProperty("rate", 168)
            _tts_engine.setProperty("volume", 1.0)
        except Exception:
            _tts_engine = None
    return _tts_engine


def speak(text: str):
    """Speak text aloud and print it."""
    print(f"\nEros: {text}\n")
    engine = _get_tts()
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


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
        print(f"\nEros: {response}\n")
        history = _update_history(history, user_input, response)


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
    try:
        result = await cns.process_input(
            user_input=user_input,
            conversation_history=history,
            user_id=USER_ID,
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
    args = parser.parse_args()

    cns = boot()

    # Boot proactive mind
    try:
        from proactive_mind import ProactiveMind
        proactive = ProactiveMind(user_name=USER_NAME)
        proactive.attach(cns, _proactive_push)
    except Exception:
        proactive = None

    if args.voice:
        asyncio.run(voice_loop(cns, proactive, continuous=args.continuous))
    else:
        print("  Type 'quit' to exit. Use --voice for voice mode.\n")
        asyncio.run(text_loop(cns, proactive))


if __name__ == "__main__":
    main()
