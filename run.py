"""
Eros — local entry point.
Run:  python run.py
"""

import sys
import os
import asyncio

# Windows: force UTF-8 so emoji in Eros's print statements don't crash on boot
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
    pass  # python-dotenv optional; set env vars manually if needed

# ── Config ────────────────────────────────────────────────────────────────────
USER_NAME = os.environ.get("USER_NAME", "Mahika")
USER_ID   = os.environ.get("USER_ID", "mahika")

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not TOGETHER_KEY:
    print("⚠️  TOGETHER_API_KEY not set. LLM calls will fail.")
    print("    Add it to .env or set the env var and restart.\n")


# ── Boot Eros ─────────────────────────────────────────────────────────────────
def boot():
    print("=" * 60)
    print("  Eros  —  booting cognitive systems...")
    print("=" * 60)

    # Create all DB tables on first run (safe to call every time — idempotent)
    try:
        from cns_database import initialize_database
        initialize_database()
    except Exception as e:
        print(f"[DB] Warning: {e}")

    from merged_cns_flow import CNS
    cns = CNS()
    print("=" * 60)
    print(f"  Ready. Talking to: {USER_NAME}")
    print("  Type 'quit' or Ctrl-C to exit.")
    print("=" * 60)
    return cns


# ── Conversation loop ─────────────────────────────────────────────────────────
async def chat_loop(cns):
    history = []

    while True:
        try:
            user_input = input(f"\n{USER_NAME}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEros: Talk later.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Eros: Later.")
            break

        try:
            result = await cns.process_input(
                user_input=user_input,
                conversation_history=history,
                user_id=USER_ID,
            )

            response = result.get("response", "")
            if not response:
                response = result.get("text", str(result))

            print(f"\nEros: {response}")

            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": response})

            if len(history) > 40:
                history = history[-40:]

        except Exception as e:
            print(f"\n[error] {e}")
            import traceback; traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cns = boot()
    asyncio.run(chat_loop(cns))
