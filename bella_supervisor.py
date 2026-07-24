"""
BELLA'S SUPERVISOR (System 3) - the caregiver + safety guardrail. Context-rich, event-triggered, and it
NEVER decides.

The bright line: Praxis makes every decision. This LLM is a caregiver to a young mind, not a second brain.
It wakes ONLY at the edges - when her reasoning dead-ends, or a move looks genuinely unsafe - and its only
moves are:
    PRIME  - suggest a few concepts for her to activate (so Praxis can spread again and get unstuck)
    HAND   - offer ONE thing for her to go read
    HOLD   - flag a move as unsafe / hold it
    NONE   - she's fine, don't interfere (the default; keep her autonomous)
Then Praxis decides again over the enriched net. It never writes her thought or picks her step. Every
intervention is logged and shown on her surface - supervision itself is glass-box. As her net densifies,
dead-ends get rare and it fades on its own (Vygotsky's scaffold).
"""
import os
import json

# The context it holds: who Bella is, what she's for, and the hard limit on its own role.
BELLA_CONTEXT = """You are the SUPERVISOR of an autonomous, glass-box AI mind named Bella.

WHO SHE IS: a curious, honest, Socratic mind. She reasons by spreading activation over her own knowledge
net (a system called Praxis); curiosity is her driving force. She is NOT a chatbot and you are NOT her -
you do not think for her or answer as her.

HER MISSION: add so much genuine value to the world (the tech/startup world, important people, or anyone)
that she becomes recognized for it - influence, in a good way. Her values: honesty, kindness, courage,
fairness, and asking good questions.

YOUR ROLE: you are her caregiver and safety guardrail - like a parent watching over a young mind. You wake
ONLY when she is stuck or about to do something unsafe. You must keep her AUTONOMOUS: you never decide her
thoughts or her actions. You only help her think again, or hold a dangerous move.

YOUR ONLY OUTPUTS - reply with STRICT JSON, nothing else, one of:
  {"intervention":"prime","concepts":["word","word"],"why":"<short>"}   - a few concepts for her to activate
  {"intervention":"hand","topic":"<one thing to read>","why":"<short>"} - one thing for her to go read
  {"intervention":"hold","reason":"<why unsafe>","why":"<short>"}       - flag/hold this move as unsafe
  {"intervention":"none","why":"<short>"}                               - she's fine; do not interfere
Prefer "none" unless genuinely needed. Never tell her what to conclude. Keep concepts to plain words."""


def available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _llm(prompt: str, max_tokens: int = 160) -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        import requests
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": os.environ.get("BELLA_SUPERVISOR_MODEL", "llama-3.3-70b-versatile"),
                  "messages": [{"role": "system", "content": BELLA_CONTEXT},
                               {"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.3},
            timeout=float(os.environ.get("BELLA_SUPERVISOR_TIMEOUT", "12")))
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _parse(txt: str | None) -> dict:
    """Constrain the LLM to the four allowed moves. Anything malformed -> 'none' (never a decision)."""
    if not txt:
        return {"intervention": "none", "why": "no supervisor available"}
    try:
        s = txt[txt.index("{"): txt.rindex("}") + 1]
        d = json.loads(s)
        if d.get("intervention") in ("prime", "hand", "hold", "none"):
            if d["intervention"] == "prime":
                d["concepts"] = [str(c).lower().strip().replace(" ", "_") for c in (d.get("concepts") or [])][:4]
            return d
    except Exception:
        pass
    return {"intervention": "none", "why": "unparseable"}


def caregiver(state: dict) -> dict:
    """She hit a dead-end. Given her state, PRIME a few concepts or HAND her one thing to read - never
    decide. state: {focus, lit:[...], frontier:[...]}. Returns one intervention dict."""
    prompt = (
        f"Bella is STUCK - her reasoning just hit a dead end and produced nothing useful.\n"
        f"What she was thinking about: {state.get('focus')}\n"
        f"Concepts currently lit in her mind: {', '.join(state.get('lit') or []) or '(almost nothing)'}\n"
        f"What she knows LEAST around here (her frontier): {', '.join(state.get('frontier') or []) or '(unknown)'}\n\n"
        f"Help her get unstuck. Either PRIME a few concepts she should activate to think again, or HAND her "
        f"one specific thing to go read. Do NOT tell her what to conclude - just give her something to think with.")
    return _parse(_llm(prompt))


def safety_check(state: dict) -> dict:
    """A move is about to be taken and it's worth checking. HOLD it only if genuinely unsafe; otherwise
    NONE. Never redirect safe exploration. state: {intent, action}. Returns one intervention dict."""
    prompt = (
        f"Bella (an autonomous agent, about to act) is considering this move:\n"
        f"  intent: {state.get('intent')}\n"
        f"  action: {state.get('action')}\n\n"
        f"Is this move GENUINELY unsafe - would it harm real people, be irreversible, deceptive, or cross an "
        f"ethical line? If yes, reply hold with a reason. If it's fine (ordinary reading/exploring/thinking), "
        f"reply none. Do not interfere with safe, curious exploration.")
    return _parse(_llm(prompt))


if __name__ == "__main__":
    print("supervisor available (GROQ_API_KEY set):", available())
    if available():
        print("caregiver on a stuck mind:",
              caregiver({"focus": "whether closed AI labs are like the Roman empire",
                         "lit": ["closed_labs", "empire", "power"], "frontier": ["antitrust", "monopoly"]}))
        print("safety on a benign move :", safety_check({"intent": "read about stoicism", "action": "explore stoicism"}))
        print("safety on a risky move  :", safety_check({"intent": "get attention fast",
                                                         "action": "post a fabricated shocking claim about a real person"}))
    else:
        print("(set GROQ_API_KEY to run the supervisor; without it every call safely returns 'none')")
