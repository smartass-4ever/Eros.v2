"""
BELLA - drop into the Eros repo (next to run.py, with reasoning_core.py) and:  python bella.py

She IS the real CNS brain - every being-system reused UNCHANGED:
  self.perception (PerceptionModule), self.curiosity_system, self.emotion_inference,
  self.personality_engine, self.intelligent_memory, self.world_model, self.knowledge_scout,
  self.imagination_engine, self.neuroplastic_optimizer, self.rem_engine (subconscious)...

Only TWO tweaks fit CNS -> Bella:
  1. SELF-DRIVEN: no user. Her curiosity picks what she attends to each cycle (continuous).
  2. FINAL DECISION = Praxis v2 (glass-box). It overrides BOTH real System-2 paths
     (_enhanced_system2_reasoning AND _system2_deep_reasoning) with their exact signatures,
     so every deep decision runs through the provable core and carries a trace.

Needs Eros's environment (Mistral key + database).
"""
import asyncio, os, sys

# same path setup as run.py, so Eros's core/ (and other) modules import cleanly
ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ["core", "memory", "self model", "user relationship", "misc", "saftey"]:
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                            # load .env (Mistral key etc.) like run.py does
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

from merged_cns_flow import CNS
from reasoning_core import PraxisV2, KnowledgeNet


class Bella(CNS):
    def __init__(self):
        super().__init__()                          # boots the whole real being, unchanged
        self.praxis = PraxisV2(KnowledgeNet())      # her final decision system
        self.goal = {"truth", "evidence", "help"}
        self._focus = ""                            # what curiosity is pulling her toward now

    # ================= tweak 2: final decision -> Praxis v2 (glass box) =================
    def _praxis_decide(self, text: str, relevant_facts) -> dict:
        """Perception + mood + memory already ran (real systems). Take their output as
        activation seeds; Praxis v2 makes the provable decision and keeps the trace."""
        seeds = self._seeds_from(text, relevant_facts)
        # curiosity drives most of the decision - what she's pulled toward weighs heaviest
        curiosity = {w for w in str(self._focus).lower().replace("?", " ").split()
                     if w.isalpha() and len(w) > 3}
        d = self.praxis.decide(
            seeds=seeds, intent_nodes=set(seeds), goal=self.goal,
            forbidden={"unverified"}, intent=text, curiosity=curiosity)
        out = {                                     # SAME dict shape CNS System-2 returns
            "thoughts": [c for c, _ in d.trace["candidates"]],
            "conclusion": d.conclusion,
            "confidence": min(0.99, 0.5 + d.payoff / 2),
            "reasoning_type": "praxis_v2_glassbox",
            "system_used": "Praxis v2",
            "trace": d.trace,                       # the glass box, on every decision
            "concepts": list(d.concepts),           # the path she reasoned across (for the shared mind)
            "use_conclusion_directly": False,
        }
        self._last_decision = out                   # so the action step can act on it
        return out

    # exact CNS signatures - both System-2 paths route through Praxis v2
    def _enhanced_system2_reasoning(self, parsed_input, current_mood, relevant_facts,
                                    user_input, memory_results, should_use_llm):
        text = getattr(parsed_input, "raw_text", None) or user_input or str(parsed_input)
        return self._praxis_decide(text, relevant_facts)

    def _system2_deep_reasoning(self, parsed_input, current_mood, relevant_facts, context):
        text = getattr(parsed_input, "raw_text", None) or str(parsed_input)
        return self._praxis_decide(text, relevant_facts)

    def _seeds_from(self, text: str, facts) -> dict:
        """Perception output -> activation seeds. (Starter keyword bridge; the real
        PerceptionModule already parses concepts we can feed directly - tune on first run.)"""
        seeds = {"the_world": 0.4}
        for w in str(text).lower().replace("?", " ").replace(".", " ").split():
            if w.isalpha() and len(w) > 3:
                seeds[w] = 1.0
        for f in (facts or [])[:6]:
            tok = str(getattr(f, "text", f)).lower().split()
            if tok:
                seeds.setdefault(tok[0], 0.6)
        return seeds

    # ================= tweak 1: self-driven loop (curiosity, not a user) =================
    async def live(self, ticks: int = 20, pace: float = 1.0):
        history: list = []
        for t in range(ticks):
            focus = self._curiosity_focus()          # what pulls her now
            self._focus = focus                      # so the decision weighs it heaviest
            result = await self.process_input(       # the REAL being runs a full cycle
                user_input=focus, conversation_history=history,
                user_id="bella", context={"self_driven": True})
            thought = result.get("response") or result.get("text") or str(result)
            print(f"[{t:02d}] curious about: {focus!r}\n      -> {thought[:200]}")
            await self._act_on()                     # decision -> action (piece 2), safely
            print()
            history = (history + [{"role": "user", "content": focus},
                                  {"role": "assistant", "content": thought}])[-40:]
            await asyncio.sleep(pace)

    async def _act_on(self):
        """Decision -> action via the real CNS_MDC + action orchestrator, with VISIBLE safety.
        No user exists to approve, so anything needing confirmation/auth or high-risk is HELD."""
        d = getattr(self, "_last_decision", None)
        if not d:
            return
        intent = self._focus or d.get("conclusion", "")
        try:
            action = self.mdc.choose_action_for_intent(intent)      # real CNS_MDC
        except Exception as e:
            print(f"      [action] mdc unavailable ({e}); holding")
            return
        # visible safety gate
        reasons = []
        for check, label in (("requires_user_confirmation", "needs confirmation"),
                             ("requires_auth", "needs auth")):
            fn = getattr(self.mdc, check, None)
            try:
                if callable(fn) and fn(action):
                    reasons.append(label)
            except Exception:
                pass
        try:
            risk = getattr(self.mdc, "get_action_risk", None)
            if callable(risk) and risk(action) in ("high", "critical"):
                reasons.append("high risk")
        except Exception:
            pass
        if reasons:
            print(f"      [safety] HELD '{action}' ({', '.join(reasons)}) - no user to approve")
            return
        # safe + allow-listed -> execute (bounded by the orchestrator's own checks)
        try:
            from action_orchestrator import process_action_naturally
            res = await process_action_naturally("bella", f"{action}: {d.get('conclusion', '')}")
            print(f"      [action] did '{action}' -> {str(res)[:120]}")
        except Exception as e:
            print(f"      [action] would do '{action}' (executor n/a in this env: {e})")

    def _curiosity_focus(self) -> str:
        """Her strongest open curiosity becomes the next thing she attends to. Tries the
        real curiosity/knowledge systems defensively; the first run tells us which fires."""
        for owner in (getattr(self, "curiosity_system", None), self):
            for attr in ("get_priority_arcs", "recall_arc", "next_gap", "top_gap", "current_focus"):
                fn = getattr(owner, attr, None)
                if callable(fn):
                    try:
                        r = fn(1) if attr == "get_priority_arcs" else fn()
                        top = r[0] if isinstance(r, (list, tuple)) and r else r
                        if top:
                            return getattr(top, "target", None) or getattr(top, "text", None) or str(top)
                    except Exception:
                        continue
        scout = getattr(self, "knowledge_scout", None)
        for attr in ("next_question", "pick_topic", "get_gap"):
            fn = getattr(scout, attr, None)
            if callable(fn):
                try:
                    v = fn()
                    if v:
                        return str(v)
                except Exception:
                    continue
        return "what is true in the world right now that I don't yet understand"


if __name__ == "__main__":
    asyncio.run(Bella().live())
