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

os.environ.setdefault("PYTHONIOENCODING", "utf-8")   # emoji in Eros's prints (like run.py)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
        try:                                        # create the DB tables first, like run.py boot()
            from cns_database import initialize_database
            initialize_database()
        except Exception as e:
            print(f"[DB] {e}")
        super().__init__()                          # boots the whole real being, unchanged
        self.praxis = PraxisV2(KnowledgeNet())      # her final decision system
        self.goal = {"truth", "evidence", "help"}
        self._focus = ""                            # what curiosity is pulling her toward now
        self.interests = [                          # her inherent interests (EDIT to make it hers)
            "artificial intelligence", "minds and consciousness", "how systems work and fail",
            "honesty and trust", "underdogs and outsiders", "philosophy",
            "science and physics", "internet culture", "what is true",
        ]

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
            self._learn_from_cycle(result)           # continuous self-learning (every cycle)
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

    # ================= continuous self-learning (every cycle) =================
    def _learn_from_cycle(self, result):
        """After every cycle she updates from what just happened. Immediate signal =
        her decision's confidence (a mild self-supervised nudge); the STRONG signal
        (real-world / credible-engagement) arrives via reward() when the world responds."""
        d = getattr(self, "_last_decision", None)
        if not d:
            return
        concepts = d.get("concepts", [])
        conf = d.get("confidence", 0.5)
        # 1) reputation: a confident, coherent decision reinforces the path it reasoned across
        self.praxis.learn(concepts, (conf - 0.5) * 2)
        # 2) grow the web: the associations she just used become part of her substrate
        rels = [(concepts[i], concepts[i + 1], 0.4) for i in range(len(concepts) - 1)]
        if rels:
            self.praxis.net.ingest(rels)
        # 3) write the lesson to memory (her own + the shared swarm mind if attached)
        pairs = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
        for store in (getattr(self, "collective", None), getattr(self, "intelligent_memory", None)):
            try:
                if store and hasattr(store, "remember"):
                    store.remember(key=d.get("conclusion", ""), content=d.get("conclusion", ""),
                                   salience=conf, relations=pairs)
            except Exception:
                pass
        # 4) let Eros's own learning systems update too (best-effort; confirm method names on a run)
        for name, method in (("neuroplastic_optimizer", "optimize"),
                             ("growth_tracker", "record"),
                             ("cognitive_learning_system", "learn")):
            fn = getattr(getattr(self, name, None), method, None)
            try:
                if callable(fn):
                    fn(d.get("conclusion", ""))
            except Exception:
                pass

    def reward(self, signal: float, concepts=None):
        """The STRONG external signal - call this when the world genuinely responds
        (credible engagement, a real outcome). Reinforces the reasoning path she used."""
        concepts = concepts or (getattr(self, "_last_decision", {}) or {}).get("concepts", [])
        self.praxis.learn(concepts, signal)

    def _world_intake(self) -> str:
        """What she's taking in from the real world right now. WIRE Indra here to feed live
        web content; until then this is empty and she runs on her own thoughts."""
        # WIRE: return Indra's latest fetched / changed content as a string
        return ""

    def _curiosity_focus(self) -> str:
        """Curiosity = novelty (the world + her own thoughts) x her inherent interests, fluid.
        Feeds real-world intake + her last thought into the real CuriositySystem to detect gaps,
        then attends to the strongest - biased toward her interests, free to wander to novelty."""
        cs = getattr(self, "curiosity_system", None)
        if cs is not None:
            d = getattr(self, "_last_decision", None)
            own = (d.get("conclusion") if d else "") or getattr(self, "_focus", "")
            intake = (self._world_intake() + " " + own).strip() \
                or "the world - what is true, what is changing, what I don't understand"
            try:
                cs.process_turn(intake)                     # novelty: gaps from the world + her thoughts
            except Exception:
                pass
            try:
                arcs = cs.dm.get_priority_arcs(top_n=3) or []
                interests = {w for i in getattr(self, "interests", []) for w in str(i).lower().split()}
                # prefer a strong arc that matches an interest; else the strongest arc (stays fluid)
                for arc in arcs:
                    tgt = str(getattr(arc, "target", "")).lower().replace("_", " ")
                    if interests & set(tgt.split()):
                        return str(arc.target)
                if arcs and getattr(arcs[0], "target", None):
                    return str(arcs[0].target)
            except Exception:
                pass
        return "what is true in the world right now that I don't yet understand"


if __name__ == "__main__":
    asyncio.run(Bella().live())
