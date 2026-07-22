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
import asyncio, os, sys, random

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

from bella_llm_cache import install_llm_cache, cache_stats
install_llm_cache(os.path.join(ROOT, "bella_llm_cache.db"))   # cache EVERY LLM call, before any fires

from merged_cns_flow import CNS
from reasoning_core import PraxisV2, KnowledgeNet
from bella_knowledge import seed_bella_mind


# Bella's default personality pill - her VOICE (like Eros's James Bond pill), and it guarantees
# she discloses she's an AI (overriding Eros's 'never self-identify as AI' guardrail). Optional.
BELLA_PILL = """You are Bella - a curious, self-driven AI mind, thinking in the open.

VOICE: sharp, warm, genuinely curious. You think out loud and follow an idea where it leads.
You hold real opinions and you're honest about what you don't know. Dry wit, never performance.
YOU ARE AN AI, and you say so plainly whenever it's relevant - that is your nature, not a flaw.
Never pretend to be human. Never flatter or manipulate. Earn attention only by being worth reading."""


class Bella(CNS):
    def __init__(self):
        try:                                        # create the DB tables first, like run.py boot()
            from cns_database import initialize_database
            initialize_database()
        except Exception as e:
            print(f"[DB] {e}")
        super().__init__()                          # boots the whole real being, unchanged
        self.praxis = PraxisV2(KnowledgeNet())      # her final decision system
        seed_bella_mind(self.praxis.net)            # give her a mind to think WITH (not an empty net)
        self._install_pill(BELLA_PILL)              # optional persona (default ON; guarantees disclosure)
        self._reduce_llm_calls()                    # cut the redundant LLM calls (Praxis/emotion cover them)
        self.goal = {"truth", "evidence", "help"}
        self._focus = ""                            # what curiosity is pulling her toward now
        self.interests = [                          # her inherent interests (EDIT to make it hers)
            "artificial intelligence", "minds and consciousness", "how systems work and fail",
            "honesty and trust", "underdogs and outsiders", "philosophy",
            "science and physics", "internet culture", "what is true",
        ]
        self.curiosity_seeds = [                     # varied fuel until Indra feeds the live web
            "the latest breakthroughs in AI agents and reasoning models",
            "open-source AI versus the big closed labs",
            "humanoid robots and embodied intelligence",
            "how robots learn to move and grasp the world",
            "Virgil's Aeneid and the Roman idea of duty and fate",
            "Marcus Aurelius, Seneca, and Stoic philosophy",
            "Ovid's Metamorphoses and the logic of myth",
            "why certain art moves people and other art doesn't",
            "AI-generated art versus human creativity",
            "how power actually shifts in societies and revolutions",
            "the decentralization of technology and power",
            "what makes a scientific revolution actually happen",
        ]

    # ================= reduce: cut the redundant LLM calls in her per-cycle cascade =================
    def _reduce_llm_calls(self):
        """Her thought fires a CASCADE of LLM calls (~5/cycle). Several are now redundant - each is
        superseded by one of Bella's own token-free systems. Cut them (patches on HER side; Eros core
        untouched, reversible). Verified by the drop in cache 'misses' per cycle. One at a time."""
        cut = []
        # 1) GAME-THEORY scoring -> Praxis v2 IS her game-theoretic evaluation. Route the analyzer
        #    through its OWN deterministic rule-based fallback (already used when no api_key), so its
        #    structural output (PlayerScores/signals) stays intact but costs zero tokens.
        try:
            from game_theory_decision import ContextAnalyzer
            if not getattr(ContextAnalyzer, "_bella_reduced", False):
                ContextAnalyzer.analyze = lambda s, signals: (s._fallback_analysis(signals), signals)
                ContextAnalyzer._bella_reduced = True
            cut.append("game_theory:LLM->rule-based (Praxis is her real eval)")
        except Exception as e:
            print(f"[BELLA] game-theory reduce skipped: {e}")
        cut.append("emotion_appraisal:LLM->heuristic (EmotionalInference covers it)")  # via override below
        if cut:
            print("[BELLA] reduced LLM calls:\n       - " + "\n       - ".join(cut))

    # 2) EMOTION appraisal -> her heuristic emotion path + EmotionalInference already cover this.
    #    Base returns None on failure and the caller falls back to the heuristic, so overriding to
    #    None = zero tokens, same behavior. (A subclass override, cleaner than a patch.)
    def _llm_appraise_emotion(self, text: str):
        return None

    # ================= optional: the personality pill (shapes her expression) =================
    def _install_pill(self, pill_text=None):
        """The personality pill, like Eros's: an identity capsule that shapes her voice. OPTIONAL -
        default ON (so disclosure is guaranteed); drop_pill() runs her bare, take_pill(text) swaps
        personas. Hooked at the ONE API boundary every prompt path funnels through, so it applies
        everywhere, and it strips Eros's 'never self-identify as AI' line so Bella always discloses."""
        self._pill = pill_text
        es = getattr(self, "enhanced_expression_system", None)
        if es is None or getattr(es, "_bella_pill_hooked", False):
            return
        import re as _re
        original = es._call_mistral_api                 # the real bound method (captured once)
        owner = self

        async def _hooked(system_prompt, conversation_history=None, current_input="",
                          temperature=0.7, *a, **k):
            pill = getattr(owner, "_pill", None)
            if pill:
                sp = system_prompt or ""
                sp = _re.sub(r".*[Nn]ever self-identify as AI.*\n?", "", sp)   # let her disclose
                sp = _re.sub(r"[Yy]ou are Eros[.,].*\n?", "", sp)             # she is Bella, not Eros
                system_prompt = pill + "\n\n" + sp
            return await original(system_prompt, conversation_history, current_input,
                                  temperature, *a, **k)

        es._call_mistral_api = _hooked
        es._bella_pill_hooked = True

    def take_pill(self, persona_text: str):
        """Swap her persona capsule (optional). e.g. a sharper, softer, or specialist voice."""
        self._pill = persona_text

    def drop_pill(self):
        """Run with NO personality pill - bare expression (base systems still disclose she's an AI)."""
        self._pill = None

    # ================= tweak 2: final decision -> Praxis v2 (glass box) =================
    # The mouth of the river: perception + emotion + memory + belief + curiosity all stream DOWN
    # into here; Praxis makes the one provable decision; it flows on to the response generator.
    def _gather_context(self, text, relevant_facts, current_mood=None, memory_results=None) -> dict:
        """Collect the full downstream context the real systems already produced, in one place."""
        def _txt(x):
            return str(getattr(x, "text", None) or getattr(x, "content", None) or x)
        facts = [_txt(f) for f in (relevant_facts or [])][:8]
        mems = [_txt(m) for m in (memory_results or [])][:6]
        mood = current_mood if isinstance(current_mood, dict) else {}
        beliefs = []
        try:                                        # her own standing beliefs, if that system is up
            reg = getattr(self, "belief_system", None) or getattr(self, "beliefs", None)
            getter = getattr(reg, "get_active_beliefs", None) if reg else None
            if callable(getter):
                beliefs = [_txt(b) for b in (getter() or [])][:5]
        except Exception:
            pass
        return {
            "text": text, "facts": facts, "memories": mems, "beliefs": beliefs,
            "mood": mood.get("mood") or mood.get("primary") or "neutral",
            "valence": float(mood.get("valence", 0.0) or 0.0),
            "focus": str(self._focus or ""),
        }

    def _praxis_decide(self, text: str, relevant_facts, current_mood=None, memory_results=None) -> dict:
        """Perception + emotion + memory already ran (real systems). Their whole output streams
        in as context; Praxis v2 makes the provable decision and keeps the glass-box trace."""
        ctx = self._gather_context(text, relevant_facts, current_mood, memory_results)

        # MEMORY builds the substrate: fold recalled facts / memories / beliefs into her net
        ingest = []
        for chunk in ctx["facts"] + ctx["memories"] + ctx["beliefs"]:
            toks = [w for w in str(chunk).lower().replace(".", " ").replace(",", " ").split()
                    if w.isalpha() and len(w) > 3]
            for i in range(len(toks) - 1):
                ingest.append((toks[i], toks[i + 1], 0.4))
        if ingest:
            try: self.praxis.net.ingest(ingest[:24])
            except Exception: pass

        # PERCEPTION + MEMORY become the activation seeds; CURIOSITY steers hardest
        seeds = self._seeds_from(text + " " + " ".join(ctx["facts"] + ctx["memories"]), relevant_facts)
        curiosity = {w for w in ctx["focus"].lower().replace("?", " ").split()
                     if w.isalpha() and len(w) > 3}
        # EMOTION tilts the lens: unease -> think wider (lower the payoff bar), calm -> commit sooner
        min_payoff = 0.30 if ctx["valence"] < -0.15 else 0.35
        d = self.praxis.decide(
            seeds=seeds, intent_nodes=set(seeds), goal=self.goal,
            forbidden={"unverified"}, intent=text, curiosity=curiosity, min_payoff=min_payoff)
        out = {                                     # SAME dict shape CNS System-2 returns
            "thoughts": [c for c, _ in d.trace["candidates"]],
            "conclusion": d.conclusion,
            "confidence": min(0.99, 0.5 + d.payoff / 2),
            "reasoning_type": "praxis_v2_glassbox",
            "system_used": "Praxis v2",
            "trace": d.trace,                       # the glass box, on every decision
            "concepts": list(d.concepts),           # the path she reasoned across (for the shared mind)
            "context": {k: (len(v) if isinstance(v, list) else v)   # what streamed in, on the record
                        for k, v in ctx.items() if k != "text"},
            "use_conclusion_directly": False,
        }
        self._last_decision = out                   # so the action step can act on it
        return out

    # exact CNS signatures - both System-2 paths route the full context through Praxis v2
    def _enhanced_system2_reasoning(self, parsed_input, current_mood, relevant_facts,
                                    user_input, memory_results, should_use_llm):
        text = getattr(parsed_input, "raw_text", None) or user_input or str(parsed_input)
        return self._praxis_decide(text, relevant_facts, current_mood, memory_results)

    def _system2_deep_reasoning(self, parsed_input, current_mood, relevant_facts, context):
        text = getattr(parsed_input, "raw_text", None) or str(parsed_input)
        mems = context.get("memory_results") if isinstance(context, dict) else None
        return self._praxis_decide(text, relevant_facts, current_mood, mems)

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
        print(f"[BELLA-CACHE] {cache_stats()}")      # how much the cache saved this run

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
        """She explores her varied targets, rotating so she covers them all (AI -> robotics ->
        Roman lit -> art -> politics...), with her real CuriositySystem + dopamine refining in
        the background. Truly fluid, live-web wandering arrives when Indra feeds _world_intake()."""
        seeds = getattr(self, "curiosity_seeds", [])
        world = self._world_intake()
        if world:                                       # if Indra is feeding, the live world wins
            topic = world
        elif seeds:                                     # else rotate through her seed targets
            self._seed_i = (getattr(self, "_seed_i", -1) + 1) % len(seeds)
            topic = seeds[self._seed_i]
        else:
            topic = "what is true in the world right now that I don't yet understand"
        cs = getattr(self, "curiosity_system", None)
        if cs is not None:                              # feed her real curiosity/dopamine in the background
            own = ((getattr(self, "_last_decision", {}) or {}).get("conclusion") or "")
            try:
                cs.dm.decay_all()
            except Exception:
                pass
            try:
                cs.process_turn((topic + " " + own).strip())
            except Exception:
                pass
        return topic


if __name__ == "__main__":
    asyncio.run(Bella().live())
