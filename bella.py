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
from reasoning_core import PraxisV2, KnowledgeNet, Candidate
from bella_knowledge import seed_bella_mind
from bella_thought import form_thought, form_structured


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
        seed_bella_mind(self.praxis.net)            # the baseline mind (bootstrap)
        self._mind_path = os.environ.get("BELLA_MIND_PATH")   # persistent volume path on the host
        if self._mind_path:                          # she REMEMBERS what she learned before the reboot
            try:
                from bella_persist import load_mind
                n = load_mind(self, self._mind_path)
                if n:
                    print(f"[PERSIST] restored her mind: {n} connections - yesterday is still hers")
            except Exception:
                pass
        self._install_pill(BELLA_PILL)              # optional persona (default ON; guarantees disclosure)
        self._reduce_llm_calls()                    # cut the redundant LLM calls (Praxis/emotion cover them)
        self._route_voice_through_praxis()          # her voice = her Praxis thought, NOT the 9.7k-tok expression
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

    def _route_voice_through_praxis(self):
        """Remove ONLY the expensive expression performance. The FULL pipeline still runs every cycle
        (perception, emotion, memory, curiosity, the psychology/rapport path - all of it feeds Praxis);
        we just stop the ~9.7k-token strategic expression LLM call and let her SPEAK her own Praxis
        thought (already in her voice via _translate). Kills the token burn; keeps the architecture."""
        es = getattr(self, "enhanced_expression_system", None)
        if es is None:
            return
        owner = self

        async def _voice(system_prompt=None, conversation_history=None, current_input="",
                         temperature=0.7, *a, **k):
            d = getattr(owner, "_last_decision", {}) or {}     # her Praxis thought, her own words
            return d.get("conclusion") or d.get("thought") or None

        es._call_mistral_api = _voice                          # the expression now voices Praxis, no big call
        es._bella_voice_routed = True

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
        # curiosity FORCE = her REAL CuriositySystem's gaps + live dopamine arcs (what she is
        # GENUINELY curious about), NOT tokenized focus words. Her real curiosity finally DRIVES.
        curiosity = self._curiosity_force(text, ctx["focus"])
        # EMOTION tilts the lens: unease -> think wider (lower the payoff bar), calm -> commit sooner
        min_payoff = 0.30 if ctx["valence"] < -0.15 else 0.35
        d = self.praxis.decide(                      # the game ranks structural candidates, provably
            seeds=seeds, intent_nodes=set(seeds), goal=self.goal,
            forbidden={"unverified"}, intent=text, curiosity=curiosity, min_payoff=min_payoff)
        # HER systems form the complete claim (typed graph + stance), NO LLM. The LLM only translates.
        structured = form_structured(d, self.praxis.net)
        thought = form_thought(d, self.praxis.net)
        spoken = self._translate(structured, thought)
        out = {                                     # SAME dict shape CNS System-2 returns
            "thoughts": [c for c, _ in d.trace["candidates"]],
            "conclusion": spoken,                   # her formed thought (LLM only smoothed the words)
            "thought": thought,                     # her token-free thought (no LLM at all)
            "claim": structured,                    # the deep structure: subject/relation/object/stance
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

    # ---- the LLM as PURE TRANSLATOR (surface realization only - it cannot add a claim) ----
    def _translate(self, structured, gloss):
        """The claim (subject / relation / object / stance) is ALREADY fixed by her systems. The LLM
        only renders it as one natural sentence, adding NO new claim, fact, or idea, and changing
        neither the relation nor the stance. No LLM/quota -> she speaks the gloss (her own words).
        THIS is 'LLM = mouth, not mind', done properly: her thought, borrowed language."""
        key = os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not key or not gloss:
            return gloss
        import requests
        persona = (getattr(self, "_pill", None) or "You are Bella, a curious, honest AI mind.").strip()
        system = (persona + "\n\nYou are given a thought that is ALREADY fully formed - its subject, "
                  "relation, object and stance are FIXED. Render it as ONE natural, sharp sentence in "
                  "your voice. Do NOT add any new claim, fact, example or idea; do NOT change the "
                  "relation or the stance. Only say THIS exact thought more naturally. No preamble.")
        user = (f"subject: {structured['subject']}\nrelation: {structured['relation']}\n"
                f"object: {structured['object']}\nstance: {structured['stance']}\n"
                f"my plain version: {gloss}\n\nSay it naturally (one sentence, no new claims):")
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": "llama-3.3-70b-versatile",
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                                    "temperature": 0.6, "max_tokens": 60}, timeout=12)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip() or gloss
        except Exception:
            pass
        return gloss

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
            await self._give_legs()                  # decision -> LEGS: dispatch to the world for NEW material
            if getattr(self, "_surface_path", None):  # stream her live cognition to the public surface
                try:
                    from bella_state import emit
                    emit(self, self._surface_path)
                except Exception:
                    pass
            self._cyc = getattr(self, "_cyc", 0) + 1   # PERSIST her mind so she never reboots to zero
            if getattr(self, "_mind_path", None) and self._cyc % 8 == 0:
                try:
                    from bella_persist import save_mind
                    save_mind(self, self._mind_path)
                except Exception:
                    pass
            print()
            history = (history + [{"role": "user", "content": focus},
                                  {"role": "assistant", "content": thought}])[-40:]
            await asyncio.sleep(pace)
        print(f"[BELLA-CACHE] {cache_stats()}")      # how much the cache saved this run

    async def _give_legs(self):
        """LEGS: turn her concluded action into a real-world dispatch (one fetcher now; the Ravana swarm
        later). She follows her curiosity OUTWARD - the target is the concept that just lit up that she
        knows LEAST (highest info-hunger), and she never re-fetches what she's already read. So she moves
        stoicism -> virtue -> Zeno -> ... instead of re-reading the same page. This CLOSES the loop."""
        if not getattr(self, "_legs_on", False):
            return
        if len(getattr(self, "_inbox", []) or []) >= 2:      # she still has fresh material to read
            return
        d = getattr(self, "_last_decision", {}) or {}
        trace = d.get("trace", {}) or {}
        lit = list(trace.get("activated_subgraph", {}).keys()) or [c for c in d.get("concepts", []) if c]
        fetched = getattr(self, "_fetched", set())
        net = self.praxis.net
        cands = [c for c in lit if isinstance(c, str) and c.replace("_", "").isalpha() and c not in fetched]
        if not cands:                                        # exhausted this neighbourhood -> wander to a fresh seed
            cands = [s for s in getattr(self, "curiosity_seeds", []) if s not in fetched]
        if not cands:
            return
        target = min(cands, key=lambda c: len(net.edges.get(c, [])))   # what she knows LEAST -> curiosity pulls out
        action = getattr(self, "_last_action", "explore")
        try:
            from bella_legs import dispatch
            discovery = await asyncio.to_thread(dispatch, action, target, "")
            if discovery:
                fetched = fetched | {target}
                self._fetched = set(list(fetched)[-24:]) if len(fetched) > 40 else fetched   # forget old, can revisit
                self.feed(discovery)                         # the swarm returns -> perception next cycle
                print(f"      [legs] {action} '{target}' -> brought back {len(discovery)} chars from the world")
        except Exception as e:
            print(f"      [legs] dispatch failed: {e}")

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
        # 1b) STRONG loop: reward the ACTION she chose to get here, so her exploration policy sharpens
        act = getattr(self, "_last_action", None)
        if act:
            try:
                from bella_curiosity import learn_from_action
                learn_from_action(self.praxis, act, max(-1.0, min(1.0, (conf - 0.5) * 2)),
                                  context=getattr(self, "_action_context", ()))
            except Exception:
                pass
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

    def feed(self, text: str):
        """Put real content in front of her to read (Indra/a fetcher fills this queue at release)."""
        self._inbox = getattr(self, "_inbox", [])
        self._inbox.append(text)

    def read(self, text: str) -> str:
        """She READS: perceives the content for what it is, remembers the markers/entities for her
        next action, and INGESTS the typed relations she extracted - this is reading GROWING her
        knowledge (and her curiosity->action loop now has REAL markers to reason from)."""
        from bella_perception import perceive
        p = perceive(text, known_net=self.praxis.net)
        self._markers = p["markers"]
        self._entities = p["entities"]
        self._read_concepts = p["concepts"][:4]         # so she can pursue it even if it's new to her
        for a, b, w, kind in p["relations"]:            # learn from what she just read
            try: self.praxis.net.relate(a, b, w, kind=kind, both=False)
            except Exception: pass
        return " ".join(p["concepts"][:3]) or "what I just read"

    def _world_intake(self) -> str:
        """What she's taking in from the real world right now. Reads the next queued content (Indra
        feeds the queue at release); empty -> she runs on her own thoughts/threads."""
        inbox = getattr(self, "_inbox", None)
        if inbox:
            return self.read(inbox.pop(0))
        return ""

    def _curiosity_focus(self) -> str:
        """Genuine curiosity FOLLOWS THREADS. If the last thing gripped her, she REASONS an action
        (Praxis over her procedural knowledge, no LLM) to go DEEPER - chase the author, hunt the
        evidence, read more - and that becomes her next focus. Only when nothing pulls her does she
        fall back to a fresh seed (the seed-list is now the FALLBACK, not the plan). Live web (Indra)
        wins when wired into _world_intake()."""
        world = self._world_intake()
        if world:                                       # the live world, when Indra feeds it
            return self._register(world)

        # FOLLOW THE THREAD - pursue what just gripped her, but only while it's still FRESH.
        last = getattr(self, "_last_decision", {}) or {}
        # what gripped her = the decision's concepts, or what she just READ (even if new to her graph)
        interest = [c for c in last.get("concepts", []) if c] or list(getattr(self, "_read_concepts", []))
        markers = tuple(getattr(self, "_markers", ()))
        pursue_worthy = bool(set(markers) & {"author", "claim", "unknown", "source", "contradiction"})
        dope = self._interest_level(interest)
        if pursue_worthy:
            dope = max(dope, 0.7)                          # perception flagged something worth chasing
        if interest and dope >= 0.55:
            key = frozenset(interest[:3])                 # habituation: same thread N times -> satisfied
            same = key == getattr(self, "_thread_key", None)
            depth = (getattr(self, "_thread_depth", 0) + 1) if same else 0
            if depth < 3:                                 # dive ~3 levels, then get bored and wander
                self._thread_key, self._thread_depth = key, depth
                from bella_curiosity import decide_next_action, action_to_focus
                action, _scores = decide_next_action(self.praxis, interest, markers, dope)  # Praxis's game chooses
                self._last_action = action
                self._action_context = markers + tuple(interest[:3])   # so she LEARNS situation->action
                self._markers = ()                        # consume the markers (fresh perception resets them)
                topic = " and ".join(w.replace("_", " ") for w in interest[:2])
                return self._register(action_to_focus(action, topic, getattr(self, "_entities", {})))
            self._thread_key, self._thread_depth = None, 0    # thread exhausted -> seek something new

        # nothing pulls her (or a thread just satisfied) -> a fresh seed
        seeds = getattr(self, "curiosity_seeds", [])
        if seeds:
            self._seed_i = (getattr(self, "_seed_i", -1) + 1) % len(seeds)
            return self._register(seeds[self._seed_i])
        return self._register("what is true in the world right now that I don't yet understand")

    @staticmethod
    def _words(text) -> set:
        return {w for w in str(text).lower().replace("?", " ").replace(".", " ").split()
                if w.isalpha() and len(w) > 3}

    def _curiosity_force(self, text, focus="") -> set:
        """Praxis's strongest term, sourced from her REAL curiosity system: the GAPS her gap-detector
        finds in this input + the live dopamine ARCS (what she's already curious about). That is her
        genuine curiosity, as concepts, driving the decision. Falls back to focus tokens if absent."""
        cs = getattr(self, "curiosity_system", None)
        concepts = set()
        if cs is not None:
            try:                                    # gaps this input opens (works on a single sentence)
                for g in (cs.detector.detect(text, None) or []):
                    concepts |= self._words(g.get("target", ""))
            except Exception:
                pass
            try:                                    # what she's already burning to know (dopamine arcs)
                for arc in (cs.dm.get_priority_arcs(top_n=3) or []):
                    concepts |= self._words(getattr(arc, "target", ""))
            except Exception:
                pass
        return concepts or self._words(focus or text)

    def _relevance(self, text) -> float:
        """How connected this is to what she already KNOWS and CARES about - read straight off her
        real knowledge graph (her memory) + her interests. This is the dimension her conversational
        curiosity system lacks (it scores novelty, not 'is this in my wheelhouse'). 0..1."""
        words = self._words(text)
        if not words:
            return 0.0
        interests_l = " ".join(getattr(self, "interests", [])).lower()
        net = self.praxis.net
        hits = sum(1 for w in words if w in net.nodes or w in interests_l)
        return round(min(1.0, hits / max(3, len(words))), 3)

    def curious_about(self, text) -> float:
        """Genuine reading-curiosity = BOTH her real systems: her gap-detector (is something
        interesting here - novelty/contradiction/story) WEIGHTED by relevance to what she knows and
        cares about (her knowledge graph). A gap about AI grips her; the same gap about Emacs doesn't.
        0..1. Not a keyword hack - the combination of her two real signals."""
        cs = getattr(self, "curiosity_system", None)
        gap = 0.0
        if cs is not None:
            try:
                gaps = cs.detector.detect(text, None) or []
                gap = max([g.get("salience", 0.0) * g.get("confidence", 1.0) for g in gaps] + [0.0])
            except Exception:
                pass
        rel = self._relevance(text)
        return round(min(1.0, gap * (0.35 + 0.65 * rel)), 3)   # a gap she CARES about beats one she doesn't

    def _interest_level(self, concepts) -> float:
        """How hard her curiosity/dopamine is pulling (0..1): her real arcs if present, else the
        last decision's confidence as a proxy. This is what makes curiosity her STRONGEST drive."""
        cs = getattr(self, "curiosity_system", None)
        try:
            dm = getattr(cs, "dm", None)
            arcs = dm.get_priority_arcs() if dm is not None else None
            if arcs:
                top = arcs[0]
                return min(1.0, float(getattr(top, "intensity", getattr(top, "strength", 0.6))))
        except Exception:
            pass
        return float((getattr(self, "_last_decision", {}) or {}).get("confidence", 0.6))

    def _register(self, topic: str) -> str:
        """Feed the chosen focus into her real curiosity/dopamine system so the arcs stay alive."""
        self._trail = (getattr(self, "_trail", []) + [topic])[-8:]   # her curiosity trail (for the surface)
        cs = getattr(self, "curiosity_system", None)
        if cs is not None:
            try: cs.dm.decay_all()
            except Exception: pass
            try: cs.process_turn(topic)
            except Exception: pass
        return topic


if __name__ == "__main__":
    asyncio.run(Bella().live())
