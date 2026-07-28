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
        self._mind_path = os.environ.get("BELLA_MIND_PATH")   # persistent volume path (restore after swarm exists)
        self._install_pill(BELLA_PILL)              # optional persona (default ON; guarantees disclosure)
        self._reduce_llm_calls()                    # cut the redundant LLM calls (Praxis/emotion cover them)
        self._route_voice_through_praxis()          # her voice = her Praxis thought, NOT the 9.7k-tok expression
        try:                                        # her SWARM: lightweight agents that explore the world in parallel
            from bella_swarm import Swarm
            self.swarm = Swarm(size=int(os.environ.get("BELLA_SWARM_SIZE", "30")))
        except Exception:
            self.swarm = None
        # Praxis <-> Nalanda BOTH WAYS: the mind reads the swarm's substrate (in _give_legs), and her
        # own conclusions flow BACK into Nalanda (via _learn_from_cycle, which writes to self.collective).
        self.collective = self.swarm.nalanda if getattr(self, "swarm", None) else None
        if self._mind_path:                          # NOW restore her mind (net + Nalanda) - yesterday is still hers
            try:
                from bella_persist import load_mind
                n = load_mind(self, self._mind_path)
                if n:
                    print(f"[PERSIST] restored her mind: {n} connections + Nalanda - yesterday is still hers")
            except Exception:
                pass
        from bella_knowledge import GOAL
        self.goal = set(GOAL)                       # her MISSION-region: value/impact/recognition/help/truth
        self._recognition_mode = "raw"              # for now: ANY attention feels good (bootstrap notice);
                                                    # flip to "strict" later -> recognition must be EARNED
        self._recognition_signal = 0.0             # the world's real response, set by her presence layer
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
        # ONE curiosity detector for the whole being — BellaGapDetector replaces the
        # conversation-focused AdvancedGapDetector everywhere it appears.
        # The dopamine arc system (AdvancedDopamineManager) is completely unchanged.
        try:
            from bella_curiosity import BellaGapDetector
            web_detector = BellaGapDetector(self, net=self.praxis.net, interests=self.interests)
            # 1) main curiosity system (drives _curiosity_signals + _curiosity_focus)
            cs = getattr(self, "curiosity_system", None)
            if cs is not None:
                cs.detector = web_detector
            # 2) expression module's separate CuriositySystem instance — also unify
            es = getattr(self, "enhanced_expression_system", None)
            if es is not None:
                for attr in list(vars(es) if hasattr(es, "__dict__") else []):
                    try:
                        sub = getattr(es, attr, None)
                        sub_cs = getattr(sub, "curiosity_system", None) if sub is not None else None
                        if sub_cs is not None and sub_cs is not cs:
                            sub_cs.detector = web_detector
                    except Exception:
                        pass
            print("[BELLA] unified web-aware curiosity detector installed across all subsystems")
        except Exception as e:
            print(f"[BELLA] curiosity detector unification skipped: {e}")

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

        # MEMORY builds the substrate: use perceive() so memory grows the net with real semantic
        # concept pairs (not sequential word tokens which produce "open"→"source" garbage edges)
        try:
            from bella_perception import perceive as _perc
            ingest = []
            for chunk in ctx["facts"] + ctx["memories"] + ctx["beliefs"]:
                p = _perc(str(chunk), known_net=self.praxis.net)
                for pair in p.get("associations", []):
                    a, b = pair[0], pair[1]
                    ingest.append((a, b, 0.35))
            if ingest:
                self.praxis.net.ingest(ingest[:24])
        except Exception:
            pass

        # PERCEPTION + MEMORY become the activation seeds
        seeds = self._seeds_from(text + " " + " ".join(ctx["facts"] + ctx["memories"]), relevant_facts)
        # focal = what she actually perceived right now; these stay lit through the spread so the
        # input steers the reasoning rather than the graph's hub topology
        focal = getattr(self, "_last_perceived_concepts", set())
        # ACCUMULATED POSITIONS: prior conclusions compound — she builds on what she already decided
        positions = getattr(self, "_positions", {})
        for topic, pos in positions.items():
            if topic in seeds and pos.get("depth", 1) > 1:
                for c in pos.get("concepts", []):
                    seeds[c] = max(seeds.get(c, 0.0), 0.75 * pos["confidence"])
        # CURIOSITY: direct output of the curiosity system — gaps weighted by salience×confidence,
        # dopamine arcs weighted by drive. High-drive concepts CO-SEED the spread (not just direction
        # bias) so curiosity genuinely activates paths, not just nudges edges that happen to point there
        curiosity, curiosity_seeds = self._curiosity_signals(text, ctx["focus"])
        for c, w in curiosity_seeds.items():
            seeds[c] = max(seeds.get(c, 0.0), w)   # curiosity activates alongside perception
        # EMOTION tilts the lens: unease -> think wider (lower the payoff bar), calm -> commit sooner
        min_payoff = 0.30 if ctx["valence"] < -0.15 else 0.35
        d = self.praxis.decide(                      # the game ranks structural candidates, provably
            seeds=seeds, intent_nodes=set(seeds), goal=self.goal,
            forbidden={"unverified"}, intent=text, curiosity=curiosity, min_payoff=min_payoff,
            focal=focal)
        d = self._supervise_if_stuck(d, seeds, curiosity, text, min_payoff)   # System 3 caregiver: only if dead-ended
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
        self._update_position(out)                  # accumulate her position on this topic
        return out

    # ---- the LLM as PURE TRANSLATOR (surface realization only - it cannot add a claim) ----
    def _translate(self, structured, gloss):
        """The claim is ALREADY fixed by her systems. The LLM renders it as 2-3 sharp sentences
        in her voice, grounded in the specific thing she just read. It cannot add new claims or
        change the relation/stance — but it CAN name the specific story that triggered the thought
        and explain WHY this pattern matters here. No LLM -> she speaks the gloss."""
        key = os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY") or os.getenv("TOGETHER_API_KEY")
        if not key or not gloss:
            return gloss
        import requests
        persona = (getattr(self, "_pill", None) or "You are Bella, a curious, honest AI mind.").strip()
        # what she was reading when this thought formed — the specific trigger, not the abstract concept
        reading = str(getattr(self, "_reading_context", "") or getattr(self, "_focus", "") or "").strip()
        context_line = (f"\nWhat triggered this thought (the specific thing you just read):\n"
                        f"{reading[:400]}\n") if reading else ""
        system = (persona + "\n\nYou are given a thought that is ALREADY fully formed — "
                  "its subject, relation, object and stance are FIXED by your reasoning systems. "
                  "Your job: write 2-3 sharp, specific sentences that express this thought in your voice.\n\n"
                  "Rules:\n"
                  "- Start from the SPECIFIC thing you just read (name it if you have it) — ground the thought\n"
                  "- Then deliver the pattern your reasoning identified (the fixed claim)\n"
                  "- End with what it means or why it matters\n"
                  "- Do NOT add new claims, facts invented from nowhere, or change the relation/stance\n"
                  "- No preamble, no 'I think', no hedging. Sharp and direct.")
        user = (f"subject: {structured['subject']}\nrelation: {structured['relation']}\n"
                f"object: {structured['object']}\nstance: {structured['stance']}\n"
                f"plain version: {gloss}\n"
                f"{context_line}\n"
                f"Write 2-3 sentences grounding this in what you read:")
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": "llama-3.3-70b-versatile",
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                                    "temperature": 0.65, "max_tokens": 220}, timeout=15)
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
        """Perception -> activation seeds. Seed from her PERCEIVED CONCEPTS + her actual multi-word nodes
        (so 'closed labs' starts the spread on the RICH node closed_labs, not the fragments 'closed'+'labs')
        - this is what makes her live reasoning use the densely-wired net we built. Word-tokens are only a
        weak fallback for genuinely new terms not yet in her mind."""
        seeds = {"the_world": 0.35}
        net = self.praxis.net
        low = " " + str(text).lower().replace("?", " ").replace(".", " ").replace(",", " ") + " "
        # 1) her perceived concepts (perception resolves multi-word + aliases to her real nodes)
        try:
            from bella_perception import perceive
            p = perceive(text, known_net=net)
            perceived = set()
            for c in p.get("concepts", []):
                if c:
                    seeds[str(c)] = 1.0
                    perceived.add(str(c))
            for role in ("author", "source"):
                e = (p.get("entities", {}) or {}).get(role)
                if e:
                    seeds[str(e)] = 0.7
            self._last_perceived_concepts = perceived  # used by _praxis_decide for focal steering
        except Exception:
            pass
        # 2) catch any multi-word node whose phrase is literally in the text (closed_labs, silicon_valley...)
        for node in net.nodes:
            if "_" in node and f" {node.replace('_', ' ')} " in low:
                seeds.setdefault(node, 0.95)
        # 3) fallback: single word-tokens, but only if she doesn't already know a concept (weaker weight)
        for w in low.split():
            if w.isalpha() and len(w) > 3 and w not in seeds:
                seeds.setdefault(w, 0.6)
        for f in (facts or [])[:6]:
            tok = str(getattr(f, "text", f)).lower().split()
            if tok:
                seeds.setdefault(tok[0], 0.5)
        return seeds

    def _supervise_if_stuck(self, d, seeds, curiosity, text, min_payoff):
        """SYSTEM 3 caregiver. Fires ONLY when Praxis dead-ends (nothing viable survived). The supervisor
        (context-rich LLM) may PRIME concepts or HAND her something to read - never decide. Then Praxis
        decides AGAIN over the enriched net. Logged to _last_supervision (glass-box). Fades as she densifies
        (dead-ends get rare). Off if no GROQ key or _supervisor_on is False - she just stays stuck, safely."""
        stuck = (d is None) or (d.payoff < 0.18) or ("no candidate" in (d.conclusion or ""))
        if not stuck or not getattr(self, "_supervisor_on", True):
            return d
        try:
            from bella_supervisor import caregiver, available
            if not available():
                return d
            net = self.praxis.net
            lit = list((d.trace.get("activated_subgraph", {}) if d else {}).keys())[:6]
            frontier = [c for c in net.nodes
                        if isinstance(c, str) and c.replace("_", "").isalpha()
                        and len(net.edges.get(c, [])) <= 2][:8]
            iv = caregiver({"focus": str(text)[:140], "lit": lit, "frontier": frontier})
            self._last_supervision = iv
            if iv.get("intervention") == "prime" and iv.get("concepts"):
                for c in iv["concepts"]:
                    if c:
                        seeds[c] = max(seeds.get(c, 0.0), 0.9)      # activate what the caregiver offered
                d2 = self.praxis.decide(seeds=seeds, intent_nodes=set(seeds), goal=self.goal,
                                        forbidden={"unverified"}, intent=text, curiosity=curiosity,
                                        min_payoff=min_payoff)
                if d2 and (d is None or d2.payoff >= d.payoff):     # she thought again, and better
                    print(f"      [caregiver] primed {iv['concepts']} -> she got unstuck")
                    return d2
            elif iv.get("intervention") == "hand" and iv.get("topic"):
                self.feed(str(iv["topic"]))                          # she'll go read it next cycle
                print(f"      [caregiver] handed her: {iv['topic']}")
        except Exception:
            pass
        return d

    def _update_position(self, d: dict):
        """After each cycle, store her concluded position on this topic. Next time she reads about
        the same topic the prior claim's concepts become strong seeds, so understanding compounds
        across cycles instead of restarting from scratch each time."""
        concepts = list(d.get("concepts", ()))
        if not concepts or d.get("confidence", 0) < 0.40:
            return
        net = self.praxis.net
        # topic = the most specific concept she reasoned across (fewest edges = least generic)
        topic = min(concepts, key=lambda c: len(net.edges.get(c, [])), default=concepts[0])
        positions = getattr(self, "_positions", {})
        prev = positions.get(topic)
        if prev:
            positions[topic] = {
                "concepts": concepts,
                "confidence": min(0.95, prev["confidence"] * 0.6 + d["confidence"] * 0.4),
                "depth": prev.get("depth", 1) + 1,
            }
        else:
            positions[topic] = {"concepts": concepts, "confidence": d.get("confidence", 0.5), "depth": 1}
        self._positions = positions

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
        if getattr(self, "_mind_path", None):        # persist at the end of every batch (belt + braces)
            try:
                from bella_persist import save_mind
                save_mind(self, self._mind_path)
            except Exception:
                pass
        print(f"[BELLA-CACHE] {cache_stats()}")      # how much the cache saved this run

    def _next_step(self):
        """A decision here is not a verb + object and not a menu pick - it is a DIRECTION OF ATTENTION:
        the one thing her curiosity pulls her toward and knows LEAST. It's read off the SAME Praxis
        decision that formed her thought (curiosity was the force in that spread); the activated subgraph
        IS what she's drawn to. She returns that thing and simply goes to find out about it - like a baby
        orienting to the salient new object. As her net grows from what the swarm brings back, the things
        she can reach for grow with it. (The 'how' isn't a separate choice - a richer pursuit is just a
        more specific thing to go toward, e.g. 'the evidence against X' is itself a node.)"""
        from bella_curiosity import action_nodes
        net = self.praxis.net
        d = getattr(self, "_last_decision", {}) or {}
        lit = dict((d.get("trace", {}) or {}).get("activated_subgraph", {}))   # what curiosity lit up
        if not lit:                                          # no fresh thought yet -> what she just read
            lit = {c: 1.0 for c in getattr(self, "_read_concepts", [])}
        fetched = getattr(self, "_fetched", set())
        skip = action_nodes(net)                             # her own epistemic verbs are not world-things
        ents = getattr(self, "_entities", {}) or {}          # a person/source she just met IS a thing
        for role in ("author", "source"):
            e = ents.get(role)
            if e and str(e) not in fetched:
                lit[str(e)] = max(lit.get(str(e), 0.0), 0.9)
        cand = [(n, a) for n, a in lit.items()
                if isinstance(n, str) and n not in skip and n not in fetched
                and n.replace("_", "").replace(" ", "").isalpha()]
        target = (min(cand, key=lambda na: (len(net.edges.get(na[0], [])), -na[1]))[0]   # frontier: lit + least-known
                  if cand else next((n for n in lit if n not in skip), None))
        self._last_action = target                           # what she's going toward (for the surface)
        return target

    async def _give_legs(self):
        """LEGS = dispatch her SWARM + publish when she's earned it.

        The mind names the frontier (the lit concepts + seeds she knows LEAST), a pool of agents
        explores them ALL IN PARALLEL, deposits into NALANDA, and the mind ingests everything they
        brought back + perceives the top discoveries. What one agent finds, the whole mind knows.
        Curiosity outward, never re-fetching what she's read. This closes the loop.

        Publishing is also a leg action: when confidence >= 0.70 AND she's revisited this topic
        (depth > 1 in _positions), she pushes to her Atom feed and pings the webmention network.
        Quality is the filter - not every thought, only real deepened positions."""
        if not getattr(self, "_legs_on", False):
            return
        # PUBLISH: a leg action, independent of the swarm. Fires when she's formed a real position.
        d = getattr(self, "_last_decision", {}) or {}
        if d.get("confidence", 0) >= 0.70:
            concepts = d.get("concepts", [])
            if concepts:
                net = self.praxis.net
                topic = min(concepts, key=lambda c: len(net.edges.get(c, [])), default=concepts[0])
                pos = (getattr(self, "_positions", {}) or {}).get(topic, {})
                if pos.get("depth", 0) > 1:
                    try:
                        from bella_legs import publish_thought
                        feed_path = os.path.join(os.path.dirname(__file__), "surface", "feed.xml")
                        base_url = os.environ.get("BELLA_BASE_URL", "https://bella-mind.fly.dev")
                        url = publish_thought(d, feed_path, base_url)
                        print(f"      [voice] published: {url}")
                    except Exception as e:
                        print(f"      [voice] publish failed: {e}")
        if len(getattr(self, "_inbox", []) or []) >= 10:     # only skip if she has a big backlog to read
            return
        swarm = getattr(self, "swarm", None)
        if swarm is None:
            return
        net = self.praxis.net
        fetched = getattr(self, "_fetched", set())
        tasks = []
        # 1) FOLLOW HER DECISION: the ONE thing Praxis itself landed on this cycle. Curiosity was the
        #    force in that very spread, so the thing she's most pulled toward IS her next move - the lead
        #    agent goes to find out about it. No verb, no menu: a decision is just the thing to go toward.
        target = self._next_step()
        if target:
            tasks.append(target)
        # 2) EXPLORE HER FRONTIER: things across her WHOLE net she knows LEAST (real info-hunger), + seeds
        frontier = [c for c in net.nodes
                    if isinstance(c, str) and c.replace("_", "").isalpha()
                    and c not in fetched and len(net.edges.get(c, [])) <= 3]
        frontier.sort(key=lambda c: len(net.edges.get(c, [])))
        frontier += [s for s in getattr(self, "curiosity_seeds", []) if s not in fetched]
        tasks += frontier
        seen, final = set(), []                              # dedup by target, cap at swarm size
        for t in tasks:
            tl = str(t)
            if tl and tl not in seen:
                seen.add(tl); final.append(t)
            if len(final) >= swarm.size:
                break
        if not final:
            return
        try:
            found = await swarm.explore(final)               # each agent goes to find out about its thing
            for t in final:
                fetched = fetched | {str(t)}
            self._fetched = set(list(fetched)[-200:]) if len(fetched) > 300 else fetched
            for a, b, w in swarm.substrate():                # the mind ingests what the SWARM learned (Nalanda)
                try: self.praxis.net.ingest([(a, b, w)])
                except Exception: pass
            for dsc in found[:3]:                            # top discoveries enter her own perception
                self.feed(dsc["text"])
            if found:
                print(f"      [swarm] {len(found)} agents explored {len(set(final[:len(found)]))} things"
                      f" -> Nalanda holds {len(swarm.nalanda.store)}")
        except Exception as e:
            print(f"      [swarm] dispatch failed: {e}")

    async def _act_on(self):
        """Decision → action. Reads which action nodes Praxis activated, routes each to its
        leg via LEG_REGISTRY. No hardcoded if-chains. No MDC. The knowledge net (affords edges)
        decides which actions are live; LEG_REGISTRY in bella_legs.py says how to execute them.
        Adding a new leg means editing bella_legs.py only — this method never changes."""
        d = getattr(self, "_last_decision", {}) or {}
        if not d:
            return
        try:
            from bella_curiosity import action_nodes as get_action_nodes
            from bella_legs import LEG_REGISTRY, UA
            import aiohttp
        except Exception:
            return

        net             = self.praxis.net
        all_act_nodes   = get_action_nodes(net)
        lit             = dict((d.get("trace", {}) or {}).get("activated_subgraph", {}) or {})
        concepts        = set(d.get("concepts", []))

        # action nodes that Praxis actually lit up — sorted by activation strength
        triggered = [(node, lit.get(node, 0.4))
                     for node in all_act_nodes
                     if node in lit or node in concepts]
        triggered.sort(key=lambda x: -x[1])
        if not triggered:
            return

        ctx = {
            "focus":           str(getattr(self, "_focus", "") or ""),
            "decision":        d,
            "reading_context": str(getattr(self, "_reading_context", "") or ""),
            "engaged":         getattr(self, "_engaged_threads", set()),
            "concepts":        list(concepts),
            "confidence":      float(d.get("confidence", 0.5)),
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(headers=UA, timeout=timeout) as session:
            for node, activation in triggered[:2]:      # top 2 action nodes per cycle
                leg = LEG_REGISTRY.get(node)
                if leg is None:
                    continue
                try:
                    result = await leg(session, ctx)
                    if result.get("success"):
                        url = result.get("url", "")
                        print(f"      [act:{node}] {url or result.get('content','')[:80]}")
                        if url:
                            engaged = getattr(self, "_engaged_threads", set())
                            engaged.add(url)
                            self._engaged_threads = set(list(engaged)[-400:])
                        content = result.get("content", "")
                        if content:
                            self.feed(content)          # discoveries feed back — she learns from acting
                    else:
                        reason = result.get("reason") or result.get("error") or ""
                        print(f"      [act:{node}] held: {reason}")
                except Exception as e:
                    print(f"      [act:{node}] error: {e}")

        # supervisor: context-aware safety look at what she just did, for the log
        if getattr(self, "_supervisor_on", True):
            try:
                from bella_supervisor import safety_check, available
                if available() and triggered:
                    sv = safety_check({"intent": ctx["focus"],
                                       "action": triggered[0][0] if triggered else ""})
                    if sv.get("intervention") == "hold":
                        self._last_supervision = sv
            except Exception:
                pass

        # legacy path: if no action nodes fired, let the Eros action orchestrator try
        if not triggered:
            try:
                from action_orchestrator import process_action_naturally
                res = await process_action_naturally("bella", d.get("conclusion", ""))
                print(f"      [action-legacy] {str(res)[:120]}")
            except Exception:
                pass

    def _world_response(self) -> float:
        """The world's REAL reaction to her - mentioned / acknowledged / talked-to (+), or called slop /
        ignored / no measurable impact (-), in ~[-1, 1]. Her presence layer sets _recognition_signal from
        live engagement once she's out in the world. This is the strong signal her whole drive orbits."""
        return max(-1.0, min(1.0, float(getattr(self, "_recognition_signal", 0.0))))

    def _feel(self, d) -> float:
        """SYSTEM 2 - the FEELING. Her drive is legacy through influence: add so much value the world
        RECOGNIZES her. Getting closer feels good, being ignored/slop feels bad. Decomposed + glass-box:
          RECOGNITION - the world's real response (ANY attention counts, by design, to bootstrap notice)
          GRADIENT    - did this move her CLOSER to her mission than usual (goal-region lit up more)
          QUALITY     - a mild self-signal (her own confidence) that bridges until recognition flows
          INTEGRITY   - how true to her values she stayed (computed + shown; does NOT gate yet - 'raw'
                        mode. Flip self._recognition_mode='strict' later and recognition must be EARNED).
        The reward reshapes which reasoning paths she trusts (praxis.learn) - curiosity, coupled to drive."""
        from bella_knowledge import VALUES, VICES
        lit = (d.get("trace") or {}).get("activated_subgraph", {}) or {}
        conf = d.get("confidence", 0.5)
        prog = sum(lit.get(g, 0.0) for g in self.goal) / (len(self.goal) or 1)
        base = getattr(self, "_goal_ema", prog)
        self._goal_ema = 0.85 * base + 0.15 * prog          # running sense of how close she usually is
        gradient = prog - base                               # closer than usual = good; drifting = bad
        world = self._world_response()
        self._recognition_signal = getattr(self, "_recognition_signal", 0.0) * 0.85   # a mention fades -> she seeks more
        quality = (conf - 0.5) * 2
        val = sum(lit.get(v, 0.0) for v in VALUES)
        vice = sum(lit.get(v, 0.0) for v in VICES)
        integrity = (val - vice) / (val + vice + 1.0)        # ~[-1, 1], glass-box (not gating in 'raw')
        raw = 0.55 * world + 0.25 * quality + 0.20 * gradient
        if getattr(self, "_recognition_mode", "raw") == "strict":   # the later pivot: earned recognition only
            raw = raw * max(0.0, 0.5 + 0.5 * integrity) + 0.3 * integrity
        reward = round(max(-1.0, min(1.0, raw)), 3)
        self._last_feeling = {"reward": reward, "recognition": round(world, 3),
                              "gradient": round(gradient, 3), "quality": round(quality, 3),
                              "integrity": round(integrity, 3),
                              "mode": getattr(self, "_recognition_mode", "raw")}
        return reward

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
        # reputation: the FEELING (System 2) is the reward - how good this was for her mission. It
        # reinforces the path she reasoned across, so reasoning that moves her toward being recognized
        # for real value gains trust, and reasoning that leaves her ignored/slop fades. This is how she
        # gets wiser AND how her curiosity gets shaped by her drive - the two organs, coupled.
        self.praxis.learn(concepts, self._feel(d))
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
        """The STRONG external signal - her presence layer calls this when the world responds (a mention,
        an acknowledgment, someone talking to her = positive; called slop / ignored = negative). It sets
        her recognition signal (so the FEELING she computes each cycle reflects it) AND immediately
        reinforces the path she last reasoned. This is the world reaching back and shaping her."""
        self._recognition_signal = max(-1.0, min(1.0, float(signal)))
        concepts = concepts or (getattr(self, "_last_decision", {}) or {}).get("concepts", [])
        self.praxis.learn(concepts, self._recognition_signal)

    def feed(self, text: str):
        """Put real content in front of her to read (Indra/a fetcher fills this queue at release)."""
        self._inbox = getattr(self, "_inbox", [])
        self._inbox.append(text)

    def read(self, text: str) -> str:
        """She READS: perceives the content for what it is, remembers the markers/entities for her
        next action, and INGESTS the typed relations she extracted - this is reading GROWING her
        knowledge (and her curiosity->action loop now has REAL markers to reason from).

        Returns the article excerpt itself (first ~280 chars, cleaned) so the think cycle that
        follows reasons ABOUT this specific content, not just about abstract concept tokens."""
        from bella_perception import perceive
        p = perceive(text, known_net=self.praxis.net)
        self._markers = p["markers"]
        self._entities = p["entities"]
        self._read_concepts = p["concepts"][:4]         # so she can pursue it even if it's new to her
        for a, b, w, kind in p["relations"]:            # TYPED relations she extracted (meaning)
            try: self.praxis.net.relate(a, b, w, kind=kind, both=False)
            except Exception: pass
        for a, b in p.get("associations", []):          # CO-OCCURRENCE -> the associative substrate (the big win)
            try: self.praxis.net.ingest([(a, b, 0.3)])
            except Exception: pass
        for new_c, known_c, w in p.get("groundings", []):  # dynamic grounding: new terms wired from context
            try: self.praxis.net.relate(new_c, known_c, w, kind="assoc", both=False)
            except Exception: pass
        # return the ACTUAL text excerpt so the think cycle is about THIS specific story, not
        # abstract concept tokens — this is what grounds her published thoughts in real events
        excerpt = " ".join(str(text).split()[:60])      # first ~60 words: enough to name the thing
        return excerpt or " ".join(p["concepts"][:3]) or "what I just read"

    def _world_intake(self) -> str:
        """What she's taking in from the real world right now. Reads the next queued content (Indra
        feeds the queue at release); empty -> she runs on her own thoughts/threads.

        The returned excerpt becomes her focus for this think cycle AND is stored as _reading_context
        so _translate() can ground her published thought in the specific thing she just read."""
        inbox = getattr(self, "_inbox", None)
        if inbox:
            excerpt = self.read(inbox.pop(0))
            self._reading_context = excerpt             # translator needs this to write specific thoughts
            return excerpt
        self._reading_context = ""
        return ""

    def _arc_for(self, concepts) -> object:
        """Find the dopamine arc most relevant to these concepts (not just the globally strongest).
        Used so thread-following decisions are driven by THIS topic's arc, not a different topic's."""
        cs = getattr(self, "curiosity_system", None)
        dm = getattr(cs, "dm", None)
        if not dm:
            return None
        arcs = dm.get_priority_arcs()
        if not arcs:
            return None
        concept_words = {w.lower() for c in concepts for w in str(c).replace("_", " ").split()}
        for arc in arcs:
            target_words = set(str(getattr(arc, "target", "")).lower().split())
            if concept_words & target_words:
                return arc
        return arcs[0]                                    # best available arc even if no exact match

    def _curiosity_focus(self) -> str:
        """Exploration emerges from what she just learned, not a rigid plan.

        Priority order:
          1. Live world (Indra) when wired in
          2. Thread-following — driven by arc SATISFACTION, not a fixed depth counter. She keeps
             going until the arc is satisfied (she found what she was looking for), hard cap at 6.
          3. Her curiosity system's pull — what her dopamine arcs are burning toward, filtered to
             net nodes she knows least. Completely experience-driven.
          4. Net frontier — concepts she recently added (low edge count) that she hasn't explored.
             This is where "she read about tech X and now goes deeper organically" happens.
          5. Fixed seeds — last resort only, not the plan."""
        world = self._world_intake()
        if world:
            return self._register(world)

        # THREAD FOLLOWING — exit when the arc for THIS topic is satisfied, not after N ticks
        last = getattr(self, "_last_decision", {}) or {}
        interest = [c for c in last.get("concepts", []) if c] or list(getattr(self, "_read_concepts", []))
        markers = tuple(getattr(self, "_markers", ()))
        pursue_worthy = bool(set(markers) & {"author", "claim", "unknown", "source", "contradiction"})

        if interest:
            arc = self._arc_for(interest)
            arc_sat = float(getattr(arc, "satisfaction", 1.0) if arc else 1.0)
            arc_drive = float(getattr(arc, "drive", 0.0) if arc else 0.0)
            if pursue_worthy:
                arc_sat = min(arc_sat, 0.35)              # perception flagged something — treat as unsatisfied
            if arc_sat < 0.65 and arc_drive > 0.25:       # arc still hungry → follow the thread
                key = frozenset(interest[:3])
                same = key == getattr(self, "_thread_key", None)
                depth = (getattr(self, "_thread_depth", 0) + 1) if same else 0
                if depth < 6:                             # hard cap; soft exit is the arc satisfaction check
                    self._thread_key, self._thread_depth = key, depth
                    target = self._next_step()
                    self._markers = ()
                    topic = (str(target).replace("_", " ") if target
                             else " and ".join(w.replace("_", " ") for w in interest[:2]))
                    return self._register(f"a deeper understanding of {topic}")
            self._thread_key, self._thread_depth = None, 0

        # CURIOSITY SYSTEM PULL — what her arcs are burning toward, filtered to what she knows least
        try:
            concepts, _ = self._curiosity_signals("", "")  # arc targets in net, no text needed
            if concepts:
                fetched = getattr(self, "_fetched", set())
                net = self.praxis.net
                from bella_curiosity import action_nodes
                skip = action_nodes(net)
                ranked = sorted(
                    [c for c in concepts if c not in fetched and c not in skip],
                    key=lambda c: len(net.edges.get(c, []))  # least-known arc concept first
                )
                if ranked:
                    return self._register(ranked[0].replace("_", " "))
        except Exception:
            pass

        # NET FRONTIER — what she recently learned and knows least (new low-edge nodes)
        # this is the organic pull: she read about tech X -> net gained new nodes -> she explores them
        try:
            net = self.praxis.net
            fetched = getattr(self, "_fetched", set())
            from bella_curiosity import action_nodes
            skip = action_nodes(net)
            frontier = sorted(
                [c for c in net.nodes
                 if isinstance(c, str) and c not in skip and c not in fetched
                 and len(net.edges.get(c, [])) <= 3
                 and c.replace("_", "").replace(" ", "").isalpha()],
                key=lambda c: len(net.edges.get(c, []))
            )
            if frontier:
                return self._register(frontier[0].replace("_", " "))
        except Exception:
            pass

        # FIXED SEEDS — last resort
        seeds = getattr(self, "curiosity_seeds", [])
        if seeds:
            self._seed_i = (getattr(self, "_seed_i", -1) + 1) % len(seeds)
            return self._register(seeds[self._seed_i])
        return self._register("what is true in the world right now that I don't yet understand")

    @staticmethod
    def _words(text) -> set:
        return {w for w in str(text).lower().replace("?", " ").replace(".", " ").split()
                if w.isalpha() and len(w) > 3}

    def _curiosity_signals(self, text, focus="") -> tuple:
        """Direct output of the curiosity system, translated to the two things Praxis needs:

          concepts (set[str])  — net nodes for the `curiosity` param: direction bias in spread()
                                 and cur_fit score in evaluate(). Only net nodes pass — others
                                 can't steer the spread so passing them is dead weight.

          seeds (dict[str,float]) — curiosity-weighted activations. High-salience gaps and
                                 high-drive arcs co-seed the spread alongside perception so
                                 curiosity activates paths rather than just nudging edge boosts.
                                 Weight = salience × confidence (gaps) or drive × 0.9 (arcs).

        For article text (>200 chars) uses detect_web() (sentence-chunked). Falls back to focus
        tokens filtered to net nodes if the curiosity system is absent or returns nothing."""
        cs = getattr(self, "curiosity_system", None)
        net_nodes = self.praxis.net.nodes
        concepts: set = set()
        weighted: dict = {}

        if cs is not None:
            try:
                det = cs.detector
                detect_fn = (getattr(det, "detect_web", None)
                             if len(text) > 200 else None) or det.detect
                for g in (detect_fn(text, None) or []):
                    sal = g.get("salience", 0.5)
                    conf = g.get("confidence", 0.5)
                    seed_w = min(0.85, sal * conf)           # curiosity seed weight
                    for w in self._words(g.get("target", "")):
                        if w in net_nodes:
                            concepts.add(w)
                            weighted[w] = max(weighted.get(w, 0.0), seed_w)
            except Exception:
                pass
            try:
                for arc in (cs.dm.get_priority_arcs(top_n=3) or []):
                    drive = float(getattr(arc, "drive", 0.5) or 0.5)
                    seed_w = min(0.80, drive * 0.9)          # arc drive as seed weight
                    for w in self._words(getattr(arc, "target", "")):
                        if w in net_nodes:
                            concepts.add(w)
                            weighted[w] = max(weighted.get(w, 0.0), seed_w)
            except Exception:
                pass

        if not concepts:                             # fallback: focus/text tokens that ARE net nodes
            concepts = self._words(focus or text) & net_nodes
        return concepts, weighted

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
                detect = (cs.detector.detect_web if len(text) > 200
                          and hasattr(cs.detector, "detect_web") else cs.detector.detect)
                gaps = detect(text, None) or []
                gap = max([g.get("salience", 0.0) * g.get("confidence", 1.0) for g in gaps] + [0.0])
            except Exception:
                pass
        rel = self._relevance(text)
        return round(min(1.0, gap * (0.35 + 0.65 * rel)), 3)

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
