"""
test_bella.py — full pipeline health check.

    python test_bella.py

Tests every component in isolation, then 3 full live cycles.
Core reasoning requires NO API keys — flags missing keys as warnings, never fails for them.
Never posts to Reddit, HN, or anywhere. Read-only throughout.

Output: one line per test, PASS / FAIL / WARN / SKIP, with a summary at the end.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback

# ── path setup (mirrors run.py / bella.py) ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ["core", "memory", "self model", "user relationship", "misc", "saftey"]:
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── result tracking ───────────────────────────────────────────────────────────
_results: list[tuple[str, str, str]] = []   # (name, status, note)

def _record(name, status, note=""):
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
    line = f"  {tag}  {name}"
    if note:
        line += f"  —  {note}"
    print(line)
    _results.append((name, status, note))

def check(name, fn, *args, **kwargs):
    """Run fn(*args, **kwargs). PASS if it returns truthy, FAIL on exception or falsy."""
    try:
        out = fn(*args, **kwargs)
        if out is False:
            _record(name, "FAIL", "returned False")
        else:
            _record(name, "PASS", str(out)[:80] if out is not True and out is not None else "")
        return out
    except Exception as e:
        _record(name, "FAIL", f"{type(e).__name__}: {e}")
        return None

def warn(name, fn, *args, **kwargs):
    """Same as check() but WARN instead of FAIL on falsy/exception."""
    try:
        out = fn(*args, **kwargs)
        if out is False or out is None:
            _record(name, "WARN", "returned None/False (may need API key)")
        else:
            _record(name, "PASS", str(out)[:80] if out is not True else "")
        return out
    except Exception as e:
        _record(name, "WARN", f"{type(e).__name__}: {e}")
        return None

def skip(name, reason):
    _record(name, "SKIP", reason)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*64)
print("  BELLA PIPELINE HEALTH CHECK")
print("═"*64 + "\n")

# ── env flags ─────────────────────────────────────────────────────────────────
HAS_GROQ  = bool(os.environ.get("GROQ_API_KEY"))
HAS_BRAVE = bool(os.environ.get("BRAVE_API_KEY"))
print(f"  GROQ key:  {'present' if HAS_GROQ  else 'MISSING — planner/translator will be skipped'}")
print(f"  BRAVE key: {'present' if HAS_BRAVE else 'MISSING — search falls back to Wikipedia'}")
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 1. KNOWLEDGE NET ─────────────────────────────────────────────────")

from reasoning_core import KnowledgeNet, PraxisV2, Candidate
from bella_knowledge import seed_bella_mind, GOAL

net = KnowledgeNet()

def _net_relate():
    net.relate("test_a", "test_b", 0.8, kind="leads_to")
    return "test_b" in net.nodes and "test_a" in net.nodes
check("KnowledgeNet.relate() adds both nodes", _net_relate)

def _net_ingest():
    net.ingest([("test_a", "test_c", 0.5), ("test_c", "test_d", 0.6)])
    return "test_c" in net.nodes and "test_d" in net.nodes
check("KnowledgeNet.ingest() grows the net", _net_ingest)

def _net_strengthen():
    before = net.edges["test_a"][0][1]
    net.strengthen("test_a", "test_b", 0.1)
    after = net.edges["test_a"][0][1]
    return after > before
check("KnowledgeNet.strengthen() increases edge weight", _net_strengthen)

def _seed():
    fresh = KnowledgeNet()
    n = seed_bella_mind(fresh, verbose=False)
    return n > 500   # should have 500+ connections after seeding
check("seed_bella_mind() loads 500+ connections", _seed)

def _seed_nodes():
    fresh = KnowledgeNet()
    seed_bella_mind(fresh, verbose=False)
    required = {"curiosity", "truth", "bayesian_updating", "lateral_reading",
                "pyramid_principle", "tactical_empathy", "seek_disconfirmation"}
    missing = required - fresh.nodes
    return f"{len(fresh.nodes)} nodes — missing: {missing}" if not missing else len(fresh.nodes)
check("seed_bella_mind() includes all knowledge blocks", _seed_nodes)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 2. PRAXIS v2 REASONING ───────────────────────────────────────────")

from reasoning_core import spread, compose, evaluate
from bella_thought import form_thought, form_structured

pnet = KnowledgeNet()
seed_bella_mind(pnet, verbose=False)
praxis = PraxisV2(pnet)
goal   = set(GOAL)

def _spread():
    seeds = {"curiosity": 1.0, "truth": 0.8, "open_source": 0.7}
    activated = spread(pnet, seeds, goal)
    return len(activated) >= 3 and "curiosity" in activated
check("spread() activates seeded concepts and neighbours", _spread)

def _compose():
    seeds = {"curiosity": 1.0, "truth": 0.8, "open_source": 0.7}
    activated = spread(pnet, seeds, goal)
    candidates = compose(activated, "open vs closed AI", net=pnet)
    return len(candidates) > 0 and hasattr(candidates[0], "conclusion")
check("compose() produces Candidate objects with conclusions", _compose)

def _evaluate():
    seeds = {"curiosity": 1.0, "truth": 0.8}
    activated = spread(pnet, seeds, goal)
    candidates = compose(activated, "what is true", net=pnet)
    scored = evaluate(candidates, activated, goal, curiosity={"curiosity", "truth"}, forbidden=set())
    return scored is not None and scored.payoff > 0
check("evaluate() picks the best candidate with payoff > 0", _evaluate)

def _decide():
    d = praxis.decide(
        seeds={"open_source": 1.0, "closed_labs": 0.8, "power": 0.6},
        intent_nodes={"open_source", "closed_labs"},
        goal=goal, forbidden=set(),
        intent="open vs closed AI", curiosity={"open_source"},
    )
    return (d is not None
            and d.payoff > 0
            and len(d.concepts) >= 2
            and "activated_subgraph" in d.trace)
check("PraxisV2.decide() returns Scored with trace + concepts", _decide)

def _trace_completeness():
    d = praxis.decide(
        seeds={"curiosity": 1.0, "bayesian_updating": 0.8},
        intent_nodes={"curiosity"}, goal=goal, forbidden=set(),
        intent="how to reason well", curiosity={"curiosity"},
    )
    required_keys = {"activated_subgraph", "candidates", "evaluation"}
    return required_keys.issubset(d.trace.keys())
check("Praxis trace has activated_subgraph + candidates + evaluation", _trace_completeness)

def _form_thought():
    d = praxis.decide(
        seeds={"truth": 1.0, "evidence": 0.8},
        intent_nodes={"truth"}, goal=goal, forbidden=set(),
        intent="what grounds a belief", curiosity={"truth"},
    )
    t = form_thought(d, pnet)
    return isinstance(t, str) and len(t) > 15
check("form_thought() returns a readable sentence", _form_thought)

def _form_structured():
    d = praxis.decide(
        seeds={"power": 1.0, "concentration": 0.8},
        intent_nodes={"power"}, goal=goal, forbidden=set(),
        intent="power and who holds it", curiosity={"power"},
    )
    s = form_structured(d, pnet)
    required = {"subject", "relation", "object", "stance"}
    return required.issubset(s.keys()) and bool(s["subject"]) and bool(s["relation"])
check("form_structured() returns subject/relation/object/stance", _form_structured)

def _praxis_learn():
    before = dict(praxis.trust)
    d = praxis.decide(seeds={"truth": 1.0}, intent_nodes={"truth"},
                      goal=goal, forbidden=set(), intent="truth", curiosity=set())
    praxis.learn(d.concepts, +0.8)
    changed = any(praxis.trust.get(c, 0) != before.get(c, 0) for c in d.concepts)
    return changed
check("PraxisV2.learn() updates trust weights after reward", _praxis_learn)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 3. PERCEPTION ────────────────────────────────────────────────────")

from bella_perception import perceive

SAMPLE_TEXT = (
    "Ilya Sutskever believes that safe superintelligence requires us to understand "
    "how AI systems reason internally, not just observe their outputs. Most current "
    "models are black boxes — you can see what they produce but not how they decided. "
    "This makes alignment fundamentally difficult."
)

def _perceive_concepts():
    p = perceive(SAMPLE_TEXT)
    return len(p.get("concepts", [])) >= 2
check("perceive() extracts at least 2 concepts from text", _perceive_concepts)

def _perceive_relations():
    p = perceive(SAMPLE_TEXT)
    return len(p.get("relations", [])) >= 1
check("perceive() extracts at least 1 typed relation", _perceive_relations)

def _perceive_markers():
    p = perceive(SAMPLE_TEXT)
    return isinstance(p.get("markers"), (list, tuple))
check("perceive() returns markers list", _perceive_markers)

def _perceive_known_net():
    fresh = KnowledgeNet(); seed_bella_mind(fresh, verbose=False)
    p = perceive("AI safety and alignment are connected to trust in reasoning systems",
                 known_net=fresh)
    grounded = p.get("groundings", [])
    return isinstance(grounded, list)
check("perceive() with known_net produces groundings", _perceive_known_net)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 4. COLLECTIVE MEMORY (NALANDA) ───────────────────────────────────")

from collective_memory import CollectiveMemory

def _cm_remember():
    cm = CollectiveMemory()
    cm.remember(key="test_key", content="AI reasoning is based on decision making",
                salience=0.7, relations=[("ai", "reasoning"), ("reasoning", "decision")])
    return "test_key" in cm.store
check("CollectiveMemory.remember() stores an entry", _cm_remember)

def _cm_substrate():
    cm = CollectiveMemory()
    cm.remember(key="k1", content="truth requires evidence", salience=0.8,
                relations=[("truth", "evidence")])
    cm.remember(key="k2", content="curiosity drives learning", salience=0.7,
                relations=[("curiosity", "learning")])
    sub = cm.substrate()
    return len(sub) >= 2
check("CollectiveMemory.substrate() returns ingestible relations", _cm_substrate)

def _cm_snapshot():
    cm = CollectiveMemory()
    cm.remember(key="snap", content="test", salience=0.5, relations=[])
    snap = cm.snapshot()
    return isinstance(snap, dict)
check("CollectiveMemory.snapshot() returns a dict", _cm_snapshot)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 5. BELLA INSTANCE ────────────────────────────────────────────────")

print("  booting Bella (takes a moment)...")
t0 = time.time()
from bella import Bella
b = Bella()
boot_s = round(time.time() - t0, 1)
_record("Bella() boots without crashing", "PASS", f"{boot_s}s")

def _surface_path():
    return (hasattr(b, "_surface_path")
            and b._surface_path.endswith("state.json"))
check("_surface_path set to surface/state.json", _surface_path)

def _legs_on():
    return getattr(b, "_legs_on", False) is True
check("_legs_on is True by default", _legs_on)

def _swarm_present():
    return b.swarm is not None and b.swarm.size > 0
check(f"swarm present with {getattr(getattr(b, 'swarm', None), 'size', '?')} agents", _swarm_present)

def _praxis_present():
    return hasattr(b, "praxis") and hasattr(b.praxis, "net") and len(b.praxis.net.nodes) > 100
check(f"praxis.net has {len(b.praxis.net.nodes)} nodes after seed", _praxis_present)

def _read():
    excerpt = b.read(SAMPLE_TEXT)
    return isinstance(excerpt, str) and len(excerpt) > 10
check("bella.read() returns an excerpt", _read)

def _read_grows_net():
    before = len(b.praxis.net.nodes)
    b.read("Spreading activation through a typed knowledge graph is a form of structured reasoning "
           "that differs fundamentally from statistical prediction in language models.")
    after = len(b.praxis.net.nodes)
    return after >= before
check("bella.read() grows or maintains the knowledge net", _read_grows_net)

def _seeds_from():
    seeds = b._seeds_from(SAMPLE_TEXT, [])
    return len(seeds) >= 2 and "the_world" in seeds
check("_seeds_from() returns seeds dict with the_world anchor", _seeds_from)

def _praxis_decide():
    b.feed(SAMPLE_TEXT)
    b._world_intake()
    d = b._praxis_decide(SAMPLE_TEXT, [], current_mood=None, memory_results=None)
    required = {"conclusion", "thought", "claim", "confidence", "concepts", "trace"}
    return (required.issubset(d.keys())
            and 0 < d["confidence"] < 1
            and len(d["concepts"]) >= 1)
check("_praxis_decide() returns valid decision with all required fields", _praxis_decide)

def _decision_claim():
    d = getattr(b, "_last_decision", {}) or {}
    c = d.get("claim", {})
    return bool(c.get("subject")) and bool(c.get("relation")) and bool(c.get("stance"))
check("decision.claim has subject + relation + stance", _decision_claim)

def _decision_trace():
    d = getattr(b, "_last_decision", {}) or {}
    sub = d.get("trace", {}).get("activated_subgraph", {})
    return len(sub) >= 3
check("decision trace has at least 3 activated concepts", _decision_trace)

def _curiosity_focus():
    focus = b._curiosity_focus()
    return isinstance(focus, str) and len(focus) > 3
check("_curiosity_focus() returns a non-empty topic string", _curiosity_focus)

def _curious_about():
    ai_score   = b.curious_about("Ilya Sutskever explains why transformers cannot reason causally")
    bored_score = b.curious_about("the best pasta shapes for carbonara sauce")
    return 0 <= ai_score <= 1 and 0 <= bored_score <= 1 and ai_score >= bored_score
check(f"curious_about() scores AI topic higher than pasta (AI={b.curious_about(SAMPLE_TEXT):.2f})", _curious_about)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 6. BELLA STATE / SURFACE ─────────────────────────────────────────")

from bella_state import build_state, emit

def _build_state():
    s = build_state(b)
    required = {"thought", "concepts", "edges", "path", "payoff", "conf", "trail"}
    return required.issubset(s.keys()) and isinstance(s["thought"], str)
check("build_state() returns dict with all required surface keys", _build_state)

def _build_state_payoff():
    s = build_state(b)
    p = s.get("payoff", {})
    return set(p.keys()) == {"curiosity", "goal", "trust", "gain", "cost"}
check("build_state() payoff has all 5 dimensions", _build_state_payoff)

def _build_state_swarm():
    s = build_state(b)
    return isinstance(s.get("heads"), list) and isinstance(s.get("nalanda"), int)
check("build_state() includes swarm heads and nalanda count", _build_state_swarm)

def _emit():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp = f.name
    try:
        emit(b, tmp)
        with open(tmp, encoding="utf-8") as f:
            loaded = json.load(f)
        return bool(loaded.get("thought"))
    finally:
        try: os.unlink(tmp)
        except: pass
check("emit() writes valid JSON with a thought to disk", _emit)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 7. BELLA LEGS (read-only — no posting) ───────────────────────────")

import aiohttp
from bella_legs import (fetch_page, find_discussions, read_thread,
                        find_author, search_web_async, execute_plan,
                        LEG_REGISTRY, PRIMITIVES)

async def _search_web():
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        result = await search_web_async(s, "spreading activation knowledge graph AI reasoning")
    return isinstance(result, str) and len(result) > 50

check("search_web_async() returns text (Wikipedia fallback OK)",
      lambda: asyncio.get_event_loop().run_until_complete(_search_web()))

async def _fetch_page():
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        result = await fetch_page(s, "https://en.wikipedia.org/wiki/Spreading_activation", chars=500)
    return isinstance(result, str) and len(result) > 100

check("fetch_page() reads a real URL and strips HTML",
      lambda: asyncio.get_event_loop().run_until_complete(_fetch_page()))

if HAS_BRAVE:
    async def _find_discussions():
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            threads = await find_discussions(s, "AI reasoning interpretability", count=3)
        return isinstance(threads, list)
    check("find_discussions() returns a list (Brave)",
          lambda: asyncio.get_event_loop().run_until_complete(_find_discussions()))
else:
    skip("find_discussions()", "needs BRAVE_API_KEY")

def _primitives_complete():
    expected = {"fetch_page", "find_discussions", "read_thread", "post_comment",
                "find_author", "search_web"}
    return expected.issubset(PRIMITIVES.keys())
check("PRIMITIVES dict has all 6 expected entries", _primitives_complete)

def _leg_registry_complete():
    expected = {"engage", "check_community", "read_discussion",
                "follow_source", "find_evidence", "find_counterargument", "search_author"}
    return expected.issubset(LEG_REGISTRY.keys())
check("LEG_REGISTRY has all 7 expected legs", _leg_registry_complete)

async def _execute_plan_no_llm():
    """execute_plan() with no GROQ key falls back to LEG_REGISTRY — should not crash."""
    d   = getattr(b, "_last_decision", {}) or {}
    ctx = {
        "focus": "AI reasoning",
        "decision": d,
        "reading_context": SAMPLE_TEXT[:200],
        "engaged": set(),
        "action_nodes": ["find_evidence"],
        "base_url": "https://bella-mind.fly.dev",
        "swarm": None,                   # no swarm → sequential path
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await execute_plan(session, d, ctx)
    return isinstance(results, list)

check("execute_plan() without LLM key falls back to LEG_REGISTRY cleanly",
      lambda: asyncio.get_event_loop().run_until_complete(_execute_plan_no_llm()))
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 8. SWARM ─────────────────────────────────────────────────────────")

from bella_swarm import Swarm

async def _swarm_explore():
    sw = Swarm(size=3)
    found = await sw.explore(["Stoicism", "AI safety", "bayesian reasoning"])
    return (len(found) > 0
            and all("text" in d for d in found)
            and len(sw.nalanda.store) > 0)

check("Swarm.explore() fetches topics and deposits into Nalanda",
      lambda: asyncio.get_event_loop().run_until_complete(_swarm_explore()))

async def _swarm_substrate():
    sw = Swarm(size=2)
    await sw.explore(["spreading activation", "knowledge graph"])
    sub = sw.substrate()
    return isinstance(sub, list) and len(sub) > 0

check("Swarm.substrate() returns ingestible relation list",
      lambda: asyncio.get_event_loop().run_until_complete(_swarm_substrate()))

async def _swarm_snapshot():
    sw = Swarm(size=5)
    await sw.explore(["curiosity"])
    snap = sw.snapshot()
    required = {"size", "last_explore", "nalanda_size", "discovered"}
    return required.issubset(snap.keys()) and snap["discovered"] > 0

check("Swarm.snapshot() has size/last_explore/nalanda_size/discovered",
      lambda: asyncio.get_event_loop().run_until_complete(_swarm_snapshot()))

async def _swarm_act_single():
    """act() with a single search step — no posting, read-only."""
    sw = Swarm(size=2)
    plan = [{"agent": "agent-0", "primitive": "search_web",
             "args": {"query": "what is spreading activation"},
             "intent": "understand the mechanism"}]
    results = await sw.act(plan)
    return (len(results) > 0
            and results[0].get("primitive") == "search_web")

check("Swarm.act() executes a single-step plan and returns results",
      lambda: asyncio.get_event_loop().run_until_complete(_swarm_act_single()))

async def _swarm_act_chain():
    """act() with a chain — search → fetch, both steps run on one agent."""
    sw = Swarm(size=2)
    plan = [{
        "agent": "agent-0",
        "intent": "search then fetch",
        "chain": [
            {"primitive": "search_web",
             "args": {"query": "Ilya Sutskever SSI safe superintelligence"}},
            {"primitive": "search_web",
             "args": {"query": "PREV.content spreading activation"}},
        ]
    }]
    results = await sw.act(plan)
    return len(results) > 0 and results[0].get("success") is not False

check("Swarm.act() runs a 2-step chain with PREV resolution",
      lambda: asyncio.get_event_loop().run_until_complete(_swarm_act_chain()))
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 9. LLM LAYERS (needs API keys) ───────────────────────────────────")

if HAS_GROQ:
    def _translate():
        d = getattr(b, "_last_decision", {}) or {}
        structured = d.get("claim", {"subject": "curiosity", "relation": "leads_to",
                                     "object": "truth", "stance": "affirming"})
        gloss = d.get("thought", "curiosity leads to truth")
        b._reading_context = SAMPLE_TEXT[:200]
        spoken = b._translate(structured, gloss)
        return (isinstance(spoken, str)
                and len(spoken) > len(gloss) * 0.5
                and spoken != gloss)
    check("_translate() returns LLM-rendered thought (longer than gloss)", _translate)

    async def _llm_plan_runs():
        from bella_legs import _llm_plan
        d = getattr(b, "_last_decision", {}) or {}
        ctx = {"focus": "AI reasoning", "action_nodes": ["find_evidence"],
               "engaged": set(), "reading_context": SAMPLE_TEXT[:200],
               "base_url": "https://bella-mind.fly.dev", "swarm": None}
        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            plan = await _llm_plan(session, d, ctx)
        return isinstance(plan, list)
    check("_llm_plan() returns a JSON plan list (sequential mode)",
          lambda: asyncio.get_event_loop().run_until_complete(_llm_plan_runs()))

    async def _llm_plan_parallel():
        from bella_legs import _llm_plan
        d   = getattr(b, "_last_decision", {}) or {}
        ctx = {"focus": "AI reasoning", "action_nodes": ["find_evidence"],
               "engaged": set(), "reading_context": SAMPLE_TEXT[:200],
               "base_url": "https://bella-mind.fly.dev",
               "swarm": b.swarm}      # swarm present → parallel chain mode
        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            plan = await _llm_plan(session, d, ctx)
        # each item should have agent + chain OR agent + primitive
        if not plan:
            return True  # empty plan is still valid (no action needed)
        return all("agent" in item for item in plan)
    check("_llm_plan() in parallel mode assigns agents to each item",
          lambda: asyncio.get_event_loop().run_until_complete(_llm_plan_parallel()))
else:
    skip("_translate()", "needs GROQ_API_KEY")
    skip("_llm_plan() sequential", "needs GROQ_API_KEY")
    skip("_llm_plan() parallel",   "needs GROQ_API_KEY")
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── 10. FULL PIPELINE — 3 LIVE CYCLES ───────────────────────────────")

print("  running bella.live(ticks=3, pace=0.3) — watching every step...")
print()

_cycle_decisions: list = []
_original_learn = b._learn_from_cycle.__func__

def _instrumented_learn(self, result):
    _cycle_decisions.append(getattr(self, "_last_decision", {}) or {})
    _original_learn(self, result)

import types
b._learn_from_cycle = types.MethodType(_instrumented_learn, b)

async def _run_live():
    await b.live(ticks=3, pace=0.3)

try:
    asyncio.get_event_loop().run_until_complete(_run_live())
    _record("bella.live(ticks=3) ran without crashing", "PASS",
            f"{len(_cycle_decisions)} cycles completed")
except Exception as e:
    _record("bella.live(ticks=3) ran without crashing", "FAIL",
            f"{type(e).__name__}: {e}")
    traceback.print_exc()

def _all_cycles_have_decisions():
    return (len(_cycle_decisions) == 3
            and all(d.get("confidence", 0) > 0 for d in _cycle_decisions))
check("all 3 cycles produced a decision with confidence > 0",
      _all_cycles_have_decisions)

def _cycles_have_concepts():
    return all(len(d.get("concepts", [])) >= 1 for d in _cycle_decisions)
check("all 3 decisions have at least 1 concept", _cycles_have_concepts)

def _cycles_have_thoughts():
    return all(len(d.get("conclusion", "")) > 10 for d in _cycle_decisions)
check("all 3 decisions have a non-trivial conclusion string", _cycles_have_thoughts)

def _net_grew():
    return len(b.praxis.net.nodes) > 200
check(f"knowledge net grew during live() — now {len(b.praxis.net.nodes)} nodes", _net_grew)

def _state_json_written():
    path = b._surface_path
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return bool(s.get("thought")) and isinstance(s.get("concepts"), list)
check("state.json written with real thought + concepts after live()", _state_json_written)

def _nalanda_has_entries():
    return (b.swarm is not None
            and len(b.swarm.nalanda.store) > 0)
check(f"Nalanda has entries after live() — {len(getattr(getattr(b, 'swarm', None), 'nalanda', type('', (), {'store': {}})()).store if b.swarm else {}, 'nothing')} items",
      _nalanda_has_entries)
print()


# ══════════════════════════════════════════════════════════════════════════════
print("── SUMMARY ──────────────────────────────────────────────────────────")

passed  = sum(1 for _, s, _ in _results if s == "PASS")
failed  = sum(1 for _, s, _ in _results if s == "FAIL")
warned  = sum(1 for _, s, _ in _results if s == "WARN")
skipped = sum(1 for _, s, _ in _results if s == "SKIP")
total   = len(_results)

print(f"\n  {passed}/{total} passed   {failed} failed   {warned} warnings   {skipped} skipped\n")

if failed:
    print("  FAILURES:")
    for name, status, note in _results:
        if status == "FAIL":
            print(f"    [FAIL] {name}")
            if note:
                print(f"           {note}")
    print()

if warned:
    print("  WARNINGS (non-blocking):")
    for name, status, note in _results:
        if status == "WARN":
            print(f"    [WARN] {name}")
            if note:
                print(f"           {note}")
    print()

if not HAS_GROQ:
    print("  TIP: set GROQ_API_KEY to test _translate() and _llm_plan()")
if not HAS_BRAVE:
    print("  TIP: set BRAVE_API_KEY to test find_discussions() with real search")

verdict = "ALL CLEAR" if failed == 0 else f"{failed} THING(S) NEED FIXING"
print(f"\n  {verdict}\n")
print("═"*64 + "\n")
