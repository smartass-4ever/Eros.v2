"""
PRAXIS v2 - the glass-box reasoning core. Bella's final decision system.

Built from the saved design. The knowledge net is the substrate; context, memory,
and goal are FORCES acting on it:
    knowledge net = concept nodes + weighted associative edges
    context       = priming (pre-activates nodes, biases the spread)
    memory        = what builds / reputation-weights the edges
    goal + intent = the DIRECTION of the spread

Two stages (System 1 / System 2):
    1. ACTIVATION  - spreading activation over the net. Pure graph algorithm, NO LLM.
                     Natively visualizable (watch nodes glow). Surfaces "what's relevant."
    2. ASSEMBLY    - COMPOSITION: activated pieces -> candidate conclusions (LLM-pluggable;
                     a heuristic composer runs it today with no LLM).
                     EVALUATION: a game with intent+goal as payoffs -> pick. PROVABLE.

Glass box: every stage records its inputs and outputs (the telemetry recorder), and
evaluation is provable (the game). This slots into Eros's CognitiveOrchestrator in place
of the direct GameTheoryDecisionEngine call - activation + composition feed the game.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict

# --------------------------------------------------------------------------- net
class KnowledgeNet:
    """Concept nodes + weighted associative edges. Memory builds/strengthens the edges."""
    def __init__(self):
        self.edges: dict[str, list[tuple[str, float, str]]] = defaultdict(list)  # src -> [(dst,w,kind)]
        self.nodes: set[str] = set()

    def relate(self, a: str, b: str, w: float, kind: str = "assoc", both: bool = True):
        self.nodes.update((a, b))
        self.edges[a].append((b, w, kind))
        if both:
            self.edges[b].append((a, w, kind))

    def strengthen(self, a: str, b: str, delta: float):
        """Reputation update from outcomes - NOT raw co-occurrence (the poison)."""
        for i, (dst, w, kind) in enumerate(self.edges[a]):
            if dst == b:
                self.edges[a][i] = (dst, min(1.0, w + delta), kind)

    def weaken(self, a: str, b: str, delta: float):
        """A path that led to a bad outcome loses reputation - she stops trusting it."""
        for i, (dst, w, kind) in enumerate(self.edges.get(a, [])):
            if dst == b:
                self.edges[a][i] = (dst, max(0.0, w - delta), kind)

    def ingest(self, relations):
        """Grow the web from experience/memory: [(a, b, weight), ...]. Seen again -> reinforced,
        new -> a fresh association. This is how memory deepens her substrate over time."""
        for a, b, w in relations:
            if any(d == b for d, _, _ in self.edges.get(a, [])):
                self.strengthen(a, b, 0.1)
            else:
                self.relate(a, b, w)

# --------------------------------------------------------------------------- stage 1: activation
def spread(net: KnowledgeNet, seeds: dict[str, float], goal: set[str],
           steps: int = 3, decay: float = 0.6, goal_bias: float = 0.5,
           breadth: float = 0.06, focal: set = None):
    """
    Spreading activation with decay, goal-directed (edges toward goal get a boost) and a
    breadth knob (lower threshold = more distant/creative concepts survive). Returns the
    activated subgraph plus a per-step trace (the visualization).

    focal: set of perceived concepts from the current input. These stay lit longer (higher
    retention) so the input steers the reasoning rather than the graph's topology. Hub nodes
    that aren't in the focal set decay normally and can't dominate the output. When focal is
    None the behaviour is identical to before.
    """
    focal = set(focal) if focal else set()
    act: dict[str, float] = defaultdict(float, seeds)
    trace = [dict(sorted(act.items(), key=lambda x: -x[1]))]
    for _ in range(steps):
        nxt: dict[str, float] = defaultdict(float)
        for node, a in act.items():
            if a < breadth:
                continue
            # focal nodes (what she just perceived) stay lit longer so they steer the spread;
            # hub nodes that aren't in the input decay at the normal rate and can't crowd them out
            retention = 0.5 + (0.15 if node in focal else 0.0)
            nxt[node] += a * retention
            for dst, w, _kind in net.edges.get(node, []):
                boost = 1.0 + (goal_bias if dst in goal else 0.0)
                nxt[dst] += a * w * decay * boost      # goal-directed spread
        m = max(nxt.values(), default=1.0)
        act = defaultdict(float, {n: v / m for n, v in nxt.items() if v / m >= breadth})
        trace.append(dict(sorted(act.items(), key=lambda x: -x[1])[:6]))
    return dict(act), trace

# --------------------------------------------------------------------------- stage 2a: composition
@dataclass
class Candidate:
    conclusion: str
    concepts: tuple[str, str]
    strength: float

def compose(activated: dict[str, float], intent: str, composer=None) -> list[Candidate]:
    """Activated pieces -> candidate conclusions. THE opaque core (LLM in prod). Heuristic
    default pairs the most-activated concepts (the guppy-effect emergent combination)."""
    if composer:                                        # pluggable LLM hook
        return composer(activated, intent)
    top = sorted(activated.items(), key=lambda x: -x[1])[:4]
    out = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            (a, av), (b, bv) = top[i], top[j]
            out.append(Candidate(f"{a} implies {b}", (a, b), round((av + bv) / 2, 3)))
    return out[:4]

# --------------------------------------------------------------------------- stage 2b: evaluation (the game)
@dataclass
class Scored:
    cand: Candidate
    intent_fit: float
    goal_fit: float
    payoff: float
    vetoed: str | None
    curiosity_fit: float = 0.0
    trust: float = 0.0            # learned reputation of the path (from outcomes)
    gain: float = 0.0             # expected information gain (how much she'd LEARN) - for action choice
    cost: float = 0.0             # effort/risk of the option - for action choice

def _edge(net, a, b) -> float:
    for dst, w, _k in (net.edges.get(a, []) if net else []):
        if dst == b:
            return w
    return 0.0

def evaluate(cands: list[Candidate], intent_nodes: set[str], goal: set[str],
             forbidden: set[str], curiosity: set[str] = frozenset(), net=None,
             gain_fn=None, cost_map=None) -> list[Scored]:
    """A game: payoff = CURIOSITY + goal + intent + TRUST + GAIN - COST. Curiosity weighted highest;
    trust lets hard-won experience weigh in; gain (information hunger) and cost make this the SAME
    engine that chooses a course of ACTION - actions are just nodes, an action-choice is just a
    decision. Hard veto. Every payoff is decomposed - that is what makes it PROVABLE, not opaque."""
    scored = []
    for c in cands:
        cs = set(c.concepts)
        a, b = c.concepts
        intent_fit = len(cs & intent_nodes) / 2
        goal_fit = len(cs & goal) / 2
        cur_fit = (len(cs & curiosity) / 2) if curiosity else 0.0
        trust = (_edge(net, a, b) + _edge(net, b, a)) / 2 if net else 0.0
        gain = max(gain_fn(a), gain_fn(b)) if gain_fn else 0.0          # where's the most to learn
        cost = max(cost_map.get(a, 0.0), cost_map.get(b, 0.0)) if cost_map else 0.0
        veto = next((f"touches forbidden '{x}'" for x in cs & forbidden), None)
        payoff = 0.0 if veto else round(0.32 * cur_fit + 0.22 * goal_fit + 0.13 * intent_fit
                                        + 0.18 * trust + 0.05 * c.strength
                                        + 0.15 * gain - 0.10 * cost, 3)
        scored.append(Scored(c, intent_fit, goal_fit, payoff, veto, cur_fit, trust, gain, cost))
    return sorted(scored, key=lambda s: -s.payoff)

# --------------------------------------------------------------------------- the core
@dataclass
class Decision:
    conclusion: str
    payoff: float
    trace: dict
    concepts: tuple = ()          # the path she reasoned across, so she can learn from the outcome

class PraxisV2:
    def __init__(self, net: KnowledgeNet):
        self.net = net

    def decide(self, seeds: dict[str, float], intent_nodes: set[str], goal: set[str],
               forbidden: set[str] = frozenset(), intent: str = "", composer=None,
               curiosity: set[str] = frozenset(), min_payoff: float = 0.35,
               gain_fn=None, cost_map=None, focal: set = None) -> Decision:
        direction = goal | set(curiosity)          # goal + curiosity steer the spread
        _focal = set(focal) if focal else set()

        def _pass(breadth, steps):
            act, tr = spread(self.net, seeds, direction, steps=steps, breadth=breadth, focal=_focal)
            cands = compose(act, intent, composer)
            ranked = evaluate(cands, intent_nodes, goal, forbidden, curiosity, net=self.net,
                              gain_fn=gain_fn, cost_map=cost_map)
            return act, tr, cands, ranked

        activated, spread_trace, candidates, ranked = _pass(0.06, 3)
        respread = None
        # RE-SPREAD: if she hit a dead end, widen the lens and think again ("think differently")
        if not ranked or ranked[0].payoff < min_payoff:
            a2, t2, c2, r2 = _pass(0.02, 4)
            if r2 and (not ranked or r2[0].payoff > ranked[0].payoff):
                activated, spread_trace, candidates, ranked = a2, t2, c2, r2
                respread = "hit a dead end -> widened the lens and thought again"

        winner = ranked[0] if ranked else None
        trace = {
            "seeds": seeds,
            "curiosity": sorted(curiosity),
            "respread": respread,
            "activation_steps": spread_trace,
            "activated_subgraph": dict(sorted(activated.items(), key=lambda x: -x[1])[:8]),
            "candidates": [(c.conclusion, c.strength) for c in candidates],
            "evaluation": [(s.cand.conclusion, s.payoff,
                            f"curiosity {s.curiosity_fit} / goal {s.goal_fit} / trust {round(s.trust,2)}"
                            + (f" / gain {round(s.gain,2)} / cost {s.cost}" if (s.gain or s.cost) else "")
                            + (f" | VETO: {s.vetoed}" if s.vetoed else ""))
                           for s in ranked],
        }
        return Decision(winner.cand.conclusion if winner else "(no candidate survived)",
                        winner.payoff if winner else 0.0, trace,
                        winner.cand.concepts if winner else ())

    def learn(self, concepts, reward: float):
        """Ex-post: the world responded. reward > 0 strengthens the path she reasoned across
        (it earned reputation); reward < 0 weakens it. This is how she gets WISER over time -
        good reasoning paths that lead to good outcomes gain trust, bad ones fade."""
        cs = list(concepts)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                if any(d == b for d, _, _ in self.net.edges.get(a, [])):
                    (self.net.strengthen if reward >= 0 else self.net.weaken)(a, b, abs(reward) * 0.15)

# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    # a tiny knowledge net (memory would build these edges from experience)
    net = KnowledgeNet()
    net.relate("open_models", "adoption", .8)
    net.relate("open_models", "hype", .5)
    net.relate("adoption", "evidence", .7)
    net.relate("adoption", "cost_drop", .6)
    net.relate("evidence", "truth", .9)
    net.relate("hype", "unverified", .8)
    net.relate("cost_drop", "truth", .5)
    net.relate("unverified", "risk", .7)

    core = PraxisV2(net)
    # Bella perceives a claim; her intent is to understand it, her goal is truth (not hype)
    d = core.decide(
        seeds={"open_models": 1.0},          # what she just perceived
        intent_nodes={"adoption", "evidence"},
        goal={"truth", "evidence"},
        forbidden={"unverified"},            # a hard constraint: don't conclude from unverified
        intent="understand whether open models are really winning",
    )

    print("PRAXIS v2 - glass-box reasoning\n" + "=" * 60)
    print("intent: understand whether open models are really winning")
    print("goal  : {truth, evidence}   forbidden: {unverified}\n")
    print("ACTIVATION (System 1) - nodes glowing, step by step:")
    for i, layer in enumerate(d.trace["activation_steps"]):
        print(f"  t{i}: " + ", ".join(f"{n}={v:.2f}" for n, v in list(layer.items())[:6]))
    print("\nCOMPOSITION - candidate conclusions:")
    for c, s in d.trace["candidates"]:
        print(f"  - {c}  (strength {s})")
    print("\nEVALUATION (the game) - provable payoffs:")
    for c, p, why in d.trace["evaluation"]:
        print(f"  {p:>5}  {c:<28} [{why}]")
    print("\nDECISION:", d.conclusion, f"(payoff {d.payoff})")
    print("\nevery step above is recorded and inspectable - this is the glass box.")
