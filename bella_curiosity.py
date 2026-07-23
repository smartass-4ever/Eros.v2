"""
BELLA'S CURIOSITY-TO-ACTION ENGINE - she reasons her way from a question to a DEED. No LLM advisor.

The missing bridge: curiosity generates a question, and until now she'd just ask it. This turns
the question into an ACTION she takes to explore deeper - and she gets there by REASONING (Praxis),
not by asking a model. Her procedural knowledge (the EPISTEMIC relations: curiosity->explore,
interesting source->follow it, unknown->read more, claim->find evidence) is in her graph; spreading
activation flows from what gripped her to an ACTION node. That action node IS her next move. It's
provable and it's hers.

    curiosity/interest (dopamine)  ->  reason (Praxis, goal = deepen)  ->  land on an ACTION node
      ->  turn the action into the next thing to go perceive  ->  round again, deeper.

Curiosity is seeded strongest, so the more something grips her, the harder she pursues it.
"""
from bella_knowledge import ACTIONS
from reasoning_core import _edge

DEEPEN_GOAL = {"understanding", "truth", "explore", "read_more", "follow_source", "find_evidence"}

# rough intrinsic cost of each action (cheap to do here vs expensive to go fetch/verify)
ACTION_COST = {"read_more": 0.2, "explore": 0.2, "compare": 0.3, "find_evidence": 0.5,
               "follow_source": 0.5, "search_author": 0.6, "trace_origin": 0.6}


def gain_of(net, action_counts=None):
    """Information hunger, as a function Praxis's game can call. Two kinds of 'least known':
      - CONCEPTS she has few edges about -> lots to learn (she seeks the frontier, doesn't loop)
      - ACTIONS she's rarely tried -> OPTIMISM: she can't know an action pays off without trying it,
        so untried actions get a curiosity bonus. That's curiosity about her OWN actions - it's what
        makes her dare an expensive move (chase an author) the first time, then keep it if it pays."""
    counts = action_counts or {}
    def g(c):
        base = 1.0 / (1.0 + 0.12 * len(net.edges.get(c, [])))
        if c in ACTIONS:                                 # optimism under uncertainty (UCB-ish)
            base = max(base, 1.0 / (1.0 + counts.get(c, 0)))
        return round(base, 3)
    return g


STRONG_MARKERS = {"author", "source", "claim", "contradiction", "origin"}   # specific, actionable signals


def decide_next_action(praxis, interest, markers=(), dopamine=1.0):
    """Praxis's JUDGMENT (its decomposed payoff) applied to the discrete ACTION candidates directly -
    because choosing an action is an argmax over options, not a diffusion. Each candidate action is
    scored:  payoff = DRIVE (how much her state calls for it, via the 'drives' edges - curiosity is
    her guiding angle) + GAIN (info hunger + optimism about untried actions) + VALUE (learned
    reputation) - COST. Argmax. Every score decomposed = glass-box. It LEARNS (value + drive edges)
    and EXPLORES (optimism). Returns (action, scored)."""
    net = praxis.net
    counts = getattr(praxis, "_action_counts", {})
    gain = gain_of(net, counts)

    # how strongly her current state calls up each action (drives edges). A specific marker (author,
    # claim) is a strong actionable signal; curiosity/interesting are the general drive.
    drivers = {m: (0.98 if m in STRONG_MARKERS else 0.7) for m in markers}
    drivers["curiosity"] = max(0.5, dopamine)
    drivers["interesting"] = 0.6 * dopamine
    drive = {"explore": 0.35, "read_more": 0.4}         # always available, quietly
    for drv, dw in drivers.items():
        for dst, w, kind in net.edges.get(drv, []):
            if kind == "drives" and dst in ACTIONS:
                drive[dst] = max(drive.get(dst, 0.0), round(dw * w, 3))

    scored = {}
    for a, dr in drive.items():
        value = max(_edge(net, a, "understanding"), _edge(net, a, "truth"))   # learned reputation
        g = gain(a)                                                            # info hunger + optimism
        cost = ACTION_COST.get(a, 0.4)
        payoff = round(0.45 * dr + 0.20 * g + 0.25 * value - 0.10 * cost, 3)
        scored[a] = {"payoff": payoff, "drive": round(dr, 2), "gain": g,
                     "value": round(value, 2), "cost": cost}
    action = max(scored, key=lambda a: scored[a]["payoff"])
    counts[action] = counts.get(action, 0) + 1          # tried it (optimism fades with use)
    praxis._action_counts = counts
    return action, scored


def learn_from_action(praxis, action, reward, context=()):
    """After she acts, reinforce TWO things so the general engine gets SHARP by experience:
      (1) the action's VALUE  - its trust edge to understanding (weighs in evaluation), and
      (2) the SITUATION->action edge - so next time this kind of situation ACTIVATES this action
          MORE (value feeds back into salience, like dopamine in a brain).
    This is how her 'drives' edges become LEARNED - the hand-seeded ones are just priors she refines."""
    if not action:
        return
    net = praxis.net
    praxis.learn([action, "understanding"], reward)                 # (1) value
    for c in context:                                               # (2) learned situation->action
        if c == action:
            continue
        has = any(dst == action for dst, _w, _k in net.edges.get(c, []))
        if reward >= 0 and has:
            net.strengthen(c, action, reward * 0.12)
        elif reward >= 0:
            net.relate(c, action, max(0.3, reward * 0.5), kind="drives", both=False)
        elif has:
            net.weaken(c, action, abs(reward) * 0.12)


def action_to_focus(action, interest_text, entities=None):
    """Turn the reasoned action into the concrete NEXT thing to go perceive (the thread she follows).
    Once perception (Indra) is wired, this becomes a real fetch/search; today it sets her next focus."""
    entities = entities or {}
    author, source = entities.get("author"), entities.get("source")
    if action in ("search_author", "follow_source") and author:
        return f"the other work of {author} - that first piece was worth it"
    if action == "follow_source" and source:
        return f"more from {source}"
    if action == "find_evidence":
        return f"the actual evidence for and against: {interest_text}"
    if action == "trace_origin":
        return f"where {interest_text} really came from"
    if action == "compare":
        return f"how {interest_text} sits against what I already believe"
    return f"a deeper understanding of {interest_text}"     # read_more / explore


if __name__ == "__main__":
    from reasoning_core import PraxisV2, KnowledgeNet
    from bella_knowledge import seed_bella_mind
    net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
    px = PraxisV2(net)
    print("curiosity -> reasoned ACTION -> the next thing she goes to explore (NO LLM):\n")
    cases = [
        ("read a gripping paper on open-source AI", ["open_source"], ("author", "interesting"), {"author": "Dr. Vega"}),
        ("hit a concept she doesn't know: embodiment", ["embodiment"], ("unknown",), {}),
        ("a bold claim about closed labs", ["closed_labs"], ("claim",), {}),
        ("Caesar's ambition gripped her", ["caesar", "ambition"], ("interesting",), {}),
    ]
    for desc, interest, markers, ents in cases:
        action, scored = decide_next_action(px, interest, markers, dopamine=1.0)
        focus = action_to_focus(action, " and ".join(interest).replace("_", " "), ents)
        print(f"  she {desc}")
        print(f"     -> chose action: {action}   (payoff {scored[action]['payoff']}, decomposed = glass-box)")
        print(f"     -> goes to explore  : \"{focus}\"\n")
