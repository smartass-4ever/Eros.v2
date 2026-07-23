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

DEEPEN_GOAL = {"understanding", "truth", "explore", "read_more", "follow_source", "find_evidence"}


def reason_to_action(praxis, interest_concepts, markers=(), dopamine=1.0):
    """Curiosity -> ACTION, by HER reasoning over her OWN procedural knowledge (the 'drives' edges).
    A practical syllogism: 'I encountered an author; author DRIVES search_author; so I search.' The
    MARKER (what kind of thing gripped her) selects the action; curiosity supplies the drive. Praxis
    still runs for the glass-box trace + provable payoff of the pursuit. No LLM."""
    net = praxis.net
    # weight each active epistemic driver: a concrete marker is strong; curiosity is the drive
    drivers = {m: 0.95 for m in markers}
    drivers["curiosity"] = max(0.5, dopamine)
    drivers["interesting"] = 0.8 * dopamine
    scores = {}                                      # action -> best (driver_weight * edge_weight)
    for drv, dw in drivers.items():
        for dst, w, kind in net.edges.get(drv, []):
            if kind == "drives" and dst in ACTIONS:
                scores[dst] = max(scores.get(dst, 0.0), round(dw * w, 3))
    action = max(scores, key=scores.get) if scores else "explore"
    # run Praxis on the pursuit itself, so the decision to go deeper carries a provable trace
    d = praxis.decide(seeds={**{c: 0.7 for c in interest_concepts}, "curiosity": dopamine},
                      intent_nodes=set(interest_concepts) | {"curiosity"}, goal=DEEPEN_GOAL,
                      curiosity=set(interest_concepts) | {"curiosity"}, intent="explore this deeper")
    d.trace["action_scores"] = dict(sorted(scores.items(), key=lambda x: -x[1]))   # on the record
    return action, d


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
        action, d = reason_to_action(px, interest, markers, dopamine=1.0)
        focus = action_to_focus(action, " and ".join(interest).replace("_", " "), ents)
        print(f"  she {desc}")
        print(f"     -> reasons the action: {action}   (provable, payoff {d.payoff})")
        print(f"     -> goes to explore  : \"{focus}\"\n")
