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


def gain_of(net):
    """Information hunger as a function Praxis's game can call: she expects to learn MOST where she
    knows LEAST (few edges -> high). This is what makes curiosity SEEK, not loop."""
    return lambda c: round(1.0 / (1.0 + 0.12 * len(net.edges.get(c, []))), 3)


def decide_next_action(praxis, interest, markers=(), dopamine=1.0):
    """NO separate scorer. This just SETS UP the action decision and lets PRAXIS'S OWN GAME choose:
    actions are nodes, reached via the 'drives' edges; the game now scores them with gain+cost folded
    in (curiosity highest, trust = learned action-value, gain = information hunger, cost = effort).
    The winning candidate's ACTION node is her move. Fully glass-box - the payoff decomposition is
    in d.trace. Returns (action, decision)."""
    net = praxis.net
    seeds = {"curiosity": max(0.5, dopamine), "interesting": 0.8 * dopamine}
    for c in interest:
        seeds[c] = 0.7
    for m in markers:                                # what KIND of thing gripped her (from perception)
        seeds[m] = 0.95
    d = praxis.decide(
        seeds=seeds, intent_nodes=set(interest) | set(markers) | {"curiosity"},
        goal=DEEPEN_GOAL | ACTIONS,                  # actions are legitimate destinations of reasoning
        curiosity=set(interest) | {"curiosity"}, intent="what should I do to explore this deeper?",
        gain_fn=gain_of(net), cost_map=ACTION_COST)
    action = next((c for c in d.concepts if c in ACTIONS), None)   # the action the game landed on
    if not action:                                    # else the top-activated action in her reasoning
        action = next((c for c in d.trace.get("activated_subgraph", {}) if c in ACTIONS), "explore")
    return action, d


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
        action, d = decide_next_action(px, interest, markers, dopamine=1.0)
        focus = action_to_focus(action, " and ".join(interest).replace("_", " "), ents)
        print(f"  she {desc}")
        print(f"     -> Praxis's game chose the action: {action}   (payoff {d.payoff})")
        print(f"     -> goes to explore  : \"{focus}\"\n")
