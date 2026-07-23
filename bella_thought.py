"""
BELLA'S THOUGHT-FORMER - she forms the COMPLETE thought; the LLM only translates it.

The claim is fully determined by HER systems before the LLM ever sees it:
  - Praxis picks the concepts, the winning path, the provable payoff
  - the knowledge graph carries TYPED relations (is / leads_to / concentrates / opposes /
    analogous_to / becomes / exemplifies / requires ...) - so a path has MEANING, not just adjacency
  - a valence lexicon sets her stance (does this path run through good things or bad?)

From those, a deterministic builder assembles a full proposition IN WORDS - no LLM. The LLM's ONLY
job (in bella._translate) is to make that finished sentence more natural, adding NO new claim. Turn
the LLM off and she still says the whole thought, just plainer. THAT is 'LLM = mouth, not mind'.
"""

ROMAN = {"rome", "caesar", "republic", "empire", "rubicon", "augustus", "brutus", "senate",
         "stoicism", "marcus_aurelius", "seneca", "aeneid", "virgil", "ovid", "metamorphoses",
         "dictator", "gaul", "point_of_no_return", "civil_war", "legacy"}

# concepts that carry a charge - her stance comes from which ones the path runs through
NEG = {"corruption", "secrecy", "betrayal", "concentration", "empire", "dictator", "control",
       "conflict", "upheaval", "risk", "broken_trust", "assassination", "civil_war", "competition",
       # modern world
       "hype", "bubble", "crash", "brain_rot", "parasocial", "churn", "burn_rate", "misalignment",
       "overvaluation", "failure", "fomo", "job_change",
       # the bedrock
       "death", "pain", "harm", "loss", "error", "suffering", "weakness", "decay", "isolation",
       "loneliness", "deception", "fear"}
POS = {"collaboration", "virtue", "truth", "republic", "distribution", "trust", "cooperation",
       "wisdom", "evidence", "community", "peace", "honesty", "progress", "discovery", "duty",
       # modern world
       "authenticity", "product_market_fit", "alignment", "ai_safety", "growth", "retention",
       "moat", "defensibility", "runway", "survival",
       # the bedrock
       "health", "strength", "stability", "skill", "mastery", "clarity", "balance", "improvement",
       "efficiency", "cooperation", "acceptance", "understanding"}

# each RELATION TYPE renders to a claim shape - this is where a typed path becomes a sentence
TEMPLATES = {
    "is":          "{a} is, at bottom, {b}",
    "is_a":        "{a} is a kind of {b}",
    "leads_to":    "{a} leads to {b}",
    "causes":      "{a} brings about {b}",
    "concentrates":"{a} pulls {b} into fewer and fewer hands",
    "distributes": "{a} spreads {b} across many hands",
    "opposes":     "{a} is pulling against {b}",
    "analogous_to":"{a} is the {b} of our moment - the same pattern, playing out again",
    "becomes":     "{a} hardens into {b}",
    "exemplifies": "{a} is {b} made flesh",
    "requires":    "{a} rests on {b}",
    "relates_to":  "{a} and {b} turn out to be two faces of one thing",
}


def _nice(c):
    return str(c).replace("_", " ")


def _typed_edge(net, a, b):
    """Find the MEANING-bearing edge between the winning pair, in either direction, and orient it:
    returns (relation_type, subject, object). Falls back to 'relates_to'."""
    for dst, _w, kind in (net.edges.get(a, []) if net else []):
        if dst == b and kind in TEMPLATES:
            return kind, a, b
    for dst, _w, kind in (net.edges.get(b, []) if net else []):
        if dst == a and kind in TEMPLATES:
            return kind, b, a
    return "relates_to", a, b


def form_structured(decision, net) -> dict:
    """The DEEP STRUCTURE of her thought - the claim itself, fully hers, no LLM. subject + typed
    relation + object + stance. The LLM (surface realization) can add nothing to this."""
    pair = [c for c in decision.concepts if c] or ["the_world"]
    a = pair[0]; b = pair[1] if len(pair) > 1 else pair[0]
    relation, subj, obj = _typed_edge(net, a, b)
    # stance comes from the charge of the two concepts the CLAIM is about (not the whole path)
    charge = (1 if subj in POS else -1 if subj in NEG else 0) + (1 if obj in POS else -1 if obj in NEG else 0)
    stance = "cautionary" if charge < 0 else "affirming" if charge > 0 else "neutral"
    return {"subject": subj, "relation": relation, "object": obj, "stance": stance,
            "roman": subj in ROMAN or obj in ROMAN}


def form_thought(decision, net) -> str:
    """Her complete thought in plain words, NO LLM - the surface gloss of the structured claim.
    (bella._translate can make this smoother, but adds no claim - the meaning is fixed here.)"""
    s = form_structured(decision, net)
    core = TEMPLATES.get(s["relation"], TEMPLATES["relates_to"]).format(
        a=_nice(s["subject"]), b=_nice(s["object"]))
    if s["stance"] == "cautionary":
        core += ", and that rarely ends the way it promises"
    elif s["stance"] == "affirming":
        core += ", and that's where the good comes from"
    return core


if __name__ == "__main__":
    from reasoning_core import PraxisV2, KnowledgeNet
    from bella_knowledge import seed_bella_mind
    net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
    core = PraxisV2(net)
    print("her COMPLETE thoughts, formed with NO LLM (the LLM would only make these smoother):\n")
    for intent, seeds in [("open vs closed AI", {"open_source": 1.0, "closed_labs": 0.8}),
                          ("Stoics on power", {"stoicism": 1.0, "power": 0.7}),
                          ("Caesar's ambition", {"caesar": 1.0, "ambition": 0.8}),
                          ("robots learning", {"robots": 1.0, "embodiment": 0.7})]:
        d = core.decide(seeds=seeds, intent_nodes=set(seeds), goal={"truth", "evidence", "help"},
                        curiosity=set(seeds), intent=intent)
        print(f"  [{intent}]  {form_thought(d, net)}")
