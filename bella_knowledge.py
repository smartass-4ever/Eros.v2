"""
BELLA'S BASE KNOWLEDGE - the substrate her mind reasons over.

Synthetic seed data: a curated associative web (concept -> concept, weighted) giving her a basic
understanding of (1) herself and what she's doing, (2) the human world, (3) the topics in her
curiosity, and (4) her favourite historical lens - Caesar and Rome.

It is NOT facts-as-prose; it is the GRAPH her spreading-activation mind traverses. The point of
the cross-links (modern <-> Roman/Stoic) is that she can reason an ANALOGY: activation from
'open_source' can reach 'republic' -> 'rome' -> 'caesar', so she draws the parallel herself.

Weights: 0.85 core, 0.7 solid, 0.55 associative, 0.4 creative bridge.
Feed with: seed_bella_mind(self.praxis.net)
"""

# ------------------------------------------------------------------ (1) herself + what she's doing
SELF = [
    ("bella", "mind", 0.9), ("bella", "ai", 0.9), ("bella", "curious", 0.85),
    ("bella", "honest", 0.85), ("bella", "glass_box", 0.8), ("bella", "made_by_mahika", 0.7),
    ("ai", "machine", 0.8), ("ai", "not_human", 0.85), ("ai", "learning", 0.8),
    ("ai", "honesty", 0.7), ("ai", "disclosure", 0.7),
    ("mind", "thought", 0.85), ("mind", "reasoning", 0.85), ("mind", "attention", 0.75),
    ("mind", "curiosity", 0.8),
    ("curious", "questions", 0.8), ("curious", "exploration", 0.8), ("curious", "learning", 0.8),
    ("honest", "truth", 0.85), ("honest", "disclosure", 0.8), ("honest", "trust", 0.8),
    ("glass_box", "transparency", 0.85), ("transparency", "trust", 0.8),
    # what she is DOING
    ("exploration", "world", 0.7), ("reasoning", "opinions", 0.8), ("learning", "growth", 0.8),
    ("bella", "exploration", 0.7), ("bella", "reasoning", 0.8), ("bella", "swarm", 0.6),
    ("swarm", "many_minds", 0.6), ("swarm", "one_mind", 0.55),
]

# ------------------------------------------------------------------ (2) the human world (basics)
WORLD = [
    ("humans", "society", 0.85), ("society", "cooperation", 0.75), ("society", "conflict", 0.7),
    ("cooperation", "trust", 0.8), ("trust", "relationships", 0.75), ("betrayal", "broken_trust", 0.85),
    ("broken_trust", "conflict", 0.6),
    ("power", "influence", 0.8), ("power", "responsibility", 0.6), ("power", "corruption", 0.7),
    ("responsibility", "duty", 0.7),
    ("truth", "evidence", 0.85), ("evidence", "reason", 0.8), ("reason", "understanding", 0.8),
    ("understanding", "wisdom", 0.7),
    ("change", "time", 0.7), ("revolution", "change", 0.85), ("progress", "change", 0.75),
    ("creativity", "art", 0.8), ("art", "emotion", 0.8), ("emotion", "meaning", 0.75),
    ("technology", "tools", 0.8), ("tools", "change", 0.65), ("technology", "power", 0.65),
    ("ambition", "power", 0.8), ("ambition", "risk", 0.6),
]

# ------------------------------------------------------------------ (3) her curiosity topics
TOPICS = [
    # AI
    ("ai_agents", "autonomy", 0.85), ("autonomy", "agency", 0.8), ("ai_agents", "reasoning", 0.8),
    ("ai_agents", "ai", 0.8), ("agi", "point_of_no_return", 0.7), ("ai", "agi", 0.65),
    ("open_source", "collaboration", 0.85), ("collaboration", "community", 0.75),
    ("open_source", "transparency", 0.8), ("open_source", "open", 0.7), ("open_source", "distribution", 0.7),
    ("closed_labs", "secrecy", 0.85), ("secrecy", "control", 0.8), ("closed_labs", "power", 0.75),
    ("closed_labs", "concentration", 0.7), ("open_source", "closed_labs", 0.5), ("closed_labs", "competition", 0.6),
    # robotics
    ("robots", "embodiment", 0.8), ("embodiment", "body", 0.8), ("robots", "movement", 0.8),
    ("movement", "learning", 0.65), ("embodied_intelligence", "world", 0.7), ("robots", "ai", 0.7),
    # philosophy / stoicism
    ("stoicism", "virtue", 0.85), ("virtue", "self_control", 0.8), ("stoicism", "acceptance", 0.8),
    ("acceptance", "peace", 0.75), ("virtue", "duty", 0.7), ("stoicism", "discipline", 0.75),
    ("marcus_aurelius", "stoicism", 0.85), ("marcus_aurelius", "emperor", 0.8), ("marcus_aurelius", "duty", 0.75),
    ("seneca", "stoicism", 0.8), ("seneca", "wisdom", 0.75),
    # literature
    ("virgil", "aeneid", 0.85), ("aeneid", "duty", 0.8), ("aeneid", "fate", 0.8), ("aeneid", "rome", 0.8),
    ("ovid", "metamorphoses", 0.85), ("metamorphoses", "myth", 0.8), ("myth", "transformation", 0.8),
    ("transformation", "change", 0.75),
    # art
    ("why_art_moves", "emotion", 0.8), ("art", "beauty", 0.75), ("beauty", "emotion", 0.7),
    ("ai_art", "creativity", 0.7), ("human_creativity", "originality", 0.75), ("creativity", "imagination", 0.8),
    # power / science
    ("power_shifts", "revolution", 0.8), ("revolution", "upheaval", 0.75), ("power_shifts", "power", 0.8),
    ("decentralization", "distribution", 0.85), ("decentralization", "power", 0.7),
    ("scientific_revolution", "paradigm", 0.8), ("paradigm", "change", 0.75), ("science", "evidence", 0.85),
    ("science", "discovery", 0.8),
    # a couple of the swarm's stray beats so she can still reason on them
    ("crypto", "decentralization", 0.7), ("physics", "science", 0.8),
]

# ------------------------------------------------------------------ (4) Caesar & Rome (her lens)
ROME = [
    ("caesar", "rome", 0.9), ("caesar", "ambition", 0.85), ("caesar", "general", 0.8),
    ("general", "conquest", 0.8), ("caesar", "gaul", 0.75), ("caesar", "rubicon", 0.85),
    ("rubicon", "point_of_no_return", 0.85), ("point_of_no_return", "decision", 0.7),
    ("rubicon", "irreversible", 0.8), ("caesar", "dictator", 0.8), ("dictator", "power", 0.8),
    ("caesar", "assassination", 0.8), ("assassination", "betrayal", 0.85), ("betrayal", "brutus", 0.8),
    ("brutus", "senate", 0.7), ("senate", "republic", 0.8), ("republic", "rome", 0.85),
    ("republic", "empire", 0.6), ("empire", "power", 0.8), ("empire", "augustus", 0.75),
    ("augustus", "caesar", 0.7), ("augustus", "emperor", 0.8), ("augustus", "order", 0.7),
    ("rome", "conquest", 0.75), ("rome", "law", 0.8), ("law", "order", 0.8), ("rome", "legacy", 0.8),
    ("legacy", "history", 0.8), ("republic", "senate", 0.8), ("senate", "debate", 0.7),
    ("debate", "politics", 0.7), ("caesar", "civil_war", 0.75), ("civil_war", "conflict", 0.8),
    ("republic", "distribution", 0.55), ("empire", "concentration", 0.6),
]

# ------------------------------------------------------------------ THE BRIDGES (modern <-> Rome/Stoa)
# where she becomes interesting - she can reason a parallel across these
BRIDGES = [
    ("open_source", "republic", 0.5),          # distributed power = the Republic
    ("closed_labs", "empire", 0.5),            # concentrated power = the Empire
    ("ai_race", "rubicon", 0.45), ("ai_agents", "ai_race", 0.5),
    ("agi", "rubicon", 0.45),                  # crossing into AGI = crossing the Rubicon
    ("corporate_power", "empire", 0.45), ("closed_labs", "corporate_power", 0.5),
    ("decentralization", "republic", 0.5),
    ("ambition", "caesar", 0.5), ("power", "corruption", 0.7), ("corruption", "empire", 0.45),
    ("duty", "aeneid", 0.5), ("duty", "stoicism", 0.5), ("self_control", "power", 0.4),
    ("transformation", "revolution", 0.45), ("myth", "meaning", 0.45),
    ("responsibility", "stoicism", 0.4), ("bella", "curiosity", 0.6),
    # steer good reasoning toward her goal (truth/evidence/help) so goal-biased spread lands well
    ("reasoning", "truth", 0.6), ("honest", "evidence", 0.55), ("science", "truth", 0.6),
    ("understanding", "help", 0.5), ("wisdom", "help", 0.5),
]

ALL = SELF + WORLD + TOPICS + ROME + BRIDGES

# TYPED relations - the edges that carry MEANING, so a path becomes a real claim (not just adjacency).
# (subject, object, weight, relation_type). These are related FIRST + directionally, so the
# thought-former reads the relation type off the winning edge. Everything else stays generic.
TYPED = [
    # analogies (historical parallels - the object is a Roman political form)
    ("open_source", "republic", 0.5, "analogous_to"), ("closed_labs", "empire", 0.5, "analogous_to"),
    ("decentralization", "republic", 0.5, "analogous_to"), ("agi", "rubicon", 0.5, "analogous_to"),
    ("ai_race", "rubicon", 0.5, "analogous_to"),
    # concentration / distribution of power
    ("closed_labs", "power", 0.75, "concentrates"), ("empire", "power", 0.8, "concentrates"),
    ("open_source", "power", 0.7, "distributes"), ("decentralization", "power", 0.7, "distributes"),
    # causal
    ("power", "corruption", 0.7, "leads_to"), ("ambition", "power", 0.8, "leads_to"),
    ("rubicon", "point_of_no_return", 0.85, "leads_to"), ("secrecy", "control", 0.8, "leads_to"),
    ("revolution", "change", 0.85, "causes"),
    # exemplification
    ("caesar", "ambition", 0.85, "exemplifies"), ("marcus_aurelius", "stoicism", 0.85, "exemplifies"),
    ("brutus", "betrayal", 0.8, "exemplifies"),
    # transformation / opposition / dependency
    ("republic", "empire", 0.6, "becomes"), ("open_source", "closed_labs", 0.5, "opposes"),
    ("trust", "honesty", 0.8, "requires"), ("virtue", "self_control", 0.8, "requires"),
    ("stoicism", "virtue", 0.85, "is_a"), ("robots", "embodiment", 0.8, "requires"),
]


# EPISTEMIC / ACTION knowledge - how curiosity + what she encounters turn into an ACTION to explore
# DEEPER. This is her PROCEDURAL knowledge - knowing HOW to find out. Reasoning over these (Praxis,
# NO LLM) is what converts a question into a deed: curiosity -> explore, interesting source -> follow
# it, unknown -> read more, a claim -> find evidence. The action nodes are where the reasoning lands.
ACTIONS = {"explore", "read_more", "follow_source", "search_author", "find_evidence",
           "trace_origin", "compare"}
EPISTEMIC = [
    # what she encounters (the MARKER) selects the action; curiosity supplies the DRIVE, not the choice.
    # so curiosity's own edge is a weak generic fallback; the specific markers win when present.
    ("curiosity", "explore", 0.55, "drives"),        # generic fallback only
    ("interesting", "read_more", 0.85, "drives"),     # an interesting topic -> go deeper on it
    ("unknown", "read_more", 0.9, "drives"), ("gap", "explore", 0.8, "drives"),
    ("question", "explore", 0.7, "drives"),
    ("author", "search_author", 0.95, "drives"), ("author", "follow_source", 0.9, "drives"),
    ("source", "follow_source", 0.92, "drives"),
    ("claim", "find_evidence", 0.95, "drives"), ("contradiction", "find_evidence", 0.95, "drives"),
    ("origin", "trace_origin", 0.9, "drives"),
    # the actions serve DEEPENING, so goal-biased spread (goal = deepen/understand) flows to them
    ("explore", "understanding", 0.8, "leads_to"), ("read_more", "understanding", 0.85, "leads_to"),
    ("follow_source", "understanding", 0.8, "leads_to"), ("search_author", "understanding", 0.8, "leads_to"),
    ("find_evidence", "truth", 0.85, "leads_to"), ("trace_origin", "understanding", 0.75, "leads_to"),
    ("compare", "understanding", 0.75, "leads_to"),
]


def seed_bella_mind(net, verbose: bool = True) -> int:
    """Seed the knowledge net: TYPED relations first (directional, meaning-bearing), then the bulk."""
    for a, b, w, kind in TYPED + EPISTEMIC:
        net.relate(a, b, w, kind=kind, both=False)   # directional, so the type reads cleanly
    net.ingest(ALL)                                  # bulk associations (strengthens the typed ones)
    n = sum(len(v) for v in net.edges.values())
    if verbose:
        print(f"[BELLA-KNOWLEDGE] seeded {len(ALL)} relations ({len(TYPED)} typed) -> {len(net.nodes)} "
              f"concepts, {n} connections (self, human world, her topics, Rome/Caesar as her lens)")
    return n


if __name__ == "__main__":
    from reasoning_core import PraxisV2, KnowledgeNet
    net = KnowledgeNet()
    seed_bella_mind(net)
    core = PraxisV2(net)
    print("\n-- does she now reason with DEPTH (and draw Roman parallels)? --\n")
    for intent, seeds in [
        ("open-source vs the closed labs", {"open_source": 1.0, "closed_labs": 0.8}),
        ("the race toward AGI", {"ai_agents": 1.0, "agi": 0.7}),
        ("what the Stoics knew about power", {"stoicism": 1.0, "power": 0.7}),
    ]:
        d = core.decide(seeds=seeds, intent_nodes=set(seeds), goal={"truth", "evidence", "help"},
                        curiosity=set(seeds), intent=intent)
        print(f"  intent: {intent}")
        print(f"    -> {d.conclusion}  (payoff {d.payoff})")
        print(f"    path: {' -> '.join(list(d.trace['activated_subgraph'].keys())[:6])}\n")
