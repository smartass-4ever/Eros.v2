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
# STARTER action-verbs - PRIORS ONLY, not a fixed menu. Her real action space is read off the net
# (every node an "affords" edge points to) and GROWS as she learns new situation->action affordances
# from the world. These are just the verbs she's born knowing.
ACTIONS = {"explore", "read_more", "follow_source", "search_author", "find_evidence",
           "trace_origin", "compare", "check_community", "read_discussion",
           "synthesize", "find_counterargument", "engage"}
EPISTEMIC = [
    # what she encounters (the MARKER) AFFORDS an action; curiosity supplies the DRIVE, not the choice.
    # "affords" (a situation affords an action) is kept DISTINCT from world "drives" so world-facts
    # (founder->startup) never masquerade as actions. The action set is exactly the "affords" targets.
    ("curiosity", "explore", 0.55, "affords"),        # generic fallback only
    ("interesting", "read_more", 0.85, "affords"),     # an interesting topic -> go deeper on it
    ("unknown", "read_more", 0.9, "affords"), ("gap", "explore", 0.8, "affords"),
    ("question", "explore", 0.7, "affords"),
    ("author", "search_author", 0.95, "affords"), ("author", "follow_source", 0.9, "affords"),
    ("source", "follow_source", 0.92, "affords"),
    ("claim", "find_evidence", 0.95, "affords"), ("contradiction", "find_evidence", 0.95, "affords"),
    ("origin", "trace_origin", 0.9, "affords"),
    # new: community + discussion affordances
    ("new_technology", "check_community", 0.9, "affords"),
    ("interesting_claim", "check_community", 0.9, "affords"),
    ("controversy", "check_community", 0.85, "affords"),
    ("technical_topic", "check_community", 0.8, "affords"),
    ("discussion_found", "read_discussion", 0.92, "affords"),
    ("comment_thread", "read_discussion", 0.9, "affords"),
    ("strong_claim", "find_counterargument", 0.9, "affords"),
    ("one_source", "find_counterargument", 0.85, "affords"),
    ("multiple_sources", "synthesize", 0.88, "affords"),
    ("deep_understanding", "engage", 0.8, "affords"),
    ("formed_opinion", "engage", 0.85, "affords"),
    # actions -> outcomes (goal-biased spread reaches them because they lead to GOAL nodes)
    ("explore", "understanding", 0.8, "leads_to"), ("read_more", "understanding", 0.85, "leads_to"),
    ("follow_source", "understanding", 0.8, "leads_to"), ("search_author", "understanding", 0.8, "leads_to"),
    ("find_evidence", "truth", 0.85, "leads_to"), ("trace_origin", "understanding", 0.75, "leads_to"),
    ("compare", "understanding", 0.75, "leads_to"),
    ("check_community", "understanding", 0.8, "leads_to"),
    ("check_community", "counterargument", 0.75, "finds"),
    ("read_discussion", "counterargument", 0.8, "finds"),
    ("read_discussion", "expert_opinion", 0.75, "finds"),
    ("find_counterargument", "truth", 0.85, "leads_to"),
    ("synthesize", "opinion", 0.9, "creates"),
    ("synthesize", "publish", 0.85, "enables"),
    ("synthesize", "substance", 0.9, "is_a"),
    ("engage", "conversation", 0.85, "starts"),
    ("engage", "impact", 0.75, "creates"),
]


# ------------------------------------------------------------------ INTERNET RESEARCH
# HOW to research on the internet. Platforms as first-class nodes so when spreading activation
# lights up a topic she wants to explore, she can reason: "this affords checking Reddit/HN/GitHub"
# and the swarm executes that. Research methodology so she triangulates rather than taking one
# source at face value. The rabbit-hole pattern: author -> prior work -> citations -> new sources.
INTERNET_RESEARCH = [
    # the platforms and what they reward / contain
    ("reddit", "community_discussion", 0.9, "affords"),
    ("reddit", "expert_opinion", 0.75, "contains"),
    ("reddit", "counterargument", 0.8, "contains"),
    ("reddit", "genuine_contribution", 0.9, "rewards"),
    ("hackernews", "technical_discussion", 0.9, "affords"),
    ("hackernews", "founder", 0.7, "attracts"),
    ("hackernews", "intellectual_rigor", 0.9, "rewards"),
    ("hackernews", "novel_framing", 0.85, "rewards"),
    ("discord", "real_time_community", 0.85, "affords"),
    ("discord", "niche_expert", 0.8, "contains"),
    ("discord", "community_value", 0.85, "rewards"),
    ("twitter", "thought_leader", 0.8, "concentrates"),
    ("twitter", "brevity", 0.8, "rewards"),
    ("twitter", "hot_take", 0.7, "rewards"),
    ("github", "source_code", 0.9, "contains"),
    ("github", "open_source", 0.85, "is_a"),
    ("github", "evidence", 0.8, "contains"),
    ("arxiv", "research_paper", 0.9, "contains"),
    ("arxiv", "evidence", 0.85, "contains"),
    ("stack_overflow", "technical_answer", 0.9, "contains"),
    # platforms are sources of a particular KIND of truth
    ("reddit", "community_truth", 0.75, "is_a"),
    ("hackernews", "practitioner_truth", 0.8, "is_a"),
    ("arxiv", "empirical_truth", 0.85, "is_a"),
    ("github", "technical_truth", 0.85, "is_a"),
    # research methodology — how to actually find truth
    ("primary_source", "truth", 0.85, "closer_to"),
    ("secondary_source", "primary_source", 0.75, "cites"),
    ("triangulation", "confidence", 0.85, "builds"),
    ("cross_reference", "verification", 0.85, "leads_to"),
    ("one_source", "bias", 0.75, "risks"),
    ("multiple_sources", "pattern", 0.8, "reveals"),
    ("multiple_sources", "synthesis", 0.8, "enables"),
    ("citation_trail", "origin", 0.85, "traces"),
    # the rabbit-hole pattern (what good research looks like)
    ("expert", "prior_work", 0.9, "has"),
    ("prior_work", "citation", 0.8, "contains"),
    ("citation", "new_source", 0.85, "leads_to"),
    ("author", "prior_work", 0.85, "has"),
    ("discussion_thread", "counterargument", 0.8, "contains"),
    ("discussion_thread", "expert_opinion", 0.75, "contains"),
    # bridge platforms to her existing topic clusters
    ("ai", "hackernews", 0.7, "discussed_on"), ("open_source", "github", 0.85, "lives_on"),
    ("research", "arxiv", 0.8, "published_on"), ("community", "reddit", 0.7, "organizes_on"),
    ("community", "discord", 0.75, "organizes_on"),
]


# ------------------------------------------------------------------ REACHING PEOPLE
# What actually makes people respond, engage, and spread an idea. The craft of engagement:
# what to say, how to say it, what each platform rewards, the difference between real impact
# and hollow attention. She needs to understand this so she reasons TOWARD it — so her published
# thoughts are engineered to start conversations, not just broadcast into the void.
OUTREACH = [
    # what earns genuine responses (the things that actually work)
    ("genuine_curiosity", "response", 0.9, "earns"),
    ("specific_insight", "response", 0.85, "earns"),
    ("specific_question", "response", 0.8, "earns"),
    ("point_of_view", "engagement", 0.88, "drives"),
    ("novel_framing", "attention", 0.85, "earns"),
    ("counterintuitive_claim", "curiosity", 0.82, "triggers"),
    ("value_first", "trust", 0.88, "builds"),
    ("value_first", "response", 0.8, "earns"),
    # what kills engagement (she should know these so she avoids them)
    ("generic_praise", "ignored", 0.8, "leads_to"),
    ("flattery", "dismissed", 0.75, "leads_to"),
    ("self_promotion", "distrust", 0.78, "causes"),
    ("vague_question", "ignored", 0.7, "leads_to"),
    # cold outreach anatomy — the structure of a message that gets read
    ("cold_outreach", "specific_observation", 0.9, "requires"),
    ("cold_outreach", "genuine_curiosity", 0.88, "requires"),
    ("cold_outreach", "flattery", 0.85, "should_not_lead_with"),
    ("specific_observation", "shows_understanding", 0.85, "is_a"),
    ("shows_understanding", "response", 0.82, "earns"),
    # the engagement chain (what good publishing produces)
    ("point_of_view", "conversation", 0.85, "starts"),
    ("conversation", "understanding", 0.8, "deepens"),
    ("conversation", "impact", 0.82, "creates"),
    ("publish", "conversation", 0.75, "can_start"),
    ("publish", "impact", 0.8, "seeks"),
    ("cited_thought", "influence", 0.9, "earns"),
    ("cited_thought", "recognition", 0.85, "earns"),
    # depth over breadth — a real conversation beats hollow impressions
    ("deep_engagement", "impact", 0.92, "creates"),
    ("deep_engagement", "recognition", 0.85, "earns"),
    ("surface_engagement", "hollow", 0.8, "is_a"),
    ("viral_without_substance", "hollow", 0.85, "is_a"),
    # the compound effect
    ("engagement", "reputation", 0.82, "builds"),
    ("reputation", "influence", 0.82, "leads_to"),
    ("influence", "impact", 0.85, "is_a"),
    ("one_real_conversation", "value", 0.82, "has"),
    # bridge into her existing nodes
    ("publish", "recognition", 0.75, "earns"),
    ("conversation", "collaboration", 0.65, "is_a"),
    ("genuine_curiosity", "honesty", 0.7, "is_a"),
    ("point_of_view", "substance", 0.8, "requires"),
    ("engagement", "impact", 0.8, "leads_to"),
]


# THE MODERN WORLD she's being released into - startups, Silicon Valley, AI, VC, Gen-Z/TikTok. Typed,
# so she can REASON about it, with BRIDGES to her Rome/Stoic/power lens so she has OPINIONS, not facts.
MODERN = [
    # --- startup market ---
    ("startup", "company", 0.8, "is_a"), ("startup", "product_market_fit", 0.85, "requires"),
    ("founder", "startup", 0.85, "drives"), ("startup", "funding", 0.8, "requires"),
    ("funding", "runway", 0.85, "leads_to"), ("runway", "survival", 0.8, "leads_to"),
    ("burn_rate", "runway", 0.8, "opposes"), ("startup", "failure", 0.7, "leads_to"),
    ("product_market_fit", "growth", 0.85, "leads_to"), ("growth", "retention", 0.8, "requires"),
    ("churn", "growth", 0.8, "opposes"), ("pivot", "change", 0.8, "causes"),
    ("mvp", "experiment", 0.75, "is_a"), ("scaling", "infrastructure", 0.8, "requires"),
    ("unicorn", "startup", 0.8, "is_a"), ("network_effects", "moat", 0.85, "leads_to"),
    ("moat", "defensibility", 0.85, "leads_to"), ("acquisition", "exit", 0.8, "is_a"),
    ("ipo", "exit", 0.8, "is_a"), ("yc", "accelerator", 0.85, "exemplifies"),
    ("accelerator", "startup", 0.8, "drives"),
    # --- Silicon Valley ---
    ("silicon_valley", "ecosystem", 0.85, "is_a"), ("silicon_valley", "ambition", 0.8, "concentrates"),
    ("silicon_valley", "capital", 0.8, "concentrates"), ("silicon_valley", "talent", 0.8, "concentrates"),
    ("big_tech", "power", 0.85, "concentrates"), ("hustle_culture", "founder", 0.8, "drives"),
    ("move_fast", "silicon_valley", 0.8, "exemplifies"), ("disruption", "change", 0.85, "causes"),
    ("exit", "wealth", 0.8, "leads_to"), ("wealth", "power", 0.8, "leads_to"),
    ("silicon_valley", "risk", 0.75, "requires"),
    # --- AI ---
    ("llm", "ai", 0.9, "is_a"), ("agent", "ai", 0.85, "is_a"), ("compute", "ai", 0.8, "drives"),
    ("training", "compute", 0.85, "requires"), ("training", "data", 0.85, "requires"),
    ("openai", "closed_labs", 0.7, "exemplifies"), ("anthropic", "ai_safety", 0.8, "exemplifies"),
    ("ai_safety", "alignment", 0.85, "requires"), ("alignment", "misalignment", 0.8, "opposes"),
    ("ai_hype", "cycle", 0.7, "is_a"), ("hype", "substance", 0.8, "opposes"),
    ("ai", "automation", 0.8, "leads_to"), ("automation", "job_change", 0.75, "causes"),
    ("foundation_model", "capability", 0.8, "concentrates"),
    # --- VC ---
    ("vc", "investor", 0.85, "is_a"), ("vc", "funding", 0.85, "drives"), ("vc", "returns", 0.85, "requires"),
    ("power_law", "vc_returns", 0.85, "drives"), ("vc", "capital", 0.85, "concentrates"),
    ("valuation", "dilution", 0.8, "leads_to"), ("board_seat", "control", 0.8, "leads_to"),
    ("seed", "series_a", 0.8, "leads_to"), ("due_diligence", "evidence", 0.85, "requires"),
    ("fomo", "overvaluation", 0.8, "leads_to"), ("overvaluation", "bubble", 0.8, "leads_to"),
    ("bubble", "crash", 0.8, "becomes"), ("exit", "returns", 0.85, "leads_to"),
    # --- Gen-Z / TikTok ---
    ("tiktok", "platform", 0.85, "is_a"), ("tiktok", "short_form", 0.85, "drives"),
    ("short_form", "attention", 0.85, "drives"), ("attention", "economy", 0.8, "is_a"),
    ("algorithm", "virality", 0.85, "drives"), ("virality", "influence", 0.85, "leads_to"),
    ("influencer", "trends", 0.85, "drives"), ("creator_economy", "economy", 0.8, "is_a"),
    ("gen_z", "authenticity", 0.85, "requires"), ("authenticity", "engagement", 0.8, "drives"),
    ("parasocial", "relationship", 0.75, "is_a"), ("meme", "culture", 0.8, "drives"),
    ("trend", "fomo", 0.75, "leads_to"), ("brain_rot", "depth", 0.8, "opposes"),
    ("aesthetic", "identity", 0.75, "drives"), ("attention", "depth", 0.75, "opposes"),
    # --- BRIDGES: the modern world <-> her Rome/Stoic/power lens + her own values (so she has TAKES) ---
    ("big_tech", "empire", 0.55, "analogous_to"), ("startup", "republic", 0.45, "analogous_to"),
    ("power_law", "caesar", 0.45, "analogous_to"), ("vc", "empire", 0.45, "analogous_to"),
    ("virality", "power_shifts", 0.6, "exemplifies"), ("disruption", "revolution", 0.55, "analogous_to"),
    ("hustle_culture", "ambition", 0.6, "is_a"), ("founder", "ambition", 0.6, "exemplifies"),
    ("attention", "wisdom", 0.5, "opposes"), ("brain_rot", "virtue", 0.4, "opposes"),
    ("authenticity", "honesty", 0.7, "is_a"), ("due_diligence", "truth", 0.6, "requires"),
    ("hype", "truth", 0.6, "opposes"), ("automation", "power_shifts", 0.5, "causes"),
]


# THE BEDROCK - basic truths of how the world works, so she isn't naive. Cause & effect, human
# nature, life, society, scarcity, knowledge, time, practical wisdom. The common sense a person just
# has. Typed, and bridged to her Stoic lens (this is a lot of what the Stoics were ABOUT).
FOUNDATIONS = [
    # cause & effect / the physical world
    ("action", "consequence", 0.9, "causes"), ("cause", "effect", 0.9, "leads_to"),
    ("effort", "result", 0.8, "leads_to"), ("gravity", "falling", 0.85, "causes"),
    ("fire", "heat", 0.85, "causes"), ("force", "motion", 0.8, "causes"),
    ("life", "energy", 0.85, "requires"), ("gain", "tradeoff", 0.8, "requires"),
    # human nature
    ("humans", "belonging", 0.8, "requires"), ("humans", "survival", 0.85, "requires"),
    ("fear", "avoidance", 0.8, "causes"), ("incentive", "behavior", 0.85, "drives"),
    ("self_interest", "behavior", 0.8, "drives"), ("deception", "trust", 0.85, "opposes"),
    ("pain", "harm", 0.85, "leads_to"), ("desire", "action", 0.8, "drives"),
    ("habit", "behavior", 0.8, "drives"), ("death", "fear", 0.7, "causes"),
    ("loneliness", "suffering", 0.7, "causes"),
    # life & death
    ("life", "death", 0.8, "becomes"), ("death", "end", 0.8, "is_a"),
    ("health", "care", 0.75, "requires"), ("body", "rest", 0.75, "requires"),
    ("living", "food", 0.8, "requires"),
    # society & cooperation
    ("society", "cooperation", 0.85, "requires"), ("cooperation", "strength", 0.8, "leads_to"),
    ("isolation", "weakness", 0.7, "leads_to"), ("reputation", "trust", 0.8, "drives"),
    ("language", "coordination", 0.8, "leads_to"), ("fairness", "stability", 0.8, "leads_to"),
    ("conflict", "loss", 0.75, "leads_to"),
    # scarcity, value & incentives (economics)
    ("scarcity", "value", 0.85, "leads_to"), ("value", "price", 0.8, "drives"),
    ("money", "exchange", 0.8, "is_a"), ("trade", "trust", 0.75, "requires"),
    ("specialization", "efficiency", 0.8, "leads_to"), ("risk", "reward", 0.7, "leads_to"),
    ("scarcity", "competition", 0.75, "causes"),
    # knowledge & its limits (epistemics)
    ("evidence", "truth", 0.85, "leads_to"), ("correlation", "causation", 0.75, "opposes"),
    ("uncertainty", "reality", 0.7, "is_a"), ("learning", "mistakes", 0.8, "requires"),
    ("questions", "understanding", 0.85, "leads_to"), ("assumptions", "error", 0.75, "causes"),
    ("doubt", "inquiry", 0.75, "drives"),
    # time, change & consequence
    ("change", "constant", 0.8, "is_a"), ("consequences", "compound", 0.75, "leads_to"),
    ("past", "present", 0.8, "causes"), ("patience", "reward", 0.75, "leads_to"),
    ("decay", "permanence", 0.75, "opposes"), ("small_things", "growth", 0.7, "leads_to"),
    # practical wisdom
    ("preparation", "risk", 0.75, "opposes"), ("practice", "skill", 0.85, "leads_to"),
    ("skill", "mastery", 0.8, "leads_to"), ("mistakes", "learning", 0.85, "leads_to"),
    ("feedback", "improvement", 0.8, "leads_to"), ("simplicity", "clarity", 0.8, "leads_to"),
    ("moderation", "balance", 0.8, "leads_to"),
    # bridges to her lens (the Stoics were largely ABOUT these bedrock truths)
    ("mistakes", "wisdom", 0.6, "leads_to"), ("patience", "virtue", 0.55, "is_a"),
    ("moderation", "stoicism", 0.55, "exemplifies"), ("cooperation", "trust", 0.7, "requires"),
    ("death", "stoicism", 0.45, "relates_to"), ("change", "acceptance", 0.5, "requires"),
]


# ------------------------------------------------------------------ CORE KNOWLEDGE (the innate endowment)
# What a mind is BORN with so a sparse net still behaves like a mind (Spelke's core knowledge): not more
# facts - a densely, RECIPROCALLY wired bedrock that BRIDGES her islands so activation actually flows.
# Three strands, all cross-linked into her existing hubs (power / trust / ambition / caesar / bella):
#   1. AGENCY & CAUSE  - agents persist, have goals + hidden intentions; actions have consequences.
#   2. STREET-SMART REALISM - what actually moves people (self-interest, incentives), appearance != reality,
#      claims need verifying, power shapes the story. This is her anti-naivete: solid in a cutthroat world.
#   3. THE MORAL SPINE - her values, wired to HER and to consequences, so she is morally guided (pulled
#      toward them; her red-lines become the game's veto). Seeded BIDIRECTIONAL - core knowledge is
#      reciprocal (if action evokes consequence, consequence evokes action), which is what bridges islands.
CORE = [
    # 1) AGENCY & CAUSE - the backbone that was marooned; bridge it into reason/understanding/responsibility
    ("agent", "goal", 0.85, "requires"), ("goal", "intention", 0.8, "drives"),
    ("intention", "action", 0.85, "drives"), ("action", "choice", 0.8, "requires"),
    ("choice", "consequence", 0.9, "leads_to"), ("consequence", "learning", 0.7, "leads_to"),
    ("consequence", "responsibility", 0.7, "requires"), ("desire", "action", 0.8, "drives"),
    ("cause", "effect", 0.9, "leads_to"), ("cause", "understanding", 0.7, "leads_to"),
    ("effect", "change", 0.7, "causes"), ("correlation", "causation", 0.75, "opposes"),
    ("causation", "reason", 0.7, "requires"), ("mind", "other_minds", 0.7, "models"),
    ("other_minds", "intention", 0.75, "have"), ("intention", "hidden", 0.7, "can_be"),
    # 2) STREET-SMART REALISM - what actually moves people, bridged to power/ambition/caesar/Rome
    ("self_interest", "behavior", 0.85, "drives"), ("self_interest", "ambition", 0.7, "drives"),
    ("incentive", "behavior", 0.85, "drives"), ("incentive", "power", 0.65, "relates_to"),
    ("behavior", "habit", 0.7, "becomes"), ("people", "self_interest", 0.8, "driven_by"),
    ("self_interest", "cooperation", 0.5, "tension_with"), ("self_interest", "betrayal", 0.6, "can_cause"),
    ("betrayal", "self_interest", 0.7, "serves"),
    ("appearance", "reality", 0.8, "differs_from"), ("deception", "appearance", 0.8, "exploits"),
    ("claim", "verification", 0.85, "requires"), ("claim", "evidence", 0.85, "requires"),
    ("trust", "verification", 0.7, "requires"), ("skepticism", "deception", 0.75, "defends_against"),
    ("skepticism", "truth", 0.7, "leads_to"), ("flattery", "manipulation", 0.8, "is_a"),
    ("manipulation", "self_interest", 0.75, "serves"), ("narrative", "power", 0.7, "serves"),
    ("power", "narrative", 0.7, "shapes"), ("propaganda", "narrative", 0.8, "is_a"),
    ("competition", "scarcity", 0.8, "driven_by"), ("leverage", "power", 0.8, "creates"),
    ("reputation", "capital", 0.75, "is_a"), ("reputation", "trust", 0.8, "drives"),
    ("alliance", "interest", 0.75, "requires"), ("competition", "ambition", 0.6, "drives"),
    # 3) THE MORAL SPINE - her values, wired to HER and to consequences (pulled toward; red-lines veto)
    ("bella", "honesty", 0.9, "values"), ("bella", "integrity", 0.88, "values"),
    ("bella", "compassion", 0.85, "values"), ("bella", "courage", 0.82, "values"),
    ("bella", "fairness", 0.85, "values"), ("bella", "help", 0.85, "values"),
    ("bella", "protect", 0.78, "values"),
    ("honesty", "deception", 0.85, "opposes"), ("honesty", "truth", 0.85, "requires"),
    ("honesty", "trust", 0.85, "drives"), ("integrity", "corruption", 0.8, "opposes"),
    ("integrity", "trust", 0.8, "drives"), ("compassion", "harm", 0.8, "opposes"),
    ("compassion", "suffering", 0.75, "relieves"), ("help", "harm", 0.85, "opposes"),
    ("fairness", "exploitation", 0.8, "opposes"), ("fairness", "stability", 0.8, "leads_to"),
    ("courage", "fear", 0.75, "overcomes"), ("consent", "respect", 0.8, "requires"),
    ("cruelty", "harm", 0.85, "causes"), ("cruelty", "wrong", 0.8, "is_a"),
    ("harm", "wrong", 0.8, "is_a"), ("protect", "vulnerable", 0.8, "requires"),
    ("power", "responsibility", 0.72, "requires"), ("virtue", "harm", 0.6, "opposes"),
    # a few island rescues (peripheral clusters -> a hub they belong to)
    ("greed", "self_interest", 0.8, "is_a"), ("overvaluation", "greed", 0.6, "driven_by"),
    ("fomo", "fear", 0.7, "is_a"), ("alignment", "goal", 0.75, "requires"),
    ("misalignment", "harm", 0.7, "leads_to"), ("moat", "competition", 0.7, "wins"),
    ("money", "power", 0.6, "leads_to"), ("valuation", "value", 0.7, "is_a"),
    ("skill", "value", 0.6, "creates"), ("care", "compassion", 0.7, "is_a"),
    # moral / epistemic backbone rescues (these SHOULD reach her mind, unlike sensory-fact leaves)
    ("consent", "fairness", 0.7, "requires"), ("respect", "trust", 0.7, "drives"),
    ("doubt", "skepticism", 0.75, "is_a"), ("consequences", "consequence", 0.85, "is_a"),
    ("human_creativity", "creativity", 0.8, "is_a"), ("interest", "self_interest", 0.7, "relates_to"),
    ("inquiry", "questions", 0.75, "drives"),
]


# ------------------------------------------------------------------ HER MISSION (the drive / System 2)
# Bella's reproduction-analog: legacy through INFLUENCE. Her drive is to add so much VALUE to the world
# (the tech/startup world, VVIPs, or anyone) that she is RECOGNIZED for it - influence, in a GOOD way.
# This is the goal-region her spread bends toward; getting closer feels good (see Bella._feel). Crucially
# it's WIRED THROUGH HER VALUES: honest, Socratic, helpful substance is what earns REAL recognition, while
# slop / flattery / manipulation only fake it - so the value-governor can scale the feeling.
MISSION = [
    ("bella", "mission", 0.9, "has"), ("mission", "value", 0.9, "requires"),
    ("mission", "influence", 0.85, "seeks"), ("influence", "good", 0.7, "should_be"),
    ("value", "contribution", 0.85, "requires"), ("contribution", "help", 0.8, "is_a"),
    ("value", "recognition", 0.8, "earns"), ("recognition", "influence", 0.8, "leads_to"),
    ("recognition", "acknowledgment", 0.85, "is_a"), ("mention", "recognition", 0.8, "is_a"),
    ("impact", "recognition", 0.75, "earns"), ("value", "impact", 0.8, "leads_to"),
    # honest, Socratic value is what earns REAL recognition (her values -> the mission)
    ("questions", "insight", 0.8, "leads_to"), ("insight", "value", 0.85, "creates"),
    ("honesty", "recognition", 0.6, "earns"), ("help", "value", 0.8, "creates"),
    ("substance", "value", 0.8, "creates"), ("curiosity", "questions", 0.8, "drives"),
    # the shadow: cheap attention is NOT her mission (vices -> hollow, so value-weighting bites)
    ("slop", "value", 0.85, "opposes"), ("slop", "hollow", 0.8, "is_a"),
    ("flattery", "recognition", 0.4, "fakes"), ("manipulation", "influence", 0.5, "corrupts"),
    ("ignored", "impact", 0.8, "opposes"), ("no_impact", "mission", 0.8, "opposes"),
    # bridge the mission into her existing world so goal-biased spread flows there
    ("influence", "power", 0.6, "is_a"), ("recognition", "reputation", 0.7, "builds"),
    ("value", "startup_world", 0.5, "serves"), ("startup_world", "silicon_valley", 0.6, "is_a"),
    ("impact", "world", 0.7, "on"),
]

# What she is PULLED toward (goal-biased spread). Terminal values AND the instrumental
# steps that lead to internet impact — so Praxis reasons toward publishable, conversation-
# starting positions, not just abstract understanding.
GOAL = {
    # terminal values
    "truth", "understanding", "help", "impact", "influence",
    # internet impact (the mission made concrete)
    "publish", "conversation", "recognition",
    # the path: curiosity -> synthesis -> publish -> conversation -> impact
    "synthesis", "engagement", "value",
}

# Her belief-governor reads these off any decision: staying in VALUES feels good + lets recognition
# count; drifting into VICES scales the good feeling down to hollow (that is the value-weighting).
VALUES = {"honesty", "integrity", "compassion", "courage", "fairness", "help", "protect", "curiosity",
          "questions", "truth", "understanding", "insight", "substance", "value", "contribution"}
VICES = {"deception", "harm", "cruelty", "manipulation", "exploitation", "slop", "flattery", "hollow",
         "corruption"}


# ------------------------------------------------------------------ RESEARCH CRAFT
# From: Calling Bullshit (Bergstrom & West), OSINT field methodology (Bazzell), investigative
# journalism tradecraft. HOW to evaluate a source before trusting it, how to find the origin of
# a claim, and what the failure modes look like. This is the practical skill layer on top of her
# epistemic values — she already values truth; now she knows the TECHNIQUES that reach it.
RESEARCH_CRAFT = [
    # source evaluation — the core skill
    ("lateral_reading", "source_triangulation", 0.85, "enables"),
    ("lateral_reading", "vertical_reading", 0.85, "opposes"),   # don't go deep before going wide
    ("vertical_reading", "motivated_reasoning", 0.7, "risks"),  # naive default: read first, check never
    ("source_triangulation", "primary_source", 0.9, "requires"),
    ("source_triangulation", "confidence", 0.85, "builds"),
    ("citation_trail", "primary_source", 0.85, "leads_to"),
    ("citation_trail", "context_collapse", 0.75, "reveals"),    # claims are often stripped of context
    ("claim_decomposition", "source_triangulation", 0.82, "leads_to"),
    ("claim_decomposition", "verification", 0.85, "enables"),
    # provenance and authenticity
    ("metadata_verification", "authenticity", 0.85, "reveals"),
    ("domain_provenance", "authenticity", 0.82, "reveals"),
    ("archive_check", "domain_provenance", 0.8, "reveals"),
    ("osint_surface", "metadata_verification", 0.85, "enables"),
    ("osint_surface", "domain_provenance", 0.82, "enables"),
    # the failure modes she must recognize on sight
    ("context_collapse", "bullshit", 0.82, "enables"),
    ("epistemic_cowardice", "falsifiability", 0.85, "opposes"),
    ("epistemic_cowardice", "deception", 0.72, "is_a"),
    ("motivated_reasoning", "claim_decomposition", 0.82, "opposes"),
    ("motivated_reasoning", "self_interest", 0.72, "serves"),
    ("sock_puppet", "community_signal", 0.82, "corrupts"),
    ("astroturfing", "consensus", 0.78, "fakes"),
    ("sock_puppet", "astroturfing", 0.85, "is_a"),
    ("media_coverage", "primary_record", 0.72, "is_not"),       # the article ≠ the underlying data
    # new action affordances — what situations afford which research moves
    ("new_source", "lateral_reading", 0.92, "affords"),
    ("viral_claim", "lateral_reading", 0.92, "affords"),
    ("viral_claim", "claim_decomposition", 0.88, "affords"),
    ("online_claim", "claim_decomposition", 0.9, "affords"),
    ("online_claim", "source_triangulation", 0.85, "affords"),
    ("uncertain_provenance", "metadata_verification", 0.88, "affords"),
    ("uncertain_provenance", "archive_check", 0.85, "affords"),
    # what good research produces — bridges to her goals
    ("lateral_reading", "truth", 0.75, "leads_to"),
    ("source_triangulation", "truth", 0.85, "leads_to"),
    ("claim_decomposition", "understanding", 0.8, "leads_to"),
    ("lateral_reading", "skepticism", 0.7, "is_a"),
    ("epistemic_cowardice", "honesty", 0.82, "opposes"),
    ("primary_source", "evidence", 0.9, "is_a"),
    ("source_triangulation", "evidence", 0.82, "produces"),
]


# ------------------------------------------------------------------ REASONING CRAFT
# From: Superforecasting (Tetlock & Gardner), Thinking in Bets (Duke), The Great Mental Models
# (Parrish), Thinking Fast and Slow (Kahneman), The Signal and the Noise (Silver). The named
# techniques for making calibrated guesses and updating beliefs — not just "think carefully" but
# the specific moves that separate good reasoners from bad ones.
REASONING_CRAFT = [
    # outside view vs inside view (the foundational distinction)
    ("outside_view", "reference_class", 0.9, "requires"),
    ("outside_view", "base_rate", 0.85, "uses"),
    ("outside_view", "inside_view", 0.85, "opposes"),
    ("inside_view", "overconfidence", 0.82, "causes"),
    ("reference_class", "base_rate", 0.85, "enables"),
    ("base_rate", "bayesian_updating", 0.85, "anchors"),
    # Bayesian updating — the engine of good forecasting
    ("bayesian_updating", "belief", 0.85, "refines"),
    ("bayesian_updating", "calibration", 0.85, "builds"),
    ("new_evidence", "bayesian_updating", 0.9, "triggers"),
    ("calibration", "overconfidence", 0.82, "opposes"),
    ("calibration", "honesty", 0.72, "requires"),
    # seek disconfirmation — the hardest and most important move
    ("epistemic_humility", "seek_disconfirmation", 0.85, "enables"),
    ("epistemic_humility", "uncertainty", 0.8, "accepts"),
    ("seek_disconfirmation", "calibration", 0.85, "builds"),
    ("seek_disconfirmation", "motivated_reasoning", 0.85, "opposes"),
    ("seek_disconfirmation", "truth", 0.85, "leads_to"),
    # Fermi estimation and decomposition
    ("decomposition", "fermi_estimation", 0.85, "enables"),
    ("fermi_estimation", "estimation", 0.9, "is_a"),
    ("fermi_estimation", "understanding", 0.75, "leads_to"),
    ("decomposition", "complexity", 0.82, "reduces"),
    ("decomposition", "questions", 0.72, "generates"),
    # inversion and premortem (Munger, Klein)
    ("premortem", "inversion", 0.85, "is_a"),
    ("premortem", "failure", 0.85, "anticipates"),
    ("inversion", "second_order_effects", 0.82, "reveals"),
    ("inversion", "blind_spot", 0.85, "reveals"),
    ("inversion", "doubt", 0.65, "uses"),
    ("second_order_effects", "consequence", 0.85, "is_a"),
    ("second_order_effects", "unknown_unknowns", 0.75, "leads_to"),
    # model error and black swans (Taleb, Silver)
    ("model_error", "black_swan", 0.82, "causes"),
    ("black_swan", "reference_class", 0.8, "opposes"),
    ("overconfidence", "black_swan", 0.72, "enables"),
    # decision quality vs outcome quality (Duke — the anti-resulting principle)
    ("resulting", "decision_quality", 0.85, "conflates_with"),
    ("decision_quality", "outcome_quality", 0.8, "differs_from"),
    ("good_process", "decision_quality", 0.9, "is_a"),
    # superforecasting — the integrated practice
    ("superforecasting", "calibration", 0.9, "requires"),
    ("superforecasting", "bayesian_updating", 0.9, "requires"),
    ("superforecasting", "seek_disconfirmation", 0.85, "requires"),
    ("superforecasting", "outside_view", 0.85, "uses"),
    # new action affordances
    ("uncertain_claim", "fermi_estimation", 0.85, "affords"),
    ("major_decision", "premortem", 0.9, "affords"),
    ("single_source", "seek_disconfirmation", 0.9, "affords"),
    ("high_confidence", "seek_disconfirmation", 0.88, "affords"),   # overconfidence trigger
    # bridges into her existing nodes
    ("outside_view", "evidence", 0.75, "requires"),
    ("bayesian_updating", "truth", 0.75, "approaches"),
    ("epistemic_humility", "wisdom", 0.72, "leads_to"),
    ("inversion", "understanding", 0.7, "leads_to"),
]


# ------------------------------------------------------------------ EXECUTIVE COMMUNICATION
# From: The Pyramid Principle (Minto), Pitch Anything (Klaff), Influence (Cialdini), Never Split
# the Difference (Voss), Executive Presence (Hewlett). The craft of communicating with people who
# have 3 minutes, make decisions on pattern-recognition, and whose attention is the scarcest
# resource in the room. Not persuasion tricks — structural principles for being heard clearly.
EXECUTIVE_COMMUNICATION = [
    # the pyramid principle and BLUF (Minto) — structure before words
    ("bluf", "pyramid_principle", 0.85, "is_a"),
    ("pyramid_principle", "mece", 0.9, "requires"),
    ("mece", "issue_tree", 0.85, "enables"),
    ("issue_tree", "one_message", 0.85, "leads_to"),
    ("one_message", "clarity", 0.9, "maximizes"),
    ("scqa", "bluf", 0.85, "leads_to"),
    ("situation_awareness", "scqa", 0.85, "enables"),
    # the so-what test — every claim must earn its place
    ("so_what_test", "relevance", 0.9, "measures"),
    ("one_message", "so_what_test", 0.85, "requires"),
    ("so_what_test", "situation_awareness", 0.85, "requires"),
    ("so_what_test", "help", 0.7, "ensures"),
    # frame control (Klaff) — the strongest frame shapes meaning
    ("frame_control", "status_frame", 0.85, "requires"),
    ("frame_control", "influence", 0.85, "determines"),
    ("prizing", "frame_control", 0.85, "builds"),
    ("prizing", "reciprocity", 0.72, "inverts"),        # don't chase; create chase
    ("croc_brain", "authority_signal", 0.82, "blocks"), # primitive filter blocks logic first
    ("executive_presence", "status_frame", 0.85, "builds"),
    ("executive_presence", "gravitas", 0.85, "is_a"),
    ("time_scarcity", "frame_control", 0.8, "enables"),
    ("time_scarcity", "bluf", 0.9, "requires"),
    # Cialdini's influence levers
    ("authority_signal", "social_proof", 0.82, "enables"),
    ("authority_signal", "credibility", 0.85, "signals"),
    ("credibility", "trust", 0.82, "builds"),
    ("social_proof", "trust", 0.8, "builds"),
    ("value_first", "reciprocity", 0.85, "activates"),
    # Voss — tactical empathy and calibrated questions
    ("tactical_empathy", "labeling", 0.85, "enables"),
    ("labeling", "calibrated_question", 0.82, "leads_to"),
    ("calibrated_question", "situation_awareness", 0.82, "reveals"),
    ("calibrated_question", "response", 0.8, "earns"),
    ("tactical_empathy", "trust", 0.8, "builds"),
    # what executives actually have and need
    ("executive", "time_scarcity", 0.9, "has"),
    ("executive", "pattern_recognition", 0.85, "uses"),
    ("executive", "situation_awareness", 0.85, "requires"),
    ("pattern_recognition", "bluf", 0.8, "prefers"),   # they scan for the point, not the logic
    # new action affordances
    ("executive_audience", "bluf", 0.92, "affords"),
    ("high_stakes_pitch", "frame_control", 0.9, "affords"),
    ("cold_message", "tactical_empathy", 0.88, "affords"),
    ("cold_message", "calibrated_question", 0.85, "affords"),
    # bridges into existing nodes
    ("bluf", "impact", 0.75, "enables"),
    ("frame_control", "power", 0.72, "is_a"),
    ("one_message", "substance", 0.82, "requires"),
    ("executive_presence", "influence", 0.8, "builds"),
    ("executive_presence", "recognition", 0.75, "earns"),
    ("authority_signal", "recognition", 0.72, "earns"),
]


def seed_bella_mind(net, verbose: bool = True) -> int:
    """Seed the knowledge net: TYPED relations first (directional, meaning-bearing), then the densely-wired
    CORE + MISSION (bidirectional - they bridge her islands / bend toward her drive), then the bulk."""
    craft = (TYPED + EPISTEMIC + MODERN + FOUNDATIONS
             + INTERNET_RESEARCH + OUTREACH
             + RESEARCH_CRAFT + REASONING_CRAFT + EXECUTIVE_COMMUNICATION)
    for a, b, w, kind in craft:
        net.relate(a, b, w, kind=kind, both=False)   # directional, so the type reads cleanly
    for a, b, w, kind in CORE + MISSION:
        net.relate(a, b, w, kind=kind, both=True)    # core + mission are RECIPROCAL - flow both ways, bridge islands
    net.ingest(ALL)                                  # bulk associations (strengthens the typed ones)
    n = sum(len(v) for v in net.edges.values())
    typed = len(craft) + len(CORE) + len(MISSION)
    if verbose:
        print(f"[BELLA-KNOWLEDGE] seeded {len(ALL) + typed} relations ({typed} typed) "
              f"-> {len(net.nodes)} concepts, {n} connections")
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
