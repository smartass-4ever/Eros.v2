"""
BELLA'S PERCEPTION - she reads real content and understands it FOR WHAT IT IS.

The linchpin. It turns a chunk of text (an article, a paper, a post) into the three things the rest
of her mind has been starved for:
  - CONCEPTS   : what it's about (seeds for reasoning)
  - ENTITIES   : who/what is in it (an author, a source) - so she can PURSUE them
  - MARKERS    : what KIND of thing this is (author / claim / unknown) - so her action choice is sharp
  - RELATIONS  : TYPED relations she extracts and INGESTS -> this is how reading GROWS her knowledge
                 (and, once her rewards are grounded in real outcomes, how she stops being ungrounded)

Heuristic first version (patterns, not deep IE). The ORGAN and the wiring are the point; real
robustness (better extraction, embeddings) and live web (Indra) come with release. It reuses the
same idea as Eros's PerceptionModule.parse_input (intent + entities), extended for web content.
"""
import re

STOP = {"the", "and", "that", "this", "with", "from", "have", "has", "was", "were", "are", "for",
        "but", "not", "you", "your", "they", "their", "them", "then", "than", "into", "over", "some",
        "what", "when", "which", "while", "will", "would", "could", "should", "about", "also", "more",
        "most", "such", "very", "been", "being", "there", "these", "those", "here", "just"}

# verb phrase in the text  ->  her relation TYPE (so a sentence becomes a typed edge she can ingest)
REL_VERBS = [
    (r"concentrat\w*", "concentrates"), (r"distribut\w*", "distributes"),
    (r"leads? to", "leads_to"), (r"causes?|drives?", "causes"), (r"becomes?|turns? into", "becomes"),
    (r"opposes?|against|versus|vs", "opposes"), (r"requires?|rests? on|depends? on", "requires"),
    (r"exemplif\w*|embodies", "exemplifies"), (r"is a|is the|is essentially", "is_a"),
]


def _word(w):
    return re.sub(r"[^a-z]", "", w.lower())


def perceive(text: str, known_net=None) -> dict:
    """Read text -> {concepts, entities, markers, relations}. All heuristic, all hers, no LLM."""
    t = (text or "").strip()
    low = t.lower()

    # ENTITIES - an author / source she could go pursue
    entities = {}
    m = (re.search(r"\bby ((?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", t)
         or re.search(r"\b(Dr\.?\s+[A-Z][a-z]+)", t)
         or re.search(r"\b([A-Z][a-z]+ et al\.?)", t))
    if m:
        entities["author"] = m.group(1).strip()

    # CONCEPTS - content words she can reason over
    seen, concepts = set(), []
    for w in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", low):
        w = _word(w)
        if w and w not in STOP and w not in seen:
            seen.add(w); concepts.append(w)
    concepts = concepts[:12]

    # MARKERS - what KIND of thing gripped her (drives which action she'll reason to)
    markers = []
    if "author" in entities:
        markers.append("author")
    if re.search(r"\b(argues?|claims?|shows?|proves?|suggests?|finds?|reveals?) that\b", low):
        markers.append("claim")
    if known_net is not None and any(c not in known_net.nodes for c in concepts[:5]):
        markers.append("unknown")
    markers.append("interesting")

    # RELATIONS - typed edges she extracts and will INGEST (reading -> knowledge)
    relations = []
    for pat, kind in REL_VERBS:
        for mm in re.finditer(rf"(\w{{4,}})\s+(?:\w+\s+){{0,2}}?(?:{pat})\s+(?:\w+\s+){{0,2}}?(\w{{4,}})", low):
            a, b = _word(mm.group(1)), _word(mm.group(2))
            if a and b and a != b and a not in STOP and b not in STOP:
                relations.append((a, b, 0.5, kind))

    return {"concepts": concepts, "entities": entities,
            "markers": tuple(dict.fromkeys(markers)), "relations": relations[:8]}


if __name__ == "__main__":
    from reasoning_core import KnowledgeNet
    from bella_knowledge import seed_bella_mind
    net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
    article = ("New research by Dr. Vega shows that open-source models are closing the gap. "
               "The paper argues that decentralization distributes power, while closed labs "
               "concentrate control. Open source opposes secrecy, and transparency becomes trust.")
    print("she reads:\n  " + article + "\n")
    p = perceive(article, known_net=net)
    print("she perceives:")
    print("  concepts :", p["concepts"][:6])
    print("  entities :", p["entities"])
    print("  markers  :", p["markers"])
    print("  relations she'll LEARN (typed):")
    for a, b, w, k in p["relations"]:
        print(f"      {a} --{k}--> {b}")
