"""
COLLECTIVE STRATIFIED MEMORY - the shared mind of the swarm.

One store, written and read by every Bella. What one learns, all can recall. Five stratified
tiers; a memory is PROMOTED upward as it recurs and matters, so the swarm's durable knowledge
(semantic / procedural) sharpens over time. Retrieval is by relevance. It also emits learned
relations to feed each Bella's reasoning net (ingest) - that is how the collective memory
deepens every individual Bella's mind.

Lives on the SovereignOrchestrator (Tier 1) so the whole swarm shares one mind.
"""
from dataclasses import dataclass, field
from collections import defaultdict
import time

TIERS = ["sensory", "working", "episodic", "semantic", "procedural"]
TIER_WEIGHT = {"sensory": 0, "working": 1, "episodic": 2, "semantic": 3, "procedural": 3}


@dataclass
class Trace:
    key: str
    content: str
    tier: str = "working"
    hits: int = 1                       # recurrence - drives promotion up the strata
    salience: float = 0.5
    relations: list = field(default_factory=list)   # [(a, b), ...] concept links for reasoning nets
    t: float = field(default_factory=time.time)


class CollectiveMemory:
    def __init__(self):
        self.store: dict[str, Trace] = {}

    def remember(self, key: str, content: str, salience: float = 0.5, relations=None, procedural=False):
        """A Bella writes what it learned. Seen again -> recurrence rises -> it promotes."""
        if key in self.store:
            tr = self.store[key]
            tr.hits += 1
            tr.salience = max(tr.salience, salience)
        else:
            tr = Trace(key, content, salience=salience, relations=list(relations or []))
            self.store[key] = tr
        self._promote(tr, procedural)
        return tr

    def _promote(self, tr: Trace, procedural: bool):
        if tr.hits >= 2 and tr.tier == "working":
            tr.tier = "episodic"
        if tr.hits >= 3 and tr.salience >= 0.5 and tr.tier == "episodic":
            tr.tier = "semantic"              # durable shared knowledge
        if procedural and tr.tier == "semantic":
            tr.tier = "procedural"            # a durable how-to-act

    def recall(self, query: str, k: int = 5):
        """Relevance retrieval - durable knowledge (semantic/procedural) surfaces first."""
        q = set(str(query).lower().split())
        def score(tr: Trace):
            overlap = len(q & set(tr.content.lower().split()))
            return overlap * 2 + TIER_WEIGHT[tr.tier] + tr.hits * 0.5 + tr.salience
        return sorted(self.store.values(), key=score, reverse=True)[:k]

    TIER_BASE = {"sensory": 0.2, "working": 0.28, "episodic": 0.36, "semantic": 0.5, "procedural": 0.6}

    def substrate(self):
        """Emit the swarm's learned relations so a mind can ingest() them into its reasoning net.
        ALL tiers flow (fresh discoveries reach the mind too), weighted by how established they are -
        working = tentative, semantic/procedural = trusted. Confidence grows with recurrence. This is
        how what one agent discovers deepens every mind."""
        rels = []
        for tr in self.store.values():
            base = self.TIER_BASE.get(tr.tier, 0.3)
            for (a, b) in tr.relations:
                rels.append((a, b, min(1.0, base + 0.05 * tr.hits)))
        return rels

    def snapshot(self):
        counts = defaultdict(int)
        for tr in self.store.values():
            counts[tr.tier] += 1
        return {t: counts[t] for t in TIERS if counts[t]}


if __name__ == "__main__":
    from reasoning_core import KnowledgeNet
    mem = CollectiveMemory()
    # three Bellas independently learn overlapping things about feminism
    mem.remember("fem-equality", "feminism is about equality and progress",
                 salience=0.7, relations=[("feminism", "equality"), ("equality", "progress")])
    mem.remember("fem-equality", "again: feminism equality progress")      # bella-2 confirms
    mem.remember("fem-equality", "third time: equality holds up")          # bella-3 confirms -> promotes
    mem.remember("fem-hype", "some feminism takes online are just hype", salience=0.4,
                 relations=[("feminism", "hype")])

    print("tiers:", mem.snapshot())
    print("recall 'feminism progress':")
    for tr in mem.recall("feminism progress", k=3):
        print(f"  [{tr.tier}] {tr.content}  (hits {tr.hits})")

    print("\nsubstrate fed to a Bella's net:")
    rels = mem.substrate()
    for r in rels:
        print("  ", r)
    net = KnowledgeNet()
    net.ingest(rels)
    print("=> a Bella that ingests the collective memory now knows",
          sum(len(v) for v in net.edges.values()), "connections it never learned itself.")
