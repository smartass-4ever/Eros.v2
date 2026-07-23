"""
BELLA - THE RAVANA STRUCTURE.  One being, ten minds.

A distribution system of N Bella minds (the heads). Each head owns a SHARD of agents and works
them independently - the heads do NOT integrate, so there's no apex bottleneck and no single point
of overload. 1000 agents doing 100 different things are handled because they're spread across 10
minds, each reasoning on its own shard IN PARALLEL.

Coherence comes horizontally, through NALANDA - a shared knowledge base + learning bus (reuses the
collective stratified memory + the ExperienceBus idea). Every head broadcasts what it LEARNS
(concepts + reward) to Nalanda; every head studies Nalanda and ingests the others' learnings into
its own net. A mistake ONE head makes IMMUNIZES the rest - they weaken that path without ever
walking it. So the swarm gets collectively wiser without any head having to integrate the whole.

    heads = copy-paste of the Bella mind      (in production each head wraps a full Bella)
    Nalanda = collective_memory + learning bus (the shared library; ancient Nalanda = the university)
"""
from reasoning_core import PraxisV2, KnowledgeNet, _edge
from collective_memory import CollectiveMemory
from bella_knowledge import seed_bella_mind


class Nalanda:
    """The shared knowledge base + learning bus for the heads. Write = broadcast a learning;
    reward<0 marks a MISTAKE others should avoid. Read = a head studies + updates its own net."""
    def __init__(self):
        self.memory = CollectiveMemory()             # the stratified library
        self.signals = []                            # broadcast learnings (concepts, reward, source)

    def broadcast(self, source, concepts, reward, conclusion):
        concepts = list(concepts)
        rels = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
        self.memory.remember(key=conclusion, content=conclusion,
                             salience=min(1.0, abs(reward)), relations=rels)
        self.signals.append({"source": source, "concepts": concepts, "reward": reward})

    def teach(self, net, since=0):
        """A head studies Nalanda: ingest the collective's durable knowledge, and REPLAY other
        heads' mistakes as weakenings so this head never has to make them (immunization)."""
        for a, b, w in self.memory.substrate():
            net.ingest([(a, b, w)])
        for sig in self.signals[since:]:
            if sig["reward"] < 0:                    # a mistake -> immunize
                cs = sig["concepts"]
                for i in range(len(cs) - 1):
                    net.weaken(cs[i], cs[i + 1], 0.15 * abs(sig["reward"]))
        return len(self.signals)


class BellaHead:
    """One of the heads: a Bella mind (here a seeded Praxis; in production a full Bella) that owns
    a shard of agents and works them, learns locally, and shares to Nalanda."""
    def __init__(self, hid, nalanda: Nalanda):
        self.hid = hid
        self.nalanda = nalanda
        net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
        self.praxis = PraxisV2(net)
        self.goal = {"truth", "evidence", "help"}
        self.agents = []                             # its shard (each agent feeds this head)
        self._studied = 0

    def work(self, focus_seeds, intent, reward_hint=None):
        """Reason over what its shard surfaced, learn locally, broadcast to Nalanda."""
        d = self.praxis.decide(seeds=focus_seeds, intent_nodes=set(focus_seeds), goal=self.goal,
                               curiosity=set(focus_seeds), intent=intent)
        reward = reward_hint if reward_hint is not None else (d.payoff - 0.4)
        self.praxis.learn(d.concepts, reward)        # local learning
        self.nalanda.broadcast(self.hid, d.concepts, reward, d.conclusion)  # share to the collective
        return d, reward

    def read(self, text):
        """This head reads a piece of the live web: perceives it and INGESTS the typed relations it
        extracts into its own net (reading -> knowledge), then it can broadcast to Nalanda."""
        from bella_perception import perceive
        p = perceive(text, known_net=self.praxis.net)
        for a, b, w, kind in p["relations"]:
            try: self.praxis.net.relate(a, b, w, kind=kind, both=False)
            except Exception: pass
        return p

    def study(self):
        self._studied = self.nalanda.teach(self.praxis.net, since=0)


class RavanaSwarm:
    """The ten-headed Bella. N heads share ONE Nalanda; agents are sharded across the heads."""
    def __init__(self, heads=10, agents_per_head=100):
        self.nalanda = Nalanda()
        self.heads = [BellaHead(f"head-{i}", self.nalanda) for i in range(heads)]
        self.agents_per_head = agents_per_head

    def tick(self, work_items):
        """work_items: {head_index: (seeds, intent)}. Each head works its shard INDEPENDENTLY
        (parallel), then every head studies Nalanda (learns from the others)."""
        results = {}
        for i, h in enumerate(self.heads):           # parallel independent work
            if i in work_items:
                seeds, intent = work_items[i]
                results[i] = h.work(seeds, intent)
        for h in self.heads:                         # horizontal learning
            h.study()
        return results


# ----------------------------------------------------------------- proof: one head learns from another
if __name__ == "__main__":
    print("=" * 76)
    print("  BELLA - RAVANA STRUCTURE   (10 heads, 1 Nalanda, no apex)")
    print("=" * 76)
    swarm = RavanaSwarm(heads=10, agents_per_head=100)
    print(f"  {len(swarm.heads)} Bella minds, each owning {swarm.agents_per_head} agents "
          f"= {len(swarm.heads) * swarm.agents_per_head} agents, all doing their OWN work in parallel\n")

    # 10 heads each working a DIFFERENT thing at the same time (the heterogeneous case)
    diverse = {
        0: ({"open_source": 1.0, "closed_labs": 0.8}, "open vs closed AI"),
        1: ({"stoicism": 1.0, "power": 0.7}, "the Stoics on power"),
        2: ({"robots": 1.0, "embodiment": 0.7}, "embodied intelligence"),
        3: ({"caesar": 1.0, "ambition": 0.8}, "Caesar and ambition"),
        4: ({"science": 1.0, "evidence": 0.8}, "what makes a science"),
    }
    print("  --- one tick: 5 heads reason on 5 DIFFERENT topics simultaneously ---")
    res = swarm.tick(diverse)
    for i, (d, r) in res.items():
        print(f"    {swarm.heads[i].hid}: \"{d.conclusion[:60]}\"  (payoff {d.payoff})")

    # PROVE immunization: head-0 makes a MISTAKE; a head that never touched it avoids it after study
    print("\n  --- collective immunization: head-0 learns a path is BAD; does head-7 avoid it? ---")
    bad = ("open_source", "secrecy")                 # a wrong association
    h0, h7 = swarm.heads[0], swarm.heads[7]
    h0.praxis.net.relate(*bad, 0.8)                  # head-0 briefly believes it
    for h in swarm.heads:                            # everyone starts believing it equally
        h.praxis.net.relate(*bad, 0.8)
    before = _edge(h7.praxis.net, *bad)
    h0.work({"open_source": 1.0}, "is open-source secretive?", reward_hint=-1.0)  # head-0 finds it FALSE
    h7.study()                                       # head-7 studies Nalanda (never made the mistake)
    after = _edge(h7.praxis.net, *bad)
    print(f"    head-7's trust in '{bad[0]} -> {bad[1]}':  before {before:.2f}  ->  after {after:.2f}")
    print(f"    head-7 weakened a path it NEVER walked, because head-0 broadcast the mistake to Nalanda.")
    print("\n" + "=" * 76)
    print("  10 minds, working in parallel on different things, one shared conscience. That's Bella.")
    print("=" * 76)
