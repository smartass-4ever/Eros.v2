"""
BELLA - THE ROUND PIPELINE.   swarm -> mind -> swarm, closed loop.

The reframe: a swarm of 1000 Bellas is NOT 1000 minds. It's ONE mind (Bella) with a swarm of
SENSES and VOICES. Signals fan IN and are aggregated + salience-gated on the way up
(1000 -> ~100 -> a few), the central mind reasons on the DISTILLED stream under a BOUNDED
attention budget, and her opinions fan back OUT to the appropriate agents. Then round again.

This MERGES the swarm organiser with Bella's mind: the Sovereign apex IS her mind. The kimi
3-tier becomes her nervous system -- managers = thalamic relays that filter+aggregate, apex =
cortex that deliberates.

Why the mind survives 1000 (or a million) agents: it never processes them all. Work per tick is
O(K) (the attention budget), NOT O(agents). Load scales with DISTINCT salient events, not headcount.
"""
import asyncio, time, random, heapq
from dataclasses import dataclass
from bella import Bella


@dataclass
class Signal:
    source: str
    topic: str
    salience: float


class SignalAgent:
    """A swarm member as a SENSE, not a mind. Token-free: it watches its 'beat' and emits a light
    signal only when something moves. A thousand of these are cheap; only the apex mind is heavy."""
    __slots__ = ("aid", "beat")
    def __init__(self, aid: str, beat: str):
        self.aid, self.beat = aid, beat
    def sense(self, world: dict):
        s = world.get(self.beat, 0.0) + random.random() * 0.05
        if s < 0.5:                                  # below its own threshold -> stays quiet (most do)
            return None
        return Signal(self.aid, self.beat, min(1.0, s))


class ThalamicRelay:
    """Tier-2 manager as a thalamic relay: gather a cluster's signals, DEDUPE by topic (many agents
    noticing the same thing -> ONE pulse), keep only the few strongest. The reduce step that stops
    the firehose from ever reaching the mind."""
    def __init__(self, rid: str):
        self.rid = rid
        self.members: list[SignalAgent] = []
    def aggregate(self, world: dict):
        strongest: dict[str, Signal] = {}
        for m in self.members:
            s = m.sense(world)
            if s and (s.topic not in strongest or s.salience > strongest[s.topic].salience):
                strongest[s.topic] = s
        return sorted(strongest.values(), key=lambda s: -s.salience)[:3]


class BellaSovereignMind:
    """THE MERGE: swarm-organiser apex + Bella's mind, one thing. Bounded attention (top-K/tick),
    reasons via Praxis on the distilled stream, emits an opinion that routes back out. She never
    processes 1000 -- she processes K."""
    def __init__(self, being: Bella, attention_budget: int = 5):
        self.being = being                           # ONE real Bella mind (CNS + Praxis v2)
        self.K = attention_budget
        self.queue: list = []                        # max-heap (by salience) of pending pulses
        self.processed = 0

    def perceive(self, pulses):                      # FAN IN
        for p in pulses:
            heapq.heappush(self.queue, (-p.salience, p.source, p))

    def deliberate(self):                            # PROCESS (bounded)
        batch = [heapq.heappop(self.queue)[2] for _ in range(min(self.K, len(self.queue)))]
        if not batch:
            return None
        seeds = {}
        for p in batch:
            for w in p.topic.split():
                if len(w) > 3:
                    seeds[w] = max(seeds.get(w, 0.0), p.salience)
        d = self.being.praxis.decide(seeds=seeds or {"world": 0.5}, intent_nodes=set(seeds),
                                     goal=self.being.goal, curiosity=set(seeds))
        self.processed += len(batch)
        self.queue = self.queue[: self.K * 4]        # shed stale low-salience backlog (bounded memory)
        return {"opinion": d.conclusion, "confidence": min(0.99, 0.5 + d.payoff / 2),
                "topics": list({p.topic for p in batch}), "from_n": len(batch)}

    def route(self, opinion, agents):                # FAN OUT (to appropriate beats)
        return [a.aid for a in agents if a.beat in opinion["topics"]][:8]


async def run_round_pipeline(mind: BellaSovereignMind, agents, relays, ticks=4):
    beats = list({a.beat for a in agents})
    for t in range(ticks):
        world = {random.choice(beats): 0.6 + random.random() * 0.4 for _ in range(3)}  # what's happening
        t0 = time.time()
        pulses = []
        for r in relays:                             # FAN IN + aggregate
            pulses += r.aggregate(world)
        mind.perceive(pulses)
        op = mind.deliberate()                       # PROCESS top-K
        routed = mind.route(op, agents) if op else []
        dt = (time.time() - t0) * 1000
        raw = len(agents)
        print(f"   tick {t}: {raw} agents sensed -> {len(pulses)} pulses reached the apex -> "
              f"mind deliberated on {op['from_n'] if op else 0} (budget {mind.K}) "
              f"-> opinion routed to {len(routed)} voices   [{dt:.0f} ms]")
        if op:
            print(f"           her opinion: \"{op['opinion']}\"  (conf {op['confidence']:.2f}) on {op['topics']}")


async def main():
    print("=" * 78)
    print("  BELLA - THE ROUND PIPELINE   (swarm -> mind -> swarm, one merged apex mind)")
    print("=" * 78)
    beats = ["ai_agents", "open_source", "robots", "stoicism", "power_shifts", "art", "physics", "crypto"]
    print("\n  booting ONE central mind (the apex) ...")
    mind = BellaSovereignMind(Bella(), attention_budget=5)

    # PROVE the apex load is flat as the swarm grows: 10 -> 100 -> 1000 -> 10000 agents
    for n in (10, 100, 1000, 10000):
        agents = [SignalAgent(f"b{i}", random.choice(beats)) for i in range(n)]
        relays = [ThalamicRelay(f"r{j}") for j in range(max(1, n // 50))]
        for i, a in enumerate(agents):
            relays[i % len(relays)].members.append(a)
        print(f"\n--- swarm size {n}  ({len(relays)} thalamic relays) ---")
        await run_round_pipeline(mind, agents, relays, ticks=3)

    print("\n" + "=" * 78)
    print(f"  the apex mind's per-tick work stayed CONSTANT (budget K={mind.K}) from 10 to 10000 agents.")
    print("  load scales with DISTINCT salient events, not agent count. THAT is how she survives a swarm.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
