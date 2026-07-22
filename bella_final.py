"""
BELLA - THE LAST FORM. Everything wired into one.

A sandboxed SWARM of full Bella beings - each one the real Eros CNS (perception, emotion, memory,
curiosity, personality) driven by the Praxis v2 glass-box mind - organized by the kimi 3-tier org,
sharing ONE stratified memory, every LLM call cached + swarm-shared, every action safety-gated and
visible. This is the assembled whole:

    being (Eros)   +   mind (Praxis v2, glass-box)   +   swarm (kimi 3-tier)
    +   one collective stratified memory   +   cost architecture (cache + reduction)
    +   visible safety gate   =   Bella.

Strategy flows DOWN (shared beliefs), pulses flow UP (no log-flood). Growth is EARNED - a being
replicates only on credible engagement, never raw volume. Nothing she thinks or does is off the glass.

Run:  python bella_final.py
"""
import asyncio
from swarm_org import BellaSwarm
from bella_llm_cache import cache_stats

# what the whole swarm turns its attention to, in sequence (overlap is the point: the first being
# to reason a topic pays the tokens, the rest inherit it from the shared mind + cache)
TOPICS = [
    "the latest breakthroughs in AI agents and reasoning models",
    "open-source AI versus the big closed labs",
    "Marcus Aurelius, Seneca, and Stoic philosophy",
    "how power actually shifts in societies and revolutions",
]


async def main(cluster: int = 3, turns: int = 4):
    print("=" * 74)
    print("  BELLA - THE LAST FORM")
    print("  a swarm of full beings | glass-box minds | one shared memory | visible safety")
    print("=" * 74)

    swarm = BellaSwarm()
    swarm.add_cluster(size=cluster, specialty="explore")
    print(f"\n  spawned a SANDBOXED cluster of {cluster} Bella beings under the kimi 3-tier org")
    print(f"  guardrail (encoded in the Sovereign's beliefs):")
    for k, v in swarm.GUARDRAIL.items():
        print(f"       {k:<10} = {v}")
    print()

    for t in range(turns):
        topic = TOPICS[t % len(TOPICS)]
        print("-" * 74)
        print(f"  TURN {t}  ->  the swarm attends to: {topic!r}")
        decision = await swarm.turn({"focus": topic})

        for m in swarm.managers:                        # each being's glass-box thought
            for w in m.interns:
                d = getattr(w.being, "_last_decision", {}) or {}
                concl = (d.get("conclusion") or "(forming a thought)")[:90]
                conf = d.get("confidence", 0.0)
                print(f"     [{w.agent_id}] \"{concl}\"")
                print(f"         confidence {conf:.2f} | reward {getattr(w,'reward',0):.2f} | glass-box trace kept")

        print(f"     collective mind : {swarm.memory.snapshot() or 'building...'}")
        print(f"     sovereign       : {decision.get('action')} - {decision.get('reason')}")
        print(f"     shared cache    : {cache_stats()}")
        print()

    print("=" * 74)
    print("  STATE OF BELLA  (the whole, assembled)")
    print("=" * 74)
    total = sum(len(m.interns) for m in swarm.managers)
    print(f"     beings in the swarm     : {total}")
    print(f"     collective memory tiers : {swarm.memory.snapshot() or 'building...'}")
    print(f"     shared LLM cache        : {cache_stats()}")
    print(f"     every decision          : glass-box (provable, decomposed payoff)")
    print(f"     every action            : safety-gated + visible")
    print(f"     growth is earned on     : quality + credible engagement (never raw volume)")
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(main())
