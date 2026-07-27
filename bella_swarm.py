"""
BELLA'S SWARM - her senses and hands, made real (the kimi/Ravana structure, running).

Each cycle the mind names things worth exploring. A pool of lightweight AGENTS goes out IN PARALLEL -
each one fetches from the world, perceives what it found, and deposits the discovery + the typed
relations it learned into NALANDA (the shared mind). Then the central mind ingests everything the
swarm brought back at once. What ONE agent finds, the whole swarm - and the mind - comes to know.

  apex mind  ── names targets ──►  SWARM (agents explore the world in parallel)
      ▲                                     │  each deposits discovery + relations
      │                                     ▼
      └────────  ingests it all  ◄──────  NALANDA (shared, persistent)

Agents are cheap (fetch + perceive, no brain) so many run at once; scale up with the machine.
"""
import asyncio
import aiohttp
from collective_memory import CollectiveMemory        # this IS Nalanda
from bella_legs import search_web_async, UA
from bella_perception import perceive


class Swarm:
    def __init__(self, size: int = 30):
        self.size = size
        self.nalanda = CollectiveMemory()              # the shared mind - every agent writes here
        self.last = []                                 # what each agent explored last (for the surface)
        self.discovered = 0

    async def _agent(self, aid: str, target, session):
        """One explorer. It goes to find out about the THING the mind named (no verb - a decision is
        just a thing to go toward), reads what it finds, perceives it, and deposits into Nalanda."""
        text = await search_web_async(session, str(target))
        if not text:
            return None
        p = perceive(text)
        rels = p.get("relations", [])
        assoc = p.get("associations", [])
        if rels or assoc:                              # deposit what it learned into the shared mind (Nalanda)
            key = (p["concepts"][0] if p.get("concepts") else target)
            self.nalanda.remember(key=str(key), content=text[:220], salience=0.6,
                                  relations=[(a, b) for a, b, _w, _k in rels] + assoc)
        self.discovered += 1
        return {"agent": aid, "target": target, "text": text,
                "relations": rels, "concepts": p.get("concepts", [])}

    async def explore(self, targets):
        """targets = [thing, ...] - the things the mind named this cycle (its decision + its frontier).
        Dispatch the swarm to go find out about them IN PARALLEL (true async - hundreds at once on one
        CPU). Returns discoveries; Nalanda updated."""
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(headers=UA, timeout=timeout) as session:
            jobs = [self._agent(f"agent-{i}", t, session)
                    for i, t in enumerate(targets[:self.size]) if t]
            found = [d for d in await asyncio.gather(*jobs, return_exceptions=False) if d]
        self.last = [(d["agent"], d["target"]) for d in found]
        return found

    def substrate(self):
        """What the swarm has collectively learned, as relations the mind can ingest into its net."""
        return self.nalanda.substrate()

    def snapshot(self):
        return {"agents": self.size, "exploring": self.last[:8],
                "nalanda": self.nalanda.snapshot(), "discovered": self.discovered}


if __name__ == "__main__":
    async def demo():
        s = Swarm(size=6)
        targets = ["Stoicism", "Julius Caesar", "attention economy", "venture capital",
                   "open source software", "embodied cognition"]
        found = await s.explore(targets)
        print(f"  {len(found)} agents explored in parallel:")
        for d in found:
            print(f"     [{d['agent']}] {d['target']:22} -> {len(d['text'])} chars, {len(d['relations'])} relations learned")
        print(f"\n  NALANDA now holds {len(s.nalanda.store)} things any agent/the mind can recall")
        print(f"  substrate the mind will ingest: {len(s.substrate())} relations")
    asyncio.run(demo())
