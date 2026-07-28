"""
BELLA'S SWARM - her senses and hands, made real (the kimi/Ravana structure, running).

Two modes — both run agents IN PARALLEL on one async event loop:

  explore(targets)  — broad curiosity: each agent fetches one topic, deposits into Nalanda.
                      The mind named the targets; agents just go find out.

  act(plan)         — targeted action: the LLM planner assigned each agent a specific
                      primitive + intent. Agents execute any bella_legs primitive, not just
                      search. Results deposit into Nalanda the same way.

  apex mind  ── names targets / planner assigns ──►  SWARM (agents work in parallel)
      ▲                                                      │  each deposits into Nalanda
      │                                                      ▼
      └────────────────── ingests it all  ◄──────────  NALANDA (shared, persistent)
"""
import asyncio
import aiohttp
from collective_memory import CollectiveMemory
from bella_legs import search_web_async, UA
from bella_perception import perceive


class Swarm:
    def __init__(self, size: int = 30):
        self.size    = size
        self.nalanda = CollectiveMemory()
        self.last    = []                   # [(agent_id, target), ...] — last explore cycle
        self.assignments = []               # [(agent_id, primitive, intent), ...] — last act cycle
        self.discovered  = 0

    # ------------------------------------------------------------------ explore (unchanged)
    async def _agent(self, aid: str, target, session):
        text = await search_web_async(session, str(target))
        if not text:
            return None
        p    = perceive(text)
        rels = p.get("relations", [])
        assoc = p.get("associations", [])
        if rels or assoc:
            key = (p["concepts"][0] if p.get("concepts") else target)
            self.nalanda.remember(key=str(key), content=text[:220], salience=0.6,
                                  relations=[(a, b) for a, b, _w, _k in rels] + assoc)
        self.discovered += 1
        return {"agent": aid, "target": target, "text": text,
                "relations": rels, "concepts": p.get("concepts", [])}

    async def explore(self, targets):
        """Broad curiosity — each agent fetches one topic in parallel. Nalanda updated."""
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(headers=UA, timeout=timeout) as session:
            jobs  = [self._agent(f"agent-{i}", t, session)
                     for i, t in enumerate(targets[:self.size]) if t]
            found = [d for d in await asyncio.gather(*jobs, return_exceptions=False) if d]
        self.last = [(d["agent"], d["target"]) for d in found]
        return found

    # ------------------------------------------------------------------ act (new)
    async def _agent_act(self, aid: str, step: dict, session, primitives: dict):
        """One targeted agent. Executes whatever primitive the planner assigned it."""
        prim_name = step.get("primitive", "")
        fn        = primitives.get(prim_name)
        if fn is None:
            return None
        args = {k: v for k, v in step.get("args", {}).items() if v not in ("", None)}
        try:
            out     = await fn(session, **args)
            # extract text so we can deposit into Nalanda regardless of output type
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = out.get("content") or out.get("text") or out.get("title") or ""
            elif isinstance(out, list) and out:
                text = " ".join(x.get("snippet", x.get("title", "")) for x in out[:3])
            else:
                text = ""
            if text:
                p   = perceive(text)
                key = p["concepts"][0] if p.get("concepts") else prim_name
                self.nalanda.remember(key=str(key), content=text[:220], salience=0.7,
                                      relations=[(a, b)
                                                 for a, b, _w, _k in p.get("relations", [])])
            self.discovered += 1
            return {"agent": aid, "primitive": prim_name,
                    "intent": step.get("intent", ""), "result": out, "success": True}
        except Exception as e:
            return {"agent": aid, "primitive": prim_name,
                    "intent": step.get("intent", ""), "error": str(e), "success": False}

    async def act(self, plan: list) -> list:
        """Targeted action — the planner assigned each step to an agent. All run IN PARALLEL.
        Each step must have complete args (no PREV references — parallel steps are independent).
        Returns list of {agent, primitive, intent, result, success}. Nalanda updated."""
        from bella_legs import PRIMITIVES   # lazy import — avoids circular at module level
        timeout = aiohttp.ClientTimeout(total=35)
        async with aiohttp.ClientSession(headers=UA, timeout=timeout) as session:
            jobs    = [self._agent_act(
                            step.get("agent", f"agent-{i}"),
                            step, session, PRIMITIVES)
                       for i, step in enumerate(plan[:self.size])]
            results = [r for r in await asyncio.gather(*jobs, return_exceptions=False) if r]
        self.assignments = [(r["agent"], r["primitive"], r.get("intent","")) for r in results]
        return results

    # ------------------------------------------------------------------ shared
    def substrate(self):
        return self.nalanda.substrate()

    def snapshot(self):
        nalanda_snap = self.nalanda.snapshot()
        return {
            "size":        self.size,
            "last_explore": self.last[:12],         # [(agent_id, topic), ...]
            "last_act":     self.assignments[:12],  # [(agent_id, primitive, intent), ...]
            "nalanda_size": (nalanda_snap.get("size") or len(getattr(self.nalanda, "store", {}))
                             if isinstance(nalanda_snap, dict) else 0),
            "discovered":   self.discovered,
        }


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
