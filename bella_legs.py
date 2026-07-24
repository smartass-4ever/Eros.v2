"""
BELLA'S LEGS - the bridge from her decision to the world, and the discovery back.

Praxis lands on a THING to go find out about (not a verb - a decision is just a direction of
attention). The legs go fetch it from the live web and bring back real prose. That discovery
re-enters her perception -> she learns typed relations from it -> the curiosity loop closes and TURNS,
instead of her re-chewing her own graph.

    Praxis lands on a thing  ->  legs go find it in the world  ->  a real discovery
      ->  back into perception (b.feed)  ->  next cycle she reasons on NEW material, not her echo.

One source (Wikipedia) stands in for the open web today; the swarm calls search_web_async directly on
the thing, so swapping in many real sources is a change here alone - the interface (a thing in, prose
out) stays identical.
"""
import aiohttp

UA = {"User-Agent": "BellaMind/1.0 (autonomous research agent; +https://bella-mind.fly.dev)"}
WIKI = "https://en.wikipedia.org/w/api.php"

_CACHE = {}          # thing -> text, so many agents don't re-fetch or hammer the source (politeness)


async def search_web_async(session, query, chars=1600):
    """Go find out about a thing: web search -> best article -> its intro prose (rich + learnable).
    Async - no thread limit, so hundreds of agents run at once on one CPU. Cached to stay polite."""
    query = str(query).replace("_", " ").strip()
    if not query:
        return None
    if query in _CACHE:
        return _CACHE[query]
    try:
        async with session.get(WIKI, params={"action": "query", "list": "search", "srsearch": query,
                                             "format": "json", "srlimit": 1}) as r:
            s = await r.json(content_type=None)
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        async with session.get(WIKI, params={"action": "query", "prop": "extracts", "exintro": 1,
                                             "explaintext": 1, "titles": title, "format": "json"}) as r:
            e = await r.json(content_type=None)
        for p in e.get("query", {}).get("pages", {}).values():
            txt = p.get("extract", "")
            if txt and len(txt) > 120:
                out = f"{title}. {txt[:chars]}"
                if len(_CACHE) < 800:
                    _CACHE[query] = out
                return out
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import asyncio

    async def demo():
        async with aiohttp.ClientSession(headers=UA) as session:
            for thing in ["Stoicism", "closed_labs", "embodiment", "venture capital"]:
                d = await search_web_async(session, thing)
                print(f"  {thing:16} -> {(d or 'nothing found')[:100]}")
    asyncio.run(demo())
