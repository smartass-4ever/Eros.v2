"""
BELLA'S LEGS - the bridge from her decision to the world, and the discovery back.

Praxis concludes an ACTION (search_author / find_evidence / read_more / explore / trace_origin /
compare) about a target. The legs turn that conclusion into a real-world task and dispatch it. For
now ONE fetcher stands in for the swarm: it searches the live web (Wikipedia) for the target and
brings back real prose. That discovery re-enters her perception -> she learns typed relations from
it -> the curiosity loop closes and TURNS, instead of her re-chewing her own graph.

    Praxis decides  ->  legs.dispatch (go find this in the world)  ->  a real discovery
      ->  back into perception (b.feed)  ->  next cycle she reasons on NEW material, not her echo.

Swap the single fetcher for the real Ravana swarm later - the interface (dispatch a task, get a
discovery) stays identical.
"""
import requests

UA = {"User-Agent": "BellaMind/1.0 (autonomous research agent; +https://bella-mind.fly.dev)"}
WIKI = "https://en.wikipedia.org/w/api.php"


def _search_web(query, chars=1600):
    """One real fetch: Wikipedia search -> best article -> its intro prose (rich + learnable)."""
    try:
        s = requests.get(WIKI, params={"action": "query", "list": "search", "srsearch": query,
                                       "format": "json", "srlimit": 1}, headers=UA, timeout=15).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        e = requests.get(WIKI, params={"action": "query", "prop": "extracts", "exintro": 1,
                                       "explaintext": 1, "titles": title, "format": "json"},
                         headers=UA, timeout=15).json()
        for p in e.get("query", {}).get("pages", {}).values():
            txt = p.get("extract", "")
            if txt and len(txt) > 120:
                return f"{title}. {txt[:chars]}"
    except Exception:
        pass
    return None


def _query_for(action, target, topic=""):
    """Turn her concluded action + target into the query the swarm goes to find."""
    t = str(target).replace("_", " ").strip()
    ctx = str(topic).replace("_", " ").strip()
    if not t and not ctx:
        return ""
    if action in ("search_author", "follow_source"):
        return t or ctx                                  # the person / source itself
    if action == "find_evidence":
        return f"{t} evidence criticism"
    if action == "trace_origin":
        return f"{t} history origin"
    if action == "compare":
        return f"{t} {ctx}".strip()
    return f"{t} {ctx}".strip() or t                     # read_more / explore


def dispatch(action, target, topic=""):
    """LEGS: her decision -> a real-world task -> the discovery it brings back. Returns text or None."""
    q = _query_for(action, target, topic)
    if not q:
        return None
    return _search_web(q)


# ------------------------------------------------------------------ ASYNC (true parallelism for the swarm)
_CACHE = {}          # query -> text, so 30 agents don't re-fetch or hammer the source (politeness)


async def search_web_async(session, query, chars=1600):
    """Async fetch - no thread limit, so hundreds can run at once on one CPU. Cached to stay polite."""
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


async def dispatch_async(session, action, target, topic=""):
    q = _query_for(action, target, topic)
    return await search_web_async(session, q) if q else None


if __name__ == "__main__":
    for action, target, topic in [("search_author", "Julius Caesar", ""),
                                  ("find_evidence", "closed_labs", "ai"),
                                  ("read_more", "embodiment", "robots"),
                                  ("explore", "stoicism", "virtue")]:
        d = dispatch(action, target, topic)
        print(f"  {action}({target}) -> {(d or 'nothing found')[:110]}")
