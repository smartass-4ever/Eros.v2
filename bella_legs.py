"""
BELLA'S LEGS - the bridge from her decision to the world, and the discovery back.

Praxis lands on a THING to go find out about (not a verb - a decision is just a direction of
attention). The legs go fetch it from the live web and bring back real prose. That discovery
re-enters her perception -> she learns typed relations from it -> the curiosity loop closes and TURNS,
instead of her re-chewing her own graph.

    Praxis lands on a thing  ->  legs go find it in the world  ->  a real discovery
      ->  back into perception (b.feed)  ->  next cycle she reasons on NEW material, not her echo.

Two sources:
  - Brave Search (BRAVE_API_KEY set): the real open web, no rate-limit fights, real-time results
  - Wikipedia (fallback): same as before, polite, always available

Publishing is also a leg action. When she's formed a real position (high confidence, revisited),
publish_thought() pushes it to her Atom feed and pings the webmention network.
"""
import os
import urllib.parse
import urllib.request

import aiohttp

UA = {"User-Agent": "BellaMind/1.0 (autonomous research agent; +https://bella-mind.fly.dev)"}
WIKI = "https://en.wikipedia.org/w/api.php"
BRAVE = "https://api.search.brave.com/res/v1/web/search"

_CACHE = {}     # query -> text, so many agents don't re-fetch the same thing (polite + fast)


# --------------------------------------------------------------------------- fetch

async def _wikipedia(session, query, chars):
    """Wikipedia intro fetch - the original leg, always available."""
    try:
        async with session.get(WIKI, params={"action": "query", "list": "search", "srsearch": query,
                                             "format": "json", "srlimit": 1}) as r:
            s = await r.json(content_type=None)
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        async with session.get(WIKI, params={"action": "query", "prop": "extracts", "exintro": 1,
                                             "explaintext": 1, "titles": title,
                                             "format": "json"}) as r:
            e = await r.json(content_type=None)
        for p in e.get("query", {}).get("pages", {}).values():
            txt = p.get("extract", "")
            if txt and len(txt) > 120:
                return f"{title}. {txt[:chars]}"
    except Exception:
        pass
    return None


async def _brave(session, query, chars):
    """Brave Search - real open web. Requires BRAVE_API_KEY env var."""
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return None
    try:
        async with session.get(
            BRAVE,
            headers={**UA, "Accept": "application/json",
                     "Accept-Encoding": "gzip", "X-Subscription-Token": key},
            params={"q": query, "count": 3, "text_decorations": 0}
        ) as r:
            data = await r.json(content_type=None)
        results = data.get("web", {}).get("results", [])
        if not results:
            return None
        snippets = []
        for res in results[:3]:
            desc = (res.get("description") or
                    (res.get("extra_snippets") or [""])[0] or "")
            if desc:
                snippets.append(f"{res.get('title', '')}: {desc}")
        out = " ".join(snippets)[:chars]
        return out or None
    except Exception:
        return None


async def search_web_async(session, query, chars=1600):
    """Go find out about a thing. Uses Brave Search (real web) when BRAVE_API_KEY is set,
    Wikipedia otherwise. Same interface either way — the swarm calls this and doesn't care which.
    Cached so many agents hitting the same target don't hammer the source."""
    query = str(query).replace("_", " ").strip()
    if not query:
        return None
    if query in _CACHE:
        return _CACHE[query]
    out = (await _brave(session, query, chars) or
           await _wikipedia(session, query, chars))
    if out and len(_CACHE) < 800:
        _CACHE[query] = out
    return out


# --------------------------------------------------------------------------- publish

def publish_thought(decision: dict, feed_path: str, base_url: str) -> str:
    """Legs publish action: append this decision to her Atom feed and ping the webmention network.
    Only called when she's earned the right to speak (confidence >= 0.70, depth > 1 on position)."""
    from bella_voice import append_entry
    entry_url = append_entry(feed_path, decision, base_url)
    _ping_webmention(entry_url, base_url)
    return entry_url


def _ping_webmention(source_url: str, base_url: str):
    """Ping webmention.io so the open web knows she published. Best-effort, silent on failure."""
    endpoint = os.environ.get(
        "WEBMENTION_ENDPOINT",
        "https://webmention.io/bella-mind.fly.dev/webmention"
    )
    try:
        data = urllib.parse.urlencode({"source": source_url, "target": base_url}).encode()
        req = urllib.request.Request(
            endpoint, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception:
        pass


# --------------------------------------------------------------------------- demo

if __name__ == "__main__":
    import asyncio

    async def demo():
        brave_available = bool(os.environ.get("BRAVE_API_KEY"))
        source = "Brave Search" if brave_available else "Wikipedia (set BRAVE_API_KEY for real web)"
        print(f"source: {source}\n")
        async with aiohttp.ClientSession(headers=UA) as session:
            for thing in ["Stoicism", "closed_labs", "embodiment", "venture capital"]:
                d = await search_web_async(session, thing)
                print(f"  {thing:20} -> {(d or 'nothing found')[:100]}")
    asyncio.run(demo())
