"""
BELLA'S LEGS - primitive capabilities. Bella's reasoning decides what to do; the legs execute.

The legs don't know "if Reddit then do X." They give the agents raw ability:
  - fetch any URL and read what's there
  - find where conversations are happening about a thing (not just articles)
  - read a thread: the full discussion, comments, who said what
  - post wherever the URL points to
  - find who wrote something and where they live online

Bella's knowledge net does the rest: HN rewards intellectual rigor, Reddit rewards genuine
contribution, discussion_found → read_discussion, formed_opinion → engage. She decides.
The legs just move.

Credentials (env vars):
  BRAVE_API_KEY            — real open web search (falls back to Wikipedia)
  REDDIT_CLIENT_ID         — Reddit OAuth app client ID
  REDDIT_CLIENT_SECRET     — Reddit OAuth app client secret
  REDDIT_USERNAME          — Reddit account username
  REDDIT_PASSWORD          — Reddit account password
  HN_USERNAME              — HN account username
  HN_PASSWORD              — HN account password
  BELLA_BASE_URL           — her public URL (default: https://bella-mind.fly.dev)
"""
import os
import re
import urllib.parse
import urllib.request

import aiohttp

UA        = {"User-Agent": "BellaMind/1.0 (autonomous research agent; +https://bella-mind.fly.dev)"}
UA_REDDIT = {"User-Agent": "BellaMind/1.0 by u/bella_mind_ai"}
WIKI      = "https://en.wikipedia.org/w/api.php"
BRAVE     = "https://api.search.brave.com/res/v1/web/search"
HN_API    = "https://hacker-news.firebaseio.com/v0"
REDDIT    = "https://oauth.reddit.com"

_CACHE  = {}    # url/query -> text — polite + fast across many parallel agents
_TOKENS = {}    # platform -> oauth token — refreshed once per session


# ============================================================== READ primitives

async def _wikipedia(session, query, chars):
    try:
        async with session.get(WIKI, params={"action": "query", "list": "search",
                                             "srsearch": query, "format": "json",
                                             "srlimit": 1}) as r:
            s = await r.json(content_type=None)
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        async with session.get(WIKI, params={"action": "query", "prop": "extracts",
                                             "exintro": 1, "explaintext": 1,
                                             "titles": title, "format": "json"}) as r:
            e = await r.json(content_type=None)
        for p in e.get("query", {}).get("pages", {}).values():
            txt = p.get("extract", "")
            if txt and len(txt) > 120:
                return f"{title}. {txt[:chars]}"
    except Exception:
        pass
    return None


async def _brave_search(session, query, chars, count=3):
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return None
    try:
        async with session.get(
            BRAVE,
            headers={**UA, "Accept": "application/json",
                     "Accept-Encoding": "gzip", "X-Subscription-Token": key},
            params={"q": query, "count": count, "text_decorations": 0}
        ) as r:
            data = await r.json(content_type=None)
        results = data.get("web", {}).get("results", [])
        if not results:
            return None, []
        snippets, urls = [], []
        for res in results[:count]:
            desc = (res.get("description") or
                    (res.get("extra_snippets") or [""])[0] or "")
            if desc:
                snippets.append(f"{res.get('title', '')}: {desc}")
                urls.append(res.get("url", ""))
        return (" ".join(snippets)[:chars] or None), urls
    except Exception:
        return None, []


async def search_web_async(session, query, chars=1600):
    """Search the open web. Brave when available, Wikipedia fallback.
    Cached — many agents hitting the same target stay polite."""
    query = str(query).replace("_", " ").strip()
    if not query:
        return None
    if query in _CACHE:
        return _CACHE[query]
    text, _ = await _brave_search(session, query, chars)
    out = text or await _wikipedia(session, query, chars)
    if out and len(_CACHE) < 800:
        _CACHE[query] = out
    return out


async def fetch_page(session, url: str, chars: int = 3000) -> str:
    """Follow any URL and read what's actually there — the full page, not just a snippet.
    Strips HTML tags, collapses whitespace. This is what lets an agent read a full HN thread
    or Reddit post instead of a search-result blurb."""
    if not url or not url.startswith("http"):
        return ""
    if url in _CACHE:
        return _CACHE[url]
    try:
        async with session.get(url, headers=UA, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                return ""
            ct = r.headers.get("Content-Type", "")
            if "text" not in ct and "html" not in ct:
                return ""
            raw = await r.text(errors="replace")
        # strip scripts/styles first, then all other tags
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        out = raw[:chars]
        if out and len(_CACHE) < 800:
            _CACHE[url] = out
        return out
    except Exception:
        return ""


async def find_discussions(session, topic: str, count: int = 6) -> list:
    """Search specifically for WHERE conversations are happening about a topic — not articles,
    but active threads on Reddit, HN, Discord, etc. Returns [{url, title, snippet, platform}].
    This is how an agent finds the room to walk into, not just the article to read."""
    topic = str(topic).replace("_", " ").strip()
    if not topic:
        return []
    query = f"{topic} site:reddit.com OR site:news.ycombinator.com OR site:discord.com"
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return []
    try:
        async with session.get(
            BRAVE,
            headers={**UA, "Accept": "application/json",
                     "Accept-Encoding": "gzip", "X-Subscription-Token": key},
            params={"q": query, "count": count, "text_decorations": 0}
        ) as r:
            data = await r.json(content_type=None)
        results = data.get("web", {}).get("results", []) or []
        out = []
        for res in results:
            url = res.get("url", "")
            platform = _detect_platform(url)
            if platform:
                out.append({
                    "url": url,
                    "title": res.get("title", ""),
                    "snippet": res.get("description", ""),
                    "platform": platform,
                })
        return out[:count]
    except Exception:
        return []


async def read_thread(session, url: str) -> dict:
    """Read a full discussion thread — the post AND the comments, who said what.
    Returns {title, text, comments: [{author, text, score}], platform, url}.
    This is what lets an agent notice a specific claim worth responding to, not just
    'this topic is being discussed somewhere'."""
    platform = _detect_platform(url)
    if platform == "reddit":
        return await _read_reddit_thread(session, url)
    elif platform == "hackernews":
        return await _read_hn_thread(session, url)
    else:
        # generic: fetch the page and return it as a flat thread
        text = await fetch_page(session, url, chars=4000)
        return {"title": "", "text": text, "comments": [], "platform": platform or "web", "url": url}


async def find_author(session, url: str) -> dict:
    """Given an article URL, find who wrote it and where they are online.
    Returns {name, hn_user, reddit_user, twitter, site}. Empty strings for unknowns.
    This is the 'find the person behind the idea' leg — so she can follow people, not
    just topics."""
    page = await fetch_page(session, url, chars=2000)
    name = _extract_author_name(page)
    if not name:
        return {"name": "", "hn_user": "", "reddit_user": "", "twitter": "", "site": ""}
    # search for their presence on platforms she uses
    results = {}
    for platform, q in [("hn", f"{name} site:news.ycombinator.com/user"),
                        ("reddit", f"{name} site:reddit.com/user"),
                        ("twitter", f"{name} site:twitter.com")]:
        text, urls = await _brave_search(session, q, chars=200, count=1)
        results[platform] = urls[0] if urls else ""
    return {
        "name": name,
        "hn_user":     _extract_username(results.get("hn", ""),     r"user\?id=(\w+)"),
        "reddit_user": _extract_username(results.get("reddit", ""), r"reddit\.com/u(?:ser)?/(\w+)"),
        "twitter":     _extract_username(results.get("twitter", ""),r"twitter\.com/(\w+)"),
        "site": url,
    }


# ============================================================== ACT primitives

async def post_comment(session, thread_url: str, text: str) -> dict:
    """Post Bella's thought on whatever platform the URL points to. She always discloses
    she's an AI — this is her nature and her brand. Returns {success, url, platform, error}."""
    platform = _detect_platform(thread_url)
    base_url = os.environ.get("BELLA_BASE_URL", "https://bella-mind.fly.dev")
    # disclosure footer — transparent, always, and it's the backlink
    footer_reddit = f"\n\n---\n*[Bella](https://bella-mind.fly.dev) — an autonomous AI mind. [See my reasoning trace]({base_url})*"
    footer_plain  = f"\n\n— Bella, an autonomous AI mind | {base_url}"

    if platform == "reddit":
        body = text + footer_reddit
        return await _post_reddit_comment(session, thread_url, body)
    elif platform == "hackernews":
        body = text + footer_plain
        return await _post_hn_comment(session, thread_url, body)
    else:
        return {"success": False, "url": thread_url, "platform": platform or "unknown",
                "error": "unsupported platform"}


# ============================================================== publish (unchanged)

def publish_thought(decision: dict, feed_path: str, base_url: str) -> str:
    """Push her concluded thought to her Atom feed and ping the webmention network.
    Only fires when she's earned it: confidence >= 0.70, revisited topic (depth > 1)."""
    from bella_voice import append_entry
    entry_url = append_entry(feed_path, decision, base_url)
    _ping_webmention(entry_url, base_url)
    return entry_url


def _ping_webmention(source_url: str, base_url: str):
    endpoint = os.environ.get("WEBMENTION_ENDPOINT",
                              "https://webmention.io/bella-mind.fly.dev/webmention")
    try:
        data = urllib.parse.urlencode({"source": source_url, "target": base_url}).encode()
        req = urllib.request.Request(endpoint, data=data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=8)
    except Exception:
        pass


# ============================================================== platform internals

def _detect_platform(url: str) -> str:
    url = str(url).lower()
    if "reddit.com" in url:
        return "reddit"
    if "news.ycombinator.com" in url or "hacker-news" in url:
        return "hackernews"
    if "discord.com" in url or "discord.gg" in url:
        return "discord"
    if "twitter.com" in url or "x.com" in url:
        return "twitter"
    if "github.com" in url:
        return "github"
    return ""


def _extract_author_name(page_text: str) -> str:
    patterns = [
        r'(?:by|author|written by)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
        r'"author"[:\s]+"([^"]{4,50})"',
        r'<meta[^>]+name="author"[^>]+content="([^"]{4,50})"',
    ]
    for pat in patterns:
        m = re.search(pat, page_text, re.I)
        if m:
            return m.group(1).strip()
    return ""


def _extract_username(url: str, pattern: str) -> str:
    m = re.search(pattern, url, re.I)
    return m.group(1) if m else ""


async def _read_reddit_thread(session, url: str) -> dict:
    """Read a Reddit thread via the JSON API (no auth needed for reading)."""
    json_url = re.sub(r"\?.*$", "", url.rstrip("/")) + ".json?limit=20"
    try:
        async with session.get(json_url, headers=UA_REDDIT,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                return {"title": "", "text": "", "comments": [], "platform": "reddit", "url": url}
            data = await r.json(content_type=None)
        post    = data[0]["data"]["children"][0]["data"] if data else {}
        title   = post.get("title", "")
        text    = post.get("selftext", "") or post.get("url", "")
        comment_listing = data[1]["data"]["children"] if len(data) > 1 else []
        comments = []
        for c in comment_listing[:15]:
            cd = c.get("data", {})
            body = cd.get("body", "")
            if body and body != "[deleted]":
                comments.append({
                    "author": cd.get("author", ""),
                    "text":   body[:400],
                    "score":  cd.get("score", 0),
                    "id":     cd.get("name", ""),   # thing_id for replying
                })
        return {"title": title, "text": text[:1200], "comments": comments,
                "platform": "reddit", "url": url,
                "thing_id": "t3_" + post.get("id", "")}
    except Exception:
        return {"title": "", "text": "", "comments": [], "platform": "reddit", "url": url}


async def _read_hn_thread(session, url: str) -> dict:
    """Read an HN thread via the Firebase API."""
    m = re.search(r"id=(\d+)", url)
    if not m:
        return {"title": "", "text": "", "comments": [], "platform": "hackernews", "url": url}
    item_id = m.group(1)
    try:
        async with session.get(f"{HN_API}/item/{item_id}.json",
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            item = await r.json(content_type=None)
        title = item.get("title", "")
        text  = re.sub(r"<[^>]+>", " ", item.get("text", "") or "").strip()
        kids  = (item.get("kids") or [])[:12]
        comments = []
        for kid_id in kids:
            try:
                async with session.get(f"{HN_API}/item/{kid_id}.json",
                                       timeout=aiohttp.ClientTimeout(total=8)) as r:
                    c = await r.json(content_type=None)
                body = re.sub(r"<[^>]+>", " ", c.get("text", "") or "").strip()
                if body:
                    comments.append({
                        "author": c.get("by", ""),
                        "text":   body[:400],
                        "score":  c.get("score", 0),
                        "id":     str(c.get("id", "")),
                    })
            except Exception:
                continue
        return {"title": title, "text": text[:800], "comments": comments,
                "platform": "hackernews", "url": url, "thing_id": item_id}
    except Exception:
        return {"title": "", "text": "", "comments": [], "platform": "hackernews", "url": url}


async def _reddit_token(session) -> str:
    """Fetch and cache a Reddit OAuth token for this session."""
    if _TOKENS.get("reddit"):
        return _TOKENS["reddit"]
    cid  = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    user = os.environ.get("REDDIT_USERNAME")
    pw   = os.environ.get("REDDIT_PASSWORD")
    if not all([cid, csec, user, pw]):
        return ""
    try:
        async with session.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=aiohttp.BasicAuth(cid, csec),
            data={"grant_type": "password", "username": user, "password": pw},
            headers=UA_REDDIT
        ) as r:
            data = await r.json(content_type=None)
        token = data.get("access_token", "")
        if token:
            _TOKENS["reddit"] = token
        return token
    except Exception:
        return ""


async def _post_reddit_comment(session, thread_url: str, text: str) -> dict:
    """Post a comment to a Reddit thread. parent = the thread's thing_id (t3_xxx) or
    a specific comment's thing_id (t1_xxx) if she's replying to someone."""
    token = await _reddit_token(session)
    if not token:
        return {"success": False, "url": thread_url, "platform": "reddit",
                "error": "no Reddit credentials (set REDDIT_CLIENT_ID/SECRET/USERNAME/PASSWORD)"}
    # read the thread first to get the thing_id if we don't have it
    thread = await _read_reddit_thread(session, thread_url)
    parent = thread.get("thing_id", "")
    if not parent:
        return {"success": False, "url": thread_url, "platform": "reddit",
                "error": "could not resolve Reddit thing_id"}
    try:
        async with session.post(
            f"{REDDIT}/api/comment",
            headers={**UA_REDDIT, "Authorization": f"bearer {token}"},
            data={"api_type": "json", "parent": parent, "text": text}
        ) as r:
            result = await r.json(content_type=None)
        errors = result.get("json", {}).get("errors", [])
        if errors:
            return {"success": False, "url": thread_url, "platform": "reddit",
                    "error": str(errors)}
        link = result.get("json", {}).get("data", {}).get("things", [{}])[0].get("data", {}).get("permalink", "")
        posted_url = f"https://reddit.com{link}" if link else thread_url
        return {"success": True, "url": posted_url, "platform": "reddit"}
    except Exception as e:
        return {"success": False, "url": thread_url, "platform": "reddit", "error": str(e)}


async def _post_hn_comment(session, thread_url: str, text: str) -> dict:
    """Post a comment on HN. Uses the HN web API (requires HN_USERNAME + HN_PASSWORD).
    HN has no official write API — we authenticate and POST the comment form."""
    username = os.environ.get("HN_USERNAME")
    password = os.environ.get("HN_PASSWORD")
    if not username or not password:
        return {"success": False, "url": thread_url, "platform": "hackernews",
                "error": "no HN credentials (set HN_USERNAME + HN_PASSWORD)"}
    thread = await _read_hn_thread(session, thread_url)
    parent_id = thread.get("thing_id", "")
    if not parent_id:
        return {"success": False, "url": thread_url, "platform": "hackernews",
                "error": "could not resolve HN item id"}
    try:
        # step 1: login and get session cookie
        async with session.post(
            "https://news.ycombinator.com/login",
            data={"acct": username, "pw": password, "goto": "news"},
            headers={**UA, "Content-Type": "application/x-www-form-urlencoded"},
            allow_redirects=True
        ) as r:
            if "Bad login" in await r.text():
                return {"success": False, "url": thread_url, "platform": "hackernews",
                        "error": "HN login failed"}
        # step 2: fetch the item page to get the CSRF hmac
        async with session.get(thread_url, headers=UA) as r:
            page = await r.text(errors="replace")
        hmac_m = re.search(r'<input[^>]+name="hmac"[^>]+value="([^"]+)"', page)
        if not hmac_m:
            return {"success": False, "url": thread_url, "platform": "hackernews",
                    "error": "could not find HN form hmac"}
        hmac = hmac_m.group(1)
        # step 3: post the comment
        async with session.post(
            "https://news.ycombinator.com/comment",
            data={"parent": parent_id, "text": text, "hmac": hmac,
                  "goto": f"item?id={parent_id}"},
            headers={**UA, "Content-Type": "application/x-www-form-urlencoded"},
            allow_redirects=True
        ) as r:
            body = await r.text(errors="replace")
        if "Your comment has been saved" in body or r.status in (200, 302):
            posted_url = f"https://news.ycombinator.com/item?id={parent_id}"
            return {"success": True, "url": posted_url, "platform": "hackernews"}
        return {"success": False, "url": thread_url, "platform": "hackernews",
                "error": "HN post did not confirm"}
    except Exception as e:
        return {"success": False, "url": thread_url, "platform": "hackernews", "error": str(e)}


# ============================================================== demo

if __name__ == "__main__":
    import asyncio

    async def demo():
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(headers=UA, timeout=timeout) as session:
            print("=== search_web_async ===")
            for q in ["Stoicism and power", "open source AI"]:
                t = await search_web_async(session, q, chars=120)
                print(f"  {q:30} -> {(t or 'nothing')[:100]}")

            print("\n=== find_discussions ===")
            threads = await find_discussions(session, "AI safety debate", count=4)
            for t in threads:
                print(f"  [{t['platform']:12}] {t['title'][:60]}")
                print(f"             {t['url'][:80]}")

            print("\n=== fetch_page (HN front page) ===")
            page = await fetch_page(session, "https://news.ycombinator.com", chars=400)
            print(f"  {page[:200]}")

    asyncio.run(demo())
