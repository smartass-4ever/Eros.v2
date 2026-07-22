"""
BELLA LLM CACHE - a transparent cache at the HTTP boundary.

Twists Mnemon's idea (cache LLM calls, SQLite-backed, cross-agent, big token cut) onto Bella's
WHOLE call cascade with ONE integration: it patches requests.post, so EVERY LLM call site
(expression, opinion, game-theory, intent, appraisal, fact-extraction, ...) is cached without
touching Eros core. In a swarm the store is SHARED on disk: the first Bella to make a given call
pays the tokens; every other Bella hits the cache for free.

Keying is exact-normalized (SAFE: only reuses a byte-identical request, never a wrong answer).
Bella's REAL Mnemon slots in at `semantic_lookup()` for fuzzy/semantic matching (more hits) when
it's available - the interface is the same (look up by content, store by content).
"""
import hashlib, json, sqlite3, time, threading

_ENDPOINT_MARK = "chat/completions"
_CACHE = None


class LLMCache:
    def __init__(self, path="bella_llm_cache.db"):
        self.path = path
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.passthrough = 0
        self.tokens_saved = 0
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE IF NOT EXISTS cache "
                    "(k TEXT PRIMARY KEY, body TEXT, tokens INTEGER, t REAL)")
        con.commit(); con.close()

    def key(self, payload: dict) -> str:
        """Exact-normalized key: model + (role, stripped content) per message."""
        msgs = payload.get("messages", []) or []
        norm = [{"role": m.get("role"), "content": (m.get("content") or "").strip()} for m in msgs]
        blob = json.dumps({"model": payload.get("model"), "messages": norm},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # --- swap point for REAL Mnemon: replace these two with Mnemon's semantic get/put ---
    def semantic_lookup(self, payload: dict):
        return self.get(self.key(payload))

    def get(self, k: str):
        con = sqlite3.connect(self.path)
        row = con.execute("SELECT body, tokens FROM cache WHERE k=?", (k,)).fetchone()
        con.close()
        return row

    def put(self, k: str, body: str, tokens: int):
        with self._lock:
            con = sqlite3.connect(self.path)
            con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                        (k, body, tokens, time.time()))
            con.commit(); con.close()

    def stats(self) -> dict:
        total = self.hits + self.misses
        rate = round(self.hits / total * 100, 1) if total else 0.0
        return {"hits": self.hits, "misses": self.misses, "passthrough": self.passthrough,
                "hit_rate_pct": rate, "tokens_saved": self.tokens_saved}


class _CachedResponse:
    """Quacks like requests.Response enough for every caller here (.status_code/.json()/.text)."""
    def __init__(self, body: str):
        self.status_code = 200
        self.text = body
        self._body = body
    def json(self):
        return json.loads(self._body)


def install_llm_cache(path="bella_llm_cache.db", verbose=True):
    """Patch requests.post so all LLM chat calls route through the cache. Idempotent."""
    global _CACHE
    import requests
    if getattr(requests, "_bella_cache_installed", False):
        return _CACHE
    _CACHE = LLMCache(path)
    real_post = requests.post

    def cached_post(url, *a, **k):
        payload = k.get("json")
        if not (isinstance(url, str) and _ENDPOINT_MARK in url
                and isinstance(payload, dict) and payload.get("messages")):
            _CACHE.passthrough += 1
            return real_post(url, *a, **k)                  # not an LLM chat call - passthrough
        row = _CACHE.semantic_lookup(payload)
        if row:                                             # HIT - cached answer, zero tokens
            _CACHE.hits += 1
            _CACHE.tokens_saved += row[1] or 0
            return _CachedResponse(row[0])
        resp = real_post(url, *a, **k)                      # MISS - one real call, then store
        try:
            if resp.status_code == 200:
                usage = resp.json().get("usage", {}) or {}
                _CACHE.put(_CACHE.key(payload), resp.text, usage.get("total_tokens", 0))
        except Exception:
            pass
        _CACHE.misses += 1
        return resp

    requests.post = cached_post
    requests._bella_cache_installed = True
    if verbose:
        print(f"[BELLA-CACHE] installed at HTTP boundary -> {path} (every LLM call cached, swarm-shared)")
    return _CACHE


def cache_stats():
    return _CACHE.stats() if _CACHE else {"note": "cache not installed"}


# ------------------------------------------------------------------ self-test (no quota needed)
if __name__ == "__main__":
    import requests, os
    db = "bella_cache_selftest.db"
    if os.path.exists(db):
        os.remove(db)

    # a FAKE llm endpoint so we can prove hit/miss with zero real tokens
    calls = {"n": 0}
    class _Fake:
        status_code = 200
        def __init__(self, content):
            self._c = content
        @property
        def text(self):
            return json.dumps({"choices": [{"message": {"content": self._c}}],
                               "usage": {"total_tokens": 9700}})
        def json(self):
            return json.loads(self.text)
    def fake_post(url, *a, **k):
        calls["n"] += 1
        return _Fake(f"llm answer #{calls['n']}")
    requests.post = fake_post

    cache = install_llm_cache(db)
    url = "https://api.groq.com/openai/v1/chat/completions"
    req = lambda msg: {"model": "llama-3.3-70b-versatile",
                       "messages": [{"role": "user", "content": msg}]}

    print("\n-- 3 Bellas ask the SAME thing (swarm), then 1 asks something NEW --")
    r1 = requests.post(url, json=req("is open-source AI winning?"))   # miss  -> real call #1
    r2 = requests.post(url, json=req("is open-source AI winning?"))   # HIT
    r3 = requests.post(url, json=req("is open-source AI winning?"))   # HIT
    r4 = requests.post(url, json=req("what is Stoic virtue?"))        # miss  -> real call #2

    print("  r1:", r1.json()["choices"][0]["message"]["content"])
    print("  r2:", r2.json()["choices"][0]["message"]["content"], "(should equal r1)")
    print("  r3:", r3.json()["choices"][0]["message"]["content"], "(should equal r1)")
    print("  r4:", r4.json()["choices"][0]["message"]["content"])
    print("  real LLM calls actually made:", calls["n"], "(of 4 requests)")
    print("  cache stats:", cache.stats())
    os.remove(db)
