"""
BELLA, LIVE. She reads the live web, thinks, and the surface shows every step.

    python bella_live.py       ->  open http://localhost:8080  and watch her think.

One process: a tiny static server for the surface (in a thread) + her cognition loop
writing live state to surface/state.json each cycle + a background HN feeder that
continuously brings her fresh things to read. Runs forever.

Env vars:
    GROQ_API_KEY          — planner + translator
    BRAVE_API_KEY         — real web search (Wikipedia fallback without it)
    BELLA_BASE_URL        — her public URL (default: https://bella-mind.fly.dev)
    BELLA_SWARM_SIZE      — parallel agents (default: 30)
    BELLA_MIND_PATH       — persist mind across restarts
    BELLA_FEED_INTERVAL   — seconds between HN batches (default: 90)
    PORT                  — HTTP port (default: 8080)
"""
import asyncio
import os
import threading
import http.server
import socketserver

HERE    = os.path.dirname(os.path.abspath(__file__))
SURFACE = os.path.join(HERE, "surface")
STATE   = os.path.join(SURFACE, "state.json")

FEED_INTERVAL  = int(os.environ.get("BELLA_FEED_INTERVAL", "90"))
CURIOSITY_GATE = float(os.environ.get("BELLA_CURIOSITY_GATE", "0.25"))
PORT           = int(os.environ.get("PORT", "8080"))

# shared ref so the webmention handler can call reward() on the live instance
_bella_ref = None


def _serve():
    feed_path = os.path.join(SURFACE, "feed.xml")

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=SURFACE, **k)

        def do_GET(self):
            if self.path in ("/feed.xml", "/feed"):
                if os.path.exists(feed_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/atom+xml; charset=utf-8")
                    self.end_headers()
                    with open(feed_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
                return
            super().do_GET()

        def do_POST(self):
            if self.path == "/webmention":
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length).decode("utf-8", errors="replace")
                print(f"[webmention] received: {body[:200]}")
                self.send_response(202)
                self.end_headers()
                # someone on the internet linked to her — that IS recognition, wire the reward
                if _bella_ref is not None:
                    try:
                        _bella_ref.reward(+0.7)
                        print("[webmention] +0.7 reward — the world responded")
                    except Exception:
                        pass
                return
            self.send_response(405)
            self.end_headers()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as s:
        s.serve_forever()


async def _hn_feeder(bella):
    """Background task: fetch fresh HN headlines every FEED_INTERVAL seconds,
    feed the ones that grip her into her inbox."""
    from bella_web import fetch_hn_headlines, fetch_article
    while True:
        try:
            heads = fetch_hn_headlines(20)
            fed   = 0
            for h in heads:
                if bella.curious_about(h["title"]) >= CURIOSITY_GATE:
                    article = fetch_article(h.get("url", "")) or h["title"]
                    bella.feed(article)
                    fed += 1
                    if fed >= 6:
                        break
            if fed:
                print(f"[FEED] {fed} articles from HN (scanned {len(heads)})")
            else:
                print(f"[FEED] nothing gripped her — running on own curiosity")
        except Exception as e:
            print(f"[FEED] error: {e}")
        await asyncio.sleep(FEED_INTERVAL)


async def main():
    global _bella_ref
    from bella import Bella
    from bella_llm_cache import cache_stats

    b = Bella()
    _bella_ref = b

    print(f"[LIVE] Bella is awake")
    print(f"[LIVE] surface  → http://localhost:{PORT}")
    print(f"[LIVE] state    → {STATE}")
    print(f"[LIVE] swarm    → {getattr(getattr(b, 'swarm', None), 'size', 0)} agents")
    print(f"[LIVE] base url → {os.environ.get('BELLA_BASE_URL', 'https://bella-mind.fly.dev')}")
    print()

    asyncio.create_task(_hn_feeder(b))

    try:
        await b.live(pace=3.0)          # ticks=None → runs forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[LIVE] shutting down...")
    finally:
        print(f"[BELLA-CACHE] {cache_stats()}")
        if getattr(b, "_mind_path", None):
            try:
                from bella_persist import save_mind
                save_mind(b, b._mind_path)
                print(f"[LIVE] mind saved → {b._mind_path}")
            except Exception as e:
                print(f"[LIVE] mind save failed: {e}")


if __name__ == "__main__":
    threading.Thread(target=_serve, daemon=True).start()
    print(f"[LIVE] surface server → http://localhost:{PORT}")
    asyncio.run(main())
