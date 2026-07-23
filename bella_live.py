"""
BELLA, LIVE. She reads the live web, thinks, and the surface shows every step.

    python bella_live.py       ->  open http://localhost:8080  and watch her think.

One process: a tiny static server for the surface (in a thread) + her cognition loop writing her
real state to surface/state.json each cycle. On the host, run this under systemd and point a domain
at it. Read-only; outbound-to-people stays human-gated.
"""
import asyncio, os, threading, http.server, socketserver

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE = os.path.join(HERE, "surface")
STATE = os.path.join(SURFACE, "state.json")


def _serve(port=8080):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=SURFACE, **k)
        def end_headers(self):
            self.send_header("Cache-Control", "no-store"); super().end_headers()
        def log_message(self, *a): pass
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), H) as s:
        s.serve_forever()


async def main():
    from bella import Bella
    from bella_web import fetch_hn_headlines, fetch_article
    from bella_state import emit
    b = Bella()
    b._surface_path = STATE
    print("[LIVE] Bella opening her eyes to the web...")
    while True:                                        # forever: forage the web, then think across it
        try:
            heads = fetch_hn_headlines(14)
            fed = 0
            for h in heads:
                if b.curious_about(h["title"]) >= 0.20:
                    b.feed(fetch_article(h["url"]) or h["title"]); fed += 1
            print(f"[LIVE] read {fed} of {len(heads)} live stories; thinking...")
        except Exception as e:
            print(f"[LIVE] web hiccup: {e}")

        # CLEAN BELLA: her own cognition ONLY - perceive -> curiosity -> Praxis glass-box -> thought.
        # NOT the full Eros pipeline (no rapport / attachment-seeking / 9.7k-token expression call).
        for _ in range(20):
            focus = b._curiosity_focus()               # her real curiosity picks what to attend to
            b._focus = focus
            try:
                b._praxis_decide(focus, relevant_facts=[])   # her provable reasoning + her own words
            except Exception as e:
                print(f"[LIVE] think hiccup: {e}")
            emit(b, STATE)                             # stream it to the surface
            d = getattr(b, "_last_decision", {}) or {}
            print(f"[{focus[:40]!r}] -> {str(d.get('conclusion',''))[:90]}")
            await asyncio.sleep(6)


if __name__ == "__main__":
    threading.Thread(target=_serve, daemon=True).start()
    print("Bella surface -> http://localhost:8080")
    asyncio.run(main())
