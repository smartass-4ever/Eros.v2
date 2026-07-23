"""
BELLA ON THE INTERNET - her eyes open.

Stage 1 of release: she READS the live web (starting with Hacker News - her world: AI, startups,
tech), perceives it, reasons, forms opinions, learns from it, and follows her curiosity across it.

READ-ONLY by design at this stage. Outbound engagement with real people stays HUMAN-GATED (her
locked rule, and what protects her). She watches and learns in the open first.
"""
import re
import requests

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"


def fetch_hn(n=6):
    """Real, current stories from Hacker News (no auth). Returns text blobs she can read."""
    out = []
    try:
        ids = requests.get(HN_TOP, timeout=15).json() or []
    except Exception as e:
        print(f"[WEB] could not reach Hacker News: {e}")
        return out
    for i in ids[: n * 4]:
        try:
            it = requests.get(HN_ITEM.format(i), timeout=15).json() or {}
        except Exception:
            continue
        title = it.get("title")
        if title:
            body = re.sub("<[^>]+>", " ", it.get("text", "") or "")
            by = it.get("by", "")
            out.append(f"{title}. {body} (posted by {by})".strip())
        if len(out) >= n:
            break
    return out


async def bella_reads_the_web(ticks=6):
    """One Bella, reading the live internet."""
    from bella import Bella
    print("[WEB] opening her eyes to the live web (Hacker News)...")
    stories = fetch_hn(ticks)
    print(f"[WEB] fetched {len(stories)} live stories\n")
    for s in stories[:3]:
        print(f"   • {s[:90]}")
    print()
    b = Bella()
    for s in stories:
        b.feed(s)
    await b.live(ticks=ticks + 2)   # read the stories, then follow her curiosity from them


def swarm_reads_the_web(heads=3, per_head=3):
    """THE SWARM on the internet: N heads each read a different slice of the live web, learn from it,
    and pool what they learn through Nalanda - so what one head reads, all the heads come to know."""
    from bella_ravana import RavanaSwarm
    swarm = RavanaSwarm(heads=heads, agents_per_head=50)
    stories = fetch_hn(heads * per_head)
    print(f"[WEB] {len(stories)} live stories, sharded across {heads} heads\n")
    for i, head in enumerate(swarm.heads):
        shard = stories[i * per_head:(i + 1) * per_head]
        learned = 0
        for s in shard:
            p = head.read(s)                         # read + ingest relations (learn)
            learned += len(p["relations"])
            if p["concepts"]:                        # share a learning to the collective
                head.nalanda.broadcast(head.hid, p["concepts"][:3], 0.5, " ".join(p["concepts"][:3]))
        print(f"   {head.hid}: read {len(shard)} stories, learned {learned} relations")
    for head in swarm.heads:                         # every head studies what the others read
        head.study()
    shared = len(swarm.nalanda.memory.store)
    print(f"\n[NALANDA] the swarm's shared mind now holds {shared} things any head can recall")
    return swarm


if __name__ == "__main__":
    # connectivity check first
    print("checking the wire to the live web...")
    s = fetch_hn(3)
    if s:
        print(f"OK - {len(s)} live stories reached:")
        for t in s:
            print("   •", t[:100])
    else:
        print("no connection to the live web from here.")
