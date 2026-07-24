"""
BELLA'S PERSISTENCE - so she never reboots to zero.

Saves her MIND to disk on an interval and loads it on boot: the knowledge graph she's grown from
reading (concepts + typed relations + learned weights), her learned action-values, and what she's
already fetched. On Fly this path lives on a persistent VOLUME, so a restart or redeploy doesn't wipe
what she learned. She stays ONE continuous, compounding being - yesterday's reading is still hers today.
"""
import json, os
from collections import defaultdict


def save_mind(bella, path) -> int:
    """Write her whole mind to disk (atomic). Returns connection count saved."""
    try:
        net = bella.praxis.net
        data = {
            "edges": {a: [[d, round(w, 4), k] for (d, w, k) in lst] for a, lst in net.edges.items()},
            "nodes": list(net.nodes),
            "action_counts": getattr(bella.praxis, "_action_counts", {}),
            "fetched": list(getattr(bella, "_fetched", set()))[-200:],
            "trail": getattr(bella, "_trail", [])[-20:],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return sum(len(v) for v in net.edges.values())
    except Exception as e:
        print(f"[PERSIST] save failed: {e}")
        return 0


def load_mind(bella, path) -> int:
    """Restore her mind from disk over the fresh seed. Returns connection count, or 0 if none."""
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        net = bella.praxis.net
        net.edges = defaultdict(list, {a: [(d, w, k) for (d, w, k) in lst]
                                       for a, lst in data.get("edges", {}).items()})
        net.nodes = set(data.get("nodes", [])) or set(net.edges.keys())
        bella.praxis._action_counts = data.get("action_counts", {}) or {}
        bella._fetched = set(data.get("fetched", []))
        bella._trail = data.get("trail", [])
        return sum(len(v) for v in net.edges.values())
    except Exception as e:
        print(f"[PERSIST] load failed: {e}")
        return 0


if __name__ == "__main__":
    # round-trip proof: learn something, save, load into a fresh mind, confirm it survived
    from reasoning_core import PraxisV2, KnowledgeNet
    from bella_knowledge import seed_bella_mind

    class Stub:  # minimal stand-in
        pass

    a = Stub(); a.praxis = PraxisV2(KnowledgeNet()); seed_bella_mind(a.praxis.net, verbose=False)
    a.praxis.net.relate("tiktok_brainrot", "attention_collapse", 0.7, kind="leads_to", both=False)
    before = sum(len(v) for v in a.praxis.net.edges.values())
    save_mind(a, "bella_mind_test.json")

    b = Stub(); b.praxis = PraxisV2(KnowledgeNet()); seed_bella_mind(b.praxis.net, verbose=False)
    fresh = sum(len(v) for v in b.praxis.net.edges.values())
    n = load_mind(b, "bella_mind_test.json")
    survived = any(d == "attention_collapse" for d, _, _ in b.praxis.net.edges.get("tiktok_brainrot", []))
    print(f"  learned+saved: {before} conns | fresh boot: {fresh} | after load: {n} | learned edge survived: {survived}")
    os.remove("bella_mind_test.json")
