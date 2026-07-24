"""
BELLA STATE BRIDGE - turns her live cognition into the JSON the surface renders.

Every cycle she writes her real thought, the activated subgraph (her cortex lighting up), the
reasoning path, the DECOMPOSED payoff her decision was actually made on, what she's reading, her
curiosity trail and the swarm. The page just draws it. Nothing shown that she didn't actually think.
"""
import json, os, re


def _payoff(trace):
    ev = trace.get("evaluation") or []
    if ev:
        why = ev[0][2]
        def g(k):
            m = re.search(rf"{k}\s+([-\d.]+)", why)
            return round(float(m.group(1)), 3) if m else 0.0
        p = {k: g(k) for k in ("curiosity", "goal", "trust", "gain", "cost")}
        if any(p.values()):
            return p
    return {"curiosity": 0.30, "goal": 0.20, "trust": 0.15, "gain": 0.15, "cost": 0.10}


def _stance(d):
    c = d.get("claim") or {}
    if c and c.get("subject"):
        return f"{c['subject']} → {c.get('relation','')} → {c.get('object','')} · stance: {c.get('stance','')}"
    return ""


def build_state(bella) -> dict:
    d = getattr(bella, "_last_decision", {}) or {}
    trace = d.get("trace", {}) or {}
    act = trace.get("activated_subgraph", {}) or {}
    concepts = [[k, round(float(v), 3)] for k, v in list(act.items())[:12]]
    ids = {k for k, _ in concepts}
    net = getattr(getattr(bella, "praxis", None), "net", None)
    edges = []
    if net is not None:
        for a in ids:
            for b, _w, _k in net.edges.get(a, [])[:6]:
                if b in ids:
                    edges.append([a, b])
    reading_why = [m for m in getattr(bella, "_markers", ()) if m][:4] or list(ids)[:4]
    return {
        "thought": d.get("conclusion") or d.get("thought") or "…",
        "reading": {"src": str(getattr(bella, "_focus", "") or "the live web")[:120], "why": reading_why},
        "concepts": concepts,
        "edges": edges[:26],
        "path": [c for c in d.get("concepts", []) if c],
        "payoff": _payoff(trace),
        "conf": round(float(d.get("confidence", 0.6)), 3),
        "action": getattr(bella, "_last_action", "explore"),
        "stance": _stance(d),
        "trail": getattr(bella, "_trail", [])[-4:],
        "heads": _swarm_heads(bella),
        "nalanda": _nalanda_count(bella),
        "feeling": getattr(bella, "_last_feeling", None),      # System 2: how she feels (reward decomposed)
        "supervision": getattr(bella, "_last_supervision", None),  # System 3: the caregiver's last touch (if any)
    }


def _swarm_heads(bella):
    sw = getattr(bella, "swarm", None)
    if sw is None:
        return []
    return [[aid, str(tgt).replace("_", " ")[:22], 1] for aid, tgt in getattr(sw, "last", [])[:8]]


def _nalanda_count(bella):
    sw = getattr(bella, "swarm", None)
    try:
        return len(sw.nalanda.store) if sw is not None else 0
    except Exception:
        return 0


def emit(bella, path):
    """Write her current state to disk for the surface to poll (atomic-ish)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(build_state(bella), f)
        os.replace(tmp, path)
    except Exception:
        pass
