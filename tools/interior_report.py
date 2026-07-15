"""
interior_report.py — read an interior-telemetry trace and summarize the
agent's inner state over a conversation.

The telemetry layer (core/interior_telemetry.py) writes one JSON object per
turn. This tool reads that log and prints a compact, readable report of the
interior: the emotional control signal over time and which motive won each
turn. It answers "what was going on inside the agent, and did the interior
actually drive the response?" without needing any plotting library.

Usage:
    python tools/interior_report.py [path-to-trace.jsonl]
    # default path: data/interior_trace.jsonl

If matplotlib is installed, also writes PNG plots next to the trace
(--plots to force, --no-plots to skip).
"""

import os
import sys
import json
from collections import Counter

# Make the report printable on a plain Windows console (matches run.py).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PLAYERS = ("memory", "curiosity", "warmth", "wit", "beliefs")
DEFAULT_PATH = os.path.join("data", "interior_trace.jsonl")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _bar(value, lo, hi, width=20):
    """A simple horizontal bar for a value in [lo, hi]."""
    if hi == lo:
        frac = 0.5
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def _valence_track(v):
    """Center-anchored track for valence in [-1, 1]."""
    half = 10
    slot = int(round(max(-1.0, min(1.0, v)) * half))
    cells = ["·"] * (2 * half + 1)
    center = half
    cells[center] = "|"
    pos = center + slot
    cells[pos] = "●"
    return "".join(cells)


def report(rows):
    if not rows:
        print("No telemetry rows found. Run a conversation with EROS_TELEMETRY=1 first.")
        return

    print("=" * 78)
    print(f"  INTERIOR TELEMETRY REPORT  —  {len(rows)} turns")
    print("=" * 78)

    # --- Emotional control signal over time ---
    print("\nEMOTIONAL CONTROL SIGNAL (valence, -1 ◀ 0 ▶ +1):\n")
    print(f"  {'turn':>4}  {'valence':>7}  {'arous':>5}  track")
    for r in rows:
        v = r.get("valence", 0.0)
        a = r.get("arousal", 0.5)
        print(f"  {r.get('turn',0):>4}  {v:>7.2f}  {a:>5.2f}  {_valence_track(v)}  {r.get('emotion','')}")

    # --- Motive competition: which player won each turn ---
    print("\nMOTIVE COMPETITION (game-theory decision core):\n")
    print(f"  {'turn':>4}  {'primary':>9}  {'secondary':>9}  {'conf':>4}  vetoed")
    winners = Counter()
    have_scores = False
    for r in rows:
        p = r.get("primary_player") or "—"
        s = r.get("secondary_player") or "—"
        c = r.get("decision_confidence", 0.0)
        vetoed = ",".join(r.get("vetoed_players", [])) or "—"
        if r.get("primary_player"):
            winners[r["primary_player"]] += 1
        if r.get("player_scores"):
            have_scores = True
        print(f"  {r.get('turn',0):>4}  {p:>9}  {s:>9}  {c:>4.2f}  {vetoed}")

    if winners:
        print("\n  Winning motive, over the conversation:")
        for player, n in winners.most_common():
            print(f"    {player:>9}: {_bar(n, 0, len(rows))}  {n}/{len(rows)}")

    # --- Per-turn player score detail (the low-dimensional drive vector) ---
    if have_scores:
        print("\nPLAYER SCORE VECTOR PER TURN (the drive that produced the choice):\n")
        header = "  turn  " + "  ".join(f"{p[:4]:>5}" for p in PLAYERS)
        print(header)
        for r in rows:
            ps = r.get("player_scores") or {}
            cells = "  ".join(f"{ps.get(p, 0.0):>5.2f}" for p in PLAYERS)
            print(f"  {r.get('turn',0):>4}  {cells}")

    # --- Response mode + load summary ---
    print("\nRESPONSE MODE DISTRIBUTION:")
    modes = Counter(r.get("response_mode") or "—" for r in rows)
    for mode, n in modes.most_common():
        print(f"    {mode:>18}: {n}")

    safety = sum(1 for r in rows if r.get("safety_intervention"))
    if safety:
        print(f"\n  Safety interventions: {safety}")

    src = Counter(r.get("emotion_source") or "unknown" for r in rows)
    print("\nEMOTION SIGNAL PROVENANCE (where the numbers come from):")
    for s, n in src.most_common():
        print(f"    {s}: {n}")
    print()


def maybe_plots(rows, path, force=False, skip=False):
    if skip or not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if force:
            print("[plots] matplotlib not installed — skipping PNGs.")
        return

    turns = [r.get("turn", i) for i, r in enumerate(rows)]
    base = os.path.splitext(path)[0]

    # 1) Emotion trajectory
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(turns, [r.get("valence", 0.0) for r in rows], marker="o", label="valence")
    ax.plot(turns, [r.get("arousal", 0.5) for r in rows], marker="s", label="arousal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Emotional control signal over the conversation")
    ax.set_xlabel("turn"); ax.set_ylabel("value"); ax.legend(); fig.tight_layout()
    fig.savefig(base + "_emotion.png", dpi=120)

    # 2) Player-score heatmap
    if any(r.get("player_scores") for r in rows):
        import numpy as np
        M = np.array([[(r.get("player_scores") or {}).get(p, 0.0) for p in PLAYERS] for r in rows]).T
        fig, ax = plt.subplots(figsize=(9, 3.2))
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(PLAYERS))); ax.set_yticklabels(PLAYERS)
        ax.set_xlabel("turn"); ax.set_title("Motive (player) scores per turn")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(base + "_players.png", dpi=120)

    print(f"[plots] wrote {base}_emotion.png" + (" and _players.png" if any(r.get('player_scores') for r in rows) else ""))


def main(argv):
    args = [a for a in argv if not a.startswith("-")]
    path = args[0] if args else DEFAULT_PATH
    force = "--plots" in argv
    skip = "--no-plots" in argv
    if not os.path.exists(path):
        print(f"Trace not found: {path}")
        print("Run Eros with EROS_TELEMETRY=1 (default) to generate one, "
              "or pass a path: python tools/interior_report.py <trace.jsonl>")
        return 1
    rows = load(path)
    report(rows)
    maybe_plots(rows, path, force=force, skip=skip)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
