# Eros

**A psyche for machines.** *(research prototype)*

Eros is a cognitive architecture that wraps a language model in a persistent
interior — memory, emotional state, opinions, and a background inner life — so
an assistant can relate to a person across time instead of resetting every
conversation. It is an exploration of what a "mind layer" for human-facing
machines could look like.

This is early, single-author research code, not a product. The sections below
are honest about what works today and what is still exploratory.

## What actually runs today

- **Local text and voice chat** (`python run.py`) — a conversational loop with a
  language model at its core, orchestrated through the cognitive pipeline below.
- **Persistent memory** — conversations and extracted facts are stored in SQLite
  and survive restarts.
- **A safety layer** — crisis (self-harm / suicide) and harmful-content messages
  are detected and short-circuit the pipeline with a caring response and crisis
  resources, before any other logic runs. Covered by tests.
- **A game-theoretic decision core** (`core/game_theory_decision.py`) — the most
  developed subsystem. It weighs competing motives (memory, curiosity, warmth,
  wit, beliefs) under veto rules that enforce self-control, and separates *who
  Eros is* (a stable personality) from *how it responds* in a given moment.
- **A constitutional belief set** (`core/eros_beliefs.py`) — always-active values
  that act as a reasoning lens rather than being re-derived each turn.
- **Background inner life** — imagination, memory consolidation ("REM"), and
  self-reflection modules that run around the main loop.

## See it for yourself (the interior is measurable)

Two self-contained tools make the claims above checkable, not rhetorical. Both
run offline with no API key (the decision core is deterministic without one):

**1. Watch the interior over a conversation.** Every turn writes a structured
snapshot — emotional control signal, the motive competition from the decision
core (which of memory/curiosity/warmth/wit/beliefs won, what was vetoed), and
where the emotion numbers came from — to `data/interior_trace.jsonl`. Read it:

```bash
python tools/interior_report.py
```

**2. Is emotion actually a control signal?** A controlled perturbation: run the
decision core twice on identical inputs, changing only the emotional state
(intact vs. clamped to neutral), and measure whether the chosen motive changes.

```bash
python tools/emotion_ablation.py
```

On the sample set, clamping emotion re-routes the decision on the emotionally
loaded turns while leaving turns dominated by other signals (a joke, a belief
question) unchanged — i.e. emotion is a low-dimensional control input, not a
label applied after the fact.

**3. One intention, broadcast to speech and action.** A common agent failure is
saying it will do something and then not doing it — because "what to say" and
"what to do" are decided separately. Eros forms intention once: a single
`SynthesizedContext` object, from which both the spoken response and the action
decision are read.

```bash
python tools/coherence_demo.py
```

It drives the real orchestrator and shows, per turn, the one intention object
and its two projections (a weather question executes an action; "play jazz" is
honestly blocked pending device setup; an emotional message stays speech-only) —
so speech and action structurally cannot diverge.

## Honest architecture note

Eros is an **orchestration layer over a language model**, not a from-scratch
mind. Some subsystems are genuinely developed (the decision core, memory,
safety); many others are heuristic scaffolding — keyword and threshold logic
that shapes prompts and routing. The interesting claim is the *composition*: how
memory, emotion, motive, and belief are combined into each response. Treat the
subsystem count as breadth of exploration, not depth of each part.

## Setup

```bash
pip install -r requirements.txt        # core deps; voice/screen groups are optional
cp .env.example .env                   # then add your LLM API key
python run.py                          # text chat
python run.py --voice                  # push-to-talk voice
python run.py --wake                   # always-on "Hey Eros"
```

An LLM API key (Groq / Together / Mistral) is required for full responses; without
one, Eros falls back to built-in patterns. See `.env.example`.

**Runtime profile.** `EROS_PROFILE=sharp python run.py` runs a tighter build that
disables non-load-bearing background modules (the consciousness-metric counters,
REM consolidation, imagination, self-reflection, the aggregated self-hub, and the
proactive scheduler) while keeping everything that shapes the response — the
decision core, emotion, memory, safety, telemetry, curiosity, personality, and
beliefs. Nothing is deleted; the default `full` profile restores everything. See
`core/eros_profile.py`.

## Repository

- `core/` — cognitive pipeline: orchestration, emotion, curiosity, the decision
  core, personality, and beliefs.
- `memory/` — relationship-aware memory that consolidates over time.
- `self model/` — the inner life: imagination, introspection, subconscious
  processing, self-reflection.
- `saftey/` — crisis and harmful-content detection (wired into the pipeline).
- `attached_assets/` — a small **sample** conversation dataset; the full training
  corpus is not published.
- `tests/` — automated tests (safety gate, interior telemetry, hybrid emotion
  appraisal, and the emotion-ablation property). Each file runs standalone:
  `python tests/test_safety.py` (or use `pytest`).
- `tools/` — `interior_report.py` and `emotion_ablation.py` (see Demonstrations).
- `API/`, `future/` — **experimental / non-functional** surfaces (a multi-tenant
  API and a Discord runtime) that depend on components not included here.

## Status

Active, early-stage research. Interfaces and internals change frequently.
Provided as a work in progress, not a supported product.
