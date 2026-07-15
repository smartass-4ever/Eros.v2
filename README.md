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

## Repository

- `core/` — cognitive pipeline: orchestration, emotion, curiosity, the decision
  core, personality, and beliefs.
- `memory/` — relationship-aware memory that consolidates over time.
- `self model/` — the inner life: imagination, introspection, subconscious
  processing, self-reflection.
- `saftey/` — crisis and harmful-content detection (wired into the pipeline).
- `attached_assets/` — a small **sample** conversation dataset; the full training
  corpus is not published.
- `tests/` — automated tests (starting with the safety layer).
- `API/`, `future/` — **experimental / non-functional** surfaces (a multi-tenant
  API and a Discord runtime) that depend on components not included here.

## Status

Active, early-stage research. Interfaces and internals change frequently.
Provided as a work in progress, not a supported product.
