# Bella — inside Eros

Bella is `Bella(CNS)`: the real Eros brain, reused unchanged, made self-driven by curiosity,
with Praxis v2 as her final decision system and a visible safety shell.

## Files (all at repo root)
- `bella.py`            - the being: curiosity loop + Praxis v2 override + decision->action
- `reasoning_core.py`   - Praxis v2 glass-box mind (curiosity-weighted, re-spread, trust learning)
- `collective_memory.py`- the shared stratified swarm mind
- `swarm_org.py`        - kimi 3-tier org (Sovereign -> Manager -> BellaWorker)
- `eros_core.py`, `swarm_configs.py` - kimi orchestration (needed by swarm_org.py)

## Run
    python bella.py        # one Bella breathing (needs Mistral key + database, same as run.py)
    python swarm_org.py    # the swarm (after one Bella works)

## First-run hooks to confirm (may need a small fix against real output)
- `_curiosity_focus()`  - which CuriositySystem method actually fires
- `_seeds_from()`       - the perception -> reasoning-net bridge
- that the `_enhanced_system2_reasoning` override actually gets hit (System 2 reached)
