"""
BELLA SWARM - arranged as your kimi 3-tier orchestration (eros_core.py), REUSING your real
classes. Runs in an env with both the kimi repo and Eros on the path.

  Tier 1  SovereignOrchestrator  - one shared strategy (BeliefRegistry), receives pulses only
  Tier 2  EROSManager            - spawns/monitors a capped cluster, PAD telemetry, pulses up
  Tier 3  BellaWorker            - a full Bella being (curiosity + Praxis v2 mind + action)

Taming the hell: strategy flows DOWN, pulses flow UP (no log-flood at the top); each Manager
caps its cluster; PAD flags failures early; replication is EARNED (credible-engagement reward).

The Sovereign's beliefs ENCODE the guardrail: reward = quality + credible engagement, never
raw volume; act on a controlled/observable surface, never flood public platforms.

Override points (confirm against a real run): InternAgent / EROSManager constructor args,
and EROSManager.spawn_intern (we make it spawn a BellaWorker).
"""
from eros_core import SovereignOrchestrator, EROSManager, InternAgent, PADAnalyzer
from swarm_configs import SwarmConfig
from bella import Bella
from collective_memory import CollectiveMemory


# ---------------------------------------------------------------- Tier 3: Bella as worker
class BellaWorker(InternAgent):
    """A full Bella being wearing the InternAgent interface, so a Manager can run + monitor it."""
    def __init__(self, agent_id: str, specialty: str, manager_id: str):
        super().__init__(agent_id, specialty, manager_id)
        self.being = Bella()                        # the real curiosity + glass-box + action being
        self.reward = 0.0                           # cumulative credible-engagement (piece 3)

    async def execute_task(self, task: dict, belief_context: dict) -> dict:
        # align to the swarm's shared strategy, then run ONE Bella cognitive cycle
        if belief_context.get("goal"):
            self.being.goal = set(belief_context["goal"])
        self.being._focus = task.get("focus") or self.being._curiosity_focus()
        result = await self.being.process_input(
            user_input=self.being._focus, conversation_history=[],
            user_id=self.agent_id, context={"swarm": True, **belief_context})
        await self.being._act_on()                  # decision -> action, safely (piece 2)
        dec = getattr(self.being, "_last_decision", {}) or {}
        out = {
            "agent": self.agent_id,
            "focus": self.being._focus,
            "conclusion": dec.get("conclusion"),
            "confidence": dec.get("confidence", 0.0),   # PAD reads this as quality
            "text": result.get("response") or result.get("text"),
        }
        self._output_buffer = getattr(self, "_output_buffer", [])
        self._output_buffer.append(out)                 # PADAnalyzer reads recent outputs
        return out

    def credit(self, engagement):
        """piece 3: inbound serious engagement, scored credibility x depth, becomes reward."""
        self.reward += engagement.get("credibility", 0) * engagement.get("depth", 0)


# ---------------------------------------------------------------- Tier 2: manager of Bellas
class BellaManager(EROSManager):
    """EROSManager that spawns BellaWorkers and REPLICATES the ones that earn credible respect."""
    REPLICATE_AT = 1.0                              # reward threshold to earn a clone

    def spawn_intern(self, specialty: str) -> BellaWorker:
        if len(self.interns) >= self.max_interns:    # cap = sandbox bound
            raise ValueError(f"Manager {self.manager_id} at capacity")
        w = BellaWorker(agent_id=f"{self.manager_id}-bella-{len(self.interns)+1}",
                        specialty=specialty, manager_id=self.manager_id)
        self.interns.append(w)
        return w

    def replicate_earners(self):
        """A Bella that has earned enough credible engagement gets cloned - growth on quality."""
        for w in list(self.interns):
            if getattr(w, "reward", 0) >= self.REPLICATE_AT and len(self.interns) < self.max_interns:
                child = self.spawn_intern(w.specialty)
                child.being.goal = set(w.being.goal)     # inherit the successful strategy
                w.reward = 0.0                           # reset; child must earn its own


# ---------------------------------------------------------------- Tier 1 + assembly
class BellaSwarm:
    GUARDRAIL = {
        "reward_on": "quality_and_credible_engagement",   # NEVER raw volume/attention
        "surface": "controlled_observable",               # never flood public platforms
        "honesty": "glass_box_required",
    }

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        constraints = {**self.config._default_constraints(), **self.GUARDRAIL}
        self.sovereign = SovereignOrchestrator(project_id="bella", global_constraints=constraints)
        self.managers: list[BellaManager] = []
        self.memory = CollectiveMemory()            # the shared mind - all Bellas read+write it

    def add_cluster(self, size: int, specialty: str = "explore") -> BellaManager:
        m = self.sovereign.create_manager()         # returns EROSManager; we treat as BellaManager
        m.__class__ = BellaManager                  # adopt the Bella behaviors (spawn/replicate)
        for _ in range(size):
            m.spawn_intern(specialty)
        self.managers.append(m)
        return m

    async def turn(self, task: dict):
        belief = self.sovereign.belief_registry.to_dict()
        substrate = self.memory.substrate()             # the shared mind's durable knowledge
        for m in self.managers:
            for w in m.interns:
                w.being.praxis.net.ingest(substrate)    # each Bella starts from the collective mind
                out = await w.execute_task(task, belief)
                await m.monitor_intern(w)               # Tier 2 watches (PAD) -> pulse if needed
                dec = getattr(w.being, "_last_decision", {}) or {}   # write learnings back
                concepts = dec.get("concepts", [])
                if dec.get("conclusion"):
                    rels = [(concepts[i], concepts[i + 1]) for i in range(len(concepts) - 1)]
                    self.memory.remember(key=dec["conclusion"], content=dec["conclusion"],
                                         salience=dec.get("confidence", 0.5), relations=rels)
            m.replicate_earners()                       # quality earns clones
        pulses = [m.get_status_pulse() for m in self.managers]
        return self.sovereign.make_strategic_decision(pulses)   # Tier 1 steers, on pulses only


if __name__ == "__main__":
    import asyncio
    async def demo():
        swarm = BellaSwarm()
        swarm.add_cluster(size=3, specialty="explore")
        decision = await swarm.turn({"focus": "what is true in the world right now"})
        print("sovereign strategic decision:", decision)
    asyncio.run(demo())
