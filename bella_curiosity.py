"""
BELLA'S ACTION SPACE - and why this file is nearly empty now.

There used to be a whole second decision system here: decide_next_action, an argmax over a fixed menu
of seven verbs, plus a learner that reinforced situation->verb "affordances". All of it is gone,
because it contradicted Praxis and it contradicted how minds actually work.

A decision is not a verb + an object chosen from a list. It is a DIRECTION OF ATTENTION - the one
thing you're pulled toward and know least. Bella's next step is read straight off Praxis's own
decision (see Bella._next_step): curiosity is the force in that spread, and the move is just the node
it lands on. She goes to find out about that thing. The "how" was never a separate choice - a richer
pursuit ("the evidence against X", "the author's other work") is simply a more specific THING to go
toward, which is itself a node. Verbs collapse into targets. Like a baby: no action-menu, just drives
over a growing knowledge net, orienting to the salient new thing.

So all that remains here is the definition of the action/epistemic-verb nodes, kept only so
_next_step can tell her own reasoning-machinery apart from things in the world worth going toward.
"""


def action_nodes(net):
    """The nodes that are Bella's own epistemic verbs (explore / read_more / find_evidence / ...),
    identified structurally as whatever an 'affords' edge points to. Not a hardcoded list - read off
    the net. _next_step uses this ONLY to skip them: her own verbs are machinery, not world-things she
    goes and learns about. Everything else in her net is fair game as a thing to go toward."""
    acts = set()
    for edges in net.edges.values():
        for dst, _w, kind in edges:
            if kind == "affords":
                acts.add(dst)
    return acts


if __name__ == "__main__":
    from reasoning_core import KnowledgeNet
    from bella_knowledge import seed_bella_mind
    net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
    print("epistemic-verb nodes (skipped as targets, read off the net - not a constant):")
    print("  ", sorted(action_nodes(net)))
    print("\neverything else she knows is a thing she can be curious about and go toward.")
    print("which one she picks is Praxis's call (Bella._next_step), curiosity as the force. One brain.")
