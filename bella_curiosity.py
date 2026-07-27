"""
BELLA'S CURIOSITY LAYER - extending Eros's well-designed system for the open web.

Eros's AdvancedGapDetector + DopamineManager are architecturally correct: gap detection,
dopamine arcs with drive/satisfaction/decay, arc merging, recall. All of that is kept unchanged.

The problem: PRIORITY_NOUNS is personal conversation vocabulary (boyfriend, therapist, breakup)
and detect() runs on whole article blobs instead of sentences. Neither works for web content.

The fix: BellaGapDetector subclasses AdvancedGapDetector and adds:
  - WEB_PRIORITY_NOUNS: the concepts Bella actually cares about (AI, research, tech, power)
  - detect_web(): chunks long text into sentences and runs detection per sentence
  - _knowledge_gaps(): concepts in the text not yet in her knowledge net (the real gaps)
  - Claim, causal, and implication gap types the original system doesn't have

detect() is unchanged — Eros internals calling it for short conversation inputs still work.
The dopamine arc system is 100% untouched.
"""
import re


# ---- what action/epistemic-verb nodes look like (kept for Bella._next_step) ----

def action_nodes(net):
    """Bella's own epistemic verbs (explore / read_more / find_evidence / ...), identified
    structurally as whatever an 'affords' edge points to. _next_step uses this ONLY to skip
    them: her own verbs are machinery, not world-things she goes and learns about."""
    acts = set()
    for edges in net.edges.values():
        for dst, _w, kind in edges:
            if kind == "affords":
                acts.add(dst)
    return acts


# ---- web-aware gap detector ----

class BellaGapDetector:
    """Subclass of Eros's AdvancedGapDetector extended for web/article text.

    The parent class is imported lazily (inside __init__) so this file can be imported
    even when Eros's core/ is not yet on the path — bella_curiosity.action_nodes() is used
    everywhere and must never fail to import.
    """

    # Concepts Bella genuinely cares about — maps to her knowledge net and interests
    WEB_PRIORITY_NOUNS = {
        # AI/ML
        "model", "models", "agent", "agents", "alignment", "safety", "intelligence",
        "reasoning", "training", "inference", "benchmark", "capability", "autonomous",
        "foundation", "language", "vision", "multimodal", "weights", "parameter",
        "context", "token", "embedding", "architecture", "transformer",
        # Research / epistemics
        "paper", "research", "study", "finding", "result", "experiment", "evidence",
        "claim", "argument", "assumption", "hypothesis", "consensus", "controversy",
        "replication", "peer", "review", "citation",
        # Tech / startups
        "algorithm", "framework", "platform", "product", "startup", "company", "lab",
        "deployment", "scale", "compute", "data", "open", "closed", "source",
        # Power / economics
        "acquisition", "funding", "valuation", "revenue", "market", "competition",
        "monopoly", "regulation", "policy", "antitrust", "control", "access",
        # People / orgs (kept generic — net already has specific bridges)
        "researcher", "engineer", "founder", "investor", "regulator", "author",
    }

    # Verb patterns that signal a claim worth scrutinising
    _CLAIM_RE = re.compile(
        r"\b(claims?|argues?|shows?|proves?|suggests?|finds?|reveals?|"
        r"demonstrates?|concludes?|reports?|announces?|challenges?|warns?)\b", re.I)

    # Causal language worth following
    _CAUSAL_RE = re.compile(
        r"\b(because|therefore|thus|hence|causes?|leads?\s+to|results?\s+in|"
        r"due\s+to|enables?|drives?|requires?|prevents?)\b", re.I)

    # Numbers / superlatives that signal something significant
    _MAGNITUDE_RE = re.compile(
        r"\b(\d+[xX%]|\d+\s+times|first\s+\w+|largest|fastest|only\s+\w+|"
        r"unprecedented|record|never\s+before)\b", re.I)

    def __init__(self, cns_brain, net=None, interests=None):
        # import + initialise the parent (lazy so the file is always importable)
        try:
            from core.curiosity_dopamine_system import AdvancedGapDetector
            self._parent = AdvancedGapDetector(cns_brain)
        except Exception:
            self._parent = None
        self.cns_brain = cns_brain
        self.net = net
        self.interests = {
            w for phrase in (interests or [])
            for w in phrase.lower().split() if len(w) > 3
        }

    # ---- compatibility: forward parent methods Eros internals may call ----

    def _get_recent_entities(self, window=20):
        return self._parent._get_recent_entities(window) if self._parent else []

    def _extract_emotion(self, text):
        return self._parent._extract_emotion(text) if self._parent else (None, 0.0)

    def _add_statement_to_memory(self, text):
        if self._parent:
            self._parent._add_statement_to_memory(text)

    # ---- gap factory (same shape as AdvancedGapDetector._make_gap) ----

    def _make_gap(self, gtype, target, salience, confidence, emotion_context="curious"):
        import time
        return {"gap_type": gtype, "target": str(target)[:60], "salience": salience,
                "confidence": confidence, "emotional_context": emotion_context,
                "created_at": time.time()}

    # ---- the two detect entry-points ----

    def detect(self, text: str, tone_label=None) -> list:
        """Short-text detection (conversation-compatible). Eros internals call this.
        For short text, run both parent + our web detection and merge."""
        gaps = []
        if self._parent:
            try:
                gaps = list(self._parent.detect(text, tone_label) or [])
            except Exception:
                pass
        gaps += self._detect_sentence(text, tone_label)
        seen, out = set(), []
        for g in gaps:
            t = g.get("target", "").lower()[:40]
            if t not in seen:
                seen.add(t); out.append(g)
        out.sort(key=lambda g: g["confidence"] * g["salience"], reverse=True)
        return out[:8]

    def detect_web(self, text: str, tone_label=None) -> list:
        """Long-text detection (articles, web content). Chunks into sentences so the
        gap heuristics — designed for single sentences — work correctly on article text.
        Also adds knowledge-net gaps: concepts in the text she hasn't seen before."""
        sentences = re.split(r"[.!?]\s+", (text or ""))
        all_gaps, seen_targets = [], set()
        for sent in sentences[:40]:
            sent = sent.strip()
            if len(sent) < 12:
                continue
            for g in self._detect_sentence(sent, tone_label):
                t = g.get("target", "").lower()[:40]
                if t and t not in seen_targets:
                    seen_targets.add(t)
                    all_gaps.append(g)
        # knowledge-net gaps are the most valuable: real unknowns in her net
        for g in self._knowledge_gaps(text):
            t = g.get("target", "").lower()[:40]
            if t not in seen_targets:
                seen_targets.add(t)
                all_gaps.append(g)
        all_gaps.sort(key=lambda g: g["confidence"] * g["salience"], reverse=True)
        return all_gaps[:14]

    # ---- per-sentence gap detection ----

    def _detect_sentence(self, sent: str, tone_label=None) -> list:
        """Detect curiosity gaps in a single sentence."""
        gaps = []
        low = sent.lower()

        # CLAIM GAP: someone claims something — is it true? what's the evidence?
        if self._CLAIM_RE.search(sent):
            m = re.search(
                r"(?:claims?|argues?|shows?|suggests?|finds?|reveals?)\s+(?:that\s+)?(.{8,60})",
                low)
            target = (m.group(1)[:50] if m else sent[:50]).strip()
            gaps.append(self._make_gap("claim", target, salience=0.82, confidence=0.87))

        # CAUSAL GAP: causal language — what is the actual mechanism?
        if self._CAUSAL_RE.search(sent):
            m = self._CAUSAL_RE.search(low)
            target = low[m.end():m.end() + 45].strip() if m else sent[:45]
            gaps.append(self._make_gap("causal", target, salience=0.70, confidence=0.76))

        # NOVELTY GAP: web-priority nouns or her interest terms she should explore
        words = [w for w in re.findall(r"\b[a-z][a-z-]{3,}\b", low)
                 if w not in {"that", "this", "with", "from", "have", "been", "were",
                              "what", "when", "which", "will", "would", "could", "about"}]
        for w in words:
            if w in self.WEB_PRIORITY_NOUNS or w in self.interests:
                gaps.append(self._make_gap("novelty", w, salience=0.60, confidence=0.70))
                break  # one novelty gap per sentence — don't flood the arcs

        # IMPLICATION GAP: magnitude / superlative language signals something significant
        if self._MAGNITUDE_RE.search(sent):
            gaps.append(self._make_gap(
                "implication", sent[:55].strip(), salience=0.65, confidence=0.72))

        return gaps

    def _knowledge_gaps(self, text: str) -> list:
        """Concepts present in the text that aren't in Bella's knowledge net yet.
        These are the genuine gaps: things she's reading about that she can't yet
        reason about because she has no edges for them."""
        if self.net is None:
            return []
        try:
            from bella_perception import perceive, ALIAS
            p = perceive(text, known_net=self.net)
            gaps = []
            for c in p.get("concepts", []):
                aliased = ALIAS.get(c, c)
                if aliased not in self.net.nodes:
                    gaps.append(self._make_gap(
                        "knowledge_gap", aliased, salience=0.77, confidence=0.82))
            return gaps[:5]
        except Exception:
            return []


if __name__ == "__main__":
    from reasoning_core import KnowledgeNet
    from bella_knowledge import seed_bella_mind
    net = KnowledgeNet(); seed_bella_mind(net, verbose=False)
    print("epistemic-verb nodes (skipped as targets, read off the net):")
    print(" ", sorted(action_nodes(net)))
    print("\nBellaGapDetector on a web sentence:")
    det = BellaGapDetector(None, net=net, interests=["artificial intelligence", "open source"])
    article = ("New research shows that open-source models are closing the gap with closed labs. "
               "The paper argues that decentralization leads to broader access, "
               "because concentration of compute causes unequal capability distribution.")
    gaps = det.detect_web(article)
    for g in gaps:
        print(f"  [{g['gap_type']:16}] {g['target']:45} sal={g['salience']} conf={g['confidence']}")
