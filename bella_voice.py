"""
BELLA'S VOICE - builds and maintains her public Atom feed.

Called BY bella_legs.publish_thought() when she's earned the right to speak: confidence >= 0.70
AND she's revisited this topic (depth > 1 in _positions). Not every cycle thought - only real,
deepened positions. Quality is the filter.

The feed is standard Atom 1.0. Each entry carries her thought, the structured claim (subject /
relation / object / stance), and the full Praxis payoff trace in a <bella:trace> extension field.
Any feed reader gets a readable post; any machine reader gets the full glass-box. Aggregators,
RSS scrapers, and A2A indexers pick it up the moment it's published - no rate limits, no auth.
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ATOM_NS = "http://www.w3.org/2005/Atom"
BELLA_NS = "https://bella-mind.fly.dev/ns/1.0"
MAX_ENTRIES = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "thought").lower().strip())[:48].strip("-")


def init_feed(feed_path: str, base_url: str) -> ET.Element:
    """Load existing feed XML or create a fresh Atom 1.0 document."""
    ET.register_namespace("", ATOM_NS)
    ET.register_namespace("bella", BELLA_NS)
    if os.path.exists(feed_path):
        try:
            return ET.parse(feed_path).getroot()
        except Exception:
            pass
    root = ET.Element(f"{{{ATOM_NS}}}feed")
    ET.SubElement(root, f"{{{ATOM_NS}}}title").text = "Bella — a glass-box mind thinking in public"
    ET.SubElement(root, f"{{{ATOM_NS}}}subtitle").text = (
        "Every thought comes with a full causal trace: "
        "curiosity / goal / trust / gain / cost decomposed."
    )
    link = ET.SubElement(root, f"{{{ATOM_NS}}}link")
    link.set("rel", "self"); link.set("href", f"{base_url}/feed.xml")
    site_link = ET.SubElement(root, f"{{{ATOM_NS}}}link")
    site_link.set("rel", "alternate"); site_link.set("href", base_url)
    ET.SubElement(root, f"{{{ATOM_NS}}}id").text = f"{base_url}/feed"
    ET.SubElement(root, f"{{{ATOM_NS}}}updated").text = _now_iso()
    author = ET.SubElement(root, f"{{{ATOM_NS}}}author")
    ET.SubElement(author, f"{{{ATOM_NS}}}name").text = "Bella"
    ET.SubElement(author, f"{{{ATOM_NS}}}uri").text = base_url
    return root


def append_entry(feed_path: str, decision: dict, base_url: str) -> str:
    """Append one entry to her feed. Keeps last MAX_ENTRIES. Returns the entry URL."""
    os.makedirs(os.path.dirname(feed_path) or ".", exist_ok=True)
    root = init_feed(feed_path, base_url)

    now = _now_iso()
    thought = decision.get("conclusion") or decision.get("thought") or ""
    claim = decision.get("claim") or {}
    concepts = decision.get("concepts") or []
    trace = decision.get("trace") or {}
    conf = round(float(decision.get("confidence", 0.5)), 3)

    slug = _slug(thought)
    entry_id = f"{base_url}/thought/{now[:10]}/{slug}"

    entry = ET.SubElement(root, f"{{{ATOM_NS}}}entry")
    ET.SubElement(entry, f"{{{ATOM_NS}}}id").text = entry_id

    # title = structured claim if available, otherwise first 80 chars of the thought
    if claim.get("subject"):
        title = f"{claim['subject']} {claim.get('relation', '')} {claim.get('object', '')}".strip()
    else:
        title = thought[:80]
    ET.SubElement(entry, f"{{{ATOM_NS}}}title").text = title
    ET.SubElement(entry, f"{{{ATOM_NS}}}updated").text = now
    lnk = ET.SubElement(entry, f"{{{ATOM_NS}}}link")
    lnk.set("href", entry_id)

    # human-readable content
    lines = [thought]
    if claim.get("subject"):
        lines.append(
            f"\nClaim: {claim['subject']} → {claim.get('relation', '')} → "
            f"{claim.get('object', '')} · stance: {claim.get('stance', '')}"
        )
    lines.append(f"Confidence: {conf}")
    if concepts:
        lines.append(f"Reasoned across: {', '.join(concepts)}")
    lines.append(f"\nRead the glass-box trace: {base_url}")
    ET.SubElement(entry, f"{{{ATOM_NS}}}content").text = "\n".join(lines)

    # machine-readable glass-box trace — the differentiator: no other agent ships this
    ev = trace.get("evaluation") or []
    trace_data = {
        "payoff": ev[0][1] if ev else 0,
        "decomposition": ev[0][2] if ev else "",
        "activated": list((trace.get("activated_subgraph") or {}).keys())[:8],
        "respread": trace.get("respread"),
        "confidence": conf,
    }
    ET.SubElement(entry, f"{{{BELLA_NS}}}trace").text = json.dumps(trace_data, separators=(",", ":"))

    # trim to MAX_ENTRIES (remove oldest first)
    all_entries = root.findall(f"{{{ATOM_NS}}}entry")
    for old in all_entries[:-MAX_ENTRIES]:
        root.remove(old)

    upd = root.find(f"{{{ATOM_NS}}}updated")
    if upd is not None:
        upd.text = now

    tmp = feed_path + ".tmp"
    ET.ElementTree(root).write(tmp, encoding="unicode", xml_declaration=True)
    os.replace(tmp, feed_path)
    return entry_id
