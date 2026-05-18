"""Markdown projector — renders claims and entities as Markdown files.

Inspired by Basic Memory's approach: the graph is projected into a
human-readable, git-friendly Markdown tree with YAML frontmatter
carrying provenance, confidence, and source metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from noodly.models.claims import Claim, ClaimStatus

logger = logging.getLogger(__name__)


def _frontmatter(fields: dict) -> str:
    """Render a simple YAML frontmatter block."""
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, datetime):
            lines.append(f"{key}: {value.isoformat()}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _slugify(text: str) -> str:
    """Turn a string into a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = slug.replace(" ", "-")
    safe = []
    for ch in slug:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
    return "".join(safe)[:80] or "unnamed"


class MarkdownProjector:
    """Projects claims from the fact ledger into a Markdown directory tree.

    Output structure::

        brain/
        ├── entities/
        │   ├── portnet.md
        │   └── jane-smith.md
        ├── claims/
        │   ├── stable/
        │   │   └── portnet-uses-python.md
        │   ├── process/
        │   └── stateful/
        └── index.md
    """

    def __init__(self, brain_dir: Path) -> None:
        self._brain_dir = brain_dir

    def project(self, claims: list[Claim]) -> int:
        """Write claims to Markdown files. Returns number of files written."""
        if not claims:
            return 0

        entities_dir = self._brain_dir / "entities"
        claims_dir = self._brain_dir / "claims"
        entities_dir.mkdir(parents=True, exist_ok=True)

        written = 0

        # Group claims by subject to build entity pages
        by_subject: dict[str, list[Claim]] = {}
        for claim in claims:
            by_subject.setdefault(claim.subject, []).append(claim)

        # Write entity pages
        for subject, subject_claims in by_subject.items():
            slug = _slugify(subject)
            path = entities_dir / f"{slug}.md"

            excluded = (ClaimStatus.superseded, ClaimStatus.rejected)
            active = [c for c in subject_claims if c.status not in excluded]
            active.sort(key=lambda c: c.truth_score, reverse=True)

            fm = _frontmatter(
                {
                    "entity": subject,
                    "claim_count": len(active),
                    "avg_truth_score": round(
                        sum(c.truth_score for c in active) / max(len(active), 1), 3
                    ),
                    "last_updated": datetime.now(timezone.utc),
                    "sources": list(
                        {
                            str(ev.artifact_id)[:8]
                            for c in active
                            for ev in c.evidence
                        }
                    )[:10],
                }
            )

            lines = [fm, "", f"# {subject}", ""]
            for claim in active:
                score_bar = "█" * int(claim.truth_score * 10)
                lines.append(
                    f"- **[{claim.status.value}]** {claim.natural_language}  "
                    f"`score={claim.truth_score:.2f}` {score_bar}"
                )
                if claim.evidence:
                    ev = claim.evidence[0]
                    if ev.source_span:
                        span_preview = ev.source_span[:120].replace("\n", " ")
                        lines.append(f"  > _{span_preview}_")
                lines.append("")

            path.write_text("\n".join(lines))
            written += 1

        # Write per-class claim files
        for claim in claims:
            if claim.status in (ClaimStatus.superseded, ClaimStatus.rejected):
                continue
            class_dir = claims_dir / claim.knowledge_class.value
            class_dir.mkdir(parents=True, exist_ok=True)

            slug = _slugify(f"{claim.subject}-{claim.predicate}-{claim.object}")
            path = class_dir / f"{slug}.md"

            fm = _frontmatter(
                {
                    "claim_id": str(claim.id),
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "status": claim.status.value,
                    "truth_score": round(claim.truth_score, 3),
                    "confidence": claim.confidence,
                    "knowledge_class": claim.knowledge_class.value,
                    "created_at": claim.created_at,
                    "last_confirmed": claim.last_confirmed_at or claim.created_at,
                    "evidence_count": len(claim.evidence),
                }
            )

            lines = [fm, "", f"# {claim.natural_language}", ""]
            lines.append(f"**Subject:** {claim.subject}  ")
            lines.append(f"**Predicate:** {claim.predicate}  ")
            lines.append(f"**Object:** {claim.object}  ")
            lines.append("")

            if claim.evidence:
                lines.append("## Evidence")
                lines.append("")
                for ev in claim.evidence:
                    direction = "supports" if ev.supports else "contradicts"
                    lines.append(f"- [{direction}] artifact `{str(ev.artifact_id)[:8]}`")
                    if ev.author:
                        lines.append(f"  - Author: {ev.author}")
                    if ev.source_span:
                        span = ev.source_span[:200].replace("\n", " ")
                        lines.append(f"  > _{span}_")
                    lines.append("")

            path.write_text("\n".join(lines))
            written += 1

        # Write index
        self._write_index(claims)
        written += 1

        logger.info("Projected %d Markdown files to %s", written, self._brain_dir)
        return written

    def _write_index(self, claims: list[Claim]) -> None:
        """Write a top-level index.md summarizing the brain."""
        excluded = (ClaimStatus.superseded, ClaimStatus.rejected)
        active = [c for c in claims if c.status not in excluded]
        by_status: dict[str, int] = {}
        for c in active:
            by_status[c.status.value] = by_status.get(c.status.value, 0) + 1

        subjects = sorted({c.subject for c in active})

        fm = _frontmatter(
            {
                "title": "Noodly Brain Index",
                "total_claims": len(active),
                "last_projected": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", "# Noodly Brain", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total active claims:** {len(active)}")
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

        if subjects:
            lines.append("## Entities")
            lines.append("")
            for subj in subjects:
                slug = _slugify(subj)
                lines.append(f"- [{subj}](entities/{slug}.md)")
            lines.append("")

        path = self._brain_dir / "index.md"
        path.write_text("\n".join(lines))
