"""Markdown projector — renders claims as consolidated, readable documents.

Phase 7 rewrite: produces entity-centric pages (all claims about an entity
in one file), topic pages (cross-entity grouping), source provenance pages,
and a dashboard index — all with file-to-file linking.

Output structure::

    brain/
    ├── index.md                        # Dashboard
    ├── entities/
    │   ├── portnet.md                  # Consolidated entity page
    │   └── singapore-customs.md
    ├── topics/
    │   ├── trade-compliance.md         # Cross-entity topic page
    │   └── http-protocol.md
    ├── sources/
    │   ├── customs-circular-04-2021.md # Source provenance page
    │   └── rfc-7540.md
    ├── conflicts/
    │   └── pending.md                  # Outstanding conflicts
    └── _manifest.json                  # Emission manifest
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from noodly.models.claims import Claim, ClaimStatus
from noodly.projection.planner import EmissionPlanner

logger = logging.getLogger(__name__)

_EXCLUDED = (ClaimStatus.superseded, ClaimStatus.rejected)


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
    """Projects claims into consolidated, readable Markdown documents.

    Produces four types of pages:
    - **Entity pages**: all claims about an entity, grouped by topic
    - **Topic pages**: cross-entity grouping of related claims
    - **Source pages**: provenance — what was extracted from each source
    - **Index**: dashboard with summary and navigation
    """

    def __init__(self, brain_dir: Path) -> None:
        self._brain_dir = brain_dir
        self._planner = EmissionPlanner(brain_dir)

    def project(
        self,
        claims: list[Claim],
        topic_map: dict[str, list[str]] | None = None,
        changed_claim_ids: set[str] | None = None,
        force_full: bool = False,
    ) -> int:
        """Render claims to consolidated Markdown files.

        Args:
            claims: all active claims from the ledger
            topic_map: optional ``{claim_id: [topic, ...]}`` from TopicClassifier
            changed_claim_ids: if provided, only re-render affected files
            force_full: if True, re-render everything regardless of manifest

        Returns:
            Number of files written.
        """
        if not claims:
            return 0

        active = [c for c in claims if c.status not in _EXCLUDED]
        if not active:
            return 0

        # Build data structures for rendering
        by_subject = self._group_by_subject(active)
        by_source = self._group_by_source(active)
        by_topic = self._group_by_topic(active, topic_map)
        relations = self._build_entity_relations(active, by_subject)
        source_slugs = {src: _slugify(src) for src in by_source}

        # Render all pages to memory
        rendered: dict[str, tuple[str, list[str]]] = {}

        for subject, subject_claims in by_subject.items():
            path = f"entities/{_slugify(subject)}.md"
            claim_ids = [str(c.id) for c in subject_claims]
            content = self._render_entity_page(
                subject, subject_claims, by_topic, relations, source_slugs
            )
            rendered[path] = (content, claim_ids)

        for topic, topic_claims in by_topic.items():
            path = f"topics/{_slugify(topic)}.md"
            claim_ids = [str(c.id) for c in topic_claims]
            content = self._render_topic_page(topic, topic_claims)
            rendered[path] = (content, claim_ids)

        for source, source_claims in by_source.items():
            path = f"sources/{_slugify(source)}.md"
            claim_ids = [str(c.id) for c in source_claims]
            content = self._render_source_page(source, source_claims)
            rendered[path] = (content, claim_ids)

        # Conflicts page
        conflicts = [c for c in active if c.conflicts_with]
        if conflicts:
            path = "conflicts/pending.md"
            claim_ids = [str(c.id) for c in conflicts]
            content = self._render_conflicts_page(conflicts)
            rendered[path] = (content, claim_ids)

        # Index
        rendered["index.md"] = (
            self._render_index(active, by_subject, by_topic, by_source),
            [],
        )

        # Plan and execute
        plan = self._planner.plan(rendered, changed_claim_ids, force_full)
        written = self._planner.execute(plan)

        logger.info(
            "Projected %d files (%s) to %s",
            written,
            plan.summary,
            self._brain_dir,
        )
        return written

    # -- Grouping helpers --

    @staticmethod
    def _group_by_subject(claims: list[Claim]) -> dict[str, list[Claim]]:
        by_subject: dict[str, list[Claim]] = {}
        for claim in claims:
            by_subject.setdefault(claim.subject, []).append(claim)
        return by_subject

    @staticmethod
    def _group_by_source(claims: list[Claim]) -> dict[str, list[Claim]]:
        by_source: dict[str, list[Claim]] = {}
        for claim in claims:
            for ev in claim.evidence:
                source = ev.source_artifact or str(ev.artifact_id)[:8]
                if source:
                    by_source.setdefault(source, []).append(claim)
                    break
        return by_source

    @staticmethod
    def _group_by_topic(
        claims: list[Claim],
        topic_map: dict[str, list[str]] | None,
    ) -> dict[str, list[Claim]]:
        by_topic: dict[str, list[Claim]] = {}
        if not topic_map:
            return by_topic
        for claim in claims:
            topics = topic_map.get(str(claim.id), [])
            for topic in topics:
                by_topic.setdefault(topic, []).append(claim)
        return by_topic

    @staticmethod
    def _build_entity_relations(
        claims: list[Claim],
        by_subject: dict[str, list[Claim]],
    ) -> dict[str, set[str]]:
        """Build co-occurrence map between entities."""
        all_subjects = set(by_subject.keys())
        relations: dict[str, set[str]] = {}

        for claim in claims:
            subj = claim.subject
            # Check if object references another known entity
            obj_lower = claim.object.lower()
            for other_subj in all_subjects:
                if other_subj != subj and other_subj.lower() in obj_lower:
                    relations.setdefault(subj, set()).add(other_subj)
                    relations.setdefault(other_subj, set()).add(subj)

            # Also link entities that share a source
            for ev in claim.evidence:
                source = ev.source_artifact or ""
                if source:
                    for other_claim in claims:
                        if other_claim.subject != subj:
                            for oev in other_claim.evidence:
                                if (oev.source_artifact or "") == source:
                                    relations.setdefault(subj, set()).add(
                                        other_claim.subject
                                    )
                                    break

        return relations

    # -- Rendering --

    def _render_entity_page(
        self,
        subject: str,
        claims: list[Claim],
        by_topic: dict[str, list[Claim]],
        relations: dict[str, set[str]],
        source_slugs: dict[str, str],
    ) -> str:
        """Render a consolidated entity page with all claims grouped by topic."""
        claims.sort(key=lambda c: c.truth_score, reverse=True)

        # Collect unique sources for this entity
        entity_sources: dict[str, str] = {}
        for c in claims:
            for ev in c.evidence:
                src = ev.source_artifact or str(ev.artifact_id)[:8]
                if src and src not in entity_sources:
                    entity_sources[src] = _slugify(src)

        # Find which topics this entity appears in
        entity_topics: list[str] = []
        for topic, topic_claims in by_topic.items():
            if any(c.subject == subject for c in topic_claims):
                entity_topics.append(topic)

        fm = _frontmatter(
            {
                "entity": subject,
                "claim_count": len(claims),
                "avg_truth_score": round(
                    sum(c.truth_score for c in claims) / max(len(claims), 1), 3
                ),
                "topics": entity_topics or ["general"],
                "sources": list(entity_sources.keys())[:10],
                "last_updated": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", f"# {subject}", ""]

        # Group claims by topic for this entity
        claim_ids = {str(c.id) for c in claims}
        topics_for_entity: dict[str, list[Claim]] = {}
        for topic, topic_claims in by_topic.items():
            matching = [c for c in topic_claims if str(c.id) in claim_ids]
            if matching:
                topics_for_entity[topic] = matching

        if topics_for_entity:
            lines.append("## Facts")
            lines.append("")
            for topic, topic_claims in sorted(topics_for_entity.items()):
                topic_slug = _slugify(topic)
                lines.append(f"### [{topic}](../topics/{topic_slug}.md)")
                lines.append("")
                for claim in sorted(topic_claims, key=lambda c: c.truth_score, reverse=True):
                    self._append_claim_bullet(lines, claim, source_slugs, prefix="../")
                lines.append("")
        else:
            # No topic map — render flat
            lines.append("## Facts")
            lines.append("")
            for claim in claims:
                self._append_claim_bullet(lines, claim, source_slugs, prefix="../")
            lines.append("")

        # Related entities
        related = relations.get(subject, set())
        if related:
            lines.append("## Related Entities")
            lines.append("")
            for rel in sorted(related):
                rel_slug = _slugify(rel)
                lines.append(f"- [{rel}]({rel_slug}.md)")
            lines.append("")

        # Source documents
        if entity_sources:
            lines.append("## Source Documents")
            lines.append("")
            for src, slug in entity_sources.items():
                lines.append(f"- [{src}](../sources/{slug}.md)")
            lines.append("")

        return "\n".join(lines)

    def _render_topic_page(
        self,
        topic: str,
        claims: list[Claim],
    ) -> str:
        """Render a cross-entity topic page."""
        claims.sort(key=lambda c: c.truth_score, reverse=True)
        entities = sorted({c.subject for c in claims})

        fm = _frontmatter(
            {
                "topic": topic,
                "entities": entities,
                "claim_count": len(claims),
                "last_updated": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", f"# {topic}", ""]
        lines.append("## Key Facts")
        lines.append("")

        # Group by entity within topic
        by_entity: dict[str, list[Claim]] = {}
        for c in claims:
            by_entity.setdefault(c.subject, []).append(c)

        for entity, entity_claims in sorted(by_entity.items()):
            entity_slug = _slugify(entity)
            lines.append(f"### [{entity}](../entities/{entity_slug}.md)")
            lines.append("")
            for claim in sorted(entity_claims, key=lambda c: c.truth_score, reverse=True):
                self._append_claim_bullet(lines, claim, {}, prefix="../")
            lines.append("")

        # Entity list
        lines.append("## Entities Involved")
        lines.append("")
        for entity in entities:
            entity_slug = _slugify(entity)
            lines.append(f"- [{entity}](../entities/{entity_slug}.md)")
        lines.append("")

        return "\n".join(lines)

    def _render_source_page(
        self,
        source: str,
        claims: list[Claim],
    ) -> str:
        """Render a source provenance page."""
        claims.sort(key=lambda c: c.truth_score, reverse=True)
        entities = sorted({c.subject for c in claims})

        fm = _frontmatter(
            {
                "source": source,
                "claims_extracted": len(claims),
                "entities": entities,
                "last_updated": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", f"# {source}", ""]
        lines.append("## Extracted Claims")
        lines.append("")
        for i, claim in enumerate(claims, 1):
            entity_slug = _slugify(claim.subject)
            score_bar = "█" * int(claim.truth_score * 10)
            lines.append(
                f"{i}. {claim.natural_language} "
                f"`score={claim.truth_score:.2f}` {score_bar} "
                f"→ [{claim.subject}](../entities/{entity_slug}.md)"
            )
        lines.append("")

        lines.append("## Entities Referenced")
        lines.append("")
        for entity in entities:
            entity_slug = _slugify(entity)
            entity_claim_count = sum(1 for c in claims if c.subject == entity)
            lines.append(
                f"- [{entity}](../entities/{entity_slug}.md) ({entity_claim_count} claims)"
            )
        lines.append("")

        return "\n".join(lines)

    def _render_conflicts_page(self, conflicts: list[Claim]) -> str:
        """Render a page of outstanding conflicts."""
        fm = _frontmatter(
            {
                "title": "Pending Conflicts",
                "conflict_count": len(conflicts),
                "last_updated": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", "# Pending Conflicts", ""]
        lines.append(
            "The following claims have unresolved conflicts. "
            "Review and resolve via `noodly resolve` or the GitLab MR workflow."
        )
        lines.append("")

        for claim in conflicts:
            entity_slug = _slugify(claim.subject)
            lines.append(f"### {claim.natural_language}")
            lines.append("")
            lines.append(
                f"- **Entity:** [{claim.subject}](../entities/{entity_slug}.md)"
            )
            lines.append(f"- **Score:** {claim.truth_score:.2f}")
            lines.append(
                f"- **Conflicts with:** {len(claim.conflicts_with)} claim(s)"
            )
            lines.append("")

        return "\n".join(lines)

    def _render_index(
        self,
        claims: list[Claim],
        by_subject: dict[str, list[Claim]],
        by_topic: dict[str, list[Claim]],
        by_source: dict[str, list[Claim]],
    ) -> str:
        """Render the dashboard index.md."""
        by_status: dict[str, int] = {}
        for c in claims:
            by_status[c.status.value] = by_status.get(c.status.value, 0) + 1

        fm = _frontmatter(
            {
                "title": "Noodly Brain Index",
                "total_claims": len(claims),
                "entity_count": len(by_subject),
                "topic_count": len(by_topic),
                "source_count": len(by_source),
                "last_projected": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", "# Noodly Brain", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total active claims:** {len(claims)}")
        lines.append(f"- **Entities:** {len(by_subject)}")
        lines.append(f"- **Topics:** {len(by_topic)}")
        lines.append(f"- **Sources:** {len(by_source)}")
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

        if by_subject:
            lines.append("## Entities")
            lines.append("")
            for subj in sorted(by_subject.keys()):
                slug = _slugify(subj)
                count = len(by_subject[subj])
                lines.append(f"- [{subj}](entities/{slug}.md) ({count} claims)")
            lines.append("")

        if by_topic:
            lines.append("## Topics")
            lines.append("")
            for topic in sorted(by_topic.keys()):
                slug = _slugify(topic)
                count = len(by_topic[topic])
                lines.append(f"- [{topic}](topics/{slug}.md) ({count} claims)")
            lines.append("")

        if by_source:
            lines.append("## Sources")
            lines.append("")
            for source in sorted(by_source.keys()):
                slug = _slugify(source)
                count = len(by_source[source])
                lines.append(f"- [{source}](sources/{slug}.md) ({count} claims)")
            lines.append("")

        return "\n".join(lines)

    # -- Helpers --

    @staticmethod
    def _append_claim_bullet(
        lines: list[str],
        claim: Claim,
        source_slugs: dict[str, str],
        prefix: str = "",
    ) -> None:
        """Append a claim as a bullet point with score bar and source link."""
        score_bar = "█" * int(claim.truth_score * 10)
        line = (
            f"- **[{claim.status.value}]** {claim.natural_language}  "
            f"`score={claim.truth_score:.2f}` {score_bar}"
        )
        lines.append(line)

        if claim.evidence:
            ev = claim.evidence[0]
            source = ev.source_artifact or str(ev.artifact_id)[:8]
            source_slug = source_slugs.get(source, _slugify(source))
            if ev.source_span:
                span_preview = ev.source_span[:120].replace("\n", " ")
                lines.append(
                    f"  > _{span_preview}_ "
                    f"— [{source}]({prefix}sources/{source_slug}.md)"
                )
            else:
                lines.append(
                    f"  — [{source}]({prefix}sources/{source_slug}.md)"
                )

    @property
    def planner(self) -> EmissionPlanner:
        """Expose the emission planner for external use (e.g., GitLab sync)."""
        return self._planner
