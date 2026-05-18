"""Compact serializer — token-efficient claim/entity formatting for agent prompts."""

from __future__ import annotations

from noodly.models.claims import Claim


class CompactSerializer:
    """Serialize claims and entities compactly for agent prompts.

    ~25 tokens per claim instead of ~100 (75% reduction).
    """

    @staticmethod
    def claim_oneliner(claim: Claim) -> str:
        """Compact single-line representation of a claim."""
        src_count = len(claim.supporting_evidence)
        return (
            f"[C:{str(claim.id)[:7]}] "
            f"{claim.subject} | {claim.predicate} | {claim.object} "
            f"({claim.knowledge_class.value}, {claim.confidence:.2f}, {src_count} src)"
        )

    @staticmethod
    def entity_oneliner(entity: dict) -> str:
        """Compact single-line representation of an entity."""
        summary = entity.get("summary", "")[:80]
        return f"[E:{entity.get('uuid', '')[:7]}] {entity.get('name', '')}: {summary}"

    @staticmethod
    def claims_block(claims: list[Claim], max_claims: int = 50) -> str:
        """Compact multi-line claims block for agent prompts."""
        lines = [CompactSerializer.claim_oneliner(c) for c in claims[:max_claims]]
        if len(claims) > max_claims:
            lines.append(f"... and {len(claims) - max_claims} more")
        return "\n".join(lines)

    @staticmethod
    def diff_summary(
        added: list[Claim],
        removed: list[Claim],
        modified: list[Claim],
    ) -> str:
        """Compact summary of claim changes for agent prompts."""
        lines: list[str] = []
        if added:
            lines.append(f"ADDED ({len(added)}):")
            for c in added[:10]:
                lines.append(f"  + {CompactSerializer.claim_oneliner(c)}")
        if removed:
            lines.append(f"REMOVED ({len(removed)}):")
            for c in removed[:10]:
                lines.append(f"  - {CompactSerializer.claim_oneliner(c)}")
        if modified:
            lines.append(f"MODIFIED ({len(modified)}):")
            for c in modified[:10]:
                lines.append(f"  ~ {CompactSerializer.claim_oneliner(c)}")
        return "\n".join(lines) or "No changes"
