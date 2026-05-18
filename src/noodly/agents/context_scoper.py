"""Context scoper — scope agent context to only relevant entities and claims."""

from __future__ import annotations

import logging

from noodly.models.claims import Claim
from noodly.scoring.ledger import FactLedger

logger = logging.getLogger(__name__)


class ContextScoper:
    """Scope agent context to only relevant entities and claims.

    Instead of dumping all entities/claims into agent prompts,
    use this to find only the ones relevant to the current change.
    """

    def __init__(self, ledger: FactLedger) -> None:
        self._ledger = ledger

    def claims_for_entities(self, entity_names: set[str]) -> list[Claim]:
        """Get all claims where any of the entity names appear as subject or object."""
        all_claims = self._ledger.list_claims(limit=10000)
        return [
            c
            for c in all_claims
            if c.subject.lower() in {n.lower() for n in entity_names}
            or c.object.lower() in {n.lower() for n in entity_names}
        ]

    def claims_for_source(self, source_uri: str) -> list[Claim]:
        """Get all claims extracted from a specific source document."""
        all_claims = self._ledger.list_claims(limit=10000)
        return [
            c
            for c in all_claims
            if any(source_uri in str(ev.artifact_id) for ev in c.evidence)
        ]

    def entities_from_claims(self, claims: list[Claim]) -> set[str]:
        """Extract unique entity names from a list of claims."""
        entities: set[str] = set()
        for c in claims:
            entities.add(c.subject)
            entities.add(c.object)
        return entities

    def related_claims(self, claims: list[Claim], max_results: int = 50) -> list[Claim]:
        """Find existing claims that share entities with the given claims."""
        entity_names = self.entities_from_claims(claims)
        related = self.claims_for_entities(entity_names)
        # Exclude claims that are in the input set
        input_ids = {str(c.id) for c in claims}
        return [c for c in related if str(c.id) not in input_ids][:max_results]
