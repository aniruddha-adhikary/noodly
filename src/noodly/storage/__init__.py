"""Storage backends for the fact ledger."""

from __future__ import annotations

from typing import Protocol

from noodly.models.claims import Claim


class LedgerBackend(Protocol):
    """Protocol for pluggable claim storage backends.

    Implementations handle persistence; the FactLedger handles scoring,
    decay, and promotion logic on top of any backend.
    """

    def load_claims(self) -> dict[str, Claim]:
        """Load all claims from the backing store."""
        ...

    def save_claim(self, claim: Claim) -> None:
        """Persist a single claim (insert or update)."""
        ...

    def save_all(self, claims: dict[str, Claim]) -> None:
        """Persist all claims (batch write)."""
        ...

    def delete_claim(self, claim_id: str) -> None:
        """Remove a claim from the backing store."""
        ...
