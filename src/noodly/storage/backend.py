"""Abstract storage backend protocol for the fact ledger."""

from __future__ import annotations

from typing import Protocol

from noodly.models.claims import Claim


class LedgerBackend(Protocol):
    """Abstract storage backend for the fact ledger.

    Implementations must provide these methods. The FactLedger delegates
    all persistence to the backend, keeping business logic in the ledger.
    """

    def load_claims(self) -> dict[str, Claim]:
        """Load all claims from storage. Returns {claim_id: Claim}."""
        ...

    def save_claim(self, claim: Claim) -> None:
        """Persist a single claim (insert or update)."""
        ...

    def save_all(self, claims: dict[str, Claim]) -> None:
        """Persist all claims (bulk write)."""
        ...

    def delete_claim(self, claim_id: str) -> None:
        """Remove a claim from storage."""
        ...
