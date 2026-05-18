"""JSON-file-backed storage backend for the fact ledger."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


class JSONBackend:
    """Stores claims as a JSON array on disk.

    This is the original v1 backend extracted from FactLedger so that
    the same scoring / decay logic can sit on top of any backend.
    """

    def __init__(self, ledger_path: Path) -> None:
        self._path = ledger_path

    def load_claims(self) -> dict[str, Claim]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text())
            claims: dict[str, Claim] = {}
            for item in data:
                claim = Claim(**item)
                claims[str(claim.id)] = claim
            return claims
        except (json.JSONDecodeError, Exception):
            logger.warning("Could not load ledger from %s, starting fresh", self._path)
            return {}

    def save_claim(self, claim: Claim) -> None:
        claims = self.load_claims()
        claims[str(claim.id)] = claim
        self._write(claims)

    def save_all(self, claims: dict[str, Claim]) -> None:
        self._write(claims)

    def delete_claim(self, claim_id: str) -> None:
        claims = self.load_claims()
        claims.pop(claim_id, None)
        self._write(claims)

    def _write(self, claims: dict[str, Claim]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [claim.model_dump(mode="json") for claim in claims.values()]
        self._path.write_text(json.dumps(data, indent=2, default=str))
