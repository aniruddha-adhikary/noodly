"""Decision cache — cache stable agent decisions like entity merges."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionCache:
    """Cache agent decisions that are unlikely to change.

    For example, "Singapore Land Authority" and "SLA" being the same entity
    is decided once and reused forever.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "agent_results"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._merges: dict[str, dict] = self._load("entity_merges.json")
        self._ontology: dict = self._load("ontology.json")

    def _load(self, filename: str) -> dict:
        path = self._dir / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt decision cache: %s", path)
        return {}

    def _save(self, filename: str, data: dict) -> None:
        path = self._dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))

    def get_merge(self, entity_a: str, entity_b: str) -> dict | None:
        """Check if we already decided these entities are (or aren't) the same."""
        key = self._merge_key(entity_a, entity_b)
        return self._merges.get(key)

    def put_merge(self, entity_a: str, entity_b: str, decision: dict) -> None:
        """Cache a merge/no-merge decision."""
        key = self._merge_key(entity_a, entity_b)
        self._merges[key] = {
            **decision,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save("entity_merges.json", self._merges)

    def get_ontology(self) -> dict:
        """Return cached ontology proposal."""
        return self._ontology

    def put_ontology(self, ontology: dict) -> None:
        """Cache ontology proposal."""
        self._ontology = {
            **ontology,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save("ontology.json", self._ontology)

    def _merge_key(self, a: str, b: str) -> str:
        return "|".join(sorted([a.lower().strip(), b.lower().strip()]))

    @property
    def merge_count(self) -> int:
        return len(self._merges)
