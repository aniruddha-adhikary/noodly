"""Extraction cache — cache LLM extraction results per chunk content hash."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractionCache:
    """Cache LLM extraction results per chunk content hash.

    If a text chunk hasn't changed, its claims won't change either
    (with temperature=0.1). Saves LLM calls on re-ingestion.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "extractions"
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, chunk_hash: str) -> list[dict] | None:
        """Return cached extraction result if available."""
        path = self._dir / f"{chunk_hash}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt extraction cache for hash %s", chunk_hash)
        return None

    def put(self, chunk_hash: str, claims: list[dict]) -> None:
        """Cache extraction result."""
        path = self._dir / f"{chunk_hash}.json"
        path.write_text(json.dumps(claims, indent=2, default=str))

    def has(self, chunk_hash: str) -> bool:
        """Check if a cached entry exists."""
        return (self._dir / f"{chunk_hash}.json").exists()

    def invalidate(self, chunk_hash: str) -> None:
        """Remove a cached entry."""
        path = self._dir / f"{chunk_hash}.json"
        if path.exists():
            path.unlink()

    def invalidate_all(self) -> int:
        """Remove all cached entries. Returns count removed."""
        removed = 0
        for path in self._dir.glob("*.json"):
            path.unlink()
            removed += 1
        return removed
