"""Source authority registry — per-source trust weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_AUTHORITY = 0.5


class AuthorityRegistry:
    """JSON-file-backed registry mapping source identifiers to trust weights.

    Sources can be authors, connector types, or specific URIs.
    Authority weights are in [0.0, 1.0] and feed into the truth-score formula.

    Usage::

        registry = AuthorityRegistry(Path("brain/authority.json"))
        registry.set("jane@company.com", 0.9)
        weight = registry.get("jane@company.com")  # 0.9
        weight = registry.get("unknown@test.com")   # 0.5 (default)
    """

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._weights: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._weights = {k: float(v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not load authority registry from %s", self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._weights, indent=2, sort_keys=True))

    def get(self, source: str) -> float:
        """Look up the authority weight for a source. Returns DEFAULT_AUTHORITY if unknown."""
        return self._weights.get(source, DEFAULT_AUTHORITY)

    def set(self, source: str, weight: float) -> None:
        """Set the authority weight for a source (clamped to [0.0, 1.0])."""
        self._weights[source] = max(0.0, min(1.0, weight))
        self._save()
        logger.info("Authority: set %s = %.2f", source, self._weights[source])

    def remove(self, source: str) -> bool:
        """Remove a source from the registry. Returns True if it existed."""
        if source in self._weights:
            del self._weights[source]
            self._save()
            return True
        return False

    def list_sources(self) -> dict[str, float]:
        """Return all registered sources and their weights."""
        return dict(sorted(self._weights.items()))

    @property
    def count(self) -> int:
        return len(self._weights)
