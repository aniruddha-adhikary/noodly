"""Source authority registry — per-source trust weights with optional topic awareness.

Supports two formats (backward-compatible):
- Flat: ``{"jane@company.com": 0.6}``
- Topic-aware: ``{"customs.gov.sg": {"_default": 0.7, "trade": 0.95, "http": 0.3}}``

Both can coexist in the same registry file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_AUTHORITY = 0.5


class AuthorityRegistry:
    """JSON-file-backed registry mapping source identifiers to trust weights.

    Supports topic-aware weights: a source can have different authority
    on different topics. When a topic is provided, the lookup order is:

    1. ``weights[source][topic]`` — exact topic match
    2. ``weights[source]["_default"]`` — source-level default
    3. ``DEFAULT_AUTHORITY`` (0.5) — global fallback

    Usage::

        registry = AuthorityRegistry(Path("brain/authority.json"))

        # Flat (backward-compat)
        registry.set("jane@company.com", 0.9)
        registry.get("jane@company.com")  # 0.9

        # Topic-aware
        registry.set("customs.gov.sg", 0.95, topic="trade")
        registry.set("customs.gov.sg", 0.3, topic="http")
        registry.get("customs.gov.sg", topic="trade")  # 0.95
        registry.get("customs.gov.sg", topic="http")    # 0.3
        registry.get("customs.gov.sg")                  # 0.5 (default)
    """

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._weights: dict[str, float | dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for k, v in data.items():
                    if isinstance(v, dict):
                        self._weights[k] = {tk: float(tv) for tk, tv in v.items()}
                    else:
                        self._weights[k] = float(v)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not load authority registry from %s", self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._weights, indent=2, sort_keys=True))

    def get(self, source: str, topic: str | None = None) -> float:
        """Look up the authority weight for a source, optionally scoped to a topic.

        Resolution order:
        1. ``weights[source][topic]`` — exact topic match
        2. ``weights[source]["_default"]`` — source-level default
        3. ``DEFAULT_AUTHORITY`` (0.5) — global fallback
        """
        entry = self._weights.get(source)
        if entry is None:
            return DEFAULT_AUTHORITY

        if isinstance(entry, dict):
            if topic and topic in entry:
                return entry[topic]
            if "_default" in entry:
                return entry["_default"]
            return DEFAULT_AUTHORITY

        return entry

    def set(self, source: str, weight: float, topic: str | None = None) -> None:
        """Set the authority weight for a source (clamped to [0.0, 1.0]).

        If ``topic`` is given, sets a topic-specific weight. The source entry
        is automatically upgraded to topic-aware format if it was flat.
        """
        clamped = max(0.0, min(1.0, weight))

        if topic is not None:
            existing = self._weights.get(source)
            if isinstance(existing, dict):
                existing[topic] = clamped
            elif isinstance(existing, (int, float)):
                self._weights[source] = {"_default": float(existing), topic: clamped}
            else:
                self._weights[source] = {topic: clamped}
            logger.info("Authority: set %s[%s] = %.2f", source, topic, clamped)
        else:
            existing = self._weights.get(source)
            if isinstance(existing, dict):
                existing["_default"] = clamped
            else:
                self._weights[source] = clamped
            logger.info("Authority: set %s = %.2f", source, clamped)

        self._save()

    def remove(self, source: str, topic: str | None = None) -> bool:
        """Remove a source or topic from the registry.

        If ``topic`` is given, removes only that topic entry.
        If the source becomes empty after removal, removes the source entirely.
        Returns True if something was removed.
        """
        if source not in self._weights:
            return False

        if topic is not None:
            entry = self._weights[source]
            if isinstance(entry, dict) and topic in entry:
                del entry[topic]
                if not entry:
                    del self._weights[source]
                self._save()
                return True
            return False

        del self._weights[source]
        self._save()
        return True

    def list_sources(self) -> dict[str, float | dict[str, float]]:
        """Return all registered sources and their weights."""
        return dict(sorted(self._weights.items()))

    def get_topics(self, source: str) -> list[str]:
        """Return the list of topics configured for a source."""
        entry = self._weights.get(source)
        if isinstance(entry, dict):
            return [k for k in sorted(entry.keys()) if k != "_default"]
        return []

    @property
    def count(self) -> int:
        return len(self._weights)
