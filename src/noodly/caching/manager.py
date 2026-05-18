"""Cache manager — coordinates cache lifecycle and invalidation."""

from __future__ import annotations

import logging
from pathlib import Path

from noodly.caching.decision_cache import DecisionCache
from noodly.caching.extraction_cache import ExtractionCache
from noodly.caching.parse_cache import ParseCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Coordinates all cache layers and handles invalidation.

    Usage::

        cache = CacheManager(brain_dir / ".cache")
        cached_doc = cache.parse.get(content_hash)
        cache.invalidate_for_source(source_uri, content_hash)
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self.parse = ParseCache(cache_dir)
        self.extraction = ExtractionCache(cache_dir)
        self.decisions = DecisionCache(cache_dir)

    def invalidate_for_source(self, content_hash: str) -> None:
        """When a source file changes, invalidate dependent caches."""
        self.parse.invalidate(content_hash)
        logger.debug("Invalidated caches for hash %s", content_hash)

    def gc(self, max_entries: int = 10000) -> int:
        """Garbage-collect old cache entries if over limit."""
        # Simple v1: just count entries, no actual GC yet
        return 0
