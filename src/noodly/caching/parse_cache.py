"""Parse cache — cache parsed Markdown output keyed by file content hash."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from noodly.parsing.parser import ParsedDocument

logger = logging.getLogger(__name__)


class ParseCache:
    """Cache parsed Markdown output keyed by file content hash.

    Avoids re-parsing unchanged files (especially slow for PDF/DOCX via MarkItDown).
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir / "content"
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, content_hash: str) -> ParsedDocument | None:
        """Return cached parse result if available."""
        md_path = self._dir / f"{content_hash}.md"
        meta_path = self._dir / f"{content_hash}.meta.json"
        if md_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                return ParsedDocument(
                    title=meta.get("title", ""),
                    markdown=md_path.read_text(),
                    source_format=meta.get("source_format", ""),
                    word_count=meta.get("word_count", 0),
                    tables_detected=meta.get("tables_detected", 0),
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache entry for hash %s", content_hash)
        return None

    def put(self, content_hash: str, doc: ParsedDocument) -> None:
        """Cache a parse result."""
        md_path = self._dir / f"{content_hash}.md"
        meta_path = self._dir / f"{content_hash}.meta.json"
        md_path.write_text(doc.markdown)
        meta_path.write_text(
            json.dumps(
                {
                    "title": doc.title,
                    "source_format": doc.source_format,
                    "word_count": doc.word_count,
                    "tables_detected": doc.tables_detected,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )

    def has(self, content_hash: str) -> bool:
        """Check if a cached entry exists."""
        return (self._dir / f"{content_hash}.md").exists()

    def invalidate(self, content_hash: str) -> None:
        """Remove a cached entry."""
        for suffix in (".md", ".meta.json"):
            path = self._dir / f"{content_hash}{suffix}"
            if path.exists():
                path.unlink()
