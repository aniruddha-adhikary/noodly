"""Local filesystem connector — watches a folder and ingests files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from noodly.connectors.base import BaseConnector
from noodly.models.artifacts import SourceArtifact, SourceType

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".sh",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".tex",
}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_text_file(path: Path) -> str | None:
    """Read a file as text, returning None if it cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        logger.warning("Cannot read file %s", path)
        return None


class LocalFSConnector(BaseConnector):
    """Ingest text files from a local directory.

    Usage::

        connector = LocalFSConnector(Path("./inbox"))
        artifacts = await connector.scan()
    """

    def __init__(self, watch_dir: Path, hash_file: Path | None = None) -> None:
        self._watch_dir = watch_dir.resolve()
        self._hash_path = hash_file or (self._watch_dir / ".hashes.json")
        self._seen_hashes: dict[str, str] = {}
        self._load_hashes()

    def _load_hashes(self) -> None:
        """Load persistent hash state from disk."""
        if self._hash_path.exists():
            try:
                self._seen_hashes = json.loads(self._hash_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load hash file %s", self._hash_path)

    def _save_hashes(self) -> None:
        """Persist hash state to disk for idempotent re-ingestion."""
        self._hash_path.parent.mkdir(parents=True, exist_ok=True)
        self._hash_path.write_text(json.dumps(self._seen_hashes, indent=2, sort_keys=True))

    async def scan(self) -> list[SourceArtifact]:
        """Walk the directory and return artifacts for new/changed files."""
        if not self._watch_dir.exists():
            logger.warning("Watch directory does not exist: %s", self._watch_dir)
            return []

        artifacts: list[SourceArtifact] = []
        dirty = False
        for path in sorted(self._watch_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name == ".hashes.json":
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            body = _read_text_file(path)
            if body is None:
                continue

            file_key = str(path.relative_to(self._watch_dir))
            content_hash = _content_hash(body)

            if self._seen_hashes.get(file_key) == content_hash:
                continue

            self._seen_hashes[file_key] = content_hash
            dirty = True

            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            artifact = SourceArtifact(
                source_type=SourceType.local_file,
                source_uri=str(path),
                title=file_key,
                body=body,
                author="",
                content_created_at=mtime,
                metadata={
                    "file_path": file_key,
                    "file_size": stat.st_size,
                    "content_hash": content_hash,
                    "mime_type": mimetypes.guess_type(str(path))[0] or "text/plain",
                },
            )
            artifacts.append(artifact)

        if dirty:
            self._save_hashes()

        logger.info("Scanned %s: found %d new/changed files", self._watch_dir, len(artifacts))
        return artifacts

    async def watch(self) -> None:
        """Poll-based watch loop (simple v1 — no inotify dependency)."""
        logger.info("Watching %s for changes (poll every 5s)", self._watch_dir)
        while True:
            await self.scan()
            await asyncio.sleep(5)
