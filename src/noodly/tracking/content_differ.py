"""Content differ — compute section-level diffs between document versions."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


@dataclass
class ModifiedSection:
    """A section that exists in both versions but with different content."""

    heading: str
    old_content: str
    new_content: str
    diff_summary: str = ""


@dataclass
class ContentDiff:
    """Diff between two versions of a parsed document."""

    source_uri: str
    old_hash: str
    new_hash: str

    added_sections: list[str] = field(default_factory=list)
    removed_sections: list[str] = field(default_factory=list)
    modified_sections: list[ModifiedSection] = field(default_factory=list)
    unchanged_sections: list[str] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.old_hash == ""

    @property
    def change_ratio(self) -> float:
        total = (
            len(self.added_sections)
            + len(self.removed_sections)
            + len(self.modified_sections)
            + len(self.unchanged_sections)
        )
        if total == 0:
            return 1.0
        changed = (
            len(self.added_sections)
            + len(self.removed_sections)
            + len(self.modified_sections)
        )
        return changed / total

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.added_sections:
            parts.append(f"+{len(self.added_sections)} added")
        if self.removed_sections:
            parts.append(f"-{len(self.removed_sections)} removed")
        if self.modified_sections:
            parts.append(f"~{len(self.modified_sections)} modified")
        if self.unchanged_sections:
            parts.append(f"={len(self.unchanged_sections)} unchanged")
        return ", ".join(parts) or "no sections"


class ContentDiffer:
    """Computes structured diffs between document versions.

    Caches previous Markdown versions in brain_dir/.content_cache/.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def diff(self, source_uri: str, new_markdown: str, new_hash: str) -> ContentDiff:
        """Compare new parsed content against cached previous version."""
        old_markdown = self._load_cached(source_uri)
        old_hash = self._load_cached_hash(source_uri)

        if old_markdown is None:
            sections = self._split_sections(new_markdown)
            self._cache(source_uri, new_markdown, new_hash)
            return ContentDiff(
                source_uri=source_uri,
                old_hash="",
                new_hash=new_hash,
                added_sections=[s for _, s in sections],
            )

        if old_hash == new_hash:
            sections = self._split_sections(new_markdown)
            return ContentDiff(
                source_uri=source_uri,
                old_hash=old_hash,
                new_hash=new_hash,
                unchanged_sections=[s for _, s in sections],
            )

        old_sections = self._split_sections(old_markdown)
        new_sections = self._split_sections(new_markdown)

        added, removed, modified, unchanged = self._align_sections(
            old_sections, new_sections
        )

        self._cache(source_uri, new_markdown, new_hash)
        return ContentDiff(
            source_uri=source_uri,
            old_hash=old_hash,
            new_hash=new_hash,
            added_sections=added,
            removed_sections=removed,
            modified_sections=modified,
            unchanged_sections=unchanged,
        )

    def _split_sections(self, markdown: str) -> list[tuple[str, str]]:
        """Split Markdown into (heading, full_section_text) pairs."""
        matches = list(_HEADING_RE.finditer(markdown))

        if not matches:
            return [("", markdown.strip())] if markdown.strip() else []

        sections: list[tuple[str, str]] = []

        if matches[0].start() > 0:
            preamble = markdown[: matches[0].start()].strip()
            if preamble:
                sections.append(("_preamble_", preamble))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()
            if content:
                sections.append((heading, content))

        return sections

    def _align_sections(
        self,
        old_sections: list[tuple[str, str]],
        new_sections: list[tuple[str, str]],
    ) -> tuple[list[str], list[str], list[ModifiedSection], list[str]]:
        """Align sections by heading and detect changes."""
        old_by_heading = {h: c for h, c in old_sections}
        new_by_heading = {h: c for h, c in new_sections}

        old_headings = set(old_by_heading.keys())
        new_headings = set(new_by_heading.keys())

        added = [new_by_heading[h] for h in new_headings - old_headings]
        removed = [old_by_heading[h] for h in old_headings - new_headings]

        modified: list[ModifiedSection] = []
        unchanged: list[str] = []

        for h in old_headings & new_headings:
            old_content = old_by_heading[h]
            new_content = new_by_heading[h]
            if old_content == new_content:
                unchanged.append(new_content)
            else:
                diff_lines = list(
                    difflib.unified_diff(
                        old_content.splitlines(),
                        new_content.splitlines(),
                        lineterm="",
                        n=1,
                    )
                )
                diff_summary = "\n".join(diff_lines[:20])
                modified.append(
                    ModifiedSection(
                        heading=h,
                        old_content=old_content,
                        new_content=new_content,
                        diff_summary=diff_summary,
                    )
                )

        return added, removed, modified, unchanged

    def _safe_key(self, source_uri: str) -> str:
        """Convert a source URI to a safe filename."""
        import hashlib

        return hashlib.sha256(source_uri.encode()).hexdigest()[:24]

    def _load_cached(self, source_uri: str) -> str | None:
        key = self._safe_key(source_uri)
        path = self._cache_dir / f"{key}.md"
        if path.exists():
            try:
                return path.read_text()
            except OSError:
                pass
        return None

    def _load_cached_hash(self, source_uri: str) -> str:
        key = self._safe_key(source_uri)
        path = self._cache_dir / f"{key}.hash"
        if path.exists():
            try:
                return path.read_text().strip()
            except OSError:
                pass
        return ""

    def _cache(self, source_uri: str, markdown: str, content_hash: str) -> None:
        key = self._safe_key(source_uri)
        (self._cache_dir / f"{key}.md").write_text(markdown)
        (self._cache_dir / f"{key}.hash").write_text(content_hash)
