"""Section-aware chunker — splits Markdown into chunks respecting headings."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A section of a document with metadata."""

    content: str
    heading: str
    index: int
    char_count: int = 0

    def __post_init__(self) -> None:
        if self.char_count == 0:
            self.char_count = len(self.content)


# Matches Markdown headings (# through ####)
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

DEFAULT_CHUNK_SIZE = 6000
MIN_CHUNK_SIZE = 200


def chunk_markdown(markdown: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[Chunk]:
    """Split Markdown into chunks, respecting section boundaries.

    Strategy:
    1. Split by headings into sections.
    2. If a section exceeds max_chars, split on paragraph boundaries.
    3. Never split mid-paragraph or mid-table.
    """
    if not markdown.strip():
        return []

    sections = _split_by_headings(markdown)

    chunks: list[Chunk] = []
    idx = 0
    for heading, content in sections:
        if len(content) <= max_chars:
            chunks.append(Chunk(content=content, heading=heading, index=idx))
            idx += 1
        else:
            sub_chunks = _split_section(content, heading, max_chars)
            for sc in sub_chunks:
                sc.index = idx
                chunks.append(sc)
                idx += 1

    return chunks


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, content) pairs by heading boundaries."""
    matches = list(_HEADING_RE.finditer(markdown))

    if not matches:
        return [("", markdown)]

    sections: list[tuple[str, str]] = []

    # Content before the first heading
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        if content:
            sections.append((heading, content))

    return sections


def _split_section(
    content: str, heading: str, max_chars: int
) -> list[Chunk]:
    """Split a long section on paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", content)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len + 2 > max_chars and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(Chunk(content=chunk_text, heading=heading, index=0))
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(Chunk(content=chunk_text, heading=heading, index=0))

    return chunks


def get_section_headings(markdown: str) -> list[str]:
    """Extract all section headings from Markdown."""
    return [m.group(2).strip() for m in _HEADING_RE.finditer(markdown)]


def content_hash(text: str) -> str:
    """Compute a short hash for content comparison."""
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]
