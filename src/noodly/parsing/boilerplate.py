"""Boilerplate stripper — remove repeated headers, footers, page numbers."""

from __future__ import annotations

import re

_PAGE_PATTERN = re.compile(r"^\s*(Page\s+\d+|[-–—]\s*\d+\s*[-–—]|\d+\s*$)")


class BoilerplateStripper:
    """Remove repeated headers, footers, and noise before extraction."""

    def __init__(self, repeat_threshold: int = 3, max_line_len: int = 100) -> None:
        self._repeat_threshold = repeat_threshold
        self._max_line_len = max_line_len

    def strip(self, markdown: str) -> str:
        """Remove boilerplate that wastes extraction tokens."""
        lines = markdown.split("\n")

        # Detect repeated short lines (headers/footers across pages)
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < self._max_line_len:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        boilerplate = {
            line
            for line, count in line_counts.items()
            if count >= self._repeat_threshold
        }

        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped in boilerplate:
                continue
            if _PAGE_PATTERN.match(stripped):
                continue
            cleaned.append(line)

        return "\n".join(cleaned)
