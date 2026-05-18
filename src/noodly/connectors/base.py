"""Base connector interface."""

from __future__ import annotations

import abc

from noodly.models.artifacts import SourceArtifact


class BaseConnector(abc.ABC):
    """Every connector must yield SourceArtifacts from its source."""

    @abc.abstractmethod
    async def scan(self) -> list[SourceArtifact]:
        """Scan the source and return new/updated artifacts."""

    @abc.abstractmethod
    async def watch(self) -> None:
        """Watch for live changes (optional — can be a no-op)."""
