"""Agentic extraction orchestrator — agents decide how to extract each file.

Supports MarkItDown (default, fast) and Docling (optional, accurate for
complex layouts). The QA agent evaluates extraction quality and can
trigger re-extraction with an alternate backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from noodly.extraction.extractor import ClaimExtractor
from noodly.models.artifacts import SourceArtifact
from noodly.models.claims import Claim
from noodly.parsing.parser import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """Agentic extraction — agents decide how to extract each file.

    Workflow:
    1. Parse with MarkItDown (fast, default)
    2. Optionally run QA agent to review quality
    3. If issues found + Docling available → re-parse with Docling
    4. QA agent compares both representations and picks best
    5. Extract claims from the chosen representation

    Usage::

        orchestrator = ExtractionOrchestrator(
            parser=parser,
            extractor=extractor,
            enable_docling=True,
        )
        claims = await orchestrator.extract(artifact)
    """

    def __init__(
        self,
        parser: DocumentParser,
        extractor: ClaimExtractor,
        qa_agent=None,
        enable_docling: bool = False,
        extraction_mode: str = "auto",
    ) -> None:
        self._parser = parser
        self._extractor = extractor
        self._qa_agent = qa_agent
        self._enable_docling = enable_docling
        self._extraction_mode = extraction_mode

    async def extract(
        self,
        artifact: SourceArtifact,
        source_path: Path | None = None,
    ) -> list[Claim]:
        """Extract claims from an artifact using the best available method.

        In ``auto`` mode (default):
        1. Parse with MarkItDown
        2. If QA agent is available, review quality
        3. If quality is poor and Docling is available, try Docling
        4. Pick the best representation
        5. Extract claims
        """
        if not artifact.body.strip() and source_path is None:
            return await self._extractor.extract(artifact)

        if self._extraction_mode == "docling" and self._enable_docling:
            return await self._extract_with_docling(artifact, source_path)

        if self._extraction_mode == "multi" and self._enable_docling:
            return await self._extract_multi(artifact, source_path)

        if self._extraction_mode == "markitdown":
            return await self._extractor.extract(artifact)

        # auto mode
        claims = await self._extractor.extract(artifact)

        if (
            self._qa_agent is not None
            and source_path is not None
            and self._enable_docling
        ):
            qa_result = await self._qa_agent.review(
                ParsedDocument(
                    title=artifact.title,
                    markdown=artifact.body,
                    source_format=source_path.suffix.lstrip(".") if source_path else "",
                ),
            )
            status = "pass"
            if hasattr(qa_result, "get"):
                status = qa_result.get("status", "pass")
            elif hasattr(qa_result, "status"):
                status = getattr(qa_result, "status", "pass")

            if status in ("fail", "poor"):
                logger.info(
                    "QA agent flagged poor quality for %s, trying Docling",
                    artifact.title,
                )
                docling_claims = await self._extract_with_docling(artifact, source_path)
                if len(docling_claims) > len(claims):
                    logger.info(
                        "Docling extracted %d claims vs MarkItDown's %d — using Docling",
                        len(docling_claims),
                        len(claims),
                    )
                    return docling_claims

        return claims

    async def _extract_with_docling(
        self,
        artifact: SourceArtifact,
        source_path: Path | None,
    ) -> list[Claim]:
        """Extract using Docling backend."""
        if source_path is None:
            return await self._extractor.extract(artifact)

        try:
            parsed = self._parser.parse(source_path, backend="docling")
            modified_artifact = SourceArtifact(
                id=artifact.id,
                source_type=artifact.source_type,
                source_uri=artifact.source_uri,
                title=parsed.title or artifact.title,
                body=parsed.markdown,
                author=artifact.author,
                content_hash=artifact.content_hash,
                metadata={**artifact.metadata, "parser_backend": "docling"},
            )
            return await self._extractor.extract(modified_artifact)
        except Exception:
            logger.exception("Docling extraction failed for %s, falling back", artifact.title)
            return await self._extractor.extract(artifact)

    async def _extract_multi(
        self,
        artifact: SourceArtifact,
        source_path: Path | None,
    ) -> list[Claim]:
        """Extract with both backends, pick the one with more claims."""
        markitdown_claims = await self._extractor.extract(artifact)

        if source_path is None or not self._enable_docling:
            return markitdown_claims

        docling_claims = await self._extract_with_docling(artifact, source_path)

        if len(docling_claims) > len(markitdown_claims):
            logger.info(
                "Multi-mode: Docling (%d claims) wins over MarkItDown (%d claims)",
                len(docling_claims),
                len(markitdown_claims),
            )
            return docling_claims

        return markitdown_claims
