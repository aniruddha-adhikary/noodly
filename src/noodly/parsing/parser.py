"""Document parser — converts binary/text formats to Markdown via MarkItDown."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Result of parsing a source file into Markdown."""

    title: str
    markdown: str
    source_format: str
    word_count: int = 0
    page_count: int = 0
    tables_detected: int = 0
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.word_count == 0 and self.markdown:
            self.word_count = len(self.markdown.split())

    @property
    def quality_score(self) -> float:
        """Heuristic quality score (0.0–1.0) for comparing parser outputs.

        Factors: word count, table presence, non-ASCII ratio (OCR noise),
        average line length, and structural markers (headings, lists).
        """
        if not self.markdown:
            return 0.0

        score = 0.0
        text = self.markdown

        # Word count contribution (more content = better, up to a point)
        wc = self.word_count or len(text.split())
        score += min(wc / 500, 0.3)  # max 0.3 from word count

        # Table presence
        if self.tables_detected > 0:
            score += 0.15

        # Structural markers (headings, lists)
        headings = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        lists = len(re.findall(r"^[-*]\s", text, re.MULTILINE))
        score += min((headings + lists) / 20, 0.2)

        # Penalize OCR noise (high non-ASCII ratio)
        if wc > 0:
            non_ascii = sum(1 for c in text if ord(c) > 127)
            noise_ratio = non_ascii / len(text) if text else 0
            if noise_ratio > 0.1:
                score *= 0.7

        # Penalize very short average line length (garbled output)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            avg_len = sum(len(ln) for ln in lines) / len(lines)
            if avg_len < 10:
                score *= 0.5

        # Penalize binary-looking content
        binary_chars = sum(1 for c in text[:500] if ord(c) < 32 and c not in "\n\r\t")
        if binary_chars > 10:
            score *= 0.1

        return min(score, 1.0)


class DocumentParser:
    """Parse documents to Markdown using MarkItDown.

    MarkItDown handles: PDF, DOCX, XLSX, PPTX, HTML, CSV, images, audio, ZIP, MSG, etc.

    Usage::

        parser = DocumentParser()
        result = parser.parse(Path("report.pdf"))
        print(result.markdown)
    """

    # Formats supported by MarkItDown
    SUPPORTED_EXTENSIONS = {
        # Documents
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        # Text
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
        ".tex",
        # Code
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
        # Config
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        # Other MarkItDown formats
        ".wav",
        ".mp3",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".zip",
        ".msg",
        ".eml",
    }

    def __init__(self, enable_docling: bool = False, ocr_enabled: bool = True) -> None:
        self._markitdown = None
        self._docling_converter = None
        self._enable_docling = enable_docling
        self._ocr_enabled = ocr_enabled

    def _get_markitdown(self):
        """Lazy-init MarkItDown to avoid import cost until needed."""
        if self._markitdown is None:
            from markitdown import MarkItDown

            self._markitdown = MarkItDown()
        return self._markitdown

    def can_parse(self, path: Path) -> bool:
        """Check if a file format is supported."""
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, path: Path, backend: str = "auto") -> ParsedDocument:
        """Parse a file to Markdown.

        Args:
            path: Path to the file.
            backend: Parser backend — "auto", "markitdown", or "docling".

        Falls back to plain text reading if the chosen backend fails.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if backend == "docling" and self._enable_docling:
            return self._parse_with_docling(path)

        ext = path.suffix.lower()

        # For plain text files, read directly (faster than MarkItDown)
        if ext in {
            ".txt",
            ".md",
            ".markdown",
            ".rst",
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
        }:
            return self._parse_text(path)

        # For binary/complex formats, use MarkItDown
        return self._parse_with_markitdown(path)

    def _parse_text(self, path: Path) -> ParsedDocument:
        """Read a plain text file directly."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Cannot read file %s", path)
            text = ""

        return ParsedDocument(
            title=path.name,
            markdown=text,
            source_format=path.suffix.lstrip("."),
            word_count=len(text.split()),
        )

    def _parse_with_markitdown(self, path: Path) -> ParsedDocument:
        """Parse using MarkItDown for binary/complex formats."""
        try:
            mid = self._get_markitdown()
            result = mid.convert(str(path))
            markdown = result.text_content or ""

            tables = markdown.count("| --- ") + markdown.count("|---")

            return ParsedDocument(
                title=result.title or path.name,
                markdown=markdown,
                source_format=path.suffix.lstrip("."),
                word_count=len(markdown.split()),
                tables_detected=tables,
            )
        except Exception:
            logger.exception("MarkItDown failed for %s, falling back to text", path)
            return self._parse_text(path)

    def _parse_with_docling(self, path: Path) -> ParsedDocument:
        """Parse using Docling for complex layouts, tables, and OCR."""
        try:
            if self._docling_converter is None:
                from docling.document_converter import DocumentConverter

                converter_kwargs: dict = {}
                if self._ocr_enabled:
                    try:
                        from docling.datamodel.pipeline_options import PdfPipelineOptions
                        from docling.pipeline.standard_pdf_pipeline import (
                            StandardPdfPipeline,
                        )

                        pdf_opts = PdfPipelineOptions(do_ocr=True)
                        converter_kwargs["pipeline_options"] = {
                            StandardPdfPipeline: pdf_opts,
                        }
                    except ImportError:
                        pass

                self._docling_converter = DocumentConverter(**converter_kwargs)

            result = self._docling_converter.convert(str(path))
            markdown = result.document.export_to_markdown()

            tables = markdown.count("| --- ") + markdown.count("|---")

            # Extract table count from Docling's structured output if available
            doc_tables = 0
            try:
                doc_tables = len(list(result.document.tables))
            except (AttributeError, TypeError):
                doc_tables = tables

            page_count = 0
            try:
                page_count = len(list(result.document.pages))
            except (AttributeError, TypeError):
                pass

            return ParsedDocument(
                title=path.name,
                markdown=markdown,
                source_format=path.suffix.lstrip("."),
                word_count=len(markdown.split()),
                page_count=page_count,
                tables_detected=doc_tables or tables,
                metadata={"parser_backend": "docling"},
            )
        except ImportError:
            logger.warning("Docling not installed. Install with: pip install noodly[docling]")
            return self._parse_with_markitdown(path)
        except Exception:
            logger.exception("Docling failed for %s, falling back to MarkItDown", path)
            return self._parse_with_markitdown(path)

    def parse_best(self, path: Path) -> ParsedDocument:
        """Parse with both backends and return the higher-quality result.

        Useful for complex documents where it's unclear which parser will
        produce better output. Compares quality scores and returns the winner.
        """
        markitdown_result = self._parse_with_markitdown(path)

        if not self._enable_docling:
            return markitdown_result

        docling_result = self._parse_with_docling(path)

        mid_score = markitdown_result.quality_score
        doc_score = docling_result.quality_score

        logger.info(
            "Parser comparison for %s: MarkItDown=%.2f, Docling=%.2f",
            path.name,
            mid_score,
            doc_score,
        )

        if doc_score > mid_score:
            docling_result.metadata["parser_comparison"] = (
                f"docling={doc_score:.2f} > markitdown={mid_score:.2f}"
            )
            return docling_result

        markitdown_result.metadata["parser_comparison"] = (
            f"markitdown={mid_score:.2f} >= docling={doc_score:.2f}"
        )
        return markitdown_result
