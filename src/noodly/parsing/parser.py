"""Document parser — converts binary/text formats to Markdown via MarkItDown."""

from __future__ import annotations

import logging
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
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        # Text
        ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json",
        ".yaml", ".yml", ".xml", ".html", ".htm", ".tex",
        # Code
        ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".sh",
        # Config
        ".toml", ".ini", ".cfg", ".conf", ".log",
        # Other MarkItDown formats
        ".wav", ".mp3", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
        ".zip", ".msg", ".eml",
    }

    def __init__(self, enable_docling: bool = False) -> None:
        self._markitdown = None
        self._docling_converter = None
        self._enable_docling = enable_docling

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
        if ext in {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts",
                   ".go", ".rs", ".java", ".c", ".cpp", ".h", ".sh",
                   ".toml", ".ini", ".cfg", ".conf", ".log"}:
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

                self._docling_converter = DocumentConverter()

            result = self._docling_converter.convert(str(path))
            markdown = result.document.export_to_markdown()

            tables = markdown.count("| --- ") + markdown.count("|---")

            return ParsedDocument(
                title=path.name,
                markdown=markdown,
                source_format=path.suffix.lstrip("."),
                word_count=len(markdown.split()),
                tables_detected=tables,
                metadata={"parser_backend": "docling"},
            )
        except ImportError:
            logger.warning(
                "Docling not installed. Install with: pip install noodly[docling]"
            )
            return self._parse_with_markitdown(path)
        except Exception:
            logger.exception("Docling failed for %s, falling back to MarkItDown", path)
            return self._parse_with_markitdown(path)
