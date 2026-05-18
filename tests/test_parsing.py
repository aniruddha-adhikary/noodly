"""Tests for the parsing module (parser, chunker, boilerplate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from noodly.parsing.boilerplate import BoilerplateStripper
from noodly.parsing.chunker import chunk_markdown, content_hash, get_section_headings
from noodly.parsing.parser import DocumentParser, ParsedDocument

# --- ParsedDocument ---


class TestParsedDocument:
    def test_auto_word_count(self):
        doc = ParsedDocument(title="t", markdown="one two three", source_format="md")
        assert doc.word_count == 3

    def test_explicit_word_count(self):
        doc = ParsedDocument(title="t", markdown="a b c", source_format="md", word_count=99)
        assert doc.word_count == 99

    def test_empty_markdown(self):
        doc = ParsedDocument(title="t", markdown="", source_format="md")
        assert doc.word_count == 0


# --- DocumentParser ---


class TestDocumentParser:
    def test_can_parse_supported(self):
        parser = DocumentParser()
        assert parser.can_parse(Path("report.pdf"))
        assert parser.can_parse(Path("notes.md"))
        assert parser.can_parse(Path("data.xlsx"))
        assert parser.can_parse(Path("slides.pptx"))

    def test_can_parse_unsupported(self):
        parser = DocumentParser()
        assert not parser.can_parse(Path("binary.exe"))
        assert not parser.can_parse(Path("archive.tar.gz"))

    def test_parse_text_file(self, tmp_path):
        f = tmp_path / "hello.md"
        f.write_text("# Hello\nWorld")
        parser = DocumentParser()
        result = parser.parse(f)
        assert result.title == "hello.md"
        assert "Hello" in result.markdown
        assert result.source_format == "md"
        assert result.word_count == 3  # "#", "Hello", "World"

    def test_parse_nonexistent_raises(self, tmp_path):
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nope.md")

    def test_parse_python_file(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    pass\n")
        parser = DocumentParser()
        result = parser.parse(f)
        assert "def hello" in result.markdown
        assert result.source_format == "py"


# --- chunk_markdown ---


class TestChunker:
    def test_empty_input(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []

    def test_single_section_under_limit(self):
        md = "# Title\nSome content here."
        chunks = chunk_markdown(md, max_chars=6000)
        assert len(chunks) == 1
        assert chunks[0].heading == "Title"
        assert chunks[0].index == 0

    def test_multiple_sections(self):
        md = "# A\nContent A\n# B\nContent B\n# C\nContent C"
        chunks = chunk_markdown(md, max_chars=6000)
        assert len(chunks) == 3
        headings = [c.heading for c in chunks]
        assert headings == ["A", "B", "C"]
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_preamble_before_heading(self):
        md = "Some preamble text\n\n# Section\nContent"
        chunks = chunk_markdown(md, max_chars=6000)
        assert len(chunks) == 2
        assert chunks[0].heading == ""
        assert chunks[1].heading == "Section"

    def test_long_section_splits(self):
        long_para = "word " * 500
        md = f"# Long\n{long_para}\n\n{long_para}"
        chunks = chunk_markdown(md, max_chars=3000)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.heading == "Long"

    def test_char_count_populated(self):
        md = "# Test\nHello world"
        chunks = chunk_markdown(md)
        assert chunks[0].char_count == len(chunks[0].content)


class TestGetSectionHeadings:
    def test_extracts_headings(self):
        md = "# A\ntext\n## B\ntext\n### C\ntext"
        headings = get_section_headings(md)
        assert headings == ["A", "B", "C"]

    def test_no_headings(self):
        assert get_section_headings("just text") == []


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_16_chars(self):
        assert len(content_hash("test")) == 16


# --- BoilerplateStripper ---


class TestBoilerplateStripper:
    def test_removes_repeated_lines(self):
        lines = ["ACME Corp Confidential"] * 5 + ["# Real Content", "Important info"]
        md = "\n".join(lines)
        stripper = BoilerplateStripper(repeat_threshold=3)
        result = stripper.strip(md)
        assert "ACME Corp Confidential" not in result
        assert "Important info" in result

    def test_removes_page_numbers(self):
        md = "Content here\nPage 1\nMore content\nPage 2\n— 3 —"
        stripper = BoilerplateStripper()
        result = stripper.strip(md)
        assert "Page 1" not in result
        assert "Page 2" not in result
        assert "Content here" in result

    def test_keeps_unique_lines(self):
        md = "Line A\nLine B\nLine C"
        stripper = BoilerplateStripper()
        result = stripper.strip(md)
        assert result == md

    def test_threshold_respected(self):
        lines = ["Header"] * 2 + ["Content"]
        md = "\n".join(lines)
        stripper = BoilerplateStripper(repeat_threshold=3)
        result = stripper.strip(md)
        assert "Header" in result
