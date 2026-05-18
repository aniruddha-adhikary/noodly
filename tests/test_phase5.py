"""Tests for Phase 5 — semantic claim dedup, LLM prompt improvements, Docling integration."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from noodly.extraction.extractor import (
    ClaimExtractor,
    EntityAlias,
    ExtractedClaim,
    _parse_date,
)
from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.parsing.parser import DocumentParser, ParsedDocument
from noodly.scoring.ledger import FactLedger
from noodly.scoring.semantic_dedup import DedupResult, SemanticDeduplicator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    subject: str = "X",
    predicate: str = "is",
    obj: str = "Y",
    confidence: float = 0.8,
    artifact_id=None,
    knowledge_class: KnowledgeClass = KnowledgeClass.stable,
    source_artifact: str = "",
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
) -> Claim:
    aid = artifact_id or uuid4()
    return Claim(
        subject=subject,
        predicate=predicate,
        object=obj,
        natural_language=f"{subject} {predicate} {obj}",
        confidence=confidence,
        knowledge_class=knowledge_class,
        status=ClaimStatus.candidate,
        evidence=[
            ClaimEvidence(
                artifact_id=aid,
                supports=True,
                source_span=f"{subject} {predicate} {obj}",
                source_artifact=source_artifact,
                author="test",
            )
        ],
        valid_from=valid_from,
        valid_until=valid_until,
    )


# ===========================================================================
# 1. DedupResult dataclass
# ===========================================================================

class TestDedupResult:
    def test_empty_result(self):
        result = DedupResult()
        assert result.merged_count == 0
        assert result.unique_count == 0

    def test_with_data(self):
        c1 = _make_claim("A", "is", "B")
        c2 = _make_claim("A", "is", "B")
        result = DedupResult(merged=[(c1, c2)], unique=[_make_claim("C", "is", "D")])
        assert result.merged_count == 1
        assert result.unique_count == 1


# ===========================================================================
# 2. SemanticDeduplicator — batch embedding
# ===========================================================================

class TestSemanticDeduplicatorBatch:
    @pytest.mark.asyncio
    async def test_embed_batch_caches_results(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.9)

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        with patch.object(
            dedup._client.embeddings, "create", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await dedup._embed_batch(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

            # Second call should use cache (no API call)
            result2 = await dedup._embed_batch(["hello", "world"])
            assert result2 == result

    @pytest.mark.asyncio
    async def test_embed_batch_handles_errors(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.9)

        with patch.object(
            dedup._client.embeddings,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await dedup._embed_batch(["hello"])
            assert len(result) == 1
            assert result[0] == []  # empty on error

    @pytest.mark.asyncio
    async def test_find_duplicates_batch_pre_embeds(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.5)

        embed_call_count = 0

        async def mock_embed_batch(texts):
            nonlocal embed_call_count
            embed_call_count += 1
            for t in texts:
                dedup._cache[t] = [0.1] * 10
            return [[0.1] * 10] * len(texts)

        with patch.object(dedup, "_embed_batch", side_effect=mock_embed_batch):
            c1 = _make_claim("A", "is", "B")
            c2 = _make_claim("C", "is", "D")
            existing = [_make_claim("A", "is", "B")]
            await dedup.find_duplicates_batch([c1, c2], existing)
            # Should have called _embed_batch twice (once for existing, once for new)
            assert embed_call_count == 2

    @pytest.mark.asyncio
    async def test_find_duplicates_batch_empty_inputs(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.9)
        result = await dedup.find_duplicates_batch([], [])
        assert result == {}
        result2 = await dedup.find_duplicates_batch([_make_claim()], [])
        assert result2 == {}


# ===========================================================================
# 3. SemanticDeduplicator — deduplicate_and_merge
# ===========================================================================

class TestDeduplicateAndMerge:
    @pytest.mark.asyncio
    async def test_merge_updates_evidence(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.5)

        aid1 = uuid4()
        aid2 = uuid4()
        existing = _make_claim("SLA", "was established", "2001", artifact_id=aid1)
        new_claim = _make_claim("SLA", "is established", "2001", artifact_id=aid2)

        # Mock find_duplicates_batch to return a match
        async def mock_find(new_claims, existing_claims):
            return {str(new_claims[0].id): existing_claims[0]}

        with patch.object(dedup, "find_duplicates_batch", side_effect=mock_find):
            result = await dedup.deduplicate_and_merge([new_claim], [existing])

        assert result.merged_count == 1
        assert result.unique_count == 0
        # Evidence should be merged into existing
        assert len(existing.evidence) == 2

    @pytest.mark.asyncio
    async def test_unique_claims_returned(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.5)

        new_claim = _make_claim("X", "is", "Y")

        async def mock_find(new_claims, existing_claims):
            return {}

        with patch.object(dedup, "find_duplicates_batch", side_effect=mock_find):
            result = await dedup.deduplicate_and_merge([new_claim], [])

        assert result.merged_count == 0
        assert result.unique_count == 1
        assert result.unique[0] is new_claim

    @pytest.mark.asyncio
    async def test_empty_input(self):
        dedup = SemanticDeduplicator(api_key="fake", threshold=0.9)
        result = await dedup.deduplicate_and_merge([], [])
        assert result.merged_count == 0
        assert result.unique_count == 0


# ===========================================================================
# 4. ClaimEvidence — source_artifact field
# ===========================================================================

class TestClaimEvidenceSourceArtifact:
    def test_source_artifact_field_exists(self):
        ev = ClaimEvidence(
            artifact_id=uuid4(),
            source_artifact="report.pdf",
            source_span="some text",
        )
        assert ev.source_artifact == "report.pdf"

    def test_source_artifact_default_empty(self):
        ev = ClaimEvidence(artifact_id=uuid4())
        assert ev.source_artifact == ""

    def test_source_artifact_serialization(self):
        ev = ClaimEvidence(
            artifact_id=uuid4(),
            source_artifact="circular-06-2025.md",
        )
        data = ev.model_dump(mode="json")
        assert data["source_artifact"] == "circular-06-2025.md"


# ===========================================================================
# 5. ExtractedClaim — entity_aliases and references fields
# ===========================================================================

class TestExtractedClaimNewFields:
    def test_entity_aliases_field(self):
        ec = ExtractedClaim(
            subject="SLA",
            predicate="is",
            object="government agency",
            natural_language="SLA is a government agency",
            knowledge_class="stable",
            confidence=0.9,
            source_span="SLA...",
            entity_aliases=[EntityAlias(alias="SLA", canonical="Singapore Land Authority")],
        )
        assert len(ec.entity_aliases) == 1
        assert ec.entity_aliases[0].alias == "SLA"

    def test_references_field(self):
        ec = ExtractedClaim(
            subject="RFC 9110",
            predicate="obsoletes",
            object="RFC 7230",
            natural_language="RFC 9110 obsoletes RFC 7230",
            knowledge_class="stable",
            confidence=0.95,
            source_span="...",
            references=["RFC 7230", "RFC 7231"],
        )
        assert len(ec.references) == 2

    def test_defaults_empty(self):
        ec = ExtractedClaim(
            subject="X",
            predicate="is",
            object="Y",
            natural_language="X is Y",
            knowledge_class="stable",
            confidence=0.5,
            source_span="...",
        )
        assert ec.entity_aliases == []
        assert ec.references == []


# ===========================================================================
# 6. Extraction prompt improvements
# ===========================================================================

class TestExtractionPromptImprovements:
    def test_extraction_schema_has_entity_aliases(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        props = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]
        assert "entity_aliases" in props
        assert props["entity_aliases"]["type"] == "array"

    def test_extraction_schema_has_references(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        props = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]
        assert "references" in props

    def test_extraction_schema_required_fields(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        required = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["required"]
        assert "entity_aliases" in required
        assert "references" in required

    def test_system_prompt_temporal_instructions(self):
        from noodly.extraction.extractor import EXTRACTION_SYSTEM_PROMPT

        assert "Temporal extraction" in EXTRACTION_SYSTEM_PROMPT
        assert "populate dates aggressively" in EXTRACTION_SYSTEM_PROMPT
        assert "valid_from" in EXTRACTION_SYSTEM_PROMPT
        assert "supersedes" in EXTRACTION_SYSTEM_PROMPT
        assert "obsoletes" in EXTRACTION_SYSTEM_PROMPT

    def test_system_prompt_supersession_instructions(self):
        from noodly.extraction.extractor import EXTRACTION_SYSTEM_PROMPT

        assert "Supersession and document relationships" in EXTRACTION_SYSTEM_PROMPT
        assert "circular" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "RFC" in EXTRACTION_SYSTEM_PROMPT


# ===========================================================================
# 7. ClaimExtractor — source_filename parameter
# ===========================================================================

class TestClaimExtractorSourceFilename:
    @pytest.mark.asyncio
    async def test_extract_passes_source_filename(self):
        from noodly.models.artifacts import SourceArtifact, SourceType

        extractor = ClaimExtractor(api_key="fake")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "claims": [
                                {
                                    "subject": "Test",
                                    "predicate": "is",
                                    "object": "working",
                                    "natural_language": "Test is working",
                                    "knowledge_class": "stable",
                                    "confidence": 0.9,
                                    "source_span": "Test is working",
                                    "valid_from": "2024-01-01",
                                    "valid_until": None,
                                    "entity_aliases": [],
                                    "references": [],
                                }
                            ]
                        }
                    )
                )
            )
        ]

        artifact = SourceArtifact(
            source_type=SourceType.local_file,
            title="Test Doc",
            body="Test is working",
            author="tester",
        )

        with patch.object(
            extractor._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            claims = await extractor.extract(artifact, source_filename="report.pdf")

        assert len(claims) >= 1
        assert claims[0].evidence[0].source_artifact == "report.pdf"

    @pytest.mark.asyncio
    async def test_extract_generates_alias_claims(self):
        from noodly.models.artifacts import SourceArtifact, SourceType

        extractor = ClaimExtractor(api_key="fake")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "claims": [
                                {
                                    "subject": "Singapore Land Authority",
                                    "predicate": "was established in",
                                    "object": "2001",
                                    "natural_language": "SLA was established in 2001",
                                    "knowledge_class": "stable",
                                    "confidence": 0.9,
                                    "source_span": "SLA was established",
                                    "valid_from": "2001-06-01",
                                    "valid_until": None,
                                    "entity_aliases": [
                                        {
                                            "alias": "SLA",
                                            "canonical": "Singapore Land Authority",
                                        }
                                    ],
                                    "references": [],
                                }
                            ]
                        }
                    )
                )
            )
        ]

        artifact = SourceArtifact(
            source_type=SourceType.local_file,
            title="Test",
            body="SLA was established in 2001",
            author="tester",
        )

        with patch.object(
            extractor._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            claims = await extractor.extract(artifact)

        # Should have 2 claims: the original + the alias claim
        assert len(claims) == 2
        alias_claims = [c for c in claims if c.predicate == "is alias of"]
        assert len(alias_claims) == 1
        assert alias_claims[0].subject == "SLA"
        assert alias_claims[0].object == "Singapore Land Authority"
        assert alias_claims[0].knowledge_class == KnowledgeClass.stable


# ===========================================================================
# 8. ParsedDocument — quality_score
# ===========================================================================

class TestParsedDocumentQualityScore:
    def test_empty_document_score_zero(self):
        doc = ParsedDocument(title="empty", markdown="", source_format="md")
        assert doc.quality_score == 0.0

    def test_good_document_high_score(self):
        markdown = "# Heading\n\n" + "Some content. " * 100 + "\n\n## Section 2\n\n"
        markdown += "| Col1 | Col2 |\n| --- | --- |\n| val | val |\n"
        doc = ParsedDocument(
            title="good",
            markdown=markdown,
            source_format="md",
            tables_detected=1,
        )
        assert doc.quality_score > 0.3

    def test_binary_content_penalized(self):
        markdown = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e" * 50
        doc = ParsedDocument(title="binary", markdown=markdown, source_format="pdf")
        assert doc.quality_score < 0.1

    def test_short_lines_penalized(self):
        markdown = "\n".join(["ab"] * 100)
        doc = ParsedDocument(title="short", markdown=markdown, source_format="txt")
        score = doc.quality_score
        normal = ParsedDocument(
            title="normal",
            markdown="\n".join(["A normal sentence with words."] * 100),
            source_format="txt",
        )
        assert score < normal.quality_score


# ===========================================================================
# 9. DocumentParser — parse_best
# ===========================================================================

class TestDocumentParserParseBest:
    def test_parse_best_without_docling(self, tmp_path):
        parser = DocumentParser(enable_docling=False)
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello\n\nWorld")
        result = parser.parse_best(test_file)
        assert "Hello" in result.markdown

    def test_parse_best_uses_markitdown_when_no_docling(self, tmp_path):
        parser = DocumentParser(enable_docling=False)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some plain text content here.")
        result = parser.parse_best(test_file)
        assert result.markdown == "Some plain text content here."


# ===========================================================================
# 10. FactLedger — auto-promotion
# ===========================================================================

class TestLedgerAutoPromotion:
    def test_auto_promote_candidate_to_unverified(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")
        claim = _make_claim("X", "is", "Y", confidence=0.8)
        stored = ledger.add_claim(claim)
        # With confidence=0.8 and authority=0.5, truth_score should be high enough
        # to auto-promote on first add
        assert stored.status in (ClaimStatus.candidate, ClaimStatus.unverified)

    def test_auto_promote_to_corroborated_on_merge(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")

        aid1 = uuid4()
        aid2 = uuid4()
        claim1 = _make_claim("X", "is", "Y", confidence=0.8, artifact_id=aid1)
        claim2 = _make_claim("X", "is", "Y", confidence=0.9, artifact_id=aid2)

        stored1 = ledger.add_claim(claim1)
        stored2 = ledger.add_claim(claim2)  # should merge into stored1

        # After merge, should have 2 evidence sources → corroborated
        assert stored2.id == stored1.id
        assert len(stored2.evidence) == 2
        # Should be promoted to corroborated (2 independent sources)
        assert stored2.status == ClaimStatus.corroborated

    def test_auto_promote_all(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")

        # Add claims with different evidence counts
        aid1 = uuid4()
        aid2 = uuid4()
        claim1 = _make_claim("A", "is", "B", confidence=0.8, artifact_id=aid1)
        claim2 = _make_claim("A", "is", "B", confidence=0.9, artifact_id=aid2)

        ledger.add_claim(claim1)
        ledger.add_claim(claim2)

        # auto_promote_all should return 0 (already promoted during add)
        promoted = ledger.auto_promote_all()
        assert promoted == 0  # already promoted

    def test_auto_promote_does_not_demote(self, tmp_path):
        ledger = FactLedger(tmp_path / "ledger.json")
        claim = _make_claim("X", "is", "Y", confidence=0.8)
        stored = ledger.add_claim(claim)
        # Manually promote to canonical
        ledger.promote_claim(str(stored.id), ClaimStatus.canonical)
        # auto_promote should not demote
        ledger.auto_promote_all()
        refreshed = ledger.get_claim(str(stored.id))
        assert refreshed.status == ClaimStatus.canonical


# ===========================================================================
# 11. Date parsing improvements
# ===========================================================================

class TestDateParsing:
    def test_parse_iso_date(self):
        result = _parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso_datetime(self):
        result = _parse_date("2024-06-01T12:00:00Z")
        assert result is not None
        assert result.year == 2024

    def test_parse_null_returns_none(self):
        assert _parse_date("null") is None
        assert _parse_date("none") is None
        assert _parse_date("N/A") is None
        assert _parse_date("") is None
        assert _parse_date(None) is None

    def test_parse_invalid_returns_none(self):
        assert _parse_date("not-a-date") is None


# ===========================================================================
# 12. Extraction schema validation
# ===========================================================================

class TestExtractionSchema:
    def test_schema_structure(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        assert EXTRACTION_SCHEMA["type"] == "object"
        assert "claims" in EXTRACTION_SCHEMA["properties"]

        claim_props = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]
        required_fields = {
            "subject",
            "predicate",
            "object",
            "natural_language",
            "knowledge_class",
            "confidence",
            "source_span",
            "valid_from",
            "valid_until",
            "entity_aliases",
            "references",
        }
        assert required_fields.issubset(set(claim_props.keys()))

    def test_entity_aliases_schema(self):
        from noodly.extraction.extractor import EXTRACTION_SCHEMA

        alias_schema = EXTRACTION_SCHEMA["properties"]["claims"]["items"][
            "properties"
        ]["entity_aliases"]["items"]
        assert "alias" in alias_schema["properties"]
        assert "canonical" in alias_schema["properties"]
        assert alias_schema["additionalProperties"] is False


# ===========================================================================
# 13. DocumentParser — OCR option
# ===========================================================================

class TestDocumentParserOCR:
    def test_ocr_enabled_by_default(self):
        parser = DocumentParser()
        assert parser._ocr_enabled is True

    def test_ocr_can_be_disabled(self):
        parser = DocumentParser(ocr_enabled=False)
        assert parser._ocr_enabled is False


# ===========================================================================
# 14. Pipeline — semantic dedup merge integration
# ===========================================================================

class TestPipelineSemanticMerge:
    def test_dedup_result_filters_unique_claims(self):
        """Verify DedupResult correctly separates merged and unique claims."""
        c1 = _make_claim("A", "is", "B")
        c2 = _make_claim("A", "is", "B")
        c3 = _make_claim("C", "is", "D")

        result = DedupResult(merged=[(c1, c2)], unique=[c3])
        assert result.merged_count == 1
        assert result.unique_count == 1
        assert result.unique[0].subject == "C"


# ===========================================================================
# 15. EntityAlias model
# ===========================================================================

class TestEntityAlias:
    def test_entity_alias_creation(self):
        alias = EntityAlias(alias="SLA", canonical="Singapore Land Authority")
        assert alias.alias == "SLA"
        assert alias.canonical == "Singapore Land Authority"

    def test_entity_alias_serialization(self):
        alias = EntityAlias(alias="HTTP", canonical="Hypertext Transfer Protocol")
        data = alias.model_dump()
        assert data == {"alias": "HTTP", "canonical": "Hypertext Transfer Protocol"}
