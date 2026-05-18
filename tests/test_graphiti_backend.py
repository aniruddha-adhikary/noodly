"""Tests for the Graphiti storage backend.

These tests mock the Graphiti client to avoid requiring a running FalkorDB
instance, focusing on the serialization/deserialization logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.storage.graphiti_backend import (
    CLAIM_MARKER,
    GraphitiBackend,
    _claim_to_edge_attributes,
    _edge_to_claim,
)


def _make_claim(**overrides) -> Claim:
    defaults = dict(
        subject="PortNet",
        predicate="uses",
        object="Python",
        natural_language="PortNet uses Python.",
        confidence=0.8,
        authority=0.7,
        status=ClaimStatus.unverified,
        knowledge_class=KnowledgeClass.stable,
        group_id="test-group",
    )
    defaults.update(overrides)
    return Claim(**defaults)


def _make_mock_edge(claim: Claim):
    """Build a mock EntityEdge from a Claim for testing deserialization."""
    attrs = _claim_to_edge_attributes(claim)
    edge = MagicMock()
    edge.attributes = attrs
    edge.name = claim.predicate
    edge.fact = f"{claim.subject} | {claim.object}"
    edge.group_id = claim.group_id
    edge.uuid = str(claim.id)
    return edge


class TestClaimEdgeSerialization:
    def test_roundtrip(self) -> None:
        original = _make_claim()
        attrs = _claim_to_edge_attributes(original)

        assert attrs[CLAIM_MARKER] is True
        assert attrs["claim_id"] == str(original.id)
        assert attrs["status"] == "unverified"
        assert attrs["knowledge_class"] == "stable"
        assert attrs["confidence"] == 0.8

        edge = _make_mock_edge(original)
        restored = _edge_to_claim(edge)

        assert restored is not None
        assert restored.id == original.id
        assert restored.subject == "PortNet"
        assert restored.predicate == "uses"
        assert restored.object == "Python"
        assert restored.confidence == 0.8
        assert restored.authority == 0.7
        assert restored.status == ClaimStatus.unverified
        assert restored.knowledge_class == KnowledgeClass.stable

    def test_evidence_roundtrip(self) -> None:
        ev = ClaimEvidence(
            artifact_id="12345678-1234-1234-1234-123456789abc",
            supports=True,
            source_span="some text",
            author="Jane",
            source_authority=0.9,
        )
        original = _make_claim(evidence=[ev])
        edge = _make_mock_edge(original)
        restored = _edge_to_claim(edge)

        assert restored is not None
        assert len(restored.evidence) == 1
        assert restored.evidence[0].author == "Jane"
        assert restored.evidence[0].source_authority == 0.9

    def test_non_claim_edge_returns_none(self) -> None:
        edge = MagicMock()
        edge.attributes = {"some_key": "some_value"}
        assert _edge_to_claim(edge) is None

    def test_empty_attributes_returns_none(self) -> None:
        edge = MagicMock()
        edge.attributes = {}
        assert _edge_to_claim(edge) is None

    def test_optional_datetime_fields(self) -> None:
        original = _make_claim(
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2025, 12, 31, tzinfo=timezone.utc),
            last_confirmed_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
        )
        edge = _make_mock_edge(original)
        restored = _edge_to_claim(edge)

        assert restored is not None
        assert restored.valid_from == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert restored.valid_until == datetime(2025, 12, 31, tzinfo=timezone.utc)
        assert restored.last_confirmed_at == datetime(2024, 6, 15, tzinfo=timezone.utc)

    def test_none_datetime_fields(self) -> None:
        original = _make_claim()
        edge = _make_mock_edge(original)
        restored = _edge_to_claim(edge)

        assert restored is not None
        assert restored.valid_from is None
        assert restored.valid_until is None
        assert restored.last_confirmed_at is None

    def test_all_statuses_roundtrip(self) -> None:
        for status in ClaimStatus:
            original = _make_claim(status=status)
            edge = _make_mock_edge(original)
            restored = _edge_to_claim(edge)
            assert restored is not None
            assert restored.status == status

    def test_all_knowledge_classes_roundtrip(self) -> None:
        for kc in KnowledgeClass:
            original = _make_claim(knowledge_class=kc)
            edge = _make_mock_edge(original)
            restored = _edge_to_claim(edge)
            assert restored is not None
            assert restored.knowledge_class == kc


class TestGraphitiBackendAsync:
    @pytest.fixture()
    def mock_brain(self):
        brain = MagicMock()
        brain.group_id = "test-group"

        graphiti = MagicMock()
        graphiti.driver = MagicMock()
        graphiti.embedder = MagicMock()
        graphiti.add_triplet = AsyncMock()
        graphiti.search = AsyncMock(return_value=[])

        brain.get_graphiti.return_value = graphiti
        return brain, graphiti

    @pytest.fixture()
    def backend(self, mock_brain):
        brain, _ = mock_brain
        return GraphitiBackend(brain)

    def test_sync_methods_raise(self, backend: GraphitiBackend) -> None:
        claim = _make_claim()
        with pytest.raises(TypeError, match="async"):
            backend.load_claims()
        with pytest.raises(TypeError, match="async"):
            backend.save_claim(claim)
        with pytest.raises(TypeError, match="async"):
            backend.save_all({str(claim.id): claim})
        with pytest.raises(TypeError, match="async"):
            backend.delete_claim(str(claim.id))

    @pytest.mark.asyncio
    async def test_save_claim_calls_add_triplet(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)
        claim = _make_claim()

        await backend.save_claim_async(claim)

        graphiti.add_triplet.assert_called_once()
        args = graphiti.add_triplet.call_args
        source_node = args[0][0]
        edge = args[0][1]
        target_node = args[0][2]

        assert source_node.name == "PortNet"
        assert target_node.name == "Python"
        assert edge.name == "uses"
        assert edge.attributes[CLAIM_MARKER] is True

    @pytest.mark.asyncio
    async def test_load_claims_filters_non_claim_edges(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)

        claim_edge = _make_mock_edge(_make_claim())
        non_claim_edge = MagicMock()
        non_claim_edge.attributes = {"regular": "edge"}

        with patch(
            "noodly.storage.graphiti_backend.EntityEdge.get_by_group_ids",
            new_callable=AsyncMock,
            return_value=[claim_edge, non_claim_edge],
        ):
            claims = await backend.load_claims_async()

        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_load_claims_empty_group(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)

        with patch(
            "noodly.storage.graphiti_backend.EntityEdge.get_by_group_ids",
            new_callable=AsyncMock,
            side_effect=Exception("No edges found"),
        ):
            claims = await backend.load_claims_async()

        assert claims == {}

    @pytest.mark.asyncio
    async def test_save_all_async(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)
        c1 = _make_claim(subject="A", predicate="is", object="B")
        c2 = _make_claim(subject="C", predicate="has", object="D")
        claims = {str(c1.id): c1, str(c2.id): c2}

        await backend.save_all_async(claims)

        assert graphiti.add_triplet.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_claim_async(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)

        mock_edge = MagicMock()
        mock_edge.delete = AsyncMock()

        with patch(
            "noodly.storage.graphiti_backend.EntityEdge.get_by_uuid",
            new_callable=AsyncMock,
            return_value=mock_edge,
        ):
            await backend.delete_claim_async("some-uuid")

        mock_edge.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_claim_nonexistent(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)

        with patch(
            "noodly.storage.graphiti_backend.EntityEdge.get_by_uuid",
            new_callable=AsyncMock,
            side_effect=Exception("Not found"),
        ):
            await backend.delete_claim_async("nonexistent")

    @pytest.mark.asyncio
    async def test_search_similar_claims(self, mock_brain) -> None:
        brain, graphiti = mock_brain
        backend = GraphitiBackend(brain)

        claim = _make_claim()
        similar = _make_claim(
            subject="PortNet",
            predicate="built_with",
            object="Python 3.11",
        )
        similar_edge = _make_mock_edge(similar)
        graphiti.search = AsyncMock(return_value=[similar_edge])

        results = await backend.search_similar_claims(claim, limit=5)

        assert len(results) == 1
        assert results[0].predicate == "built_with"
        graphiti.search.assert_called_once()
