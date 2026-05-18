"""Tests for the agents module (toolkit, compact_serializer, context_scoper, model_router)."""

from __future__ import annotations

import json
from pathlib import Path

from noodly.agents.compact_serializer import CompactSerializer
from noodly.agents.context_scoper import ContextScoper
from noodly.agents.model_router import get_model
from noodly.agents.toolkit import TOOL_DEFINITIONS, AgentToolkit
from noodly.caching.manager import CacheManager
from noodly.models.claims import Claim, KnowledgeClass
from noodly.scoring.ledger import FactLedger
from noodly.tracking.changelog import ChangeEvent, ChangeLog, ChangeType


def _claim(subject: str, predicate: str, obj: str, **kw) -> Claim:
    return Claim(subject=subject, predicate=predicate, object=obj, **kw)


# --- CompactSerializer ---


class TestCompactSerializer:
    def test_claim_oneliner(self):
        c = _claim("Apple", "is", "Company", confidence=0.9)
        line = CompactSerializer.claim_oneliner(c)
        assert "Apple" in line
        assert "is" in line
        assert "Company" in line
        assert "0.90" in line

    def test_entity_oneliner(self):
        e = {"uuid": "abc12345", "name": "Google", "summary": "A tech company"}
        line = CompactSerializer.entity_oneliner(e)
        assert "Google" in line
        assert "A tech company" in line

    def test_claims_block(self):
        claims = [_claim("A", "is", "B"), _claim("C", "is", "D")]
        block = CompactSerializer.claims_block(claims)
        assert "A" in block
        assert "C" in block
        lines = block.strip().split("\n")
        assert len(lines) == 2

    def test_claims_block_truncation(self):
        claims = [_claim(f"E{i}", "is", "X") for i in range(60)]
        block = CompactSerializer.claims_block(claims, max_claims=10)
        assert "... and 50 more" in block

    def test_diff_summary_added(self):
        added = [_claim("A", "is", "B")]
        result = CompactSerializer.diff_summary(added, [], [])
        assert "ADDED (1)" in result

    def test_diff_summary_no_changes(self):
        result = CompactSerializer.diff_summary([], [], [])
        assert result == "No changes"


# --- ModelRouter ---


class TestModelRouter:
    def test_known_tasks(self):
        assert get_model("claim_extraction") == "gpt-4o-mini"
        assert get_model("ontology_alignment") == "gpt-4o"

    def test_no_llm_tasks(self):
        assert get_model("boilerplate_detection") is None
        assert get_model("table_syntax_check") is None

    def test_unknown_task_uses_default(self):
        assert get_model("unknown_task") == "gpt-4o-mini"
        assert get_model("unknown_task", default="gpt-4o") == "gpt-4o"


# --- ContextScoper ---


class TestContextScoper:
    def _make_ledger(self, tmp_path: Path, claims: list[Claim]) -> FactLedger:
        ledger = FactLedger(tmp_path / "ledger.json")
        for c in claims:
            ledger.add_claim(c)
        return ledger

    def test_claims_for_entities(self, tmp_path):
        claims = [
            _claim("Google", "is", "Company"),
            _claim("Apple", "is", "Company"),
            _claim("Google", "acquired", "YouTube"),
        ]
        ledger = self._make_ledger(tmp_path, claims)
        scoper = ContextScoper(ledger)
        result = scoper.claims_for_entities({"Google"})
        assert len(result) == 2

    def test_entities_from_claims(self, tmp_path):
        claims = [_claim("A", "is", "B"), _claim("C", "is", "D")]
        ledger = self._make_ledger(tmp_path, claims)
        scoper = ContextScoper(ledger)
        entities = scoper.entities_from_claims(claims)
        assert entities == {"A", "B", "C", "D"}

    def test_related_claims(self, tmp_path):
        c1 = _claim("Google", "is", "Company")
        c2 = _claim("Google", "acquired", "YouTube")
        c3 = _claim("Apple", "is", "Company")
        ledger = self._make_ledger(tmp_path, [c1, c2, c3])
        scoper = ContextScoper(ledger)
        related = scoper.related_claims([c1])
        assert any(c.predicate == "acquired" for c in related)


# --- AgentToolkit ---


class TestAgentToolkit:
    def _make_toolkit(self, tmp_path: Path, claims: list[Claim] | None = None) -> AgentToolkit:
        ledger = FactLedger(tmp_path / "ledger.json")
        for c in (claims or []):
            ledger.add_claim(c)
        changelog = ChangeLog(tmp_path / "changelog.json")
        cache = CacheManager(tmp_path / "cache")
        return AgentToolkit(ledger=ledger, changelog=changelog, cache=cache)

    def test_tool_definitions_valid(self):
        assert len(TOOL_DEFINITIONS) == 6
        for td in TOOL_DEFINITIONS:
            assert td["type"] == "function"
            assert "name" in td["function"]
            assert "parameters" in td["function"]

    def test_find_claims_by_entity(self, tmp_path):
        claims = [
            _claim("Google", "is", "Company"),
            _claim("Apple", "is", "Company"),
        ]
        tk = self._make_toolkit(tmp_path, claims)
        result = tk.execute("find_claims_by_entity", {"entity_name": "Google"})
        assert "Google" in result
        assert "Apple" not in result

    def test_find_claims_no_match(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        result = tk.execute("find_claims_by_entity", {"entity_name": "Nobody"})
        assert "No claims found" in result

    def test_find_similar_claims(self, tmp_path):
        claims = [_claim("Google", "is", "Company")]
        tk = self._make_toolkit(tmp_path, claims)
        result = tk.execute(
            "find_similar_claims",
            {"subject": "Google", "predicate": "is", "object": "Organization"},
        )
        assert "Google" in result

    def test_get_recent_changes(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        tk._changelog.emit(ChangeEvent(ChangeType.claim_added, entity_id="E1"))
        result = tk.execute("get_recent_changes", {"minutes": 60})
        assert "claim_added" in result

    def test_get_recent_changes_none(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        result = tk.execute("get_recent_changes", {"minutes": 1})
        assert "No changes" in result

    def test_get_entity_history(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        tk._changelog.emit(
            ChangeEvent(ChangeType.claim_added, entity_id="Google", payload={"note": "test"})
        )
        result = tk.execute("get_entity_history", {"entity_name": "Google"})
        assert "claim_added" in result

    def test_get_claim_coverage(self, tmp_path):
        claims = [
            _claim("Google", "is", "Company", knowledge_class=KnowledgeClass.stable),
            _claim("Google", "acquired", "YouTube", knowledge_class=KnowledgeClass.process),
        ]
        tk = self._make_toolkit(tmp_path, claims)
        result = tk.execute("get_claim_coverage", {"entity_name": "Google"})
        data = json.loads(result)
        assert data["total_claims"] == 2
        assert "stable" in data["by_class"]

    def test_read_source_section(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        result = tk.execute(
            "read_source_section",
            {"source_uri": "test.md", "section_heading": "Intro"},
        )
        assert "v1" in result

    def test_unknown_tool(self, tmp_path):
        tk = self._make_toolkit(tmp_path)
        result = tk.execute("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_fuzzy_match(self, tmp_path):
        assert AgentToolkit._fuzzy_match("hello world", "hello there") > 0.0
        assert AgentToolkit._fuzzy_match("hello world", "hello world") == 1.0
        assert AgentToolkit._fuzzy_match("a b c", "x y z") == 0.0
        assert AgentToolkit._fuzzy_match("", "hello") == 0.0
