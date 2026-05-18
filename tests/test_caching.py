"""Tests for the caching module (parse_cache, extraction_cache, decision_cache, manager)."""

from __future__ import annotations

from noodly.caching.decision_cache import DecisionCache
from noodly.caching.extraction_cache import ExtractionCache
from noodly.caching.manager import CacheManager
from noodly.caching.parse_cache import ParseCache
from noodly.parsing.parser import ParsedDocument


# --- ParseCache ---


class TestParseCache:
    def test_put_and_get(self, tmp_path):
        cache = ParseCache(tmp_path)
        doc = ParsedDocument(
            title="Report", markdown="# Hello\nWorld",
            source_format="md", word_count=2, tables_detected=1,
        )
        cache.put("abc123", doc)
        result = cache.get("abc123")
        assert result is not None
        assert result.title == "Report"
        assert result.markdown == "# Hello\nWorld"
        assert result.word_count == 2
        assert result.tables_detected == 1

    def test_get_missing(self, tmp_path):
        cache = ParseCache(tmp_path)
        assert cache.get("nonexistent") is None

    def test_has(self, tmp_path):
        cache = ParseCache(tmp_path)
        assert not cache.has("abc")
        doc = ParsedDocument(title="t", markdown="x", source_format="md")
        cache.put("abc", doc)
        assert cache.has("abc")

    def test_invalidate(self, tmp_path):
        cache = ParseCache(tmp_path)
        doc = ParsedDocument(title="t", markdown="x", source_format="md")
        cache.put("abc", doc)
        assert cache.has("abc")
        cache.invalidate("abc")
        assert not cache.has("abc")
        assert cache.get("abc") is None


# --- ExtractionCache ---


class TestExtractionCache:
    def test_put_and_get(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        claims = [{"subject": "A", "predicate": "is", "object": "B"}]
        cache.put("chunk1", claims)
        result = cache.get("chunk1")
        assert result == claims

    def test_get_missing(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        assert cache.get("nope") is None

    def test_has(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        assert not cache.has("x")
        cache.put("x", [])
        assert cache.has("x")

    def test_invalidate(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        cache.put("x", [{"a": 1}])
        cache.invalidate("x")
        assert not cache.has("x")

    def test_invalidate_all(self, tmp_path):
        cache = ExtractionCache(tmp_path)
        cache.put("a", [])
        cache.put("b", [])
        cache.put("c", [])
        removed = cache.invalidate_all()
        assert removed == 3
        assert not cache.has("a")


# --- DecisionCache ---


class TestDecisionCache:
    def test_put_and_get_merge(self, tmp_path):
        cache = DecisionCache(tmp_path)
        cache.put_merge("SLA", "Singapore Land Authority", {"should_merge": True})
        result = cache.get_merge("SLA", "Singapore Land Authority")
        assert result is not None
        assert result["should_merge"] is True

    def test_get_merge_missing(self, tmp_path):
        cache = DecisionCache(tmp_path)
        assert cache.get_merge("X", "Y") is None

    def test_merge_key_order_independent(self, tmp_path):
        cache = DecisionCache(tmp_path)
        cache.put_merge("A", "B", {"merged": True})
        assert cache.get_merge("B", "A") is not None

    def test_merge_key_case_insensitive(self, tmp_path):
        cache = DecisionCache(tmp_path)
        cache.put_merge("Apple", "APPLE Inc", {"merged": True})
        assert cache.get_merge("apple", "apple inc") is not None

    def test_merge_count(self, tmp_path):
        cache = DecisionCache(tmp_path)
        assert cache.merge_count == 0
        cache.put_merge("A", "B", {"merged": True})
        cache.put_merge("C", "D", {"merged": False})
        assert cache.merge_count == 2

    def test_ontology(self, tmp_path):
        cache = DecisionCache(tmp_path)
        assert cache.get_ontology() == {}
        cache.put_ontology({"entity_types": ["Person", "Company"]})
        ont = cache.get_ontology()
        assert "Person" in ont["entity_types"]

    def test_persistence(self, tmp_path):
        cache1 = DecisionCache(tmp_path)
        cache1.put_merge("A", "B", {"merged": True})
        cache2 = DecisionCache(tmp_path)
        assert cache2.get_merge("A", "B") is not None
        assert cache2.merge_count == 1


# --- CacheManager ---


class TestCacheManager:
    def test_creates_subdirs(self, tmp_path):
        cache_dir = tmp_path / "cache"
        mgr = CacheManager(cache_dir)
        assert cache_dir.exists()
        assert mgr.parse is not None
        assert mgr.extraction is not None
        assert mgr.decisions is not None

    def test_invalidate_for_source(self, tmp_path):
        mgr = CacheManager(tmp_path / "cache")
        doc = ParsedDocument(title="t", markdown="x", source_format="md")
        mgr.parse.put("hash1", doc)
        assert mgr.parse.has("hash1")
        mgr.invalidate_for_source("hash1")
        assert not mgr.parse.has("hash1")

    def test_gc_returns_zero(self, tmp_path):
        mgr = CacheManager(tmp_path / "cache")
        assert mgr.gc() == 0
