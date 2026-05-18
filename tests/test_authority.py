"""Tests for the source authority registry."""

from __future__ import annotations

from noodly.scoring.authority import DEFAULT_AUTHORITY, AuthorityRegistry


def test_get_default(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    assert registry.get("unknown@test.com") == DEFAULT_AUTHORITY


def test_set_and_get(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    registry.set("jane@company.com", 0.9)
    assert registry.get("jane@company.com") == 0.9


def test_clamp_weight(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    registry.set("high", 1.5)
    assert registry.get("high") == 1.0
    registry.set("low", -0.3)
    assert registry.get("low") == 0.0


def test_remove(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    registry.set("bob", 0.7)
    assert registry.remove("bob") is True
    assert registry.get("bob") == DEFAULT_AUTHORITY
    assert registry.remove("bob") is False


def test_list_sources(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    registry.set("alice", 0.8)
    registry.set("bob", 0.6)
    sources = registry.list_sources()
    assert sources == {"alice": 0.8, "bob": 0.6}


def test_persistence(tmp_path):
    path = tmp_path / "authority.json"
    r1 = AuthorityRegistry(path)
    r1.set("jane", 0.95)

    r2 = AuthorityRegistry(path)
    assert r2.get("jane") == 0.95


def test_count(tmp_path):
    registry = AuthorityRegistry(tmp_path / "authority.json")
    assert registry.count == 0
    registry.set("a", 0.5)
    registry.set("b", 0.6)
    assert registry.count == 2
