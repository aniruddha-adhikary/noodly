"""Tests for the local filesystem connector — persistent hash tracking."""

from __future__ import annotations

import json

import pytest

from noodly.connectors.local_fs import LocalFSConnector


@pytest.mark.asyncio
async def test_scan_finds_files(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    artifacts = await connector.scan()
    assert len(artifacts) == 2
    titles = {a.title for a in artifacts}
    assert "notes.md" in titles
    assert "process.txt" in titles


@pytest.mark.asyncio
async def test_scan_idempotent(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    first = await connector.scan()
    assert len(first) == 2
    second = await connector.scan()
    assert len(second) == 0


@pytest.mark.asyncio
async def test_hash_file_created(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    await connector.scan()
    hash_path = tmp_inbox / ".hashes.json"
    assert hash_path.exists()
    data = json.loads(hash_path.read_text())
    assert "notes.md" in data
    assert "process.txt" in data


@pytest.mark.asyncio
async def test_hash_persistence_across_instances(tmp_inbox):
    c1 = LocalFSConnector(tmp_inbox)
    await c1.scan()

    c2 = LocalFSConnector(tmp_inbox)
    artifacts = await c2.scan()
    assert len(artifacts) == 0


@pytest.mark.asyncio
async def test_modified_file_detected(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    await connector.scan()

    (tmp_inbox / "notes.md").write_text("# Updated content\nNew information here.")

    c2 = LocalFSConnector(tmp_inbox)
    artifacts = await c2.scan()
    assert len(artifacts) == 1
    assert artifacts[0].title == "notes.md"


@pytest.mark.asyncio
async def test_new_file_detected(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    await connector.scan()

    (tmp_inbox / "extra.md").write_text("Extra content.")

    c2 = LocalFSConnector(tmp_inbox)
    artifacts = await c2.scan()
    assert len(artifacts) == 1
    assert artifacts[0].title == "extra.md"


@pytest.mark.asyncio
async def test_hashes_json_skipped(tmp_inbox):
    connector = LocalFSConnector(tmp_inbox)
    await connector.scan()

    c2 = LocalFSConnector(tmp_inbox)
    artifacts = await c2.scan()
    titles = {a.title for a in artifacts}
    assert ".hashes.json" not in titles


@pytest.mark.asyncio
async def test_unsupported_extension_skipped(tmp_inbox):
    (tmp_inbox / "image.png").write_bytes(b"\x89PNG")
    connector = LocalFSConnector(tmp_inbox)
    artifacts = await connector.scan()
    titles = {a.title for a in artifacts}
    assert "image.png" not in titles


@pytest.mark.asyncio
async def test_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    connector = LocalFSConnector(empty)
    artifacts = await connector.scan()
    assert len(artifacts) == 0


@pytest.mark.asyncio
async def test_missing_directory(tmp_path):
    connector = LocalFSConnector(tmp_path / "nonexistent")
    artifacts = await connector.scan()
    assert len(artifacts) == 0
