"""Shared fixtures for noodly tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_brain(tmp_path):
    """Return a temporary brain directory."""
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir()
    return brain_dir


@pytest.fixture
def tmp_inbox(tmp_path):
    """Return a temporary inbox directory with sample files."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "notes.md").write_text(
        "# Team Notes\nJane owns the billing service.\nBob handles deployments."
    )
    (inbox / "process.txt").write_text("Deploys happen every Tuesday at 2pm.")
    return inbox
