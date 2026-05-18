"""Emission planner — computes minimal file operations for projection.

Maintains a ``_manifest.json`` that tracks content hashes per file,
enabling incremental updates instead of full re-renders.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass
class PlannedFile:
    """A file that needs to be written."""

    path: str
    content: str
    claim_ids: list[str] = field(default_factory=list)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class EmissionPlan:
    """What the projector should do this cycle."""

    files_to_create: list[PlannedFile] = field(default_factory=list)
    files_to_update: list[PlannedFile] = field(default_factory=list)
    files_to_delete: list[str] = field(default_factory=list)
    files_unchanged: int = 0
    reason: str = "incremental"

    @property
    def total_changes(self) -> int:
        return len(self.files_to_create) + len(self.files_to_update) + len(self.files_to_delete)

    @property
    def summary(self) -> str:
        parts = []
        if self.files_to_create:
            parts.append(f"{len(self.files_to_create)} create")
        if self.files_to_update:
            parts.append(f"{len(self.files_to_update)} update")
        if self.files_to_delete:
            parts.append(f"{len(self.files_to_delete)} delete")
        if self.files_unchanged:
            parts.append(f"{self.files_unchanged} unchanged")
        return f"[{self.reason}] {', '.join(parts) or 'no changes'}"


class EmissionManifest:
    """Tracks what was emitted and content hashes for incremental updates."""

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._data: dict = {"version": MANIFEST_VERSION, "files": {}, "claim_to_files": {}}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                if data.get("version") == MANIFEST_VERSION:
                    self._data = data
                else:
                    logger.info("Manifest version mismatch, starting fresh")
            except (json.JSONDecodeError, ValueError):
                logger.warning("Could not load manifest from %s", self._path)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data["last_emission"] = datetime.now(timezone.utc).isoformat()
        self._path.write_text(json.dumps(self._data, indent=2))

    def get_file_hash(self, path: str) -> str | None:
        """Get the stored content hash for a file."""
        entry = self._data["files"].get(path)
        if entry:
            return entry.get("content_hash")
        return None

    def get_all_files(self) -> set[str]:
        """Get all file paths tracked by the manifest."""
        return set(self._data["files"].keys())

    def get_files_for_claims(self, claim_ids: set[str]) -> set[str]:
        """Find which files contain any of the given claim IDs."""
        affected: set[str] = set()
        c2f = self._data.get("claim_to_files", {})
        for cid in claim_ids:
            if cid in c2f:
                affected.update(c2f[cid])
        return affected

    def update(self, plan: EmissionPlan) -> None:
        """Update manifest after executing a plan."""
        now = datetime.now(timezone.utc).isoformat()

        for pf in plan.files_to_create + plan.files_to_update:
            self._data["files"][pf.path] = {
                "content_hash": pf.content_hash,
                "claim_ids": pf.claim_ids,
                "last_written": now,
            }
            for cid in pf.claim_ids:
                self._data.setdefault("claim_to_files", {}).setdefault(cid, [])
                if pf.path not in self._data["claim_to_files"][cid]:
                    self._data["claim_to_files"][cid].append(pf.path)

        for path in plan.files_to_delete:
            file_entry = self._data["files"].pop(path, None)
            if file_entry:
                for cid in file_entry.get("claim_ids", []):
                    c2f = self._data.get("claim_to_files", {})
                    if cid in c2f and path in c2f[cid]:
                        c2f[cid].remove(path)
                        if not c2f[cid]:
                            del c2f[cid]

        self.save()

    @property
    def is_empty(self) -> bool:
        return not self._data.get("files")


class EmissionPlanner:
    """Plans minimal file operations for projection.

    Usage::

        planner = EmissionPlanner(brain_dir)
        plan = planner.plan(rendered_files, changed_claim_ids)
        planner.execute(plan)
    """

    def __init__(self, brain_dir: Path) -> None:
        self._brain_dir = brain_dir
        self._manifest = EmissionManifest(brain_dir / "_manifest.json")

    def plan(
        self,
        rendered_files: dict[str, tuple[str, list[str]]],
        changed_claim_ids: set[str] | None = None,
        force_full: bool = False,
    ) -> EmissionPlan:
        """Compute the minimal set of file operations needed.

        Args:
            rendered_files: mapping of ``{path: (content, [claim_ids])}``
            changed_claim_ids: if provided, only check files containing these claims
            force_full: if True, compare all files regardless of changes

        Returns:
            An ``EmissionPlan`` with create/update/delete lists.
        """
        if self._manifest.is_empty or force_full:
            return self._plan_full(rendered_files)

        if changed_claim_ids is not None:
            return self._plan_incremental(rendered_files, changed_claim_ids)

        return self._plan_full(rendered_files)

    def _plan_full(self, rendered_files: dict[str, tuple[str, list[str]]]) -> EmissionPlan:
        """Full plan: compare all rendered files with manifest."""
        plan = EmissionPlan(reason="full")
        existing_files = self._manifest.get_all_files()
        rendered_paths = set(rendered_files.keys())

        for path, (content, claim_ids) in rendered_files.items():
            pf = PlannedFile(path=path, content=content, claim_ids=claim_ids)
            old_hash = self._manifest.get_file_hash(path)

            if old_hash is None:
                plan.files_to_create.append(pf)
            elif old_hash != pf.content_hash:
                plan.files_to_update.append(pf)
            else:
                plan.files_unchanged += 1

        for path in existing_files - rendered_paths:
            plan.files_to_delete.append(path)

        return plan

    def _plan_incremental(
        self,
        rendered_files: dict[str, tuple[str, list[str]]],
        changed_claim_ids: set[str],
    ) -> EmissionPlan:
        """Incremental plan: only check files affected by changed claims."""
        plan = EmissionPlan(reason="incremental")

        affected_files = self._manifest.get_files_for_claims(changed_claim_ids)
        # Always check index and any new files
        affected_files.add("index.md")

        for path, (content, claim_ids) in rendered_files.items():
            pf = PlannedFile(path=path, content=content, claim_ids=claim_ids)
            old_hash = self._manifest.get_file_hash(path)

            if old_hash is None:
                plan.files_to_create.append(pf)
            elif path in affected_files and old_hash != pf.content_hash:
                plan.files_to_update.append(pf)
            else:
                plan.files_unchanged += 1

        return plan

    def execute(self, plan: EmissionPlan) -> int:
        """Write planned files to disk. Returns files written."""
        written = 0

        for pf in plan.files_to_create + plan.files_to_update:
            full_path = self._brain_dir / pf.path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(pf.content)
            written += 1

        for path in plan.files_to_delete:
            full_path = self._brain_dir / path
            if full_path.exists():
                full_path.unlink()
                logger.info("Deleted orphaned file: %s", path)

        self._manifest.update(plan)

        logger.info("Emission: %s → %d files written", plan.summary, written)
        return written

    @property
    def manifest(self) -> EmissionManifest:
        return self._manifest
