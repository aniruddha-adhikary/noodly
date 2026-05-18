"""GitLab knowledge projector — syncs rendered Markdown to a GitLab repo.

Supports two modes:
- **Full sync**: regenerates all files, compares with remote, commits only diffs.
- **Incremental sync**: given changed claim subjects, updates only affected files.

Uses the Commits API to batch create/update/delete files in a single commit,
keeping the GitLab history clean and meaningful.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from noodly.dispatch.gitlab_handler import GitLabClient, GitLabConfig
from noodly.models.claims import Claim
from noodly.projection.markdown import _frontmatter, _slugify

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a GitLab sync operation."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0
    commit_sha: str = ""

    @property
    def total_changes(self) -> int:
        return len(self.created) + len(self.updated) + len(self.deleted)

    @property
    def summary(self) -> str:
        parts = []
        if self.created:
            parts.append(f"{len(self.created)} created")
        if self.updated:
            parts.append(f"{len(self.updated)} updated")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        if self.unchanged:
            parts.append(f"{self.unchanged} unchanged")
        return ", ".join(parts) or "no changes"


class GitLabProjector:
    """Projects claim Markdown files to a GitLab repository.

    Uses the existing ``MarkdownProjector`` to render files locally,
    then syncs them to GitLab via the API — either as a full sync
    (compare everything) or incremental (only affected entities).

    Usage::

        projector = GitLabProjector(config)
        result = await projector.sync_full(claims)
        result = await projector.sync_incremental(claims, changed_subjects={"HTTP/2"})
    """

    def __init__(self, config: GitLabConfig) -> None:
        self._config = config
        self._client = GitLabClient(config)

    async def sync_full(
        self,
        claims: list[Claim],
        commit_message: str = "",
    ) -> SyncResult:
        """Full sync: render all claims, compare with remote, commit diffs.

        Steps:
        1. Render all claims to in-memory Markdown (reusing MarkdownProjector logic).
        2. Fetch the existing file tree from GitLab.
        3. Diff: create new files, update changed files, delete removed files.
        4. Commit all changes in a single API call.
        """
        if not self._config.token:
            logger.warning("GitLab token not configured, skipping sync")
            return SyncResult()

        local_files = self._render_all(claims)
        remote_files = await self._list_remote_files()
        return await self._commit_diff(
            local_files,
            remote_files,
            commit_message or self._auto_message(claims, mode="full"),
        )

    async def sync_incremental(
        self,
        claims: list[Claim],
        changed_subjects: set[str],
        commit_message: str = "",
    ) -> SyncResult:
        """Incremental sync: only update files for changed entities.

        Steps:
        1. Filter claims to only those with subjects in ``changed_subjects``.
        2. Render affected entity pages and per-class claim files.
        3. Fetch only the affected remote files (by path prefix).
        4. Commit targeted updates.
        """
        if not self._config.token:
            logger.warning("GitLab token not configured, skipping sync")
            return SyncResult()

        affected_claims = [c for c in claims if c.subject in changed_subjects]
        if not affected_claims:
            return SyncResult()

        local_files = self._render_all(affected_claims)

        # Also re-render index with full claim list
        local_files["index.md"] = self._render_index(claims)

        remote_files = {}
        for path in local_files:
            content = await self._get_remote_file(path)
            if content is not None:
                remote_files[path] = content

        return await self._commit_diff(
            local_files,
            remote_files,
            commit_message
            or self._auto_message(claims, mode="incremental", subjects=changed_subjects),
            incremental=True,
        )

    def _render_all(self, claims: list[Claim]) -> dict[str, str]:
        """Render claims to an in-memory {path: content} dict.

        Mirrors MarkdownProjector's output structure but writes to memory
        instead of disk.
        """
        from datetime import datetime, timezone

        from noodly.models.claims import ClaimStatus

        files: dict[str, str] = {}

        # Group by subject for entity pages
        by_subject: dict[str, list[Claim]] = {}
        for claim in claims:
            by_subject.setdefault(claim.subject, []).append(claim)

        excluded = (ClaimStatus.superseded, ClaimStatus.rejected)

        # Entity pages
        for subject, subject_claims in by_subject.items():
            slug = _slugify(subject)
            active = [c for c in subject_claims if c.status not in excluded]
            active.sort(key=lambda c: c.truth_score, reverse=True)

            fm = _frontmatter(
                {
                    "entity": subject,
                    "claim_count": len(active),
                    "avg_truth_score": round(
                        sum(c.truth_score for c in active) / max(len(active), 1), 3
                    ),
                    "last_updated": datetime.now(timezone.utc),
                    "sources": list({str(ev.artifact_id)[:8] for c in active for ev in c.evidence})[
                        :10
                    ],
                }
            )

            lines = [fm, "", f"# {subject}", ""]
            for claim in active:
                score_bar = "█" * int(claim.truth_score * 10)
                lines.append(
                    f"- **[{claim.status.value}]** {claim.natural_language}  "
                    f"`score={claim.truth_score:.2f}` {score_bar}"
                )
                if claim.evidence:
                    ev = claim.evidence[0]
                    if ev.source_span:
                        span_preview = ev.source_span[:120].replace("\n", " ")
                        lines.append(f"  > _{span_preview}_")
                lines.append("")

            files[f"entities/{slug}.md"] = "\n".join(lines)

        # Per-class claim files
        for claim in claims:
            if claim.status in excluded:
                continue
            slug = _slugify(f"{claim.subject}-{claim.predicate}-{claim.object}")
            path = f"claims/{claim.knowledge_class.value}/{slug}.md"

            fm = _frontmatter(
                {
                    "claim_id": str(claim.id),
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "status": claim.status.value,
                    "truth_score": round(claim.truth_score, 3),
                    "confidence": claim.confidence,
                    "knowledge_class": claim.knowledge_class.value,
                    "created_at": claim.created_at,
                    "last_confirmed": claim.last_confirmed_at or claim.created_at,
                    "evidence_count": len(claim.evidence),
                }
            )

            lines = [fm, "", f"# {claim.natural_language}", ""]
            lines.append(f"**Subject:** {claim.subject}  ")
            lines.append(f"**Predicate:** {claim.predicate}  ")
            lines.append(f"**Object:** {claim.object}  ")
            lines.append("")

            if claim.evidence:
                lines.append("## Evidence")
                lines.append("")
                for ev in claim.evidence:
                    direction = "supports" if ev.supports else "contradicts"
                    lines.append(f"- [{direction}] artifact `{str(ev.artifact_id)[:8]}`")
                    if ev.source_artifact:
                        lines.append(f"  - Source: {ev.source_artifact}")
                    if ev.author:
                        lines.append(f"  - Author: {ev.author}")
                    if ev.source_span:
                        span = ev.source_span[:200].replace("\n", " ")
                        lines.append(f"  > _{span}_")
                    lines.append("")

            files[path] = "\n".join(lines)

        # Index
        files["index.md"] = self._render_index(claims)

        return files

    def _render_index(self, claims: list[Claim]) -> str:
        """Render the top-level index.md."""
        from datetime import datetime, timezone

        from noodly.models.claims import ClaimStatus

        excluded = (ClaimStatus.superseded, ClaimStatus.rejected)
        active = [c for c in claims if c.status not in excluded]
        by_status: dict[str, int] = {}
        for c in active:
            by_status[c.status.value] = by_status.get(c.status.value, 0) + 1

        subjects = sorted({c.subject for c in active})

        fm = _frontmatter(
            {
                "title": "Noodly Brain Index",
                "total_claims": len(active),
                "last_projected": datetime.now(timezone.utc),
            }
        )

        lines = [fm, "", "# Noodly Brain", ""]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total active claims:** {len(active)}")
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

        if subjects:
            lines.append("## Entities")
            lines.append("")
            for subj in subjects:
                slug = _slugify(subj)
                lines.append(f"- [{subj}](entities/{slug}.md)")
            lines.append("")

        return "\n".join(lines)

    async def _list_remote_files(self) -> dict[str, str]:
        """Fetch the full file tree under knowledge_path from GitLab."""
        session = await self._client._get_session()
        knowledge_path = self._config.knowledge_path
        result: dict[str, str] = {}

        try:
            # Use repository tree API with recursive flag
            url = f"{self._config.api_url}/projects/{self._config.project_id}/repository/tree"
            page = 1
            while True:
                async with session.get(
                    url,
                    params={
                        "ref": self._config.target_branch,
                        "path": knowledge_path,
                        "recursive": "true",
                        "per_page": "100",
                        "page": str(page),
                    },
                ) as resp:
                    if resp.status != 200:
                        break
                    items = await resp.json()
                    if not items:
                        break
                    for item in items:
                        if item.get("type") == "blob":
                            # Strip knowledge_path prefix to get relative path
                            full_path = item["path"]
                            rel = full_path
                            if full_path.startswith(f"{knowledge_path}/"):
                                rel = full_path[len(knowledge_path) + 1 :]
                            result[rel] = ""  # content fetched lazily
                    page += 1
        except Exception:
            logger.exception("Failed to list remote files from GitLab")

        return result

    async def _get_remote_file(self, rel_path: str) -> str | None:
        """Fetch a single file's content from GitLab."""
        session = await self._client._get_session()
        full_path = f"{self._config.knowledge_path}/{rel_path}"
        encoded = full_path.replace("/", "%2F")
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/repository/files/{encoded}/raw"
        )

        try:
            async with session.get(url, params={"ref": self._config.target_branch}) as resp:
                if resp.status == 200:
                    return await resp.text()
                return None
        except Exception:
            return None

    async def _commit_diff(
        self,
        local_files: dict[str, str],
        remote_files: dict[str, str],
        commit_message: str,
        incremental: bool = False,
    ) -> SyncResult:
        """Compare local vs remote and commit the diff."""
        result = SyncResult()
        actions: list[dict] = []
        knowledge_path = self._config.knowledge_path

        # Determine creates and updates
        for rel_path, content in local_files.items():
            full_path = f"{knowledge_path}/{rel_path}"
            if rel_path not in remote_files:
                actions.append(
                    {
                        "action": "create",
                        "file_path": full_path,
                        "content": content,
                    }
                )
                result.created.append(rel_path)
            else:
                # For full sync, fetch content to compare
                remote_content = await self._get_remote_file(rel_path)
                if remote_content is not None and self._content_changed(content, remote_content):
                    actions.append(
                        {
                            "action": "update",
                            "file_path": full_path,
                            "content": content,
                        }
                    )
                    result.updated.append(rel_path)
                else:
                    result.unchanged += 1

        # For full sync, detect deletions (files in remote but not local)
        if not incremental:
            for rel_path in remote_files:
                if rel_path not in local_files:
                    full_path = f"{knowledge_path}/{rel_path}"
                    actions.append(
                        {
                            "action": "delete",
                            "file_path": full_path,
                        }
                    )
                    result.deleted.append(rel_path)

        if not actions:
            logger.info("GitLab sync: no changes to commit")
            return result

        # Commit all changes in a single API call
        session = await self._client._get_session()
        url = f"{self._config.api_url}/projects/{self._config.project_id}/repository/commits"
        payload = {
            "branch": self._config.target_branch,
            "commit_message": commit_message,
            "actions": actions,
        }

        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    logger.error("GitLab commit failed: %s", data)
                else:
                    result.commit_sha = data.get("id", "")
                    logger.info(
                        "GitLab sync committed: %s (%s)",
                        result.commit_sha[:8],
                        result.summary,
                    )
        except Exception:
            logger.exception("Failed to commit to GitLab")

        return result

    @staticmethod
    def _content_changed(local: str, remote: str) -> bool:
        """Compare content by stripping volatile frontmatter fields.

        The ``last_updated`` and ``last_projected`` fields change every run,
        so we hash everything except those lines for comparison.
        """

        def _stable_hash(text: str) -> str:
            lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("last_updated:") or stripped.startswith("last_projected:"):
                    continue
                lines.append(line)
            return hashlib.sha256("\n".join(lines).encode()).hexdigest()

        return _stable_hash(local) != _stable_hash(remote)

    @staticmethod
    def _auto_message(
        claims: list[Claim],
        mode: str = "full",
        subjects: set[str] | None = None,
    ) -> str:
        """Generate a commit message."""
        if mode == "incremental" and subjects:
            subj_list = ", ".join(sorted(subjects)[:5])
            suffix = f" (+{len(subjects) - 5} more)" if len(subjects) > 5 else ""
            return f"noodly: update knowledge for {subj_list}{suffix}"
        return f"noodly: full sync — {len(claims)} claims"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
