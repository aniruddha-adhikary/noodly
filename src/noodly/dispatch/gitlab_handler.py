"""GitLab MR handler — creates merge requests for manual conflict resolution.

Supports both gitlab.com and self-hosted/enterprise GitLab instances via
configurable base URL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from noodly.dispatch.dispatcher import EventHandler, HandlerResult
from noodly.resolution.detector import ConflictPair
from noodly.tracking.changelog import ChangeEvent, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class GitLabConfig:
    """Configuration for GitLab integration."""

    url: str = "https://gitlab.com"
    token: str = ""
    project_id: str = ""
    target_branch: str = "main"
    knowledge_path: str = "knowledge"
    assignee_ids: list[int] | None = None
    labels: list[str] | None = None

    @property
    def api_url(self) -> str:
        """API base URL.

        Handles both instance URLs (``https://gitlab.com``) and full
        project URLs (``https://gitlab.com/org/repo``) by extracting
        the scheme + host automatically.
        """
        parsed = urlparse(self.url.rstrip("/"))
        base = f"{parsed.scheme}://{parsed.netloc}"
        return f"{base}/api/v4"


class GitLabClient:
    """HTTP client for GitLab API v4.

    Supports self-hosted and gitlab.com instances.

    Usage::

        client = GitLabClient(
            url="https://gitlab.mycompany.com",
            token="glpat-xxxxx",
            project_id="42",
        )
        mr = await client.create_merge_request(...)
    """

    def __init__(self, config: GitLabConfig) -> None:
        self._config = config
        self._session = None

    async def _get_session(self):
        """Lazy-init aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
            except ImportError:
                raise ImportError(
                    "aiohttp is required for GitLab integration. "
                    "Install with: pip install aiohttp"
                )
            self._session = aiohttp.ClientSession(
                headers={
                    "PRIVATE-TOKEN": self._config.token,
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def create_branch(self, branch_name: str, ref: str = "main") -> dict:
        """Create a new branch."""
        session = await self._get_session()
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/repository/branches"
        )
        async with session.post(
            url, json={"branch": branch_name, "ref": ref}
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error("GitLab create branch failed: %s", data)
            return data

    async def create_or_update_file(
        self,
        branch: str,
        file_path: str,
        content: str,
        commit_message: str,
    ) -> dict:
        """Create or update a file in the repository."""
        session = await self._get_session()
        encoded_path = file_path.replace("/", "%2F")
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/repository/files/{encoded_path}"
        )

        payload = {
            "branch": branch,
            "content": content,
            "commit_message": commit_message,
        }

        async with session.get(
            url, params={"ref": branch}
        ) as check_resp:
            if check_resp.status == 200:
                async with session.put(url, json=payload) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        logger.error("GitLab update file failed: %s", data)
                    return data
            else:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        logger.error("GitLab create file failed: %s", data)
                    return data

    async def create_merge_request(
        self,
        source_branch: str,
        title: str,
        description: str,
        target_branch: str | None = None,
        assignee_ids: list[int] | None = None,
        labels: list[str] | None = None,
    ) -> dict:
        """Create a merge request."""
        session = await self._get_session()
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/merge_requests"
        )
        payload: dict = {
            "source_branch": source_branch,
            "target_branch": target_branch or self._config.target_branch,
            "title": title,
            "description": description,
        }
        if assignee_ids or self._config.assignee_ids:
            payload["assignee_ids"] = assignee_ids or self._config.assignee_ids
        if labels or self._config.labels:
            payload["labels"] = ",".join(labels or self._config.labels or [])

        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error("GitLab create MR failed: %s", data)
            else:
                logger.info("GitLab MR created: %s", data.get("web_url", ""))
            return data

    async def get_merge_request(self, mr_iid: int) -> dict:
        """Get a merge request by IID."""
        session = await self._get_session()
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/merge_requests/{mr_iid}"
        )
        async with session.get(url) as resp:
            return await resp.json()

    async def add_mr_note(self, mr_iid: int, body: str) -> dict:
        """Add a comment to a merge request."""
        session = await self._get_session()
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/merge_requests/{mr_iid}/notes"
        )
        async with session.post(url, json={"body": body}) as resp:
            return await resp.json()

    async def commit_knowledge_files(
        self,
        branch: str,
        files: dict[str, str],
        commit_message: str,
    ) -> dict:
        """Commit multiple knowledge files in a single commit.

        Args:
            branch: Target branch name.
            files: {file_path: content} mapping.
            commit_message: Commit message.
        """
        session = await self._get_session()
        url = (
            f"{self._config.api_url}/projects/{self._config.project_id}"
            f"/repository/commits"
        )
        actions = []
        for path, content in files.items():
            full_path = f"{self._config.knowledge_path}/{path}"
            actions.append(
                {
                    "action": "create",
                    "file_path": full_path,
                    "content": content,
                }
            )

        payload = {
            "branch": branch,
            "commit_message": commit_message,
            "actions": actions,
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error("GitLab commit failed: %s", data)
            return data

    async def push_knowledge(
        self,
        files: dict[str, str],
        branch: str | None = None,
        commit_message: str = "Update knowledge files from noodly",
    ) -> dict:
        """Push rendered knowledge files to the GitLab repo.

        This commits all files to the specified branch (or target_branch).
        """
        target = branch or self._config.target_branch
        return await self.commit_knowledge_files(target, files, commit_message)


class GitLabMRHandler(EventHandler):
    """Creates GitLab merge requests for manual conflict resolution.

    When a conflict cannot be auto-resolved (score delta below threshold),
    this handler:
    1. Creates a branch named ``noodly/conflict-{conflict_id}``
    2. Commits both conflicting claims as separate Markdown files
    3. Opens an MR with the conflict description
    4. Human reviews and merges/closes to resolve

    Usage::

        handler = GitLabMRHandler(gitlab_config)
        dispatcher.register(handler, event_types=[ChangeType.conflict_detected])
    """

    name = "gitlab_mr"

    def __init__(self, config: GitLabConfig) -> None:
        self._config = config
        self._client = GitLabClient(config)

    def accepts(self, event: ChangeEvent) -> bool:
        return event.change_type == ChangeType.conflict_detected

    async def handle(self, event: ChangeEvent) -> HandlerResult:
        """Create a GitLab MR for a conflict event."""
        if not self._config.token:
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=False,
                action_taken="skipped",
                details={"reason": "No GitLab token configured"},
            )

        payload = event.payload or {}
        resolution_status = payload.get("resolution", "")

        if resolution_status == "auto":
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=True,
                action_taken="skipped",
                details={"reason": "Auto-resolved, no MR needed"},
            )

        try:
            mr_data = await self._create_conflict_mr(event)
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=True,
                action_taken="mr_created",
                details={
                    "mr_iid": mr_data.get("iid"),
                    "mr_url": mr_data.get("web_url", ""),
                },
            )
        except Exception:
            logger.exception("Failed to create GitLab MR for conflict %s", event.id)
            return HandlerResult(
                handler_name=self.name,
                event_id=event.id,
                success=False,
                action_taken="error",
            )

    async def escalate_conflict(self, conflict: ConflictPair) -> dict:
        """Create a GitLab MR for a ConflictPair (called by ConflictResolver)."""
        branch_name = f"noodly/conflict-{str(conflict.id)[:8]}"

        try:
            await self._client.create_branch(
                branch_name, self._config.target_branch
            )
        except Exception:
            logger.warning("Branch creation failed (may already exist)")

        claim_a_content = self._render_claim_markdown(conflict.claim_a, "Claim A")
        claim_b_content = self._render_claim_markdown(conflict.claim_b, "Claim B")

        files = {
            f"conflicts/{str(conflict.id)[:8]}/claim_a.md": claim_a_content,
            f"conflicts/{str(conflict.id)[:8]}/claim_b.md": claim_b_content,
            f"conflicts/{str(conflict.id)[:8]}/README.md": self._render_conflict_readme(conflict),
        }

        await self._client.commit_knowledge_files(
            branch_name,
            files,
            f"noodly: conflict detected - {conflict.description[:80]}",
        )

        mr_data = await self._client.create_merge_request(
            source_branch=branch_name,
            title=f"[noodly conflict] {conflict.description[:100]}",
            description=self._render_mr_description(conflict),
            labels=["noodly", "conflict-resolution"],
        )

        return {
            "mr_iid": mr_data.get("iid"),
            "mr_url": mr_data.get("web_url", ""),
            "branch": branch_name,
        }

    async def _create_conflict_mr(self, event: ChangeEvent) -> dict:
        """Create MR from a ChangeEvent."""
        conflict_id = str(event.id)[:8]
        branch_name = f"noodly/conflict-{conflict_id}"
        payload = event.payload or {}

        try:
            await self._client.create_branch(
                branch_name, self._config.target_branch
            )
        except Exception:
            logger.warning("Branch creation may have failed")

        description = payload.get("description", "Conflict detected")
        claim_a = payload.get("claim_a", "Unknown")
        claim_b = payload.get("claim_b", "Unknown")

        readme_content = (
            f"# Conflict: {description}\n\n"
            f"**Claim A:** {claim_a}\n\n"
            f"**Claim B:** {claim_b}\n\n"
            f"## Resolution\n\n"
            f"Review both claims and merge this MR to accept the resolution, "
            f"or close it to reject.\n"
        )

        files = {
            f"conflicts/{conflict_id}/README.md": readme_content,
        }

        await self._client.commit_knowledge_files(
            branch_name,
            files,
            f"noodly: conflict - {description[:80]}",
        )

        return await self._client.create_merge_request(
            source_branch=branch_name,
            title=f"[noodly conflict] {description[:100]}",
            description=(
                f"## Conflict Detected\n\n"
                f"**Claim A:** {claim_a}\n\n"
                f"**Claim B:** {claim_b}\n\n"
                f"### How to resolve\n\n"
                f"1. Review the conflicting claims in the "
                f"`{self._config.knowledge_path}/conflicts/"
                f"{conflict_id}/` directory\n"
                f"2. **Merge** this MR to accept the proposed resolution\n"
                f"3. **Close** this MR to reject and keep both claims\n\n"
                f"_Generated by noodly conflict resolution system_"
            ),
            labels=["noodly", "conflict-resolution"],
        )

    @staticmethod
    def _render_claim_markdown(claim, label: str) -> str:
        """Render a claim as Markdown for the MR."""
        lines = [
            f"# {label}",
            "",
            f"**Subject:** {claim.subject}",
            f"**Predicate:** {claim.predicate}",
            f"**Object:** {claim.object}",
            "",
            f"> {claim.natural_language}",
            "",
            f"- **Confidence:** {claim.confidence:.2f}",
            f"- **Truth Score:** {claim.truth_score:.3f}",
            f"- **Knowledge Class:** {claim.knowledge_class.value}",
            f"- **Status:** {claim.status.value}",
            f"- **Evidence count:** {len(claim.evidence)}",
            f"- **Created:** {claim.created_at.isoformat()}",
        ]
        if claim.valid_from:
            lines.append(f"- **Valid from:** {claim.valid_from.isoformat()}")
        if claim.valid_until:
            lines.append(f"- **Valid until:** {claim.valid_until.isoformat()}")
        return "\n".join(lines)

    @staticmethod
    def _render_conflict_readme(conflict: ConflictPair) -> str:
        """Render the conflict README for the MR."""
        return (
            f"# Conflict: {conflict.description}\n\n"
            f"**Type:** {conflict.conflict_type}\n"
            f"**Detected:** {conflict.detected_at.isoformat()}\n"
            f"**Detected by:** {conflict.detected_by}\n"
            f"**Score delta:** {conflict.score_delta:.3f}\n\n"
            f"## Claims\n\n"
            f"- **Claim A** ({conflict.claim_a.subject} {conflict.claim_a.predicate}): "
            f'"{conflict.claim_a.object}" (score={conflict.claim_a.truth_score:.3f})\n'
            f"- **Claim B** ({conflict.claim_b.subject} {conflict.claim_b.predicate}): "
            f'"{conflict.claim_b.object}" (score={conflict.claim_b.truth_score:.3f})\n\n'
            f"## Resolution\n\n"
            f"Review both claims (see `claim_a.md` and `claim_b.md`) and:\n"
            f"- **Merge** this MR to accept the resolution\n"
            f"- **Close** this MR to reject\n"
        )

    @staticmethod
    def _render_mr_description(conflict: ConflictPair) -> str:
        """Render the MR description."""
        return (
            f"## Conflict Detected\n\n"
            f"**Subject:** {conflict.claim_a.subject}\n"
            f"**Predicate:** {conflict.claim_a.predicate}\n\n"
            f"### Claim A\n"
            f"> {conflict.claim_a.natural_language}\n"
            f"- Object: `{conflict.claim_a.object}`\n"
            f"- Score: {conflict.claim_a.truth_score:.3f}\n\n"
            f"### Claim B\n"
            f"> {conflict.claim_b.natural_language}\n"
            f"- Object: `{conflict.claim_b.object}`\n"
            f"- Score: {conflict.claim_b.truth_score:.3f}\n\n"
            f"### Score Delta: {conflict.score_delta:.3f}\n\n"
            f"### How to resolve\n\n"
            f"1. Review the conflicting claims in the `conflicts/` directory\n"
            f"2. **Merge** this MR to accept Claim A as winner\n"
            f"3. **Close** this MR to reject and keep both claims\n\n"
            f"_Generated by noodly conflict resolution system_"
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
