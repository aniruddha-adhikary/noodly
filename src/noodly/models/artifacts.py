"""Source artifact models — the raw evidence layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Where the artifact originated."""

    local_file = "local_file"
    slack_message = "slack_message"
    email = "email"
    document = "document"
    meeting_transcript = "meeting_transcript"
    github_issue = "github_issue"
    github_pr = "github_pr"
    notion_page = "notion_page"
    web_page = "web_page"
    manual = "manual"


class SourceArtifact(BaseModel):
    """An immutable record of raw evidence before any extraction.

    Every piece of ingested content — a file, a Slack message, an email —
    becomes a SourceArtifact first.  It is never mutated; updates create
    new artifacts that reference the original via ``replaces_id``.
    """

    id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    source_uri: str = ""
    title: str = ""
    body: str = ""
    author: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_created_at: datetime | None = None

    # Metadata
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    # Lineage
    replaces_id: UUID | None = None
