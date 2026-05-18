"""Claim models — the truth pipeline layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Lifecycle of a claim through the truth pipeline."""

    candidate = "candidate"
    unverified = "unverified"
    corroborated = "corroborated"
    owner_confirmed = "owner_confirmed"
    canonical = "canonical"
    superseded = "superseded"
    rejected = "rejected"


class KnowledgeClass(str, Enum):
    """Decay-policy category for knowledge."""

    stable = "stable"
    process = "process"
    tacit = "tacit"
    stateful = "stateful"


class ClaimEvidence(BaseModel):
    """Links a claim to a supporting or refuting source artifact."""

    artifact_id: UUID
    episode_id: str = ""
    supports: bool = True
    source_span: str = ""
    author: str = ""
    author_role: str = ""
    source_authority: float = 0.5


class Claim(BaseModel):
    """A normalized assertion extracted from evidence.

    Example: "Procedure PA-17 applies to permit amendments."

    Claims carry scores and metadata that the truth-maintenance engine
    uses to decide whether to promote them into WorkingTruth.
    """

    id: UUID = Field(default_factory=uuid4)
    subject: str
    predicate: str
    object: str
    natural_language: str = ""

    # Scoring
    confidence: float = 0.5
    authority: float = 0.5
    recency: float = 1.0
    specificity: float = 0.5

    # Status
    status: ClaimStatus = ClaimStatus.candidate
    knowledge_class: KnowledgeClass = KnowledgeClass.process

    # Evidence
    evidence: list[ClaimEvidence] = Field(default_factory=list)

    # Time
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    last_confirmed_at: datetime | None = None

    # Grouping
    group_id: str = "default"

    @property
    def truth_score(self) -> float:
        """Composite score used for ranking and promotion decisions."""
        corroboration = min(1.0, len(self.evidence) * 0.25) if self.evidence else 0.1
        conflict_penalty = 1.0
        for ev in self.evidence:
            if not ev.supports:
                conflict_penalty *= 0.7
        return (
            self.confidence
            * self.authority
            * self.recency
            * corroboration
            * conflict_penalty
        )
