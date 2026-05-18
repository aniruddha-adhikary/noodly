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
    source_artifact: str = ""
    author: str = ""
    author_role: str = ""
    source_authority: float = 0.5
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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

    # Truth maintenance
    superseded_by: UUID | None = None
    supersedes: UUID | None = None
    conflicts_with: list[UUID] = Field(default_factory=list)

    @property
    def supporting_evidence(self) -> list[ClaimEvidence]:
        """Evidence that supports this claim."""
        return [ev for ev in self.evidence if ev.supports]

    @property
    def refuting_evidence(self) -> list[ClaimEvidence]:
        """Evidence that contradicts this claim."""
        return [ev for ev in self.evidence if not ev.supports]

    @property
    def truth_score(self) -> float:
        """Composite score used for ranking and promotion decisions.

        Formula: confidence x authority x recency x corroboration x conflict_penalty
        - authority: max source_authority from supporting evidence, or self.authority
        - corroboration: scales with number of independent supporting sources
        - conflict_penalty: 0.7 per piece of refuting evidence
        """
        # Authority from evidence (use best supporting source)
        evidence_authority = self.authority
        if self.supporting_evidence:
            evidence_authority = max(ev.source_authority for ev in self.supporting_evidence)

        corroboration = min(1.0, len(self.supporting_evidence) * 0.25) if self.evidence else 0.1
        conflict_penalty = 0.7 ** len(self.refuting_evidence)

        # Additional penalty for known conflicts with other claims
        if self.conflicts_with:
            conflict_penalty *= 0.9 ** len(self.conflicts_with)

        return (
            self.confidence
            * evidence_authority
            * self.recency
            * corroboration
            * conflict_penalty
        )
