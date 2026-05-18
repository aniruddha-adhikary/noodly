"""Resolution strategies — configurable algorithms for auto-resolving conflicts."""

from __future__ import annotations

from enum import Enum

from noodly.models.claims import Claim
from noodly.resolution.detector import ConflictPair


class AutoResolveStrategy(str, Enum):
    """Available strategies for automatic conflict resolution."""

    RECENCY_WINS = "recency_wins"
    AUTHORITY_WINS = "authority_wins"
    MAJORITY_WINS = "majority_wins"
    HIGHER_SCORE = "higher_score"


def resolve_by_strategy(
    conflict: ConflictPair,
    strategy: AutoResolveStrategy,
) -> tuple[Claim, Claim, str]:
    """Apply a strategy to pick winner and loser.

    Returns (winner, loser, rationale).
    """
    a = conflict.claim_a
    b = conflict.claim_b

    if strategy == AutoResolveStrategy.AUTHORITY_WINS:
        a_auth = _max_authority(a)
        b_auth = _max_authority(b)
        if a_auth >= b_auth:
            winner, loser = a, b
            rationale = (
                f"Authority: {a_auth:.3f} vs {b_auth:.3f}. "
                f"Claim '{a.object}' from higher-authority source wins."
            )
        else:
            winner, loser = b, a
            rationale = (
                f"Authority: {b_auth:.3f} vs {a_auth:.3f}. "
                f"Claim '{b.object}' from higher-authority source wins."
            )

    elif strategy == AutoResolveStrategy.RECENCY_WINS:
        if a.created_at >= b.created_at:
            winner, loser = a, b
            rationale = (
                f"Recency: '{a.object}' is newer "
                f"({a.created_at.date()} vs {b.created_at.date()})."
            )
        else:
            winner, loser = b, a
            rationale = (
                f"Recency: '{b.object}' is newer "
                f"({b.created_at.date()} vs {a.created_at.date()})."
            )

    elif strategy == AutoResolveStrategy.MAJORITY_WINS:
        a_support = len(a.supporting_evidence)
        b_support = len(b.supporting_evidence)
        if a_support >= b_support:
            winner, loser = a, b
            rationale = f"Majority: '{a.object}' has {a_support} vs {b_support} supporting sources."
        else:
            winner, loser = b, a
            rationale = f"Majority: '{b.object}' has {b_support} vs {a_support} supporting sources."

    elif strategy == AutoResolveStrategy.HIGHER_SCORE:
        if a.truth_score >= b.truth_score:
            winner, loser = a, b
            rationale = (
                f"Score: {a.truth_score:.3f} vs {b.truth_score:.3f}. "
                f"Claim '{a.object}' has higher composite truth score."
            )
        else:
            winner, loser = b, a
            rationale = (
                f"Score: {b.truth_score:.3f} vs {a.truth_score:.3f}. "
                f"Claim '{b.object}' has higher composite truth score."
            )
    else:
        winner, loser = a, b
        rationale = f"Fallback: first claim wins (unknown strategy '{strategy}')."

    return winner, loser, rationale


def _max_authority(claim: Claim) -> float:
    """Get the highest authority score from supporting evidence."""
    if claim.supporting_evidence:
        return max(ev.source_authority for ev in claim.supporting_evidence)
    return claim.authority
