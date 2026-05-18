"""PostgreSQL storage backend — production-grade option with Alembic migrations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


class PostgreSQLBackend:
    """PostgreSQL-backed storage for the fact ledger.

    Uses asyncpg for async connections. Tables are managed via Alembic migrations.
    Install with: pip install noodly[postgresql]

    Usage::

        backend = PostgreSQLBackend("postgresql://user:pass@localhost/noodly")
        await backend.connect()
        claims = backend.load_claims()
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        """Establish connection pool."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pip install noodly[postgresql]"
            )
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        logger.info("Connected to PostgreSQL at %s", self._dsn.split("@")[-1])

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()

    async def ensure_tables(self) -> None:
        """Create tables if they don't exist (for quick setup without Alembic)."""
        if not self._pool:
            await self.connect()
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    knowledge_class TEXT NOT NULL DEFAULT 'process',
                    truth_score REAL NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject);
                CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
                CREATE INDEX IF NOT EXISTS idx_claims_class ON claims(knowledge_class);
                CREATE INDEX IF NOT EXISTS idx_claims_score ON claims(truth_score DESC);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    entity_id TEXT,
                    source_uri TEXT,
                    payload JSONB,
                    agent TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS resolutions (
                    id TEXT PRIMARY KEY,
                    conflict_id TEXT NOT NULL,
                    winner_id TEXT,
                    loser_id TEXT,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    resolved_by TEXT NOT NULL,
                    details JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_resolutions_conflict
                    ON resolutions(conflict_id);
            """)
        logger.info("PostgreSQL tables ensured")

    def load_claims(self) -> dict[str, Claim]:
        """Load all claims from PostgreSQL (sync wrapper)."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync load_claims from async context. "
                "Use load_claims_async instead."
            )
        return loop.run_until_complete(self.load_claims_async())

    async def load_claims_async(self) -> dict[str, Claim]:
        """Load all claims from PostgreSQL."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, data FROM claims")
            claims: dict[str, Claim] = {}
            for row in rows:
                try:
                    data = json.loads(row["data"])
                    claim = Claim(**data)
                    claims[str(claim.id)] = claim
                except Exception:
                    logger.warning("Failed to deserialize claim %s", row["id"])
            return claims

    def save_claim(self, claim: Claim) -> None:
        """Save a single claim (sync wrapper)."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use save_claim_async from async context.")
        loop.run_until_complete(self.save_claim_async(claim))

    async def save_claim_async(self, claim: Claim) -> None:
        """Save a single claim to PostgreSQL."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        data = json.dumps(claim.model_dump(mode="json"), default=str)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO claims (id, data, subject, predicate, object,
                                    status, knowledge_class, truth_score,
                                    created_at, updated_at)
                VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    status = EXCLUDED.status,
                    knowledge_class = EXCLUDED.knowledge_class,
                    truth_score = EXCLUDED.truth_score,
                    updated_at = EXCLUDED.updated_at
                """,
                str(claim.id),
                data,
                claim.subject,
                claim.predicate,
                claim.object,
                claim.status.value,
                claim.knowledge_class.value,
                claim.truth_score,
                claim.created_at,
                datetime.now(timezone.utc),
            )

    def save_all(self, claims: dict[str, Claim]) -> None:
        """Save all claims (sync wrapper)."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use save_all_async from async context.")
        loop.run_until_complete(self.save_all_async(claims))

    async def save_all_async(self, claims: dict[str, Claim]) -> None:
        """Bulk save all claims to PostgreSQL."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for claim in claims.values():
                    data = json.dumps(claim.model_dump(mode="json"), default=str)
                    await conn.execute(
                        """
                        INSERT INTO claims (id, data, subject, predicate, object,
                                            status, knowledge_class, truth_score,
                                            created_at, updated_at)
                        VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO UPDATE SET
                            data = EXCLUDED.data,
                            status = EXCLUDED.status,
                            knowledge_class = EXCLUDED.knowledge_class,
                            truth_score = EXCLUDED.truth_score,
                            updated_at = EXCLUDED.updated_at
                        """,
                        str(claim.id),
                        data,
                        claim.subject,
                        claim.predicate,
                        claim.object,
                        claim.status.value,
                        claim.knowledge_class.value,
                        claim.truth_score,
                        claim.created_at,
                        datetime.now(timezone.utc),
                    )

    def delete_claim(self, claim_id: str) -> None:
        """Delete a claim (sync wrapper)."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use delete_claim_async from async context.")
        loop.run_until_complete(self.delete_claim_async(claim_id))

    async def delete_claim_async(self, claim_id: str) -> None:
        """Delete a claim from PostgreSQL."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM claims WHERE id = $1", claim_id)

    async def record_audit(
        self,
        event_type: str,
        entity_id: str = "",
        source_uri: str = "",
        payload: dict | None = None,
        agent: str = "",
    ) -> None:
        """Record an audit log entry."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (event_type, entity_id, source_uri, payload, agent)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                """,
                event_type,
                entity_id,
                source_uri,
                json.dumps(payload or {}, default=str),
                agent,
            )

    async def record_resolution(
        self,
        resolution_id: str,
        conflict_id: str,
        winner_id: str | None,
        loser_id: str | None,
        strategy: str,
        confidence: float,
        resolved_by: str,
        details: dict | None = None,
    ) -> None:
        """Record a conflict resolution."""
        if not self._pool:
            await self.connect()
            await self.ensure_tables()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO resolutions (id, conflict_id, winner_id, loser_id,
                                         strategy, confidence, resolved_by, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                resolution_id,
                conflict_id,
                winner_id,
                loser_id,
                strategy,
                confidence,
                resolved_by,
                json.dumps(details or {}, default=str),
            )
