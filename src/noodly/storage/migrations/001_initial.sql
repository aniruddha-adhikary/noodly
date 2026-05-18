-- Initial schema for noodly PostgreSQL backend
-- Applied via Alembic or manually

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

CREATE INDEX IF NOT EXISTS idx_resolutions_conflict ON resolutions(conflict_id);
