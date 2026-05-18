"""Configuration for Noodly."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = {"env_prefix": "NOODLY_", "env_file": ".env", "extra": "ignore"}

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_username: str | None = None
    falkordb_password: str | None = None
    falkordb_database: str = "default"

    # Paths
    brain_dir: Path = Path("./brain")
    watch_dir: Path = Path("./inbox")

    # Graph
    group_id: str = "default"

    # Phase 3: parsing & agents
    enable_qa_agent: bool = False
    enable_graph_agent: bool = False
    chunk_size: int = 6000
    qa_change_threshold: float = 0.05

    # Phase 4: storage backend
    storage_backend: str = "json"  # "json" or "postgresql"
    postgresql_dsn: str = ""

    # Phase 4/8: semantic dedup
    enable_semantic_dedup: bool = False
    semantic_dedup_threshold: float = 0.92
    embedding_model: str = "text-embedding-3-large"
    enable_ingestion_embeddings: bool = True  # embed claims at ingestion for semantic dedup
    embedding_dim: int = 3072  # dimension for text-embedding-3-large

    # Phase 8: claim promotion thresholds
    promote_threshold: float = 0.15  # truth_score threshold for candidate → unverified
    high_authority_threshold: float = 0.8  # source_authority threshold for auto-unverified
    corroboration_count: int = 2  # independent sources needed for corroborated

    # Phase 4: conflict resolution
    enable_conflict_resolution: bool = False
    auto_resolve_threshold: float = 0.3
    resolve_strategy: str = "authority_wins"
    conflict_similarity_threshold: float = 0.8

    # Phase 4: event dispatch
    enable_event_dispatch: bool = False
    audit_log_path: str = ""  # empty = brain_dir/audit.jsonl

    # Phase 4: Docling
    enable_docling: bool = False
    extraction_mode: str = "auto"  # auto, markitdown, docling, multi

    # Phase 7: parallel LLM dispatch
    llm_max_concurrent: int = 8
    llm_rate_limit_rpm: int = 500
    llm_retry_max: int = 3
    llm_request_timeout: float = 30.0

    # Phase 7: emission planning
    emission_mode: str = "incremental"  # incremental | full

    # Phase 7: topic clustering
    enable_topic_clustering: bool = True
    topic_model: str = "gpt-4o-mini"

    # Phase 4: GitLab integration
    gitlab_url: str = "https://gitlab.com"
    gitlab_token: str = ""
    gitlab_project_id: str = ""
    gitlab_target_branch: str = "main"
    gitlab_knowledge_path: str = "knowledge"
    enable_gitlab_handler: bool = False
    enable_gitlab_projection: bool = False

    # Phase 7: topic-aware authority
    authority_topic_inference: str = "llm"  # llm | keyword


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
