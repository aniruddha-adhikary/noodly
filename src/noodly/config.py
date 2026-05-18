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


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
