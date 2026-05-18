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
    falkordb_database: str = "noodly"

    # Paths
    brain_dir: Path = Path("./brain")
    watch_dir: Path = Path("./inbox")

    # Graph
    group_id: str = "default"

    # Storage backend — set to true to store claims as Graphiti edges
    use_graphiti_backend: bool = False


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
