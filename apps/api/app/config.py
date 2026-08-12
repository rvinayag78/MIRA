from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: apps/api/app/config.py → ../../..
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_CANDIDATES = (_REPO_ROOT / ".env", Path(".env"))
_ENV_FILES = tuple(str(p) for p in _ENV_CANDIDATES if p.exists())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://mira:mira@localhost:5432/mira"
    redis_url: str = "redis://localhost:6379"
    audio_dir: Path = Path("./data/audio")
    tts_dir: Path = Path("./data/tts")
    api_cors_origins: str = "http://localhost:3000"
    rerank_enabled: bool = False

    assemblyai_api_key: str = ""
    voyage_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""

    voyage_embed_model: str = "voyage-context-4"
    voyage_embed_dim: int = 1024
    voyage_rerank_model: str = "rerank-2.5"
    anthropic_route_model: str = "claude-haiku-4-5-20251001"
    anthropic_ground_model: str = "claude-sonnet-5-20250514"

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 8

    # Eval thresholds
    recall_at_k_threshold: float = 0.7
    faithfulness_threshold: float = 0.85

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
