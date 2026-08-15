from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict

def _env_files() -> tuple[str, ...]:
    here = Path(__file__).resolve().parent
    candidates = [here / ".env", Path(".env")]
    for parent in here.parents:
        candidates.append(parent / ".env")
    seen: list[str] = []
    for path in candidates:
        resolved = str(path)
        if path.is_file() and resolved not in seen:
            seen.append(resolved)
    return tuple(seen)


_ENV_FILES = _env_files()


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

    def _asyncpg_parts(self) -> tuple[str, bool | None]:
        """Normalize DSN for asyncpg (scheme, SSL, Neon query params)."""
        url = self.database_url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        parsed = urlparse(url)
        ssl: bool | None = None
        kept: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            lower = key.lower()
            if lower in {"sslmode", "ssl", "channel_binding"}:
                if lower == "sslmode" and value.lower() in {"require", "verify-full", "verify-ca"}:
                    ssl = True
                elif lower == "sslmode" and value.lower() == "disable":
                    ssl = False
                continue
            kept.append((key, value))
        dsn = urlunparse(parsed._replace(query=urlencode(kept)))
        if ssl is None and "neon.tech" in dsn:
            ssl = True
        return dsn, ssl

    @property
    def asyncpg_dsn(self) -> str:
        return self._asyncpg_parts()[0]

    @property
    def asyncpg_ssl(self) -> bool | None:
        return self._asyncpg_parts()[1]


@lru_cache
def get_settings() -> Settings:
    return Settings()
