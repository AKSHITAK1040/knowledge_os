from __future__ import annotations

from dataclasses import dataclass, field
import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    app_name: str = "KnowledgeOS"
    environment: str = os.getenv("KNOWLEDGEOS_ENV", "development")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    voyage_model: str = os.getenv("VOYAGE_MODEL", "voyage-3-large")
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/knowledgeos")
    redis_dsn: str = os.getenv("REDIS_DSN", "redis://localhost:6379/0")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    api_rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    max_agent_retries: int = int(os.getenv("MAX_AGENT_RETRIES", "2"))
    memory_decay_halflife_hours: float = float(os.getenv("MEMORY_DECAY_HALFLIFE_HOURS", "72"))
    local_state_path: str = os.getenv("KNOWLEDGEOS_STATE_PATH", ".knowledgeos/knowledgeos.sqlite3")
    enable_metrics: bool = field(default_factory=lambda: _env_bool("ENABLE_METRICS", True))


settings = Settings()
