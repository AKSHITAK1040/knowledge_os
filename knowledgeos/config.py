from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file if present
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    try:
        for line in _env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    app_name: str = "KnowledgeOS"
    environment: str = os.getenv("KNOWLEDGEOS_ENV", "development")

    # Primary LLM: Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Optional Providers
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    voyage_model: str = os.getenv("VOYAGE_MODEL", "voyage-3-large")
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/knowledgeos")
    redis_dsn: str = os.getenv("REDIS_DSN", "redis://localhost:6379/0")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")

    # Platform settings
    api_rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    max_agent_retries: int = int(os.getenv("MAX_AGENT_RETRIES", "2"))
    memory_decay_halflife_hours: float = float(os.getenv("MEMORY_DECAY_HALFLIFE_HOURS", "72"))
    local_state_path: str = os.getenv("KNOWLEDGEOS_STATE_PATH", ".knowledgeos/knowledgeos.sqlite3")
    enable_metrics: bool = field(default_factory=lambda: _env_bool("ENABLE_METRICS", True))


settings = Settings()
