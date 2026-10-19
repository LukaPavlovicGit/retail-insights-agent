from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = None
    gcp_project: str | None = None
    primary_model: str = "gemini-3.8-flash"
    fallback_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    dataset: str = "bigquery-public-data.thelook_ecommerce"
    db_path: Path = Path("ria.db")
    data_dir: Path = Path("data")
    bq_max_bytes: int = 1_000_000_000
    bq_timeout_s: int = 60
    max_repairs: int = 2
    max_llm_calls_per_turn: int = 8
    row_cap: int = 200
    window_turns: int = 8

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file, override=False)
        g = os.environ.get
        return cls(
            gemini_api_key=g("GEMINI_API_KEY") or None,
            gcp_project=g("GOOGLE_CLOUD_PROJECT") or None,
            primary_model=g("RIA_PRIMARY_MODEL", cls.primary_model),
            fallback_model=g("RIA_FALLBACK_MODEL", cls.fallback_model),
            db_path=Path(g("RIA_DB_PATH", "ria.db")),
            data_dir=Path(g("RIA_DATA_DIR", "data")),
            bq_max_bytes=int(g("RIA_BQ_MAX_BYTES", str(cls.bq_max_bytes))),
        )

    @property
    def persona_path(self) -> Path:
        return self.data_dir / "persona.md"

    @property
    def users_path(self) -> Path:
        return self.data_dir / "users.yaml"

    @property
    def trios_dir(self) -> Path:
        return self.data_dir / "golden_trios"

    @property
    def schema_path(self) -> Path:
        return self.data_dir / "schema.json"
