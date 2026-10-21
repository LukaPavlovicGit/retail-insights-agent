"""Runtime settings for the agent, read once from the environment (and an optional .env file).

`Settings` is the single place for model names, BigQuery limits, retry and call budgets, and data file locations.
It is passed into `graph.deps.Deps`, the CLI, the MCP server launcher and the evals; nothing else reads env vars
(the MCP server child process is the one exception, and it gets its values from `tools.open_mcp_toolbox`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable configuration with defaults for local development.

    Limits: `bq_max_bytes` caps bytes scanned per query, `bq_timeout_s` the BigQuery wait, `row_cap` the rows
    returned and the LIMIT the guard enforces, `max_repairs` the SQL repair rounds per turn,
    `max_llm_calls_per_turn` the LLM budget per turn, `window_turns` how many recent turns go to the model
    before older ones are summarised, `scope_column` the `products` column that user scopes filter on, `min_group_size`
    the smallest customer group the guard lets a query show. `embedding_dim` must match the vector table created by `db.init_schema`.
    """

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
    scope_column: str = "brand"
    min_group_size: int = 10

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        """Build settings from env vars, loading `env_file` first without overriding variables already set.

        Only the keys, model names, db path, data dir, byte cap, scope column and minimum group size are overridable from the environment;
        the other limits keep their class defaults.
        """
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
            scope_column=g("RIA_SCOPE_COLUMN", cls.scope_column),
            min_group_size=int(g("RIA_MIN_GROUP_SIZE", str(cls.min_group_size))),
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

    @property
    def definitions_path(self) -> Path:
        return self.data_dir / "definitions.md"
