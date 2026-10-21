"""Runtime settings for the agent, read once from the environment (and an optional .env file).

`Settings` is the single place for model names, BigQuery limits, retry and call budgets, and data file locations.
It is passed into `graph.deps.Deps`, the CLI, the MCP server launcher and the evals; nothing else reads env vars
(the MCP server child process is the one exception, and it gets its values from `tools.open_mcp_toolbox`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable configuration with defaults for local development.

    Limits: `bq_max_bytes` caps bytes scanned per query, `bq_timeout_s` the BigQuery wait, `row_cap` the rows
    returned and the LIMIT the guard enforces, `max_repairs` the SQL repair rounds per turn,
    `max_llm_calls_per_turn` the LLM budget per turn, `window_turns` how many recent turns go to the model
    before older ones are summarised, `scope_column` the `products` column that user scopes filter on. `embedding_dim` must match the vector table created by `db.init_schema`.
    `persona_override` points `persona_path` at another file (used to try a candidate persona in the persona gate).
    Business defaults (churn window, revenue statuses, minimum group size) are not settings; they live in `data/org_defaults.yaml` (see `OrgDefaults`).
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
    persona_override: Path | None = None

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        """Build settings from env vars, loading `env_file` first without overriding variables already set.

        Only the keys, model names, db path, data dir, byte cap and scope column are overridable from the environment;
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
        )

    @property
    def persona_path(self) -> Path:
        return self.persona_override or self.data_dir / "persona.md"

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
    def org_defaults_path(self) -> Path:
        return self.data_dir / "org_defaults.yaml"


_STATUSES = ("Complete", "Shipped", "Processing", "Cancelled", "Returned")


def _int_in(key: str, value: object, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{key} must be an integer from {lo} to {hi}, got {value!r}")
    return value


@dataclass(frozen=True)
class OrgDefaults:
    """Org-wide business defaults from `data/org_defaults.yaml`: churn window, statuses that count as revenue, and
    the smallest customer group the guard lets a query show."""

    churn_window_days: int = 90
    revenue_statuses: tuple[str, ...] = ("Complete", "Shipped", "Processing")
    min_group_size: int = 10

    @classmethod
    def load(cls, path: Path) -> "OrgDefaults":
        """Read `path`; a missing or empty file gives the defaults, an invalid value raises `ValueError` naming the key."""
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{p.name} must be a mapping of keys to values")
        unknown = set(raw) - {"churn_window_days", "revenue_statuses", "min_group_size"}
        if unknown:
            raise ValueError(f"unknown key in {p.name}: {', '.join(sorted(map(str, unknown)))}")
        kwargs: dict = {}
        if "churn_window_days" in raw:
            kwargs["churn_window_days"] = _int_in("churn_window_days", raw["churn_window_days"], 1, 3650)
        if "min_group_size" in raw:
            kwargs["min_group_size"] = _int_in("min_group_size", raw["min_group_size"], 1, 1000)
        if "revenue_statuses" in raw:
            s = raw["revenue_statuses"]
            if not isinstance(s, list) or not s or any(x not in _STATUSES for x in s):
                raise ValueError(f"revenue_statuses must be a non-empty list of {', '.join(_STATUSES)}, got {s!r}")
            kwargs["revenue_statuses"] = tuple(s)
        return cls(**kwargs)


def join_statuses(statuses: tuple[str, ...]) -> str:
    """`("A", "B", "C")` -> "A, B or C"."""
    return statuses[0] if len(statuses) == 1 else ", ".join(statuses[:-1]) + " or " + statuses[-1]


def render_definitions(d: OrgDefaults) -> str:
    """The business definitions text shown to the model, with the org defaults substituted."""
    return f"""# Business definitions (org-wide defaults; read on every answer)

- Churn: a customer who made no purchase in the last {d.churn_window_days} days. Churn rate for a period is churned customers over customers seen.
- Revenue: sum of order_items.sale_price for items with status {join_statuses(d.revenue_statuses)}. Other statuses are excluded.
- Active customer: at least one purchase in the period.
- New customer: first purchase falls in the period.

If the user specifies a different window or statuses in their question, use theirs for that answer and say so."""
