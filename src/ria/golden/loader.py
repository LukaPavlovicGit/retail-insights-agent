"""The `Trio` record and the loader for seed Golden Trios stored as YAML files.

A Golden Trio is an analyst-approved example: a question, the SQL that answers it, and the report an analyst would
write. Seed trios live one per file under the data directory's golden_trios folder and are loaded by `ria init`.
`Trio` is shared by the store, retrieval, and the report-saving node.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

REQUIRED = ("question", "sql", "report")


@dataclass
class Trio:
    """One Golden Trio. `id` is None until stored. `tags` are the product categories it suits, or "all"; retrieval
    matches them against the user's scope. `status` is promoted, pending or rejected. `source` is "seed" (from
    YAML) or "user" (saved from chat). `author` and `created_at` are informational."""

    id: int | None
    question: str
    sql: str
    report: str
    tags: list[str] = field(default_factory=lambda: ["all"])
    status: str = "promoted"
    source: str = "seed"
    author: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Plain dict for graph state (which must stay serialisable for the checkpointer)."""
        return asdict(self)


def load_trios_dir(path: Path) -> list[Trio]:
    """Read every `*.yaml` file in `path` (sorted by name) into promoted seed `Trio`s.

    Each file needs non-empty `question`, `sql` and `report`; a missing one raises `ValueError` naming the file, so a
    bad seed fails `ria init` loudly. `tags` defaults to ["all"].
    """
    trios: list[Trio] = []
    for file in sorted(Path(path).glob("*.yaml")):
        data = yaml.safe_load(file.read_text()) or {}
        for key in REQUIRED:
            if not data.get(key):
                raise ValueError(f"{file.name}: missing required key '{key}'")
        trios.append(
            Trio(
                id=None,
                question=str(data["question"]).strip(),
                sql=str(data["sql"]).strip(),
                report=str(data["report"]).strip(),
                tags=[str(t) for t in (data.get("tags") or ["all"])],
                status="promoted",
                source="seed",
                author=str(data.get("author", "")),
                created_at=str(data.get("created_at") or date.today().isoformat()),
            )
        )
    return trios
