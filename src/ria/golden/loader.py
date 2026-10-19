from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

REQUIRED = ("question", "sql", "report")


@dataclass
class Trio:
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
        return asdict(self)


def load_trios_dir(path: Path) -> list[Trio]:
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
