from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from sqlite_vec import serialize_float32

from ria.golden.loader import Trio


def _row(r: sqlite3.Row) -> Trio:
    return Trio(
        id=r["id"], question=r["question"], sql=r["sql"], report=r["report"], tags=json.loads(r["tags"]),
        status=r["status"], source=r["source"], author=r["author"], created_at=r["created_at"],
    )


class TrioStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, t: Trio) -> int:
        cur = self.conn.execute(
            "INSERT INTO trios (question, sql, report, tags, status, source, author, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (t.question, t.sql, t.report, json.dumps(t.tags), t.status, t.source, t.author,
             t.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def replace_seed(self, trios: list[Trio]) -> list[int]:
        seed_ids = [r["id"] for r in self.conn.execute("SELECT id FROM trios WHERE source = 'seed'")]
        if seed_ids:
            marks = ",".join("?" * len(seed_ids))
            self.conn.execute(f"DELETE FROM trio_vec WHERE trio_id IN ({marks})", seed_ids)
            self.conn.execute(f"DELETE FROM trios WHERE id IN ({marks})", seed_ids)
        return [self.insert(t) for t in trios]

    def get(self, trio_id: int) -> Trio | None:
        r = self.conn.execute("SELECT * FROM trios WHERE id = ?", (trio_id,)).fetchone()
        return _row(r) if r else None

    def list(self, status: str | None = None) -> list[Trio]:
        if status:
            rows = self.conn.execute("SELECT * FROM trios WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self.conn.execute("SELECT * FROM trios ORDER BY id")
        return [_row(r) for r in rows]

    def set_status(self, trio_id: int, status: str) -> None:
        self.conn.execute("UPDATE trios SET status = ? WHERE id = ?", (status, trio_id))
        self.conn.commit()

    def set_embedding(self, trio_id: int, vec: list[float]) -> None:
        self.conn.execute("DELETE FROM trio_vec WHERE trio_id = ?", (trio_id,))
        self.conn.execute("INSERT INTO trio_vec (trio_id, embedding) VALUES (?, ?)", (trio_id, serialize_float32(vec)))
        self.conn.commit()

    def missing_embeddings(self) -> list[Trio]:
        rows = self.conn.execute(
            "SELECT t.* FROM trios t LEFT JOIN trio_vec v ON v.trio_id = t.id WHERE v.trio_id IS NULL ORDER BY t.id"
        )
        return [_row(r) for r in rows]

    def search(
        self, vec: list[float], categories: list[str] | None, k: int = 3, max_distance: float = 0.5
    ) -> list[tuple[Trio, float]]:
        rows = self.conn.execute(
            "SELECT trio_id, distance FROM trio_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialize_float32(vec), max(k * 4, 8)),
        ).fetchall()
        out: list[tuple[Trio, float]] = []
        wanted = set(categories or [])
        for r in rows:
            if r["distance"] > max_distance:
                continue
            t = self.get(r["trio_id"])
            if t is None or t.status != "promoted":
                continue
            if categories is not None and "all" not in t.tags and not (wanted & set(t.tags)):
                continue
            out.append((t, float(r["distance"])))
            if len(out) == k:
                break
        return out
