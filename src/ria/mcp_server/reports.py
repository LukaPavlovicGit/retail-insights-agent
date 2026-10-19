from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Report:
    id: int
    user_id: str
    thread_id: str
    title: str
    question: str
    sql: str
    body: str
    tags: list[str]
    created_at: str


def _row(r: sqlite3.Row) -> Report:
    return Report(r["id"], r["user_id"], r["thread_id"], r["title"], r["question"], r["sql"], r["body"],
                  json.loads(r["tags"]), r["created_at"])


class ReportStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, user_id: str, thread_id: str, title: str, question: str, sql: str, body: str, tags: list[str]) -> int:
        cur = self.conn.execute(
            "INSERT INTO reports (user_id, thread_id, title, question, sql, body, tags, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, thread_id, title, question, sql, body, json.dumps(list(tags)),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, user_id: str, report_id: int) -> Report | None:
        r = self.conn.execute("SELECT * FROM reports WHERE user_id = ? AND id = ?", (user_id, report_id)).fetchone()
        return _row(r) if r else None

    def search(self, user_id: str, text: str | None = None, thread_id: str | None = None) -> list[Report]:
        sql = "SELECT * FROM reports WHERE user_id = ?"
        params: list = [user_id]
        if text:
            like = f"%{text.lower()}%"
            sql += " AND (LOWER(title) LIKE ? OR LOWER(body) LIKE ? OR LOWER(question) LIKE ?)"
            params += [like, like, like]
        if thread_id:
            sql += " AND thread_id = ?"
            params.append(thread_id)
        sql += " ORDER BY id"
        return [_row(r) for r in self.conn.execute(sql, params)]

    def delete(self, user_id: str, ids: list[int]) -> int:
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        cur = self.conn.execute(f"DELETE FROM reports WHERE user_id = ? AND id IN ({marks})", [user_id, *ids])
        self.conn.commit()
        return int(cur.rowcount)
