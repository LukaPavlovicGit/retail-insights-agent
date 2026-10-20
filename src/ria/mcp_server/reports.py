"""The saved-reports library: per-user storage of analyses the user chose to keep.

`ReportStore` is used only by the MCP server's `reports_*` tools. Every query is filtered by `user_id`, so one user cannot
read or delete another's reports even if the model passes the wrong ids. Depends on the `reports` table from `ria.db`.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Report:
    """A saved report: the user's `question`, the `sql` that answered it, the PII-masked analysis `body`, a generated `title` and `tags`, and the `thread_id` it came from."""

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
    """CRUD over the `reports` table on an open connection. Reports are scoped to `user_id` in every method."""

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
        """One of this user's reports by id, or None (also None if the id belongs to someone else)."""
        r = self.conn.execute("SELECT * FROM reports WHERE user_id = ? AND id = ?", (user_id, report_id)).fetchone()
        return _row(r) if r else None

    def search(self, user_id: str, text: str | None = None, thread_id: str | None = None) -> list[Report]:
        """The user's reports in id order. `text` keeps those whose title, body or question contains it (case-insensitive
        substring); `thread_id` keeps those saved in that conversation. Both filters combine with AND; neither means all."""
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
        """Delete the listed reports that belong to `user_id` and return how many rows were removed (ids owned by others are ignored)."""
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        cur = self.conn.execute(f"DELETE FROM reports WHERE user_id = ? AND id IN ({marks})", [user_id, *ids])
        self.conn.commit()
        return int(cur.rowcount)
