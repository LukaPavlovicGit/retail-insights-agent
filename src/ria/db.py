from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec
import yaml

DDL = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, categories TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS user_prefs (
  user_id TEXT PRIMARY KEY, format TEXT NOT NULL DEFAULT 'prose', depth TEXT NOT NULL DEFAULT 'standard',
  charts INTEGER NOT NULL DEFAULT 0, notes TEXT NOT NULL DEFAULT '', brevity_streak INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, thread_id TEXT NOT NULL, title TEXT NOT NULL,
  question TEXT NOT NULL, sql TEXT NOT NULL, body TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trios (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, sql TEXT NOT NULL, report TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '["all"]', status TEXT NOT NULL DEFAULT 'promoted', source TEXT NOT NULL DEFAULT 'seed',
  author TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (
  turn_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_id TEXT NOT NULL, intent TEXT, outcome TEXT NOT NULL,
  started_at TEXT NOT NULL, duration_ms INTEGER NOT NULL, tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0, bq_bytes INTEGER NOT NULL DEFAULT 0, fallback_used INTEGER NOT NULL DEFAULT 0,
  repairs INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL, node TEXT NOT NULL, model TEXT NOT NULL,
  tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL, latency_ms INTEGER NOT NULL,
  fallback INTEGER NOT NULL DEFAULT 0);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    conn.executescript(DDL)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS trio_vec USING vec0("
        f"trio_id INTEGER PRIMARY KEY, embedding float[{embedding_dim}] distance_metric=cosine)"
    )
    conn.commit()


def _row_to_user(row: sqlite3.Row) -> dict:
    cats = json.loads(row["categories"])
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row["role"],
        "categories": None if cats == ["*"] else cats,
    }


def load_users(conn: sqlite3.Connection, path: Path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    for u in raw:
        for key in ("id", "name", "role", "categories"):
            if key not in u:
                raise ValueError(f"{path}: user entry missing '{key}'")
        conn.execute(
            "INSERT OR REPLACE INTO users (id, name, role, categories) VALUES (?, ?, ?, ?)",
            (u["id"], u["name"], u["role"], json.dumps(list(u["categories"]))),
        )
    conn.commit()
    return [get_user(conn, u["id"]) for u in raw]


def get_user(conn: sqlite3.Connection, user_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None
