"""Per-turn traces and aggregate agent metrics.

`write_trace` (the last graph node) saves one `TurnTrace` per turn: timings, LLM calls, SQL attempts, guard events,
bytes scanned. `TraceStore` writes it to SQLite (`traces` plus a per-call `llm_ledger`) and optionally appends a JSON line
to a file. `ria trace` and `ria stats` read it back; the evals read traces to check behaviour. Depends on `ria.db` tables only.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


SUCCESS_OUTCOMES = ("ok", "saved", "deleted", "cancelled", "listed", "shown", "rated", "help")


@dataclass
class TurnTrace:
    """Everything recorded about one turn. `node_timings`, `llm_calls` (ledger dicts), `sql_attempts` and
    `guard_events` are copied from graph state. Token totals, `fallback_used` and `repairs` are derived, not stored
    fields: `repairs` is the number of SQL attempts after the first, so guard rejections count too. The trace holds
    SQL and metadata but not the user's question or the answer text."""

    turn_id: str
    thread_id: str
    user_id: str
    intent: str | None
    outcome: str
    started_at: str
    duration_ms: int
    node_timings: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    sql_attempts: list[dict] = field(default_factory=list)
    guard_events: list[str] = field(default_factory=list)
    bq_bytes: int = 0
    error: str | None = None
    rating: int | None = None  # +1 / -1 once the user rated the answer
    rating_note: str = ""
    corrected: bool = False  # the user corrected this turn's answer on a later turn

    @property
    def tokens_in(self) -> int:
        return sum(int(c.get("tokens_in", 0)) for c in self.llm_calls)

    @property
    def tokens_out(self) -> int:
        return sum(int(c.get("tokens_out", 0)) for c in self.llm_calls)

    @property
    def fallback_used(self) -> bool:
        return any(c.get("fallback") for c in self.llm_calls)

    @property
    def repairs(self) -> int:
        return max(len(self.sql_attempts) - 1, 0)

    def to_dict(self) -> dict:
        """Stored fields plus the derived totals, for the JSONL log."""
        d = asdict(self)
        d.update(tokens_in=self.tokens_in, tokens_out=self.tokens_out, fallback_used=self.fallback_used, repairs=self.repairs)
        return d


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank style percentile (`pct` between 0 and 1) of a list of ints; 0 for an empty list."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct * (len(ordered) - 1))))
    return int(ordered[idx])


class TraceStore:
    """Persist and query `TurnTrace`s. `jsonl_path`, when set (live runs), also gets one JSON line per saved turn."""

    def __init__(self, conn: sqlite3.Connection, jsonl_path: Path | None = None):
        self.conn = conn
        self.jsonl_path = jsonl_path

    def save(self, t: TurnTrace) -> None:
        """Upsert the turn row and replace its `llm_ledger` rows, so saving the same turn id twice does not duplicate
        ledger entries. The JSONL line is appended on every save."""
        payload = json.dumps(
            {"node_timings": t.node_timings, "llm_calls": t.llm_calls, "sql_attempts": t.sql_attempts,
             "guard_events": t.guard_events, "error": t.error}
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO traces (turn_id, thread_id, user_id, intent, outcome, started_at, duration_ms, "
            "tokens_in, tokens_out, bq_bytes, fallback_used, repairs, payload, rating, rating_note, corrected) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.turn_id, t.thread_id, t.user_id, t.intent, t.outcome, t.started_at, t.duration_ms, t.tokens_in,
             t.tokens_out, t.bq_bytes, int(t.fallback_used), t.repairs, payload, t.rating, t.rating_note,
             int(t.corrected)),
        )
        self.conn.execute("DELETE FROM llm_ledger WHERE turn_id = ?", (t.turn_id,))
        for c in t.llm_calls:
            self.conn.execute(
                "INSERT INTO llm_ledger (turn_id, node, model, tokens_in, tokens_out, latency_ms, fallback) VALUES (?,?,?,?,?,?,?)",
                (t.turn_id, c.get("node", ""), c.get("model", ""), int(c.get("tokens_in", 0)),
                 int(c.get("tokens_out", 0)), int(c.get("latency_ms", 0)), int(bool(c.get("fallback")))),
            )
        self.conn.commit()
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(t.to_dict()) + "\n")

    def _from_row(self, r: sqlite3.Row) -> TurnTrace:
        p = json.loads(r["payload"])
        return TurnTrace(
            turn_id=r["turn_id"], thread_id=r["thread_id"], user_id=r["user_id"], intent=r["intent"],
            outcome=r["outcome"], started_at=r["started_at"], duration_ms=r["duration_ms"],
            node_timings=p.get("node_timings", []), llm_calls=p.get("llm_calls", []),
            sql_attempts=p.get("sql_attempts", []), guard_events=p.get("guard_events", []),
            bq_bytes=r["bq_bytes"], error=p.get("error"),
            rating=r["rating"], rating_note=r["rating_note"], corrected=bool(r["corrected"]),
        )

    def set_rating(self, turn_id: str, rating: int, note: str = "") -> bool:
        """Record a +1 / -1 rating (and optional note) on a saved turn. False when no such turn exists."""
        cur = self.conn.execute("UPDATE traces SET rating = ?, rating_note = ? WHERE turn_id = ?", (rating, note, turn_id))
        self.conn.commit()
        return cur.rowcount > 0

    def set_corrected(self, turn_id: str) -> bool:
        """Mark a saved turn as corrected by the user. False when no such turn exists."""
        cur = self.conn.execute("UPDATE traces SET corrected = 1 WHERE turn_id = ?", (turn_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def get(self, turn_id: str) -> TurnTrace | None:
        """Load one trace by turn id, or None."""
        r = self.conn.execute("SELECT * FROM traces WHERE turn_id = ?", (turn_id,)).fetchone()
        return self._from_row(r) if r else None

    def stats(self, since: str | None = None) -> dict:
        """Aggregate metrics over all turns, or those started at or after the ISO timestamp `since`.

        Returns `{"turns": 0}` when empty. Otherwise rates over all turns: success (`SUCCESS_OUTCOMES`), refusal, error (error, gave_up) and fallback-model use; `no_data` counts in none of the first
        three. `outage_turns` counts turns that failed with `LLMUnavailable` or whose last SQL attempt was refused by
        the open circuit breaker. Also mean repairs (analysis turns only), guard rejections by check code, p50 and p95 latency,
        mean tokens and BigQuery bytes per turn, counts of good and bad ratings, the correction rate (corrected
        analysis turns / analysis turns), and turn counts by intent.
        """
        if since:
            rows = self.conn.execute("SELECT * FROM traces WHERE started_at >= ?", (since,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM traces").fetchall()
        traces = [self._from_row(r) for r in rows]
        n = len(traces)
        if n == 0:
            return {"turns": 0}
        rejections: Counter = Counter()
        for t in traces:
            for e in t.guard_events:
                code, _, status = e.partition(":")
                if status == "rejected":
                    rejections[code] += 1
        durations = [t.duration_ms for t in traces]
        analysis = [t for t in traces if t.intent == "analysis"]
        return {
            "turns": n,
            "success_rate": sum(t.outcome in SUCCESS_OUTCOMES for t in traces) / n,
            "refusal_rate": sum(t.outcome == "refused" for t in traces) / n,
            "error_rate": sum(t.outcome in ("error", "gave_up") for t in traces) / n,
            "fallback_rate": sum(t.fallback_used for t in traces) / n,
            "outage_turns": sum(
                (t.error or "").startswith("LLMUnavailable")
                or bool(t.sql_attempts and t.sql_attempts[-1].get("kind") == "circuit_open")
                for t in traces
            ),
            "mean_repairs": (sum(t.repairs for t in analysis) / len(analysis)) if analysis else 0.0,
            "guard_rejections": dict(rejections),
            "p50_ms": _percentile(durations, 0.5),
            "p95_ms": _percentile(durations, 0.95),
            "mean_tokens": sum(t.tokens_in + t.tokens_out for t in traces) / n,
            "mean_bq_bytes": sum(t.bq_bytes for t in traces) / n,
            "rated_good": sum(t.rating == 1 for t in traces),
            "rated_bad": sum(t.rating == -1 for t in traces),
            "correction_rate": (sum(t.corrected for t in analysis) / len(analysis)) if analysis else 0.0,
            "by_intent": dict(Counter(t.intent or "unknown" for t in traces)),
        }
