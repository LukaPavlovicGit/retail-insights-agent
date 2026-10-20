"""Per-user answer preferences (format, depth, charts) learned from what the user says, and their storage.

`extract_preferences` reads phrases like "use tables" or "shorter" with plain regexes (no LLM); `PrefsStore` applies the
changes and persists them in `user_prefs`. The graph reads preferences to shape prompts and rendering, and writes them
from `ack_preference` and `update_memory`. Preferences are per user, not per thread, so they carry over between chats.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone

DEPTHS = ["brief", "standard", "deep"]
FORMATS = {"table", "bullets", "prose"}


@dataclass
class Prefs:
    """One user's settings. `format` is table, bullets or prose; `depth` is brief, standard or deep; `charts` is
    stored and shown in the summary but nothing renders charts yet. `brevity_streak` counts consecutive
    "shorter" requests; at three, depth drops one step. `notes` is stored but not currently written anywhere."""

    format: str = "prose"
    depth: str = "standard"
    charts: bool = False
    notes: str = ""
    brevity_streak: int = 0

    def summary(self) -> str:
        """One-line description used in prompts and in the acknowledgement shown to the user."""
        return f"Format: {self.format}. Depth: {self.depth}. Charts: {'yes' if self.charts else 'no'}."


_FORMAT_PATTERNS = [
    (re.compile(r"\b(as|in|use|prefer|show|with|want)\b[^.?!]{0,20}\btables?\b", re.I), "table"),
    (re.compile(r"\bbullet(s| points?| list)?\b", re.I), "bullets"),
    (re.compile(r"\b(prose|paragraphs?|narrative)\b", re.I), "prose"),
]
_BREVITY = re.compile(r"\b(shorter|brief(er|ly)?|concise|too long|less detail|summar(y|ize) only)\b", re.I)
_DEPTH_UP = re.compile(r"\b(deeper|more detail(s|ed)?|in depth|elaborate|expand on)\b", re.I)
_CHARTS_NO = re.compile(r"\b(no|without|skip|don'?t (want|need|include))\b[^.?!]{0,15}\b(charts?|graphs?)\b", re.I)
_CHARTS_YES = re.compile(
    r"\b(include|add|show|want|with|prefer|use|give|make|like)\b[^.?!]{0,20}\b(charts?|graphs?|plots?|visuals?)\b(?!\s+of\b)",
    re.I,
)


def extract_preferences(text: str) -> dict:
    """Pull preference changes out of a message. Returns a dict with any of `format` (table, bullets, prose; first
    match wins), `brevity` (True), `depth_up` (True), `charts` (True or False; a "no charts" phrase beats a "charts" one).
    Empty dict when nothing matches. Patterns are deliberately narrow (for example "chart of ..." is a data
    request, not a preference), and a miss simply leaves preferences unchanged."""
    out: dict = {}
    for pattern, fmt in _FORMAT_PATTERNS:
        if pattern.search(text):
            out["format"] = fmt
            break
    if _BREVITY.search(text):
        out["brevity"] = True
    if _DEPTH_UP.search(text):
        out["depth_up"] = True
    if _CHARTS_NO.search(text):
        out["charts"] = False
    elif _CHARTS_YES.search(text):
        out["charts"] = True
    return out


class PrefsStore:
    """SQLite-backed `Prefs` per user id. Created by `Deps`."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, user_id: str) -> Prefs:
        """Stored preferences, or the defaults for a user with no row."""
        r = self.conn.execute("SELECT * FROM user_prefs WHERE user_id = ?", (user_id,)).fetchone()
        if not r:
            return Prefs()
        return Prefs(r["format"], r["depth"], bool(r["charts"]), r["notes"], r["brevity_streak"])

    def save(self, user_id: str, p: Prefs) -> None:
        """Upsert the full preferences row and stamp `updated_at`."""
        self.conn.execute(
            "INSERT OR REPLACE INTO user_prefs (user_id, format, depth, charts, notes, brevity_streak, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, p.format, p.depth, int(p.charts), p.notes, p.brevity_streak,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def apply(self, user_id: str, changes: dict) -> tuple[Prefs, bool]:
        """Apply the dict from `extract_preferences`; returns `(new_prefs, changed)` and saves only if something changed.

        `depth_up` raises depth one step and resets the brevity streak. `brevity` adds to the streak, and the
        third consecutive one lowers depth one step and resets it. Note that `changed` is True whenever the streak
        moves, even if the visible settings stay the same.
        """
        old = self.get(user_id)
        new = replace(old)
        if changes.get("format") in FORMATS:
            new.format = changes["format"]
        if "charts" in changes:
            new.charts = bool(changes["charts"])
        if changes.get("depth_up"):
            new.depth = DEPTHS[min(DEPTHS.index(new.depth) + 1, len(DEPTHS) - 1)]
            new.brevity_streak = 0
        elif changes.get("brevity"):
            new.brevity_streak += 1
            if new.brevity_streak >= 3:
                new.depth = DEPTHS[max(DEPTHS.index(new.depth) - 1, 0)]
                new.brevity_streak = 0
        changed = new != old
        if changed:
            self.save(user_id, new)
        return new, changed
