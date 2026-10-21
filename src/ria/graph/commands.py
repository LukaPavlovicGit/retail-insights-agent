"""Slash-command parsing for the pre-router: fixed commands the user can type, recognised in code with no model call.

`parse_command` returns a `ParsedCommand` for `/reports`, `/report <id>`, `/delete <filter>`, `/schema`, `/good [note]`,
`/bad [note]` and `/help`, and None for anything else (so the text goes on to the LLM router). No dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HELP_TEXT = """Commands (they work even when the language model is down):
- /reports: list your saved reports
- /report <id>: show one saved report
- /delete <filter>: delete reports ("from this conversation", "all", or "mentioning <subject>"); you confirm first
- /schema: show the tables and columns I can query
- /good [note], /bad [note]: rate my last answer
- /help: show this list
Anything else is treated as a question."""


@dataclass
class ParsedCommand:
    """A recognised slash command. `kind` is reports, report, delete, schema, rating or help. `arg` is the report id (as text)
    for `report`; `rating` (+1 or -1) and `note` are for `rating`; `filter_scope` ("conversation", "all", "text" or None
    when no filter was given) and `filter_text` are for `delete`."""

    kind: str
    arg: str = ""
    rating: int = 0
    note: str = ""
    filter_scope: str | None = None
    filter_text: str | None = None


_SIMPLE = {"/reports": "reports", "/schema": "schema", "/help": "help"}
_REPORT = re.compile(r"/report\s+(\d+)", re.I)
_DELETE = re.compile(r"/delete(?:\s+(.*))?", re.I | re.S)
_RATING = re.compile(r"/(good|bad)(?:\s+(.*))?", re.I | re.S)
_CONVERSATION = re.compile(r"(from\s+)?(this\s+)?conversation", re.I)
_PREFIX = re.compile(r"^(mentioning|about)\s+", re.I)


def parse_command(text: str) -> ParsedCommand | None:
    """Parse `text` (case-insensitive, surrounding whitespace ignored) into a `ParsedCommand`, or None if it is not a known command."""
    t = (text or "").strip()
    if t.lower() in _SIMPLE:
        return ParsedCommand(kind=_SIMPLE[t.lower()])
    if m := _REPORT.fullmatch(t):
        return ParsedCommand(kind="report", arg=m.group(1))
    if m := _RATING.fullmatch(t):
        return ParsedCommand(kind="rating", rating=1 if m.group(1).lower() == "good" else -1, note=(m.group(2) or "").strip())
    if m := _DELETE.fullmatch(t):
        rest = (m.group(1) or "").strip()
        if not rest:
            return ParsedCommand(kind="delete")
        if _CONVERSATION.fullmatch(rest):
            return ParsedCommand(kind="delete", filter_scope="conversation")
        if rest.lower() == "all":
            return ParsedCommand(kind="delete", filter_scope="all")
        return ParsedCommand(kind="delete", filter_scope="text", filter_text=_PREFIX.sub("", rest).strip())
    return None
