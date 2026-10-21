"""Deterministic pre-router: recognises slash commands before the LLM router so they work with both models down.

Runs between `begin_turn` and `router`. For a command it sets `intent` (and `command`, plus the report filter for
`/delete`) and `nodes_common.route_intent` sends the turn straight to that command's node; otherwise it writes nothing
and the turn continues to `router`. Depends on `graph.commands`.
"""
from __future__ import annotations

from dataclasses import asdict

from ria.graph.commands import parse_command
from ria.graph.deps import Deps

COMMAND_INTENTS = {
    "reports": "report_list", "report": "report_show", "delete": "report_delete",
    "schema": "schema_overview", "rating": "rating", "help": "help",
}


def make_pre_router(deps: Deps):
    """Node `pre_router`: no LLM call, no I/O. Returns `{}` for plain text, else the command's `intent` and `command` dict."""
    async def pre_router(state: dict) -> dict:
        cmd = parse_command(state.get("question") or "")
        if cmd is None:
            return {}
        out: dict = {"intent": COMMAND_INTENTS[cmd.kind], "command": asdict(cmd)}
        if cmd.kind == "delete":
            out.update(report_filter_scope=cmd.filter_scope, report_filter_text=cmd.filter_text)
        return out

    return pre_router
