"""Run one conversational turn through the compiled graph and summarise the result.

`run_turn` is the single entry point used by the CLI chat loop and the evals. It starts a fresh turn or
resumes a paused one, and reports either the rendered answer or a pending confirmation. It depends on
LangGraph's `Command` for resuming and on the state keys defined in `graph.state`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command


@dataclass
class TurnResult:
    """What a caller needs after a turn: the `turn_id` (key into the trace store), the final `rendered` text,
    the `interrupt` payload if the graph is waiting for confirmation, and the `outcome` label.

    `outcome` comes from the graph (for example ok, refused, no_data, gave_up, error, saved, deleted, cancelled)
    or is `needs_confirmation` when paused; in that case `rendered` is None and `interrupt` is set.
    """

    turn_id: str
    rendered: str | None
    interrupt: dict | None
    outcome: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_turn_id() -> str:
    return uuid4().hex[:12]


async def run_turn(app, user: dict, thread_id: str, text: str, *, resume: dict | None = None, now=None, new_turn_id=None) -> TurnResult:
    """Invoke the graph for `thread_id`.

    Without `resume`, sends the user's text as a new turn (new turn id and start time; `now` and `new_turn_id`
    are injectable for tests). With `resume`, continues the paused thread via `Command(resume=...)` and ignores
    `text`. The thread id is the checkpointer key, so the same id carries conversation history across turns.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        result = await app.ainvoke(Command(resume=resume), cfg)
    else:
        payload = {
            "user": user, "thread_id": thread_id, "messages": [HumanMessage(content=text)], "question": text,
            "turn_id": (new_turn_id or _new_turn_id)(), "started_at": (now or _now)(),
        }
        result = await app.ainvoke(payload, cfg)
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        return TurnResult(result.get("turn_id", ""), None, interrupts[0].value, "needs_confirmation")
    return TurnResult(result.get("turn_id", ""), result.get("rendered"), None, result.get("outcome") or "ok")
