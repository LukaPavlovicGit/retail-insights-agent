from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command


@dataclass
class TurnResult:
    turn_id: str
    rendered: str | None
    interrupt: dict | None
    outcome: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_turn_id() -> str:
    return uuid4().hex[:12]


async def run_turn(app, user: dict, thread_id: str, text: str, *, resume: dict | None = None, now=None, new_turn_id=None) -> TurnResult:
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
