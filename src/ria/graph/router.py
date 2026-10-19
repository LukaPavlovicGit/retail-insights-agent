from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ria.graph.deps import Deps
from ria.graph.nodes_common import base_system, call_llm, window
from ria.graph.prompts import REFUSAL_TEXT, ROUTER_SYSTEM
from ria.memory.prefs import extract_preferences

class RouteDecision(BaseModel):
    intent: Literal["analysis", "schema_question", "report_save", "report_delete", "preference_update", "off_topic"]
    reason: str = ""
    report_filter_scope: Literal["conversation", "text", "all"] | None = None
    report_filter_text: str | None = None


def make_router(deps: Deps):
    async def router(state: dict) -> dict:
        question = (state.get("question") or "").strip()
        if not question:
            return {"intent": "off_topic", "router_reason": "empty message"}
        ctx = base_system(deps, state)
        system = ROUTER_SYSTEM.format(**{k: v for k, v in ctx.items() if k != "prefs_obj"})
        res, calls = await call_llm(deps, state, node="router", system=system, messages=window(state, deps), schema=RouteDecision)
        d: RouteDecision = res.parsed
        return {
            "intent": d.intent, "router_reason": d.reason, "llm_calls": calls,
            "report_filter_scope": d.report_filter_scope, "report_filter_text": d.report_filter_text,
        }

    return router


def make_refuse(deps: Deps):
    async def refuse(state: dict) -> dict:
        return {"rendered": REFUSAL_TEXT, "outcome": "refused"}

    return refuse


def make_ack_preference(deps: Deps):
    async def ack_preference(state: dict) -> dict:
        changes = extract_preferences(state.get("question", ""))
        prefs, changed = deps.prefs_store.apply(state["user"]["id"], changes)
        if changed:
            text = f"Noted. From now on: {prefs.summary()}"
        elif changes.get("brevity"):
            text = f"Understood, I'll keep it tight. Current settings: {prefs.summary()}"
        else:
            text = f"I didn't catch a specific preference. Current settings: {prefs.summary()} You can say \"use tables\", \"bullet points\", \"shorter\", or \"more detail\"."
        return {"rendered": text, "outcome": "ok"}

    return ack_preference
