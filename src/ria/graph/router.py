"""Intent routing and the two cheap terminal nodes it can lead to (refusal and preference acknowledgement).

The router classifies each message into one intent with a structured LLM call; `nodes_common.route_intent` then maps
the intent to the next node (see `build.INTENT_ROUTES`). Refusals and preference acknowledgements are canned or
rule-based and make no further LLM calls. Depends on `graph.nodes_common`, `graph.prompts` and `memory.prefs`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ria.graph.commands import HELP_TEXT
from ria.graph.deps import Deps
from ria.graph.nodes_common import base_system, call_llm, window
from ria.graph.prompts import REFUSAL_TEXT, ROUTER_SYSTEM
from ria.memory.prefs import extract_preferences

class RouteDecision(BaseModel):
    """Structured router output: one `intent` plus a short `reason`. For `report_delete` the model also fills
    `report_filter_scope` ("conversation", "text" or "all") and `report_filter_text`, which `resolve_targets` consumes."""

    intent: Literal["analysis", "schema_question", "report_save", "report_delete", "preference_update", "off_topic"]
    reason: str = ""
    report_filter_scope: Literal["conversation", "text", "all"] | None = None
    report_filter_text: str | None = None
    is_correction: bool = False


def make_router(deps: Deps):
    """Node `router`: reads the question, recent messages, summary, user scope and preferences; writes `intent`,
    `router_reason` and the report filter fields. One LLM call (`ROUTER_SYSTEM`, structured `RouteDecision`); an empty
    message is classified off_topic without calling the model. The next node comes from `route_intent`."""
    async def router(state: dict) -> dict:
        question = (state.get("question") or "").strip()
        if not question:
            return {"intent": "off_topic", "router_reason": "empty message"}
        ctx = base_system(deps, state)
        system = ROUTER_SYSTEM.format(**{k: v for k, v in ctx.items() if k != "prefs_obj"})
        res, calls = await call_llm(deps, state, node="router", system=system, messages=window(state, deps), schema=RouteDecision)
        d: RouteDecision = res.parsed
        previous = state.get("last_turn_id")
        correction_of = previous if d.is_correction and previous else None
        if correction_of:
            deps.trace_store.set_corrected(correction_of)
        return {
            "intent": d.intent, "router_reason": d.reason, "llm_calls": calls,
            "report_filter_scope": d.report_filter_scope, "report_filter_text": d.report_filter_text,
            "correction_of": correction_of,
        }

    return router


def make_rating(deps: Deps):
    """Node `rating`: for `/good` and `/bad`. Records the command's +1 / -1 and note on the trace of `last_turn_id` (the
    previous answer-bearing turn of this thread); with none yet it says so and records nothing. No LLM call. Followed by `finalize`."""
    async def rating(state: dict) -> dict:
        last = state.get("last_turn_id")
        if not last:
            return {"rendered": "Nothing to rate yet. Ask a question first.", "outcome": "ok"}
        cmd = state.get("command") or {}
        value = int(cmd.get("rating", 0))
        deps.trace_store.set_rating(last, value, cmd.get("note", ""))
        return {"rendered": f"Thanks, recorded a thumbs {'up' if value > 0 else 'down'} for the last answer.", "outcome": "rated"}

    return rating


def make_help(deps: Deps):
    """Node `help`: for `/help`. Renders `HELP_TEXT`, outcome "help"; no LLM call. Followed by `finalize`."""
    async def help_node(state: dict) -> dict:
        return {"rendered": HELP_TEXT, "outcome": "help"}

    return help_node


def make_refuse(deps: Deps):
    """Node `refuse`: for `off_topic` intents. Writes the fixed `REFUSAL_TEXT` with outcome "refused"; no LLM call.
    Followed by `finalize`."""
    async def refuse(state: dict) -> dict:
        return {"rendered": REFUSAL_TEXT, "outcome": "refused"}

    return refuse


def make_ack_preference(deps: Deps):
    """Node `ack_preference`: for `preference_update` intents. Parses the question with the regex rules in
    `memory.prefs.extract_preferences`, applies the changes to the stored preferences, and writes a confirmation
    of the current settings, or a hint listing example phrases if nothing was recognised. Brevity requests count
    toward `brevity_streak` (three in a row lower the depth one step).
    No LLM call. Followed by `finalize`; `update_memory` skips this intent so changes are not applied twice."""
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
