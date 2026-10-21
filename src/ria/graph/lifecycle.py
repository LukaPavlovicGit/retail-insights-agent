"""Nodes that run on every turn regardless of intent: start-of-turn setup, end-of-turn finalisation, memory, tracing, and failure.

`begin_turn` opens the turn (and trims long history); `finalize`, `update_memory` and `write_trace` always run last,
in that order, so every turn (including failures and refusals) is masked, learned from and traced. `fail` turns an
error into a user-facing message. Depends on `graph.state`, `guards.pii_mask`, `memory.prefs` and `observability.trace`.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from ria.graph.deps import Deps
from ria.graph.nodes_common import call_llm
from ria.graph.prompts import ERROR_TEXT, SUMMARIZE_SYSTEM
from ria.graph.state import new_turn_fields
from ria.guards.pii_mask import mask_pii
from ria.llm.client import LLMUnavailable
from ria.memory.prefs import extract_preferences
from ria.observability.trace import TurnTrace


ANSWER_OUTCOMES = {"ok", "no_data", "gave_up", "refused", "error", "saved", "deleted", "cancelled"}


def make_begin_turn(deps: Deps):
    """Node `begin_turn`: the first node of every turn. Resets all turn-level state via `new_turn_fields` (conversation-level
    keys such as messages, summary and last_analysis are kept). When the history is longer than the model window
    (`2 * window_turns` messages plus a small margin) it summarises the older messages with one LLM call
    (`SUMMARIZE_SYSTEM`), stores the summary, and removes those messages from state. Followed by `router`, or `fail`
    if the summary call errored."""
    async def begin_turn(state: dict) -> dict:
        msgs = list(state.get("messages", []))
        question = state.get("question") or (msgs[-1].content if msgs and isinstance(msgs[-1], HumanMessage) else "")
        out = new_turn_fields(state.get("turn_id") or "", question, state.get("started_at") or deps.now(), deps.clock())
        keep = 2 * deps.settings.window_turns
        if len(msgs) > keep + 2:
            older = msgs[:-keep]
            transcript = "\n".join(f"{m.type}: {m.content}" for m in older)
            try:
                res, calls = await call_llm(
                    deps, out, node="summarize",
                    system=SUMMARIZE_SYSTEM.format(existing=state.get("summary") or "(none)"),
                    messages=[HumanMessage(content=transcript)],
                )
            except LLMUnavailable:  # models down: keep the messages as they are so commands still work
                out["guard_events"] = ["summary:skipped_llm_unavailable"]
            else:
                out["summary"] = res.text.strip()
                out["llm_calls"] = calls
                out["messages"] = [RemoveMessage(id=m.id) for m in older if m.id]
        return out

    return begin_turn


def make_finalize(deps: Deps):
    """Node `finalize`: the single exit for every branch. Masks PII (emails, phone numbers, street addresses) in
    `rendered` as a last line of defence, substitutes `ERROR_TEXT` if no branch produced text, appends the answer to
    `messages` as an AI message, and, for a successful analysis, records `last_question`, `last_sql` and
    `last_analysis` so "save this as a report" works on a later turn. No LLM call. Followed by `update_memory`."""
    async def finalize(state: dict) -> dict:
        rendered = state.get("rendered")
        outcome = state.get("outcome") or ("error" if state.get("error") else "ok")
        if rendered is None:
            rendered = ERROR_TEXT.format(turn_id=state.get("turn_id", "?"))
            outcome = "error" if outcome in ("", "ok") else outcome
        masked, _ = mask_pii(rendered)
        out: dict = {"rendered": masked, "outcome": outcome, "messages": [AIMessage(content=masked)]}
        # Only answer-bearing turns can be rated or marked corrected later; listings, help, `/schema` and ratings leave it alone.
        if outcome in ANSWER_OUTCOMES and state.get("intent") not in ("rating", "schema_overview"):
            out["last_turn_id"] = state["turn_id"]
        if state.get("intent") == "analysis" and outcome == "ok" and state.get("analysis"):
            out.update(last_question=state.get("question"), last_sql=state.get("sql"), last_analysis=state.get("analysis"))
        return out

    return finalize


def make_update_memory(deps: Deps):
    """Node `update_memory`: quietly picks up formatting preferences ("use tables", "shorter") stated inside any
    message and saves them for next time. Skipped for `preference_update` (already handled by `ack_preference`) and
    `schema_question`. Rule-based, no LLM call, and writes no state. Followed by `write_trace`."""
    async def update_memory(state: dict) -> dict:
        if state.get("intent") in ("preference_update", "schema_question") or state.get("command"):  # commands are not free text
            return {}
        changes = extract_preferences(state.get("question", ""))
        if changes:
            deps.prefs_store.apply(state["user"]["id"], changes)
        return {}

    return update_memory


def make_write_trace(deps: Deps):
    """Node `write_trace`: builds a `TurnTrace` from turn-level state (intent, outcome, timings, LLM ledger, SQL
    attempts, guard events, bytes scanned, error) and saves it through `deps.trace_store`. The last node before END;
    it writes no state. Duration is measured from the `t0` that `begin_turn` recorded."""
    async def write_trace(state: dict) -> dict:
        trace = TurnTrace(
            turn_id=state["turn_id"], thread_id=state.get("thread_id", ""), user_id=state["user"]["id"],
            intent=state.get("intent") or None, outcome=state.get("outcome") or "ok", started_at=state["started_at"],
            duration_ms=int((deps.clock() - state.get("t0", deps.clock())) * 1000),
            node_timings=list(state.get("node_timings", [])), llm_calls=list(state.get("llm_calls", [])),
            sql_attempts=list(state.get("sql_attempts", [])), guard_events=list(state.get("guard_events", [])),
            bq_bytes=int(state.get("result_bytes", 0) or 0), error=state.get("error"),
        )
        deps.trace_store.save(trace)
        return {}

    return write_trace


def make_fail(deps: Deps):
    """Node `fail`: target of every edge when `state["error"]` is set. Picks a user-facing message by error text:
    LLM-call budget exceeded, model unavailable (nothing was lost), or a generic message that quotes the turn id so
    the trace can be found. The error string itself is kept in state for the trace. Followed by `finalize`."""
    async def fail(state: dict) -> dict:
        err = state.get("error") or "unknown error"
        if "BudgetExceeded" in err:
            text = "I hit the per-question budget before finishing. Try a narrower question, or ask again in a moment."
        elif "LLMUnavailable" in err:
            text = "The language model is unavailable right now (rate limit or outage). Nothing was lost; please retry shortly."
        else:
            text = ERROR_TEXT.format(turn_id=state.get("turn_id", "?"))
        return {"rendered": text, "outcome": "error"}

    return fail
