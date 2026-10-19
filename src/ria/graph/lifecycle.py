from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from ria.graph.deps import Deps
from ria.graph.nodes_common import call_llm
from ria.graph.prompts import ERROR_TEXT, SUMMARIZE_SYSTEM
from ria.graph.state import new_turn_fields
from ria.guards.pii_mask import mask_pii
from ria.memory.prefs import extract_preferences
from ria.observability.trace import TurnTrace


def make_begin_turn(deps: Deps):
    async def begin_turn(state: dict) -> dict:
        msgs = list(state.get("messages", []))
        question = state.get("question") or (msgs[-1].content if msgs and isinstance(msgs[-1], HumanMessage) else "")
        out = new_turn_fields(state.get("turn_id") or "", question, state.get("started_at") or deps.now(), deps.clock())
        keep = 2 * deps.settings.window_turns
        if len(msgs) > keep + 2:
            older = msgs[:-keep]
            transcript = "\n".join(f"{m.type}: {m.content}" for m in older)
            res, calls = await call_llm(
                deps, out, node="summarize",
                system=SUMMARIZE_SYSTEM.format(existing=state.get("summary") or "(none)"),
                messages=[HumanMessage(content=transcript)],
            )
            out["summary"] = res.text.strip()
            out["llm_calls"] = calls
            out["messages"] = [RemoveMessage(id=m.id) for m in older if m.id]
        return out

    return begin_turn


def make_finalize(deps: Deps):
    async def finalize(state: dict) -> dict:
        rendered = state.get("rendered")
        outcome = state.get("outcome") or ("error" if state.get("error") else "ok")
        if rendered is None:
            rendered = ERROR_TEXT.format(turn_id=state.get("turn_id", "?"))
            outcome = "error" if outcome in ("", "ok") else outcome
        masked, _ = mask_pii(rendered)
        out: dict = {"rendered": masked, "outcome": outcome, "messages": [AIMessage(content=masked)]}
        if state.get("intent") == "analysis" and outcome == "ok" and state.get("analysis"):
            out.update(last_question=state.get("question"), last_sql=state.get("sql"), last_analysis=state.get("analysis"))
        return out

    return finalize


def make_update_memory(deps: Deps):
    async def update_memory(state: dict) -> dict:
        if state.get("intent") in ("preference_update", "schema_question"):
            return {}
        changes = extract_preferences(state.get("question", ""))
        if changes:
            deps.prefs_store.apply(state["user"]["id"], changes)
        return {}

    return update_memory


def make_write_trace(deps: Deps):
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
