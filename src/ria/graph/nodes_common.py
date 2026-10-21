"""Helpers shared by all graph nodes: the LLM call wrapper with its per-turn budget, timing, prompt context, and the generic edge functions.

Every node calls the model through `call_llm`, which enforces `max_llm_calls_per_turn` and records the call in
the turn's ledger. `timed` wraps each node when the graph is built. Depends on `graph.deps` and `llm.client`; the
node modules in this package depend on it.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ria.config import join_statuses, render_definitions
from ria.graph.deps import Deps
from ria.llm.client import LLMResult, LLMUnavailable


class BudgetExceeded(RuntimeError):
    """Raised when a turn would exceed its LLM-call budget."""


Node = Callable[[dict], Awaitable[dict]]


def timed(name: str, fn: Node, clock=time.perf_counter) -> Node:
    """Wrap a node to append `{node, ms}` to `node_timings` and to convert `LLMUnavailable` and `BudgetExceeded`
    into `{"error": ..., "outcome": "error"}`, which the edge functions route to `fail`. Other exceptions are
    deliberately not caught: they are bugs and should surface."""
    async def wrapped(state: dict) -> dict:
        t0 = clock()
        try:
            out = await fn(state)
        except (LLMUnavailable, BudgetExceeded) as e:
            out = {"error": f"{type(e).__name__}: {e}", "outcome": "error"}
        out = dict(out or {})
        out["node_timings"] = list(state.get("node_timings", [])) + [{"node": name, "ms": int((clock() - t0) * 1000)}]
        return out

    wrapped.__name__ = name
    return wrapped


async def call_llm(
    deps: Deps, state: dict, *, node: str, system: str, messages: Sequence[BaseMessage],
    schema: type[BaseModel] | None = None,
) -> tuple[LLMResult, list[dict]]:
    """Make one LLM call for `node` and return `(result, updated_llm_calls)`; the caller puts the list back in state
    under `llm_calls`. Raises `BudgetExceeded` before calling if the turn already used its allowance, so a repair
    loop cannot spend without limit. With `schema`, the result's `parsed` holds the validated pydantic object."""
    calls = list(state.get("llm_calls", []))
    if len(calls) >= deps.settings.max_llm_calls_per_turn:
        raise BudgetExceeded(f"turn already used {len(calls)} LLM calls")
    res = await deps.llm.complete(system=system, messages=messages, schema=schema, node=node)
    return res, calls + [res.ledger()]


def window(state: dict, deps: Deps) -> list[BaseMessage]:
    """The most recent `2 * window_turns` messages (user and assistant), the part of history sent to the model."""
    msgs = list(state.get("messages", []))
    return msgs[-(2 * deps.settings.window_turns):]


def _plural(column: str) -> str:
    return {"brand": "brands", "category": "categories"}.get(column, column + "s")


def scope_text(state: dict, column: str) -> str:
    """Plain-language description of the user's scope on `column` (the `Settings.scope_column`), for prompts and user-facing messages."""
    scope = state["user"].get("scope")
    plural = _plural(column)
    return f"all {plural}" if scope is None else f"only these {plural}: " + ", ".join(scope)


def base_system(deps: Deps, state: dict) -> dict:
    """Common template values for prompts: persona, business definitions, user name and role, scope text and `scope_column`, preference summary (and the
    `Prefs` object itself under `prefs_obj`), and the conversation summary block when one exists. Callers pick the keys
    their prompt template uses."""
    prefs = deps.prefs_store.get(state["user"]["id"])
    summary = state.get("summary") or ""
    org = deps.org_defaults()
    return {
        "persona": deps.persona(),
        "definitions": render_definitions(org),
        "revenue_statuses": join_statuses(org.revenue_statuses),
        "user_name": state["user"]["name"],
        "user_role": state["user"]["role"],
        "scope": scope_text(state, deps.settings.scope_column),
        "scope_column": deps.settings.scope_column,
        "prefs": prefs.summary(),
        "prefs_obj": prefs,
        "summary_block": f"Summary of earlier conversation: {summary}" if summary else "",
    }


def route_intent(state: dict) -> str:
    """Edge after `router`: error set -> "fail"; otherwise the intent name, which `build.INTENT_ROUTES` maps to a node.
    A missing intent is treated as off_topic."""
    return "fail" if state.get("error") else state.get("intent") or "off_topic"


def after(ok_target: str) -> Callable[[dict], str]:
    """Build an edge function for a straight-line step: go to `ok_target`, or to "fail" if `state["error"]` is set."""
    def _route(state: dict) -> str:
        return "fail" if state.get("error") else ok_target

    _route.__name__ = f"after_{ok_target}"
    return _route
