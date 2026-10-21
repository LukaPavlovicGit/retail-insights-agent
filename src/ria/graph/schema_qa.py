"""Node for "what data do you have?" questions.

Answers from the schema digest only (personal columns already removed by `guards.schema.schema_digest`), so it
makes no BigQuery call. Depends on `graph.nodes_common` and `graph.prompts.SCHEMA_SYSTEM`; reached from the router's `schema_question` intent.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from ria.graph.deps import Deps
from ria.graph.nodes_common import base_system, call_llm
from ria.graph.prompts import SCHEMA_SYSTEM


def make_schema_overview(deps: Deps):
    """Node `schema_overview`: for `/schema`. Renders `deps.schema_text` under a one-line intro (outcome "ok"); no LLM call. Followed by `finalize`."""
    async def schema_overview(state: dict) -> dict:
        return {"rendered": f"Here are the tables and columns I can query:\n\n{deps.schema_text}", "outcome": "ok"}

    return schema_overview


def make_schema_answer(deps: Deps):
    """Node `schema_answer`: reads `question` and the user's preferences; writes `rendered` (outcome "ok"). One LLM
    call (`SCHEMA_SYSTEM`) grounded in `deps.schema_text`. Followed by `finalize`, or `fail` if an error is set."""
    async def schema_answer(state: dict) -> dict:
        ctx = base_system(deps, state)
        system = SCHEMA_SYSTEM.format(persona=ctx["persona"], schema_text=deps.schema_text, prefs=ctx["prefs"])
        res, calls = await call_llm(deps, state, node="schema_answer", system=system,
                                    messages=[HumanMessage(content=state["question"])])
        return {"rendered": res.text.strip(), "outcome": "ok", "llm_calls": calls}

    return schema_answer
