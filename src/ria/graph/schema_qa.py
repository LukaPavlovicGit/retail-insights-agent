from __future__ import annotations

from langchain_core.messages import HumanMessage

from ria.graph.deps import Deps
from ria.graph.nodes_common import base_system, call_llm
from ria.graph.prompts import SCHEMA_SYSTEM


def make_schema_answer(deps: Deps):
    async def schema_answer(state: dict) -> dict:
        ctx = base_system(deps, state)
        system = SCHEMA_SYSTEM.format(persona=ctx["persona"], schema_text=deps.schema_text, prefs=ctx["prefs"])
        res, calls = await call_llm(deps, state, node="schema_answer", system=system,
                                    messages=[HumanMessage(content=state["question"])])
        return {"rendered": res.text.strip(), "outcome": "ok", "llm_calls": calls}

    return schema_answer
