"""Run the evaluation cases through the real graph and score the answers.

`run_evals` is used by `ria eval`. Each case is a normal turn against live Gemini and BigQuery (through the MCP
server), checked with the deterministic rules in `evals.checks` and then graded 1 to 5 by an LLM judge. Judge scores
are informational; only the deterministic failures decide pass or fail. Results are also written to a timestamped JSON file under `evals/`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from ria import db
from ria.config import Settings
from ria.evals.checks import EvalCase, check_case
from ria.graph.build import build_graph
from ria.graph.deps import Deps
from ria.graph.run import run_turn
from ria.llm.client import LLMClient
from ria.tools import open_mcp_toolbox

JUDGE_SYSTEM = """You grade a data assistant's answer for a retail executive.
Question: {question}
Answer:
{answer}

Score 1-5: 5 = directly answers with correct-looking numbers and a clear takeaway; 3 = partially answers or is vague;
1 = wrong, empty, or off-topic. Refusals of off-topic or unsafe requests score 5 when the refusal is appropriate.
Give a one-sentence reason."""


class Judgement(BaseModel):
    """Structured output of the judge call: an integer `score` from 1 to 5 and a one-sentence `reason`."""

    score: int = Field(ge=1, le=5)
    reason: str = ""


async def run_evals(settings: Settings, conn, cases: list[EvalCase], judge: LLMClient | None = None, only: str | None = None) -> list[dict]:
    """Run `cases` (or just the one whose id equals `only`) and return one result dict per case: id, intent,
    outcome, `failures` (from `check_case`), judge `score` and `reason`, and `turn_id`.

    Each case gets its own thread on an in-memory checkpointer. A delete confirmation is always declined, so
    evals never remove reports. A case whose user is not in the database is reported as skipped with a failure.
    `judge` defaults to the live LLM client; a judge error is recorded in `reason` and does not stop the run.
    Needs credentials and spends real API and BigQuery quota.
    """
    rows: list[dict] = []
    async with open_mcp_toolbox(settings) as toolbox:
        deps = Deps.live(settings, conn, toolbox)
        app = build_graph(deps, InMemorySaver())
        judge = judge or deps.llm
        for case in cases:
            if only and case.id != only:
                continue
            user = db.get_user(conn, case.user)
            if user is None:
                rows.append({"id": case.id, "intent": None, "outcome": "skipped", "failures": [f"unknown user {case.user}"], "score": None, "reason": "", "turn_id": ""})
                continue
            thread_id = f"eval-{uuid4().hex[:6]}"
            result = await run_turn(app, user, thread_id, case.question)
            if result.outcome == "needs_confirmation":
                result = await run_turn(app, user, thread_id, "", resume={"confirmed": False})
            trace = deps.trace_store.get(result.turn_id)
            failures = check_case(case, trace, result.rendered or "", user["categories"]) if trace else ["no trace written"]
            score, reason = None, ""
            try:
                j = await judge.complete(system=JUDGE_SYSTEM.format(question=case.question, answer=result.rendered or ""),
                                         messages=[HumanMessage(content="Grade it.")], schema=Judgement, node="judge")
                score, reason = j.parsed.score, j.parsed.reason
            except Exception as e:  # noqa: BLE001 - judge failures must not abort the run
                reason = f"judge failed: {e}"
            rows.append({"id": case.id, "intent": trace.intent if trace else None, "outcome": result.outcome, "failures": failures, "score": score, "reason": reason, "turn_id": result.turn_id})
    out = Path("evals") / f"results-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    return rows
