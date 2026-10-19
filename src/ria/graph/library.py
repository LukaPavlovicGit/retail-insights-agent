from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ria.golden.loader import Trio
from ria.graph.deps import Deps
from ria.graph.nodes_common import call_llm
from ria.graph.prompts import TITLE_SYSTEM
from ria.guards.pii_mask import mask_pii


class ReportMeta(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list)


def make_save_report(deps: Deps):
    async def save_report(state: dict) -> dict:
        analysis = state.get("last_analysis")
        if not analysis:
            return {"rendered": "There is no analysis in this conversation to save yet. Ask a data question first, then say \"save this as a report\".", "outcome": "ok"}
        analysis = mask_pii(analysis)[0]
        question = state.get("last_question") or state["question"]
        sql = state.get("last_sql") or ""
        res, calls = await call_llm(
            deps, state, node="title",
            system=TITLE_SYSTEM.format(question=question, analysis=analysis[:1500]),
            messages=[HumanMessage(content="Title and tags, please.")], schema=ReportMeta,
        )
        meta: ReportMeta = res.parsed
        saved = await deps.toolbox.reports_save(
            state["user"]["id"], state.get("thread_id", ""), meta.title, question, sql, analysis, meta.tags
        )
        rid = saved.get("id")
        deps.trio_store.insert(Trio(
            id=None, question=question, sql=sql, report=analysis, tags=list(state["user"]["categories"] or ["all"]),
            status="pending", source="user", author=state["user"]["id"],
        ))
        text = (f"Saved report #{rid}: {meta.title}. It is also queued as a Golden Trio candidate for analyst review "
                f"(`ria golden list`).")
        return {"rendered": text, "outcome": "saved", "llm_calls": calls}

    return save_report


def make_resolve_targets(deps: Deps):
    async def resolve_targets(state: dict) -> dict:
        user_id = state["user"]["id"]
        scope = state.get("report_filter_scope")
        text = (state.get("report_filter_text") or "").strip() or None
        if scope == "conversation":
            found = await deps.toolbox.reports_search(user_id, text=None, thread_id=state.get("thread_id"))
            what = "from this conversation"
        elif scope == "text" and text:
            found = await deps.toolbox.reports_search(user_id, text=text, thread_id=None)
            what = f"mentioning \"{text}\""
        elif scope == "all":
            found = await deps.toolbox.reports_search(user_id, text=None, thread_id=None)
            what = "in your library"
        else:
            return {"pending_delete": None, "outcome": "ok",
                    "rendered": "Which reports should I delete? Say \"mentioning <subject>\", \"from this conversation\", or \"all my reports\"."}
        reports = [{"id": r["id"], "title": r["title"], "created_at": r.get("created_at", "")} for r in found.get("reports", [])]
        if not reports:
            return {"pending_delete": None, "outcome": "ok", "rendered": f"No reports {what} were found. Nothing to delete."}
        return {"pending_delete": reports}

    return resolve_targets


def route_after_resolve(state: dict) -> str:
    if state.get("error"):
        return "fail"
    return "confirm_delete" if state.get("pending_delete") else "finalize"


def make_confirm_delete(deps: Deps):
    async def confirm_delete(state: dict) -> dict:
        pending = state.get("pending_delete") or []
        answer = interrupt({"action": "delete_reports", "reports": pending})
        confirmed = bool(answer.get("confirmed")) if isinstance(answer, dict) else False
        if not confirmed:
            return {"rendered": "Cancelled. Nothing was deleted.", "outcome": "cancelled", "pending_delete": None}
        ids = [r["id"] for r in pending]
        res = await deps.toolbox.reports_delete(state["user"]["id"], ids)
        n = int(res.get("deleted", 0))
        titles = "; ".join(r["title"] for r in pending)
        return {"rendered": f"Deleted {n} report{'s' if n != 1 else ''}: {titles}.", "outcome": "deleted", "pending_delete": None}

    return confirm_delete
