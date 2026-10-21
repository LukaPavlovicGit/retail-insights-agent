"""Nodes for the saved-reports branch: save the last analysis, and delete reports after the user confirms.

Reports live in the MCP server's SQLite library, reached only through `deps.toolbox`; every call is scoped to the
current user id. Saving also queues the analysis as a pending Golden Trio candidate for analyst review. Deleting is
a two-step flow (`resolve_targets` finds matches, `confirm_delete` pauses on a LangGraph `interrupt`) so nothing is
deleted without an explicit yes. Depends on `graph.nodes_common`, `graph.prompts.TITLE_SYSTEM`, `guards.pii_mask` and `golden`.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ria.golden.loader import Trio
from ria.graph.deps import Deps
from ria.graph.nodes_common import call_llm
from ria.graph.prompts import TITLE_SYSTEM
from ria.guards.pii_mask import mask_pii
from ria.llm.client import LLMUnavailable


class ReportMeta(BaseModel):
    """Structured output of the titling call: a short `title` and 1 to 3 lowercase `tags` for a saved report."""

    title: str
    tags: list[str] = Field(default_factory=list)


def make_save_report(deps: Deps):
    """Node `save_report`: reads `last_analysis`, `last_question` and `last_sql` (set by `finalize` after the last
    successful analysis); writes `rendered` and `outcome` "saved". One LLM call (`TITLE_SYSTEM`, structured
    `ReportMeta`) for title and tags, then `reports_save` through the toolbox and a `pending` Golden Trio insert
    tagged with the user's scope values. The analysis is PII-masked before it is stored. With nothing to save it
    replies with guidance and makes no calls. Always followed by `finalize` (or `fail`)."""
    async def save_report(state: dict) -> dict:
        analysis = state.get("last_analysis")
        if not analysis:
            return {"rendered": "There is no analysis in this conversation to save yet. Ask a data question first, then say \"save this as a report\".", "outcome": "ok"}
        analysis = mask_pii(analysis)[0]
        question = state.get("last_question") or state["question"]
        sql = state.get("last_sql") or ""
        calls = list(state.get("llm_calls", []))
        try:
            res, calls = await call_llm(
                deps, state, node="title",
                system=TITLE_SYSTEM.format(question=question, analysis=analysis[:1500]),
                messages=[HumanMessage(content="Title and tags, please.")], schema=ReportMeta,
            )
            meta: ReportMeta = res.parsed
        except LLMUnavailable:  # models down: still save, titled with the start of the question
            meta = ReportMeta(title=question[:80])
        saved = await deps.toolbox.reports_save(
            state["user"]["id"], state.get("thread_id", ""), meta.title, question, sql, analysis, meta.tags
        )
        rid = saved.get("id")
        deps.trio_store.insert(Trio(
            id=None, question=question, sql=sql, report=analysis, tags=list(state["user"]["scope"] or ["all"]),
            status="pending", source="user", author=state["user"]["id"],
        ))
        text = (f"Saved report #{rid}: {meta.title}. It is also queued as a Golden Trio candidate for analyst review "
                f"(`ria golden list`).")
        return {"rendered": text, "outcome": "saved", "llm_calls": calls}

    return save_report


def make_list_reports(deps: Deps):
    """Node `list_reports`: for `/reports`. Calls `reports_search` for this user and renders `#id  title  (created_at)`
    lines, newest first, or "You have no saved reports."; outcome "listed". No LLM call. Followed by `finalize`."""
    async def list_reports(state: dict) -> dict:
        found = await deps.toolbox.reports_search(state["user"]["id"])
        reports = sorted(found.get("reports", []), key=lambda r: r["id"], reverse=True)
        if not reports:
            return {"rendered": "You have no saved reports.", "outcome": "listed"}
        lines = [f"#{r['id']}  {r['title']}  ({r.get('created_at', '')})" for r in reports]
        return {"rendered": "Your saved reports:\n" + "\n".join(lines), "outcome": "listed"}

    return list_reports


def make_show_report(deps: Deps):
    """Node `show_report`: for `/report <id>`. Fetches the report with `reports_get` (author only) and renders its title,
    question, body and SQL in a code block; an unknown or foreign id gives "No report #N in your library."; outcome
    "shown". No LLM call. Followed by `finalize`."""
    async def show_report(state: dict) -> dict:
        rid = int((state.get("command") or {}).get("arg") or 0)
        report = (await deps.toolbox.reports_get(state["user"]["id"], rid)).get("report")
        if not report:
            return {"rendered": f"No report #{rid} in your library.", "outcome": "shown"}
        text = (f"Report #{report['id']}: {report['title']}\n\nQuestion: {report['question']}\n\n{report['body']}\n\n"
                f"SQL:\n```sql\n{report['sql']}\n```")
        return {"rendered": text, "outcome": "shown"}

    return show_report


def make_resolve_targets(deps: Deps):
    """Node `resolve_targets`: turns the router's `report_filter_scope` ("conversation", "text" with
    `report_filter_text`, or "all") into a `reports_search` for this user and writes `pending_delete` (id, title,
    created_at per report). If the scope is unclear or nothing matches it writes a `rendered` reply instead, so no
    confirmation is requested. No LLM call. Next node is chosen by `route_after_resolve`."""
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
    """Edge after `resolve_targets`: error set -> fail; reports found -> confirm_delete; otherwise -> finalize
    (the node already wrote the reply)."""
    if state.get("error"):
        return "fail"
    return "confirm_delete" if state.get("pending_delete") else "finalize"


def make_confirm_delete(deps: Deps):
    """Node `confirm_delete`: pauses the turn with `interrupt({"action": "delete_reports", "reports": [...]})` and
    waits for a resume value like `{"confirmed": True}`. Anything other than a dict with a truthy `confirmed`
    cancels. On yes it calls `reports_delete` with the listed ids (the server also filters by user id) and writes
    `rendered` with the count; on no it deletes nothing. Because `interrupt` re-runs this node from the top on
    resume, it must stay free of side effects before the interrupt call. Always followed by `finalize`."""
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
