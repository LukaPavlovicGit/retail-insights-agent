from __future__ import annotations

import asyncio
import uuid
from contextlib import closing
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from ria import db
from ria.bootstrap import chat_loop, run_init
from ria.config import Settings
from ria.golden.store import TrioStore
from ria.observability.report import render_stats, render_trace
from ria.observability.trace import TraceStore

app = typer.Typer(help="Retail Insights Agent", no_args_is_help=True)
golden_app = typer.Typer(help="Review Golden Trio candidates")
app.add_typer(golden_app, name="golden")
console = Console()


def _settings() -> Settings:
    return Settings.from_env()


def _conn(settings: Settings):
    conn = db.connect(settings.db_path)
    db.init_schema(conn, settings.embedding_dim)
    return conn


@app.command()
def init() -> None:
    """Load users and Golden Trios, embed them, cache the live schema."""
    settings = _settings()
    if not settings.gemini_api_key:
        console.print("[red]GEMINI_API_KEY is not set (see .env.example).[/red]")
        raise typer.Exit(1)
    from ria.golden.embed import GeminiEmbedder
    from ria.mcp_server.bigquery import BigQueryRunner

    conn = _conn(settings)
    runner = BigQueryRunner(settings.gcp_project, settings.dataset, settings.bq_max_bytes, settings.bq_timeout_s, settings.row_cap)

    def categories() -> list[str]:
        res = runner.query(f"SELECT DISTINCT category FROM `{settings.dataset}.products` ORDER BY 1")
        if not res["ok"]:
            raise RuntimeError(res["error"])
        return [r["category"] for r in res["rows"]]

    report = run_init(settings, conn, GeminiEmbedder(settings), schema_fetch=runner.live_schema, live_categories=categories)
    console.print(f"users: {report['users']}  trios: {report['trios']}  newly embedded: {report['embedded']}  schema: {report['schema_source']}")
    if report["unknown_categories"]:
        console.print(f"[yellow]Categories in users.yaml not found in the dataset: {report['unknown_categories']}[/yellow]")
    if report["category_check"] == "failed":
        console.print("[yellow]Could not validate categories against BigQuery.[/yellow]")
    if report["schema_source"] == "static":
        console.print("[yellow]Could not read the live schema from BigQuery; using the built-in one. Run `ria doctor`.[/yellow]")


@app.command()
def doctor() -> None:
    """Check the Gemini key, Google credentials, and a BigQuery dry run."""
    settings = _settings()
    ok = True
    console.print(f"GEMINI_API_KEY: {'set' if settings.gemini_api_key else '[red]missing[/red]'}")
    ok &= bool(settings.gemini_api_key)
    console.print(f"GOOGLE_CLOUD_PROJECT: {settings.gcp_project or '[yellow]not set (default credentials project will be used)[/yellow]'}")
    try:
        import google.auth

        creds, project = google.auth.default()
        console.print(f"Google credentials: ok (project {project or settings.gcp_project})")
    except Exception as e:  # noqa: BLE001
        console.print(f"Google credentials: [red]{e}[/red]  -> run `gcloud auth application-default login`")
        ok = False
    if ok:
        from ria.mcp_server.bigquery import BigQueryRunner

        r = BigQueryRunner(settings.gcp_project, settings.dataset, settings.bq_max_bytes, settings.bq_timeout_s, settings.row_cap)
        dry = r.dry_run(f"SELECT 1 FROM `{settings.dataset}.products` LIMIT 1")
        console.print(f"BigQuery dry run: {'ok' if dry['ok'] else '[red]' + str(dry['error']) + '[/red]'}")
        ok &= bool(dry["ok"])
        from ria.llm.gemini import GeminiClient
        from langchain_core.messages import HumanMessage

        try:
            res = asyncio.run(GeminiClient(settings).complete(system="Reply with OK.", messages=[HumanMessage(content="ping")], node="doctor"))
            console.print(f"Gemini: ok ({res.model})")
        except Exception as e:  # noqa: BLE001
            console.print(f"Gemini: [red]{e}[/red]")
            ok = False
    console.print("[green]All checks passed.[/green]" if ok else "[red]Fix the items above, then run `ria init`.[/red]")
    raise typer.Exit(0 if ok else 1)


@app.command()
def chat(user: str = typer.Option(..., "--user", "-u", help="Demo user id from data/users.yaml"),
         thread: str | None = typer.Option(None, "--thread", "-t", help="Conversation id to resume")) -> None:
    """Start the chat loop."""
    settings = _settings()
    conn = _conn(settings)
    u = db.get_user(conn, user)
    if u is None:
        console.print(f"[red]Unknown user '{user}'. Run `ria init` and pick an id from data/users.yaml.[/red]")
        raise typer.Exit(1)
    if not settings.gemini_api_key:
        console.print("[red]GEMINI_API_KEY is not set (see .env.example).[/red]")
        raise typer.Exit(1)
    if TrioStore(conn).missing_embeddings():
        console.print("[yellow]Golden Trios are not embedded yet; run `ria init`.[/yellow]")

    async def main() -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        from ria.graph.build import build_graph
        from ria.graph.deps import Deps
        from ria.tools import open_mcp_toolbox

        thread_id = thread or uuid.uuid4().hex[:8]
        session = PromptSession()

        async def prompt_fn(p: str) -> str:
            with patch_stdout():
                return await session.prompt_async(p)

        def print_fn(text: str) -> None:
            console.print(Markdown(str(text)))

        def status_fn(text: str) -> None:
            console.print(text, style="dim", markup=False)

        async with open_mcp_toolbox(settings) as toolbox, AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as cp:
            deps = Deps.live(settings, conn, toolbox)
            graph = build_graph(deps, cp)
            await chat_loop(graph, u, thread_id, prompt_fn, print_fn, status_fn=status_fn)

    asyncio.run(main())


@app.command()
def trace(turn_id: str) -> None:
    """Show one turn's trace as a tree."""
    with closing(_conn(_settings())) as conn:
        t = TraceStore(conn).get(turn_id)
    if t is None:
        console.print(f"turn {turn_id} not found")
        raise typer.Exit(1)
    console.print(render_trace(t))


@app.command()
def stats(since: str | None = typer.Option(None, help="ISO timestamp lower bound")) -> None:
    """Agent-level metrics."""
    with closing(_conn(_settings())) as conn:
        data = TraceStore(conn).stats(since=since)
    console.print(render_stats(data))


@golden_app.command("list")
def golden_list(status: str | None = typer.Option(None, help="pending | promoted | rejected")) -> None:
    with closing(_conn(_settings())) as conn:
        trios = TrioStore(conn).list(status=status)
    for t in trios:
        console.print(f"#{t.id} [{t.status}] ({t.source}) {t.question}", markup=False)


@golden_app.command("promote")
def golden_promote(trio_id: int) -> None:
    with closing(_conn(_settings())) as conn:
        TrioStore(conn).set_status(trio_id, "promoted")
    console.print(f"promoted #{trio_id}; run `ria init` if it has no embedding yet")


@golden_app.command("reject")
def golden_reject(trio_id: int) -> None:
    with closing(_conn(_settings())) as conn:
        TrioStore(conn).set_status(trio_id, "rejected")
    console.print(f"rejected #{trio_id}")


@app.command("eval")
def eval_cmd(only: str | None = typer.Option(None, help="Run a single case id"),
             cases: Path = typer.Option(Path("evals/questions.yaml"), help="Case file")) -> None:
    """Run the evaluation set against live services (needs credentials)."""
    from rich.table import Table

    from ria.evals.checks import load_cases
    from ria.evals.runner import run_evals

    settings = _settings()
    with closing(_conn(settings)) as conn:
        rows = asyncio.run(run_evals(settings, conn, load_cases(cases), only=only))
    table = Table(title="Eval results")
    for col in ("id", "outcome", "score", "failures"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["id"], r["outcome"], str(r["score"]), "; ".join(r["failures"]) or "-")
    console.print(table)
    failed = sum(1 for r in rows if r["failures"])
    console.print(f"{len(rows) - failed}/{len(rows)} cases passed deterministic checks")
    raise typer.Exit(1 if failed else 0)


if __name__ == "__main__":
    app()
