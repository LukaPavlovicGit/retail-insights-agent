"""Non-interactive pieces of the CLI, kept separate so they are testable without a terminal.

`run_init` backs `ria init` (load users and seed Golden Trios, embed them, cache the live schema, check user categories).
`chat_loop` is the read-eval-print loop behind `ria chat`; it drives `graph.run.run_turn` and handles the delete confirmation.
All I/O (prompting, printing, BigQuery, embeddings) is injected, so tests call these with fakes."""
from __future__ import annotations

import asyncio
import json
from typing import Callable

from ria import db
from ria.config import Settings
from ria.golden.embed import Embedder
from ria.golden.loader import load_trios_dir
from ria.golden.store import TrioStore
from ria.graph.run import run_turn
from ria.guards.schema import STATIC_SCHEMA


def run_init(settings: Settings, conn, embedder: Embedder, schema_fetch: Callable[[], dict] | None,
             live_categories: Callable[[], list[str]] | None) -> dict:
    """Prepare the local database and data files for chat. Safe to re-run.

    Loads users.yaml, replaces the seed Golden Trios, and embeds every trio that has no vector yet (including
    user-saved candidates). If `schema_fetch` returns a schema containing every static table it is
    written to `settings.schema_path`; any failure or partial result keeps the built-in schema. If
    `live_categories` is given, user categories missing from the dataset are reported, not rejected.
    Returns a summary dict (counts, `schema_source` "live" or "static", `unknown_categories`, `category_check`
    "ok" or "failed") that the CLI prints. Embedding errors propagate; schema and category lookups never raise.
    """
    users =db.load_users(conn, settings.users_path)
    trios = load_trios_dir(settings.trios_dir)
    store = TrioStore(conn)
    store.replace_seed(trios)
    missing = store.missing_embeddings()
    if missing:
        vectors = asyncio.run(embedder.embed_many([t.question for t in missing], kind="document"))
        for t, v in zip(missing, vectors):
            store.set_embedding(t.id, v)
    schema_source = "static"
    if schema_fetch is not None:
        try:
            live = schema_fetch()
            if live and set(live) >= set(STATIC_SCHEMA):
                settings.schema_path.parent.mkdir(parents=True, exist_ok=True)
                settings.schema_path.write_text(json.dumps(live, indent=2))
                schema_source = "live"
        except Exception:  # noqa: BLE001 - fall back to the static schema
            schema_source = "static"
    unknown: list[str] = []
    category_check = "ok"
    if live_categories is not None:
        try:
            known = set(live_categories())
            for u in users:
                for c in u["categories"] or []:
                    if c not in known:
                        unknown.append(c)
        except Exception:  # noqa: BLE001
            unknown = []
            category_check = "failed"
    return {"users": len(users), "trios": len(trios), "embedded": len(missing), "schema_source": schema_source,
            "unknown_categories": sorted(set(unknown)), "category_check": category_check}


def parse_confirmation(text: str) -> bool:
    """Only an explicit "yes" or "y" confirms a delete; anything else, including empty input, cancels."""
    return text.strip().lower() in {"yes", "y"}


def format_interrupt(payload: dict) -> str:
    """Render the `confirm_delete` interrupt payload (`{"action", "reports": [{id, title, created_at}]}`) as a prompt."""
    reports = payload.get("reports", [])
    lines = [f"This will permanently delete {len(reports)} report(s):", ""]
    lines += [f"- #{r['id']}  {r['title']}  ({r.get('created_at', '')})" for r in reports]
    lines += ["", "Type 'yes' to confirm, anything else to cancel."]
    return "\n".join(lines)


async def chat_loop(app, user: dict, thread_id: str, prompt_fn, print_fn, run_turn_fn=run_turn, status_fn=None) -> None:
    """Run the interactive loop for one user and thread until EOF or Ctrl-C at the prompt.

    `prompt_fn` is async; `status_fn` prints the dim turn line (defaults to `print_fn`). Each input goes through
    `run_turn_fn`. If the turn pauses with `needs_confirmation` (the graph's delete interrupt), the loop shows the
    reports, asks for confirmation, and resumes the same thread with `{"confirmed": bool}`; a missing answer counts
    as "no". Exceptions from a turn are printed and the loop continues, so one bad turn never ends the session.
    """
    status_fn = status_fn or print_fn
    print_fn(f"Chatting as {user['name']} ({user['role']}). Thread {thread_id}. Ctrl-D to exit.")
    while True:
        try:
            text = await prompt_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            print_fn("bye")
            return
        if not text.strip():
            continue
        try:
            result = await run_turn_fn(app, user, thread_id, text)
            if result.outcome == "needs_confirmation" and result.interrupt:
                print_fn(format_interrupt(result.interrupt))
                try:
                    answer = await prompt_fn("confirm> ")
                except (EOFError, KeyboardInterrupt):
                    answer = ""
                result = await run_turn_fn(app, user, thread_id, "", resume={"confirmed": parse_confirmation(answer)})
            print_fn(result.rendered or "(no answer)")
            status_fn(f"[turn {result.turn_id} · {result.outcome}]")
        except KeyboardInterrupt:
            print_fn("Turn cancelled.")
        except Exception as e:  # noqa: BLE001 - the loop must never die on one bad turn
            print_fn(f"Something went wrong: {type(e).__name__}: {e}")
