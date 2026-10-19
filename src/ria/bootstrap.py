"""Non-interactive pieces of the CLI, kept separate so they are testable without a terminal."""
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
    users = db.load_users(conn, settings.users_path)
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
    return text.strip().lower() in {"yes", "y"}


def format_interrupt(payload: dict) -> str:
    reports = payload.get("reports", [])
    lines = [f"This will permanently delete {len(reports)} report(s):", ""]
    lines += [f"- #{r['id']}  {r['title']}  ({r.get('created_at', '')})" for r in reports]
    lines += ["", "Type 'yes' to confirm, anything else to cancel."]
    return "\n".join(lines)


async def chat_loop(app, user: dict, thread_id: str, prompt_fn, print_fn, run_turn_fn=run_turn, status_fn=None) -> None:
    """`prompt_fn` is async; `status_fn` prints the dim turn line (defaults to `print_fn`)."""
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
