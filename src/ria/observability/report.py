"""Rich renderers for `ria trace` and `ria stats`: a trace as a tree, aggregate stats as a table.

Pure presentation over `observability.trace` data; user-controlled strings (SQL, errors) are escaped so Rich does not read them as markup.
"""
from __future__ import annotations

from rich.markup import escape
from rich.table import Table
from rich.tree import Tree

from ria.observability.trace import TurnTrace


def render_trace(t: TurnTrace) -> Tree:
    """One turn as a tree: header, node timings, LLM calls (tokens, latency, fallback flag), SQL attempts with their
    results, guard events, and any error."""
    root = Tree(
        f"turn {escape(t.turn_id)}  user={escape(t.user_id)} intent={escape(str(t.intent))} "
        f"outcome={escape(t.outcome)} {t.duration_ms}ms"
    )
    nodes = root.add("nodes")
    for n in t.node_timings:
        nodes.add(f"{escape(str(n['node']))}: {n['ms']}ms")
    llm = root.add(f"llm calls ({len(t.llm_calls)}, tokens in/out {t.tokens_in}/{t.tokens_out})")
    for c in t.llm_calls:
        flag = f" {escape('[fallback]')}" if c.get("fallback") else ""
        llm.add(
            f"{escape(str(c['node']))}: {escape(str(c['model']))} "
            f"{c.get('tokens_in', 0)}/{c.get('tokens_out', 0)} tok {c.get('latency_ms', 0)}ms{flag}"
        )
    sql = root.add(f"sql attempts ({len(t.sql_attempts)}, repairs={t.repairs}, bytes={t.bq_bytes})")
    for i, a in enumerate(t.sql_attempts, 1):
        status = f"error: {escape(str(a['error']))}" if a.get("error") else f"{a.get('row_count', 0)} rows"
        sql.add(f"#{i} {status}\n{escape(a['sql'])}")
    guards = root.add("guard events")
    for e in t.guard_events:
        guards.add(escape(e))
    if t.error:
        root.add(f"error: {escape(t.error)}")
    return root


def render_stats(s: dict) -> Table:
    """Format the dict from `TraceStore.stats` as a two-column table; with zero turns it shows only the turn count."""
    table = Table(title="Agent stats")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("turns", str(s.get("turns", 0)))
    if s.get("turns", 0) == 0:
        return table
    for key in ("success_rate", "refusal_rate", "error_rate", "fallback_rate"):
        table.add_row(key, f"{s[key]:.0%}")
    table.add_row("mean_repairs", f"{s['mean_repairs']:.2f}")
    table.add_row("guard_rejections", ", ".join(f"{k}={v}" for k, v in s["guard_rejections"].items()) or "none")
    table.add_row("latency p50/p95 ms", f"{s['p50_ms']}/{s['p95_ms']}")
    table.add_row("mean tokens/turn", f"{s['mean_tokens']:.0f}")
    table.add_row("mean BQ bytes/turn", f"{s['mean_bq_bytes']:.0f}")
    table.add_row("by_intent", ", ".join(f"{k}={v}" for k, v in s["by_intent"].items()))
    return table
