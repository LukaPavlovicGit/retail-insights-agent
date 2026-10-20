"""Parse the model's SQL into an AST and enforce the safety rules in code, never in the prompt.

Rules, in order: exactly one SELECT (or set operation); only tables from the allowed dataset and schema; columns
resolved against the schema; no personal columns on `users`, including whole-row references such as `SELECT u`;
per-user category scope injected into every SELECT that reads products or order_items; a row cap. The graph's `guard_sql` node calls
`guard_sql` before every BigQuery dry run, so the model is never trusted to enforce any of this itself.
Depends on sqlglot and `guards.schema`; it does no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

from ria.guards.schema import PII_COLUMNS, STATIC_SCHEMA, qualify_schema

SET_OP = getattr(exp, "SetOperation", exp.Union)


SCOPE_REASON = (
    "Your access is limited to specific product categories, so every query must involve "
    "products through order_items (join order_items.product_id = products.id)."
)


@dataclass
class GuardResult:
    """Outcome of `guard_sql`. When `ok`, `sql` is the rewritten, safe-to-run BigQuery SQL and `reason` is None. When
    rejected, `sql` is None and `reason` is a message the repair prompt can act on. `events` lists the checks
    that ran (`code:ok`, `code:rejected`, `scope:injected`, `limit:clamped` and so on) and ends up in the trace."""

    ok: bool
    sql: str | None
    events: list[str] = field(default_factory=list)
    reason: str | None = None


def _tables_of(select: exp.Select) -> dict[str, str]:
    """table name -> alias for tables whose nearest enclosing SELECT is `select`."""
    out: dict[str, str] = {}
    for t in select.find_all(exp.Table):
        if t.parent_select is select and t.db:  # no db = CTE reference, not a dataset table
            out[t.name] = t.alias or t.name
    return out


def guard_sql(
    sql: str,
    *,
    categories: list[str] | None,
    dataset: str,
    row_cap: int = 200,
    schema: dict | None = None,
) -> GuardResult:
    """Validate and rewrite `sql`; returns a `GuardResult` and never raises on bad SQL.

    `categories` is the user's scope: a list means every SELECT over products or order_items is filtered to those
    categories (an inner join to products is added through order_items when the SELECT has none; a SELECT whose
    tables include neither is rejected, as is a query where nothing could be scoped), while None means unrestricted. `dataset` is `project.dataset`. `schema` defaults to the static
    schema. The LIMIT is added if missing and clamped to `row_cap` if larger or not a plain number.
    """
    events: list[str] = []

    def reject(code: str, reason: str) -> GuardResult:
        events.append(f"{code}:rejected")
        return GuardResult(False, None, events, reason)

    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except ParseError as e:
        return reject("parse", f"SQL could not be parsed: {e}")
    if len(statements) != 1:
        return reject("single_statement", "Exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, SET_OP)):
        return reject("select_only", "Only SELECT queries are allowed")
    events.append("select_only:ok")

    catalog, db = dataset.split(".")
    cte_names = {cte.alias for cte in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if t.name in cte_names and not t.db:
            continue
        if (t.catalog, t.db) != (catalog, db):
            return reject("dataset", f"Table {t.sql(dialect='bigquery')} is outside the allowed dataset {dataset}")
        if t.name not in (schema or STATIC_SCHEMA):
            return reject("table", f"Table {t.name} is not available to this assistant (allowed: orders, order_items, products, users)")
    events.append("dataset:ok")
    events.append("table:ok")

    try:
        tree = qualify(tree, schema=qualify_schema(dataset, schema), dialect="bigquery")
    except OptimizeError as e:
        return reject("resolve", f"Could not resolve columns: {e}")

    alias_to_tables: dict[str, set[str]] = {}
    for t in tree.find_all(exp.Table):
        alias_to_tables.setdefault(t.alias or t.name, set()).add(t.name)
    # sqlglot >= 30 resolves a bare table alias (whole-row reference) to exp.TableColumn
    for col in tree.find_all(exp.Column, exp.TableColumn):
        if (isinstance(col, exp.TableColumn) or not col.table) and "users" in alias_to_tables.get(col.name, set()):
            return reject("pii", "Whole-row references to users are not allowed")
        if isinstance(col, exp.Column) and "users" in alias_to_tables.get(col.table, set()) and col.name in PII_COLUMNS:
            return reject("pii", f"Column users.{col.name} is personal data and may not be queried")
    events.append("pii:ok")

    if categories is None:
        events.append("scope:unrestricted")
    else:
        injected = 0
        for sel in list(tree.find_all(exp.Select)):
            sources = _tables_of(sel)
            if "products" in sources:
                alias = sources["products"]
            elif "order_items" in sources:
                oi = sources["order_items"]
                sel.join(
                    f"`{dataset}.products` AS ria_p",
                    on=f"ria_p.id = {oi}.product_id",
                    join_type="inner",
                    dialect="bigquery",
                    copy=False,
                )
                alias = "ria_p"
                events.append("scope:join_added")
            elif sources:
                return reject("scope", SCOPE_REASON)
            else:
                continue
            sel.where(exp.column("category", table=alias, quoted=True).isin(*categories), copy=False)
            injected += 1
        if injected == 0:
            return reject("scope", SCOPE_REASON)
        events.append("scope:injected")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(row_cap)
        events.append("limit:added")
    else:
        try:
            current = int(limit.expression.this)
        except (TypeError, ValueError, AttributeError):
            current = row_cap + 1
        if current > row_cap:
            tree = tree.limit(row_cap)
            events.append("limit:clamped")
        else:
            events.append("limit:kept")

    return GuardResult(True, tree.sql(dialect="bigquery"), events, None)
