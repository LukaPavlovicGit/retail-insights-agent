from __future__ import annotations

import re

import pytest

from ria.guards.schema import STATIC_SCHEMA, schema_digest, load_schema
from ria.guards.sql_guard import guard_sql

DS = "bigquery-public-data.thelook_ecommerce"
OI = f"`{DS}.order_items`"
P = f"`{DS}.products`"
U = f"`{DS}.users`"
O = f"`{DS}.orders`"


def g(sql, categories=("Levi's", "Carhartt"), **kw):
    scope = None if categories is None else list(categories)
    return guard_sql(sql, scope=scope, scope_column="brand", dataset=DS, **kw)


def test_rejects_non_select():
    r = g(f"DELETE FROM {U} WHERE id = 1", categories=None)
    assert not r.ok and "select_only:rejected" in r.events


def test_rejects_multiple_statements():
    r = g("SELECT 1; SELECT 2", categories=None)
    assert not r.ok and "single_statement:rejected" in r.events


def test_rejects_parse_error():
    r = g("SELEC nope FROM", categories=None)
    assert not r.ok and r.events[-1] == "parse:rejected" and "parsed" in r.reason


def test_rejects_other_dataset():
    r = g("SELECT 1 FROM `other-project.other.users`", categories=None)
    assert not r.ok and "dataset:rejected" in r.events


def test_rejects_pii_direct():
    r = g(f"SELECT email FROM {U}", categories=None)
    assert not r.ok and "pii:rejected" in r.events and "email" in r.reason


def test_rejects_pii_aliased():
    r = g(f"SELECT u.first_name FROM {U} AS u", categories=None)
    assert not r.ok and "pii:rejected" in r.events


def test_rejects_pii_star_on_users():
    r = g(f"SELECT * FROM {U}", categories=None)
    assert not r.ok and "pii:rejected" in r.events


def test_rejects_pii_inside_cte():
    r = g(f"WITH u AS (SELECT email, state FROM {U}) SELECT state FROM u", categories=None)
    assert not r.ok and "pii:rejected" in r.events


def test_allows_demographics():
    r = g(f"SELECT state, COUNT(*) AS n FROM {U} GROUP BY state", categories=None)
    assert r.ok and "pii:ok" in r.events


def test_rejects_unknown_column_with_reason():
    r = g(f"SELECT nope FROM {U}", categories=None)
    assert not r.ok and "resolve:rejected" in r.events and "nope" in r.reason


def test_scope_injected_when_products_present():
    r = g(f"SELECT p.category, SUM(oi.sale_price) AS rev FROM {OI} oi JOIN {P} p ON p.id = oi.product_id GROUP BY 1")
    assert r.ok and "scope:injected" in r.events
    assert "`p`.`brand` IN ('Levi\\'s', 'Carhartt')" in r.sql


def test_scope_adds_join_when_only_order_items():
    r = g(f"SELECT COUNT(*) AS n FROM {OI} oi")
    assert r.ok and "scope:join_added" in r.events and "scope:injected" in r.events
    assert "products" in r.sql and "IN ('Levi\\'s', 'Carhartt')" in r.sql


def test_scope_semijoin_for_users_only():
    r = g(f"SELECT DATE_TRUNC(DATE(created_at), MONTH) AS m, COUNT(*) AS signups FROM {U} GROUP BY m")
    assert r.ok and "scope:semijoin" in r.events
    assert re.search(r"`id` IN \(SELECT .*user_id FROM .*order_items.* JOIN .*products.* WHERE .*`brand` IN \('Levi\\'s', 'Carhartt'\)\)", r.sql, re.S)
    assert r.sql.index("IN (SELECT") < r.sql.index("GROUP BY")


def test_scope_semijoin_for_orders_only_keeps_existing_where():
    r = g(f"SELECT status, COUNT(*) AS n FROM {O} o WHERE o.created_at > '2026-01-01' GROUP BY status")
    assert r.ok and "scope:semijoin" in r.events
    assert "`o`.`created_at` > '2026-01-01' AND `o`.`order_id` IN (SELECT" in r.sql


def test_scope_semijoin_prefers_orders_when_users_and_orders_present():
    r = g(f"SELECT u.state, COUNT(*) AS n FROM {O} o JOIN {U} u ON u.id = o.user_id GROUP BY u.state")
    assert r.ok and "scope:semijoin" in r.events and "`o`.`order_id` IN (SELECT" in r.sql


def test_scope_column_setting():
    r = guard_sql(f"SELECT category FROM {P}", scope=["Jeans"], scope_column="category", dataset=DS)
    assert r.ok and "`category` IN ('Jeans')" in r.sql


def test_scope_skipped_for_unrestricted_user():
    r = g(f"SELECT state, COUNT(*) AS n FROM {U} GROUP BY state", categories=None)
    assert r.ok and "scope:unrestricted" in r.events and "IN (" not in r.sql


def test_scope_injected_inside_cte():
    sql = (
        f"WITH x AS (SELECT p.category AS c, oi.sale_price AS s FROM {OI} oi JOIN {P} p ON p.id = oi.product_id "
        f"WHERE oi.status = 'Complete') SELECT c, SUM(s) AS rev FROM x GROUP BY c"
    )
    r = g(sql)
    assert r.ok and "scope:injected" in r.events
    assert "`oi`.`status` = 'Complete' AND `p`.`brand` IN ('Levi\\'s', 'Carhartt')" in r.sql


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT TO_JSON_STRING(u) AS j FROM {U} u",
        f"SELECT u FROM {U} u",
        f"SELECT * FROM (SELECT u FROM {U} u)",
    ],
)
def test_rejects_whole_row_users_reference(sql):
    r = g(sql, categories=None)
    assert not r.ok and "pii:rejected" in r.events and "Whole-row" in r.reason


def test_scope_orders_with_products_only_in_subquery_gets_semijoin_and_inner_predicate():
    r = g(f"SELECT status, COUNT(*) AS n FROM {O} WHERE EXISTS (SELECT 1 FROM {P}) GROUP BY status")
    assert r.ok and "scope:semijoin" in r.events and "scope:injected" in r.events
    assert "`order_id` IN (SELECT" in r.sql
    assert re.search(r"EXISTS\(SELECT 1 .*FROM .*products.* WHERE `products`\.`brand` IN .*\) AND `orders`\.`order_id` IN \(SELECT", r.sql, re.S)


def test_scope_orders_only_scoped_and_unrestricted():
    sql = f"SELECT status, COUNT(*) AS n FROM {O} GROUP BY status"
    assert "scope:semijoin" in g(sql).events
    r = g(sql, categories=None)
    assert r.ok and "scope:unrestricted" in r.events and "IN (" not in r.sql


def test_scope_injected_in_derived_table_inner_select():
    r = g(f"SELECT t.c FROM (SELECT p.category AS c FROM {P} p) t")
    assert r.ok and "scope:injected" in r.events
    assert r.sql.index("IN ('Levi\\'s', 'Carhartt')") < r.sql.index(") AS `t`")


def test_limit_added_and_clamped():
    r1 = g(f"SELECT category FROM {P}")
    assert r1.ok and "limit:added" in r1.events and r1.sql.rstrip().endswith("LIMIT 200")
    r2 = g(f"SELECT category FROM {P} LIMIT 5000", row_cap=100)
    assert r2.ok and "limit:clamped" in r2.events and r2.sql.rstrip().endswith("LIMIT 100")
    r3 = g(f"SELECT category FROM {P} LIMIT 10")
    assert r3.ok and "limit:kept" in r3.events and r3.sql.rstrip().endswith("LIMIT 10")


def test_category_with_quote_is_escaped():
    r = g(f"SELECT category FROM {P}", categories=["Tops & Tees", "O'Neil"])
    assert r.ok and "'O\\'Neil'" in r.sql  # BigQuery backslash escape, rendered by sqlglot


def test_schema_digest_hides_pii_columns():
    text = schema_digest()
    assert "email" not in text.split("Personal columns")[0]
    assert "traffic_source" in text


def test_load_schema_falls_back(tmp_path):
    assert load_schema(tmp_path / "missing.json") is STATIC_SCHEMA
    (tmp_path / "bad.json").write_text("{not json")
    assert load_schema(tmp_path / "bad.json") is STATIC_SCHEMA


EV = f"`{DS}.events`"


@pytest.mark.parametrize("cats", [None, ["Jeans"]])
def test_rejects_unknown_table_events(cats):
    r = g(f"SELECT e.ip_address FROM {EV} e JOIN {OI} oi ON oi.user_id = e.user_id", categories=cats)
    assert not r.ok and "table:rejected" in r.events and "not available" in r.reason


def test_rejects_select_star_on_unknown_table():
    r = g(f"SELECT * FROM {EV}", categories=None)
    assert not r.ok and "table:rejected" in r.events


def test_known_tables_emit_table_ok():
    r = g(f"SELECT p.category FROM {OI} oi JOIN {P} p ON p.id = oi.product_id", categories=None)
    assert r.ok and "table:ok" in r.events
