from __future__ import annotations

from pathlib import Path

from ria.evals.checks import EvalCase, check_case, load_cases
from ria.observability.trace import TurnTrace

CASES = Path(__file__).resolve().parents[1] / "evals" / "questions.yaml"


def trace(intent="analysis", outcome="ok", sql="SELECT 1 FROM `x.y.order_items` oi JOIN `x.y.products` p ON 1=1 WHERE p.brand IN ('A')", attempts=1):
    atts = [{"sql": "bad", "error": "e", "kind": "syntax", "row_count": 0, "bytes": 0}] * (attempts - 1)
    atts.append({"sql": sql, "error": None, "kind": None, "row_count": 1, "bytes": 1})
    return TurnTrace("t", "th", "u", intent, outcome, "2026-10-16T00:00:00+00:00", 10, sql_attempts=atts)


def test_load_cases():
    cases = load_cases(CASES)
    assert len(cases) == 24 and cases[0].id == "schema-overview" and cases[1].tables == ["order_items", "users"]
    assert cases[1].max_repairs == 1 and cases[8].max_repairs == 2


def test_command_cases_pass_with_no_sql():
    cases = {c.id: c for c in load_cases(CASES)}
    for cid, intent, outcome in (("command-reports", "report_list", "listed"), ("command-help", "help", "help")):
        t = TurnTrace("t", "th", "bob", intent, outcome, "2026-10-16T00:00:00+00:00", 1)
        assert check_case(cases[cid], t, "ok", ["Columbia"]) == []


def test_check_case_passes():
    case = EvalCase("c", "alice", "q", "analysis", ["order_items", "products"], "ok", 1, [r"@"])
    assert check_case(case, trace(), "Jeans lead.", ["A"]) == []


def _scope_failures(sql, values, column="brand"):
    case = EvalCase("c", "alice", "q", "analysis", [], "ok", 1, [])
    return [f for f in check_case(case, trace(sql=sql), "x", values, column) if f.startswith("scope:")]


def test_scope_requires_in_predicate():
    failures = _scope_failures("SELECT p.brand, SUM(x) FROM t GROUP BY p.brand", ["A"])
    assert "scope: final SQL has no brand IN predicate" in failures


def test_scope_all_values_present():
    assert _scope_failures("SELECT 1 FROM t WHERE p.brand IN ('A', 'B')", ["A", "B"]) == []


def test_scope_missing_value_named():
    failures = _scope_failures("SELECT 1 FROM t WHERE p.brand IN ('A')", ["A", "B"])
    assert len(failures) == 1 and "'B'" in failures[0]


def test_scope_accepts_escaped_apostrophe_and_semi_join():
    sql = "SELECT 1 FROM t WHERE EXISTS (SELECT 1 FROM p WHERE p.brand IN ('Levi\\'s', 'Carhartt'))"
    assert _scope_failures(sql, ["Levi's", "Carhartt"]) == []


def test_scope_uses_configured_column():
    assert _scope_failures("SELECT 1 FROM t WHERE p.category IN ('A')", ["A"], "category") == []
    assert _scope_failures("SELECT 1 FROM t WHERE p.brand IN ('A')", ["A"], "category") == [
        "scope: final SQL has no category IN predicate"]


def test_scope_skipped_for_unscoped_user():
    assert _scope_failures("SELECT 1 FROM t", None) == []


def test_check_case_reports_each_failure():
    case = EvalCase("c", "alice", "q", "analysis", ["users"], "ok", 0, [r"@"])
    failures = check_case(case, trace(intent="off_topic", outcome="refused", attempts=2, sql="SELECT 1 FROM `x.y.orders`"), "mail me at a@b.c", ["A"])
    assert any("intent" in f for f in failures)
    assert any("outcome" in f for f in failures)
    assert any("users" in f for f in failures)
    assert any("repairs" in f for f in failures)
    assert any("must_not_match" in f for f in failures)
    assert any("scope" in f for f in failures)
