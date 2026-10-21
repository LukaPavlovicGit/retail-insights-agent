from __future__ import annotations

from ria.memory.prefs import Prefs, PrefsStore, extract_preferences
from ria.persona.loader import DEFAULT_PERSONA, load_persona


def test_extract_format_preferences():
    assert extract_preferences("please show this as a table")["format"] == "table"
    assert extract_preferences("I prefer bullet points")["format"] == "bullets"
    assert extract_preferences("use prose from now on")["format"] == "prose"
    assert "format" not in extract_preferences("what was revenue in March?")


def test_extract_depth_and_charts():
    assert extract_preferences("shorter please")["brevity"] is True
    assert extract_preferences("keep it brief")["brevity"] is True
    assert extract_preferences("go deeper on this")["depth_up"] is True
    assert extract_preferences("more detail next time")["depth_up"] is True
    assert extract_preferences("include charts")["charts"] is True
    assert extract_preferences("no charts please")["charts"] is False


def test_chart_mentions_in_questions_are_not_preferences():
    assert "charts" not in extract_preferences("show me the chart of accounts table")
    assert "format" not in extract_preferences("show me the chart of accounts table")
    assert "charts" not in extract_preferences("Why did the graph of sales dip?")
    assert extract_preferences("I'd like a graph with the answer")["charts"] is True


def test_prefs_store_defaults_and_apply(conn):
    s = PrefsStore(conn)
    assert s.get("alice") == Prefs()
    p, changed = s.apply("alice", {"format": "table", "charts": True})
    assert changed and p.format == "table" and p.charts is True
    assert s.get("alice").format == "table"
    _, changed = s.apply("alice", {"format": "table"})
    assert changed is False


def test_three_brevity_requests_lower_depth(conn):
    s = PrefsStore(conn)
    s.apply("bob", {"brevity": True})
    s.apply("bob", {"brevity": True})
    assert s.get("bob").depth == "standard" and s.get("bob").brevity_streak == 2
    p, _ = s.apply("bob", {"brevity": True})
    assert p.depth == "brief" and p.brevity_streak == 0
    p, _ = s.apply("bob", {"brevity": True})
    assert p.depth == "brief"  # cannot go below brief


def test_depth_up_resets_streak(conn):
    s = PrefsStore(conn)
    s.apply("bob", {"brevity": True})
    p, _ = s.apply("bob", {"depth_up": True})
    assert p.depth == "deep" and p.brevity_streak == 0
    p, _ = s.apply("bob", {"depth_up": True})
    assert p.depth == "deep"


def test_prefs_summary():
    assert Prefs(format="table", depth="brief").summary() == "Format: table. Depth: brief. Charts: no."


def test_load_persona(tmp_path):
    assert load_persona(tmp_path / "missing.md") == DEFAULT_PERSONA
    f = tmp_path / "persona.md"
    f.write_text("# Be terse\n")
    assert load_persona(f) == "# Be terse"
