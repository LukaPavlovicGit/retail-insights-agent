from __future__ import annotations

import pytest

from ria.graph.commands import HELP_TEXT, ParsedCommand, parse_command


@pytest.mark.parametrize("text", ["hello", "/unknown", "", "show me /reports", "/report abc", "/report", "/reportsx", "/goodness"])
def test_non_commands_return_none(text):
    assert parse_command(text) is None


def test_simple_commands_ignore_case_and_whitespace():
    assert parse_command("  /REPORTS \n").kind == "reports"
    assert parse_command("/Schema").kind == "schema"
    assert parse_command("/help").kind == "help"
    assert parse_command("/report 3") == ParsedCommand(kind="report", arg="3")


def test_delete_filters():
    assert parse_command("/delete from this conversation") == ParsedCommand(kind="delete", filter_scope="conversation")
    assert parse_command("/delete conversation").filter_scope == "conversation"
    assert parse_command("/delete ALL").filter_scope == "all"
    c = parse_command("/delete mentioning Acme")
    assert (c.filter_scope, c.filter_text) == ("text", "Acme")
    c = parse_command("/delete about Jeans Q2")
    assert (c.filter_scope, c.filter_text) == ("text", "Jeans Q2")
    assert parse_command("/delete Acme").filter_text == "Acme"
    assert parse_command("/delete").filter_scope is None


def test_ratings():
    assert parse_command("/good") == ParsedCommand(kind="rating", rating=1)
    c = parse_command("/bad too slow")
    assert (c.kind, c.rating, c.note) == ("rating", -1, "too slow")


def test_help_text_lists_every_command():
    for cmd in ("/reports", "/report", "/delete", "/schema", "/good", "/bad", "/help"):
        assert cmd in HELP_TEXT
