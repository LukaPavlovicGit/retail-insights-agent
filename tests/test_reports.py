from ria.mcp_server.reports import ReportStore


def test_save_search_delete_scoped_to_user(conn):
    s = ReportStore(conn)
    a1 = s.save("alice", "t1", "Q1 Jeans review", "how did jeans do", "SELECT 1", "Jeans grew 10%.", ["jeans"])
    a2 = s.save("alice", "t2", "Acme account", "acme", "SELECT 2", "Acme spend fell.", [])
    b1 = s.save("bob", "t1", "Acme swimwear", "acme swim", "SELECT 3", "Acme bought swim.", [])
    assert [r.id for r in s.search("alice")] == [a1, a2]
    assert [r.id for r in s.search("alice", text="acme")] == [a2]
    assert [r.id for r in s.search("alice", thread_id="t1")] == [a1]
    assert [r.id for r in s.search("bob", text="acme")] == [b1]
    assert s.get("alice", b1) is None and s.get("bob", b1).title == "Acme swimwear"
    assert s.delete("alice", [a2, b1]) == 1  # bob's report is untouched
    assert s.get("bob", b1) is not None and s.get("alice", a2) is None


def test_search_matches_title_body_and_question(conn):
    s = ReportStore(conn)
    s.save("alice", "t", "Title", "question about Client X", "SELECT 1", "body", [])
    s.save("alice", "t", "Client X summary", "q", "SELECT 1", "body", [])
    s.save("alice", "t", "T", "q", "SELECT 1", "mentions client x in body", [])
    s.save("alice", "t", "Other", "q", "SELECT 1", "nothing", [])
    assert len(s.search("alice", text="client x")) == 3
