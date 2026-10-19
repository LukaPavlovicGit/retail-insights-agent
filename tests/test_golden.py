from __future__ import annotations

from pathlib import Path

import pytest

from ria.golden.embed import FakeEmbedder
from ria.golden.loader import Trio, load_trios_dir
from ria.golden.retrieve import retrieve_trios
from ria.golden.store import TrioStore
from ria.guards.sql_guard import guard_sql

SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "golden_trios"


def trio(q, tags=("all",), **kw):
    return Trio(id=None, question=q, sql="SELECT 1", report="r", tags=list(tags), **kw)


@pytest.fixture
def store(conn):
    return TrioStore(conn)


async def test_fake_embedder_is_deterministic_and_semantic():
    e = FakeEmbedder(dim=32)
    a = await e.embed("who are our top customers by spend")
    b = await e.embed("top customers by total spend")
    c = await e.embed("monthly revenue trend")
    assert a == await e.embed("who are our top customers by spend")
    assert len(a) == 32

    def cos(x, y):
        dot = sum(i * j for i, j in zip(x, y))
        nx = sum(i * i for i in x) ** 0.5
        ny = sum(i * i for i in y) ** 0.5
        return dot / (nx * ny)

    assert cos(a, b) > cos(a, c)


def test_load_trios_dir(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "01.yaml").write_text(
        "question: Top customers\nsql: SELECT 1\nreport: Customers rock.\ntags: [all]\nauthor: analyst\ncreated_at: 2026-01-01\n"
    )
    (d / "02.yaml").write_text("question: Jeans\nsql: SELECT 2\nreport: Jeans rock.\ntags: [Jeans]\n")
    trios = load_trios_dir(d)
    assert [t.question for t in trios] == ["Top customers", "Jeans"]
    assert trios[0].author == "analyst" and trios[1].tags == ["Jeans"] and trios[1].source == "seed"


def test_load_trios_dir_reports_bad_file(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "bad.yaml").write_text("question: Only a question\n")
    with pytest.raises(ValueError) as e:
        load_trios_dir(d)
    assert "bad.yaml" in str(e.value) and "sql" in str(e.value)


def test_store_roundtrip_and_status(store):
    i = store.insert(trio("Q1", source="user", status="pending"))
    t = store.get(i)
    assert t.question == "Q1" and t.status == "pending"
    assert [x.id for x in store.list(status="pending")] == [i]
    store.set_status(i, "promoted")
    assert store.get(i).status == "promoted"


def test_replace_seed_keeps_user_trios(store):
    store.replace_seed([trio("A"), trio("B")])
    u = store.insert(trio("U", source="user", status="pending"))
    ids = store.replace_seed([trio("C")])
    questions = sorted(t.question for t in store.list())
    assert questions == ["C", "U"] and len(ids) == 1


async def test_search_filters_by_status_and_tags(store):
    e = FakeEmbedder(dim=32)
    ids = {}
    for q, tags, status in [
        ("top customers by spend", ["all"], "promoted"),
        ("top customers by spend in jeans", ["Jeans"], "promoted"),
        ("top customers by spend in swimwear", ["Swim"], "promoted"),
        ("top customers by spend pending", ["all"], "pending"),
    ]:
        i = store.insert(trio(q, tags, status=status))
        store.set_embedding(i, await e.embed(q, kind="document"))
        ids[q] = i
    assert store.missing_embeddings() == []
    vec = await e.embed("top customers by spend")
    got = [t.question for t, _ in store.search(vec, categories=["Jeans"], k=5)]
    assert "top customers by spend" in got
    assert "top customers by spend in jeans" in got
    assert "top customers by spend in swimwear" not in got
    assert "top customers by spend pending" not in got
    got_all = [t.question for t, _ in store.search(vec, categories=None, k=5)]
    assert "top customers by spend in swimwear" in got_all


async def test_search_respects_k_and_distance(store):
    e = FakeEmbedder(dim=32)
    for q in ["revenue by month", "revenue by week", "revenue by day", "elephants in the zoo"]:
        i = store.insert(trio(q))
        store.set_embedding(i, await e.embed(q, kind="document"))
    vec = await e.embed("revenue by month")
    res = store.search(vec, categories=None, k=2)
    assert len(res) == 2 and res[0][0].question == "revenue by month" and res[0][1] < 1e-6
    far = store.search(await e.embed("elephants in the zoo"), categories=None, k=4, max_distance=0.1)
    assert [t.question for t, _ in far] == ["elephants in the zoo"]


async def test_retrieve_trios(store):
    e = FakeEmbedder(dim=32)
    for q in ["monthly revenue trend", "top customers"]:
        i = store.insert(trio(q))
        store.set_embedding(i, await e.embed(q, kind="document"))
    got = await retrieve_trios(store, e, "show me the monthly revenue trend", categories=None, k=1)
    assert [t.question for t in got] == ["monthly revenue trend"]


def test_seed_trios_pass_guard_for_scoped_and_unscoped_users():
    trios = load_trios_dir(SEED_DIR)
    assert len(trios) == 15
    for t in trios:
        for cats in (None, ["Jeans", "Pants"]):
            r = guard_sql(t.sql, categories=cats, dataset="bigquery-public-data.thelook_ecommerce")
            assert r.ok, f"{t.question} ({cats}): {r.reason}"
