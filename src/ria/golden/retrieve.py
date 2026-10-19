from __future__ import annotations

from ria.golden.embed import Embedder
from ria.golden.loader import Trio
from ria.golden.store import TrioStore


async def retrieve_trios(
    store: TrioStore, embedder: Embedder, question: str, categories: list[str] | None, k: int = 3
) -> list[Trio]:
    vec = await embedder.embed(question, kind="query")
    return [t for t, _ in store.search(vec, categories, k=k)]
