"""Find the Golden Trios most similar to a user's question. Called by the graph's `retrieve_trios` node;
combines `golden.embed` (vectors) with `golden.store` (search and scope filtering)."""
from __future__ import annotations

from ria.golden.embed import Embedder
from ria.golden.loader import Trio
from ria.golden.store import TrioStore


async def retrieve_trios(
    store: TrioStore, embedder: Embedder, question: str, scope: list[str] | None, k: int = 3
) -> list[Trio]:
    """Embed `question` as a query and return up to `k` promoted trios visible to a user with `scope` (None = all), nearest first."""
    vec = await embedder.embed(question, kind="query")
    return [t for t, _ in store.search(vec, scope, k=k)]
