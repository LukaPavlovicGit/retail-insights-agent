"""Text embeddings for Golden Trio retrieval.

`Embedder` is the interface; `GeminiEmbedder` calls the Gemini embedding API and `FakeEmbedder` is a deterministic
offline stand-in for tests. Questions are embedded as "query" and stored trios as "document", the task types the
Gemini model is trained for. Used by `ria init` (documents), `golden.retrieve` (queries) and `graph.deps`.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Literal, Protocol

from ria.config import Settings

Kind = Literal["query", "document"]


class Embedder(Protocol):
    """Turns text into fixed-length float vectors of length `dim`. `kind` selects the task type ("query" for user
    questions, "document" for stored trios). Implementations: `GeminiEmbedder` (real) and `FakeEmbedder` (tests)."""

    dim: int

    async def embed_many(self, texts: list[str], kind: Kind = "document") -> list[list[float]]: ...

    async def embed(self, text: str, kind: Kind = "query") -> list[float]: ...


class GeminiEmbedder:
    """Embedder over `settings.embedding_model`, truncated to `settings.embedding_dim` dimensions so the vectors
    fit the `trio_vec` table. Needs `settings.gemini_api_key`; the SDK is imported when constructed."""

    def __init__(self, settings: Settings):
        from google import genai

        self.dim = settings.embedding_dim
        self.model = settings.embedding_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def embed_many(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        resp = await self._client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
        )
        return [list(e.values) for e in resp.embeddings]

    async def embed(self, text: str, kind: Kind = "query") -> list[float]:
        return (await self.embed_many([text], kind))[0]


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for tests. Similar wording -> nearby vectors."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_many(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed(self, text: str, kind: Kind = "query") -> list[float]:
        return self._vec(text)
