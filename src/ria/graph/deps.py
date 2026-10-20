"""Dependency container passed to every graph node.

`Deps` bundles settings, the LLM client, embedder, toolbox, stores and the schema text so nodes take one
argument and tests can swap any piece. `Deps.live` builds the real Gemini-backed set; `Deps.for_tests` builds
the fake-friendly set. Depends on almost every lower layer (guards, llm, golden, memory, observability, persona, tools).
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ria.config import Settings
from ria.golden.embed import Embedder, FakeEmbedder
from ria.golden.store import TrioStore
from ria.guards.schema import load_schema, schema_digest
from ria.llm.client import LLMClient
from ria.memory.prefs import PrefsStore
from ria.observability.trace import TraceStore
from ria.persona.loader import load_persona
from ria.tools import ToolBox


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Deps:
    """Everything a node may use. Built once per process by `live` or `for_tests`, never by nodes.

    `schema` is the table-to-columns map used by the SQL guard; `schema_text` is the digest of it shown to the
    model (personal columns left out). `clock` (monotonic seconds) and `now` (ISO UTC string) are injectable so
    timings and timestamps are deterministic in tests.
    """

    settings: Settings
    llm: LLMClient
    embedder: Embedder
    toolbox: ToolBox
    conn: sqlite3.Connection
    trio_store: TrioStore
    prefs_store: PrefsStore
    trace_store: TraceStore
    schema: dict
    schema_text: str
    clock: Callable[[], float] = field(default=time.perf_counter)
    now: Callable[[], str] = field(default=_now)

    def persona(self) -> str:
        """Persona text for prompts, re-read from disk on each call so edits to persona.md apply without a restart."""
        return load_persona(self.settings.persona_path)

    @classmethod
    def for_tests(cls, conn, settings, llm, toolbox, embedder=None) -> "Deps":
        """Deps with caller-supplied `llm` and `toolbox` (usually `FakeLLM` and `FakeToolBox`), a hashing `FakeEmbedder`
        by default, and no JSONL trace file."""
        schema = load_schema(settings.schema_path)
        return cls(
            settings=settings, llm=llm, embedder=embedder or FakeEmbedder(settings.embedding_dim), toolbox=toolbox,
            conn=conn, trio_store=TrioStore(conn), prefs_store=PrefsStore(conn), trace_store=TraceStore(conn),
            schema=schema, schema_text=schema_digest(schema, settings.dataset),
        )

    @classmethod
    def live(cls, settings, conn, toolbox) -> "Deps":
        """Production Deps: `GeminiClient` and `GeminiEmbedder`, and traces also appended to `traces.jsonl` next to the database."""
        from ria.golden.embed import GeminiEmbedder
        from ria.llm.gemini import GeminiClient

        schema = load_schema(settings.schema_path)
        return cls(
            settings=settings, llm=GeminiClient(settings), embedder=GeminiEmbedder(settings), toolbox=toolbox,
            conn=conn, trio_store=TrioStore(conn), prefs_store=PrefsStore(conn),
            trace_store=TraceStore(conn, jsonl_path=settings.db_path.parent / "traces.jsonl"),
            schema=schema, schema_text=schema_digest(schema, settings.dataset),
        )
