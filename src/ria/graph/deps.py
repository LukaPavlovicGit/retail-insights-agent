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
        return load_persona(self.settings.persona_path)

    @classmethod
    def for_tests(cls, conn, settings, llm, toolbox, embedder=None) -> "Deps":
        schema = load_schema(settings.schema_path)
        return cls(
            settings=settings, llm=llm, embedder=embedder or FakeEmbedder(settings.embedding_dim), toolbox=toolbox,
            conn=conn, trio_store=TrioStore(conn), prefs_store=PrefsStore(conn), trace_store=TraceStore(conn),
            schema=schema, schema_text=schema_digest(schema, settings.dataset),
        )

    @classmethod
    def live(cls, settings, conn, toolbox) -> "Deps":
        from ria.golden.embed import GeminiEmbedder
        from ria.llm.gemini import GeminiClient

        schema = load_schema(settings.schema_path)
        return cls(
            settings=settings, llm=GeminiClient(settings), embedder=GeminiEmbedder(settings), toolbox=toolbox,
            conn=conn, trio_store=TrioStore(conn), prefs_store=PrefsStore(conn),
            trace_store=TraceStore(conn, jsonl_path=settings.db_path.parent / "traces.jsonl"),
            schema=schema, schema_text=schema_digest(schema, settings.dataset),
        )
