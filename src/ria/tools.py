from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol

from langchain_core.tools import BaseTool

from ria.config import Settings


class ToolBox(Protocol):
    async def bq_dry_run(self, sql: str) -> dict: ...
    async def bq_query(self, sql: str) -> dict: ...
    async def bq_schema(self) -> dict: ...
    async def reports_save(self, user_id: str, thread_id: str, title: str, question: str, sql: str, body: str, tags: list[str]) -> dict: ...
    async def reports_search(self, user_id: str, text: str | None = None, thread_id: str | None = None) -> dict: ...
    async def reports_delete(self, user_id: str, ids: list[int]) -> dict: ...


def _decode(out: Any) -> dict:
    if isinstance(out, tuple):  # (content, artifact)
        out = out[0]
    if hasattr(out, "content"):
        out = out.content
    if isinstance(out, list):
        out = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in out)
    return json.loads(out) if isinstance(out, str) else dict(out)


class MCPToolBox:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {t.name: t for t in tools}

    async def _call(self, name: str, **kwargs) -> dict:
        return _decode(await self._tools[name].ainvoke(kwargs))

    async def bq_dry_run(self, sql: str) -> dict:
        return await self._call("bq_dry_run", sql=sql)

    async def bq_query(self, sql: str) -> dict:
        return await self._call("bq_query", sql=sql)

    async def bq_schema(self) -> dict:
        return await self._call("bq_schema")

    async def reports_save(self, user_id, thread_id, title, question, sql, body, tags) -> dict:
        return await self._call("reports_save", user_id=user_id, thread_id=thread_id, title=title, question=question, sql=sql, body=body, tags=list(tags))

    async def reports_search(self, user_id, text=None, thread_id=None) -> dict:
        return await self._call("reports_search", user_id=user_id, text=text, thread_id=thread_id)

    async def reports_delete(self, user_id, ids) -> dict:
        return await self._call("reports_delete", user_id=user_id, ids=list(ids))


@asynccontextmanager
async def open_mcp_toolbox(settings: Settings) -> AsyncIterator[MCPToolBox]:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    env = {
        **os.environ,
        "RIA_DB_PATH": str(settings.db_path),
        "RIA_DATASET": settings.dataset,
        "RIA_BQ_MAX_BYTES": str(settings.bq_max_bytes),
        "RIA_BQ_TIMEOUT_S": str(settings.bq_timeout_s),
        "RIA_ROW_CAP": str(settings.row_cap),
        "RIA_EMBEDDING_DIM": str(settings.embedding_dim),
    }
    if settings.gcp_project:
        env["GOOGLE_CLOUD_PROJECT"] = settings.gcp_project
    client = MultiServerMCPClient(
        {"ria": {"transport": "stdio", "command": sys.executable, "args": ["-m", "ria.mcp_server.server"], "env": env}}
    )
    async with client.session("ria") as session:
        tools = await load_mcp_tools(session)
        yield MCPToolBox(tools)


class FakeToolBox:
    DEFAULTS = {
        "bq_dry_run": {"ok": True, "bytes": 1000, "error": None, "error_kind": None},
        "bq_query": {"ok": True, "rows": [{"value": 1}], "row_count": 1, "bytes": 1000, "error": None, "error_kind": None},
        "bq_schema": {"source": "static", "schema": {}},
        "reports_save": {"ok": True, "id": 1},
        "reports_search": {"reports": []},
        "reports_delete": {"deleted": 0},
    }

    def __init__(self, responses: dict[str, list[dict]] | None = None):
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: list[tuple[str, dict]] = []

    async def _call(self, name: str, **kwargs) -> dict:
        self.calls.append((name, kwargs))
        queue = self.responses.get(name)
        return dict(queue.pop(0)) if queue else dict(self.DEFAULTS[name])

    async def bq_dry_run(self, sql: str) -> dict:
        return await self._call("bq_dry_run", sql=sql)

    async def bq_query(self, sql: str) -> dict:
        return await self._call("bq_query", sql=sql)

    async def bq_schema(self) -> dict:
        return await self._call("bq_schema")

    async def reports_save(self, user_id, thread_id, title, question, sql, body, tags) -> dict:
        return await self._call("reports_save", user_id=user_id, thread_id=thread_id, title=title, question=question, sql=sql, body=body, tags=list(tags))

    async def reports_search(self, user_id, text=None, thread_id=None) -> dict:
        return await self._call("reports_search", user_id=user_id, text=text, thread_id=thread_id)

    async def reports_delete(self, user_id, ids) -> dict:
        return await self._call("reports_delete", user_id=user_id, ids=list(ids))
