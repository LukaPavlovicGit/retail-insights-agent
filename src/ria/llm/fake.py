from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ria.llm.client import LLMResult


class FakeLLM:
    """Scripted LLM for tests. `script` maps node name -> list of outputs consumed in order.

    An output may be a str, a pydantic instance, a dict (validated against `schema`), or an
    Exception instance (raised).
    """

    def __init__(self, script: dict[str, list[Any]] | None = None, default: str = "ok"):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None = None,
        node: str = "",
    ) -> LLMResult:
        self.calls.append({"node": node, "system": system, "messages": list(messages), "schema": schema})
        queue = self.script.get(node)
        out: Any = queue.pop(0) if queue else self.default
        if isinstance(out, Exception):
            raise out
        parsed = None
        if schema is not None:
            parsed = schema.model_validate(out) if isinstance(out, dict) else out
            text = parsed.model_dump_json() if isinstance(parsed, BaseModel) else str(parsed)
        else:
            text = out if isinstance(out, str) else str(out)
        return LLMResult(text, parsed, "fake", 10, 5, 1, False, node)
