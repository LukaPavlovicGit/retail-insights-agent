from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class LLMUnavailable(RuntimeError):
    """Raised when every model in the chain failed."""


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    fallback: bool
    node: str

    def ledger(self) -> dict:
        return {
            "node": self.node,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "fallback": self.fallback,
        }


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None = None,
        node: str = "",
    ) -> LLMResult: ...


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_fences(text: str) -> str:
    m = _FENCE.match(text)
    return (m.group(1) if m else text).strip()
