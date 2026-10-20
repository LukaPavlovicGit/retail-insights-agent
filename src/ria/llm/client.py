"""Provider-neutral LLM interface used by the graph.

Defines the `LLMClient` protocol, the `LLMResult` every call returns, the `LLMUnavailable` error, and a helper for
cleaning model SQL. `llm.gemini.GeminiClient` is the real implementation and `llm.fake.FakeLLM` the test one, so nodes never import a provider SDK.
"""
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
    """One completed model call: raw `text`, the validated pydantic object in `parsed` (None when no schema was
    requested), the `model` that answered, token counts, `latency_ms`, whether the `fallback` model was used, and the
    calling `node` name. `ledger()` turns it into the dict stored in `llm_calls` and later in the trace."""

    text: str
    parsed: Any
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    fallback: bool
    node: str

    def ledger(self) -> dict:
        """Metadata only (node, model, tokens, latency, fallback flag); the prompt and answer text are not kept."""
        return {
            "node": self.node,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "fallback": self.fallback,
        }


class LLMClient(Protocol):
    """What a node needs from a model: `complete(system, messages, schema, node)` returns an `LLMResult`.

    With `schema` (a pydantic class) the reply is parsed into it and returned as `result.parsed`. Implementations:
    `GeminiClient` (real, with retries and fallback model) and `FakeLLM` (scripted, for tests). Raises `LLMUnavailable` if no model could answer.
    """

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
    """Remove a surrounding Markdown code fence (with or without a language tag) and trim whitespace; unfenced text is only trimmed."""
    m = _FENCE.match(text)
    return (m.group(1) if m else text).strip()
