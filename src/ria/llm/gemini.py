from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from ria.config import Settings
from ria.llm.client import LLMResult, LLMUnavailable

_RETRYABLE = re.compile(
    r"\b(429|500|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|DEADLINE_EXCEEDED|DEADLINE|timed out|timeout)\b",
    re.IGNORECASE,
)


def is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 500, 503, 504):
        return True
    return bool(_RETRYABLE.search(f"{type(exc).__name__}: {exc}"))


def _usage(msg: Any) -> tuple[int, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


def _text_of(msg: Any) -> str:
    t = getattr(msg, "text", None)
    if isinstance(t, str):  # property in langchain-core 1.x (a callable str subclass; don't call it)
        return str(t)
    if callable(t):
        t = t()
        if isinstance(t, str):
            return t
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class GeminiClient:
    retries = 3
    backoff_s = 2.0
    down_for_s = 60.0

    def __init__(self, settings: Settings, *, primary=None, fallback=None, sleep=asyncio.sleep, clock=time.monotonic):
        self.settings = settings
        self.primary_name = settings.primary_model
        self.fallback_name = settings.fallback_model
        self.primary = primary or self._make(settings.primary_model)
        self.fallback = fallback or self._make(settings.fallback_model)
        self._sleep = sleep
        self._clock = clock
        self._primary_down_until = 0.0

    def _make(self, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=self.settings.gemini_api_key, temperature=0, max_retries=0, timeout=60
        )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None = None,
        node: str = "",
    ) -> LLMResult:
        chain = [("primary", self.primary, self.primary_name), ("fallback", self.fallback, self.fallback_name)]
        if self._clock() < self._primary_down_until:
            chain = chain[1:]
        payload = [SystemMessage(content=system), *messages]
        last_err: Exception | None = None
        for label, model, name in chain:
            retryable_failures = 0
            for attempt in range(self.retries):
                t0 = self._clock()
                try:
                    runnable = model.with_structured_output(schema, include_raw=True) if schema else model
                    out = await runnable.ainvoke(payload)
                except Exception as e:  # noqa: BLE001 - provider errors vary by SDK
                    last_err = e
                    if not is_retryable(e):
                        break
                    retryable_failures += 1
                    if attempt < self.retries - 1:
                        await self._sleep(self.backoff_s * (2**attempt))
                    continue
                latency = int((self._clock() - t0) * 1000)
                if schema:
                    raw, parsed = out["raw"], out["parsed"]
                    if parsed is None:
                        last_err = RuntimeError(f"structured output failed: {out.get('parsing_error')}")
                        continue
                    tin, tout = _usage(raw)
                    return LLMResult(_text_of(raw), parsed, name, tin, tout, latency, label == "fallback", node)
                tin, tout = _usage(out)
                return LLMResult(_text_of(out), None, name, tin, tout, latency, label == "fallback", node)
            # Only a budget fully spent on retryable errors marks the primary down; a 400 or parse failure does not.
            if label == "primary" and retryable_failures == self.retries:
                self._primary_down_until = self._clock() + self.down_for_s
        raise LLMUnavailable(f"All models failed; last error: {last_err}")
