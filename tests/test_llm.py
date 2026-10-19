from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from ria.config import Settings
from ria.llm.client import LLMUnavailable, strip_fences
from ria.llm.fake import FakeLLM
from ria.llm.gemini import GeminiClient


class Decision(BaseModel):
    intent: str
    reason: str = ""


class StubModel:
    """Stands in for ChatGoogleGenerativeAI: pops scripted outcomes per ainvoke."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.structured = False

    def with_structured_output(self, schema, include_raw=False, **kw):
        s = StubModel([])
        s.outcomes = self.outcomes  # share the parent's script
        s.structured = True
        s.calls_ref = self
        return s

    async def ainvoke(self, messages):
        owner = getattr(self, "calls_ref", self)
        owner.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        if self.structured and not isinstance(out, dict):
            return {"raw": out, "parsed": Decision(intent=out.text), "parsing_error": None}
        return out


def ai(text="hi"):
    return AIMessage(content=text, usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})


class Quota(Exception):
    code = 429


@pytest.fixture
def no_sleep():
    async def sleep(_):
        return None
    return sleep


async def test_success_first_try(no_sleep):
    primary = StubModel([ai("ok")])
    c = GeminiClient(Settings(), primary=primary, fallback=StubModel([]), sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")], node="router")
    assert r.text == "ok" and r.model == "gemini-3.8-flash" and not r.fallback
    assert r.tokens_in == 7 and r.tokens_out == 3 and r.node == "router"
    assert r.ledger()["node"] == "router"


async def test_structured_output(no_sleep):
    primary = StubModel([ai("analysis")])
    c = GeminiClient(Settings(), primary=primary, fallback=StubModel([]), sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")], schema=Decision)
    assert isinstance(r.parsed, Decision) and r.parsed.intent == "analysis"


async def test_retries_on_429_then_succeeds():
    sleeps = []

    async def sleep(s):
        sleeps.append(s)

    primary = StubModel([Quota(), Quota(), ai("ok")])
    c = GeminiClient(Settings(), primary=primary, fallback=StubModel([]), sleep=sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r.text == "ok" and not r.fallback and primary.calls == 3
    assert sleeps == [2.0, 4.0]


async def test_falls_back_after_retries_exhausted(no_sleep):
    primary = StubModel([Quota(), Quota(), Quota()])
    fallback = StubModel([ai("fb")])
    c = GeminiClient(Settings(), primary=primary, fallback=fallback, sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r.text == "fb" and r.fallback and r.model == "gemini-3.5-flash-lite"


async def test_non_retryable_error_goes_straight_to_fallback(no_sleep):
    primary = StubModel([ValueError("bad request")])
    fallback = StubModel([ai("fb")])
    c = GeminiClient(Settings(), primary=primary, fallback=fallback, sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r.fallback and primary.calls == 1
    primary.outcomes.append(ai("back"))
    r2 = await c.complete(system="s", messages=[HumanMessage("q")])
    assert primary.calls == 2 and r2.text == "back" and not r2.fallback  # not marked down


async def test_structured_retries_then_fallback(no_sleep):
    primary = StubModel([Quota(), Quota(), Quota()])
    c = GeminiClient(Settings(), primary=primary, fallback=StubModel([ai("fb")]), sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")], schema=Decision)
    assert r.parsed.intent == "fb" and r.fallback and primary.calls == 3


async def test_parse_failures_fall_back_without_marking_primary_down(no_sleep):
    bad = {"raw": ai("x"), "parsed": None, "parsing_error": "bad json"}
    primary = StubModel([bad, bad, bad, ai("again")])
    c = GeminiClient(Settings(), primary=primary, fallback=StubModel([ai("fb")]), sleep=no_sleep)
    r = await c.complete(system="s", messages=[HumanMessage("q")], schema=Decision)
    assert r.parsed.intent == "fb" and r.fallback and primary.calls == 3
    r2 = await c.complete(system="s", messages=[HumanMessage("q")], schema=Decision)
    assert primary.calls == 4 and r2.parsed.intent == "again" and not r2.fallback


async def test_schema_binding_error_is_a_provider_error(no_sleep):
    class BadBind(StubModel):
        def with_structured_output(self, schema, include_raw=False, **kw):
            raise TypeError("cannot bind")

    c = GeminiClient(Settings(), primary=BadBind([]), fallback=BadBind([]), sleep=no_sleep)
    with pytest.raises(LLMUnavailable):
        await c.complete(system="s", messages=[HumanMessage("q")], schema=Decision)


async def test_raises_when_both_fail(no_sleep):
    c = GeminiClient(Settings(), primary=StubModel([Quota()] * 3), fallback=StubModel([Quota()] * 3), sleep=no_sleep)
    with pytest.raises(LLMUnavailable):
        await c.complete(system="s", messages=[HumanMessage("q")])


async def test_primary_marked_down_is_skipped(no_sleep):
    t = [100.0]
    primary = StubModel([Quota()] * 3 + [ai("back")])
    fallback = StubModel([ai("fb1"), ai("fb2")])
    c = GeminiClient(Settings(), primary=primary, fallback=fallback, sleep=no_sleep, clock=lambda: t[0])
    r1 = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r1.fallback and primary.calls == 3
    r2 = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r2.text == "fb2" and primary.calls == 3  # primary skipped while down
    t[0] += 61
    r3 = await c.complete(system="s", messages=[HumanMessage("q")])
    assert r3.text == "back" and not r3.fallback


async def test_fake_llm_scripts_per_node():
    f = FakeLLM({"router": [Decision(intent="analysis")], "analyze": ["Revenue rose."]}, default="dflt")
    r1 = await f.complete(system="s", messages=[], schema=Decision, node="router")
    assert r1.parsed.intent == "analysis" and r1.model == "fake"
    r2 = await f.complete(system="s", messages=[], node="analyze")
    assert r2.text == "Revenue rose."
    r3 = await f.complete(system="s", messages=[], node="other")
    assert r3.text == "dflt" and len(f.calls) == 3


async def test_fake_llm_accepts_dict_for_schema_and_raises_exceptions():
    f = FakeLLM({"router": [{"intent": "off_topic"}, LLMUnavailable("down")]})
    r = await f.complete(system="s", messages=[], schema=Decision, node="router")
    assert r.parsed.intent == "off_topic"
    with pytest.raises(LLMUnavailable):
        await f.complete(system="s", messages=[], schema=Decision, node="router")


def test_strip_fences():
    assert strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"
    assert strip_fences("```\nSELECT 2\n```\n") == "SELECT 2"
    assert strip_fences("SELECT 3") == "SELECT 3"
