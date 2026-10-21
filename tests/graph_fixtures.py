from __future__ import annotations

from langchain_core.messages import HumanMessage

from ria.graph.deps import Deps
from ria.graph.state import new_turn_fields
from ria.llm.fake import FakeLLM
from ria.tools import FakeToolBox

ALICE = {"id": "alice", "name": "Alice Chen", "role": "VP Denim and Workwear", "scope": ["Levi's", "Carhartt", "Dockers", "Wrangler"]}
CAROL = {"id": "carol", "name": "Carol Diaz", "role": "CEO", "scope": None}


def make_deps(conn, settings, script=None, toolbox=None, default="ok") -> Deps:
    return Deps.for_tests(conn, settings, FakeLLM(script, default=default), toolbox or FakeToolBox())


def turn_state(question: str, user=ALICE, thread_id="t1", turn_id="turn-1", **extra) -> dict:
    state = {"user": user, "thread_id": thread_id, "messages": [HumanMessage(content=question, id="h1")], "summary": ""}
    state.update(new_turn_fields(turn_id, question, "2026-10-16T10:00:00+00:00", 0.0))
    state.update(extra)
    return state
