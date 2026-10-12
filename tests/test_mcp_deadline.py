"""A tool call must not outlive the run that asked for it either.

The LLM path was bounded first; the tool path kept its own fixed allowance, so
`Limits(deadline_s=5)` could still sit inside an MCP call for the client's full
sixty seconds and only then hand control back. A task's deadline that only the
model respects is not a deadline.

The fixture server answers the handshake and then never answers a call, which is
the case a mock cannot cover: whether a real wait is actually interruptible.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.mcp import MCPClient, MCPError
from corecoder.tools.base import Tool

SERVER = Path(__file__).parent / "fake_hanging_mcp_server.py"

#: The client's own allowance is CALL_TIMEOUT (60s). Anything near that here
#: would mean the deadline was ignored; the assertions use far smaller numbers.
FIXED_ALLOWANCE = 60.0


@pytest.fixture
def client():
    instance = MCPClient("fake", sys.executable, [str(SERVER)])
    try:
        yield instance
    finally:
        instance.close()


# --- against a real, silent server -------------------------------------------


def test_a_remote_call_is_bounded_by_the_run_deadline(client):
    tool = client.tools[0]
    started = time.monotonic()

    with pytest.raises(MCPError, match="no answer"):
        tool.execute_with_deadline(time.monotonic() + 0.3)

    elapsed = time.monotonic() - started
    assert elapsed < FIXED_ALLOWANCE / 4, (
        f"waited {elapsed:.1f}s against a {FIXED_ALLOWANCE:g}s fixed allowance: "
        "the call is not reading the run's deadline"
    )
    assert elapsed >= 0.25, "it returned before the deadline it was given"


def test_a_call_that_starts_after_the_deadline_does_not_start(client):
    """Identifiable as a deadline stop, not as a slow or broken server."""
    tool = client.tools[0]

    with pytest.raises(MCPError, match="deadline had already passed"):
        tool.execute_with_deadline(time.monotonic() - 1.0)


def test_without_a_deadline_the_client_allowance_is_what_applies(client):
    """The default is unchanged: this is an addition, not a replacement.

    Not exercised by calling it -- that would wait the full allowance. The
    arithmetic is what matters and it is visible in the timeout the request
    actually got.
    """
    assert client.call_timeout == FIXED_ALLOWANCE

    seen = {}
    client._request = lambda method, params, timeout: seen.update(timeout=timeout) or {}
    client.call_tool("hang", {})

    assert seen["timeout"] == FIXED_ALLOWANCE


def test_a_deadline_shorter_than_the_allowance_wins(client):
    seen = {}
    client._request = lambda method, params, timeout: seen.update(timeout=timeout) or {}
    client.call_tool("hang", {}, deadline_at=time.monotonic() + 2.0)

    assert seen["timeout"] < FIXED_ALLOWANCE
    assert seen["timeout"] <= 2.0


# --- through the agent --------------------------------------------------------


class _RecordsDeadline(Tool):
    name, description = "probe", "records the deadline it was handed"
    from typing import ClassVar

    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def __init__(self):
        self.seen: list[float | None] = []

    def execute(self, **kwargs) -> str:
        raise AssertionError("the agent should call execute_with_deadline")

    def execute_with_deadline(self, deadline_at, **kwargs) -> str:
        self.seen.append(deadline_at)
        return "ok"


def _agent_with(tool: Tool, calls: int, **limits) -> Agent:
    from corecoder.limits import Limits

    script = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name=tool.name, arguments={})])
              for i in range(calls)]
    script.append(LLMResponse(content="done"))
    return Agent(llm=ScriptedLLM(script), tools=[tool],
                 limits=Limits(max_rounds=10, **limits))


def test_the_agent_hands_every_tool_call_the_turn_deadline():
    tool = _RecordsDeadline()
    before = time.monotonic()
    _agent_with(tool, calls=2, deadline_s=30).chat("go")

    assert len(tool.seen) == 2
    assert all(d is not None for d in tool.seen)
    assert all(before < d <= before + 30.5 for d in tool.seen)


def test_a_turn_without_a_deadline_hands_the_tool_none():
    tool = _RecordsDeadline()
    _agent_with(tool, calls=1).chat("go")

    assert tool.seen == [None]


def test_the_parallel_path_gets_the_deadline_too():
    """The single-call path is the easy one; batches go through the pool."""
    tool = _RecordsDeadline()
    from corecoder.limits import Limits

    script = [
        LLMResponse(tool_calls=[ToolCall(id=c, name="probe", arguments={}) for c in ("a", "b", "c")]),
        LLMResponse(content="done"),
    ]
    agent = Agent(llm=ScriptedLLM(script), tools=[tool],
                  limits=Limits(max_rounds=10, deadline_s=30))
    agent.chat("go")

    assert len(tool.seen) == 3, "the batch did not all reach the tool"
    assert all(d is not None for d in tool.seen), (
        "a pooled call lost the deadline -- the pool is where it is easiest to drop"
    )
