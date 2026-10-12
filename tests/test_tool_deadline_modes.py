"""Tools state whether they can enforce the deadline, and are held to it.

Two kinds exist and the difference is not cosmetic:

* **enforced** -- the tool bounds its own wait (a subprocess timeout with a
  process-group kill, a network read timeout). The deadline holds.
* **cooperative** -- the tool can be *told* the deadline but cannot act on it.
  An in-process GIS function that blocks returns when it returns; nothing
  outside it can interrupt the frame it is in.

Claiming "enforced" without overriding the entry point is worse than claiming
nothing, so the first test here is the guard: the declaration has to be backed
by an override, not by intention.
"""

from __future__ import annotations

import time
from typing import ClassVar

import pytest

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.limits import LimitExceeded, Limits
from corecoder.llm import LLMResponse, ToolCall
from corecoder.mcp import MCPTool
from corecoder.tools.base import Tool
from corecoder.tools.bash import BashTool

_SCHEMA: dict = {
    "type": "object",
    "properties": {"seconds": {"type": "number"}},
    "required": [],
}


# --- the declarations, held to what they claim --------------------------------


@pytest.mark.parametrize("cls", [MCPTool, BashTool])
def test_a_tool_that_claims_enforcement_overrides_the_entry_point(cls):
    assert cls.deadline_mode == "enforced"
    assert cls.execute_with_deadline is not Tool.execute_with_deadline, (
        f"{cls.__name__} claims an enforceable deadline but inherits the default, "
        "which ignores it -- the claim would be false"
    )


def test_the_local_gis_tool_is_declared_cooperative():
    """Honest, not aspirational: an in-process function cannot be interrupted.

    It may still *read* the deadline -- and once it does, it stops being purely
    cooperative -- so the assertion is only that it does not claim enforcement
    it cannot deliver.
    """
    from strata.geobench.tools import GeoBenchTool

    assert GeoBenchTool.deadline_mode == "cooperative"


def test_nothing_claims_enforcement_while_ignoring_the_deadline():
    """The default entry point is the tell: it ignores the argument entirely."""
    import inspect

    source = inspect.getsource(Tool.execute_with_deadline)
    assert "return self.execute(**kwargs)" in source


# --- the overrun record -------------------------------------------------------


class _Slow(Tool):
    """A cooperative tool that blocks for a while -- like a GIS function would."""

    name, description = "slow", "blocks for the requested time"
    parameters: ClassVar[dict] = _SCHEMA

    def execute(self, seconds: float = 0.0) -> str:
        time.sleep(seconds)
        return "done"


def _agent(tool: Tool, deadline_s: float | None) -> Agent:
    script = [
        LLMResponse(tool_calls=[ToolCall(id="c1", name=tool.name, arguments={"seconds": 0.2})]),
        LLMResponse(content="done"),
    ]
    return Agent(llm=ScriptedLLM(script), tools=[tool],
                 limits=Limits(max_rounds=5, deadline_s=deadline_s))


def test_a_tool_that_returns_late_is_recorded_with_its_mode():
    """Cooperative tools cannot be stopped, so they are counted instead.

    A run whose tools overran is still a valid run. A report that does not say
    so is not.
    """
    agent = _agent(_Slow(), deadline_s=0.05)

    with pytest.raises(LimitExceeded):
        agent.chat("go")

    assert len(agent.deadline_overruns) == 1
    name, late, mode = agent.deadline_overruns[0]
    assert (name, mode) == ("slow", "cooperative")
    assert late > 0


def test_a_tool_that_returns_in_time_is_not_recorded():
    agent = _agent(_Slow(), deadline_s=30)

    agent.chat("go")

    assert agent.deadline_overruns == []


def test_nothing_is_recorded_without_a_deadline():
    """No deadline means no overrun to report -- not an overrun of zero."""
    agent = _agent(_Slow(), deadline_s=None)

    agent.chat("go")

    assert agent.deadline_overruns == []


def test_a_tool_rejected_after_deadline_is_not_recorded_as_a_late_call():
    """A skipped call is different from a tool that actually returned late."""
    class _SlowUnsafe(_Slow):
        concurrency_safe = False

    script = [
        LLMResponse(tool_calls=[
            ToolCall(id="slow", name="slow", arguments={"seconds": 0.2}),
            ToolCall(id="bash", name="bash", arguments={"command": "echo skipped"}),
        ]),
        LLMResponse(content="done"),
    ]
    agent = Agent(
        llm=ScriptedLLM(script),
        tools=[_SlowUnsafe(), BashTool()],
        limits=Limits(max_rounds=5, deadline_s=0.05),
    )

    with pytest.raises(LimitExceeded):
        agent.chat("go")

    assert [(name, mode) for name, _, mode in agent.deadline_overruns] == [
        ("slow", "cooperative")
    ]


# --- bash reads the deadline --------------------------------------------------


def test_bash_does_not_start_a_command_it_cannot_finish_in_time():
    out = BashTool().execute_with_deadline(time.monotonic() - 1.0, "echo hi")

    assert out.startswith("Error")
    assert "deadline had already passed" in out


def test_bash_caps_its_wait_by_the_remaining_time():
    """Its own default is 120s; the run's remaining time outranks it."""
    started = time.monotonic()
    out = BashTool().execute_with_deadline(time.monotonic() + 0.5, "sleep 5")
    elapsed = time.monotonic() - started

    assert "timed out" in out
    assert elapsed < 0.9, f"waited {elapsed:.1f}s; the run's remaining time was 0.5s"


def test_bash_still_runs_normally_within_the_deadline():
    out = BashTool().execute_with_deadline(time.monotonic() + 30, "echo hi")

    assert "hi" in out
    assert not out.startswith("Error")
