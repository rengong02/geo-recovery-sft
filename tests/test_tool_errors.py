"""A failed call has to look like a failure to everyone who asks.

The recorder and the repeat counter read the same return value. If they disagree
about which strings are failures, the counts in a report stop meaning anything --
and the retry numbers P1 is judged on are exactly those counts.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.tool_errors import (
    REPEAT_NUDGE_AT,
    call_key,
    is_error_result,
    with_repeat_marker,
)
from corecoder.tools.base import Tool

_SCHEMA: dict = {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}


class _Fails(Tool):
    name, description = "flaky", "always reports failure"
    parameters: ClassVar[dict] = _SCHEMA

    def execute(self, x: str = "") -> str:
        return f"Error: cannot use {x!r}"


class _Works(Tool):
    name, description = "fine", "always succeeds"
    parameters: ClassVar[dict] = _SCHEMA

    def execute(self, x: str = "") -> str:
        return f"used {x!r}"


# --- the predicate ------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Error: unknown tool 'nope'",
    "Error executing bash: boom",
    "Error running command: no such file",
])
def test_every_failure_shape_is_recognised(text):
    """Three shapes, three layers: adapter, argument binding, unknown tool.

    `_exec_tool`'s own shape has no colon after "Error", which is why this
    predicate lives in one place instead of being re-spelled by each caller.
    """
    assert is_error_result(text)


@pytest.mark.parametrize("text", [
    "Loaded data and created DataFrame named gdp",
    "",
    "Errors: none",
    "  \n Error: indented is still a failure",
])
def test_ordinary_results_are_not_mistaken_for_failures(text):
    """`Errors: none` is the trap: it starts with the same six letters."""
    assert is_error_result(text) == text.strip().startswith("Error:")


def test_call_key_ignores_argument_order():
    """Re-sending the same call with reordered keys is the same call."""
    assert call_key("t", {"a": 1, "b": 2}) == call_key("t", {"b": 2, "a": 1})
    assert call_key("t", {"a": 1}) != call_key("t", {"a": 2})
    assert call_key("t", {"a": 1}) != call_key("u", {"a": 1})


def test_the_marker_waits_for_a_second_failure():
    """The first failure is new information; the second is not."""
    plain = "Error: Column 'X' not found"
    assert with_repeat_marker(plain, 1) == plain
    assert with_repeat_marker(plain, REPEAT_NUDGE_AT) != plain
    assert "#2" in with_repeat_marker(plain, 2)


# --- the counter, through a real chat loop ------------------------------------


def _tool_replies(agent: Agent) -> list[str]:
    return [m["content"] for m in agent.messages if m.get("role") == "tool"]


def _chat(tool: Tool, arguments: dict, times: int) -> Agent:
    script = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name=tool.name, arguments=arguments)])
              for i in range(times)]
    script.append(LLMResponse(content="done"))
    agent = Agent(llm=ScriptedLLM(script), tools=[tool])
    agent.chat("go")
    return agent


def test_identical_failures_are_counted_once_under_one_key():
    agent = _chat(_Fails(), {"x": "a"}, 3)

    assert list(agent.failure_repeats.values()) == [3], (
        "three identical failures are one repeated call, not three calls"
    )


def test_the_third_reply_tells_the_model_it_is_repeating_itself():
    replies = _tool_replies(_chat(_Fails(), {"x": "a"}, 3))

    assert "Identical call" not in replies[0], "the first failure is still news"
    assert "Identical call" in replies[1]
    assert "#3" in replies[2]


def test_different_arguments_are_counted_separately():
    """Guessing column names is one call per guess, not one repeated call."""
    tool = _Fails()
    script = [
        LLMResponse(tool_calls=[ToolCall(id="c1", name="flaky", arguments={"x": "NAME"})]),
        LLMResponse(tool_calls=[ToolCall(id="c2", name="flaky", arguments={"x": "ADMIN"})]),
        LLMResponse(content="done"),
    ]
    agent = Agent(llm=ScriptedLLM(script), tools=[tool])
    agent.chat("go")

    assert sorted(agent.failure_repeats.values()) == [1, 1]


def test_a_successful_call_is_not_counted():
    agent = _chat(_Works(), {"x": "a"}, 2)

    assert agent.failure_repeats == {}


def test_the_counter_never_retries_on_its_own():
    """The model makes the call; the harness only comments on the result.

    A framework that silently re-sends would make "the model repeated itself"
    and "the agent retried it" the same event in the record.
    """
    tool = _Fails()
    script = [
        LLMResponse(tool_calls=[ToolCall(id="c1", name="flaky", arguments={"x": "a"})]),
        LLMResponse(content="stopping"),
    ]
    agent = Agent(llm=ScriptedLLM(script), tools=[tool])
    result = agent.chat("go")

    assert result == "stopping"
    assert len(_tool_replies(agent)) == 1, "the harness re-sent a call the model made once"
