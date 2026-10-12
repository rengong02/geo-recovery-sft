"""Run limits: a turn that hits a budget stops with a *named* reason.

Upstream had only `max_rounds`, and hitting it returned the bare string
"(reached maximum tool-call rounds)". Two problems with that: a model can burn
arbitrary tokens without ever reaching the round cap, and the caller cannot tell
"stopped early" from "finished" without string-matching. An evaluation that
records a truncated run as a completed one produces numbers nobody should trust.

So each limit raises `LimitExceeded(kind, limit, observed)`, and the kind is
data rather than prose.
"""

import time

import pytest

from corecoder.agent import Agent
from corecoder.limits import (
    COST,
    DEADLINE,
    ROUNDS,
    TOKENS,
    TOOL_CALLS,
    LimitExceeded,
    Limits,
)
from corecoder.llm import LLMResponse, ToolCall


class LoopingLLM:
    """Always asks for another tool call, so a limit is the only way out."""

    def __init__(self, tokens_per_turn: int = 0, cost_per_turn: float = 0.0, delay: float = 0.0):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.model = "looping-fake"
        self._n = 0
        self._tokens = tokens_per_turn
        self._cost = cost_per_turn
        self._delay = delay

    def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None) -> LLMResponse:
        self._n += 1
        if self._delay:
            time.sleep(self._delay)
        self.total_prompt_tokens += self._tokens
        return LLMResponse(tool_calls=[ToolCall(id=f"c{self._n}", name="todo_write",
                                                arguments={"tasks": []})])

    @property
    def estimated_cost(self) -> float | None:
        return self._cost * self._n


def _agent(llm, **limits_kwargs):
    return Agent(llm=llm, tools=[], limits=Limits(**limits_kwargs))


# --- the headline change: it raises, and it says which limit -----------------


def test_round_cap_raises_instead_of_returning_a_string():
    agent = _agent(LoopingLLM(), max_rounds=3)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == ROUNDS
    assert exc.value.limit == 3
    assert exc.value.observed == 4  # the round that was refused, not the last one run


def test_tool_call_budget_trips_before_the_round_cap():
    """A model making many calls per round must not get a free pass."""
    agent = _agent(LoopingLLM(), max_rounds=50, max_tool_calls=2)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == TOOL_CALLS
    assert exc.value.limit == 2


def test_token_budget_counts_prompt_and_completion():
    agent = _agent(LoopingLLM(tokens_per_turn=100), max_rounds=50, max_tokens=250)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == TOKENS
    assert exc.value.observed > 250


def test_cost_budget_uses_the_delta_for_this_turn():
    llm = LoopingLLM(cost_per_turn=0.5)
    agent = _agent(llm, max_rounds=50, max_cost_usd=1.0)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == COST
    assert exc.value.observed > 1.0


def test_deadline_trips_on_wall_clock():
    agent = _agent(LoopingLLM(delay=0.05), max_rounds=10_000, deadline_s=0.15)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == DEADLINE
    assert exc.value.observed >= 0.15


# --- budgets are per turn, so one Agent can serve many -----------------------


class PreloadedLLM(LoopingLLM):
    """A looping model whose token counter starts already in the thousands.

    Stands in for a long-lived Agent whose earlier turns already spent tokens.
    """

    def __init__(self, already_spent: int, tokens_per_turn: int):
        super().__init__(tokens_per_turn=tokens_per_turn)
        self.total_prompt_tokens = already_spent


def test_the_budget_is_measured_per_turn_not_over_the_agents_lifetime():
    """An Agent that already spent 5000 tokens must still get a fresh 300.

    With a lifetime-wide budget this turn would trip on its very first call
    (5000 > 300) instead of after the ~3 calls it is allowed.
    """
    llm = PreloadedLLM(already_spent=5000, tokens_per_turn=100)
    agent = _agent(llm, max_rounds=50, max_tokens=300)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == TOKENS
    # measured from this turn's start: 100, 200, 300, then 400 trips it.
    assert exc.value.observed == 400


# --- unconfigured limits must not change anything ----------------------------


def test_no_extra_limits_means_the_turn_just_finishes():
    """Only `max_rounds` is on by default; everything else stays off."""

    class Once:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        model = "once"

        def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
            return LLMResponse(content="hello")

        @property
        def estimated_cost(self):
            return None

    assert Agent(llm=Once(), tools=[], limits=Limits()).chat("go") == "hello"


def test_a_cost_budget_is_ignored_when_no_price_is_known():
    """Scripted and unpriced models report None; that must not trip or explode."""
    agent = _agent(LoopingLLM(cost_per_turn=0.0), max_rounds=2, max_cost_usd=0.001)

    with pytest.raises(LimitExceeded) as exc:
        agent.chat("go")

    assert exc.value.kind == ROUNDS  # not COST -- there was no price to compare


# --- the exception reads well in the places that surface it ------------------


def test_the_message_names_the_limit_and_both_numbers():
    e = LimitExceeded(ROUNDS, 50, 51)
    assert "rounds" in str(e)
    assert "50" in str(e) and "51" in str(e)


def test_the_agent_can_read_cost_from_a_real_llm():
    """Regression for the P0 this shipped with.

    `LLM.estimated_cost` is a `@property`, but the agent originally called it
    with parentheses, so a real client raised `TypeError: 'float' object is not
    callable` the moment a budget was consulted. All 225 tests passed anyway,
    because the fakes had implemented it as a method -- the fakes disagreed with
    production, so the tests proved nothing about production.
    """
    from corecoder.llm import LLM

    llm = LLM(model="deepseek-chat", api_key="dummy")
    agent = Agent(llm=llm, tools=[], limits=Limits(max_cost_usd=1.0))

    agent._cost_delta(None)  # must not raise


def test_every_fake_agrees_with_the_production_llm_interface():
    """Guard the class of bug above: a fake that drifts hides real breakage.

    Reading the attribute on the *class* yields the property object itself, so
    a fake that defines it as a plain method is caught here rather than in
    production.
    """
    from corecoder.demo import ScriptedLLM
    from corecoder.llm import LLM

    assert isinstance(LLM.__dict__["estimated_cost"], property), (
        "LLM.estimated_cost changed shape -- update Agent._cost_delta and the fakes"
    )
    for fake in (LoopingLLM, PreloadedLLM, ScriptedLLM):
        assert isinstance(fake.estimated_cost, property), (
            f"{fake.__name__}.estimated_cost must be a property, like LLM's"
        )


def test_passing_both_limits_and_max_rounds_is_an_error():
    """Two spellings of the same thing must not silently disagree."""
    with pytest.raises(ValueError, match="not both"):
        Agent(llm=LoopingLLM(), tools=[], max_rounds=7, limits=Limits(max_rounds=9))


def test_the_default_round_cap_is_still_fifty():
    """`max_rounds` now defaults to None so "not passed" is distinguishable."""
    assert Agent(llm=LoopingLLM(), tools=[]).limits.max_rounds == 50


def test_explicitly_passing_the_default_value_still_conflicts():
    """`max_rounds=50, limits=...` must not be silently resolved in favour of limits.

    With `max_rounds: int = 50` the constructor could not tell "caller passed 50"
    from "caller passed nothing", so this spelled-out conflict went unnoticed.
    """
    with pytest.raises(ValueError, match="not both"):
        Agent(llm=LoopingLLM(), tools=[], max_rounds=50, limits=Limits(max_rounds=3))


def test_sub_agent_gets_its_own_round_budget():
    """Documented semantics: each `chat()` has its own budget, sub-agents too.

    A sub-agent therefore does NOT inherit the parent's token/cost/deadline
    allowance -- it is a separate turn with its own clock. That is deliberate
    (a sub-agent is its own unit of work) but it means the parent's token budget
    is not a ceiling on the whole tree; see the note in AgentTool.
    """
    from corecoder.tools.agent import AgentTool

    parent = Agent(llm=LoopingLLM(), tools=[], limits=Limits(max_rounds=2, max_tokens=10_000))
    tool = AgentTool()
    tool._parent_agent = parent

    out = tool.execute(task="anything")

    assert "stopped early" in out
    assert "rounds" in out  # the sub-agent's own 20-round cap, not the parent's 2


def test_history_stays_valid_when_a_budget_trips_mid_turn():
    """A limit raised after the model replied must not leave a dangling turn.

    The assistant message used to be appended *after* the budget checks, so an
    exception left `messages` holding a user turn with no reply and, worse, tool
    calls with no results. A caller that caught the error and reused the Agent
    would send that broken transcript as the next request.
    """
    agent = _agent(LoopingLLM(), max_rounds=50, max_tool_calls=2)

    with pytest.raises(LimitExceeded):
        agent.chat("go")

    answered = {
        m.get("tool_call_id") for m in agent.messages if m["role"] == "tool"
    }
    for m in agent.messages:
        if m["role"] != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            assert call["id"] in answered, (
                f"tool call {call['id']} has no reply; the transcript is invalid"
            )


def test_the_round_cap_leaves_a_transcript_that_can_be_reused():
    """After a budget stop the Agent must still be able to take another turn."""
    agent = _agent(LoopingLLM(), max_rounds=2)

    with pytest.raises(LimitExceeded):
        agent.chat("go")

    # swap in a model that just answers, and take a second turn on the same Agent
    class _Answer:
        total_prompt_tokens = 0
        total_completion_tokens = 0

        def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
            return LLMResponse(content="recovered")

        @property
        def estimated_cost(self):
            return None

    agent.llm = _Answer()
    assert agent.chat("again") == "recovered"


def test_sub_agent_reports_a_budget_stop_as_text():
    """A tool must never raise into the parent loop -- errors are return values."""
    from corecoder.tools.agent import AgentTool

    parent = Agent(llm=LoopingLLM(), tools=[], limits=Limits(max_rounds=2))
    tool = AgentTool()
    tool._parent_agent = parent

    out = tool.execute(task="do something long")

    assert "stopped early" in out
    assert "rounds" in out
