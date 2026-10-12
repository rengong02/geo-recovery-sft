"""A request must not be able to outlive the run that asked for it.

`Limits.deadline_s` was checked only *between* rounds, so one wedged stream
blocked everything behind it -- observed sitting from 00:18 to 01:14, taking
every later sample of an experiment with it. Two separate causes, both of them
the same mistake at different levels:

* the OpenAI client retried underneath our own retry loop, so the policies
  multiplied;
* the deadline was passed *down* as a duration, so each retry loop re-based it
  and every attempt started with a fresh allowance. A request bounded at 100ms
  ran for three seconds and issued nine requests.

So the deadline is an absolute monotonic time, and it is computed once per
`chat()`. These tests use a clock that only advances when something sleeps, so
they assert on time without waiting for it.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import httpx
import pytest
from openai import APITimeoutError, InternalServerError

from corecoder import llm as llm_module
from corecoder.agent import Agent
from corecoder.limits import DEADLINE, LimitExceeded, Limits
from corecoder.llm import LLM, LiteLLM, RequestBudgetExceeded


class _Clock:
    """A monotonic clock that advances only inside `sleep`.

    Patching the module's `time` is what makes these assertions exact: real
    timing would make them slow and flaky, and "it took about a second" is not
    a claim a test can make.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.start = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    @property
    def elapsed(self) -> float:
        return self.now - self.start


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))


def _server_error() -> InternalServerError:
    request = httpx.Request("POST", "https://example.invalid")
    return InternalServerError("boom", response=httpx.Response(500, request=request), body=None)


def _llm(**kwargs) -> LLM:
    llm = LLM(model="test-model", api_key="sk-test", **kwargs)
    llm.client = mock.Mock()
    return llm


class _DeadStream:
    """A stream that dies after the request was accepted but before any output."""

    def __iter__(self):
        raise _timeout_error()


# --- the two retry policies ---------------------------------------------------


def test_the_sdk_does_not_retry_on_its_own():
    """Two policies multiply, and neither one is then the budget."""
    with mock.patch.object(llm_module, "OpenAI") as openai_cls:
        LLM(model="test-model", api_key="sk-test")

    assert openai_cls.call_args.kwargs["max_retries"] == 0, (
        "the OpenAI client is retrying as well, so the two policies multiply"
    )
    assert openai_cls.call_args.kwargs["timeout"] == LLM.DEFAULT_TIMEOUT


def test_a_retry_that_does_not_fit_is_not_started():
    clock = _Clock()
    count = {"n": 0}

    def boom(**_kwargs):
        count["n"] += 1
        raise _timeout_error()

    llm = _llm()
    llm.client.chat.completions.create.side_effect = boom

    with mock.patch.object(llm_module, "time", clock), pytest.raises(RequestBudgetExceeded):
        llm._call_with_retry({"model": "m"}, deadline_at=clock.monotonic() + 0.05)

    assert count["n"] == 1
    assert clock.elapsed <= 0.05


def test_a_5xx_backoff_that_does_not_fit_is_not_taken():
    """The other backoff branch. It slept first and checked afterwards.

    Leaving this one unconditional kept the budget claim false for 5xx: a 100ms
    request still sat through a one-second sleep before noticing.
    """
    clock = _Clock()
    count = {"n": 0}

    def boom(**_kwargs):
        count["n"] += 1
        raise _server_error()

    llm = _llm()
    llm.client.chat.completions.create.side_effect = boom

    with mock.patch.object(llm_module, "time", clock), pytest.raises(RequestBudgetExceeded):
        llm._call_with_retry({"model": "m"}, deadline_at=clock.monotonic() + 0.05)

    assert count["n"] == 1, "a 5xx retry was started that could not finish in time"
    assert clock.elapsed <= 0.05


def test_an_exhausted_budget_raises_before_any_request():
    llm = _llm()

    with pytest.raises(RequestBudgetExceeded, match="used up"):
        llm._call_with_retry({"model": "m"}, deadline_at=0.0)

    llm.client.chat.completions.create.assert_not_called()


def test_the_deadline_not_the_attempt_count_is_the_bound():
    """With time to spare, retries still happen."""
    clock = _Clock()
    count = {"n": 0}

    def boom(**_kwargs):
        count["n"] += 1
        raise _timeout_error()

    llm = _llm()
    llm.client.chat.completions.create.side_effect = boom

    with mock.patch.object(llm_module, "time", clock), pytest.raises(APITimeoutError):
        llm._call_with_retry({"model": "m"}, deadline_at=clock.monotonic() + 30.0)

    assert count["n"] == 3


# --- the outer streaming-retry loop -------------------------------------------


@pytest.mark.parametrize("budget", [0.1, 1.5, 2.5, 5.0, 30.0])
def test_the_streaming_retry_loop_never_outlives_its_budget(budget):
    """The property that actually matters: total time never exceeds the grant.

    Asserting the shape instead ("one deadline is passed to every attempt")
    would have been a test with no teeth: the outer loop's own backoff guard
    bounds this today, so re-basing the request deadline alone changes nothing.
    What was wrong was the *combination* -- unconditional `sleep(1) + sleep(2)`
    plus a per-attempt deadline -- and at a 0.1s grant that ran for 3.0s and
    issued three requests.

    So this asserts the outcome across several grants, which is what a report
    would quote, and which fails against the original loop.
    """
    clock = _Clock()
    llm = _llm(timeout=budget)
    requests = {"n": 0}

    def create(**_kwargs):
        requests["n"] += 1
        return _DeadStream()

    llm.client.chat.completions.create.side_effect = create

    with mock.patch.object(llm_module, "time", clock), pytest.raises((RequestBudgetExceeded, APITimeoutError)):
        llm.chat([{"role": "user", "content": "x"}])

    assert clock.elapsed <= budget, (
        f"a {budget}s grant ran for {clock.elapsed:.2f}s and issued "
        f"{requests['n']} requests"
    )


def test_a_streaming_retry_happens_when_the_budget_allows_it():
    """The same loop still retries on a stream death with time to spare."""
    clock = _Clock()
    llm = _llm(timeout=30.0)
    requests = {"n": 0}

    def create(**_kwargs):
        requests["n"] += 1
        return _DeadStream()

    llm.client.chat.completions.create.side_effect = create

    with mock.patch.object(llm_module, "time", clock), pytest.raises(APITimeoutError):
        llm.chat([{"role": "user", "content": "x"}])

    assert requests["n"] == 3


# --- the LiteLLM path ---------------------------------------------------------


def _fake_litellm(monkeypatch, error: Exception) -> dict:
    """Install a stand-in `litellm` module and count the calls it receives."""
    calls = {"n": 0}
    module = types.ModuleType("litellm")

    def completion(**_kwargs):
        calls["n"] += 1
        raise error

    module.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", module)
    return calls


def test_the_litellm_path_does_not_back_off_past_the_deadline(monkeypatch):
    """Same rule on the other backend, which has its own retry loop."""
    clock = _Clock()
    calls = _fake_litellm(monkeypatch, RuntimeError("503 server error"))
    llm = LiteLLM(model="ollama/llama3")

    with mock.patch.object(llm_module, "time", clock), pytest.raises(RequestBudgetExceeded):
        llm._call_with_retry({"model": "ollama/llama3"},
                             deadline_at=clock.monotonic() + 0.05)

    assert calls["n"] == 1
    assert clock.elapsed <= 0.05


def test_the_litellm_path_reports_an_exhausted_budget_before_importing(monkeypatch):
    """No `litellm` is installed here, so reaching the import would be visible."""
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    llm = LiteLLM(model="ollama/llama3")

    with pytest.raises(RequestBudgetExceeded):
        llm._call_with_retry({"model": "ollama/llama3"}, deadline_at=0.0)


# --- how the caller's timeout combines with the client's ----------------------


@pytest.mark.parametrize(("client_timeout", "caller_timeout", "expected"), [
    (100.0, 10.0, 10.0),    # tightened
    (100.0, 500.0, 100.0),  # never loosened
    (42.0, None, 42.0),     # caller said nothing
    (None, 5.0, 5.0),       # client unbounded
])
def test_the_caller_timeout_tightens_but_never_loosens(client_timeout, caller_timeout, expected):
    llm = _llm(timeout=client_timeout)
    seen = {}

    def capture(params, max_retries=3, deadline_at=None):
        seen["deadline_at"] = deadline_at
        raise RequestBudgetExceeded("stop here")

    llm._call_with_retry = capture
    with mock.patch.object(llm_module, "time", _Clock()), pytest.raises(RequestBudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}], timeout=caller_timeout)

    if expected is None:
        assert seen["deadline_at"] is None
    else:
        assert seen["deadline_at"] == pytest.approx(1000.0 + expected)


def test_a_timeout_is_configurable():
    assert _llm().timeout == LLM.DEFAULT_TIMEOUT
    assert _llm(timeout=7.0).timeout == 7.0


# --- what the agent does with it ----------------------------------------------


class _ExhaustedLLM:
    """A backend that always runs out of its request budget."""

    total_prompt_tokens = 0
    total_completion_tokens = 0
    estimated_cost = None

    def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
        raise RequestBudgetExceeded("out of time")


def test_an_expired_request_budget_is_a_deadline_stop_not_a_crash():
    """`truncated` and `crashed` call for different responses.

    A slow task is a normal outcome of a bounded run; a crash is not. Reporting
    one as the other hides which happened.
    """
    agent = Agent(llm=_ExhaustedLLM(), tools=[], limits=Limits(max_rounds=3, deadline_s=10))

    with pytest.raises(LimitExceeded) as caught:
        agent.chat("hi")

    assert caught.value.kind == DEADLINE


def test_the_same_error_without_a_task_deadline_is_not_dressed_up_as_one():
    """The budget then came from the client's own timeout, not a deadline."""
    agent = Agent(llm=_ExhaustedLLM(), tools=[], limits=Limits(max_rounds=3))

    with pytest.raises(RequestBudgetExceeded):
        agent.chat("hi")


def test_the_agent_passes_its_remaining_deadline_down():
    """Without this the deadline is only checked between rounds."""
    seen = {}

    class _RecordingLLM(_ExhaustedLLM):
        def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
            seen["timeout"] = timeout
            raise RequestBudgetExceeded("out of time")

    agent = Agent(llm=_RecordingLLM(), tools=[], limits=Limits(max_rounds=3, deadline_s=10))
    with pytest.raises(LimitExceeded):
        agent.chat("hi")

    assert seen["timeout"] is not None and 0 < seen["timeout"] <= 10


# --- the subclass trap --------------------------------------------------------
#
# `LiteLLM` writes its own `__init__` and never calls the base one, so any
# attribute set only inside `LLM.__init__` is missing on it. That has caused two
# separate breakages, both caught late by unrelated tests: `saw_reasoning` raised
# AttributeError from `_drain`, and `timeout` from `chat`.

SHARED_ATTRIBUTES = ("saw_reasoning", "timeout")


@pytest.mark.parametrize("cls", [LLM, LiteLLM])
def test_every_backend_can_answer_the_shared_attributes(cls):
    """Constructed without `__init__`, which is the state a skipped one leaves."""
    probe = cls.__new__(cls)

    missing = [name for name in SHARED_ATTRIBUTES if not hasattr(probe, name)]
    assert not missing, (
        f"{cls.__name__} is missing {missing}. Declare them at class level on LLM, "
        "not only in its __init__. Add new shared attributes to SHARED_ATTRIBUTES."
    )


@pytest.mark.parametrize("cls", [LLM, LiteLLM])
def test_every_backend_accepts_the_same_retry_contract(cls):
    """A caller that passes `deadline_at` to one must work against the other."""
    import inspect

    params = inspect.signature(cls._call_with_retry).parameters
    assert {"params", "max_retries", "deadline_at"} <= set(params), (
        f"{cls.__name__}._call_with_retry does not accept the shared contract"
    )


# --- the stream itself --------------------------------------------------------
#
# The deadline guarded the request and the backoffs but not the chunk loop, so a
# provider that kept trickling chunks crossed it and still handed back a
# perfectly normal-looking reply. A silently truncated answer is worse than a
# refusal: nothing in the result says it is incomplete.


def _chunk(text: str):
    from types import SimpleNamespace

    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=None))])


class _Stream:
    """A stream that advances a clock between chunks, like a trickling provider."""

    def __init__(self, clock: _Clock, plan):
        self.clock = clock
        self.plan = list(plan)
        self.closed = False

    def __iter__(self):
        for advance, chunk in self.plan:
            if advance:
                self.clock.now += advance
            yield chunk

    def close(self):
        self.closed = True


def test_a_stream_that_outlives_its_deadline_is_abandoned_not_returned():
    """The repro: first chunk in time, second after the deadline.

    Without the check this returned the full text over a 2s run on a 1s grant.
    """
    clock = _Clock()
    stream = _Stream(clock, [(0.0, _chunk("first")), (2.0, _chunk("late"))])
    llm = _llm(timeout=1.0)
    llm.client.chat.completions.create.return_value = stream

    with mock.patch.object(llm_module, "time", clock), pytest.raises(RequestBudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}])

    assert stream.closed, "the abandoned stream was left open"
    # Not `clock.elapsed <= 1.0`: the fake advances the clock *before* yielding,
    # which is what a real gap looks like, and the check can only run once a
    # chunk arrives. The overrun is the gap. Asserting the deadline on the nose
    # here would be asserting something the mechanism cannot deliver.
    assert clock.elapsed == 2.0, "the overrun should be exactly the inter-chunk gap"


def test_a_stream_that_finishes_inside_the_deadline_is_returned_normally():
    """The check must not fire on a stream that behaves."""
    clock = _Clock()
    stream = _Stream(clock, [(0.0, _chunk("all")), (0.1, _chunk(" good"))])
    llm = _llm(timeout=30.0)
    llm.client.chat.completions.create.return_value = stream

    with mock.patch.object(llm_module, "time", clock):
        response = llm.chat([{"role": "user", "content": "x"}])

    assert response.content == "all good"
    assert not stream.closed


def test_the_litellm_backend_bounds_the_stream_too(monkeypatch):
    """`LiteLLM` inherits `chat` but writes its own `_call_with_retry`.

    Signatures matching is not enough -- the two backends have to behave the
    same, and this is the path where they differ.
    """
    clock = _Clock()
    stream = _Stream(clock, [(0.0, _chunk("first")), (2.0, _chunk("late"))])
    module = types.ModuleType("litellm")
    module.completion = lambda **_kwargs: stream
    monkeypatch.setitem(sys.modules, "litellm", module)

    llm = LiteLLM(model="ollama/llama3", timeout=1.0)
    with mock.patch.object(llm_module, "time", clock), pytest.raises(RequestBudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}])

    assert stream.closed


def test_each_attempt_gets_the_budget_remaining_not_a_fresh_one():
    """This is what bounds a stream that stops producing chunks entirely.

    The chunk check cannot help there -- the loop is parked inside `next()` --
    so the only thing that ends the wait is the request timeout, and it has to
    be the budget *remaining*, not a fresh full one. The three numbers below
    are 25, then 25 minus the 10s the first attempt consumed and the 1s backoff,
    then what is left after the second attempt and its 2s backoff.
    """
    clock = _Clock()
    llm = _llm()
    timeouts: list[float | None] = []

    def create(**kwargs):
        timeouts.append(kwargs.get("timeout"))
        clock.now += 10.0  # a slow attempt; a real client would abort at its own timeout
        raise _timeout_error()

    llm.client.chat.completions.create.side_effect = create

    with mock.patch.object(llm_module, "time", clock), pytest.raises(APITimeoutError):
        llm._call_with_retry({"model": "m"}, deadline_at=clock.monotonic() + 25.0)

    assert timeouts == [pytest.approx(25.0), pytest.approx(14.0), pytest.approx(2.0)], (
        "an attempt was handed a fresh allowance instead of what was left"
    )
