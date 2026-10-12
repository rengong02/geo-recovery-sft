"""Run limits: why a turn stopped, as data rather than a shrug.

`max_rounds` alone is a blunt instrument. A model that loops without calling
tools burns tokens without ever reaching the cap; an MCP server that wedges
hangs the turn forever; a cheap model that makes 40 cheap calls and an
expensive one that makes 3 look identical. And when a run does stop, the
upstream behaviour is to return the string "(reached maximum tool-call rounds)",
which loses the one fact an evaluation needs: *which* limit tripped.

So each limit is a named kind with the value that tripped it, and tripping one
raises `LimitExceeded` rather than returning text. Raising is deliberate --
a caller that forgets to check a return value silently records a truncated run
as a finished one, which is exactly the failure mode that makes benchmark
numbers untrustworthy.
"""

import time
from dataclasses import dataclass

#: Kinds, in the order a turn is likely to hit them.
ROUNDS = "rounds"
TOOL_CALLS = "tool_calls"
TOKENS = "tokens"
COST = "cost"
DEADLINE = "deadline"


class LimitExceeded(RuntimeError):
    """A configured run limit was reached; the turn did not finish."""

    def __init__(self, kind: str, limit: float, observed: float):
        self.kind = kind
        self.limit = limit
        self.observed = observed
        super().__init__(f"{kind} limit exceeded: {observed:g} > {limit:g}")

    def __str__(self) -> str:
        return super().__str__()


@dataclass
class Limits:
    """What a single `Agent.chat` turn is allowed to spend.

    Every field except `max_rounds` is off by default, so an interactive session
    behaves exactly as before until someone configures a budget.
    """

    max_rounds: int = 50
    #: Total tool calls across the turn.
    max_tool_calls: int | None = None
    #: Prompt + completion tokens, measured from the start of the turn.
    max_tokens: int | None = None
    #: Estimated USD, measured from the start of the turn. Ignored when the
    #: model is missing from the pricing table and no estimate is possible.
    max_cost_usd: float | None = None
    #: Wall-clock seconds for the whole turn.
    deadline_s: float | None = None

    def check_rounds(self, rounds: int) -> None:
        if rounds > self.max_rounds:
            raise LimitExceeded(ROUNDS, self.max_rounds, rounds)

    def check_tool_calls(self, calls: int) -> None:
        if self.max_tool_calls is not None and calls > self.max_tool_calls:
            raise LimitExceeded(TOOL_CALLS, self.max_tool_calls, calls)

    def check_tokens(self, tokens: int) -> None:
        if self.max_tokens is not None and tokens > self.max_tokens:
            raise LimitExceeded(TOKENS, self.max_tokens, tokens)

    def check_cost(self, cost_usd: float | None) -> None:
        if self.max_cost_usd is None or cost_usd is None:
            return
        if cost_usd > self.max_cost_usd:
            raise LimitExceeded(COST, self.max_cost_usd, cost_usd)

    def check_deadline(self, started_at: float) -> None:
        if self.deadline_s is None:
            return
        elapsed = time.monotonic() - started_at
        if elapsed > self.deadline_s:
            raise LimitExceeded(DEADLINE, self.deadline_s, elapsed)

    def remaining_seconds(self, started_at: float) -> float | None:
        """Seconds left before the deadline, or None when unbounded.

        **The LLM path consumes this.** `Agent.chat` passes it to every request,
        where it becomes the budget for that request and all of its retries and
        backoffs (see `LLM.chat`). That is what makes the deadline more than a
        between-rounds check.

        **The tool path consumes it only where the tool opts in.** `Agent` passes
        the same absolute deadline through the internal
        `execute_with_deadline()` entry point; `MCPTool` uses it to tighten the
        client's fixed allowance. In-process tools that inherit the default
        implementation may still run until their own call returns, because
        Python cannot safely cancel arbitrary work in place. The deadline is
        therefore hard for LLM calls and deadline-aware remote tools, but not a
        blanket cancellation guarantee for every `Tool` implementation.
        """
        if self.deadline_s is None:
            return None
        return max(0.0, self.deadline_s - (time.monotonic() - started_at))
