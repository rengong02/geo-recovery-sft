"""Run a planned tool call for real, and keep what came back.

The plan says what to call; only execution says what happens. Every observation
in a sample comes from here, because a written-out observation is a fabricated
one -- and a fabricated observation is worse than a missing one, since it looks
like data.

Two renderings are kept for every step, and the reason is a conflict the design
had to resolve:

* `raw` is the tool's complete return, verbatim. Validation compares against
  this, because it is the thing the tool actually produced.
* `visible` is what the model sees: the same text under a **fixed, versioned**
  truncation rule. Training uses this.

Storing only one of them forces a choice between comparing truncated strings
(which hides real differences) and training on full ones (which the context
budget cannot pay for). Keeping both, with the renderer version recorded,
means training and validation can disagree about length without disagreeing
about content.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from corecoder.tool_errors import is_error_result

#: Bump when the truncation rule changes. Samples carry the version they were
#: built with, so a renderer change is visible as a data change rather than a
#: silent shift in what the model was trained to read.
RENDERER_VERSION = "r2"

#: Characters of a tool result the model sees. A frame description can run to
#: several thousand; the head carries the column list, which is what F1/F2 need.
VISIBLE_HEAD = 1200

TRUNCATION_NOTE = "…[{dropped} chars truncated, renderer {version}]"


def render_column_names(columns) -> str:
    """The loader's column list, as a JSON array.

    The v1 renderer joined with `", "`, and a column actually called
    `Population, total` became two fragments that no parser can split back --
    measured on the shipped data, 24 of 192 loader observations could not be
    round-tripped. A separator that can appear inside the values is not a
    separator.

    JSON is used rather than another delimiter because quoting and escaping are
    already specified: a name with a comma, a quote, a newline or a non-ASCII
    character survives unchanged, and `json.loads` recovers the exact list.
    """
    import json as _json

    return _json.dumps([str(column) for column in columns], ensure_ascii=False)


def render_visible(raw: str, head: int = VISIBLE_HEAD) -> str:
    if len(raw) <= head:
        return raw
    dropped = len(raw) - head
    return raw[:head] + TRUNCATION_NOTE.format(dropped=dropped, version=RENDERER_VERSION)


@dataclass
class Step:
    """One call and its outcome, in both renderings."""

    call_id: str
    tool: str
    arguments: dict
    raw: str
    visible: str
    failed: bool
    seconds: float

    def as_messages(self) -> list[dict]:
        return [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": self.call_id,
                "type": "function",
                "function": {"name": self.tool, "arguments": self.arguments},
            }]},
            {"role": "tool", "tool_call_id": self.call_id, "content": self.visible},
        ]

    def as_record(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "observation_raw": self.raw,
            "observation_visible": self.visible,
            "failed": self.failed,
            "seconds": round(self.seconds, 4),
        }


@dataclass
class Execution:
    steps: list[Step] = field(default_factory=list)

    @property
    def failed_steps(self) -> list[Step]:
        return [s for s in self.steps if s.failed]

    def as_records(self) -> list[dict]:
        return [s.as_record() for s in self.steps]


def execute_plan(
    calls: list[tuple[str, dict]],
    tools: dict[str, Any],
    on_step: Callable[[Step], None] | None = None,
) -> Execution:
    """Run `(tool_name, arguments)` pairs in order against a live tool set.

    Order matters and is not an optimisation: these tools share one state dict,
    so a step that reads a frame genuinely depends on the step that built it.

    An error return is a **step, not an exception**. A call that failed is
    exactly what a recovery trace is made of, and dropping it would delete the
    only evidence that the model can see a failure and change course.
    """
    import time

    execution = Execution()
    for index, (name, arguments) in enumerate(calls, 1):
        call_id = f"call_{index}"
        tool = tools.get(name)
        started = time.perf_counter()
        if tool is None:
            raw, failed = f"Error: tool {name!r} is not in this profile", True
        else:
            try:
                raw = str(tool.execute(**arguments))
                failed = is_error_result(raw)
            except Exception as exc:  # noqa: BLE001 - recorded, never raised
                raw, failed = f"Error: {type(exc).__name__}: {exc}", True
        seconds = time.perf_counter() - started

        step = Step(call_id=call_id, tool=name, arguments=arguments, raw=raw,
                    visible=render_visible(raw), failed=failed, seconds=seconds)
        execution.steps.append(step)
        if on_step is not None:
            on_step(step)
    return execution
