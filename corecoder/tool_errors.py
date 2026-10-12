"""Telling a tool failure from a tool result, and noticing when one repeats.

CoreCoder reports a failed tool call as text rather than raising, so one broken
tool cannot kill the loop. That makes "did this fail?" a string question -- and
the answer has to be the same everywhere it is asked. A recorder and a retry
counter that disagree about what counts as an error turn the counts in a report
into noise, which is the one thing this project cannot afford.

The counting here is deliberately *not* a framework retry. A harness that
quietly re-sends a call makes "the model repeated itself" and "the agent retried
it" the same event in the record, and they call for opposite fixes: one is a
model-behaviour finding, the other a harness bug.
"""

from __future__ import annotations

import json

#: How a failed call comes back. Three shapes, because the failure is caught at
#: three layers: inside the tool adapter, by the agent's argument binding, and
#: by the agent's unknown-tool check. `_exec_tool`'s third shape ("Error
#: executing {name}: ...") has no colon after `Error`, which is exactly why the
#: predicate lives in one place instead of being spelled out per caller.
_ERROR_PREFIXES = ("Error:", "Error executing ", "Error running ")


def is_error_result(text: str) -> bool:
    """Whether a tool's return value reports a failure rather than a result."""
    return text.lstrip().startswith(_ERROR_PREFIXES)


def call_key(name: str, arguments: dict) -> str:
    """A stable identity for "the same call", for counting repeats.

    Sorted keys: the model re-sending a call with its keys in another order is
    the same call, and counting it twice would overstate the very behaviour
    being measured.
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(arguments)
    return f"{name}:{rendered}"


#: A second identical failure is where the reply stops being information and
#: starts being a nudge: the first one the model has not seen yet.
REPEAT_NUDGE_AT = 2

REPEAT_MARKER = (
    "\n[Identical call, failure #{n}. Repeating it unchanged will fail the same "
    "way. Change the arguments, or spend a call finding out what this one needs.]"
)


def with_repeat_marker(text: str, repeats: int) -> str:
    """Append the nudge once a call has repeated a deterministic failure."""
    if repeats < REPEAT_NUDGE_AT:
        return text
    return text + REPEAT_MARKER.format(n=repeats)
