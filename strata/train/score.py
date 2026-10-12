"""Grading an attempt against the semantic-role gold.

Not a boolean, because a boolean hides the distinction this project cares most
about. Two wrong answers look nothing alike in the data:

* the model names a column that **is not there** -- loud, and recoverable by
  reading the error, which is the p2c failure;
* the model names a column that **is there and means something else** -- silent:
  the frame keeps its shape, the call succeeds, no error is raised, and only the
  meaning is wrong.

The second is the one worth being able to count, and a pass/fail collapses it
into the first. So an attempt is classified into an outcome, and the detail
carries enough to explain it.

For F2 the same split applies with different evidence. A join on the wrong keys
does not fail either -- it returns the same 251x96 frame with an empty column --
so the outcome is decided by **how much of the value column actually matched**,
which is measured, not asserted.
"""

from __future__ import annotations

from collections.abc import Callable

from .reference import MIN_CORRECT_FILL_RATE, confusable_columns
from .termination import valid_v2_termination

SCORER_VERSION = "v2.1-status-schema"

OUTCOME_CORRECT = "correct"
OUTCOME_PICKED_NONEXISTENT = "picked_nonexistent_column"
OUTCOME_PICKED_CONFUSABLE = "picked_confusable_column"
OUTCOME_WRONG_KEY_PAIR = "wrong_key_pair"
OUTCOME_NO_TERMINATION = "no_termination"
OUTCOME_MALFORMED = "malformed_termination"
OUTCOME_REPORTED_ERROR = "reported_error"

#: Outcomes that mean the model produced an answer and it was wrong. Separated
#: from the format failures so an accuracy figure is not deflated by a model
#: that simply never learned the output shape -- those are two different
#: findings and one number cannot report both.
ANSWERED_WRONG = frozenset({
    OUTCOME_PICKED_NONEXISTENT, OUTCOME_PICKED_CONFUSABLE, OUTCOME_WRONG_KEY_PAIR,
})

FORMAT_FAILURES = frozenset({OUTCOME_NO_TERMINATION, OUTCOME_MALFORMED})


#: Distinguishes "the model never finished" from "the model finished in a shape
#: nothing can read". Both are format problems, and they are not the same
#: problem: the first is a truncated run, the second is prose where JSON was
#: asked for -- and pooling them would make one number stand for two findings.
_ABSENT = object()


def _termination_of(attempt):
    """`_ABSENT` when there is no termination, `None` when it is unusable, else
    the dict."""
    if not isinstance(attempt, dict):
        return _ABSENT
    if "termination" not in attempt or attempt.get("termination") is None:
        return _ABSENT
    termination = attempt.get("termination")
    return termination if isinstance(termination, dict) else None


def _score_f1(sample: dict, attempt: dict, frame_columns: list[str],
              **_) -> tuple[str, dict]:
    gold = sample.get("gold") or {}
    expected = str(gold.get("column", ""))
    termination = _termination_of(attempt)
    if termination is _ABSENT:
        return OUTCOME_NO_TERMINATION, {"expected": expected}
    if termination is None:
        return OUTCOME_MALFORMED, {"expected": expected, "termination": "unreadable"}

    selected = termination.get("selected_column")
    if not isinstance(selected, str) or not selected.strip():
        return OUTCOME_MALFORMED, {"expected": expected, "selected": selected}

    if selected == expected:
        return OUTCOME_CORRECT, {"selected": selected}

    detail = {"expected": expected, "selected": selected,
              "in_frame": selected in frame_columns}
    # Real, and still wrong. Only a near-miss of the gold earns this label; a
    # plainly unrelated column is a miss, not a confusion, and calling both the
    # same would make the count mean nothing.
    if selected in frame_columns and selected in confusable_columns(expected, frame_columns):
        return OUTCOME_PICKED_CONFUSABLE, detail
    return OUTCOME_PICKED_NONEXISTENT, detail


def _score_f2(sample: dict, attempt: dict,
              fill_rate: Callable[[tuple[str, str]], float] | None = None,
              **_) -> tuple[str, dict]:
    gold = sample.get("gold") or {}
    expected_pair = tuple(gold.get("join_keys") or [])
    termination = _termination_of(attempt)
    if termination is _ABSENT:
        return OUTCOME_NO_TERMINATION, {"expected": list(expected_pair)}
    if termination is None:
        return OUTCOME_MALFORMED, {"expected": list(expected_pair),
                                   "termination": "unreadable"}

    keys = termination.get("join_keys")
    if not isinstance(keys, list) or len(keys) != 2 or not all(
            isinstance(k, str) and k.strip() for k in keys):
        return OUTCOME_MALFORMED, {"expected": list(expected_pair), "selected": keys}

    if tuple(keys) == expected_pair:
        achieved = fill_rate(expected_pair) if fill_rate else None
        return OUTCOME_CORRECT, {"selected": keys, "fill_rate": achieved}

    achieved = fill_rate(tuple(keys)) if fill_rate else None
    expected_rate = fill_rate(expected_pair) if fill_rate else None
    return OUTCOME_WRONG_KEY_PAIR, {
        "expected": list(expected_pair), "selected": keys,
        "fill_rate": achieved, "expected_fill_rate": expected_rate,
        "expected_fill_floor": MIN_CORRECT_FILL_RATE,
    }


def score_attempt(sample: dict, attempt, frame_columns: list[str] | None = None,
                  fill_rate: Callable[[tuple[str, str]], float] | None = None,
                  ) -> tuple[str, dict]:
    """Classify one attempt. Never raises: a scorer that dies on a bad attempt
    cannot be used to score a batch of them."""
    try:
        family = str(sample.get("family") or "").upper()
        if sample.get("termination_schema_version", sample.get("schema_version", 1)) == 2:
            termination = _termination_of(attempt)
            if termination is _ABSENT:
                return OUTCOME_NO_TERMINATION, {}
            if not valid_v2_termination(termination, family):
                return OUTCOME_MALFORMED, {"termination": termination,
                                          "reason": "invalid_v2_termination_contract"}
            if termination["status"] == "error":
                return OUTCOME_REPORTED_ERROR, {
                    "error_code": termination["error_code"],
                    "error_message": termination["error_message"],
                }
        if family == "F2":
            return _score_f2(sample, attempt or {}, fill_rate=fill_rate)
        return _score_f1(sample, attempt or {}, frame_columns=frame_columns or [])
    except Exception as exc:  # noqa: BLE001 - a bad attempt is data, not a crash
        return OUTCOME_MALFORMED, {"error": f"{type(exc).__name__}: {exc}"}


def summarise(outcomes: list[str]) -> dict:
    """Accuracy over answered attempts, with format failures reported apart.

    Pooling them would let a model that never emits a termination look like a
    model that answers wrongly, and the two call for different work.
    """
    total = len(outcomes)
    correct = sum(1 for o in outcomes if o == OUTCOME_CORRECT)
    format_failures = sum(1 for o in outcomes if o in FORMAT_FAILURES)
    reported_errors = outcomes.count(OUTCOME_REPORTED_ERROR)
    answered = total - format_failures - reported_errors
    by_outcome: dict[str, int] = {}
    for outcome in outcomes:
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "answered": answered,
        "accuracy_of_answered": round(correct / answered, 4) if answered else None,
        "format_failures": format_failures,
        "reported_errors": reported_errors,
        "by_outcome": by_outcome,
    }
