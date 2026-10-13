"""A recovery trace contains one deliberate contract violation -- and only one.

V1 requires every call to match its tool's argument contract. A recovery trace
breaks that on purpose: the wrong call *is* the error being taught, and deleting
it would delete the lesson. Measured before this allowance: the 32
`missing_argument` samples failed validation for being exactly what they are.

The allowance is bounded, and these tests are about the bound. Loosening a
validator until the data passes is how a gate stops meaning anything; what is
allowed here is one violation, declared in provenance, and the count is asserted
so a second one cannot hide behind the first.
"""

from __future__ import annotations

import copy
import json

from strata.train.sample import BLOCKER_BAD_ARGUMENTS, validate_sample
from strata.train.synth import build_synth_f2_recovery_batch
from strata.train.synth_families import f2_family_cases


def _missing_argument_sample() -> dict:
    """The kind that violates the contract: `geokey` is left out."""
    cases = [c for c in f2_family_cases("shared_root__full", 3)]
    samples, _ = build_synth_f2_recovery_batch(cases)
    for sample in samples:
        if sample["lineage"].get("recovery_error_kind") == "missing_argument":
            return sample
    raise AssertionError("no missing_argument sample was produced")


def test_one_violation_is_allowed_and_only_for_a_recovery_trace():
    sample = _missing_argument_sample()
    assert validate_sample(sample).ok, validate_sample(sample).blockers


def test_the_allowance_does_not_apply_without_the_provenance_marker():
    """Without the declaration it is an ordinary broken sample."""
    sample = copy.deepcopy(_missing_argument_sample())
    del sample["lineage"]["recovery_error_kind"]
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(sample).blockers


def test_a_second_violation_is_still_refused():
    """The bound, stated as a test rather than as a comment."""
    sample = copy.deepcopy(_missing_argument_sample())
    bad = {"id": "call_extra", "type": "function",
           "function": {"name": "merge_dataframes", "arguments": {}}}
    messages = sample["messages"]
    insert_at = next(i for i, m in enumerate(messages)
                     if m.get("role") == "assistant" and m.get("tool_calls"))
    messages.insert(insert_at + 1, {"role": "assistant", "content": None,
                                    "tool_calls": [bad]})
    messages.insert(insert_at + 2, {"role": "tool", "tool_call_id": "call_extra",
                                    "content": "Error: nope"})
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(sample).blockers


def test_the_marker_is_provenance_and_not_prompt_text():
    """It must not reach system/user/messages, or the trace names its own error."""
    sample = _missing_argument_sample()
    assert sample["lineage"]["recovery_error_kind"] == "missing_argument"
    assert "recovery_error_kind" not in json.dumps(sample["messages"])


# --- recovery allowance bounds ----------------------------------------------


def test_a_forged_marker_is_not_a_licence():
    """An unknown kind is a forgery, not a declaration.

    The first version accepted any non-empty string, so a sample that simply
    wrote `{"recovery_error_kind": "whatever"}` bought itself one free contract
    violation.
    """
    sample = copy.deepcopy(_missing_argument_sample())
    sample["lineage"]["recovery_error_kind"] = "not_a_kind"
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(sample).blockers


def test_the_allowance_must_land_on_the_recorded_failed_step():
    """Otherwise it forgives a different call than the one that failed.

    The trace would then contain a contract violation nobody meant to teach,
    while the deliberate error went unchecked.
    """
    sample = copy.deepcopy(_missing_argument_sample())
    for step in sample.get("steps") or []:
        step["failed"] = False           # no step is recorded as failing
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(sample).blockers


def test_more_than_one_failed_step_disables_the_allowance():
    """Exactly one failure is what a recovery trace is."""
    sample = copy.deepcopy(_missing_argument_sample())
    steps = sample.get("steps") or []
    if len(steps) >= 2:
        steps[0]["failed"] = True
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(sample).blockers


def test_the_known_kinds_are_the_three_that_exist():
    from strata.train.sample import RECOVERY_ERROR_KINDS
    from strata.train.synth import ERROR_KINDS

    assert set(ERROR_KINDS) == set(RECOVERY_ERROR_KINDS)
    assert RECOVERY_ERROR_KINDS == {"wrong_role", "value_as_key", "missing_argument"}
