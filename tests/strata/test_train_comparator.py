"""Comparing a stored observation against a fresh replay of the same call.

Requiring the two to be byte-identical was the first design and it was wrong:
timestamps, temp paths and base64 blobs differ between two runs of the *same*
correct call, so a byte rule refuses good samples and -- worse -- the refusals
look like data rot when they are not.

The replacement compares in layers, and the layering is the whole point:

* structure (which call, which tool, which arguments) is **exact**, because a
  difference there is a real difference;
* floats get a **stated** tolerance rather than a rounding that nobody wrote down;
* timestamps and paths are **normalised**, because two runs legitimately differ;
* base64 and image payloads are compared by **manifest** -- length class, not
  content -- since the bytes carry no meaning the task depends on.

A comparison that only says "not equal" is not actionable, so every difference
is reported with the layer it was found in.
"""

from __future__ import annotations

from strata.train.comparator import (
    COMPARATOR_VERSION,
    FLOAT_DECIMALS,
    compare_executions,
    compare_observations,
    compare_steps,
    normalise,
)


def _step(**overrides) -> dict:
    step = {
        "call_id": "call_1",
        "tool": "get_unique_values",
        "arguments": {"dataframe_name": "countries", "column": "NAME_EN"},
        "observation_raw": "Unique values in column 'NAME_EN': ['Indonesia', 'Chile']",
        "failed": False,
    }
    step.update(overrides)
    return step


# --- the normalisation --------------------------------------------------------


def test_an_identical_observation_needs_no_normalisation():
    assert normalise("plain text") == "plain text"


def test_timestamps_are_normalised_away():
    """Two runs of the same call are minutes apart. That is not a difference in
    the data."""
    left, right = normalise("started 2025-01-16T02:25:36.580Z"), \
        normalise("started 2026-10-05T11:02:09.001Z")
    assert left == right


def test_temp_paths_are_normalised_away():
    """Each run gets its own workspace, by design."""
    assert normalise("wrote /tmp/strata-run-8f2a/out.png") == \
        normalise("wrote /tmp/strata-run-c91b/out.png")


def test_floats_are_rounded_to_a_stated_precision():
    """A tolerance that is written down can be argued with; a tolerance that is
    implicit cannot."""
    assert FLOAT_DECIMALS >= 1
    assert normalise("mean 11.8655060897") == normalise("mean " + str(round(11.8655060897, FLOAT_DECIMALS)))


def test_a_float_beyond_the_tolerance_is_a_difference():
    left = normalise("mean 11.8655060897")
    right = normalise("mean " + str(round(11.8655060897, FLOAT_DECIMALS) + 10 ** (-FLOAT_DECIMALS) * 5))
    assert left != right


def test_base64_is_compared_by_length_not_content():
    """The bytes of a rendered PNG vary with font hinting; the task does not
    depend on them."""
    a = normalise("image: " + "QUJD" * 40)
    b = normalise("image: " + "REVG" * 40)
    assert a == b


def test_a_different_base64_size_is_still_a_difference():
    """Same rule, but a payload that changed *size* is worth knowing about."""
    assert normalise("image: " + "QUJD" * 40) != normalise("image: " + "QUJD" * 41)


# --- the layers ---------------------------------------------------------------


def test_identical_steps_are_equal():
    comparison = compare_steps(_step(), _step())
    assert comparison.equal
    assert comparison.diffs == []


def test_a_different_tool_is_a_structural_difference():
    comparison = compare_steps(_step(), _step(tool="load_data"))
    assert not comparison.equal
    assert any(d.layer == "structural" and d.field == "tool" for d in comparison.diffs)


def test_different_arguments_are_a_structural_difference():
    comparison = compare_steps(_step(), _step(arguments={"column": "NAME"}))
    assert not comparison.equal
    assert any(d.layer == "structural" and d.field == "arguments" for d in comparison.diffs)


def test_a_different_observation_is_reported_with_its_layer():
    comparison = compare_steps(_step(), _step(observation_raw="totally different"))
    assert not comparison.equal
    assert any(d.layer == "observation" for d in comparison.diffs)


def test_a_normalised_away_difference_is_not_a_difference():
    comparison = compare_steps(
        _step(observation_raw="wrote /tmp/run-a/f.png"),
        _step(observation_raw="wrote /tmp/run-b/f.png"))
    assert comparison.equal, comparison.diffs


def test_a_failure_flip_is_reported():
    """A call that errored last time and succeeded now is a real change, even if
    the text happened to match."""
    comparison = compare_steps(_step(), _step(failed=True, observation_raw="same text"))
    assert not comparison.equal
    assert any(d.field == "failed" for d in comparison.diffs)


def test_every_difference_is_listed_not_just_the_first():
    comparison = compare_steps(_step(), _step(tool="load_data", observation_raw="other",
                                              failed=True))
    assert len(comparison.diffs) >= 3


# --- the report ---------------------------------------------------------------


def test_a_matching_execution_reports_clean():
    report = compare_executions([_step()], [_step()])
    assert report["equal"]
    assert report["diffs"] == []
    assert report["comparator_version"] == COMPARATOR_VERSION


def test_a_length_change_is_reported_rather_than_crashing():
    report = compare_executions([_step(), _step(call_id="call_2")], [_step()])
    assert not report["equal"]
    assert report["missing"] or report["extra"]


def test_duplicate_call_ids_are_reported_not_collapsed():
    expected = [_step(), _step(call_id="call_1", observation_raw="second")]
    actual = [_step(), _step(call_id="call_1", observation_raw="second")]
    report = compare_executions(expected, actual)
    assert not report["equal"]
    assert report["duplicate_expected"] == ["call_1"]
    assert report["duplicate_actual"] == ["call_1"]


def test_the_report_names_the_offending_step():
    report = compare_executions([_step(), _step(call_id="call_2", observation_raw="x")],
                                [_step(), _step(call_id="call_2", observation_raw="y")])
    assert not report["equal"]
    assert report["diffs"][0]["call_id"] == "call_2"


def test_the_comparator_is_versioned_because_its_rules_can_change():
    report = compare_executions([_step()], [_step()])
    assert report["comparator_version"] == COMPARATOR_VERSION


def test_comparing_a_raw_string_uses_the_same_rules_as_steps():
    diffs = compare_observations("wrote /tmp/a/x", "wrote /tmp/b/x")
    assert diffs == []
