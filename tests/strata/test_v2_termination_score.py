"""The v2 public termination contract is also a grading contract."""

import pytest

from strata.train.score import score_attempt, summarise


def sample(family="F1"):
    return {"schema_version": 2, "termination_schema_version": 2, "family": family,
            "gold": {"column": "Population, total", "join_keys": ["iso", "iso3"]}}


@pytest.mark.parametrize("status", ["error", "failed", "anything", None])
def test_v2_cannot_pass_merely_by_including_the_gold_column(status):
    outcome, _ = score_attempt(
        sample(), {"termination": {"status": status, "schema_version": 2,
                                    "selected_column": "Population, total"}})
    assert outcome != "correct"


@pytest.mark.parametrize("version", [None, 1, "2", True])
def test_v2_requires_integer_schema_version_2(version):
    outcome, _ = score_attempt(
        sample(), {"termination": {"status": "done", "schema_version": version,
                                    "selected_column": "Population, total"}})
    assert outcome == "malformed_termination"


def test_v2_valid_error_is_not_malformed_and_is_not_an_answer():
    outcome, detail = score_attempt(
        sample(), {"termination": {"status": "error", "schema_version": 2,
                                    "error_code": "missing_column", "error_message": "cannot resolve"}})
    assert outcome == "reported_error"
    assert detail["error_code"] == "missing_column"
    summary = summarise(["correct", "reported_error", "no_termination"])
    assert summary["correct"] == 1
    assert summary["answered"] == 1
    assert summary["reported_errors"] == 1
    assert summary["format_failures"] == 1


def test_v2_successful_f2_is_checked_with_the_same_version_contract():
    outcome, _ = score_attempt(
        sample("F2"), {"termination": {"status": "done", "schema_version": 2,
                                        "join_keys": ["iso", "iso3"]}},
        fill_rate=lambda pair: 1.0)
    assert outcome == "correct"
