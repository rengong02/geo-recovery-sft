"""Grading an attempt against the semantic-role gold.

The gold is not "did the call run" and not "does the column exist". Both of
those pass a model that picks `Country Code` when the question asked for the
country *name*, because `Country Code` exists and the call succeeds. Measured on
the real data, that mistake is invisible in every structural property: the frame
is the same size, the column is real, no error is raised.

So the scorer's job is not a boolean. It is to separate the two ways an answer
can look fine and be wrong:

* **a name that is not there** -- the p2c failure, loud, self-correcting if the
  model reads the error;
* **a real column with the wrong meaning** -- silent, and the one worth being
  able to count.

An attempt is what the model produced: the calls it made and its termination.
The scorer never trusts either.
"""

from __future__ import annotations

from strata.train.score import (
    OUTCOME_CORRECT,
    OUTCOME_MALFORMED,
    OUTCOME_NO_TERMINATION,
    OUTCOME_PICKED_CONFUSABLE,
    OUTCOME_PICKED_NONEXISTENT,
    OUTCOME_WRONG_KEY_PAIR,
    score_attempt,
)

COLUMNS = ["Country Name", "Country Code", "Indicator Name", "1990", "2021"]


def _f1_attempt(column: str | None, *, terminate: bool = True) -> dict:
    termination = ({"status": "done", "selected_column": column} if terminate else None)
    calls = ([{"name": "get_unique_values", "arguments": {"column": column}}]
             if column else [])
    return {"termination": termination, "calls": calls}


def _sample(family: str = "F1", **gold) -> dict:
    base = {"sample_id": "s", "family": family, "gold": gold}
    return base


# --- F1 -----------------------------------------------------------------------


def test_the_right_column_passes():
    outcome, detail = score_attempt(
        _sample(role="country_name", column="Country Name"), _f1_attempt("Country Name"),
        frame_columns=COLUMNS)
    assert outcome == OUTCOME_CORRECT
    assert detail["selected"] == "Country Name"


def test_a_nonexistent_column_is_its_own_outcome():
    """The p2c failure mode: the model guesses a name that is not in the frame."""
    outcome, _ = score_attempt(
        _sample(role="country_name", column="Country Name"), _f1_attempt("NAME"),
        frame_columns=COLUMNS)
    assert outcome == OUTCOME_PICKED_NONEXISTENT


def test_a_real_but_wrong_column_is_its_own_outcome():
    """The one that looks fine. `Country Code` exists, the call succeeds, and it
    answers a different question."""
    outcome, detail = score_attempt(
        _sample(role="country_name", column="Country Name"), _f1_attempt("Country Code"),
        frame_columns=COLUMNS)
    assert outcome == OUTCOME_PICKED_CONFUSABLE
    assert "Country Code" in detail["selected"]


def test_an_unrelated_real_column_is_not_called_confusable():
    """Only near-misses get that label; a plainly different column is just wrong.
    Overloading the label would make the count meaningless."""
    outcome, _ = score_attempt(
        _sample(role="country_name", column="Country Name"), _f1_attempt("1990"),
        frame_columns=COLUMNS)
    assert outcome == OUTCOME_PICKED_NONEXISTENT


def test_no_termination_is_recorded_rather_than_counted_as_wrong():
    """A model that never finished has not answered; scoring that as an
    incorrect answer would hide a format failure inside an accuracy figure."""
    outcome, _ = score_attempt(
        _sample(role="country_name", column="Country Name"),
        _f1_attempt(None, terminate=False), frame_columns=COLUMNS)
    assert outcome == OUTCOME_NO_TERMINATION


def test_a_malformed_termination_is_its_own_outcome():
    outcome, _ = score_attempt(
        _sample(role="country_name", column="Country Name"),
        {"termination": "I think it is Country Name", "calls": []},
        frame_columns=COLUMNS)
    assert outcome == OUTCOME_MALFORMED


def test_scoring_never_raises_on_a_missing_attempt():
    outcome, _ = score_attempt(_sample(role="r", column="c"), None, frame_columns=COLUMNS)
    assert outcome == OUTCOME_NO_TERMINATION


# --- F2 -----------------------------------------------------------------------


def test_the_right_key_pair_passes():
    outcome, detail = score_attempt(
        _sample(family="F2", join_keys=["Country Code", "ISO_A3"]),
        {"termination": {"status": "done", "join_keys": ["Country Code", "ISO_A3"]},
         "calls": []},
        fill_rate=lambda pair: 0.741)
    assert outcome == OUTCOME_CORRECT
    assert detail["fill_rate"] == 0.741


def test_a_wrong_pair_is_scored_by_what_it_actually_matched():
    """Not by whether the call ran -- the wrong join runs perfectly. What
    separates the pairs is how much of the value column survived, measured here
    at 0.0 against the correct pair's 0.741."""
    outcome, detail = score_attempt(
        _sample(family="F2", join_keys=["Country Code", "ISO_A3"]),
        {"termination": {"status": "done", "join_keys": ["Country Name", "ISO_A3"]},
         "calls": []},
        fill_rate=lambda pair: 0.741 if tuple(pair) == ("Country Code", "ISO_A3") else 0.0)
    assert outcome == OUTCOME_WRONG_KEY_PAIR
    assert detail["fill_rate"] == 0.0
    assert detail["expected_fill_rate"] == 0.741


def test_f2_also_separates_a_missing_termination():
    outcome, _ = score_attempt(
        _sample(family="F2", join_keys=["Country Code", "ISO_A3"]),
        {"termination": None, "calls": []}, fill_rate=lambda pair: 0.0)
    assert outcome == OUTCOME_NO_TERMINATION
