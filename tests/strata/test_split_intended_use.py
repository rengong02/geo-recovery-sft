"""A diagnostic set is not a failed training set.

Two readiness gates contradict an orthogonal diagnostic design by construction:

* the per-family sample minimum wants dozens per family; a diagnostic cell is
  small on purpose, because its value is the comparison rather than the count;
* the content-schema gate wants one fingerprint per split, but isolating a
  single variable *requires* the cells to share everything else -- including the
  content schema.

Measured on the orthogonal F2 set, all four crossing fingerprints are
`(shape, full)` / `(shape, partial)` pairs. The crossings are the design working,
so a gate that blocks them is blocking the design.

The resolution is not to weaken the gate: the crossings are still reported, for
both uses. What changes is that a diagnostic set says plainly that it is not
training data, instead of failing a gate it was never meant to pass.
"""

from __future__ import annotations

import pytest

from strata.train.split import INTENDED_USES, build_split_report
from strata.train.synth import build_synth_f2_batch
from strata.train.synth_families import all_f2_family_cases


@pytest.fixture(scope="module")
def orthogonal_samples() -> list[dict]:
    """All six cells: enough groups for a three-way split to be possible at all.

    A smaller slice fails the structural gates first and never reaches the
    fingerprint gate this file is about.
    """
    cases = all_f2_family_cases(4)
    samples, report = build_synth_f2_batch(cases)
    for row in report:
        assert row["gold_ok"] and row["validate_ok"], row
    return samples


def test_both_uses_are_named():
    assert set(INTENDED_USES) == {"training", "diagnostic"}


def test_a_diagnostic_set_is_not_ready_because_it_is_not_training_data(orthogonal_samples):
    report = build_split_report(orthogonal_samples, intended_use="diagnostic")
    assert report["training_ready"] is False
    assert report["training_readiness_reasons"] == ["intended_use_is_diagnostic"]


def test_the_crossings_are_reported_for_both_uses(orthogonal_samples):
    """Not suppressed for the diagnostic case.

    Hiding them would make the design look clean by making the report less
    informative -- and the crossings are the thing a reader most needs to see.
    """
    diagnostic = build_split_report(orthogonal_samples, intended_use="diagnostic")
    training = build_split_report(orthogonal_samples, intended_use="training")

    assert diagnostic["content_schema_fingerprint_crossings"]
    assert (diagnostic["content_schema_fingerprint_crossings"]
            == training["content_schema_fingerprint_crossings"])


def test_training_still_treats_the_crossing_as_a_blocker(orthogonal_samples):
    """The gate is not weakened -- only made inapplicable where it cannot apply."""
    report = build_split_report(orthogonal_samples, intended_use="training")
    assert "content_schema_cross_split_leakage" in report["training_readiness_reasons"]


def test_the_use_is_recorded_on_the_report(orthogonal_samples):
    assert build_split_report(orthogonal_samples, intended_use="diagnostic")["intended_use"] == "diagnostic"
    assert build_split_report(orthogonal_samples)["intended_use"] == "training"
