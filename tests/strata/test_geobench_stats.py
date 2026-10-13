"""The test that says "not significant" has to be as reproducible as the number.

P1's negative result rests on this function. If it were wrong -- too lenient, or
mis-signed -- the conclusion "the hint has no effect" would be wrong in the
direction that matters, so its behaviour is pinned against an independently computed exact value from the same raw vectors.
"""

from __future__ import annotations

import pytest

from strata.geobench import stats
from strata.geobench.stats import permutation_p


def test_the_p1a_vectors_give_the_independently_computed_p():
    """The real arms, from `evals/ablation/p1a_schema_hints`.

    An independent calculation gives 0.4126 for these vectors; this pins the same number so a
    later change to the test cannot quietly move it.
    """
    on = [2, 0, 2, 1, 0, 0, 1]
    off = [0, 0, 0, 6, 0, 2, 13]

    diff, p = permutation_p(on, off)

    assert diff == pytest.approx(6 / 7 - 21 / 7)
    assert p == pytest.approx(0.4126, abs=5e-4), (
        "the P1a conclusion depends on this value; if it moved, say so in the report"
    )


def test_identical_arms_are_never_significant():
    diff, p = permutation_p([1, 2, 3, 4], [1, 2, 3, 4])

    assert diff == 0
    assert p == pytest.approx(1.0)


def test_completely_separated_arms_are_significant():
    """Every value in one arm above every value in the other."""
    _, p = permutation_p([10, 11, 12, 13, 14], [0, 1, 2, 3, 4])

    assert p < 0.05


def test_the_sign_follows_the_argument_order():
    """A negative difference must mean "the first arm is smaller"."""
    diff, _ = permutation_p([1, 1, 1], [5, 5, 5])

    assert diff < 0


def test_one_observation_per_arm_is_allowed_and_conservative():
    """A single sample per arm cannot establish anything, and says so."""
    _, p = permutation_p([1], [100])

    assert p == pytest.approx(1.0), (
        "with one observation a side, no split is more extreme than the observed "
        "one, so the honest p is 1 -- not a small number that would read as a result"
    )


def test_an_empty_arm_is_an_error_not_a_zero():
    """A silent 0.0 p would read as a strong result."""
    with pytest.raises(ValueError, match="at least one"):
        permutation_p([], [1, 2, 3])
    with pytest.raises(ValueError, match="at least one"):
        permutation_p([1, 2, 3], [])


def test_paired_permutation_uses_within_task_differences():
    assert hasattr(stats, "paired_permutation_p")
    diff, p = stats.paired_permutation_p([10, 10, 10], [0, 0, 0])

    assert diff == 10
    assert p == pytest.approx(0.25)


def test_paired_permutation_rejects_mismatched_arms():
    assert hasattr(stats, "paired_permutation_p")
    with pytest.raises(ValueError, match="same number"):
        stats.paired_permutation_p([1, 2], [1])
