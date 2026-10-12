"""The statistics an ablation is allowed to draw conclusions from.

Small and dependency-free on purpose. The first P1 result rested on a reading
of two means, which is how a raw 71% drop in failures turned out to be one
control run; the test that says "not significant" has to be as reproducible as
the number it is judging.
"""

from __future__ import annotations

import itertools
import statistics


def permutation_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided permutation test on the difference in means, enumerated exactly.

    Chosen over a t-test because ablation arms are tiny and plainly not normal:
    the P1a control was `[0, 0, 0, 6, 0, 2, 13]`. Enumerating every split makes
    no distributional assumption at all, and at n=7 a side it is 3432
    combinations -- cheaper than deciding whether the assumptions hold.

    Returns `(observed_mean_difference, p)`. `a` first, so the sign is readable:
    a negative difference means `a` is smaller.
    """
    if not a or not b:
        raise ValueError("permutation_p needs at least one observation per arm")

    observed = statistics.mean(a) - statistics.mean(b)
    pool = list(a) + list(b)
    n = len(a)
    hits = total = 0
    for combo in itertools.combinations(range(len(pool)), n):
        chosen = set(combo)
        left = [pool[i] for i in chosen]
        right = [pool[i] for i in range(len(pool)) if i not in chosen]
        total += 1
        if abs(statistics.mean(left) - statistics.mean(right)) >= abs(observed) - 1e-12:
            hits += 1
    return observed, hits / total


def paired_permutation_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Exact sign-flip test on paired within-task differences."""
    if not a or not b:
        raise ValueError("paired_permutation_p needs at least one observation per arm")
    if len(a) != len(b):
        raise ValueError("paired_permutation_p needs the same number of observations")

    differences = [left - right for left, right in zip(a, b)]
    observed = statistics.mean(differences)
    hits = total = 0
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        candidate = statistics.mean(diff * sign for diff, sign in zip(differences, signs))
        total += 1
        if abs(candidate) >= abs(observed) - 1e-12:
            hits += 1
    return observed, hits / total
