"""The recovery trace must teach the failure the model actually has.

The first recovery bundle opened every trace with `geokey=f"{geo_key}__missing"`.
Measured on it: all 144 errors were `KeyError`, every wrong `statkey` was a valid
column, and the invented name **announced itself** -- a solver could learn "if it
ends in `__missing`, pick another" without understanding keys at all.

Meanwhile the failure actually observed at inference (192/192 failing merges) is
`ValueError: merge on str and float64 columns`, from using a real column that
holds values. The bundle contained no such case, so it taught a rule that does
not apply, and the SFT came out worse than the base.

Three properties are required of every kind, and asserted here:

1. the wrong call **really fails**, produced by executing it;
2. the recovery **really succeeds**;
3. nothing in the call **names itself as wrong**.
"""

from __future__ import annotations

import pytest

from strata.geobench.tools import benchmark_state
from strata.train.executor import execute_plan
from strata.train.synth import (
    ERROR_KINDS,
    build_synth_f2_recovery_plan,
    fresh_synth_tools,
)
from strata.train.synth_families import f2_family_cases

CASE = f2_family_cases("shared_root__full", 1)[0]


def _run(kind: str):
    state = benchmark_state()
    tools = fresh_synth_tools([CASE.data_table, CASE.geo_table], state)
    return execute_plan(build_synth_f2_recovery_plan(CASE, kind), tools)


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_the_wrong_call_fails_for_real(kind):
    execution = _run(kind)
    assert execution.steps[2].failed, f"{kind} did not fail"
    assert execution.steps[2].raw.startswith("Error"), execution.steps[2].raw[:80]


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_the_recovery_succeeds(kind):
    execution = _run(kind)
    assert not execution.steps[3].failed, execution.steps[3].raw[:120]


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_no_call_names_itself_as_wrong(kind):
    """A giveaway in the arguments turns a recovery lesson into string matching."""
    _, arguments = build_synth_f2_recovery_plan(CASE, kind)[2]
    for value in arguments.values():
        assert "__missing" not in str(value), arguments
        assert "wrong" not in str(value).lower(), arguments


def test_at_least_one_kind_reproduces_the_error_seen_at_inference():
    """The dtype failure is the one the model actually hits; it must be present."""
    raw = _run("value_as_key").steps[2].raw
    assert "ValueError" in raw and "merge on" in raw


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        build_synth_f2_recovery_plan(CASE, "not-a-kind")
