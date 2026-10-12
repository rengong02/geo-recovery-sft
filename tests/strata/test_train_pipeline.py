"""The F1 pipeline: template -> real execution -> reference check -> sample.

These tests are about the seams, because the seams are where the data can go
wrong quietly:

* the executor must keep the tool's **actual** return, and keep enough of it for
  validation to compare against -- a truncation that loses the difference is
  worse than no truncation, because it looks like agreement;
* a failure must be recorded as a step, never raised away -- the recovery trace
  is the reason failure samples exist at all;
* the reference check must catch a template whose "wrong" guess is a real
  column, since that template would emit a failure that is not one.
"""

from __future__ import annotations

from strata.train.executor import (
    RENDERER_VERSION,
    execute_plan,
    render_visible,
)


def test_expanded_synthetic_sample_records_its_generator_provenance():
    from strata.train.synth import build_synth_f1_batch
    from strata.train.synth_families import FAMILY_VERSION, f1_family_cases

    samples, report = build_synth_f1_batch(f1_family_cases("year_role", count=1))

    assert report[0]["gold_ok"]
    generator = samples[0]["source_provenance"]["generator"]
    assert generator["module"] == "strata.train.synth_families"
    assert generator["version"] == FAMILY_VERSION








def test_expanded_synthetic_sample_replays_with_its_declared_runtime():
    from strata.train.replay import replay_all
    from strata.train.synth import build_synth_f1_batch
    from strata.train.synth_families import f1_family_cases

    samples, _ = build_synth_f1_batch(f1_family_cases("year_role", count=1))
    report = replay_all(samples)

    assert report["replayed_equal"] == 1, report["reports"]
    assert report["tools_changed"] == 0, report["reports"]
    assert report["stale"] == 0, report["reports"]


class _FakeTool:
    """Stands in for a GeoBench tool: returns what it is told, or raises."""

    def __init__(self, result=None, raises: Exception | None = None):
        self.result, self.raises = result, raises
        self.calls: list[dict] = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.result


# --- executor -----------------------------------------------------------------


def test_the_raw_observation_is_kept_in_full():
    """Validation compares against raw, so truncating it there would compare
    two views of the same shortening and always agree."""
    long_text = "x" * 5000
    execution = execute_plan([("load_data", {"dataset": "d"})],
                             {"load_data": _FakeTool(long_text)})
    step = execution.steps[0]
    assert len(step.raw) == 5000
    assert len(step.visible) < len(step.raw)


def test_truncation_is_versioned_and_says_what_it_dropped():
    visible = render_visible("y" * 3000)
    assert RENDERER_VERSION in visible
    assert "truncated" in visible


def test_short_results_are_not_annotated():
    """A marker on a result that was not truncated is noise the model has to
    learn to ignore."""
    assert render_visible("short") == "short"


def test_a_failing_call_is_a_step_not_an_exception():
    execution = execute_plan([("get_unique_values", {"column": "NAME"})],
                             {"get_unique_values": _FakeTool("Error: Column 'NAME' not found")})
    assert len(execution.steps) == 1
    assert execution.steps[0].failed


def test_a_tool_that_raises_is_recorded_rather_than_propagated():
    """Some upstream tools raise instead of returning an error string. Both must
    land in the trace, or the sample silently loses the hard failures."""
    execution = execute_plan([("x", {})],
                             {"x": _FakeTool(raises=ValueError("cannot unpack non-iterable"))})
    assert execution.steps[0].failed
    assert "ValueError" in execution.steps[0].raw


def test_steps_are_paired_into_messages_by_id():
    execution = execute_plan([("a", {}), ("b", {})],
                             {"a": _FakeTool("ok-a"), "b": _FakeTool("ok-b")})
    messages = [m for s in execution.steps for m in s.as_messages()]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_1", "call_2"]


# --- reference ----------------------------------------------------------------








# --- the assembled sample -----------------------------------------------------
















# --- F2: the join family ------------------------------------------------------










def test_synthetic_samples_replay_with_their_declared_runtime():
    from strata.train.replay import replay_all
    from strata.train.synth import (
        build_synth_f1_batch,
        build_synth_f2_batch,
        f1_synth_cases,
        f2_synth_cases,
    )

    f1, _ = build_synth_f1_batch(f1_synth_cases(2))
    f2, _ = build_synth_f2_batch(f2_synth_cases(2))
    report = replay_all(f1 + f2)

    assert report["replayed_equal"] == 4, report["reports"]
    assert report["stale"] == 0


