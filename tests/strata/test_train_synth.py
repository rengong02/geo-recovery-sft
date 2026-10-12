"""Generated training tables, and the tool surface they need declared.

Two things this file exists to hold down, both of which were broken first:

* **a loader bound to one table.** The first version took a single spec, so it
  answered for exactly one handle and rejected every other case with an error
  that read like bad data. It is a registry now.
* **a surface hardcoded to the benchmark's six.** The validator and the schema
  builder both read `CORE_TOOLS`, so every synthetic sample failed as
  `unknown_tool`/`bad_tool_schema`. That was the check working and the interface
  being wrong: a sample must be judged against the tools *it declares*.
"""

from __future__ import annotations

import json

from strata.geobench.tools import benchmark_state
from strata.train.reference import confusable_columns
from strata.train.sample import CORE_TOOLS, allowed_tools, validate_sample
from strata.train.synth import (
    LOAD_TOOL_NAME,
    build_synth_f1_batch,
    build_synth_f2_batch,
    f1_synth_cases,
    f2_synth_cases,
    fresh_synth_tools,
)

CASES = f1_synth_cases(12)


# --- the generated tables -----------------------------------------------------


def test_generation_is_deterministic():
    """A sample that does not replay to itself is not a sample."""
    from strata.train.synth import build_table

    spec = CASES[0].table
    assert build_table(spec).equals(build_table(spec))


def test_entity_namespace_is_part_of_a_generated_world():
    from strata.train.synth import SynthTableSpec, build_table

    first = SynthTableSpec(
        handle="world-a", columns=("Region", "region_id"), n_rows=4,
        seed=11, id_column="region_id", entity_namespace="world-a",
    )
    second = SynthTableSpec(
        handle="world-b", columns=("Region", "region_id"), n_rows=4,
        seed=11, id_column="region_id", entity_namespace="world-b",
    )

    assert not set(build_table(first)["region_id"]) & set(build_table(second)["region_id"])
    assert first.as_dict()["entity_namespace"] == "world-a"


def test_the_near_miss_column_is_present_and_is_not_the_answer():
    """The whole point of the family: a real column that is the wrong answer.

    `GDP per capita (current US$)` and `GDP (current US$)` both exist. A checker
    that only asks whether the chosen column exists passes the wrong one.
    """
    case = next(c for c in CASES if c.correct_column.endswith("(current US$)"))
    state = benchmark_state()
    tools = fresh_synth_tools([case.table], state)
    tools[LOAD_TOOL_NAME].execute(handle=case.table.handle, output_dataframe_name="f")
    columns = list(state["data_store"]["f"].columns)

    assert case.correct_column in columns
    assert "GDP (current US$)" in columns
    assert "GDP (current US$)" in confusable_columns(case.correct_column, columns)


def test_every_generated_case_has_a_valid_gold():
    """The reference check, over the whole generated batch."""
    for case in CASES:
        columns = case.table.columns
        assert case.correct_column in columns, case.case_id
        offenders = [g for g in case.wrong_guesses if g in columns]
        assert not offenders, (case.case_id, offenders)


def test_wrong_guesses_really_fail_when_executed():
    """The failure trace has to be a failure. A guess that resolved would put a
    fabricated difficulty in the data."""
    case = CASES[0]
    state = benchmark_state()
    tools = fresh_synth_tools([c.table for c in CASES], state)
    tools[LOAD_TOOL_NAME].execute(handle=case.table.handle, output_dataframe_name="f")

    for guess in case.wrong_guesses:
        result = tools["get_unique_values"].execute(dataframe_name="f", column=guess)
        assert result.startswith("Error"), (guess, result[:80])


# --- the loader -----------------------------------------------------------------


def test_the_loader_answers_for_any_registered_handle():
    """Not just the first one. The single-spec version answered for one case and
    errored on the other eleven."""
    state = benchmark_state()
    tools = fresh_synth_tools([c.table for c in CASES], state)
    for case in CASES[:4]:
        result = tools[LOAD_TOOL_NAME].execute(
            handle=case.table.handle, output_dataframe_name="f")
        assert not result.startswith("Error"), (case.case_id, result[:80])
        assert case.correct_column in state["data_store"]["f"].columns


def test_an_unknown_handle_is_an_error_not_a_crash():
    state = benchmark_state()
    tools = fresh_synth_tools([CASES[0].table], state)
    result = tools[LOAD_TOOL_NAME].execute(handle="syn_nope", output_dataframe_name="f")
    assert result.startswith("Error")
    assert "syn_nope" in result


def test_the_description_comes_from_the_frame_that_was_stored():
    """Built from the frame, not the spec: a spec that disagreed with its
    generator would otherwise be invisible."""
    case = CASES[0]
    state = benchmark_state()
    tools = fresh_synth_tools([case.table], state)
    message = tools[LOAD_TOOL_NAME].execute(handle=case.table.handle,
                                            output_dataframe_name="f")
    frame = state["data_store"]["f"]
    assert f"Rows: {len(frame)}" in message
    for column in frame.columns:
        assert str(column) in message


# --- the declared surface -------------------------------------------------------


def test_the_synthetic_surface_is_five_tools_including_ours():
    tools = fresh_synth_tools([CASES[0].table], benchmark_state())
    assert len(tools) == 5
    assert LOAD_TOOL_NAME in tools
    assert "load_data" not in tools and "load_geodata" not in tools


def test_synthetic_surface_does_not_require_catalog_tool_builder(monkeypatch):
    import strata.geobench.tools as geobench_tools

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("synthetic tools must not load the catalog builder")

    monkeypatch.setattr(geobench_tools, "build_geobench_tools", fail_if_loaded)
    tools = fresh_synth_tools([CASES[0].table], benchmark_state())
    assert len(tools) == 5


def test_the_allowed_tool_set_comes_from_the_record():
    """Not from `CORE_TOOLS`. Hardcoding it made every synthetic sample invalid."""
    record = {"tools": [{"type": "function", "function": {
        "name": LOAD_TOOL_NAME, "parameters": {"type": "object",
                                               "properties": {"handle": {}},
                                               "required": ["handle"]}}}]}
    assert allowed_tools(record) == frozenset({LOAD_TOOL_NAME})
    assert allowed_tools({}) == CORE_TOOLS


def test_the_generated_batch_validates_end_to_end():
    """Gold and V1 both, over the real generated set."""
    samples, report = build_synth_f1_batch(CASES)
    assert len(samples) == len(CASES)
    for row in report:
        assert row["gold_ok"], (row["sample_id"], row["gold_problems"])
        assert row["validate_ok"], (row["sample_id"], row["blockers"])


def test_the_generated_f2_batch_has_a_real_join_signal():
    samples, report = build_synth_f2_batch(f2_synth_cases(3))
    assert len(samples) == 3
    for row in report:
        assert row["gold_ok"], (row["sample_id"], row["gold_problems"])
        assert row["validate_ok"], (row["sample_id"], row["blockers"])
    for sample in samples:
        assert sample["family"] == "F2"
        assert sample["source_provenance"]["generator"]["specs"]
        assert sample["gold"]["join_keys"]


def test_a_synthetic_sample_declares_its_own_provenance():
    """`source_kind` is what stops a generated sample being mistaken for a
    benchmark one downstream."""
    samples, _ = build_synth_f1_batch(CASES[:2])
    for sample in samples:
        assert sample["source_provenance"]["source_kind"] == "synthetic_generator"
        assert sample["source_provenance"]["dataset_names"]
        assert json.dumps(sample["source_provenance"]).count("syn_") >= 1


def test_tool_server_can_build_the_declared_synthetic_profile():
    from scripts.tool_server import build_synthetic_tools

    case = CASES[0]
    tools, state = build_synthetic_tools([case.table.as_dict()])

    assert LOAD_TOOL_NAME in tools
    result = tools[LOAD_TOOL_NAME].execute(
        handle=case.table.handle, output_dataframe_name="frame",
    )
    assert not result.startswith("Error")
    assert case.correct_column in state["data_store"]["frame"].columns


def test_tool_server_measures_synthetic_join_fill_rate():
    from scripts.tool_server import synthetic_fill_rate
    from strata.train.synth_families import f2_family_cases

    case = f2_family_cases("shared_root__full", count=1)[0]
    specs = [case.data_table.as_dict(), case.geo_table.as_dict()]

    correct = synthetic_fill_rate(
        specs, "stats_frame", "geo_frame", case.value_column,
        case.data_table.handle, case.geo_table.handle,
        (case.data_key, case.geo_key),
    )
    wrong = synthetic_fill_rate(
        specs, "stats_frame", "geo_frame", case.value_column,
        case.data_table.handle, case.geo_table.handle,
        case.wrong_pairs[0],
    )

    assert correct >= 0.5
    assert wrong == 0.0


def test_synthetic_manifest_exposes_its_loader_tool():
    samples, _ = build_synth_f1_batch(CASES[:1])
    manifest = samples[0]["messages"][0]["content"]

    assert LOAD_TOOL_NAME in manifest
    assert "load_data(" not in manifest
    assert "load_geodata(" not in manifest


def test_each_f1_sample_only_exposes_its_own_table_handle():
    samples, _ = build_synth_f1_batch(CASES[:2])

    first_manifest = samples[0]["messages"][0]["content"]
    assert CASES[0].table.handle in first_manifest
    assert CASES[1].table.handle not in first_manifest


def test_each_f2_sample_only_exposes_its_two_table_handles():
    cases = f2_synth_cases(2)
    samples, _ = build_synth_f2_batch(cases)

    first_manifest = samples[0]["messages"][0]["content"]
    assert cases[0].data_table.handle in first_manifest
    assert cases[0].geo_table.handle in first_manifest
    assert cases[1].data_table.handle not in first_manifest
    assert cases[1].geo_table.handle not in first_manifest


def test_year_family_question_identifies_the_target_year():
    from strata.train.synth_families import f1_family_cases

    case = f1_family_cases("year_role", count=1)[0]

    assert case.correct_column in case.question


def test_naming_family_contains_distinct_column_conventions():
    from strata.train.synth_families import f1_family_cases

    cases = f1_family_cases("naming_convention", count=12)
    columns = {tuple(case.table.columns) for case in cases}

    assert len(columns) >= 3


def test_naming_family_gold_column_matches_each_convention():
    from strata.train.synth_families import f1_family_cases

    cases = f1_family_cases("naming_convention", count=12)

    assert all(case.correct_column in case.table.columns for case in cases)


def test_a_benchmark_record_still_needs_the_whole_benchmark_surface():
    """The guard Codex put there is kept: a surface drawn from the benchmark's
    six must be all six, not a convenient subset."""
    partial = {"tools": [{"type": "function", "function": {
        "name": "get_unique_values",
        "parameters": {"type": "object",
                       "properties": {"dataframe_name": {}, "column": {}},
                       "required": ["dataframe_name", "column"]}}}],
        "messages": []}
    assert not validate_sample(partial).ok
