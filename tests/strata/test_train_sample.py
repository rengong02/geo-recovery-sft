"""A training sample is a well-formed tool-call conversation, or it is not a sample.

The p2c audit produced 18 records that looked like trajectories and were not:
they had a *plan* (`tool_call_plan`) and no per-call observation, no `role`, no
`tool_call_id`. Nothing about them was machine-checked — the export gate counted
strings and compared task IDs.

So this validator is the first thing built, before any template, because the
format is the part that cannot be retrofitted: a missing observation cannot be
recovered afterwards without inventing it.

Two failures it exists to catch, both found by Codex on the earlier gate:

* **a renamed reserved task walked through.** Matching on the task id string
  means `RENAMED_DERIVED` is a new task as far as the gate can tell. The sample
  has to be refused by its *lineage* -- which dataset it came from -- because
  renaming does not change where the data came from.
* **structure was never checked.** `role` order, `tool_call_id` pairing and a
  machine-parseable termination are what make a trace usable; counting strings
  does not verify any of them.
"""

from __future__ import annotations

import json

from strata.train.executor import RENDERER_VERSION
from strata.train.sample import (
    BLOCKER_BAD_ARGUMENTS,
    BLOCKER_BAD_ROLE_ORDER,
    BLOCKER_BAD_STEP_ALIGNMENT,
    BLOCKER_BAD_TERMINATION,
    BLOCKER_BAD_TOOL_CALL_ID,
    BLOCKER_BAD_TOOL_SCHEMA,
    BLOCKER_BAD_VERSION,
    BLOCKER_MISSING_LINEAGE,
    BLOCKER_RESERVED_LINEAGE,
    BLOCKER_UNKNOWN_TOOL,
    BLOCKER_UNPAIRED_TOOL_CALL,
    CORE_TOOL_ARGUMENTS,
    CORE_TOOLS,
    RESERVED_LINEAGE_MARKERS,
    SAMPLE_SCHEMA_VERSION,
    TERMINATION_SCHEMA_VERSION,
    lineage_of,
    sample_to_json,
    validate_sample,
)

RESERVED_MARKER = min(RESERVED_LINEAGE_MARKERS)


def _calls(*specs) -> list[dict]:
    return [{"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(specs, 1)]


def _sample(**overrides) -> dict:
    """A sample that passes every check, so each test spoils exactly one thing."""
    sample = {
        "sample_id": "f1:demo:0001",
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "task_text": "Which column holds GDP per capita in the frame?",
        "source": "template",
        "lineage": {
            "source_lineage": "synthetic:gdp-family",
            "dataset_snapshot_id": "sha256:notreserved",
            "schema_variant_family": "gdp_v1",
            "generator_family": "f1_column_role",
            "template_id": "f1_col_role/0001",
            "seed": 17,
        },
        "renderer_version": RENDERER_VERSION,
        "comparator_version": "c1",
        "termination_schema_version": TERMINATION_SCHEMA_VERSION,
        "tool_source_digest": "sha256:tools",
        "dataset_digest": "sha256:data",
        "gold": {"role": "gdp_per_capita", "column": "GDP per capita (current US$)"},
        "tools": [
            {"type": "function", "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {key: {} for key in required | optional},
                    "required": sorted(required),
                },
            }}
            for name, (required, optional) in CORE_TOOL_ARGUMENTS.items()
        ],
        "steps": [{
            "call_id": "call_1",
            "tool": "get_unique_values",
            "arguments": {"dataframe_name": "gdp", "column": "GDP per capita (current US$)"},
            "observation_raw": "['GDP per capita (current US$)']",
            "observation_visible": "['GDP per capita (current US$)']",
            "failed": False,
            "seconds": 0.01,
        }],
        "messages": [
            {"role": "system", "content": "You have 6 tools."},
            {"role": "user", "content": "Which column holds GDP per capita?"},
            {"role": "assistant", "content": None, "tool_calls": _calls(
                ("get_unique_values", {"dataframe_name": "gdp", "column": "GDP per capita (current US$)"}))},
            {"role": "tool", "tool_call_id": "call_1", "content": "['GDP per capita (current US$)']"},
            {"role": "assistant", "content": json.dumps(
                {"status": "done", "schema_version": TERMINATION_SCHEMA_VERSION,
                 "selected_column": "GDP per capita (current US$)"})},
        ],
    }
    sample.update(overrides)
    return sample


# --- the shape that passes ----------------------------------------------------


def test_a_well_formed_sample_passes():
    verdict = validate_sample(_sample())
    assert verdict.ok, verdict.blockers
    assert verdict.blockers == []


def test_a_self_declared_executor_is_rejected():
    sample = _sample()
    for schema in sample["tools"]:
        if schema["function"]["name"] == "get_unique_values":
            schema["function"]["name"] = "invented_tool"
    for message in sample["messages"]:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] == "get_unique_values":
                call["function"]["name"] = "invented_tool"
    for step in sample["steps"]:
        if step["tool"] == "get_unique_values":
            step["tool"] = "invented_tool"

    verdict = validate_sample(sample)
    assert not verdict.ok
    assert BLOCKER_UNKNOWN_TOOL in verdict.blockers
    assert BLOCKER_BAD_TOOL_SCHEMA in verdict.blockers


def test_the_core_tool_set_is_six_for_the_first_batch():
    """F1/F2 use six; F5 adds a seventh. The count is asserted so the doc and
    the code cannot drift apart again."""
    assert len(CORE_TOOLS) == 6
    assert "make_choropleth_map" not in CORE_TOOLS


# --- structure ----------------------------------------------------------------


def test_roles_must_follow_the_conversation_shape():
    """system -> user -> (assistant/tool)* -> assistant. A trace that starts with
    a tool message is not a conversation, it is a fragment."""
    broken = _sample()
    broken["messages"] = [{"role": "tool", "tool_call_id": "call_1", "content": "orphan"}]
    assert BLOCKER_BAD_ROLE_ORDER in validate_sample(broken).blockers


def test_a_tool_call_without_its_observation_is_refused():
    """This is exactly what the p2c candidates were: calls with no results."""
    broken = _sample()
    broken["messages"] = [m for m in broken["messages"] if m["role"] != "tool"]
    assert BLOCKER_UNPAIRED_TOOL_CALL in validate_sample(broken).blockers


def test_an_observation_without_its_call_is_refused():
    broken = _sample()
    broken["messages"] = broken["messages"][:3] + [
        {"role": "tool", "tool_call_id": "call_999", "content": "no such call"},
        broken["messages"][-1],
    ]
    assert BLOCKER_UNPAIRED_TOOL_CALL in validate_sample(broken).blockers


def test_two_calls_may_not_share_one_observation():
    """Matching by name instead of id would let this through; matching by id
    cannot."""
    broken = _sample()
    broken["messages"] = broken["messages"][:1] + [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content": None, "tool_calls": _calls(
            ("get_unique_values", {"dataframe_name": "g", "column": "a"}),
            ("get_unique_values", {"dataframe_name": "g", "column": "b"}))},
        {"role": "tool", "tool_call_id": "call_1", "content": "only one response"},
        broken["messages"][-1],
    ]
    assert BLOCKER_UNPAIRED_TOOL_CALL in validate_sample(broken).blockers


def test_a_tool_outside_the_core_set_is_refused():
    """The narrow tool surface is the lever; a sample that uses a seventh tool
    trains a different problem than the one being measured."""
    broken = _sample()
    broken["messages"][2]["tool_calls"] = _calls(("make_choropleth_map", {"dataframe_name": "g"}))
    assert BLOCKER_UNKNOWN_TOOL in validate_sample(broken).blockers


def test_arguments_must_be_an_object():
    broken = _sample()
    broken["messages"][2]["tool_calls"] = [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_unique_values", "arguments": "not-an-object"}}]
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(broken).blockers


def test_malformed_function_schema_is_refused_without_crashing():
    broken = _sample()
    broken["messages"][2]["tool_calls"][0]["function"] = "not-an-object"
    verdict = validate_sample(broken)
    assert BLOCKER_BAD_ARGUMENTS in verdict.blockers


def test_arguments_must_match_the_core_tool_schema():
    broken = _sample()
    broken["messages"][2]["tool_calls"][0]["function"]["arguments"]["invented"] = True
    assert BLOCKER_BAD_ARGUMENTS in validate_sample(broken).blockers


def test_model_tool_schema_must_cover_exactly_the_core_tools():
    broken = _sample()
    broken["tools"] = [tool for tool in broken["tools"]
                       if tool["function"]["name"] != "load_data"]
    assert BLOCKER_BAD_TOOL_SCHEMA in validate_sample(broken).blockers


def test_tool_call_ids_must_be_nonempty_and_unique():
    broken = _sample()
    broken["messages"][2]["tool_calls"][0]["id"] = ""
    assert BLOCKER_BAD_TOOL_CALL_ID in validate_sample(broken).blockers


def test_tool_observation_ids_must_follow_call_order():
    broken = _sample()
    second_call = {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "call_2", "type": "function",
            "function": {"name": "get_unique_values",
                          "arguments": {"dataframe_name": "gdp", "column": "Year"}},
        }],
    }
    broken["messages"] = broken["messages"][:-1] + [
        second_call,
        {"role": "tool", "tool_call_id": "call_2", "content": "year"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_3", "type": "function",
            "function": {"name": "get_unique_values",
                          "arguments": {"dataframe_name": "gdp", "column": "Year"}},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "country"},
        broken["messages"][-1],
    ]
    broken["steps"] = [
        broken["steps"][0],
        {**broken["steps"][0], "call_id": "call_2", "observation_raw": "year",
         "observation_visible": "year"},
    ]
    assert BLOCKER_UNPAIRED_TOOL_CALL in validate_sample(broken).blockers


def test_messages_and_steps_must_describe_the_same_call():
    broken = _sample()
    broken["steps"][0]["tool"] = "load_data"
    assert BLOCKER_BAD_STEP_ALIGNMENT in validate_sample(broken).blockers


def test_sample_versions_are_checked():
    for field in ("schema_version", "renderer_version", "comparator_version",
                  "termination_schema_version"):
        broken = _sample()
        broken[field] = "wrong"
        assert BLOCKER_BAD_VERSION in validate_sample(broken).blockers, field


def test_termination_must_be_machine_parseable():
    """A free-text final answer cannot be scored. The prose may accompany it."""
    broken = _sample()
    broken["messages"] = broken["messages"][:-1] + [
        {"role": "assistant", "content": "I think the answer is GDP per capita."}]
    assert BLOCKER_BAD_TERMINATION in validate_sample(broken).blockers


def test_a_termination_without_status_is_refused():
    broken = _sample()
    broken["messages"] = broken["messages"][:-1] + [
        {"role": "assistant", "content": json.dumps({"selected_column": "x"})}]
    assert BLOCKER_BAD_TERMINATION in validate_sample(broken).blockers


def test_a_failed_call_followed_by_a_recovery_is_valid():
    """Failure traces are required, not tolerated: without them the data cannot
    support a claim about error recovery."""
    recovered = _sample()
    recovered["messages"] = [
        {"role": "system", "content": "You have 6 tools."},
        {"role": "user", "content": "Which column holds GDP per capita?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_unique_values",
                "arguments": {"dataframe_name": "gdp", "column": "GDP"}}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "Error: Column 'GDP' not found in DataFrame"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_2", "type": "function", "function": {
                "name": "get_unique_values",
                "arguments": {"dataframe_name": "gdp",
                              "column": "GDP per capita (current US$)"}}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "['...']"},
        {"role": "assistant", "content": json.dumps(
            {"status": "done", "schema_version": TERMINATION_SCHEMA_VERSION,
             "selected_column": "GDP per capita (current US$)"})},
    ]
    recovered["steps"] = [
        {**recovered["steps"][0],
         "arguments": {"dataframe_name": "gdp", "column": "GDP"},
         "observation_raw": "Error: Column 'GDP' not found in DataFrame",
         "observation_visible": "Error: Column 'GDP' not found in DataFrame",
         "failed": True},
        {**recovered["steps"][0], "call_id": "call_2",
         "arguments": {"dataframe_name": "gdp",
                       "column": "GDP per capita (current US$)"},
         "observation_raw": "['...']", "observation_visible": "['...']"},
    ]
    assert validate_sample(recovered).ok, validate_sample(recovered).blockers


def test_v2_validator_rejects_a_termination_missing_its_schema_version():
    broken = _sample()
    broken["messages"][-1]["content"] = json.dumps(
        {"status": "done", "selected_column": "GDP per capita (current US$)"})
    assert BLOCKER_BAD_TERMINATION in validate_sample(broken).blockers


# --- lineage ------------------------------------------------------------------


def test_every_lineage_field_is_required():
    """A sample without lineage cannot be held out of the reserved cluster, so
    the field is not optional bookkeeping."""
    for field in ("source_lineage", "dataset_snapshot_id", "schema_variant_family",
                  "generator_family", "template_id", "seed"):
        broken = _sample()
        del broken["lineage"][field]
        assert BLOCKER_MISSING_LINEAGE in validate_sample(broken).blockers, field


def test_a_renamed_reserved_task_is_still_refused():
    """The bypass Codex demonstrated on the earlier gate.

    Renaming the task changes the id and nothing else -- the sample still comes
    from the reserved data, so it is still refused.
    """
    renamed = _sample()
    renamed["sample_id"] = "RENAMED_DERIVED"
    renamed["lineage"]["source_lineage"] = f"derived-from:{RESERVED_MARKER}:renamed"
    assert BLOCKER_RESERVED_LINEAGE in validate_sample(renamed).blockers


def test_a_reserved_dataset_snapshot_is_refused_even_with_a_clean_name():
    derived = _sample()
    derived["sample_id"] = "looks-fine"
    derived["lineage"]["dataset_snapshot_id"] = RESERVED_MARKER
    assert BLOCKER_RESERVED_LINEAGE in validate_sample(derived).blockers


def test_lineage_is_read_from_the_nested_field_only():
    """`lineage_of` is the single place that knows where lineage lives, so a
    future format change touches one function instead of every caller."""
    assert lineage_of(_sample())["seed"] == 17
    assert lineage_of({"lineage": None}) == {}


# --- serialization ------------------------------------------------------------


def test_samples_round_trip_through_json():
    """One sample per line, so a build can be appended to and diffed."""
    line = sample_to_json(_sample())
    assert json.loads(line)["sample_id"] == "f1:demo:0001"


def test_the_termination_schema_is_versioned():
    """Compatibility is a claim; a version makes it a checkable one."""
    assert TERMINATION_SCHEMA_VERSION >= 1
