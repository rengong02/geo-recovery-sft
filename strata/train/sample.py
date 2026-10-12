"""What a training sample is, and the checks that make it one.

A sample is a **complete tool-call conversation**: a system prompt with the tool
schemas, a user task, then alternating assistant `tool_call`s and their `tool`
observations, ending in a structured termination. Nothing here is a "plan" --
a plan is what the model intends, and a plan with no observation is not
trainable, because the observation is the part that cannot be recovered later
without inventing it.

This module is deliberately the first thing built. The format is the piece that
cannot be retrofitted: the p2c audit produced 18 records shaped like
trajectories that had calls and no results, and no amount of downstream work
could turn them into samples.

Two properties are worth calling out because they were found the hard way:

* **Refusal is by lineage, not by name.** Matching a reserved task on its id
  string means a renamed record passes -- Codex demonstrated exactly that on the
  earlier gate. Where the data came from does not change when you rename the
  task, so that is what gets checked.
* **Every blocker is reported, not just the first.** Fixing a sample one
  refusal at a time is slower than seeing the whole list.

The validator is pure: it reads a mapping and returns a verdict, and it never
raises on malformed input, because a validator that crashes on bad data cannot
be used to triage a batch of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..provenance import RESERVED_TASK_IDS
from .comparator import COMPARATOR_VERSION
from .executor import RENDERER_VERSION

#: The tools the first batch (F1 field grounding, F2 join-key alignment) trains
#: on. Deliberately six: a wide tool surface is itself a failure cause for small
#: models, which is the whole reason this project narrows it. F5 adds
#: `make_choropleth_map` on top of these, for seven.
CORE_TOOLS: frozenset[str] = frozenset({
    "load_data",
    "load_geodata",
    "get_unique_values",
    "filter_categorical",
    "merge_dataframes",
    "calculate_column_statistics",
})

CORE_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "load_data": (frozenset({"dataset", "output_dataframe_name"}), frozenset()),
    "load_geodata": (frozenset({"geodataset", "output_geodataframe_name"}), frozenset()),
    "get_unique_values": (frozenset({"dataframe_name", "column"}), frozenset()),
    "filter_categorical": (
        frozenset({"dataframe_name", "filters", "output_dataframe_name"}),
        frozenset(),
    ),
    "merge_dataframes": (
        frozenset({"dataframe_name", "geodataframe_name", "statkey", "geokey",
                   "output_dataframe_name"}),
        frozenset(),
    ),
    "calculate_column_statistics": (
        frozenset({"dataframe_name", "column_name", "output_variable_name"}),
        frozenset({"include_quantiles", "additional_quantiles"}),
    ),
}

SYNTHETIC_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "load_synthetic_table": (
        frozenset({"handle", "output_dataframe_name"}), frozenset()),
}

#: The real-data line's loader. Same signature as the generated-table one because
#: the two play the same role -- name a table, get it into the store -- and
#: keeping the shape identical means the surface a real sample declares is judged
#: by the same rule as a synthetic one rather than by a special case.
REAL_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "load_real_table": (
        frozenset({"handle", "output_dataframe_name"}), frozenset()),
}

EXTENDED_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "make_choropleth_map": (
        frozenset({"dataframe_name", "mappingkey", "legendtext", "colormap"}),
        frozenset()),
}

REGISTERED_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    **CORE_TOOL_ARGUMENTS,
    **SYNTHETIC_TOOL_ARGUMENTS,
    **REAL_TOOL_ARGUMENTS,
    **EXTENDED_TOOL_ARGUMENTS,
}

#: Markers whose presence in a sample's lineage means the sample is derived from
#: the reserved evaluation cluster and may not be trained on. The nine task ids
#: come from the export gate so there is one list, not two that drift.
#:
#: Checked as a **substring of `source_lineage`** and as an **exact
#: `dataset_snapshot_id`**, so a derived record is caught even when its own name
#: says nothing about where it came from.
RESERVED_LINEAGE_MARKERS: frozenset[str] = frozenset(RESERVED_TASK_IDS) | {
    "geobenchx:benchmark_set",
}

#: The termination payload schema. Versioned because "compatible" is a claim,
#: and a version makes it a checkable one.
TERMINATION_SCHEMA_VERSION = 2
SAMPLE_SCHEMA_VERSION = 2

LINEAGE_FIELDS = (
    "source_lineage",
    "dataset_snapshot_id",
    "schema_variant_family",
    "generator_family",
    "template_id",
    "seed",
)

BLOCKER_BAD_ROLE_ORDER = "bad_role_order"
BLOCKER_UNPAIRED_TOOL_CALL = "unpaired_tool_call"
BLOCKER_UNKNOWN_TOOL = "unknown_tool"
#: The recovery error kinds a contract-violating call may legitimately be.
#: Declared here rather than imported from `synth` because `synth` imports this
#: module -- and because the validator must not treat "some non-empty string" as
#: a licence: a forged marker would buy a free exemption.
RECOVERY_ERROR_KINDS = frozenset({"wrong_role", "value_as_key", "missing_argument"})

BLOCKER_BAD_ARGUMENTS = "bad_arguments"
BLOCKER_BAD_TERMINATION = "bad_termination"
BLOCKER_BAD_TOOL_CALL_ID = "bad_tool_call_id"
BLOCKER_BAD_STEP_ALIGNMENT = "bad_step_alignment"
BLOCKER_BAD_VERSION = "bad_version"
BLOCKER_BAD_TOOL_SCHEMA = "bad_tool_schema"
BLOCKER_MISSING_LINEAGE = "missing_lineage"
BLOCKER_RESERVED_LINEAGE = "reserved_lineage"


@dataclass
class Verdict:
    sample_id: str
    ok: bool
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "ok": self.ok, "blockers": self.blockers}


def lineage_of(record: dict) -> dict:
    """The one place that knows where lineage lives.

    Keep it that way: when the format moves, one function changes instead of
    every reader.
    """
    lineage = record.get("lineage")
    return lineage if isinstance(lineage, dict) else {}


# --- structure ----------------------------------------------------------------


def _role_order_ok(messages: list) -> bool:
    """`system`, `user`, then (assistant-with-calls, tool) pairs, then a plain
    assistant. Anything else is a fragment, not a conversation."""
    if len(messages) < 3:
        return False
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return False

    index = 2
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            return False
        calls = message.get("tool_calls")
        if index == len(messages) - 1:
            return not calls
        if not isinstance(calls, list) or len(calls) != 1:
            return False
        if messages[index + 1].get("role") != "tool":
            return False
        if not isinstance(messages[index + 1].get("content"), str):
            return False
        index += 2
    return False


def _tool_call_ids(messages: list) -> tuple[list[str], list[str], bool]:
    called: list[str] = []
    observed: list[str] = []
    invalid = False
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        invalid = True
                    else:
                        called.append(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                invalid = True
            else:
                observed.append(call_id)
    if len(called) != len(set(called)) or len(observed) != len(set(observed)):
        invalid = True
    return called, observed, invalid


def _iter_calls(messages: list):
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    yield call


def declared_arguments(record: dict) -> dict[str, tuple[set[str], set[str]]]:
    """required/optional per tool, **as the record declares them**.

    Read from the sample's own tool schemas rather than from a module-level
    table. The table is still consulted -- for the tools it knows, a declaration
    that disagrees with the real contract is a defect and is refused -- but a
    workspace may expose tools the table has never heard of, and refusing those
    outright made every synthetic sample invalid.
    """
    contracts: dict[str, tuple[set[str], set[str]]] = {}
    for schema in record.get("tools") or []:
        if not isinstance(schema, dict):
            continue
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        required = parameters.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            continue
        contracts[name] = (set(required), set(properties) - set(required))
    return contracts


def _arguments_ok(call: dict, contracts: dict[str, tuple[set[str], set[str]]]) -> bool:
    if call.get("type") != "function":
        return False
    function = call.get("function")
    if not isinstance(function, dict):
        return False
    name = function.get("name")
    arguments = function.get("arguments")
    if name not in contracts or not isinstance(arguments, dict):
        return False
    required, optional = contracts[name]
    keys = set(arguments)
    return required <= keys <= required | optional


def _steps_align(record: dict, messages: list) -> bool:
    steps = record.get("steps")
    if not isinstance(steps, list):
        return False

    pairs: list[tuple[dict, dict]] = []
    index = 2
    while index + 1 < len(messages):
        assistant = messages[index]
        tool = messages[index + 1]
        if assistant.get("role") != "assistant" or tool.get("role") != "tool":
            break
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            break
        pairs.append((calls[0], tool))
        index += 2

    if len(steps) != len(pairs):
        return False
    for step, (call, tool_message) in zip(steps, pairs):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(step, dict) or not isinstance(function, dict):
            return False
        if step.get("call_id") != call.get("id"):
            return False
        if step.get("tool") != function.get("name"):
            return False
        if step.get("arguments") != function.get("arguments"):
            return False
        if not isinstance(step.get("observation_raw"), str):
            return False
        if step.get("observation_visible") != tool_message.get("content"):
            return False
        if not isinstance(step.get("failed"), bool):
            return False
        seconds = step.get("seconds")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return False
    return True


def _versions_ok(record: dict) -> bool:
    return all(record.get(field) == expected for field, expected in {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "comparator_version": COMPARATOR_VERSION,
        "termination_schema_version": TERMINATION_SCHEMA_VERSION,
    }.items())


def _tool_schema_ok(record: dict) -> bool:
    """Is the declared tool surface well-formed, and honest about known tools?"""
    schemas = record.get("tools")
    if not isinstance(schemas, list) or not schemas:
        return False

    seen: set[str] = set()
    for schema in schemas:
        if not isinstance(schema, dict) or schema.get("type") != "function":
            return False
        function = schema.get("function")
        if not isinstance(function, dict):
            return False
        name = function.get("name")
        if not isinstance(name, str) or name in seen:
            return False
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return False
        properties = parameters.get("properties")
        declared_required = parameters.get("required")
        if not isinstance(properties, dict) or not isinstance(declared_required, list):
            return False

        # A sample may use a different profile, but it cannot invent the
        # executor behind that profile. New tools must be added to this
        # registry before they can enter an export.
        if name not in REGISTERED_TOOL_ARGUMENTS:
            return False
        required, optional = REGISTERED_TOOL_ARGUMENTS[name]
        if set(properties) != required | optional or set(declared_required) != required:
            return False
        seen.add(name)

    # A surface drawn only from the benchmark profile must be the whole profile:
    # a record that silently declares five of the six would validate its own
    # calls while claiming a surface the model was never given.
    if seen <= CORE_TOOLS:
        return seen == CORE_TOOLS
    return True


def _termination_ok(messages: list, record: dict | None = None) -> bool:
    # An empty transcript is malformed, not an exception. This validator exists
    # to triage bad records, and one that raises on the emptiest possible input
    # cannot be pointed at a batch.
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return False
    content = last.get("content")
    if not isinstance(content, str):
        return False
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return False
    if record and record.get("termination_schema_version", record.get("schema_version", 1)) == 2:
        from .termination import valid_v2_termination

        family = str(record.get("family") or ("F2" if "join_keys" in (record.get("gold") or {}) else "F1"))
        return valid_v2_termination(payload, family)
    return isinstance(payload, dict) and isinstance(payload.get("status"), str)


# --- lineage ------------------------------------------------------------------


def missing_lineage(record: dict) -> list[str]:
    lineage = lineage_of(record)
    return [name for name in LINEAGE_FIELDS
            if not str(lineage.get(name, "")).strip()]


def reserved_hits(record: dict, markers=frozenset(RESERVED_LINEAGE_MARKERS)) -> list[str]:
    """Which reserved markers this sample traces back to.

    Substring on `source_lineage` so a derived-and-renamed record is caught,
    exact match on `dataset_snapshot_id` because that field is an identifier
    rather than a sentence.
    """
    lineage = lineage_of(record)
    source = str(lineage.get("source_lineage") or "")
    snapshot = str(lineage.get("dataset_snapshot_id") or "")
    return sorted(m for m in markers if m in source or m == snapshot)


# --- the verdict --------------------------------------------------------------


def allowed_tools(record: dict) -> frozenset[str]:
    """The tool surface this sample declares.

    Read from the record rather than taken from `CORE_TOOLS`, because a
    workspace may legitimately expose a different set -- the synthetic one
    replaces the two catalog loaders with a single generated-table loader.
    Hardcoding the benchmark's six made every synthetic sample fail as
    `unknown_tool`, which was the check working and the interface being wrong.

    A record that declares nothing falls back to the benchmark six, so older
    artifacts keep being checked against the surface they were built with.
    """
    declared = record.get("tools")
    if isinstance(declared, list):
        names = {str((t.get("function") or {}).get("name"))
                 for t in declared if isinstance(t, dict)}
        names.discard("None")
        if names:
            return frozenset(names)
    return CORE_TOOLS


def validate_sample(record: Any, markers=frozenset(RESERVED_LINEAGE_MARKERS)) -> Verdict:
    sample_id = str(record.get("sample_id", "?")) if isinstance(record, dict) else "?"

    if not isinstance(record, dict):
        return Verdict(sample_id=sample_id, ok=False, blockers=[BLOCKER_BAD_ROLE_ORDER])

    messages = record.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        messages = []

    blockers: list[str] = []

    if not _role_order_ok(messages):
        blockers.append(BLOCKER_BAD_ROLE_ORDER)

    called, observed, invalid_ids = _tool_call_ids(messages)
    if invalid_ids:
        blockers.append(BLOCKER_BAD_TOOL_CALL_ID)
    if called != observed:
        blockers.append(BLOCKER_UNPAIRED_TOOL_CALL)

    # Provenance only; it never reaches the prompt. Restricted to the known
    # kinds -- anything else is not a declaration, it is a forgery.
    recovery_kind = str((lineage_of(record) or {}).get("recovery_error_kind") or "")
    if recovery_kind not in RECOVERY_ERROR_KINDS:
        recovery_kind = ""
    # The one call that may break the contract is the one the trace recorded as
    # failing. Without this the allowance could forgive a *different* call, and
    # the trace would contain a violation nobody meant to teach.
    failed_call_ids = {str(step.get("call_id"))
                       for step in (record.get("steps") or [])
                       if isinstance(step, dict) and step.get("failed")}
    if len(failed_call_ids) != 1:
        recovery_kind = ""
    contract_violations = 0
    for call in _iter_calls(messages):
        raw_function = call.get("function")
        if not isinstance(raw_function, dict):
            blockers.append(BLOCKER_BAD_ARGUMENTS)
            continue
        function = raw_function
        name = function.get("name")
        if name not in allowed_tools(record) or name not in REGISTERED_TOOL_ARGUMENTS:
            blockers.append(BLOCKER_UNKNOWN_TOOL)
            continue
        required, optional = REGISTERED_TOOL_ARGUMENTS[name]
        if not _arguments_ok(call, {name: (set(required), set(optional))}):
            # A recovery trace contains **one** call that deliberately breaks the
            # contract -- that call *is* the error it teaches from, and removing
            # it would remove the lesson. The relaxation is bounded: exactly one
            # such call, and only when the sample's provenance declares itself a
            # recovery trace. Anything else still fails, and the count is
            # asserted so a second violation cannot hide inside the allowance.
            if (recovery_kind and contract_violations < 1
                    and str(call.get("id")) in failed_call_ids):
                contract_violations += 1
                continue
            blockers.append(BLOCKER_BAD_ARGUMENTS)

    if not _termination_ok(messages, record):
        blockers.append(BLOCKER_BAD_TERMINATION)

    if not _versions_ok(record):
        blockers.append(BLOCKER_BAD_VERSION)

    if not _tool_schema_ok(record):
        blockers.append(BLOCKER_BAD_TOOL_SCHEMA)

    if not _steps_align(record, messages):
        blockers.append(BLOCKER_BAD_STEP_ALIGNMENT)

    if missing_lineage(record):
        blockers.append(BLOCKER_MISSING_LINEAGE)

    if reserved_hits(record, markers):
        blockers.append(BLOCKER_RESERVED_LINEAGE)

    # Deduplicated but order-preserving: a sample with three bad calls should
    # say "unknown_tool" once, not three times.
    seen: list[str] = []
    for blocker in blockers:
        if blocker not in seen:
            seen.append(blocker)
    return Verdict(sample_id=sample_id, ok=not seen, blockers=seen)


def sample_to_json(record: dict) -> str:
    """One sample per line: appendable, diffable, and greppable."""
    return json.dumps(record, ensure_ascii=False)
