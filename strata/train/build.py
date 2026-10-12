"""Assemble a sample: template -> real execution -> conversation -> validated record.

The pipeline is one direction only, and each stage only sees the previous one:

    template  declares a role and a plan (no observations)
    executor  runs the plan, keeps raw + visible results (never invented)
    reference checks the declaration against the loaded frame
    builder   wires them into one record and validates it

Nothing writes an observation by hand, and nothing re-uses another stage's
conclusion: the reference does not read the template's claim as truth, and the
validator does not read the reference's verdict.

Timing and provenance are carried on the record because a sample whose tools
have since changed is not the sample that was validated -- the digests are what
make that checkable instead of remembered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .comparator import COMPARATOR_VERSION
from .executor import RENDERER_VERSION, Execution, execute_plan
from .reference import check_f1_gold, check_f2_gold
from .sample import (
    CORE_TOOLS,
    SAMPLE_SCHEMA_VERSION,
    TERMINATION_SCHEMA_VERSION,
    validate_sample,
)
from .templates import (
    F1Case,
    build_f1_plan,
    build_f2_plan,
    f1_termination,
    f2_termination,
)


def digest_of_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of_frame(frame) -> str:
    """A snapshot id for the loaded frame: schema, shape, and content.

    Content is included because schema and shape alone are **not** identifying
    here. Every World Bank indicator in this benchmark arrives as the same
    266x69 wide table with the same column names, so "GDP per capita" and
    "Forest area" hashed identically -- two different datasets sharing one
    snapshot id. That silently collapses the group split (the same data lands in
    train and validation) and weakens lineage-based refusal.

    Geometry is excluded: it is the one column whose hashing is both slow and
    sensitive to representation rather than content.
    """
    columns = [str(c) for c in getattr(frame, "columns", [])]
    shape = getattr(frame, "shape", ("?", "?"))
    try:
        import pandas as pd

        values = frame.drop(columns=["geometry"], errors="ignore")
        content = str(int(pd.util.hash_pandas_object(values, index=False).sum()))
    except Exception:  # noqa: BLE001 - a missing digest must not lose the frame
        content = "unavailable"

    return digest_of_text(json.dumps(
        {"columns": columns, "shape": list(shape), "content": content},
        ensure_ascii=False))


def digest_of_tools(geobenchx_root: Path | None = None) -> str:
    root = geobenchx_root or (Path(__file__).resolve().parents[2].parent / "GeoBenchX")
    target = Path(root) / "geobenchx" / "tools.py"
    try:
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return "sha256:unavailable"


def digest_of_runtime(source_kind: str = "geobenchx_catalog",
                      geobenchx_root: Path | None = None,
                      generator_module: str = "strata.train.synth") -> str:
    """Digest the executable sources used by a sample's tool profile."""
    root = geobenchx_root or (Path(__file__).resolve().parents[2].parent / "GeoBenchX")
    paths = [
        Path(root) / "geobenchx" / "tools.py",
        Path(__file__).resolve().parents[1] / "geobench" / "tools.py",
    ]
    if source_kind in ("synthetic_generator", "real_source_snapshot"):
        paths.append(Path(__file__).with_name("synth.py"))
        paths.append(Path(__file__).with_name("slim_tools.py"))
        generator_sources = {
            "strata.train.synth": Path(__file__).with_name("synth.py"),
            "strata.train.synth_families": Path(__file__).with_name("synth_families.py"),
            "strata.train.shape_ood": Path(__file__).with_name("shape_ood.py"),
            "strata.train.real_data": Path(__file__).with_name("real_data.py"),
        }
        generator_path = generator_sources.get(generator_module)
        if generator_path is None:
            return "sha256:unavailable"
        if generator_path not in paths:
            paths.append(generator_path)

    digest = hashlib.sha256()
    try:
        for path in paths:
            digest.update(str(path.name).encode("utf-8"))
            digest.update(path.read_bytes())
    except OSError:
        return "sha256:unavailable"
    return "sha256:" + digest.hexdigest()


def tool_manifest(tools: dict, names=None) -> str:
    """The tool surface as the model is told about it.

    Names, what each does, and its parameter names -- not the full JSON schema,
    which runs to thousands of characters for six tools and would crowd out the
    thing being trained.

    `names` defaults to the tools actually present. A workspace with a
    different surface therefore gets a manifest that matches the schemas and
    executor behind the sample rather than the benchmark profile this module
    happens to know about.
    """
    lines = ["You have exactly these tools:"]
    for name in sorted(tools if names is None else names):
        tool = tools.get(name)
        if tool is None:
            continue
        params = ", ".join(sorted((tool.parameters or {}).get("properties", {})))
        lines.append(f"- {name}({params}): {(tool.description or '').strip().splitlines()[0]}")
    lines.append("Call one tool at a time and read its result before deciding what to do next.")
    # The contract is published in full. v1 told the model only that the final
    # object needs a `status`, while the scorer also required `selected_column`,
    # `join_keys` and `schema_version` -- so a format failure could be the
    # model failing to guess a schema nobody showed it. A terminated run must
    # not be attributable to a hidden requirement.
    lines.append(
        "Finish with a JSON object. Two shapes are accepted:\n"
        '  success: {"status": "done", "schema_version": 2, '
        '"selected_column": "<exact column name>"}   -- for a column question\n'
        '  success: {"status": "done", "schema_version": 2, '
        '"join_keys": ["<stats column>", "<geo column>"]}   -- for a join task\n'
        '  failure: {"status": "error", "schema_version": 2, '
        '"error_code": "<short code>", "error_message": "<brief reason>"}\n'
        "Use the fields the task asks for; the values must be column names that "
        "appear in a frame you have observed. A tool error is an observation to "
        "react to, not a reason to stop -- only use the error shape when you "
        "cannot proceed.")
    return "\n".join(lines)


def tool_schemas(tools: dict) -> list[dict]:
    """Return the exact JSON tool schemas exposed to the model.

    Iterates the tools actually present rather than the module-level constant:
    a workspace whose surface differs (the synthetic one swaps in its own
    loader) must declare **its** tools, or the record would describe a surface
    the model was never given.
    """
    schemas = []
    for name in sorted(tools):
        tool = tools.get(name)
        if tool is None:
            continue
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (tool.description or "").strip(),
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        })
    return schemas


def _assemble(case, plan: list[tuple[str, dict]], termination: dict, gold_fn,
              family: str, generator_family: str, frames: list[str], tools: dict,
              state: dict, seed: int, geobenchx_root: Path | None,
              source_kind: str = "geobenchx_catalog",
              generator_module: str = "strata.train.synth",
              generator_version: str = "synth/v1",
              lineage_extra: dict | None = None) -> dict:
    """Run the plan and wire the result into a record.

    Shared by F1 and F2 because the shape is the same and the differences are
    data, not logic. The one thing it must never do is accept an observation
    from a caller -- everything the model reads comes back from `execute_plan`.

    `gold_fn` is called *after* the plan runs, with the data store, because part
    of the gold is only knowable from the loaded data: the scorer has to know
    which columns the frame actually has, or it cannot tell a column that is
    absent from one that is present and means something else.
    """
    execution: Execution = execute_plan(plan, tools)

    store = state.get("data_store") or {}
    digests = [digest_of_frame(store.get(name)) for name in frames if store.get(name) is not None]
    dataset_digest = digest_of_text("|".join(digests)) if digests else ""
    source_datasets = sorted({
        str(value)
        for _tool, arguments in plan
        for key in ("dataset", "geodataset")
        for value in [arguments.get(key)]
        if value
    })
    # A generated table is named by its handle, and the handle is what a reader
    # needs to regenerate it -- without this the provenance of a synthetic
    # sample would say "some table" and nothing more.
    synth_handles = sorted({
        str(arguments["handle"])
        for _tool, arguments in plan
        if arguments.get("handle")
    })

    source_provenance = {
        "source_kind": source_kind,
        "dataset_names": source_datasets + synth_handles,
    }
    if source_kind in ("synthetic_generator", "real_source_snapshot"):
        tables = []
        for attribute in ("table", "data_table", "geo_table"):
            table = getattr(case, attribute, None)
            if table is not None and hasattr(table, "as_dict"):
                tables.append(table.as_dict())
        if tables:
            source_provenance["generator"] = {
                "module": generator_module,
                "version": generator_version,
                "specs": tables,
            }

    messages: list[dict] = [
        {"role": "system", "content": tool_manifest(tools)},
        {"role": "user", "content": case.question},
    ]
    for step in execution.steps:
        messages.extend(step.as_messages())
    messages.append({"role": "assistant",
                     "content": json.dumps(termination, ensure_ascii=False)})

    return {
        "sample_id": f"{family.lower()}:{case.case_id}",
        "family": family,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "task_text": case.question,
        "source": "template",
        "source_provenance": source_provenance,
        "tool_registry": {
            "profile": source_kind,
            "tool_names": sorted(tools),
        },
        "tools": tool_schemas(tools),
        "lineage": {
            "source_lineage": case.source_lineage,
            "dataset_snapshot_id": dataset_digest,
            "schema_variant_family": case.variant_family,
            "generator_family": generator_family,
            "template_id": case.case_id,
            "seed": seed,
            # Provenance only. It must not reach system/user/messages: a trace
            # whose prompt names the error type it is about to make has handed
            # the model the answer.
            **(lineage_extra or {}),
        },
        "renderer_version": RENDERER_VERSION,
        "comparator_version": COMPARATOR_VERSION,
        "termination_schema_version": TERMINATION_SCHEMA_VERSION,
        "tool_source_digest": digest_of_runtime(
            source_kind, geobenchx_root, generator_module=generator_module),
        "dataset_digest": dataset_digest,
        "gold": gold_fn(store),
        "steps": execution.as_records(),
        "messages": messages,
    }


def build_f1_sample(case: F1Case, tools: dict, state: dict,
                    seed: int = 0, geobenchx_root: Path | None = None) -> dict:
    """One F1 sample, executed for real and checked before it is returned."""
    return _assemble(
        case, build_f1_plan(case), f1_termination(case),
        lambda store: {
            "role": case.role, "column": case.correct_column,
            "frame_columns": list(store.get(case.frame).columns
                                   if store.get(case.frame) is not None else []),
        },
        family="F1", generator_family="f1_column_role", frames=[case.frame],
        tools=tools, state=state, seed=seed, geobenchx_root=geobenchx_root)


def build_f2_sample(case, tools: dict, state: dict,
                    seed: int = 0, geobenchx_root: Path | None = None) -> dict:
    """One F2 sample: both sides loaded, a wrong pair tried for real, then the right one."""
    return _assemble(
        case, build_f2_plan(case), f2_termination(case),
        lambda store: {"join_keys": [case.data_key, case.geo_key],
                       "value_column": case.value_column},
        family="F2", generator_family="f2_join_key",
        frames=[case.data_frame, case.geo_frame],
        tools=tools, state=state, seed=seed, geobenchx_root=geobenchx_root)


def fresh_tools() -> tuple[dict, dict]:
    """A new state and the core tool set bound to it.

    One environment per sample, never shared: these tools write into a common
    `data_store`, so a frame left behind by the previous case is exactly the
    kind of cross-contamination that makes a batch look cleaner than it is.
    """
    from ..geobench.tools import benchmark_state, build_geobench_tools

    state = benchmark_state()
    tools = {t.name: t for t in build_geobench_tools(state=state,
                                                    only=tuple(sorted(CORE_TOOLS)))}
    return tools, state


def _build_batch(cases, checker, builder, geobenchx_root, tools_factory) -> tuple[list[dict], list[dict]]:
    """Shared shape for F1 and F2: fresh environment, check gold, build, validate.

    A case whose gold fails the reference check or whose record fails validation
    is **still built and still reported** -- a batch that silently drops its bad
    rows reports a clean pass rate over an unknown denominator.
    """
    samples: list[dict] = []
    report: list[dict] = []

    for seed, case in enumerate(cases):
        tools, state = tools_factory()
        gold = checker(case, tools, state)
        sample = builder(case, tools, state, seed=seed, geobenchx_root=geobenchx_root)
        verdict = validate_sample(sample)
        row = {
            "sample_id": sample["sample_id"],
            "gold_ok": gold.ok,
            "gold_problems": gold.problems,
            "validate_ok": verdict.ok,
            "blockers": verdict.blockers,
            "failed_steps": sum(1 for s in sample["steps"] if s["failed"]),
        }
        row.update(gold.as_dict())
        report.append(row)
        samples.append(sample)

    return samples, report


def build_f1_batch(cases: list[F1Case], geobenchx_root: Path | None = None,
                   tools_factory=fresh_tools) -> tuple[list[dict], list[dict]]:
    return _build_batch(cases, check_f1_gold, build_f1_sample, geobenchx_root, tools_factory)


def build_f2_batch(cases: list, geobenchx_root: Path | None = None,
                   tools_factory=fresh_tools) -> tuple[list[dict], list[dict]]:
    return _build_batch(cases, check_f2_gold, build_f2_sample, geobenchx_root, tools_factory)
