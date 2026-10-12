"""Replay a sample's calls against the current tools and compare the results.

This is the adapter the design called for: the existing
`scripts/replay_trajectory.py` reads the old JSONL `steps` field, and a sample's
chat messages cannot be fed to it. Same idea, different input.

The two checks answer different questions and both are needed:

* **the tool digest** says *something about the tools changed*;
* **the comparator** says *whether that change mattered to this sample*.

Reporting only the first would fail every sample on a comment change. Reporting
only the second would let a real behavioural change through whenever the
normalisation happened to absorb it. Together they separate "the source moved"
from "the data is stale", which is the distinction that decides whether to
rebuild or to ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .build import COMPARATOR_VERSION, digest_of_runtime, fresh_tools
from .comparator import compare_executions
from .executor import execute_plan


@dataclass
class ReplayReport:
    sample_id: str
    tool_digest_before: str
    tool_digest_now: str
    steps_replayed: int
    comparison: dict

    @property
    def tools_changed(self) -> bool:
        return self.tool_digest_before != self.tool_digest_now

    @property
    def stale(self) -> bool:
        """The tools moved **and** it changed what this sample sees.

        A digest change alone is not staleness: this project has been bitten by
        treating "the file hash moved" as "the data is wrong".
        """
        return self.tools_changed and not self.comparison.get("equal", False)

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "comparator_version": COMPARATOR_VERSION,
            "tool_digest_before": self.tool_digest_before,
            "tool_digest_now": self.tool_digest_now,
            "tools_changed": self.tools_changed,
            "steps_replayed": self.steps_replayed,
            "equal": bool(self.comparison.get("equal")),
            "stale": self.stale,
            "comparison": self.comparison,
        }


def planned_calls(sample: dict) -> list[tuple[str, dict]]:
    """The calls in the order the conversation makes them.

    Read from the messages rather than from a separate plan field, so there is
    one source of truth: if the two could disagree, the sample would be
    ambiguous about what it is asking the model to do.
    """
    calls: list[tuple[str, dict]] = []
    for message in sample.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            calls.append((str(function.get("name")), function.get("arguments") or {}))
    return calls


def _synthetic_tools_factory(sample: dict):
    from ..geobench.tools import benchmark_state
    from .synth import SynthTableSpec, fresh_synth_tools

    generator = (sample.get("source_provenance") or {}).get("generator") or {}
    raw_specs = generator.get("specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("synthetic sample is missing generator specs")

    specs = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise TypeError("synthetic generator spec is not an object")
        specs.append(SynthTableSpec(
            handle=str(raw_spec["handle"]),
            columns=tuple(str(column) for column in raw_spec["columns"]),
            n_rows=int(raw_spec["n_rows"]),
            seed=int(raw_spec["seed"]),
            kind=str(raw_spec.get("kind", "stats")),
            id_column=raw_spec.get("id_column"),
            variant_family=str(raw_spec.get("variant_family", "synth/v1")),
            entity_namespace=raw_spec.get("entity_namespace"),
        ))

    def factory():
        state = benchmark_state()
        return fresh_synth_tools(specs, state), state

    return factory


def _real_tools_factory(sample: dict, sources_dir=None):
    """Rebuild the real-data workspace from the spec the sample carries.

    The snapshots are read from disk and their hashes re-checked, so a replay
    against a different snapshot fails here rather than quietly comparing two
    different datasets.
    """
    from ..geobench.tools import benchmark_state
    from .real_data import DEFAULT_SOURCES_DIR, RealTableSpec, fresh_real_tools

    generator = (sample.get("source_provenance") or {}).get("generator") or {}
    raw_specs = generator.get("specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("real-data sample is missing its source specs")
    specs = [RealTableSpec.from_dict(raw) for raw in raw_specs]
    directory = Path(sources_dir) if sources_dir is not None else DEFAULT_SOURCES_DIR

    def factory():
        state = benchmark_state()
        return fresh_real_tools(specs, state, directory), state

    return factory


def _default_tools_factory(sample: dict):
    source_kind = (sample.get("source_provenance") or {}).get("source_kind")
    if source_kind == "synthetic_generator":
        return _synthetic_tools_factory(sample)
    if source_kind == "real_source_snapshot":
        return _real_tools_factory(sample)
    return fresh_tools


def replay_sample(sample: dict, tools_factory=None,
                  geobenchx_root=None, sources_dir=None) -> ReplayReport:
    calls = planned_calls(sample)
    if tools_factory is not None:
        factory = tools_factory
    elif (sample.get("source_provenance") or {}).get("source_kind") == "real_source_snapshot":
        factory = _real_tools_factory(sample, sources_dir)
    else:
        factory = _default_tools_factory(sample)
    tools, _state = factory()
    execution = execute_plan(calls, tools)

    comparison = compare_executions(sample.get("steps") or [], execution.as_records())
    source_provenance = sample.get("source_provenance") or {}
    generator = source_provenance.get("generator") or {}
    return ReplayReport(
        sample_id=str(sample.get("sample_id", "?")),
        tool_digest_before=str(sample.get("tool_source_digest", "")),
        tool_digest_now=digest_of_runtime(
            (sample.get("source_provenance") or {}).get("source_kind",
             "geobenchx_catalog"), geobenchx_root,
            generator_module=str(generator.get("module", "strata.train.synth"))),
        steps_replayed=len(execution.steps),
        comparison=comparison,
    )


def replay_all(samples: list[dict], tools_factory=None,
               geobenchx_root=None, sources_dir=None) -> dict:
    reports = [replay_sample(s, tools_factory, geobenchx_root, sources_dir)
               for s in samples]
    return {
        "samples": len(reports),
        "replayed_equal": sum(1 for r in reports if r.comparison.get("equal")),
        "tools_changed": sum(1 for r in reports if r.tools_changed),
        "stale": sum(1 for r in reports if r.stale),
        "reports": [r.as_dict() for r in reports],
    }
