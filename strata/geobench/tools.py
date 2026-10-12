"""GeoBenchX's tools, re-exposed as CoreCoder tools.

Why an adapter rather than "just point MCP at them": GeoBenchX's tools are
LangChain `StructuredTool`s whose `state` parameter is a LangGraph state dict
(`data_store` / `image_store` / `html_store`). CoreCoder has no such state, so
something has to hold it and hand it in on every call. That wrapper is this
module.

Two details that would silently break a run if missed:

1. **This GeoBenchX version publishes `state` in its tool schema.** Its
   `args_schema` lists `state` as a *required* property, and CoreCoder's
   `Agent._tool_schemas()` passes `Tool.parameters` straight through, so nothing
   downstream would filter it. Left as-is, the model would be asked to invent a
   value for a framework-internal parameter. It is stripped here. (Stated as a
   fact about this version's tools, not about LangChain in general.)
2. **The score is a judgement of the tool plan, not of the GIS result.**
   `score_task_solution()` serialises the candidate and the reference into
   tool-call text and hands *that* to an LLM judge; the judge never executes a
   tool and never sees a return value, a DataFrame, an image or the final
   natural-language answer. So the metric is "does this trajectory match the
   reference plan", not "was the geography right". The tools still have to
   *run* -- that is what proves the state hand-off and argument schemas work --
   but their output text only matters to the model.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corecoder.tools.base import Tool

from .evidence import (
    DEFAULT_SCHEMA_PREVIEW,
    EvidenceStore,
    classify,
    render,
    render_image,
    summarize,
    summarize_image,
    validate_schema_preview,
)


def geobenchx_root() -> Path:
    """Locate the GeoBenchX checkout.

    `GEOBENCHX_ROOT` wins; otherwise a sibling of this repo, which is where the
    project keeps it (`work/CoreCoder` and `work/GeoBenchX`).
    """
    import os

    env = os.getenv("GEOBENCHX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "GeoBenchX"


def _catalog_root(root: Path) -> Path:
    """Find GeoBenchX's data directory on case-sensitive and case-insensitive OSes."""
    for name in ("Data", "data"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / "Data"


def load_geobenchx_tools_module(root: Path | None = None):
    """Import `geobenchx.tools`, adding the checkout to `sys.path` if needed.

    GeoBenchX is not installed as a package, so this is how we reach it. The
    import is deliberately local: GeoBenchX needs GDAL and its datasets, and
    nothing else in CoreCoder should pay that cost.

    A non-interactive matplotlib backend is forced here, for a reproduced
    failure rather than as a precaution. GeoBenchX picks its backend when
    `tools.py` imports pyplot, and ten of its tools call `plt.show()`. CoreCoder
    runs one round's tool calls on a thread pool (`Agent._exec_tools_parallel`),
    and macOS' interactive backend refuses to make a window off the main thread:
    that is the `Cannot create a GUI FigureManager outside the main thread`
    error seen in the first baseline. Hence `MPLBACKEND` is assigned rather than
    `setdefault`-ed -- a benchmark run that opens a window is not a headless run.
    """
    import os
    import sys

    root = root or geobenchx_root()
    data_root = _catalog_root(root)
    os.environ.setdefault("STATDATAPATH", str(data_root / "StatData"))
    os.environ.setdefault("GEODATAPATH", str(data_root / "GeoData"))
    os.environ.setdefault("SCRATCHPATH", str(root / "scratch"))
    os.environ["MPLBACKEND"] = "Agg"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("geobenchx.tools")
    _use_headless_backend()
    return module


def _use_headless_backend() -> None:
    """Assert the non-interactive backend, in case the environment wanted Agg.

    `MPLBACKEND` is read only when pyplot is first imported, so it does nothing
    in a process that already imported matplotlib -- a test session, say. This
    closes that gap. Idempotent by design: once the backend is Agg there is
    nothing to switch, so a figure drawn mid-run is never torn down.
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


def benchmark_state() -> dict:
    """A fresh GeoBenchX state for one task.

    Fresh, never shared: the state carries the task's DataFrames, so one dict
    handed to two tasks would let the second read the first's data.

    `visualize` is off. GeoBenchX's tools guard ten `plt.show()` calls on it,
    and a benchmark run is headless. The maps are not lost by turning it off --
    each tool base64-encodes its figure into `image_store` *before* the show
    guard -- so a task can still use its own imagery. `_use_headless_backend`
    then covers the case where a tool shows a figure anyway.
    """
    return {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}


#: The 24 tools GeoBenchX wires into its own agent, in its own order. Keeping
#: the list explicit (rather than globbing `*_tool`) means an upstream rename
#: fails loudly here instead of silently shrinking the profile.
GEOBENCH_TOOL_NAMES: tuple[str, ...] = (
    "load_data",
    "load_geodata",
    "make_choropleth_map",
    "make_bivariate_map",
    "merge_dataframes",
    "get_unique_values",
    "filter_categorical",
    "filter_numerical",
    "select_features_by_spatial_relationship",
    "filter_points_by_raster_values",
    "create_buffer",
    "get_raster_path",
    "get_raster_description",
    "get_values_from_raster_with_geometries",
    "analyze_raster_overlap",
    "calculate_line_lengths",
    "calculate_columns",
    "scale_column_by_value",
    "make_heatmap",
    "visualize_geographies",
    "get_centroids",
    "generate_contours_display",
    "reject_task",
    "calculate_column_statistics",
)


def _drop_injected_state(schema: dict) -> dict:
    """Remove `state` from a JSON schema's properties and required list.

    Left in, it is a required parameter the model can never supply correctly.
    """
    schema = {**schema}
    props = dict(schema.get("properties", {}))
    props.pop("state", None)
    schema["properties"] = props
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r != "state"]
    return schema


#: GeoBenchX answers a bad column with `Error: Column 'X' not found in DataFrame`
#: and nothing else, so the model's only strategy is to guess -- one name per
#: call. On one task it spent 21 calls guessing at a 27-column frame and tried
#: `NAME_EN` ninth, which is a column the frame has. Naming the columns it could
#: have asked for turns that into one call.
_COLUMN_NOT_FOUND = re.compile(r"Column '([^']*)' not found")
_FRAME_NOT_FOUND = re.compile(r"(?:GeoDataFrame|DataFrame) '([^']*)' not found")
_ANY_FRAME_MISSING = re.compile(r"not found in data store")

#: Long enough for the World Bank frames (69 columns), short enough that the
#: hint never crowds out the conversation it is trying to rescue.
MAX_HINTED_COLUMNS = 80
MAX_HINTED_FRAMES = 40


def _named_frames(kwargs: dict, store: dict) -> list[str]:
    """The stored frames this call referred to, in argument order.

    `output_*` and `*column*` keys are excluded: one names a frame the call is
    about to create, the other holds a column name that could collide with a
    frame's. Guessing wrong there would be worse than saying nothing.
    """
    seen: list[str] = []
    for key, value in kwargs.items():
        if not key.endswith("_name") or key.startswith("output") or "column" in key:
            continue
        if isinstance(value, str) and value in store and value not in seen:
            seen.append(value)
    return seen


def _with_schema_hint(out: str, kwargs: dict, state: dict) -> str:
    """Append what the call could have asked for, to a not-found error.

    Bounded and factual: the columns of the frames the call named, or the names
    of the frames that do exist. Nothing is inferred and nothing is corrected --
    the model still has to choose, it just stops choosing blind.
    """
    store = state.get("data_store") or {}
    if not isinstance(store, dict) or not store:
        return out

    if _COLUMN_NOT_FOUND.search(out):
        frames = _named_frames(kwargs, store)
        if not frames:
            return out
        lines = []
        for name in frames:
            columns = [str(c) for c in store[name].columns]
            shown = columns[:MAX_HINTED_COLUMNS]
            rest = len(columns) - len(shown)
            lines.append(f"  '{name}': {', '.join(shown)}" + (f"  (+{rest} more)" if rest else ""))
        return out + "\nColumns actually available:\n" + "\n".join(lines)

    if _FRAME_NOT_FOUND.search(out) or _ANY_FRAME_MISSING.search(out):
        names = [str(n) for n in store][:MAX_HINTED_FRAMES]
        rest = len(store) - len(names)
        return out + "\nFrames actually available: " + ", ".join(names) + (f"  (+{rest} more)" if rest else "")

    return out


#: How much of a tool's raw text still reaches the model when evidence is on.
#:
#: A single flat number, deliberately **not** per artifact type. Making it
#: per-type is P2's intervention, and P1a already showed what building an
#: intervention on an unmeasured causal claim costs.
EVIDENCE_TEXT_HEAD = 2_000


class _EvidenceBlock:
    """Turns one call's result into an addressable artifact plus a summary.

    **Only replaces the raw text when the block is smaller than it.** Measured
    first, and the measurement changed this class: with an unconditional block,
    `load_geodata` results went from ~1,800 characters to ~4,700 -- two and a
    half times *worse*, because those results are already short and the block
    added a head of the original on top of a summary. The control arm's whole
    advantage is being cheap.

    So the rule is now self-correcting rather than a threshold somebody has to
    keep calibrated: file the artifact unconditionally, then hand over whichever
    of the two texts is shorter. It cannot be worse than the control by
    construction, and how much better it is becomes the measurable quantity.
    """

    def __init__(self, store, schema_preview: str = DEFAULT_SCHEMA_PREVIEW):
        self.store = store
        self.schema_preview = validate_schema_preview(schema_preview)

    def __call__(self, label: str, raw: str, produced: dict,
                 images: list[dict] | None = None,
                 html_items: list[dict] | None = None) -> str:
        ref = self.store.put(label, raw, "text")
        block = self._render(label, raw, produced, ref, images or [], html_items or [])
        delivered = block if len(block) < len(raw) else raw
        self.store.note(len(raw), len(delivered))
        return delivered

    def _render(self, label: str, raw: str, produced: dict, ref,
                images: list[dict], html_items: list[dict]) -> str:
        lines = [f"[evidence] {label}"]
        for name, value in produced.items():
            artifact = self.store.put(name, _describe(value), classify(value), ".json")
            lines.append(render(
                summarize(value, schema_preview=self.schema_preview), name, artifact
            ))
        for index, item in enumerate(images):
            encoded = item.get("base64", "")
            try:
                payload = base64.b64decode(encoded, validate=False)
            except (binascii.Error, TypeError, ValueError):
                if isinstance(encoded, str):
                    payload = encoded.encode("utf-8")
                elif isinstance(encoded, (bytes, bytearray, memoryview)):
                    payload = bytes(encoded)
                else:
                    payload = repr(encoded).encode("utf-8")
            summary = summarize_image(payload)
            summary["description"] = item.get("description") or f"{label}-image-{index}"
            # Summarised, **not archived**. The design says format, dimensions
            # and hash -- and duplicating the payload cost 543 MB across one
            # nine-task run, because GeoBenchX stores a base64 PNG for every map
            # any tool draws. It also buys nothing for the question at hand: an
            # image was never in the model's context, so keeping a second copy
            # is not evidence about context.
            lines.append(render_image(summary))
        for index, item in enumerate(html_items):
            payload = item.get("html", "")
            if not isinstance(payload, (str, bytes)):
                payload = str(payload)
            name = item.get("description") or f"{label}-html-{index}"
            artifact = self.store.put(f"{label}-html-{index}", payload, "html", ".html")
            summary = {"kind": "html", "bytes": artifact.size, "description": name}
            lines.append(render(summary, name, artifact))
        if not produced and not images and not html_items:
            lines.append("  (no artifacts produced)")
        head = raw[:EVIDENCE_TEXT_HEAD]
        lines.append(f"  text: {head}" + ("\n  ... (truncated)" if len(raw) > len(head) else ""))
        lines.append(f"  full: {ref.path} ({ref.size:,} bytes, sha256 {ref.sha256[:12]})")
        return "\n".join(lines)


def _describe(value) -> str:
    try:
        return value.to_json() if hasattr(value, "to_json") else repr(value)
    except Exception:  # noqa: BLE001
        return repr(value)


class GeoBenchTool(Tool):
    """One GeoBenchX tool, wrapped so CoreCoder can call it.

    Holds the LangGraph-shaped `state` dict so successive calls share the
    DataFrames and GeoDataFrames they build up.
    """

    #: Every one of these tools reads or writes the *same* `data_store`, and the
    #: model's order is the dataflow: a step that computes a column and a step
    #: that reads it are a dependency, not two independent jobs. Measured with a
    #: scripted LLM asking for such a pair in one response: **27 of 30 batches
    #: failed** with `DataFrame 'out' not found`, because the reader reached the
    #: store first. One in-order batch instead of a pool is a cheap fix for a
    #: failure that was being blamed on the model.
    concurrency_safe = False

    def __init__(self, spec, state: dict, record: Callable[[str, dict], None],
                 on_result: Callable[[str, str], None] | None = None,
                 schema_hints: bool = False,
                 evidence: EvidenceStore | None = None,
                 evidence_schema_preview: str = DEFAULT_SCHEMA_PREVIEW):
        self.name = spec.name
        self.description = spec.description or spec.name
        self.parameters = _drop_injected_state(spec.args_schema.model_json_schema())
        self._func = spec.func
        self._state = state
        self._record = record
        self._on_result = on_result
        self._schema_hints = schema_hints
        self._evidence = (
            _EvidenceBlock(evidence, evidence_schema_preview)
            if evidence is not None else None
        )

    def execute(self, **kwargs: Any) -> str:
        # Recorded before running: the trajectory is the scored artefact, and a
        # call that raises still happened as far as the transcript is concerned.
        self._record(self.name, kwargs)
        # Which frames the call is about to create: captured before, because
        # afterwards there is no way to tell its output from an earlier one's.
        before = set(self._state.get("data_store") or {})
        before_images = len(self._state.get("image_store") or [])
        before_html = len(self._state.get("html_store") or [])
        try:
            out = str(self._func(state=self._state, **kwargs))
        except Exception as e:  # noqa: BLE001
            # boundary: the agent gets an error string, not a traceback
            out = f"Error: {e}"
        # Enriched before `on_result`, so what gets recorded is what the model
        # actually saw -- a hint the model received but the record does not show
        # would make the next run's numbers unexplainable.
        if self._schema_hints:
            out = _with_schema_hint(out, kwargs, self._state)
        if self._evidence is not None:
            store = self._state.get("data_store") or {}
            produced = {k: v for k, v in store.items() if k not in before}
            images = (self._state.get("image_store") or [])[before_images:]
            html_items = (self._state.get("html_store") or [])[before_html:]
            out = self._evidence(self.name, out, produced, images, html_items)
        if self._on_result is not None:
            # Kept so a run can be diagnosed afterwards. Without it the only
            # record is the plan, and "it computed the right answer" becomes an
            # inference from a plausible-looking trajectory rather than evidence.
            self._on_result(self.name, out)
        return out


def build_geobench_tools(
    state: dict | None = None,
    record: Callable[[str, dict], None] | None = None,
    only: tuple[str, ...] | None = None,
    root: Path | None = None,
    on_result: Callable[[str, str], None] | None = None,
    schema_hints: bool = False,
    evidence: EvidenceStore | None = None,
    evidence_schema_preview: str = DEFAULT_SCHEMA_PREVIEW,
) -> list[GeoBenchTool]:
    """Wrap GeoBenchX's tools for CoreCoder.

    `only` narrows the set -- the first benchmark profile deliberately exposes a
    subset rather than all 24, because a wide tool surface makes both the model
    and the evaluation harder for no benefit.

    `state` defaults to a fresh `benchmark_state()`, which is what one task
    should get. It is a fresh dict per call, so the default is already isolated;
    pass your own only when you are deliberately sharing state across calls.
    """
    module = load_geobenchx_tools_module(root)
    if state is None:
        state = benchmark_state()
    if record is None:
        record = lambda name, args: None

    names = only if only is not None else GEOBENCH_TOOL_NAMES
    tools: list[GeoBenchTool] = []
    for name in names:
        spec = getattr(module, f"{name}_tool", None)
        if spec is None:
            raise AttributeError(
                f"GeoBenchX has no {name}_tool; upstream renamed or dropped it. "
                "Update GEOBENCH_TOOL_NAMES rather than letting the profile shrink silently."
            )
        tools.append(GeoBenchTool(
            spec,
            state,
            record,
            on_result,
            schema_hints,
            evidence,
            evidence_schema_preview,
        ))
    return tools
