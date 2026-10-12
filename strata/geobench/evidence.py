"""Making a GIS tool's result addressable, and describing it by type.

A GeoBenchX tool hands the model raw text. For a table that text is often the
whole frame -- one tool result in the baseline run was **247,772 characters**,
most of it a World Bank frame's 69 columns. The model needs to know what it
got, not to read all of it.

Two separate jobs, kept separate:

* **addressability.** The full result goes to a stable directory with a content
  hash, so a later step can go and look. Written into a task workspace that is
  deleted on the way out, it would not be addressable, so the store refuses
  that -- the word means the path still resolves afterwards.
* **a typed summary.** What a vector layer, a raster and a table each need to
  say about themselves is different, and a wall of text says none of it.

**Nothing here decides how much of the raw text the model still gets.** That is
a budget question, and budgets belong to P2; this module only measures. It is
duck-typed on purpose, so the summary rules can be tested without a GIS stack
installed.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How many column names a summary spells out. The World Bank frames have 69;
#: listing every year is noise, and the count plus a sample says the same thing.
MAX_SUMMARY_COLUMNS = 24
MAX_SAMPLE_ROWS = 3
SCHEMA_PREVIEWS = ("head", "head_tail")
DEFAULT_SCHEMA_PREVIEW = "head_tail"

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_label(label: str) -> str:
    """A logical name turned into something that is safe as a filename.

    Dots are stripped from the ends as well as dashes: `..` is made of
    characters this allows, so a label like `../../etc/passwd` would otherwise
    keep a leading `..` and the "safe" name would still climb a directory.
    """
    cleaned = _SAFE_LABEL.sub("-", label).strip("-.")
    return cleaned[:60] or "artifact"


@dataclass
class EvidenceRef:
    """Where one artifact went, and how to tell whether it changed."""

    label: str
    #: Relative to the store root, so it survives a move. None once the store
    #: has hit its cap and stopped writing: the metadata is still true, the
    #: file is not there.
    path: str | None
    sha256: str
    size: int
    kind: str

    def as_dict(self) -> dict:
        return {"label": self.label, "path": self.path, "sha256": self.sha256,
                "size": self.size, "kind": self.kind}


class EvidenceStore:
    """Writes results somewhere that is still there when the run is over."""

    #: Total bytes one store will write. Reached in a real run: archiving every
    #: map GeoBenchX draws put 543 MB on disk for nine tasks and the run hit its
    #: wall clock. A cap turns "how big can this get" into a stated bound rather
    #: than a surprise, and a store that has hit it says so.
    DEFAULT_MAX_BYTES = 64 * 1024 * 1024

    #: A single artifact larger than this is fingerprinted, not stored.
    #:
    #: Measured: one `Countries` frame serialised to GeoJSON is **21 MB** -- every
    #: polygon's coordinates at full precision -- so three calls filled 21.7 MB
    #: with the map images already excluded. Keeping bytes like that on the off
    #: chance somebody re-reads them is not what "addressable" has to mean.
    DEFAULT_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024

    def __init__(self, root: str | Path, max_bytes: int | None = None,
                 max_artifact_bytes: int | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.refs: list[EvidenceRef] = []
        self.max_bytes = self.DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
        self.max_artifact_bytes = (
            self.DEFAULT_MAX_ARTIFACT_BYTES if max_artifact_bytes is None
            else max_artifact_bytes
        )
        self.too_large = 0
        self.written_bytes = 0
        self.capped = False
        #: What the arm actually did, per call. Recorded because "evidence is
        #: on" is not a result: how often it engaged, and how much it saved, is
        #: what a paired experiment has to compare.
        self.calls = 0
        self.replaced = 0
        self.bytes_before = 0
        self.bytes_after = 0

    def note(self, before: int, after: int) -> None:
        self.calls += 1
        self.bytes_before += before
        self.bytes_after += after
        if after < before:
            self.replaced += 1

    def stats(self) -> dict:
        """The numbers a control/treatment comparison is allowed to quote."""
        return {
            "calls": self.calls,
            "replaced": self.replaced,
            "kept": self.calls - self.replaced,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_saved": self.bytes_before - self.bytes_after,
            "artifacts": len(self.refs),
            "stored": sum(1 for r in self.refs if r.path is not None),
            "written_bytes": self.written_bytes,
            "capped": self.capped,
            "too_large": self.too_large,
        }

    def put(self, label: str, payload: str | bytes, kind: str, suffix: str = ".txt") -> EvidenceRef:
        """Store one artifact under a content-addressed name.

        The digest is in the filename so the same result written twice is one
        file, and so a report can quote a hash instead of a path that moved.
        """
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        digest = hashlib.sha256(data).hexdigest()
        name = f"{safe_label(label)}-{digest[:12]}{suffix}"

        # Over the cap the metadata is still returned -- what it was and how big
        # it was are true either way -- but nothing more is written, and `capped`
        # says so. A ref whose path does not exist would be the one thing worse
        # than not storing it: a report could quote a path that resolves to
        # nothing.
        stored = (len(data) <= self.max_artifact_bytes
                  and self.written_bytes + len(data) <= self.max_bytes)
        if stored:
            (self.root / name).write_bytes(data)
            self.written_bytes += len(data)
        else:
            self.capped = True
            if len(data) > self.max_artifact_bytes:
                self.too_large += 1

        ref = EvidenceRef(label=label, path=name if stored else None,
                          sha256=digest, size=len(data), kind=kind)
        self.refs.append(ref)
        return ref


# --- what kind of thing is this ------------------------------------------------
#
# Duck-typed rather than imported: `geopandas` and `rasterio` are heavy, the
# summary rules are the interesting part, and a test of them should not need a
# GIS stack installed to run.


def _is_frame(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "__len__")


def _is_vector(value: Any) -> bool:
    """A CRS *or* a geometry column is enough to call it a layer.

    Either alone is a strong signal, and requiring both would classify a layer
    whose geometry accessor raises as a plain table -- telling the model it has
    a table when it has a map is worse than admitting a field is unreadable.
    """
    try:
        return _is_frame(value) and (hasattr(value, "crs") or hasattr(value, "geometry"))
    except Exception:  # noqa: BLE001 - an accessor that raises is still a signal
        return True


def _is_raster(value: Any) -> bool:
    return hasattr(value, "transform") and hasattr(value, "shape") and hasattr(value, "crs")


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and not _is_frame(value)


def classify(value: Any) -> str:
    """The artifact's type, by what it can do rather than by its class name."""
    if _is_vector(value):
        return "vector"
    if _is_raster(value):
        return "raster"
    if _is_frame(value):
        return "table"
    if _is_array(value):
        return "array"
    return "other"


# --- the summaries -------------------------------------------------------------


def validate_schema_preview(schema_preview: str) -> str:
    """Validate the schema preview policy used by an evidence arm."""
    if schema_preview not in SCHEMA_PREVIEWS:
        choices = ", ".join(SCHEMA_PREVIEWS)
        raise ValueError(f"unknown schema preview {schema_preview!r}; choose {choices}")
    return schema_preview


def _preview_columns(names: list, schema_preview: str = DEFAULT_SCHEMA_PREVIEW) -> list:
    """Keep the configured schema end(s) within the fixed column budget."""
    validate_schema_preview(schema_preview)
    if len(names) <= MAX_SUMMARY_COLUMNS:
        return names
    if schema_preview == "head":
        return names[:MAX_SUMMARY_COLUMNS]
    head = MAX_SUMMARY_COLUMNS // 2
    tail = MAX_SUMMARY_COLUMNS - head
    return names[:head] + names[-tail:]


def _columns(frame: Any, schema_preview: str = DEFAULT_SCHEMA_PREVIEW) -> dict[str, str]:
    """Column names from both ends with their dtypes, and a count of the rest.

    The dtype lookup reads the *original* column object, not its stringified
    name: a frame with non-string column labels would otherwise be asked for a
    column that does not exist.
    """
    names = list(frame.columns)
    shown = _preview_columns(names, schema_preview)
    out: dict[str, str] = {}
    for column in shown:
        try:
            out[str(column)] = str(frame[column].dtype)
        except Exception:  # noqa: BLE001 - one unreadable dtype must not lose the rest
            out[str(column)] = "?"
    if len(names) > len(shown):
        marker = {"...": f"+{len(names) - len(shown)} more"}
        head = MAX_SUMMARY_COLUMNS if schema_preview == "head" else MAX_SUMMARY_COLUMNS // 2
        out = dict(list(out.items())[:head] + list(marker.items()) + list(out.items())[head:])
    return out


def _sample(frame: Any, shown: list) -> list[dict]:
    """A couple of rows, cut to the same columns the summary already names.

    Bounded on both axes for the same reason: a sample of every column of a
    69-column frame puts the whole schema back into the summary it was cut from.
    """
    allowed = {str(c) for c in shown}
    try:
        head = frame.head(MAX_SAMPLE_ROWS)
        return [{str(k): str(v)[:60] for k, v in row.items() if str(k) in allowed}
                for row in head.to_dict("records")]
    except Exception:  # noqa: BLE001 - a sample is a nicety, never the point
        return []


def summarize_frame(
    frame: Any,
    limit: int | None = None,
    schema_preview: str = DEFAULT_SCHEMA_PREVIEW,
) -> dict:
    """What a table says about itself: shape, schema, and a taste of the rows."""
    validate_schema_preview(schema_preview)
    shown = _preview_columns(list(frame.columns), schema_preview)
    summary: dict[str, Any] = {
        "kind": classify(frame),
        "rows": len(frame),
        "columns": _columns(frame, schema_preview),
        "sample": _sample(frame, shown),
    }
    if summary["kind"] == "vector":
        summary["crs"] = str(getattr(frame, "crs", None))
        # Each read guarded separately: a frame that cannot report its geometry
        # should still report its bounds and its row count.
        try:
            summary["geometry_column"] = str(frame.geometry.name)
        except Exception:  # noqa: BLE001
            summary["geometry_column"] = None
        try:
            summary["bounds"] = [round(float(b), 6) for b in frame.total_bounds]
        except Exception:  # noqa: BLE001
            summary["bounds"] = None
        summary["invalid_geometries"] = _invalid_geometries(frame)
    return summary


def _invalid_geometries(frame: Any) -> int | None:
    """How many geometries are broken, or None when it cannot be determined.

    Not zero-on-failure: "we did not check" and "all of them are fine" are
    different statements, and a summary that conflates them is worse than one
    that admits the gap.
    """
    try:
        return int((~frame.geometry.is_valid).sum())
    except Exception:  # noqa: BLE001
        return None


def summarize_raster(raster: Any) -> dict:
    """What a raster says about itself: grid, georeferencing, and value range."""
    summary: dict[str, Any] = {
        "kind": "raster",
        "shape": list(getattr(raster, "shape", [])),
        "dtype": str(getattr(raster, "dtypes", getattr(raster, "dtype", "?"))),
        "crs": str(getattr(raster, "crs", None)),
        "transform": str(getattr(raster, "transform", None)),
        "nodata": getattr(raster, "nodata", None),
    }
    try:
        summary["bounds"] = [round(float(b), 6) for b in raster.bounds]
    except Exception:  # noqa: BLE001
        summary["bounds"] = None
    return summary


def summarize_array(array: Any) -> dict:
    """What a bare array says about itself. No CRS: an array does not have one."""
    summary: dict[str, Any] = {
        "kind": "array",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    # Non-numeric or empty: the shape and dtype are still worth reporting, so
    # the range is simply absent rather than the whole summary failing.
    with contextlib.suppress(Exception):
        summary["min"] = float(array.min())
        summary["max"] = float(array.max())
        summary["mean"] = round(float(array.mean()), 6)
    return summary


#: PNG/JPEG/GIF headers carry the dimensions; reading them beats importing an
#: image library to ask two integers.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GIF_MAGIC = (b"GIF87a", b"GIF89a")


def summarize_image(payload: bytes) -> dict:
    """Format, dimensions and hash -- enough to know what the picture is."""
    summary: dict[str, Any] = {
        "kind": "image",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()[:16],
    }
    if payload.startswith(_PNG_MAGIC) and len(payload) >= 24:
        width, height = struct.unpack(">II", payload[16:24])
        summary.update(format="png", width=width, height=height)
    elif payload[:6] in _GIF_MAGIC and len(payload) >= 10:
        width, height = struct.unpack("<HH", payload[6:10])
        summary.update(format="gif", width=width, height=height)
    elif payload[:2] == b"\xff\xd8":
        summary["format"] = "jpeg"
    else:
        summary["format"] = "unknown"
    return summary


def summarize(
    value: Any,
    limit: int | None = None,
    schema_preview: str = DEFAULT_SCHEMA_PREVIEW,
) -> dict:
    """The typed summary for whatever a tool produced."""
    if _is_vector(value) or _is_frame(value):
        return summarize_frame(value, limit, schema_preview)
    if _is_raster(value):
        return summarize_raster(value)
    if _is_array(value):
        return summarize_array(value)
    return {"kind": "other", "type": type(value).__name__}


def summarize_base64_image(payload: str) -> dict:
    """For GeoBenchX's `image_store`, which holds base64 rather than bytes."""
    try:
        return summarize_image(base64.b64decode(payload, validate=False))
    except Exception:  # noqa: BLE001
        return {"kind": "image", "format": "undecodable", "bytes": len(payload)}


# --- the text the model actually reads -----------------------------------------


def render(summary: dict, label: str = "", ref: EvidenceRef | None = None) -> str:
    """One compact block: what it is, what matters about it, and where it is.

    Deliberately not the whole frame. A summary that reproduced the input would
    be the problem it exists to solve.
    """
    lines = [f"[evidence] {label} ({summary.get('kind', 'other')})"]
    if summary.get("description"):
        lines.append(f"  description: {summary['description']}")
    for key in ("rows", "crs", "geometry_column", "bounds", "invalid_geometries",
                "shape", "dtype", "nodata", "format", "width", "height",
                "min", "max", "mean", "bytes"):
        if summary.get(key) is not None:
            lines.append(f"  {key}: {summary[key]}")
    columns = summary.get("columns")
    if columns:
        lines.append("  columns: " + ", ".join(f"{k}:{v}" for k, v in columns.items()))
    sample = summary.get("sample")
    if sample:
        lines.append(f"  sample: {sample}")
    if ref is not None:
        if ref.path is None:
            # Said plainly rather than omitted: a reader has to know the bytes
            # are not on disk, not wonder why there is no line for them.
            lines.append(f"  raw: NOT STORED, store cap reached "
                         f"({ref.size:,} bytes, sha256 {ref.sha256[:12]})")
        else:
            lines.append(f"  raw: {ref.path} ({ref.size:,} bytes, sha256 {ref.sha256[:12]})")
    return "\n".join(lines)


def render_image(summary: dict) -> str:
    """An image's line in the block: what it is, not the picture itself.

    No artifact reference, because the payload is deliberately not stored: see
    `_EvidenceBlock`, where duplicating it cost 543 MB in one nine-task run.
    """
    parts = [f"  image: {summary.get('description', '')}".rstrip()]
    for key in ("format", "width", "height", "bytes", "sha256"):
        if summary.get(key) is not None:
            parts.append(f"{key}={summary[key]}")
    return " ".join(parts)
