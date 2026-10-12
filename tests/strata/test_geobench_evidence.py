"""What a tool result should say about itself, and where the rest of it goes.

Synthetic on purpose. The rules under test -- which fields a vector layer has
to report, when a summary must admit it does not know something, how large a
summary is next to the frame it describes -- have nothing to do with whether a
GIS stack is installed, and a test that needs one runs in fewer places.

The size comparison at the end is the point of the whole module: a summary that
grew with the frame would be the problem it exists to solve.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.geobench.evidence import (
    MAX_SUMMARY_COLUMNS,
    EvidenceStore,
    classify,
    render,
    safe_label,
    summarize,
    summarize_array,
    summarize_base64_image,
    summarize_frame,
    summarize_image,
    summarize_raster,
)

# --- stand-ins, duck-typed the way the module sees them -----------------------


class _BoolSeries:
    def __init__(self, flags):
        self.flags = list(flags)

    def __invert__(self):
        return _BoolSeries([not f for f in self.flags])

    def sum(self):
        return sum(self.flags)


class _GeometrySeries:
    """A geometry column: a name, and validity per feature."""

    def __init__(self, name="geometry", invalid=0, count=2):
        self.name = name
        self._flags = [i >= invalid for i in range(count)]

    @property
    def is_valid(self):
        return _BoolSeries(self._flags)


class _Series:
    def __init__(self, values, dtype="object"):
        self.values = list(values)
        self.dtype = dtype

    def __len__(self):
        return len(self.values)


class _Table:
    """A frame with no geometry: a table, and nothing else."""

    def __init__(self, columns: dict, rows: int | None = None):
        self._cols = {k: v if isinstance(v, _Series) else _Series(v) for k, v in columns.items()}
        self._rows = rows if rows is not None else len(next(iter(self._cols.values())))

    @property
    def columns(self):
        return list(self._cols)

    def __len__(self):
        return self._rows

    def __getitem__(self, key):
        return self._cols[key]

    def head(self, n):
        return _Head(self._cols, min(n, self._rows))

    def to_dict(self, orient):
        assert orient == "records"
        return [{k: v.values[i] for k, v in self._cols.items()} for i in range(min(2, self._rows))]


class _Head(_Table):
    pass


class _Vector(_Table):
    """A frame with a CRS and a geometry column."""

    def __init__(self, columns, crs="EPSG:4326", invalid=0, bounds=(0.0, 0.0, 1.0, 1.0)):
        super().__init__(columns)
        self.crs = crs
        self.geometry = _GeometrySeries("geometry", invalid, len(self))
        self._bounds = bounds

    @property
    def total_bounds(self):
        return self._bounds


class _UnreadableValidity(_Table):
    """Geometries exist; whether they are valid cannot be computed."""

    crs = "EPSG:4326"
    total_bounds = (0.0, 0.0, 1.0, 1.0)

    @property
    def geometry(self):
        return _UnreadableGeometry()


class _UnreadableGeometry:
    name = "geometry"

    @property
    def is_valid(self):
        raise RuntimeError("validity not computable")


class _NoGeometryAtAll(_Table):
    """A frame that claims a CRS and a geometry, and can report neither well."""

    crs = "EPSG:4326"

    @property
    def geometry(self):
        raise RuntimeError("geometry accessor unavailable")

    @property
    def total_bounds(self):
        raise RuntimeError("bounds unavailable")


class _Raster:
    shape = (1, 120, 200)
    dtypes = "float32"
    crs = "EPSG:32633"
    transform = "| 30.0, 0.0, 300000.0 | 0.0, -30.0, 5000000.0 |"
    nodata = -9999.0
    bounds = (300000.0, 4994000.0, 306000.0, 5000000.0)


class _FakeArray:
    shape = (2, 3)
    dtype = "float64"

    def min(self):
        return 1.0

    def max(self):
        return 9.0

    def mean(self):
        return 4.5


def _image_b64(width: int, height: int) -> str:
    import base64

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height) + b"\x00" * 8
    return base64.b64encode(png).decode()


# --- classification -----------------------------------------------------------


def test_each_artifact_kind_is_recognised_by_what_it_can_do():
    assert classify(_Vector({"a": [1]})) == "vector"
    assert classify(_Table({"a": [1]})) == "table"
    assert classify(_Raster()) == "raster"
    assert classify(_FakeArray()) == "array"
    assert classify(object()) == "other"


class _FakeArray:
    shape = (2, 3)
    dtype = "float64"

    def min(self):
        return 1.0

    def max(self):
        return 9.0

    def mean(self):
        return 4.5


# --- the summaries ------------------------------------------------------------


def test_a_vector_summary_carries_what_a_vector_needs():
    frame = _Vector({"NAME_EN": ["A", "B"], "POP": _Series([1, 2], "int64")}, invalid=1,
                    bounds=(1.5, 2.5, 3.5, 4.5))

    summary = summarize_frame(frame)

    assert summary["kind"] == "vector"
    assert summary["rows"] == 2
    assert summary["crs"] == "EPSG:4326"
    assert summary["bounds"] == [1.5, 2.5, 3.5, 4.5]
    assert summary["invalid_geometries"] == 1
    assert summary["columns"] == {"NAME_EN": "object", "POP": "int64"}


def test_a_table_summary_does_not_invent_a_crs():
    summary = summarize_frame(_Table({"a": [1, 2, 3]}))

    assert summary["kind"] == "table"
    assert "crs" not in summary
    assert summary["rows"] == 3


def test_a_wide_frame_is_summarised_by_count_not_by_listing_every_column():
    """The World Bank frames have 69 columns, mostly years."""
    frame = _Table({f"c{i}": [1] for i in range(MAX_SUMMARY_COLUMNS + 45)})

    summary = summarize_frame(frame)

    assert len(summary["columns"]) == MAX_SUMMARY_COLUMNS + 1  # names + the marker
    assert summary["columns"]["..."] == "+45 more"


def test_a_wide_frame_summary_keeps_the_tail_of_the_schema():
    frame = _Table({f"c{i}": [1] for i in range(MAX_SUMMARY_COLUMNS + 45)})

    summary = summarize_frame(frame)

    assert "c68" in summary["columns"]


def test_a_wide_frame_can_reproduce_the_legacy_head_only_preview():
    frame = _Table({f"c{i}": [1] for i in range(MAX_SUMMARY_COLUMNS + 45)})

    summary = summarize_frame(frame, schema_preview="head")

    assert "c0" in summary["columns"]
    assert f"c{MAX_SUMMARY_COLUMNS - 1}" in summary["columns"]
    assert "c68" not in summary["columns"]
    assert summary["columns"]["..."] == "+45 more"


def test_an_unknown_schema_preview_is_rejected():
    frame = _Table({"c0": [1]})

    with pytest.raises(ValueError, match="schema preview"):
        summarize_frame(frame, schema_preview="middle")


def test_unknown_validity_is_none_not_zero():
    """"We did not check" and "all of them are fine" are different statements."""
    summary = summarize_frame(_UnreadableValidity({"a": [1]}))

    assert summary["invalid_geometries"] is None


def test_a_raster_summary_carries_georeferencing_and_a_nodata():
    summary = summarize_raster(_Raster())

    assert summary["kind"] == "raster"
    assert summary["shape"] == [1, 120, 200]
    assert summary["crs"] == "EPSG:32633"
    assert summary["nodata"] == -9999.0
    assert summary["bounds"][0] == 300000.0


def test_an_array_summary_has_no_crs_because_an_array_has_none():
    summary = summarize_array(_FakeArray())

    assert summary["kind"] == "array"
    assert "crs" not in summary
    assert (summary["min"], summary["max"]) == (1.0, 9.0)


def test_an_image_summary_reads_the_dimensions_from_the_header():
    summary = summarize_base64_image(_image_b64(640, 480))

    assert summary["format"] == "png"
    assert (summary["width"], summary["height"]) == (640, 480)
    assert summary["sha256"]


def test_an_undecodable_image_says_so_rather_than_guessing():
    summary = summarize_base64_image("not really base64 !!!")

    assert summary["kind"] == "image"
    assert summary["format"] in ("undecodable", "unknown")


def test_a_jpeg_is_recognised_without_a_decoder():
    assert summarize_image(b"\xff\xd8\xff\xe0rest")["format"] == "jpeg"


def test_a_gif_header_reports_its_dimensions_without_a_decoder():
    payload = b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00" * 8

    summary = summarize_image(payload)

    assert summary["format"] == "gif"
    assert (summary["width"], summary["height"]) == (320, 240)


def test_images_are_summarised_and_not_duplicated(tmp_path: Path):
    """State side effects are reported -- but an image's bytes are not copied.

    Archiving every map GeoBenchX draws cost **543 MB across one nine-task
    run**, and the run hit its wall clock before finishing. An image is also the
    one kind of artifact that was never in the model's context: GeoBenchX keeps
    it in `image_store` and the tool returns a short string, so a second copy is
    not evidence about context. Format, dimensions, size and hash are what the
    design asks for, and they are what this reports.
    """
    from strata.geobench.tools import GeoBenchTool

    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {}}

    state = {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}
    png = base64.b64decode(_image_b64(12, 8))

    def produce(state):
        state["image_store"].append({
            "type": "map",
            "description": "synthetic map",
            "base64": base64.b64encode(png).decode(),
        })
        state["html_store"].append({
            "type": "interactive_map",
            "description": "synthetic html",
            "html": "<html><body>map</body></html>",
        })
        return "result " + ("x" * 4_000)

    spec = SimpleNamespace(
        name="synthetic_visualize",
        description="synthetic visualizer",
        args_schema=_Schema(),
        func=produce,
    )
    store = EvidenceStore(tmp_path / "evidence")
    tool = GeoBenchTool(spec, state, lambda name, args: None, evidence=store)

    result = tool.execute()

    assert "[evidence]" in result
    assert "image:" in result and "format=png" in result
    assert "width=12" in result and "height=8" in result
    assert not any(ref.kind == "image" for ref in store.refs), (
        "the image payload was duplicated into the store again"
    )
    # HTML is small and is text the model may well have been shown, so it is
    # still filed.
    assert any(ref.kind == "html" for ref in store.refs)


def test_evidence_block_delivers_the_selected_schema_preview(tmp_path: Path):
    from strata.geobench.tools import _EvidenceBlock

    frame = _Table({f"c{i}": [1] for i in range(MAX_SUMMARY_COLUMNS + 45)})
    raw = "result " + ("x" * 4_000)

    head = _EvidenceBlock(EvidenceStore(tmp_path / "head"), "head")
    head_tail = _EvidenceBlock(EvidenceStore(tmp_path / "head-tail"), "head_tail")

    head_text = head("synthetic_table", raw, {"frame": frame})
    head_tail_text = head_tail("synthetic_table", raw, {"frame": frame})

    assert "c68" not in head_text
    assert "c68" in head_tail_text


@pytest.mark.parametrize("encoded", ["abc", None, 123])
def test_malformed_image_evidence_does_not_break_the_tool_result(tmp_path: Path, encoded):
    from strata.geobench.tools import GeoBenchTool

    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {}}

    state = {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}

    def produce(state):
        state["image_store"].append({"description": "broken image", "base64": encoded})
        return "result " + ("x" * 4_000)

    spec = SimpleNamespace(
        name="synthetic_broken_visualize",
        description="synthetic broken visualizer",
        args_schema=_Schema(),
        func=produce,
    )
    store = EvidenceStore(tmp_path / "evidence")
    tool = GeoBenchTool(spec, state, lambda name, args: None, evidence=store)

    result = tool.execute()

    assert "[evidence]" in result
    # A broken payload must not stop the block from being produced, and it must
    # not be filed either -- it is summarised as undecodable and nothing more.
    assert not any(ref.kind == "image" for ref in store.refs)
    assert "image:" in result


# --- the store ----------------------------------------------------------------


def test_the_store_is_content_addressed(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence")

    first = store.put("countries", "hello", "text")
    second = store.put("countries", "hello", "text")

    assert first.path == second.path, "the same content should be one file"
    assert first.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert first.size == 5


def test_a_stored_artifact_still_resolves_afterwards(tmp_path: Path):
    """Addressable means the path works later, not that bytes were written."""
    root = tmp_path / "evidence"
    ref = EvidenceStore(root).put("countries", "payload", "table", ".jsonl")

    assert (root / ref.path).read_text(encoding="utf-8") == "payload"


def test_the_stored_path_is_relative_so_it_survives_a_move(tmp_path: Path):
    ref = EvidenceStore(tmp_path / "e").put("x", "y", "text")

    assert not Path(ref.path).is_absolute()
    assert "/" not in ref.path


def test_a_label_with_separators_cannot_escape_the_directory():
    assert safe_label("../../etc/passwd") == "etc-passwd"
    assert safe_label("Countries (2020)") == "Countries-2020"
    assert safe_label("") == "artifact"


def test_bytes_are_stored_as_bytes(tmp_path: Path):
    ref = EvidenceStore(tmp_path / "e").put("img", b"\x89PNG\r\n\x1a\n", "image", ".png")

    assert ref.size == 8


# --- the text the model reads -------------------------------------------------

WIDE_COLUMNS = 69


def _wide_frame() -> _Table:
    """69 columns -- the shape of a World Bank frame -- and 200 distinguishable rows."""
    return _Table({name: [f"r{i}" for i in range(200)] for name in
                   ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
                   + [str(y) for y in range(1960, 1960 + WIDE_COLUMNS - 4)]})


def _raw_text(frame: _Table, rows: int = 200) -> str:
    """What GeoBenchX hands over today: every field of every row, as text.

    Built explicitly rather than estimated, so the comparison at the bottom
    measures two real strings and not an order of magnitude.
    """
    return "\n".join(
        ", ".join(f"{name}=r{i}" for name in frame.columns) for i in range(rows)
    )


def test_the_summary_is_far_smaller_than_the_text_it_replaces():
    """The whole point: a summary that grew with the frame solves nothing."""
    frame = _wide_frame()

    summary_text = render(summarize_frame(frame), "gdp_per_capita")
    raw_text = _raw_text(frame)

    assert len(summary_text) * 20 < len(raw_text), (
        f"summary {len(summary_text)} chars vs raw {len(raw_text)}"
    )
    assert len(summary_text.splitlines()) < 12


def test_the_summary_size_does_not_follow_the_row_count():
    """Rows are what a table has most of; the summary must not track them."""
    small = _wide_frame()
    large = _wide_frame()
    large._rows = 20_000

    small_text = render(summarize_frame(small))
    large_text = render(summarize_frame(large))

    # A hundred times the rows, and the only growth is the digits of the count.
    assert len(large_text) - len(small_text) < 5, (
        f"{len(small_text)} -> {len(large_text)} chars for 100x the rows"
    )


def test_a_vector_that_cannot_report_its_geometry_still_reports_its_shape():
    """One unreadable field must not cost the whole summary."""
    summary = summarize_frame(_NoGeometryAtAll({"NAME_EN": ["A"]}))

    assert summary["rows"] == 1
    assert summary["geometry_column"] is None
    assert summary["bounds"] is None


def test_the_summary_does_not_reproduce_the_data():
    frame = _wide_frame()

    text = render(summarize_frame(frame), "gdp_per_capita")

    assert "1960" in text, "the schema should be named"
    # Two sampled rows carry a value for it; the other 198 rows do not appear.
    assert "r0" in text and "r1" in text
    assert "r199" not in text


def test_the_rendered_block_names_the_artifact():
    store = EvidenceStore(Path("/tmp") / "unused-evidence")
    ref = store.put("gdp", "raw text", "table")

    text = render(summarize_frame(_wide_frame()), "gdp", ref)

    assert ref.path in text
    assert ref.sha256[:12] in text
    assert "bytes" in text


@pytest.mark.parametrize("kind_value", [{"kind": "other", "type": "int"}])
def test_a_result_it_does_not_understand_says_only_that(kind_value):
    assert summarize(object())["kind"] == "other"


# --- what a paired experiment is allowed to quote -----------------------------
#
# "Evidence is on" is not a result. How often the block engaged, and how much
# it saved, is what a control/treatment comparison has to compare.


def test_the_store_records_what_the_arm_actually_did(tmp_path: Path):
    store = EvidenceStore(tmp_path / "e")

    store.note(before=1000, after=200)   # replaced
    store.note(before=50, after=50)      # kept: the block was not smaller
    stats = store.stats()

    assert stats["calls"] == 2
    assert stats["replaced"] == 1
    assert stats["kept"] == 1
    assert stats["bytes_saved"] == 800


def test_an_untouched_store_reports_zeroes_not_nothing(tmp_path: Path):
    """A missing key and a zero mean different things in a report."""
    stats = EvidenceStore(tmp_path / "e").stats()

    assert stats["calls"] == 0
    assert stats["bytes_saved"] == 0


# --- the cap ------------------------------------------------------------------
#
# Archiving every map GeoBenchX draws put 543 MB on disk for nine tasks and the
# run hit its wall clock. A cap turns "how big can this get" into a bound.


def test_a_store_stops_writing_once_it_hits_its_cap(tmp_path: Path):
    store = EvidenceStore(tmp_path / "e", max_bytes=100)

    first = store.put("a", "x" * 60, "text")
    second = store.put("b", "y" * 60, "text")   # would take it to 120

    assert first.path is not None and (store.root / first.path).exists()
    assert second.path is None, "the cap was exceeded and it wrote anyway"
    assert store.capped is True


def test_a_capped_ref_still_reports_what_it_was(tmp_path: Path):
    """The metadata stays true; only the file is missing.

    A ref whose path resolves to nothing would be worse than no ref at all:
    a report could quote a path a reader cannot open.
    """
    store = EvidenceStore(tmp_path / "e", max_bytes=10)
    ref = store.put("big", "0123456789abcdef", "text")

    assert ref.path is None
    assert ref.size == 16
    assert ref.sha256 == hashlib.sha256(b"0123456789abcdef").hexdigest()


def test_a_capped_store_says_so_in_its_stats(tmp_path: Path):
    store = EvidenceStore(tmp_path / "e", max_bytes=10)
    store.put("a", "x" * 5, "text")
    store.put("b", "y" * 50, "text")
    stats = store.stats()

    assert stats["capped"] is True
    assert stats["artifacts"] == 2
    assert stats["stored"] == 1
    assert stats["written_bytes"] == 5


def test_the_cap_is_generous_enough_for_a_normal_run():
    """64 MB: the frames and text of a task, without the map images."""
    assert EvidenceStore.DEFAULT_MAX_BYTES == 64 * 1024 * 1024


def test_a_single_huge_artifact_is_fingerprinted_not_stored(tmp_path: Path):
    """One `Countries` frame serialises to 21 MB of GeoJSON coordinates.

    Measured, after the map images were already excluded. Keeping bytes like
    that on the off chance somebody re-reads them is not what "addressable"
    has to mean, so the fingerprint is kept and the payload is not.
    """
    store = EvidenceStore(tmp_path / "e", max_artifact_bytes=1_000)

    small = store.put("text", "x" * 500, "text")
    huge = store.put("frame", "y" * 5_000, "table")

    assert small.path is not None
    assert huge.path is None
    assert huge.size == 5_000 and huge.sha256
    assert store.stats()["too_large"] == 1
