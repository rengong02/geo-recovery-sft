"""Renderer r2: the column list must survive the trip back.

v1 joined column names with `", "`, so a column actually called `Population,
total` was displayed as two fragments and no parser could split them back --
measured on the shipped data, 24 of 192 loader observations could not be
round-tripped. A separator that can occur inside the values is not a separator.

These tests are the v2 gate for that (§3 of the protocol draft): the rendering
must be lossless for the characters that broke v1, and a duplicate name must be
detectable rather than silently collapsed.
"""

from __future__ import annotations

import json

from strata.train.executor import RENDERER_VERSION, render_column_names
from strata.train.sample import SAMPLE_SCHEMA_VERSION, TERMINATION_SCHEMA_VERSION


def _roundtrip(columns: list[str]) -> list[str]:
    return json.loads(render_column_names(columns))


def test_the_renderer_is_v2():
    assert RENDERER_VERSION == "r2"
    assert SAMPLE_SCHEMA_VERSION == 2
    assert TERMINATION_SCHEMA_VERSION == 2


def test_the_v1_breaking_case_round_trips():
    """The exact name that could not be recovered under `", "`."""
    columns = ["Country Name", "Population, total", "2021"]
    assert _roundtrip(columns) == columns


def test_commas_quotes_newlines_and_unicode_survive():
    columns = [
        "Population, total",
        'Say "hi"',
        "two\nlines",
        "Ünïcode ✓",
        "back\\slash",
        "tab\there",
    ]
    assert _roundtrip(columns) == columns


def test_an_empty_name_survives():
    """An empty string is a legal column name and must not vanish."""
    assert _roundtrip(["", "a"]) == ["", "a"]


def test_duplicate_names_are_visible_after_a_round_trip():
    """Not silently collapsed: the caller can see the frame has two of them.

    Rejecting duplicates is the generator's job, not the renderer's -- a
    renderer that quietly deduplicated would hide a real defect in the frame.
    """
    columns = ["value", "value"]
    recovered = _roundtrip(columns)
    assert recovered == columns
    assert len(recovered) != len(set(recovered))


def test_non_string_columns_are_rendered_as_their_string_form():
    """A frame with integer column labels still renders and recovers."""
    assert _roundtrip([1990, "1991"]) == ["1990", "1991"]


def test_the_rendering_is_json_not_a_delimiter_join():
    """The property, stated directly: the output parses as a JSON array."""
    rendered = render_column_names(["a, b", "c"])
    assert json.loads(rendered) == ["a, b", "c"]
    assert rendered.startswith("[") and rendered.endswith("]")
