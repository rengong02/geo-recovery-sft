"""Content-based grouping for independently generated training samples."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_FINGERPRINT_VERSION = "content-schema-v1"


def _normalise_spec(spec: Any) -> dict | None:
    if not isinstance(spec, dict):
        return None
    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) and column for column in columns
    ):
        return None
    kind = spec.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    id_column = spec.get("id_column")
    if id_column is not None and (
        not isinstance(id_column, str) or not id_column.strip()
    ):
        return None
    return {
        "kind": kind,
        "columns": sorted(columns),
        "id_column": id_column,
    }


def content_schema_fingerprint(sample: dict) -> str | None:
    """Return a stable content-schema fingerprint for a synthetic sample.

    Handles, seeds, row counts, generator family names and gold answers are
    intentionally excluded. The frame order is retained, while column order
    is normalised conservatively so renaming a family cannot evade the gate.
    ``None`` means the sample does not contain enough synthetic provenance to
    compute the fingerprint.
    """
    provenance = sample.get("source_provenance")
    if not isinstance(provenance, dict):
        return None
    if provenance.get("source_kind") != "synthetic_generator":
        return None
    generator = provenance.get("generator")
    if not isinstance(generator, dict):
        return None
    specs = generator.get("specs")
    if not isinstance(specs, list) or not specs:
        return None
    normalised = [_normalise_spec(spec) for spec in specs]
    if any(spec is None for spec in normalised):
        return None
    payload = json.dumps(
        {"version": SCHEMA_FINGERPRINT_VERSION, "specs": normalised},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
