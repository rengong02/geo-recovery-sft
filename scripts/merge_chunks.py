#!/usr/bin/env python
"""Validate source, configuration and row identity before merging baseline chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_state import coverage_for, read_jsonl_samples, samples_digest, write_json_atomic
from strata.train.score import summarise

CHUNK_NAME = re.compile(r"chunk_(\d+)\.json$")
CONFIG_KEYS = (
    "evaluation_kind", "checkpoint_kind", "model", "device", "attention_implementation",
    "sample_schema_versions", "tool_protocol", "decoding", "max_turns", "scorer_version",
    "source_manifest", "report_schema_version",
)


def validate_chunk_report(report: dict, samples: list[dict], *, chunk_start: int,
                          chunk_size: int, source_manifest: dict | None) -> list[dict]:
    if not isinstance(report, dict) or not isinstance(report.get("rows"), list):
        raise TypeError("chunk must contain rows")
    rows = report["rows"]
    identifiers = [row.get("sample_id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(rows):
        raise ValueError("invalid row in chunk")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate sample_id in chunk")
    intended = samples[chunk_start:chunk_start + chunk_size]
    if not intended or identifiers != [sample["sample_id"] for sample in intended][:len(rows)]:
        raise ValueError("foreign/out-of-order sample_id or wrong chunk boundary")
    if report.get("samples_digest") != samples_digest(intended):
        raise ValueError("chunk samples_digest does not match the full source slice")
    if report.get("n") != len(rows) or report.get("summary") != summarise([row["outcome"] for row in rows]):
        raise ValueError("chunk summary/n does not match rows under the current scorer")
    if report.get("evaluation_kind") not in {"base", "sft-probe"}:
        raise ValueError("invalid evaluation_kind")
    if report.get("checkpoint_kind") != report["evaluation_kind"]:
        raise ValueError("checkpoint_kind disagrees with evaluation_kind")
    if source_manifest is not None and report.get("source_manifest") != source_manifest:
        raise ValueError("chunk source_manifest mismatch")
    if "coverage" in report and report["coverage"] != coverage_for(intended, rows):
        raise ValueError("chunk coverage does not match rows")
    return rows


def merge_reports(chunks_dir: Path, samples: list[dict], chunk_size: int = 32,
                   source_manifest: dict | None = None) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows = []
    reports = []
    configuration = None
    for path in sorted(chunks_dir.glob("chunk_*.json")):
        match = CHUNK_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid chunk filename: {path.name}")
        start = int(match.group(1))
        if start % chunk_size:
            raise ValueError(f"wrong chunk boundary: {path.name}")
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(validate_chunk_report(report, samples, chunk_start=start,
                                          chunk_size=chunk_size, source_manifest=source_manifest))
        current = {key: report.get(key) for key in CONFIG_KEYS}
        if configuration is not None and current != configuration:
            raise ValueError(f"mixed evaluation configuration: {path.name}")
        configuration = current
        reports.append(report)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample_id across chunks")
    coverage = coverage_for(samples, rows)
    return {
        "what": f"validated merge of {len(reports)} chunk reports; partial coverage is not a baseline",
        **(configuration or {}),
        "samples_digest": samples_digest(samples),
        "chunks": len(reports),
        "coverage": coverage,
        "n": len(rows),
        "summary": summarise([row["outcome"] for row in rows]),
        "merged_summary": summarise([row["outcome"] for row in rows]),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    try:
        source = json.loads(args.source_manifest.read_text(encoding="utf-8")) if args.source_manifest else None
        report = merge_reports(args.chunks, read_jsonl_samples(args.samples), args.chunk_size, source)
        write_json_atomic(args.out, report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    coverage = report["coverage"]
    print(f"coverage: {coverage['measured']}/{coverage['intended']} "
          f"({'complete' if coverage['complete'] else 'INCOMPLETE'}); wrote {args.out}")
    return 0 if coverage["complete"] else 75


if __name__ == "__main__":
    raise SystemExit(main())
