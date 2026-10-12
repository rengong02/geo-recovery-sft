"""Durable, fail-closed state for offline baseline jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path


def samples_digest(samples: list[dict]) -> str:
    normalized = []
    for sample in samples:
        record = dict(sample)
        if isinstance(record.get("steps"), list):
            record["steps"] = [
                {key: value for key, value in step.items() if key != "seconds"}
                if isinstance(step, dict) else step
                for step in record["steps"]
            ]
        normalized.append(record)
    payload = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in normalized
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def digest_mapping(mapping: dict) -> str:
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl_samples(path: Path) -> list[dict]:
    samples = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        identifier = record.get("sample_id") if isinstance(record, dict) else None
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"invalid sample_id at {path}:{line_number}")
        if identifier in seen:
            raise ValueError(f"duplicate sample_id {identifier!r} at {path}:{line_number}")
        seen.add(identifier)
        samples.append(record)
    return samples


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another job holds lock: {path}") from None
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def progress_metadata(samples: list[dict], *, model: str, evaluation_kind: str,
                      device: str, attention_implementation: str, max_turns: int,
                      max_new_tokens: int, source_manifest: dict | None = None) -> dict:
    return {
        "report_schema_version": 2,
        "model": model,
        "evaluation_kind": evaluation_kind,
        "checkpoint_kind": evaluation_kind,
        "requested_device": device,
        "requested_attention_implementation": attention_implementation,
        "samples_digest": samples_digest(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "sample_schema_versions": sorted({sample.get("schema_version") for sample in samples}, key=str),
        "tool_protocol": "native_tools_when_supported; structured_assistant_tool_messages",
        "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "max_turns": max_turns,
        "source_manifest": source_manifest or {},
    }


def validate_rows(rows: list[dict], identifiers: list[str]) -> None:
    seen = set()
    allowed = set(identifiers)
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("invalid progress row")
        identifier = row.get("sample_id")
        if identifier in seen:
            raise ValueError(f"duplicate progress sample_id {identifier!r}")
        if identifier not in allowed:
            raise ValueError(f"foreign progress sample_id {identifier!r}")
        if not isinstance(row.get("outcome"), str) or not row["outcome"]:
            raise ValueError(f"missing outcome for {identifier!r}")
        seen.add(identifier)


def load_progress(path: Path, expected: dict) -> tuple[dict, list[dict]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": "metadata", "metadata": expected},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return expected, []

    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if not lines or not lines[0].endswith(b"\n"):
        raise ValueError("incomplete progress metadata; use a new output directory")
    header = json.loads(lines[0])
    if header.get("record_type") != "metadata" or header.get("metadata") != expected:
        raise ValueError("progress metadata mismatch; use a new output directory")
    rows = []
    committed = len(lines[0])
    for index, line in enumerate(lines[1:], 1):
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            suffix = hashlib.sha256(line).hexdigest()[:16]
            quarantine = path.with_name(f"{path.name}.torn-{suffix}.bin")
            if not quarantine.exists():
                with quarantine.open("xb") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            with path.open("r+b") as handle:
                handle.truncate(committed)
                handle.flush()
                os.fsync(handle.fileno())
            break
        record = json.loads(line)
        if record.get("record_type") != "result" or not isinstance(record.get("row"), dict):
            raise ValueError(f"invalid progress record at line {index + 1}")
        rows.append(record["row"])
        committed += len(line)
    validate_rows(rows, expected["sample_ids"])
    return expected, rows


def append_progress_row(path: Path, row: dict) -> None:
    payload = json.dumps({"record_type": "result", "row": row}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def coverage_for(samples: list[dict], rows: list[dict]) -> dict:
    identifiers = [sample["sample_id"] for sample in samples]
    validate_rows(rows, identifiers)
    measured = {row["sample_id"] for row in rows}
    missing = [identifier for identifier in identifiers if identifier not in measured]
    return {"intended": len(identifiers), "measured": len(rows), "missing": len(missing),
            "complete": not missing, "missing_sample_ids": missing}


def runtime_source_manifest(root: Path, model: str, tool_python: str) -> dict:
    files = {}
    for folder in ("corecoder", "strata", "scripts"):
        for path in sorted((root / folder).rglob("*.py")):
            files[str(path.relative_to(root))] = file_digest(path)
    upstream = Path(os.getenv("GEOBENCHX_ROOT", root.parent / "GeoBenchX"))
    for path in sorted((upstream / "geobenchx").rglob("*.py")):
        files["GeoBenchX/" + str(path.relative_to(upstream))] = file_digest(path)
    checkpoint = Path(model)
    model_files = {}
    if checkpoint.is_dir():
        for path in sorted(checkpoint.iterdir()):
            if path.is_file() and (
                path.suffix in {".safetensors", ".bin"}
                or path.name in {"config.json", "generation_config.json", "tokenizer.json",
                                 "tokenizer_config.json", "special_tokens_map.json",
                                 "vocab.json", "merges.txt", "model.safetensors.index.json",
                                 "pytorch_model.bin.index.json"}
            ):
                model_files[path.name] = file_digest(path)
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "pandas", "numpy", "tokenizers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"runtime_source_digest": digest_mapping(files), "environment": versions,
            "checkpoint_digest": digest_mapping(model_files) if model_files else None,
            "tool_python": str(Path(tool_python).resolve())}
