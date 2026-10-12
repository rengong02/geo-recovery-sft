"""Regression tests for resumable baseline jobs and chunk validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_baseline = load_script("resume_run_baseline", ROOT / "scripts" / "run_baseline.py")
merge_chunks = load_script("resume_merge_chunks", ROOT / "scripts" / "merge_chunks.py")


def sample(sample_id: str) -> dict:
    return {"sample_id": sample_id, "schema_version": 2, "steps": []}


def test_jsonl_loader_rejects_duplicate_ids_and_accepts_missing_final_newline(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(sample("a")) + "\n" + json.dumps(sample("a")), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_baseline.read_jsonl_samples(path)
    path.write_text("\n" + json.dumps(sample("a")), encoding="utf-8")
    assert run_baseline.read_jsonl_samples(path) == [sample("a")]


def test_progress_journal_round_trips_rows_and_rejects_config_drift(tmp_path):
    samples = [sample("a"), sample("b")]
    metadata = run_baseline.progress_metadata(
        samples,
        model="model",
        evaluation_kind="base",
        device="cuda",
        attention_implementation="sdpa",
        max_turns=6,
        max_new_tokens=1024,
        source_manifest={"input_file_sha256": "sha256:source"},
    )
    path = tmp_path / "progress.jsonl"

    loaded_metadata, rows = run_baseline.load_progress(path, metadata)
    assert loaded_metadata == metadata
    assert rows == []

    row = {"sample_id": "a", "outcome": "correct", "attempt": {}}
    run_baseline.append_progress_row(path, row)
    _, rows = run_baseline.load_progress(path, metadata)
    assert rows == [row]

    changed = dict(metadata, max_new_tokens=512)
    with pytest.raises(ValueError, match="progress metadata mismatch"):
        run_baseline.load_progress(path, changed)


def test_progress_journal_rejects_duplicate_rows(tmp_path):
    samples = [sample("a")]
    metadata = run_baseline.progress_metadata(
        samples,
        model="model",
        evaluation_kind="base",
        device="cpu",
        attention_implementation="eager",
        max_turns=1,
        max_new_tokens=8,
    )
    path = tmp_path / "progress.jsonl"
    run_baseline.load_progress(path, metadata)
    row = {"sample_id": "a", "outcome": "no_termination"}
    run_baseline.append_progress_row(path, row)
    run_baseline.append_progress_row(path, row)

    with pytest.raises(ValueError, match="duplicate progress sample_id"):
        run_baseline.load_progress(path, metadata)


def test_merge_rejects_duplicate_ids_instead_of_counting_set_coverage(tmp_path):
    samples = [sample("a"), sample("b")]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(row) for row in samples) + "\n", encoding="utf-8"
    )
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    report = {
        "model": "model",
        "evaluation_kind": "base",
        "checkpoint_kind": "base",
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "max_turns": 1,
        "rows": [
            {"sample_id": "a", "outcome": "correct"},
            {"sample_id": "a", "outcome": "correct"},
        ],
        "summary": {"n": 2, "correct": 2, "accuracy": 1.0,
                    "answered": 2, "accuracy_of_answered": 1.0,
                    "format_failures": 0,
                    "by_outcome": {"correct": 2}},
    }
    (chunks / "chunk_00000.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        merge_chunks.validate_chunk_report(
            report, samples, chunk_start=0, chunk_size=2, source_manifest=None
        )


def metadata_for(samples):
    return run_baseline.progress_metadata(
        samples, model="model", evaluation_kind="base", device="cpu",
        attention_implementation="eager", max_turns=1, max_new_tokens=8,
    )


def test_interruption_keeps_completed_raw_attempts_and_resume_skips_them(tmp_path):
    samples = [sample("a"), sample("b")]
    progress = tmp_path / "progress.jsonl"
    metadata = metadata_for(samples)
    _, rows = run_baseline.load_progress(progress, metadata)

    def interrupted(record):
        if record["sample_id"] == "b":
            raise KeyboardInterrupt
        return {"sample_id": "a", "outcome": "correct", "raw_turns": ["raw output"],
                "attempt": {"termination": {"status": "done"}, "calls": []}}

    with pytest.raises(KeyboardInterrupt):
        run_baseline.run_pending_samples(samples, rows, interrupted, progress)

    _, restored = run_baseline.load_progress(progress, metadata)
    assert restored == rows
    assert restored[0]["raw_turns"] == ["raw output"]
    evaluated = []

    def finish(record):
        evaluated.append(record["sample_id"])
        return {"sample_id": record["sample_id"], "outcome": "no_termination"}

    assert run_baseline.run_pending_samples(samples, restored, finish, progress) == "complete"
    assert evaluated == ["b"]


def test_budget_stop_is_incomplete_and_does_not_evaluate_a_new_sample(tmp_path):
    progress = tmp_path / "progress.jsonl"
    samples = [sample("a")]
    run_baseline.load_progress(progress, metadata_for(samples))
    rows = []

    def forbidden(record):
        raise AssertionError("must stop before computing a sample")

    assert run_baseline.run_pending_samples(samples, rows, forbidden, progress, deadline=0) == "budget_exhausted"
    assert rows == []


def test_uncommitted_tail_is_quarantined_but_committed_corruption_is_refused(tmp_path):
    progress = tmp_path / "progress.jsonl"
    metadata = metadata_for([sample("a"), sample("b")])
    run_baseline.load_progress(progress, metadata)
    row = {"sample_id": "a", "outcome": "correct"}
    run_baseline.append_progress_row(progress, row)
    with progress.open("ab") as handle:
        handle.write(b'{"record_type":"result","row":')
    _, restored = run_baseline.load_progress(progress, metadata)
    assert restored == [row]
    assert len(list(tmp_path.glob("*.torn-*.bin"))) == 1
    with progress.open("ab") as handle:
        handle.write(b"broken-json\n")
    with pytest.raises(ValueError):
        run_baseline.load_progress(progress, metadata)


def test_completed_resume_does_not_load_the_model(monkeypatch, tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    samples = [sample("a")]
    samples_path.write_text(json.dumps(samples[0]) + "\n", encoding="utf-8")
    source = {"runtime_source_digest": "fixed", "checkpoint_digest": "fixed"}
    monkeypatch.setattr(run_baseline, "runtime_source_manifest", lambda *args: source)
    metadata = metadata_for(samples)
    metadata["source_manifest"] = source
    metadata["scorer_version"] = "v2.1-status-schema"
    progress = tmp_path / "progress.jsonl"
    run_baseline.load_progress(progress, metadata)
    run_baseline.append_progress_row(progress, {"sample_id": "a", "outcome": "correct"})

    def forbidden(*args):
        raise AssertionError("completed resume must not load a checkpoint")

    monkeypatch.setattr(run_baseline, "load_checkpoint", forbidden)
    args = SimpleNamespace(
        samples=samples_path, out=tmp_path / "report.json", model="model", limit=0,
        progress=progress, tool_python=None, source_manifest=None,
        device="cpu", attn_implementation="eager", evaluation_kind="base",
        max_turns=1, max_new_tokens=8, resume=True, budget_seconds=0,
    )
    assert run_baseline.run_resumable(args) == 0
    assert json.loads(args.out.read_text())["coverage"]["complete"]


def chunk_report(samples, rows):
    from strata.train.score import summarise

    return {"model": "model", "evaluation_kind": "base", "checkpoint_kind": "base",
            "device": "cuda", "attention_implementation": "sdpa",
            "samples_digest": run_baseline.samples_digest(samples),
            "decoding": {"do_sample": False, "max_new_tokens": 8}, "max_turns": 1,
            "n": len(rows), "summary": summarise([row["outcome"] for row in rows]), "rows": rows}


@pytest.mark.parametrize("damage", ["digest", "foreign_id", "summary", "source"])
def test_chunk_validation_rejects_source_and_row_damage(damage):
    samples = [sample("a")]
    report = chunk_report(samples, [{"sample_id": "a", "outcome": "correct"}])
    source = None
    if damage == "digest":
        report["samples_digest"] = "sha256:wrong"
    elif damage == "foreign_id":
        report["rows"][0]["sample_id"] = "other"
    elif damage == "summary":
        report["summary"]["correct"] = 0
    else:
        source = {"runtime_source_digest": "expected"}
    with pytest.raises(ValueError):
        merge_chunks.validate_chunk_report(report, samples, chunk_start=0, chunk_size=1, source_manifest=source)


def test_valid_chunks_have_different_slice_digests_but_one_full_source_digest(tmp_path):
    samples = [sample("a"), sample("b")]
    for index, record in enumerate(samples):
        report = chunk_report([record], [{"sample_id": record["sample_id"], "outcome": "correct"}])
        (tmp_path / f"chunk_{index:05d}.json").write_text(json.dumps(report), encoding="utf-8")
    merged = merge_chunks.merge_reports(tmp_path, samples, chunk_size=1)
    assert merged["samples_digest"] == run_baseline.samples_digest(samples)
    assert merged["coverage"]["complete"]
    assert len(merged["rows"]) == 2


@pytest.mark.parametrize("damage", ["corrupt_json", "model", "decoding", "kind"])
def test_merge_refuses_one_corrupt_or_mixed_chunk(damage, tmp_path):
    samples = [sample("a"), sample("b")]
    for index, record in enumerate(samples):
        report = chunk_report([record], [{"sample_id": record["sample_id"], "outcome": "correct"}])
        if index == 1 and damage == "model":
            report["model"] = "another-model"
        elif index == 1 and damage == "decoding":
            report["decoding"]["max_new_tokens"] = 16
        elif index == 1 and damage == "kind":
            report["evaluation_kind"] = report["checkpoint_kind"] = "sft-probe"
        text = "broken-json" if index == 1 and damage == "corrupt_json" else json.dumps(report)
        (tmp_path / f"chunk_{index:05d}.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        merge_chunks.merge_reports(tmp_path, samples, chunk_size=1)


def test_file_lock_is_atomic_and_never_unlinked(tmp_path):
    state = load_script("resume_baseline_state", ROOT / "scripts" / "baseline_state.py")
    lock = tmp_path / "job.lock"
    with state.exclusive_lock(lock):
        second = state.exclusive_lock(lock)
        with pytest.raises(RuntimeError, match="another job holds lock"):
            second.__enter__()
    assert lock.exists()
    with state.exclusive_lock(lock):
        assert lock.exists()
