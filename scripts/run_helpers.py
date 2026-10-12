"""Run the shape-OOD base evaluation and gated 48-step SFT task."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "evals" / "train" / "f2_shape_ood_20261009_r4"
sys.path.insert(0, str(ROOT))

from scripts.baseline_state import (
    coverage_for,
    digest_mapping,
    exclusive_lock,
    file_digest,
    progress_metadata,
    read_jsonl_samples,
    runtime_source_manifest,
    samples_digest,
    write_json_atomic,
)
from scripts.run_sft import choose_rows, encode_row
from strata.train.sample import validate_sample
from strata.train.schema_groups import content_schema_fingerprint
from strata.train.score import SCORER_VERSION, summarise


def verify_candidate(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "shape-ood-manifest-v1":
        raise ValueError("not a shape-OOD candidate manifest")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("candidate manifest has no artifact hashes")
    for name, expected in artifact_hashes.items():
        target = (bundle_dir / name).resolve()
        if not target.is_file() or file_digest(target) != expected:
            raise ValueError(f"candidate artifact changed or missing: {name}")
    return manifest


def validate_shape_ood_dataset(samples: list[dict], split: dict, manifest: dict) -> None:
    if samples_digest(samples) != manifest.get("samples_digest"):
        raise ValueError("candidate samples_digest mismatch")
    identifiers = [sample.get("sample_id") for sample in samples]
    assignments = split.get("assignments") or {}
    if len(identifiers) != len(set(identifiers)) or set(assignments) != set(identifiers):
        raise ValueError("candidate split must cover every unique sample exactly once")
    if not split.get("training_ready"):
        raise ValueError("candidate split is not training_ready")
    counts = {name: list(assignments.values()).count(name)
              for name in ("train", "validation", "test")}
    if counts != {"train": 48, "validation": 48, "test": 48}:
        raise ValueError(f"candidate split counts are not 48/48/48: {counts}")
    if split.get("content_schema_fingerprint_crossings"):
        raise ValueError("candidate content schema crosses explicit splits")

    families: dict[str, set[str]] = {}
    worlds: list[str] = []
    for sample in samples:
        verdict = validate_sample(sample)
        if not verdict.ok:
            raise ValueError(f"invalid candidate sample {sample.get('sample_id')}: {verdict.blockers}")
        lineage = sample.get("lineage") or {}
        family = str(lineage.get("generator_family") or "")
        families.setdefault(family, set()).add(assignments[sample["sample_id"]])
        if sample.get("train_eligible") is not False:
            raise ValueError("candidate samples must remain train_eligible=false")
        fingerprint = content_schema_fingerprint(sample)
        if fingerprint is None:
            raise ValueError(f"missing content schema fingerprint: {sample['sample_id']}")
        specs = ((sample.get("source_provenance") or {}).get("generator") or {}).get("specs")
        if not isinstance(specs, list) or len(specs) != 2:
            raise ValueError(f"candidate sample has invalid generator specs: {sample['sample_id']}")
        namespaces = {str(spec.get("entity_namespace") or "") for spec in specs}
        if len(namespaces) != 1 or "" in namespaces:
            raise ValueError(f"candidate sample has invalid world namespace: {sample['sample_id']}")
        worlds.extend(namespaces)
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("generator family leaks across candidate splits")
    if len(worlds) != len(set(worlds)):
        raise ValueError("candidate worlds are not independent")


def load_candidate_bundle(bundle_dir: Path) -> tuple[dict, list[dict], dict]:
    manifest = verify_candidate(bundle_dir)
    samples = read_jsonl_samples(bundle_dir / "samples.jsonl")
    split = json.loads((bundle_dir / "split_report.json").read_text(encoding="utf-8"))
    validate_shape_ood_dataset(samples, split, manifest)
    return manifest, samples, split


def baseline_command(samples: Path, output: Path, model: Path, kind: str,
                     manifest: Path, tool_python: str, budget_seconds: float) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "scripts" / "run_baseline.py"),
            "--samples", str(samples), "--out", str(output), "--model", str(model),
            "--evaluation-kind", kind, "--device", "cuda", "--attn-implementation", "sdpa",
            "--tool-python", tool_python, "--max-turns", "6", "--max-new-tokens", "1024",
            "--resume", "--source-manifest", str(manifest),
            "--budget-seconds", str(budget_seconds)]


def run_logged(command: list[str], path: Path, lock_descriptor: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        print("running:", " ".join(command), flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   pass_fds=(lock_descriptor,))
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def validate_base_report(report: dict, samples: list[dict], metadata: dict,
                         require_complete: bool = True) -> None:
    if report.get("metadata") != metadata:
        raise ValueError("base report metadata mismatch")
    if report.get("evaluation_kind") != "base":
        raise ValueError("stage requires a base evaluation report")
    if report.get("model") != metadata["model"] or report.get("samples_digest") != samples_digest(samples):
        raise ValueError("base report source/model mismatch")
    rows = report.get("rows")
    coverage = coverage_for(samples, rows) if isinstance(rows, list) else None
    if coverage is None or report.get("coverage") != coverage or (require_complete and not coverage["complete"]):
        raise ValueError("base report is incomplete or coverage is inconsistent")
    if report.get("n") != len(rows) or report.get("summary") != summarise([row["outcome"] for row in rows]):
        raise ValueError("base report summary mismatch")
    if any(not isinstance(row.get("attempt"), dict) or not isinstance(row.get("raw_turns"), list)
           for row in rows):
        raise ValueError("base rows must retain raw attempts and turns")


def validate_checkpoint(checkpoint: Path, samples: list[dict], split: dict) -> None:
    manifest = json.loads((checkpoint / "probe_manifest.json").read_text(encoding="utf-8"))
    train = choose_rows(samples, split, "train", 0, 7)
    validation = choose_rows(samples, split, "validation", 0, 8)
    expected = {
        "trajectory": "gold_only", "device": "cuda", "dtype": "bfloat16",
        "attention_implementation": "sdpa", "train_count": 48, "seen_train_count": 48,
        "eval_split": "validation", "eval_count": 48, "steps": 48,
        "learning_rate": 5e-6, "seed": 7, "source_samples_digest": samples_digest(samples),
        "method": "full_parameter_sft; not LoRA or QLoRA",
        "label_policy": "assistant_body_and_im_end_only; context_and_headers_masked",
        "train_sample_digest": samples_digest(train), "eval_sample_digest": samples_digest(validation),
        "train_sample_ids": [sample["sample_id"] for sample in train],
        "eval_sample_ids": [sample["sample_id"] for sample in validation],
        "step_sample_ids": [sample["sample_id"] for sample in train],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("SFT checkpoint does not match the shape-OOD 48-step recipe")
    losses = manifest.get("losses") or []
    if len(losses) != 48 or not all(isinstance(loss, (float, int)) and math.isfinite(loss) for loss in losses):
        raise ValueError("SFT checkpoint has no complete finite 48-step loss record")
    if samples_digest(read_jsonl_samples(checkpoint / "eval_samples.jsonl")) != samples_digest(validation):
        raise ValueError("SFT validation samples changed")
    if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise ValueError("SFT checkpoint is incomplete")


def check_tool_runtime(samples: list[dict], tool_python: str) -> None:
    from scripts.run_baseline import ToolServer

    server = ToolServer(tool_python)
    try:
        sample = samples[0]
        server.configure_for_sample(sample)
        for step in sample["steps"]:
            result = server.call(step["tool"], step["arguments"])
            if bool(result.get("failed")) != bool(step["failed"]):
                raise ValueError(f"CPU tool preflight disagrees with {sample['sample_id']}")
    finally:
        server.close()


def check_sft_encoding(samples: list[dict], split: dict, model: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    selected = choose_rows(samples, split, "train", 0, 7) + choose_rows(samples, split, "validation", 0, 8)
    examples = [encode_row(tokenizer, sample, "gold_only", 4096) for sample in selected]
    return {
        "train_and_validation_encoded": len(examples),
        "max_sequence_tokens": max(len(example["input_ids"]) for example in examples),
        "min_supervised_tokens": min(sum(label != -100 for label in example["labels"]) for example in examples),
        "max_length": 4096,
        "test_not_encoded": True,
    }


def interrupt_on_termination(signum, _frame):
    raise KeyboardInterrupt(f"task received signal {signum}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("base", "sft"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path(os.getenv("STRATA_BUNDLE_DIR", str(DEFAULT_BUNDLE))))
    parser.add_argument("--run-root", type=Path, default=Path(os.getenv(
        "STRATA_RUN_ROOT", "<lab-root>/experiments/strata_shape_ood_20261009")))
    parser.add_argument("--model", type=Path, default=Path(os.getenv(
        "STRATA_MODEL", "<lab-root>/models/Qwen3-0.6B")))
    parser.add_argument("--budget-seconds", type=float,
                        default=float(os.getenv("BUDGET_SECONDS", "0")))
    args = parser.parse_args()
    if args.budget_seconds < 0:
        parser.error("budget-seconds must be nonnegative")

    signal.signal(signal.SIGTERM, interrupt_on_termination)
    started = time.monotonic()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GEOBENCHX_ROOT"] = str(ROOT.parent / "GeoBenchX")

    manifest, samples, split = load_candidate_bundle(args.bundle_dir)
    if not args.model.is_dir() or not (args.model / "config.json").is_file() or not list(args.model.glob("*.safetensors")):
        raise ValueError("base model must be an already downloaded local checkpoint")
    tool_python = os.getenv("STRATA_TOOL_PYTHON") or sys.executable
    source = runtime_source_manifest(ROOT, str(args.model), tool_python)
    source["task_bundle"] = manifest
    metadata = progress_metadata(
        samples, model=str(args.model), evaluation_kind="base", device="cuda",
        attention_implementation="sdpa", max_turns=6, max_new_tokens=1024,
        source_manifest=source)
    metadata["scorer_version"] = SCORER_VERSION
    check_tool_runtime(samples, tool_python)
    base_output = args.run_root / "base" / "report.json"

    if args.check_only:
        base_complete = False
        if base_output.exists():
            report = json.loads(base_output.read_text(encoding="utf-8"))
            validate_base_report(report, samples, metadata, require_complete=False)
            base_complete = report["coverage"]["complete"]
        encoding = check_sft_encoding(samples, split, args.model) if args.stage == "sft" else None
        print(json.dumps({
            "check_only": True,
            "gpu_model_loaded": False,
            "stage": args.stage,
            "samples": len(samples),
            "samples_digest": samples_digest(samples),
            "candidate_manifest": str(args.bundle_dir / "manifest.json"),
            "base_complete": base_complete,
            "run_root": str(args.run_root),
            "sft_encoding": encoding,
            "sft_gate": "ready" if base_complete else "WAIT_FOR_COMPLETE_BASE",
        }, ensure_ascii=False, indent=2))
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.bundle_dir / "manifest.json"
    samples_path = args.bundle_dir / "samples.jsonl"
    split_path = args.bundle_dir / "split_report.json"
    with exclusive_lock(args.run_root / "platform_task.lock") as lock:
        if args.stage == "base":
            return run_logged(
                baseline_command(samples_path, base_output, args.model, "base",
                                 manifest_path, tool_python, args.budget_seconds),
                args.run_root / "logs" / "base.log", lock.fileno())

        if not base_output.exists():
            raise ValueError("run the complete shape-OOD base task first")
        validate_base_report(json.loads(base_output.read_text(encoding="utf-8")), samples, metadata)
        sft_root = args.run_root / "sft_gold48_epoch1_lr5e-6"
        checkpoint = sft_root / "checkpoint"
        recipe = {
            "candidate_manifest": manifest,
            "source_samples_digest": samples_digest(samples),
            "base_metadata_digest": digest_mapping(metadata),
            "train_split": "train", "eval_split": "validation",
            "trajectory": "gold_only", "train_count": 48, "steps": 48,
            "learning_rate": 5e-6, "seed": 7, "max_length": 4096,
            "method": "full_parameter_bfloat16; not LoRA/GRPO",
        }
        recipe_path = sft_root / "recipe.json"
        if recipe_path.exists() and json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
            raise ValueError("SFT recipe changed; choose a new STRATA_RUN_ROOT")
        write_json_atomic(recipe_path, recipe)
        if not checkpoint.exists():
            command = [sys.executable, "-u", str(ROOT / "scripts" / "run_sft.py"),
                       "--samples", str(samples_path), "--split-report", str(split_path),
                       "--model", str(args.model), "--output", str(checkpoint),
                       "--trajectory", "gold_only", "--train-limit", "0", "--eval-limit", "0",
                       "--eval-split", "validation", "--steps", "48", "--learning-rate", "5e-6",
                       "--max-length", "4096", "--seed", "7", "--device", "cuda"]
            result = run_logged(command, args.run_root / "logs" / "sft_train.log", lock.fileno())
            if result:
                return result
        validate_checkpoint(checkpoint, samples, split)
        remaining = args.budget_seconds - (time.monotonic() - started) if args.budget_seconds else 0
        if args.budget_seconds and remaining <= 0:
            print("SFT checkpoint saved; soft budget reached before validation. Re-run the same task.")
            return 75
        return run_logged(
            baseline_command(checkpoint / "eval_samples.jsonl",
                             sft_root / "validation_report.json", checkpoint, "sft-probe",
                             manifest_path, tool_python, remaining),
            args.run_root / "logs" / "sft_validation.log", lock.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
