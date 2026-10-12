"""Run the real-data F2 recovery base evaluation and gated recovery SFT task.

Same shape as the synthetic runner, with the checks the real data makes
necessary. Those are not decoration: a real candidate's claims are that the
snapshots are the recorded ones and that no country appears in two splits, and
neither is visible in an accuracy number.

`--check-only` deliberately does **not** require a model checkpoint. The point of
a dry run is to be runnable anywhere -- on a laptop, before the GPU is booked --
and a dry run that needs an A100 image to start is one that never gets run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_state import (
    exclusive_lock,
    file_digest,
    progress_metadata,
    read_jsonl_samples,
    runtime_source_manifest,
    samples_digest,
    write_json_atomic,
)
from scripts.run_helpers import baseline_command, check_tool_runtime, run_logged
from strata.train.real_data import MANIFEST_VERSION as EXPECTED_MANIFEST_VERSION
from strata.train.real_data import SOURCES
from strata.train.score import SCORER_VERSION

SFT_STEPS = 48
SFT_LEARNING_RATE = 5e-6
SFT_SEED = 7
SFT_MAX_LENGTH = 4096
REQUIRED_PROVENANCE = ("source_id", "source_url", "source_sha256", "source_licence",
                       "source_attribution", "pool", "subset_row_ids", "canonical_key",
                       "generator_version")


def expected_source_hashes() -> dict[str, str]:
    return {name: "sha256:" + meta["sha256"] for name, meta in SOURCES.items()}


def verify_candidate(bundle_dir: Path) -> dict:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != EXPECTED_MANIFEST_VERSION:
        raise ValueError("not a real-data recovery candidate manifest")
    if manifest.get("training_release") is not False:
        raise ValueError("candidate is not marked as an unreleased training candidate")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("candidate manifest has no artifact hashes")
    for name, expected in hashes.items():
        target = (bundle_dir / name).resolve()
        if not target.is_file() or file_digest(target) != expected:
            raise ValueError(f"candidate artifact changed or missing: {name}")
    return manifest


def validate_real_dataset(samples: list[dict], split: dict, manifest: dict) -> None:
    """Everything that has to hold before a GPU is spent on this candidate."""
    if samples_digest(samples) != manifest.get("samples_digest"):
        raise ValueError("candidate samples_digest mismatch")
    identifiers = [sample.get("sample_id") for sample in samples]
    assignments = split.get("assignments") or {}
    if len(identifiers) != len(set(identifiers)) or set(assignments) != set(identifiers):
        raise ValueError("candidate split must cover every unique sample exactly once")

    pools = {name: set(members) for name, members in (manifest.get("entity_pools") or {}).items()}
    if len(pools) != 3:
        raise ValueError("candidate must carry three entity pools")
    union = [entity for members in pools.values() for entity in members]
    if len(union) != len(set(union)):
        raise ValueError("candidate entity pools overlap")

    expected_hashes = expected_source_hashes()
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for sample in samples:
        sample_id = sample.get("sample_id")
        lineage = sample.get("lineage") or {}
        split_name = str(lineage.get("intended_split") or "")
        pool = str(lineage.get("entity_pool") or "")
        if assignments.get(sample_id) != split_name:
            raise ValueError(f"{sample_id}: split report and lineage disagree")
        if pool not in pools:
            raise ValueError(f"{sample_id}: unknown entity pool {pool!r}")
        counts[split_name] = counts.get(split_name, 0) + 1

        specs = ((sample.get("source_provenance") or {}).get("generator") or {}).get("specs")
        if not isinstance(specs, list) or len(specs) != 2:
            raise ValueError(f"{sample_id}: expected two source specs")
        for spec in specs:
            for key in REQUIRED_PROVENANCE:
                if not spec.get(key):
                    raise ValueError(f"{sample_id}: source spec is missing {key}")
            if spec["source_sha256"] != expected_hashes.get(str(spec["source_id"])):
                raise ValueError(f"{sample_id}: source hash is not the recorded snapshot")
            for entity in spec["subset_row_ids"]:
                if entity not in pools[pool]:
                    raise ValueError(f"{sample_id}: entity {entity} is outside {pool}")
                if entity in seen and seen[entity] != split_name:
                    raise ValueError(f"entity {entity} appears in {seen[entity]} and {split_name}")
                seen[entity] = split_name

        failed = [step for step in sample.get("steps") or [] if step.get("failed")]
        if len(failed) != 1:
            raise ValueError(f"{sample_id}: {len(failed)} failed steps, expected one")
        if not str(failed[0].get("observation_raw") or "").startswith("Error:"):
            raise ValueError(f"{sample_id}: the failed step is not an error observation")
        if (sample.get("steps") or [])[-1].get("failed"):
            raise ValueError(f"{sample_id}: the trace does not recover")

    if set(counts) != {"train", "validation", "test"}:
        raise ValueError(f"candidate splits are {sorted(counts)}, expected three")
    if len(set(counts.values())) != 1:
        raise ValueError(f"candidate splits are uneven: {counts}")
    if counts != manifest.get("counts"):
        raise ValueError("candidate split counts disagree with its manifest")


def load_candidate_bundle(bundle_dir: Path) -> tuple[dict, list[dict], dict]:
    manifest = verify_candidate(bundle_dir)
    samples = read_jsonl_samples(bundle_dir / "samples.jsonl")
    split = json.loads((bundle_dir / "split_report.json").read_text(encoding="utf-8"))
    validate_real_dataset(samples, split, manifest)
    return manifest, samples, split


def task_metadata(samples: list[dict], model: str, tool_python: str,
                  source_manifest: dict) -> dict:
    metadata = progress_metadata(
        samples, model=model, evaluation_kind="base", device="cuda",
        attention_implementation="sdpa", max_turns=6, max_new_tokens=1024,
        source_manifest=source_manifest)
    metadata["scorer_version"] = SCORER_VERSION
    return metadata


def training_split_report(candidate_split: dict) -> dict:
    derived = dict(candidate_split)
    derived["training_ready"] = True
    derived["derived_from"] = "candidate split_report.json"
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("base", "sft"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path(os.getenv("STRATA_BUNDLE_DIR", "")) if
                        os.getenv("STRATA_BUNDLE_DIR") else None)
    parser.add_argument("--run-root", type=Path, default=Path(os.getenv(
        "STRATA_RUN_ROOT", "")) if os.getenv("STRATA_RUN_ROOT") else None)
    parser.add_argument("--model", type=Path, default=Path(os.getenv(
        "STRATA_MODEL", "")) if os.getenv("STRATA_MODEL") else None)
    parser.add_argument("--budget-seconds", type=float,
                        default=float(os.getenv("BUDGET_SECONDS", "0")))
    args = parser.parse_args()
    # No defaults that point at another experiment. The synthetic runner's two
    # defaults still name its own run root, which is exactly how a second
    # experiment's results end up written into the first one's directory.
    if args.bundle_dir is None:
        parser.error("--bundle-dir is required (no default: this line must not "
                     "silently reuse another candidate)")
    if args.run_root is None:
        parser.error("--run-root is required (no default: results must not land "
                     "in another experiment's directory)")
    if args.budget_seconds < 0:
        parser.error("budget-seconds must be nonnegative")

    started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest, samples, split = load_candidate_bundle(args.bundle_dir)
    model_present = bool(args.model) and args.model.is_dir() and (
        args.model / "config.json").is_file() and bool(list(args.model.glob("*.safetensors")))
    if not args.check_only and not model_present:
        raise ValueError("base model must be an already downloaded local checkpoint")

    tool_python = os.getenv("STRATA_TOOL_PYTHON") or sys.executable
    check_tool_runtime(samples, tool_python)

    if args.check_only:
        print(json.dumps({
            "check_only": True,
            "gpu_model_loaded": False,
            "stage": args.stage,
            "samples": len(samples),
            "samples_digest": samples_digest(samples),
            "candidate_manifest": str(args.bundle_dir / "manifest.json"),
            "counts": manifest.get("counts"),
            "kind_counts": manifest.get("kind_counts"),
            "entities_used_per_split": split.get("entities_used_per_split"),
            "model_present": model_present,
            "run_root": str(args.run_root),
        }, ensure_ascii=False, indent=2))
        return 0

    source = runtime_source_manifest(ROOT, str(args.model), tool_python)
    source["task_bundle"] = manifest
    metadata = task_metadata(samples, str(args.model), tool_python, source)

    args.run_root.mkdir(parents=True, exist_ok=True)
    base_output = args.run_root / "base" / "report.json"
    with exclusive_lock(args.run_root / "platform_task.lock") as lock:
        if args.stage == "base":
            return run_logged(
                baseline_command(args.bundle_dir / "samples.jsonl", base_output,
                                 args.model, "base", args.bundle_dir / "manifest.json",
                                 tool_python, args.budget_seconds),
                args.run_root / "logs" / "base.log", lock.fileno())

        if not base_output.exists():
            raise ValueError("run the complete real-data base task first")
        report = json.loads(base_output.read_text(encoding="utf-8"))
        if report.get("metadata") != metadata:
            raise ValueError("base report metadata does not match this candidate")
        if not report.get("coverage", {}).get("complete"):
            raise ValueError("run the complete real-data base task first")

        per_split = len(samples) // 3
        sft_root = args.run_root / f"sft_recovery{per_split}_epoch1_lr5e-6"
        checkpoint = sft_root / "checkpoint"
        sft_split_path = sft_root / "split_report.json"
        derived_split = training_split_report(split)
        if sft_split_path.exists() and json.loads(sft_split_path.read_text(encoding="utf-8")) != derived_split:
            raise ValueError("derived SFT split report changed; choose a new --run-root")
        write_json_atomic(sft_split_path, derived_split)
        recipe = {
            "candidate_manifest": manifest,
            "source_samples_digest": samples_digest(samples),
            "train_split": "train", "eval_split": "validation",
            "trajectory": "recovery", "train_count": per_split, "steps": SFT_STEPS,
            "learning_rate": SFT_LEARNING_RATE, "seed": SFT_SEED,
            "max_length": SFT_MAX_LENGTH,
            "method": "full_parameter_bfloat16; not LoRA/GRPO",
            "label_policy": ("failed_tool_call_context_only; "
                             "successful_recovery_and_termination_supervised"),
        }
        recipe_path = sft_root / "recipe.json"
        if recipe_path.exists() and json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
            raise ValueError("SFT recipe changed; choose a new --run-root")
        write_json_atomic(recipe_path, recipe)
        if not checkpoint.exists():
            command = [sys.executable, "-u", str(ROOT / "scripts" / "run_sft.py"),
                       "--samples", str(args.bundle_dir / "samples.jsonl"),
                       "--split-report", str(sft_split_path),
                       "--model", str(args.model), "--output", str(checkpoint),
                       "--trajectory", "recovery", "--train-limit", "0", "--eval-limit", "0",
                       "--eval-split", "validation", "--steps", str(SFT_STEPS),
                       "--learning-rate", str(SFT_LEARNING_RATE),
                       "--max-length", str(SFT_MAX_LENGTH), "--seed", str(SFT_SEED),
                       "--device", "cuda"]
            result = run_logged(command, args.run_root / "logs" / "sft_train.log",
                                lock.fileno())
            if result:
                return result
        remaining = (args.budget_seconds - (time.monotonic() - started)
                     if args.budget_seconds else 0)
        if not checkpoint.exists():
            raise ValueError("SFT checkpoint was not produced")
        return run_logged(
            baseline_command(checkpoint / "eval_samples.jsonl",
                             sft_root / "validation_report.json", checkpoint, "sft-probe",
                             args.bundle_dir / "manifest.json", tool_python, remaining),
            args.run_root / "logs" / "sft_validation.log", lock.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
