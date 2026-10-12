"""Build and gate the independent F2 recovery candidate (r2).

r1's builder lived *inside* the candidate directory it produced, so the only way
to reproduce it was to run a file that the manifest was simultaneously hashing.
This is the same recipe at a versioned path, plus the gates r2 needs:

* **three error kinds, counted.** r1 emitted one self-labelling failure. The
  rebuilt bundle emits three realistic ones -- a wrong-but-present key from the
  other frame, a real numeric column used as a key, and an omitted required
  parameter -- and the counts are checked per kind *and* per split, because an
  even global split can still be lumpy where it matters.
* **the declared kind is derived back from the arguments.** A record that says
  `value_as_key` while its failed call passed the geo key would train the wrong
  lesson while every aggregate looked right. The kind is recomputed from the
  failed call and must match.
* **sensitive files are absent from the artifact.** Not "we remembered to
  exclude them" -- the output directory is required to contain exactly the
  expected artifacts and nothing else.

Every check here fails closed and runs before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ("samples.jsonl", "build_report.json", "replay_report.json",
             "split_report.json", "manifest.json", "audit.json")
SENSITIVE_NAMES = frozenset({"lab_ssh_key", ".env", "id_rsa", "id_ed25519"})
COUNT_PER_CELL = 24
EXPECTED_SAMPLES = 144
EXACT_SPLIT_COUNTS = {"train": 48, "validation": 48, "test": 48}
# r2 is not an independent candidate: it is r1 with one factor swapped, so the
# comparison between them can be attributed. That makes "the same worlds" a
# requirement to enforce, not a coincidence to tolerate.
PAIRED_BASELINE_NAME = "recovery-baseline"
PAIRED_BASELINE_DIR = ROOT / "evals" / "train" / PAIRED_BASELINE_NAME
WORLD_POLICY = "same_world_paired_ablation"
CHANGED_FACTOR = "recovery_error_kind"
INTENDED_USE = "paired_error_kind_ablation_candidate"
# Wall-clock noise, not content: a paired ablation cannot require equal timings.
PAIRING_IGNORED_STEP_FIELDS = frozenset({"seconds"})


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(output: Path, name: str, value) -> None:
    (output / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def failed_kind_from_arguments(arguments: dict, case) -> str:
    """Recompute the error kind from the call that actually failed.

    The three kinds are distinguishable by the argument that is wrong, and that
    is the point: each is a plausible-looking call a model would really make.
    Returns `unknown:<...>` rather than guessing, so an unrecognised signature
    fails the gate instead of being counted as one of the three.
    """
    arguments = arguments or {}
    geokey = arguments.get("geokey")
    if not arguments or geokey is None or not str(geokey).strip():
        return "missing_argument"
    statkey = str(arguments.get("statkey") or "")
    if statkey == case.value_column:
        return "value_as_key"
    if statkey == case.geo_key:
        return "wrong_role"
    return f"unknown:{statkey}"


def observation_signature(observation: str, length: int = 60) -> str:
    return (observation or "").strip()[:length]


def check_kind_distribution(rows: list[dict], assignments: dict[str, str],
                            kinds: tuple[str, ...]) -> dict:
    """Global and per-split balance, both required.

    A bundle can be 48/48/48 overall and still put every `missing_argument` in
    test, which would make the split an error-type split rather than a shape
    split -- two variables on one axis, with no way afterwards to say which one
    moved.
    """
    overall = Counter(row["recovery_error_kind"] for row in rows)
    per_split: dict[str, Counter] = {name: Counter() for name in EXACT_SPLIT_COUNTS}
    for row in rows:
        per_split[assignments[row["sample_id"]]][row["recovery_error_kind"]] += 1

    expected_overall = {kind: EXPECTED_SAMPLES // len(kinds) for kind in kinds}
    expected_per_split = {kind: EXACT_SPLIT_COUNTS["train"] // len(kinds) for kind in kinds}
    problems = []
    if dict(overall) != expected_overall:
        problems.append(f"kind counts are {dict(overall)}, expected {expected_overall}")
    for split, counts in per_split.items():
        if dict(counts) != expected_per_split:
            problems.append(f"{split} kind counts are {dict(counts)}, expected {expected_per_split}")

    return {
        "overall": {kind: overall.get(kind, 0) for kind in kinds},
        "per_split": {split: {kind: counts.get(kind, 0) for kind in kinds}
                      for split, counts in per_split.items()},
        "problems": problems,
    }


def check_failed_step_alignment(samples: list[dict], cases_by_id: dict) -> dict:
    """The declared kind must be the kind the trace actually contains."""
    mismatches = []
    signatures: dict[str, set[str]] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        lineage = sample.get("lineage") or {}
        # Via `template_id`, not by rebuilding the sample_id: the two must not be
        # able to disagree about which case this trace came from.
        case = cases_by_id.get(str(lineage.get("template_id") or ""))
        if case is None:
            mismatches.append(f"{sample_id}: no case for template_id {lineage.get('template_id')!r}")
            continue
        declared = str(lineage.get("recovery_error_kind") or "")
        failed = [step for step in sample["steps"] if step.get("failed")]
        if len(failed) != 1:
            mismatches.append(f"{sample_id}: {len(failed)} failed steps")
            continue
        step = failed[0]
        derived = failed_kind_from_arguments(step.get("arguments"), case)
        if derived != declared:
            mismatches.append(f"{sample_id}: declared {declared!r}, arguments say {derived!r}")
        if not str(step.get("observation_raw") or "").startswith("Error:"):
            mismatches.append(f"{sample_id}: failed step is not an error observation")
        if sample["steps"][-1].get("failed"):
            mismatches.append(f"{sample_id}: trace does not end in a successful call")
        signatures.setdefault(declared or derived, set()).add(
            observation_signature(step.get("observation_raw") or ""))

    wrong_key_signatures = signatures.get("wrong_role", set()) | signatures.get("value_as_key", set())
    omitted_signatures = signatures.get("missing_argument", set())
    # An omitted required parameter must not read like a wrong key: they are the
    # two halves of the recovery lesson, and if they collapsed into one message
    # the bundle would only be teaching one of them.
    if wrong_key_signatures and omitted_signatures and (wrong_key_signatures & omitted_signatures):
        mismatches.append("omitted-parameter and wrong-key failures share an observation signature")

    return {
        "mismatches": mismatches,
        "signatures": {kind: sorted(values) for kind, values in sorted(signatures.items())},
    }


def check_sensitive_files(output: Path) -> list[str]:
    found = []
    if not output.exists():
        return found
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name in SENSITIVE_NAMES:
            found.append(str(path.relative_to(output)))
    return found


def pairing_view(sample: dict) -> dict:
    """Everything a paired ablation requires to be *identical* across bundles.

    The failing call is the factor under test, so its arguments and observation
    become a placeholder. Two things are left out on purpose:

    * `seconds` -- wall clock, not content;
    * `messages` -- they embed the failing call verbatim, so they would always
      differ; the initial system/user turns are compared separately instead.

    Keeping this list explicit is the point: a field that silently stopped being
    compared would turn "one factor changed" into an unchecked claim.
    """
    steps = []
    for step in sample.get("steps") or []:
        step = dict(step)
        if step.get("failed"):
            step["arguments"] = "<the ablation factor>"
            step["observation_raw"] = "<the ablation factor>"
            step["observation_visible"] = "<the ablation factor>"
        steps.append({key: value for key, value in step.items()
                      if key not in PAIRING_IGNORED_STEP_FIELDS})
    lineage = {key: value for key, value in (sample.get("lineage") or {}).items()
               if key != CHANGED_FACTOR}
    return {
        "family": sample.get("family"),
        "task_text": sample.get("task_text"),
        "source_provenance": sample.get("source_provenance"),
        "tool_registry": sample.get("tool_registry"),
        "tools": sample.get("tools"),
        "lineage": lineage,
        "dataset_digest": sample.get("dataset_digest"),
        "gold": sample.get("gold"),
        "steps": steps,
    }


def check_paired_baseline(samples: list[dict], baseline_samples: list[dict],
                          baseline_name: str = PAIRED_BASELINE_NAME) -> dict:
    """Require that this bundle is its baseline with exactly one factor changed.

    Run as a gate because the claim is what makes the two bundles comparable. If
    a world, a prompt or a gold had drifted, the experiment would still produce
    numbers -- they would just no longer answer the question they were run for.
    """
    mine = {str(sample["sample_id"]): sample for sample in samples}
    theirs = {str(sample["sample_id"]): sample for sample in baseline_samples}
    problems: list[str] = []

    if set(mine) != set(theirs):
        problems.append(
            f"sample ids differ from {baseline_name}: "
            f"{len(set(mine) - set(theirs))} only here, "
            f"{len(set(theirs) - set(mine))} only there")

    diverged: list[str] = []
    kind_changed = 0
    prompt_diverged: list[str] = []
    for sample_id in sorted(set(mine) & set(theirs)):
        here, there = mine[sample_id], theirs[sample_id]
        if pairing_view(here) != pairing_view(there):
            diverged.append(sample_id)
        if (here.get("lineage") or {}).get(CHANGED_FACTOR) != \
                (there.get("lineage") or {}).get(CHANGED_FACTOR):
            kind_changed += 1
        if (here.get("messages") or [])[:2] != (there.get("messages") or [])[:2]:
            prompt_diverged.append(sample_id)

    if diverged:
        problems.append(
            f"{len(diverged)} samples differ outside {CHANGED_FACTOR!r}, "
            f"e.g. {diverged[:3]}")
    if prompt_diverged:
        problems.append(
            f"{len(prompt_diverged)} samples' initial system/user turns differ, "
            f"e.g. {prompt_diverged[:3]}")
    if kind_changed != len(mine):
        problems.append(
            f"only {kind_changed}/{len(mine)} samples changed {CHANGED_FACTOR!r}")

    return {
        "paired_baseline_bundle": baseline_name,
        "world_policy": WORLD_POLICY,
        "changed_factor": CHANGED_FACTOR,
        "baseline_samples": len(theirs),
        "shared_sample_ids": len(set(mine) & set(theirs)),
        "identical_outside_changed_factor": not diverged and set(mine) == set(theirs),
        "initial_prompts_identical": not prompt_diverged,
        "changed_error_kinds": kind_changed,
        "divergent_samples": diverged[:10],
        "prompt_divergent_samples": prompt_diverged[:10],
        "problems": problems,
    }


def load_paired_baseline(baseline_dir: Path) -> list[dict]:
    from scripts.baseline_state import read_jsonl_samples

    samples_path = baseline_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"paired baseline samples are required to prove the pairing: {samples_path}")
    return read_jsonl_samples(samples_path)


def build_candidate(output: Path, count_per_cell: int = COUNT_PER_CELL,
                    paired_baseline: Path | None = None) -> dict:
    from scripts.run_baseline import samples_digest
    from strata.train.replay import replay_all
    from strata.train.sample import RECOVERY_ERROR_KINDS, validate_sample
    from strata.train.schema_groups import content_schema_fingerprint
    from strata.train.shape_ood import (
        RECOVERY_SHAPE_OOD_GENERATOR_VERSION,
        assign_recovery_shape_ood_splits,
        recovery_shape_ood_family_cases,
    )
    from strata.train.split import build_split_report
    from strata.train.synth import build_synth_f2_recovery_batch

    output = output.resolve()
    stray = check_sensitive_files(output)
    if stray:
        raise ValueError(f"sensitive files present in candidate output: {stray}")
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError("recovery evidence already exists; do not overwrite it")
    output.mkdir(parents=True, exist_ok=True)

    kinds = tuple(sorted(RECOVERY_ERROR_KINDS))
    cases = recovery_shape_ood_family_cases(count_per_cell)
    samples, build = build_synth_f2_recovery_batch(cases)
    cases_by_id = {case.case_id: case for case in cases}
    assignments = assign_recovery_shape_ood_splits(samples)
    split = build_split_report(samples, assignments=assignments)
    for sample in samples:
        sample["train_eligible"] = False
        sample["usage_role"] = "f2_recovery_candidate"
        sample["intended_split"] = assignments[sample["sample_id"]]

    replay = replay_all(samples)
    gold_ok = sum(bool(row["gold_ok"]) for row in build)
    validator_ok = sum(bool(row["validate_ok"]) for row in build)
    fresh_ok = sum(validate_sample(sample).ok for sample in samples)
    failed_step_counts = Counter(
        sum(bool(step["failed"]) for step in sample["steps"]) for sample in samples)
    worlds = [(case.data_table.entity_namespace, case.geo_table.entity_namespace)
              for case in cases]
    world_names = [data_world for data_world, _geo in worlds]
    content_splits = split["content_schema_fingerprint_splits"]
    crossings = split["content_schema_fingerprint_crossings"]
    split_counts = Counter(assignments.values())

    errors = []
    if len(samples) != EXPECTED_SAMPLES or len(cases) != EXPECTED_SAMPLES:
        errors.append(f"recovery recipe must contain {EXPECTED_SAMPLES} samples")
    if gold_ok != len(samples) or validator_ok != len(samples):
        errors.append("gold/reference gate failed")
    if fresh_ok != len(samples):
        errors.append("fresh validator gate failed")
    if replay["replayed_equal"] != len(samples) or replay["tools_changed"]:
        errors.append("fresh replay gate failed")
    if dict(failed_step_counts) != {1: EXPECTED_SAMPLES}:
        errors.append(f"each recovery trace must have one failed step: {dict(failed_step_counts)}")
    if len(world_names) != len(set(world_names)):
        errors.append("world namespace is shared across candidate samples")
    if any(data != geo for data, geo in worlds):
        errors.append("data and geo tables do not share their case world")
    if dict(split_counts) != EXACT_SPLIT_COUNTS:
        errors.append(f"unexpected explicit split counts: {dict(split_counts)}")
    if crossings:
        errors.append(f"content schema crosses explicit splits: {crossings}")
    if not split["training_ready"]:
        errors.append(f"explicit candidate split is not ready: {split['training_readiness_reasons']}")
    if any(content_schema_fingerprint(sample) is None for sample in samples):
        errors.append("candidate is missing a content-schema fingerprint")
    if any(not row.get("recovered") for row in build):
        errors.append("candidate contains a trace that does not recover")

    kinds_report = check_kind_distribution(build, assignments, kinds)
    errors.extend(kinds_report["problems"])
    alignment = check_failed_step_alignment(samples, cases_by_id)
    errors.extend(alignment["mismatches"])

    baseline_dir = paired_baseline if paired_baseline is not None else PAIRED_BASELINE_DIR
    pairing = check_paired_baseline(samples, load_paired_baseline(baseline_dir),
                                    baseline_dir.name)
    errors.extend(pairing["problems"])

    if errors:
        raise ValueError("recovery candidate gate failed:\n  - " + "\n  - ".join(errors))

    samples_path = output / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8")
    digest_now = samples_digest(samples)
    write_json(output, "build_report.json", {
        "what": "F2 recovery paired ablation data built offline from real tool execution",
        "not": "not a model run, not API-backed, not yet released as a training task",
        "generator": "strata.train.shape_ood",
        "generator_version": RECOVERY_SHAPE_OOD_GENERATOR_VERSION,
        "trajectory": "recovery",
        "paired_baseline": pairing,
        "failure_contract": (
            "one of three realistic merge failures -- wrong_role (the other frame's key "
            "name), value_as_key (a real numeric column used as a key), missing_argument "
            "(the required geokey omitted) -- followed by the correct merge"
        ),
        "recovery_error_kinds": list(kinds),
        "kind_counts": kinds_report["overall"],
        "kind_counts_per_split": kinds_report["per_split"],
        "failed_step_kind_alignment": "declared kind recomputed from the failed call's arguments",
        "failed_step_observation_signatures": alignment["signatures"],
        "samples_digest": digest_now,
        "samples": len(samples),
        "gold_checked_ok": gold_ok,
        "v1_validate_ok": validator_ok,
        "fresh_validate_ok": fresh_ok,
        "failed_step_counts": {str(k): v for k, v in sorted(failed_step_counts.items())},
        "recovered_ok": sum(1 for row in build if row.get("recovered")),
        "report": build,
    })
    write_json(output, "replay_report.json", replay)
    write_json(output, "split_report.json", split)

    artifact_names = ("samples.jsonl", "build_report.json", "replay_report.json",
                      "split_report.json")
    manifest = {
        "manifest_version": "shape-ood-recovery-manifest-v1",
        "candidate_revision": "r2",
        "intended_use": INTENDED_USE,
        "training_release": False,
        "trajectory": "recovery",
        "label_policy": "failed_tool_call_context_only; successful_recovery_and_termination_supervised",
        # Same worlds as the baseline, on purpose. This bundle is r1 with one
        # factor swapped, so a run on it is only interpretable next to a run on
        # r1 -- and copying those worlds is what makes that comparison clean
        # rather than a coincidence nobody recorded.
        "paired_baseline_bundle": PAIRED_BASELINE_NAME,
        "world_policy": WORLD_POLICY,
        "changed_factor": CHANGED_FACTOR,
        "paired_baseline": pairing,
        "generator": {
            "module": "strata.train.shape_ood",
            "version": RECOVERY_SHAPE_OOD_GENERATOR_VERSION,
            "count_per_cell": count_per_cell,
            "cells": 6,
        },
        "split_policy": {
            "shared_root": "train",
            "case_only": "validation",
            "disjoint_root": "test",
            "full_and_partial_same_shape_same_split": True,
            "explicit_assignments": True,
        },
        "counts": dict(split_counts),
        "recovery_error_kinds": list(kinds),
        "kind_counts": kinds_report["overall"],
        "kind_counts_per_split": kinds_report["per_split"],
        "samples_digest": digest_now,
        "worlds": {
            "count": len(world_names),
            "unique": len(set(world_names)),
            "data_geo_namespace_pairs_match": all(data == geo for data, geo in worlds),
        },
        "content_schema_fingerprint_count": len(content_splits),
        "content_schema_crossings": crossings,
        "source_hashes": {
            relative: digest(ROOT / relative) for relative in (
                "strata/train/synth.py",
                "strata/train/shape_ood.py",
                "strata/train/build.py",
                "strata/train/replay.py",
                "strata/train/split.py",
                "strata/train/schema_groups.py",
                "scripts/build_recovery_candidate.py",
            )
        },
        "artifact_hashes": {name: digest(output / name) for name in artifact_names},
    }
    write_json(output, "manifest.json", manifest)

    audit = {
        "scope": "f2_recovery_candidate_only",
        "candidate_revision": "r2",
        "intended_use": INTENDED_USE,
        "paired_baseline_bundle": PAIRED_BASELINE_NAME,
        "world_policy": WORLD_POLICY,
        "changed_factor": CHANGED_FACTOR,
        "paired_baseline": pairing,
        "training_release": False,
        "model_inference_or_training_runs": 0,
        "api_calls": 0,
        "samples": len(samples),
        "unique_worlds": len(set(world_names)),
        "gold_ok": gold_ok,
        "validator_ok": validator_ok,
        "fresh_validate_ok": fresh_ok,
        "replayed_equal": replay["replayed_equal"],
        "replay_tools_changed": replay["tools_changed"],
        "recovered_ok": sum(1 for row in build if row.get("recovered")),
        "failed_step_counts": {str(k): v for k, v in sorted(failed_step_counts.items())},
        "kind_counts": kinds_report["overall"],
        "kind_counts_per_split": kinds_report["per_split"],
        "failed_step_kind_mismatches": alignment["mismatches"],
        "split_counts": dict(split_counts),
        "training_ready": split["training_ready"],
        "training_readiness_reasons": split["training_readiness_reasons"],
        "content_schema_crossings": crossings,
        "sensitive_files_present": [],
        "samples_digest": digest_now,
    }
    audit["artifact_hashes"] = {
        name: digest(output / name) for name in (*artifact_names, "manifest.json")}
    write_json(output, "audit.json", audit)

    unexpected = sorted(
        str(path.relative_to(output)) for path in output.rglob("*")
        if path.is_file() and path.name not in ARTIFACTS)
    if unexpected:
        raise ValueError(f"candidate directory contains unexpected files: {unexpected}")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count-per-cell", type=int, default=COUNT_PER_CELL)
    parser.add_argument("--paired-baseline", type=Path, default=PAIRED_BASELINE_DIR,
                        help="candidate this bundle must differ from in exactly one factor")
    args = parser.parse_args()
    build_candidate(args.out_dir, args.count_per_cell, args.paired_baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
