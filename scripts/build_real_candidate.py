"""Build the real-data F2 recovery smoke candidate, and gate it before writing.

This is the CPU smoke step of the real-data sub-line: 12-24 traces over published
tables, enough to prove the mechanism end to end without spending a GPU.

The gates here are the ones the synthetic line needed plus the ones this data
makes necessary. Real tables bring three risks the generated ones cannot have:

* **leakage between splits.** Synthetic worlds are disjoint by construction; real
  countries are not, so the partition is checked rather than trusted -- every
  case must draw only from its own pool, and no entity may appear in two splits.
* **provenance that describes the wrong table.** A subset of Natural Earth is not
  "the Natural Earth table". Every sample has to carry the source hash, the
  licence, the pool and the exact row ids it used, and the build refuses if any
  of those is missing.
* **a mistake that is not a mistake.** A wrong key that merges successfully would
  put a fabricated difficulty in the data, so each case's failed call is
  re-executed and its kind recomputed from the arguments it actually passed.
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
             "split_report.json", "manifest.json", "metadata.json", "audit.json")
REQUIRED_PROVENANCE = ("source_id", "source_url", "source_sha256", "source_licence",
                       "source_attribution", "source_version", "pool",
                       "subset_row_ids", "canonical_key", "generator_version")
REQUIRED_LINEAGE = ("recovery_error_kind", "entity_pool", "intended_split", "canonical_key")
COUNT_PER_POOL = 6
SENSITIVE_NAMES = frozenset({"lab_ssh_key", ".env", "id_rsa", "id_ed25519"})


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(output: Path, name: str, value) -> None:
    (output / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_provenance(samples: list[dict]) -> list[str]:
    """Every sample must say which snapshot, which subset, and under what licence."""
    problems: list[str] = []
    for sample in samples:
        specs = ((sample.get("source_provenance") or {}).get("generator") or {}).get("specs")
        if not isinstance(specs, list) or len(specs) != 2:
            problems.append(f"{sample['sample_id']}: expected two source specs")
            continue
        for spec in specs:
            missing = [key for key in REQUIRED_PROVENANCE if not spec.get(key)]
            if missing:
                problems.append(f"{sample['sample_id']}: spec missing {missing}")
            if not spec.get("subset_row_ids"):
                problems.append(f"{sample['sample_id']}: spec names no subset rows")
        lineage = sample.get("lineage") or {}
        absent = [key for key in REQUIRED_LINEAGE if not lineage.get(key)]
        if absent:
            problems.append(f"{sample['sample_id']}: lineage missing {absent}")
    return problems


def check_no_entity_leaks(samples: list[dict], pools: dict) -> list[str]:
    """An entity may live in exactly one pool, and a case may use only its own.

    Checked on the emitted candidate rather than on the builder's intent, because
    the failure this prevents -- the same country in training and in test -- is
    invisible in every aggregate number.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for sample in samples:
        lineage = sample.get("lineage") or {}
        pool = str(lineage.get("entity_pool") or "")
        split = str(lineage.get("intended_split") or "")
        if pool not in pools:
            problems.append(f"{sample['sample_id']}: unknown pool {pool!r}")
            continue
        allowed = set(pools[pool])
        for spec in ((sample.get("source_provenance") or {}).get("generator")
                     or {}).get("specs") or []:
            for entity in spec.get("subset_row_ids") or []:
                if entity not in allowed:
                    problems.append(f"{sample['sample_id']}: entity {entity} is outside {pool}")
                if entity in seen and seen[entity] != split:
                    problems.append(f"entity {entity} appears in both {seen[entity]} and {split}")
                seen[entity] = split
    return problems


def check_split_integrity(samples: list[dict], pools: dict) -> list[str]:
    """One pool per split, one split per pool, and the recorded source hashes.

    The synthetic line's cross-split check is a content-schema fingerprint, and
    it is defined only for generated samples (`content_schema_fingerprint`
    returns None for anything but `synthetic_generator`). That is correct rather
    than a gap: a real case's "schema" is the same two published tables for every
    case, so a fingerprint would be identical everywhere and would check nothing.
    What actually leaks here is **entities**, and that is checked directly.
    """
    from strata.train.real_data import SOURCES

    problems: list[str] = []
    split_to_pool: dict[str, set[str]] = {}
    pool_to_split: dict[str, set[str]] = {}
    for sample in samples:
        lineage = sample.get("lineage") or {}
        split, pool = str(lineage.get("intended_split") or ""), str(lineage.get("entity_pool") or "")
        split_to_pool.setdefault(split, set()).add(pool)
        pool_to_split.setdefault(pool, set()).add(split)
        expected = {name: "sha256:" + meta["sha256"] for name, meta in SOURCES.items()}
        for spec in ((sample.get("source_provenance") or {}).get("generator")
                     or {}).get("specs") or []:
            recorded = str(spec.get("source_sha256") or "")
            if recorded != expected.get(str(spec.get("source_id") or "")):
                problems.append(f"{sample['sample_id']}: source hash {recorded!r} is not the "
                                f"recorded snapshot for {spec.get('source_id')!r}")
    for split, names in split_to_pool.items():
        if len(names) != 1:
            problems.append(f"split {split!r} draws from {sorted(names)} pools")
    for pool, names in pool_to_split.items():
        if len(names) != 1:
            problems.append(f"pool {pool!r} feeds {sorted(names)} splits")
    return problems


def check_failed_call(sample: dict) -> list[str]:
    """Recompute the error kind from the failed call's own arguments."""
    problems: list[str] = []
    failed = [step for step in sample.get("steps") or [] if step.get("failed")]
    if len(failed) != 1:
        return [f"{sample['sample_id']}: {len(failed)} failed steps, expected 1"]
    step = failed[0]
    arguments = step.get("arguments") or {}
    declared = str((sample.get("lineage") or {}).get("recovery_error_kind") or "")
    if not str(step.get("observation_raw") or "").startswith("Error:"):
        problems.append(f"{sample['sample_id']}: failed step is not an error observation")
    if arguments.get("geokey") is None:
        derived = "missing_argument"
    elif str(arguments.get("statkey")) == "population_total":
        derived = "value_as_key"
    elif str(arguments.get("statkey")) == "ADM0_A3":
        derived = "wrong_role"
    else:
        derived = f"unknown:{arguments.get('statkey')}"
    if derived != declared:
        problems.append(f"{sample['sample_id']}: declared {declared!r} but arguments say {derived!r}")
    if (sample.get("steps") or [])[-1].get("failed"):
        problems.append(f"{sample['sample_id']}: trace does not end in a successful call")
    return problems


def build_candidate(output: Path, sources_dir: Path, count_per_pool: int = COUNT_PER_POOL,
                    ) -> dict:
    from strata.train.real_data import (
        CANONICAL_KEY,
        MANIFEST_VERSION,
        SOURCE_KIND,
        SOURCES,
        build_real_recovery_batch,
        entity_pools,
        real_recovery_cases,
    )
    from strata.train.replay import replay_all
    from strata.train.sample import RECOVERY_ERROR_KINDS, validate_sample

    output = output.resolve()
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError("real-data evidence already exists; do not overwrite it")
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name in SENSITIVE_NAMES:
            raise ValueError(f"sensitive file in output directory: {path}")

    sources_dir = Path(sources_dir).resolve()
    pools = entity_pools(sources_dir)
    cases = real_recovery_cases(count_per_pool, sources_dir=sources_dir)
    samples, build = build_real_recovery_batch(cases, sources_dir)
    replay = replay_all(samples, sources_dir=sources_dir)

    kinds = tuple(sorted(RECOVERY_ERROR_KINDS))
    kind_counts = Counter(row["recovery_error_kind"] for row in build)
    per_split = Counter((row["split"], row["recovery_error_kind"]) for row in build)
    split_counts = Counter(case.split for case in cases)

    errors: list[str] = []
    expected = count_per_pool * len(pools)
    if len(samples) != expected:
        errors.append(f"expected {expected} cases, built {len(samples)}")
    if not all(row["gold_ok"] for row in build):
        errors.append(f"gold failed: {[r['sample_id'] for r in build if not r['gold_ok']]}")
    if not all(row["validate_ok"] for row in build):
        errors.append(f"validator failed: {[r['sample_id'] for r in build if not r['validate_ok']]}")
    if not all(validate_sample(s).ok for s in samples):
        errors.append("fresh validator rejected a sample")
    if replay["replayed_equal"] != len(samples) or replay["tools_changed"]:
        errors.append(f"fresh replay failed: {replay['replayed_equal']}/{len(samples)}")
    if not all(row["recovered"] for row in build):
        errors.append("a trace does not recover")
    errors.extend(check_provenance(samples))
    errors.extend(check_split_integrity(samples, pools))
    errors.extend(check_no_entity_leaks(samples, pools))
    for sample in samples:
        errors.extend(check_failed_call(sample))

    if len(pools) != 3 or len({frozenset(v) for v in pools.values()}) != 3:
        errors.append("entity pools are not three distinct sets")
    union = [key for members in pools.values() for key in members]
    if len(union) != len(set(union)):
        errors.append("entity pools overlap")

    if errors:
        raise ValueError("real-data smoke gate failed:\n  - " + "\n  - ".join(errors))

    output.mkdir(parents=True, exist_ok=True)
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8")
    write_json(output, "build_report.json", {
        "what": "real-data F2 recovery smoke candidate, built offline from real tool execution",
        "not": "not a model run, not API-backed, not a released training task, not a capability claim",
        "source_kind": SOURCE_KIND,
        "sources": {name: {key: value for key, value in meta.items() if key != "filename"}
                    for name, meta in SOURCES.items()},
        "canonical_key": CANONICAL_KEY,
        "entity_pool_assignment": {name: list(members) for name, members in pools.items()},
        "cases": len(cases),
        "recovery_error_kinds": list(kinds),
        "kind_counts": {kind: kind_counts.get(kind, 0) for kind in kinds},
        "kind_counts_by_split": {f"{split}|{kind}": count
                                 for (split, kind), count in sorted(per_split.items())},
        "split_counts": dict(split_counts),
        "report": build,
    })
    write_json(output, "replay_report.json", replay)

    # The split is by entity pool, so this report records the pool assignment
    # rather than a grouped split computed from content: every real case shares
    # the same two tables and the same columns, so a content-based grouping would
    # call them all one group. What has to hold is that the pools do not overlap,
    # and `check_no_entity_leaks` checks that directly.
    assignments = {sample["sample_id"]: sample["lineage"]["intended_split"]
                   for sample in samples}
    write_json(output, "split_report.json", {
        "policy": "by_entity_pool",
        "pool_to_split": {"pool_a": "train", "pool_b": "validation", "pool_c": "test"},
        "pool_sizes": {name: len(members) for name, members in pools.items()},
        "assignments": assignments,
        "split_counts": dict(split_counts),
        "entity_pools_pairwise_disjoint": True,
        "entities_used_per_split": {
            split: len({entity for sample in samples
                        if sample["lineage"]["intended_split"] == split
                        for spec in sample["source_provenance"]["generator"]["specs"]
                        for entity in spec["subset_row_ids"]})
            for split in sorted(set(assignments.values()))},
    })

    from scripts.run_baseline import samples_digest

    artifact_names = ("samples.jsonl", "build_report.json", "replay_report.json",
                      "split_report.json")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "candidate_id": output.name,
        "revision": "r1",
        "stage": "cpu-candidate",
        "intended_use": "real_data_recovery_candidate_pending_sft_authorisation",
        "training_release": False,
        "gpu_runs": 0,
        "trajectory": "recovery",
        "count_per_pool": count_per_pool,
        "canonical_key": CANONICAL_KEY,
        "sources": {name: {"url": meta["url"], "licence": meta["licence"],
                           "licence_url": meta["licence_url"],
                           "attribution": meta["attribution"],
                           "sha256": "sha256:" + meta["sha256"]}
                    for name, meta in SOURCES.items()},
        "entity_pools": {name: list(members) for name, members in pools.items()},
        "counts": dict(split_counts),
        "kind_counts": {kind: kind_counts.get(kind, 0) for kind in kinds},
        "samples_digest": samples_digest(samples),
        "artifact_hashes": {name: digest(output / name) for name in artifact_names},
        "not": "no GPU run, no capability claim, must not be merged with the synthetic results",
    }
    write_json(output, "manifest.json", manifest)

    write_json(output, "metadata.json", {
        "candidate_id": output.name,
        "revision": "smoke-1",
        "stage": "cpu-smoke",
        "training_release": False,
        "gpu_runs": 0,
        "intended_use": "mechanism_proof_for_the_real_data_sub_line",
        "splits": {"pool_a": "train", "pool_b": "validation", "pool_c": "test"},
        "entity_pools": {name: len(members) for name, members in pools.items()},
        "canonical_key": CANONICAL_KEY,
        "count_per_pool": count_per_pool,
        "sources": sorted(SOURCES),
        "artifacts": {name: digest(output / name) for name in artifact_names},
        "samples_digest": manifest["samples_digest"],
    })
    write_json(output, "audit.json", {
        "scope": "real_data_smoke_only",
        "gpu_runs": 0,
        "api_calls": 0,
        "samples": len(samples),
        "gold_ok": sum(1 for row in build if row["gold_ok"]),
        "validate_ok": sum(1 for row in build if row["validate_ok"]),
        "replayed_equal": replay["replayed_equal"],
        "kind_counts": {kind: kind_counts.get(kind, 0) for kind in kinds},
        "split_counts": dict(split_counts),
        "entity_pools_pairwise_disjoint": True,
        "subset_row_ids_recorded": True,
        "sensitive_files_present": [],
    })
    print(json.dumps(json.loads((output / "audit.json").read_text(encoding="utf-8")),
                     indent=2, ensure_ascii=False))
    return json.loads((output / "audit.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, default=None)
    parser.add_argument("--count-per-pool", type=int, default=COUNT_PER_POOL)
    args = parser.parse_args()
    from strata.train.real_data import DEFAULT_SOURCES_DIR

    build_candidate(args.out_dir, args.sources_dir or DEFAULT_SOURCES_DIR,
                    args.count_per_pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
