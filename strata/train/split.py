"""Deterministic lineage-aware train/validation/test splitting.

Random row splits are not meaningful for tool-use samples: two rows can share a
dataset snapshot or schema family and differ only in wording. This module keeps
all samples connected by either leakage-bearing value in one component, then
assigns whole components to splits. If the data cannot support three disjoint
components, it returns ``blocked`` instead of inventing a split.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema_groups import content_schema_fingerprint

GROUP_FIELDS = ("dataset_snapshot_id", "schema_variant_family")
SPLITS = ("train", "validation", "test")
MIN_SAMPLES_PER_SPLIT = 2
MIN_SAMPLES_PER_GENERATOR_FAMILY = 24


@dataclass
class SplitResult:
    ok: bool
    reason: str | None
    assignments: dict[str, str]
    groups: list[list[str]]


def _components(samples: list[dict], fields: tuple[str, ...]) -> tuple[list[list[int]], str | None]:
    parent = list(range(len(samples)))
    sample_ids: set[str] = set()
    seen_values: dict[tuple[str, str], int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or "").strip()
        if not sample_id:
            return [], "missing_sample_id"
        if sample_id in sample_ids:
            return [], "duplicate_sample_id"
        sample_ids.add(sample_id)
        lineage = sample.get("lineage")
        if not isinstance(lineage, dict):
            return [], "missing_lineage"
        for field in fields:
            value = str(lineage.get(field) or "").strip()
            if not value:
                return [], f"missing_lineage:{field}"
            key = (field, value)
            if key in seen_values:
                union(index, seen_values[key])
            else:
                seen_values[key] = index

    groups: dict[int, list[int]] = {}
    for index in range(len(samples)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values()), None


def make_group_split(samples: list[dict], fields: tuple[str, ...] = GROUP_FIELDS) -> SplitResult:
    components, reason = _components(samples, fields)
    if reason:
        return SplitResult(False, reason, {}, [])
    if len(components) < len(SPLITS):
        return SplitResult(False, "insufficient_independent_groups", {}, [
            [str(samples[index].get("sample_id")) for index in group]
            for group in components
        ])

    ordered = sorted(
        components,
        key=lambda group: (-len(group), tuple(str(samples[index]["sample_id"]) for index in group)),
    )
    assignments: dict[str, str] = {}
    split_sizes = {split: 0 for split in SPLITS}
    for index, group in enumerate(ordered):
        if index < len(SPLITS):
            split = SPLITS[index]
        else:
            split = min(SPLITS, key=lambda candidate: (split_sizes[candidate], candidate))
        for sample_index in group:
            assignments[str(samples[sample_index]["sample_id"])] = split
        split_sizes[split] += len(group)

    return SplitResult(True, None, assignments, [
        [str(samples[index]["sample_id"]) for index in group]
        for group in ordered
    ])


def _explicit_split(samples: list[dict], assignments: dict[str, str],
                    fields: tuple[str, ...]) -> SplitResult:
    sample_ids = {str(sample.get("sample_id")) for sample in samples}
    assignment_ids = set(assignments)
    unknown = sorted(assignment_ids - sample_ids)
    if unknown:
        return SplitResult(False, f"assignment_unknown_sample_id:{unknown[0]}", {}, [])
    missing = sorted(sample_ids - assignment_ids)
    if missing:
        return SplitResult(False, f"assignment_missing_sample_id:{missing[0]}", {}, [])
    invalid = sorted(
        (sample_id, split)
        for sample_id, split in assignments.items()
        if split not in SPLITS
    )
    if invalid:
        sample_id, split = invalid[0]
        return SplitResult(False, f"assignment_invalid_split:{sample_id}:{split}", {}, [])
    for field in fields:
        values: dict[str, set[str]] = {}
        for sample in samples:
            sample_id = str(sample.get("sample_id"))
            value = str((sample.get("lineage") or {}).get(field) or "").strip()
            values.setdefault(value, set()).add(assignments[sample_id])
        crossings = sorted(
            (value, splits) for value, splits in values.items() if len(splits) > 1
        )
        if crossings:
            value, _splits = crossings[0]
            return SplitResult(
                False,
                f"assignment_group_field_cross_split:{field}:{value}",
                {},
                [],
            )
    groups = [
        sorted(sample_id for sample_id, split in assignments.items() if split == name)
        for name in SPLITS
        if any(split == name for split in assignments.values())
    ]
    return SplitResult(True, None, dict(assignments), groups)


def _ood_report(samples: list[dict], assignments: dict[str, str], fields: tuple[str, ...]) -> dict:
    train_values = {
        field: {str((sample.get("lineage") or {}).get(field))
                for sample in samples if assignments.get(str(sample.get("sample_id"))) == "train"}
        for field in fields
    }
    candidates = []
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        if assignments.get(sample_id) != "test":
            continue
        lineage = sample.get("lineage") or {}
        if all(str(lineage.get(field)) not in train_values[field] for field in fields):
            candidates.append(sample_id)
    return {
        "eligible": bool(candidates),
        "count": len(candidates),
        "sample_ids": candidates,
    }


def _split_values(samples: list[dict], assignments: dict[str, str], field: str) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        value = str((sample.get("lineage") or {}).get(field) or "").strip()
        split = assignments.get(sample_id)
        if value and split:
            values.setdefault(value, set()).add(split)
    return {value: sorted(splits) for value, splits in sorted(values.items())}


def _content_schema_splits(samples: list[dict], assignments: dict[str, str]) -> tuple[dict[str, list[str]], list[str]]:
    values: dict[str, set[str]] = {}
    missing: list[str] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        if (sample.get("source_provenance") or {}).get("source_kind") != "synthetic_generator":
            continue
        fingerprint = content_schema_fingerprint(sample)
        if fingerprint is None:
            missing.append(sample_id)
            continue
        split = assignments.get(sample_id)
        if split:
            values.setdefault(fingerprint, set()).add(split)
    return {value: sorted(splits) for value, splits in sorted(values.items())}, missing


#: Which split each F2 **shape** belongs to, for the paired diagnostic panel.
#:
#: The automatic group split cannot express this: it groups by content schema,
#: so the same shape's `full` and `partial` cells land in different splits, and
#: it reports `content_schema_cross_split_leakage` -- correctly, because that is
#: leakage. But a paired *diagnostic* is not a training set, and its whole point
#: is to compare two cells of one shape against each other. So the assignment is
#: stated explicitly: a shape and both its coverages go to one split, and the
#: shapes are held out from each other.
DIAGNOSTIC_SHAPE_SPLITS = {
    "shared_root": "train",
    "case_only": "validation",
    "disjoint_root": "test",
}


def assign_diagnostic_shapes(samples: list[dict]) -> dict[str, str]:
    """Explicit `{sample_id: split}` for the F2 shape panel.

    Raises on a sample whose shape is not in the map: silently defaulting would
    put an unclassified shape somewhere, and the panel's conclusions depend on
    which shapes the model saw.
    """
    assignments: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        family = str((sample.get("lineage") or {}).get("generator_family") or "")
        shape = family.removeprefix("f2_synth/").split("__", 1)[0]
        if shape not in DIAGNOSTIC_SHAPE_SPLITS:
            raise ValueError(f"unmapped F2 shape {shape!r} in {sample_id!r}")
        assignments[sample_id] = DIAGNOSTIC_SHAPE_SPLITS[shape]
    return assignments


#: What a sample set is *for*. The readiness gates below assume training data,
#: and two of them contradict a diagnostic set by construction:
#:
#: * `insufficient_samples_per_generator_family` wants >= 24 per family, but a
#:   diagnostic cell is small on purpose -- its value is the comparison, not the
#:   sample count;
#: * `content_schema_cross_split_leakage` wants one fingerprint per split, but
#:   isolating a single variable *requires* the cells to share everything else,
#:   content schema included. Measured on the orthogonal F2 set: **all four**
#:   crossing fingerprints are `(shape, full)` / `(shape, partial)` pairs --
#:   the crossings are the design working, not a leak.
#:
#: So the distinction is explicit rather than inferred, and a diagnostic set is
#: reported `training_ready=False` for the honest reason that it is not training
#: data -- not because it failed a gate it was never meant to pass.
INTENDED_USES = ("training", "diagnostic")


def build_split_report(samples: list[dict], fields: tuple[str, ...] = GROUP_FIELDS,
                       intended_use: str = "training",
                       assignments: dict[str, str] | None = None) -> dict:
    if intended_use not in INTENDED_USES:
        raise ValueError(f"unknown intended_use {intended_use!r}; choose from {INTENDED_USES}")
    result = make_group_split(samples, fields)
    if assignments is not None:
        explicit = _explicit_split(samples, assignments, fields)
        if result.reason and result.reason != "insufficient_independent_groups":
            result = SplitResult(False, result.reason, {}, [])
        else:
            result = explicit
    split_counts = {
        split: sum(1 for assignment in result.assignments.values() if assignment == split)
        for split in SPLITS
    }
    report = {
        "group_fields": list(fields),
        "samples": len(samples),
        "status": "ok" if result.ok else "blocked",
        "reason": result.reason,
        "groups": result.groups,
        "assignments": result.assignments,
    }
    report["ood"] = (
        _ood_report(samples, result.assignments, fields)
        if result.ok else {"eligible": False, "count": 0, "sample_ids": []}
    )
    readiness_reasons = []
    if not result.ok:
        readiness_reasons.append(result.reason)
    if any(count < MIN_SAMPLES_PER_SPLIT for count in split_counts.values()):
        readiness_reasons.append("insufficient_samples_per_split")
    if not report["ood"]["eligible"]:
        readiness_reasons.append("no_combination_ood_test")
    source_kinds = {
        str((sample.get("source_provenance") or {}).get("source_kind") or "unknown")
        for sample in samples
    }
    if "geobenchx_catalog" in source_kinds:
        readiness_reasons.append("catalog_provenance_requires_review")
    generator_family_splits = _split_values(samples, result.assignments, "generator_family")
    generator_family_counts = {
        family: sum(
            1 for sample in samples
            if str((sample.get("lineage") or {}).get("generator_family") or "").strip()
            == family
        )
        for family in generator_family_splits
    }
    if generator_family_splits and len(generator_family_splits) < 2:
        readiness_reasons.append("insufficient_generator_families")
    if any(len(splits) > 1 for splits in generator_family_splits.values()):
        readiness_reasons.append("generator_family_cross_split_leakage")
    if (intended_use == "training" and generator_family_counts
            and any(count < MIN_SAMPLES_PER_GENERATOR_FAMILY
                    for count in generator_family_counts.values())):
        readiness_reasons.append("insufficient_samples_per_generator_family")
    content_schema_splits, missing_content_schema = _content_schema_splits(
        samples, result.assignments)
    if missing_content_schema:
        readiness_reasons.append("missing_content_schema_fingerprint")
    # Reported either way; only a blocker for training data.
    fingerprint_crossings = sorted(
        fingerprint for fingerprint, splits in content_schema_splits.items()
        if len(splits) > 1)
    if fingerprint_crossings and intended_use == "training":
        readiness_reasons.append("content_schema_cross_split_leakage")
    if intended_use == "diagnostic":
        # Never training data, said plainly instead of via a gate it cannot pass.
        readiness_reasons.append("intended_use_is_diagnostic")
    report["split_counts"] = split_counts
    report["source_kinds"] = sorted(source_kinds)
    report["generator_family_splits"] = generator_family_splits
    report["generator_family_counts"] = generator_family_counts
    report["content_schema_fingerprint_splits"] = content_schema_splits
    report["content_schema_fingerprint_count"] = len(content_schema_splits)
    report["missing_content_schema_fingerprint"] = missing_content_schema
    report["intended_use"] = intended_use
    report["content_schema_fingerprint_crossings"] = fingerprint_crossings
    report["training_ready"] = not readiness_reasons
    report["training_readiness_reasons"] = readiness_reasons
    return report
