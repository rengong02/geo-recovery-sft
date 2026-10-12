"""Group split tests for the training data gate."""

from __future__ import annotations

import pytest

from strata.train.schema_groups import content_schema_fingerprint
from strata.train.split import (
    GROUP_FIELDS,
    build_split_report,
    make_group_split,
)


def _sample(sample_id: str, snapshot: str, variant: str,
            source_kind: str = "independent_local") -> dict:
    return {
        "sample_id": sample_id,
        "lineage": {
            "dataset_snapshot_id": snapshot,
            "schema_variant_family": variant,
        },
        "source_provenance": {"source_kind": source_kind},
    }


def test_group_split_never_shares_snapshot_or_schema_values():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-b", "schema-b"),
        _sample("c", "data-c", "schema-c"),
    ]
    result = make_group_split(samples)
    assert result.ok
    assert set(result.assignments.values()) == {"train", "validation", "test"}
    for field in GROUP_FIELDS:
        values_by_split = {
            split: {s["lineage"][field] for s in samples
                    if result.assignments[s["sample_id"]] == split}
            for split in ("train", "validation", "test")
        }
        assert not (values_by_split["train"] & values_by_split["validation"])
        assert not (values_by_split["train"] & values_by_split["test"])
        assert not (values_by_split["validation"] & values_by_split["test"])


def test_split_is_blocked_when_fewer_than_three_independent_groups_exist():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-a", "schema-a"),
    ]
    result = make_group_split(samples)
    assert not result.ok
    assert result.reason == "insufficient_independent_groups"


def test_report_exposes_combination_ood_coverage():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-b", "schema-b"),
        _sample("c", "data-c", "schema-c"),
    ]
    report = build_split_report(samples)
    assert report["status"] == "ok"
    assert report["training_ready"] is False
    assert report["ood"]["eligible"] is True
    assert report["ood"]["count"] >= 1


def test_explicit_assignments_must_cover_exactly_the_input_samples():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-b", "schema-b"),
        _sample("c", "data-c", "schema-c"),
    ]

    report = build_split_report(
        samples,
        assignments={"a": "train", "b": "validation"},
    )

    assert report["status"] == "blocked"
    assert report["reason"] == "assignment_missing_sample_id:c"
    assert report["training_ready"] is False


def test_explicit_assignments_reject_unknown_ids_and_invalid_splits():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-b", "schema-b"),
        _sample("c", "data-c", "schema-c"),
    ]

    unknown = build_split_report(
        samples,
        assignments={"a": "train", "b": "validation", "c": "test", "extra": "test"},
    )
    invalid = build_split_report(
        samples,
        assignments={"a": "train", "b": "validation", "c": "holdout"},
    )

    assert unknown["reason"] == "assignment_unknown_sample_id:extra"
    assert invalid["reason"] == "assignment_invalid_split:c:holdout"


def test_explicit_assignments_do_not_bypass_lineage_validation():
    samples = [
        _sample("a", "data-a", "schema-a"),
        _sample("b", "data-b", "schema-b"),
        _sample("c", "data-c", "schema-c"),
    ]
    del samples[1]["lineage"]["schema_variant_family"]

    report = build_split_report(
        samples,
        assignments={"a": "train", "b": "validation", "c": "test"},
    )

    assert report["reason"] == "missing_lineage:schema_variant_family"
    assert report["training_ready"] is False


def test_explicit_assignments_do_not_bypass_group_field_leakage():
    samples = [
        _sample("a", "data-a", "schema-shared"),
        _sample("b", "data-b", "schema-shared"),
        _sample("c", "data-c", "schema-c"),
    ]

    report = build_split_report(
        samples,
        assignments={"a": "train", "b": "validation", "c": "test"},
    )

    assert report["reason"] == "assignment_group_field_cross_split:schema_variant_family:schema-shared"
    assert report["training_ready"] is False


def test_catalog_backed_samples_are_not_training_ready():
    samples = [
        _sample("a", "data-a", "schema-a", "geobenchx_catalog"),
        _sample("b", "data-b", "schema-b", "geobenchx_catalog"),
        _sample("c", "data-c", "schema-c", "geobenchx_catalog"),
    ]
    report = build_split_report(samples)
    assert "catalog_provenance_requires_review" in report["training_readiness_reasons"]
    assert report["training_ready"] is False


def test_a_generator_family_cannot_leak_across_splits():
    samples = [
        {
            **_sample("a", "data-a", "schema-a"),
            "lineage": {
                "dataset_snapshot_id": "data-a",
                "schema_variant_family": "schema-a",
                "generator_family": "same-template",
            },
        },
        {
            **_sample("b", "data-b", "schema-b"),
            "lineage": {
                "dataset_snapshot_id": "data-b",
                "schema_variant_family": "schema-b",
                "generator_family": "same-template",
            },
        },
        {
            **_sample("c", "data-c", "schema-c"),
            "lineage": {
                "dataset_snapshot_id": "data-c",
                "schema_variant_family": "schema-c",
                "generator_family": "same-template",
            },
        },
    ]
    report = build_split_report(samples)
    assert report["generator_family_splits"]["same-template"] == [
        "test", "train", "validation"
    ]
    assert "generator_family_cross_split_leakage" in report["training_readiness_reasons"]
    assert report["training_ready"] is False


def test_training_readiness_requires_enough_examples_per_generator_family():
    samples = []
    for index, family in enumerate(("family-a", "family-b", "family-c")):
        sample = _sample(f"sample-{index}", f"data-{index}", f"schema-{index}")
        sample["lineage"]["generator_family"] = family
        samples.append(sample)

    report = build_split_report(samples)

    assert "insufficient_samples_per_generator_family" in report[
        "training_readiness_reasons"
    ]
    assert report["training_ready"] is False


def _synthetic_sample(sample_id: str, snapshot: str, family: str,
                      *, columns: list[str] | None = None,
                      row_count: int = 30) -> dict:
    columns = columns or ["District", "DISTRICT_ID", "population_2021"]
    return {
        "sample_id": sample_id,
        "lineage": {
            "dataset_snapshot_id": snapshot,
            "schema_variant_family": f"synth/f2/{family}",
            "generator_family": f"f2_synth/{family}",
        },
        "source_provenance": {
            "source_kind": "synthetic_generator",
            "generator": {
                "specs": [{
                    "handle": f"data-{sample_id}",
                    "columns": columns,
                    "n_rows": row_count,
                    "seed": 1,
                    "kind": "stats",
                    "id_column": "DISTRICT_ID",
                    "variant_family": f"synth/f2/{family}",
                }]
            },
        },
    }


def test_content_schema_fingerprint_ignores_instance_identity():
    first = _synthetic_sample("a", "data-a", "family-a", row_count=30)
    second = _synthetic_sample("b", "data-b", "family-b", row_count=45)
    second["source_provenance"]["generator"]["specs"][0]["columns"] = [
        "population_2021", "DISTRICT_ID", "District"
    ]
    assert content_schema_fingerprint(first) == content_schema_fingerprint(second)


def test_report_blocks_content_schema_reuse_across_splits():
    samples = [
        _synthetic_sample("a", "data-a", "family-a"),
        _synthetic_sample("b", "data-b", "family-b"),
        _synthetic_sample("c", "data-c", "family-c"),
    ]
    report = build_split_report(samples)
    assert report["content_schema_fingerprint_splits"]
    assert "content_schema_cross_split_leakage" in report[
        "training_readiness_reasons"
    ]
    assert report["training_ready"] is False


@pytest.mark.parametrize("specs", [None, [{}]])
def test_report_blocks_synthetic_samples_without_valid_content_schema(specs):
    sample = _synthetic_sample("missing", "data-missing", "missing")
    if specs is None:
        del sample["source_provenance"]["generator"]["specs"]
    else:
        sample["source_provenance"]["generator"]["specs"] = specs

    report = build_split_report([
        sample,
        _synthetic_sample("valid-a", "data-a", "valid-a"),
        _synthetic_sample("valid-b", "data-b", "valid-b"),
    ])

    assert content_schema_fingerprint(sample) is None
    assert report["missing_content_schema_fingerprint"] == ["missing"]
    assert "missing_content_schema_fingerprint" in report[
        "training_readiness_reasons"
    ]
    assert report["training_ready"] is False


@pytest.mark.parametrize("generator", [["malformed"], {"specs": [{
    "columns": ["District"],
    "kind": "stats",
    "id_column": "",
}]}])
def test_content_schema_fingerprint_rejects_malformed_generator_metadata(generator):
    sample = _synthetic_sample("malformed", "data-malformed", "malformed")
    sample["source_provenance"]["generator"] = generator

    assert content_schema_fingerprint(sample) is None
