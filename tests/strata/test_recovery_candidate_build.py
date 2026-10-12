"""The r2 candidate's own gates, and the artifact they produced.

The gates in `scripts/build_recovery_candidate.py` are the ones that decide
whether 144 traces may be called a recovery bundle. A gate nobody exercises is
indistinguishable from a gate that always passes, so each one is broken on
purpose here and required to fail.

The last block runs the same checks against the bundle on disk, because the
interesting failure is not "the function is wrong" but "the artifact was built
before the function was".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_recovery_candidate import (
    ARTIFACTS,
    CHANGED_FACTOR,
    EXACT_SPLIT_COUNTS,
    EXPECTED_SAMPLES,
    PAIRED_BASELINE_NAME,
    WORLD_POLICY,
    check_failed_step_alignment,
    check_kind_distribution,
    check_paired_baseline,
    check_sensitive_files,
    failed_kind_from_arguments,
    load_paired_baseline,
    observation_signature,
    pairing_view,
)
from strata.train.sample import RECOVERY_ERROR_KINDS
from strata.train.shape_ood import recovery_shape_ood_family_cases
from strata.train.synth_families import f2_family_cases

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "evals" / "candidates" / "recovery-ablation"
CASE = f2_family_cases("shared_root__full", 1)[0]


def _step(arguments: dict, failed: bool = True, raw: str = "Error: nope") -> dict:
    return {"call_id": "call_3", "tool": "merge_dataframes", "arguments": arguments,
            "observation_raw": raw, "failed": failed}


def _merge_arguments(**overrides) -> dict:
    arguments = {"dataframe_name": "stats_frame", "geodataframe_name": "geo_frame",
                 "statkey": CASE.data_key, "geokey": CASE.geo_key,
                 "output_dataframe_name": "merged"}
    arguments.update(overrides)
    return arguments


# --- deriving the kind from the call, not from the label -----------------------


def test_the_three_kinds_are_recovered_from_their_arguments():
    assert failed_kind_from_arguments(
        _merge_arguments(statkey=CASE.geo_key), CASE) == "wrong_role"
    assert failed_kind_from_arguments(
        _merge_arguments(statkey=CASE.value_column), CASE) == "value_as_key"
    omitted = _merge_arguments()
    omitted.pop("geokey")
    assert failed_kind_from_arguments(omitted, CASE) == "missing_argument"


def test_a_blank_geokey_counts_as_omitted_not_as_a_wrong_key():
    """An empty string would sail through a `not in` check and change the failure
    class while still being reported as a wrong key."""
    assert failed_kind_from_arguments(_merge_arguments(geokey="  "), CASE) == "missing_argument"


def test_an_unrecognised_statkey_is_not_folded_into_a_known_kind():
    derived = failed_kind_from_arguments(_merge_arguments(statkey="mystery_column"), CASE)
    assert derived.startswith("unknown:")
    assert derived not in RECOVERY_ERROR_KINDS


def test_the_declared_kind_must_be_the_kind_in_the_trace():
    sample = {
        "sample_id": "f2:case",
        "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "value_as_key"},
        "steps": [_step(_merge_arguments(statkey=CASE.geo_key)), _step({}, failed=False, raw="ok")],
    }
    problems = check_failed_step_alignment([sample], {CASE.case_id: CASE})["mismatches"]
    assert problems and "declared 'value_as_key'" in problems[0]


def test_a_trace_that_does_not_recover_is_refused():
    """One failed call, and it is the last one: nothing was recovered."""
    sample = {
        "sample_id": "f2:case",
        "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "value_as_key"},
        "steps": [_step({}, failed=False, raw="ok"),
                  _step(_merge_arguments(statkey=CASE.value_column))],
    }
    problems = check_failed_step_alignment([sample], {CASE.case_id: CASE})["mismatches"]
    assert any("does not end in a successful call" in problem for problem in problems)


def test_two_failed_calls_are_refused_before_anything_else():
    sample = {
        "sample_id": "f2:case",
        "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "value_as_key"},
        "steps": [_step(_merge_arguments(statkey=CASE.value_column)),
                  _step(_merge_arguments(statkey=CASE.value_column))],
    }
    problems = check_failed_step_alignment([sample], {CASE.case_id: CASE})["mismatches"]
    assert any("2 failed steps" in problem for problem in problems)


def test_an_omitted_parameter_must_not_read_like_a_wrong_key():
    """If the two collapse into one message, the bundle teaches only one lesson.

    Checked across the bundle, because a collision is a statement about two
    traces: one trace alone can never demonstrate that the classes are separate.
    """
    omitted = _merge_arguments()
    omitted.pop("geokey")
    shared_message = "Error: ValueError: merge on str and float64 columns"
    samples = [
        {"sample_id": "f2:omitted",
         "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "missing_argument"},
         "steps": [_step(omitted, raw=shared_message), _step({}, failed=False, raw="ok")]},
        {"sample_id": "f2:wrongkey",
         "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "value_as_key"},
         "steps": [_step(_merge_arguments(statkey=CASE.value_column), raw=shared_message),
                   _step({}, failed=False, raw="ok")]},
    ]
    problems = check_failed_step_alignment(samples, {CASE.case_id: CASE})["mismatches"]
    assert any("share an observation signature" in problem for problem in problems)


def test_distinct_failure_classes_do_not_trip_the_collision_check():
    omitted = _merge_arguments()
    omitted.pop("geokey")
    samples = [
        {"sample_id": "f2:omitted",
         "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "missing_argument"},
         "steps": [_step(omitted, raw="Error: TypeError: missing 1 required positional argument"),
                   _step({}, failed=False, raw="ok")]},
        {"sample_id": "f2:wrongkey",
         "lineage": {"template_id": CASE.case_id, "recovery_error_kind": "value_as_key"},
         "steps": [_step(_merge_arguments(statkey=CASE.value_column),
                         raw="Error: ValueError: merge on str and float64"),
                   _step({}, failed=False, raw="ok")]},
    ]
    report = check_failed_step_alignment(samples, {CASE.case_id: CASE})
    assert report["mismatches"] == []
    assert set(report["signatures"]) == {"missing_argument", "value_as_key"}


def test_an_unknown_template_is_reported_rather_than_skipped():
    sample = {"sample_id": "f2:ghost", "lineage": {"template_id": "nope"},
              "steps": [_step(_merge_arguments()), _step({}, failed=False)]}
    problems = check_failed_step_alignment([sample], {})["mismatches"]
    assert problems and "no case for template_id" in problems[0]


# --- balance, globally and where it matters ------------------------------------


def _balanced_rows() -> tuple[list[dict], dict[str, str]]:
    """48 per kind, 16 per kind per split -- the shape the real bundle has."""
    kinds = sorted(RECOVERY_ERROR_KINDS)
    splits = ("train", "validation", "test")
    rows: list[dict] = []
    assignments: dict[str, str] = {}
    for split_index, split in enumerate(splits):
        for kind_index, kind in enumerate(kinds):
            for index in range(16):
                sample_id = f"{kind}_{split}_{index}"
                rows.append({"sample_id": sample_id, "recovery_error_kind": kind})
                assignments[sample_id] = split
    return rows, assignments


def test_a_balanced_bundle_passes():
    rows, assignments = _balanced_rows()
    report = check_kind_distribution(rows, assignments, tuple(sorted(RECOVERY_ERROR_KINDS)))
    assert report["problems"] == []


def test_balance_overall_does_not_hide_a_lumpy_split():
    """48/48/48 overall is satisfiable while test holds one kind only."""
    rows, assignments = _balanced_rows()
    kinds = sorted(RECOVERY_ERROR_KINDS)
    for row in rows:
        if row["recovery_error_kind"] == kinds[0]:
            assignments[row["sample_id"]] = "test"
        else:
            assignments[row["sample_id"]] = "train" if row["sample_id"].endswith(
                ("_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7")) else "validation"
    report = check_kind_distribution(rows, assignments, tuple(sorted(RECOVERY_ERROR_KINDS)))
    assert report["overall"] == {kind: 48 for kind in kinds}
    assert any("kind counts" in problem for problem in report["problems"])


# --- the artifact must not carry a secret --------------------------------------


def test_a_sensitive_file_in_the_output_directory_is_found(tmp_path):
    (tmp_path / "lab_ssh_key").write_text("not-a-real-key", encoding="utf-8")
    assert check_sensitive_files(tmp_path) == ["lab_ssh_key"]


def test_a_clean_output_directory_reports_nothing(tmp_path):
    (tmp_path / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    assert check_sensitive_files(tmp_path) == []


def test_the_builder_refuses_a_directory_holding_a_secret(tmp_path):
    """Cheap because the check runs before any case is generated."""
    from scripts.build_recovery_candidate import build_candidate

    (tmp_path / ".env").write_text("x=1", encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive files present"):
        build_candidate(tmp_path)


def test_the_builder_refuses_to_overwrite_evidence(tmp_path):
    from scripts.build_recovery_candidate import build_candidate

    (tmp_path / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_candidate(tmp_path)


def test_observation_signature_collapses_whitespace_runs():
    assert observation_signature("  Error:   x  ") == "Error:   x"


# --- the pairing that makes the ablation interpretable --------------------------


def _pairable(sample_id: str = "f2:case", *, kind: str = "value_as_key",
              digest: str = "sha256:world", prompt: str = "Join them.",
              failed_arguments: dict | None = None, seconds: float = 0.1) -> dict:
    """One sample shaped like the real ones, small enough to vary by hand."""
    failed = {"call_id": "call_3", "tool": "merge_dataframes",
              "arguments": failed_arguments or {"statkey": "population_2021"},
              "observation_raw": "Error: ValueError: merge on str and float64",
              "observation_visible": "Error: ValueError: merge on str and float64",
              "failed": True, "seconds": seconds}
    ok = {"call_id": "call_4", "tool": "merge_dataframes", "arguments": {"statkey": "id"},
          "observation_raw": "Merged.", "observation_visible": "Merged.",
          "failed": False, "seconds": seconds}
    return {
        "sample_id": sample_id, "family": "F2", "task_text": prompt,
        "source_provenance": {"source_kind": "synthetic_generator"},
        "tool_registry": {"profile": "synthetic_generator"},
        "tools": [{"type": "function", "function": {"name": "merge_dataframes"}}],
        "dataset_digest": digest,
        "lineage": {"template_id": sample_id, CHANGED_FACTOR: kind},
        "gold": {"join_keys": ["id", "code"]},
        "messages": [{"role": "system", "content": "You have exactly these tools:"},
                     {"role": "user", "content": prompt}],
        "steps": [ok, failed],
    }


def test_a_paired_bundle_passes_when_only_the_failing_call_moved():
    baseline = _pairable()
    candidate = _pairable(kind="missing_argument",
                          failed_arguments={"statkey": "id"},
                          seconds=9.9)
    candidate["steps"][1]["observation_raw"] = "Error: TypeError: missing 1 required argument"
    report = check_paired_baseline([candidate], [baseline])
    assert report["problems"] == []
    assert report["identical_outside_changed_factor"] is True
    assert report["changed_error_kinds"] == 1


def test_a_drifted_world_is_not_a_paired_ablation():
    report = check_paired_baseline(
        [_pairable(kind="missing_argument", digest="sha256:other-world")], [_pairable()])
    assert any("differ outside" in problem for problem in report["problems"])
    assert report["identical_outside_changed_factor"] is False


def test_a_drifted_prompt_is_reported_separately_from_the_trajectory():
    """A prompt that moved is a different question, not a different failure."""
    report = check_paired_baseline(
        [_pairable(kind="missing_argument", prompt="Join them differently.")], [_pairable()])
    assert any("initial system/user turns differ" in problem for problem in report["problems"])
    assert report["initial_prompts_identical"] is False


def test_a_pairing_that_changed_nothing_is_refused():
    """Identical bundles are a rerun, not an ablation: nothing was varied."""
    report = check_paired_baseline([_pairable()], [_pairable()])
    assert any("only 0/1 samples changed" in problem for problem in report["problems"])


def test_a_missing_baseline_sample_is_reported():
    report = check_paired_baseline([_pairable()], [_pairable(sample_id="f2:other")])
    assert any("sample ids differ" in problem for problem in report["problems"])


def test_timings_are_not_part_of_the_pairing():
    report = check_paired_baseline([_pairable(seconds=42.0)], [_pairable(seconds=0.001)])
    assert not any("differ outside" in problem for problem in report["problems"])


def test_pairing_view_keeps_the_initial_turns_out_of_the_trajectory_comparison():
    """The messages embed the failing call, so including them would make every
    pair differ and turn a real gate into a permanently red one."""
    left, right = _pairable(), _pairable(kind="missing_argument",
                                        failed_arguments={"statkey": "id"})
    assert pairing_view(left) == pairing_view(right)
    assert left["messages"] == right["messages"]


def test_a_missing_paired_baseline_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="paired baseline samples are required"):
        load_paired_baseline(tmp_path)


# --- the bundle on disk --------------------------------------------------------


def _bundle_or_skip():
    if not (BUNDLE / "manifest.json").is_file():
        pytest.skip("r2 candidate has not been built in this checkout")
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))


def test_the_built_candidate_has_exactly_the_expected_files():
    _bundle_or_skip()
    assert sorted(p.name for p in BUNDLE.iterdir() if p.is_file()) == sorted(ARTIFACTS)


def test_the_built_candidate_counts_are_balanced_everywhere():
    manifest = _bundle_or_skip()
    kinds = tuple(sorted(RECOVERY_ERROR_KINDS))
    assert manifest["kind_counts"] == {kind: 48 for kind in kinds}
    for split in EXACT_SPLIT_COUNTS:
        assert manifest["kind_counts_per_split"][split] == {kind: 16 for kind in kinds}


def test_every_built_trace_failed_for_the_kind_it_declares():
    from scripts.run_recovery import load_candidate_bundle

    if not (BUNDLE / "manifest.json").is_file():
        pytest.skip("r2 candidate has not been built in this checkout")
    _manifest, samples, _split = load_candidate_bundle(BUNDLE)
    cases = {case.case_id: case for case in recovery_shape_ood_family_cases(24)}
    report = check_failed_step_alignment(samples, cases)
    assert report["mismatches"] == []


def test_the_built_candidate_contains_the_failure_the_model_actually_hits():
    """`ValueError: merge on str and float64` is the observed inference failure;
    a bundle without it trains a rule that does not apply."""
    _bundle_or_skip()
    build = json.loads((BUNDLE / "build_report.json").read_text(encoding="utf-8"))
    signatures = build["failed_step_observation_signatures"]["value_as_key"]
    assert any("ValueError" in signature and "merge on" in signature
               for signature in signatures), signatures


def test_the_built_candidate_holds_no_revision_marker_in_its_prompts():
    """`recovery_error_kind` is provenance. If it reached a prompt, the model
    would be told the answer it is being asked to discover."""
    if not (BUNDLE / "samples.jsonl").is_file():
        pytest.skip("r2 candidate has not been built in this checkout")
    for line in (BUNDLE / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        assert sample["lineage"]["recovery_error_kind"]
        for message in sample["messages"]:
            rendered = json.dumps(message, ensure_ascii=False)
            assert "recovery_error_kind" not in rendered
            assert sample["lineage"]["recovery_error_kind"] not in rendered


def test_the_built_candidate_matches_the_declared_sample_count():
    _bundle_or_skip()
    assert sum(1 for _ in (BUNDLE / "samples.jsonl").open(encoding="utf-8")) == EXPECTED_SAMPLES


def test_the_built_candidate_declares_itself_a_paired_ablation():
    """The manifest says what this bundle is, so a later reader cannot mistake
    the shared worlds for an independence failure."""
    manifest = _bundle_or_skip()
    assert manifest["paired_baseline_bundle"] == PAIRED_BASELINE_NAME
    assert manifest["world_policy"] == WORLD_POLICY
    assert manifest["changed_factor"] == CHANGED_FACTOR
    assert manifest["intended_use"] == "paired_error_kind_ablation_candidate"
    audit = json.loads((BUNDLE / "audit.json").read_text(encoding="utf-8"))
    assert audit["paired_baseline_bundle"] == PAIRED_BASELINE_NAME
    assert audit["world_policy"] == WORLD_POLICY
    assert audit["changed_factor"] == CHANGED_FACTOR


def test_the_built_candidate_really_is_its_baseline_with_one_factor_changed():
    """Recheck the pairing from the two files, not from the recorded verdict."""
    _bundle_or_skip()
    baseline_dir = BUNDLE.parent / PAIRED_BASELINE_NAME
    if not (baseline_dir / "samples.jsonl").is_file():
        pytest.skip("paired baseline is not present in this checkout")
    mine = [json.loads(line) for line in
            (BUNDLE / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    theirs = load_paired_baseline(baseline_dir)
    report = check_paired_baseline(mine, theirs, PAIRED_BASELINE_NAME)
    assert report["problems"] == []
    assert report["shared_sample_ids"] == EXPECTED_SAMPLES
    assert report["changed_error_kinds"] == EXPECTED_SAMPLES

