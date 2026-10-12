"""Real published tables as task worlds.

The synthetic line earns its claim of independence by generating it. This one
cannot: the entities are countries and they exist once, so the separation has to
be built and then checked. Every check here is one that would otherwise be a
sentence in a docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strata.train.real_data import (
    DEFAULT_SOURCES_DIR,
    GENERATOR_VERSION,
    GEO_SOURCE,
    POOL_NAMES,
    SOURCES,
    SPLIT_FOR_POOL,
    STATS_SOURCE,
    RealTableSpec,
    build_real_recovery_plan,
    build_real_table,
    canonical_keys,
    entity_pools,
    real_recovery_cases,
    verify_source,
)

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "evals" / "candidates" / "real-smoke"


# --- the snapshots --------------------------------------------------------------


def test_the_shipped_snapshots_are_the_recorded_ones():
    """A different file is a different source, so the hash is a gate, not a note."""
    for source_id in SOURCES:
        assert verify_source(source_id, DEFAULT_SOURCES_DIR).startswith("sha256:")


def test_a_changed_snapshot_is_refused(tmp_path):
    source = SOURCES[STATS_SOURCE]
    (tmp_path / source["filename"]).write_bytes(b"not the published file")
    with pytest.raises(ValueError, match="A different file is a different source"):
        verify_source(STATS_SOURCE, tmp_path)


def test_a_missing_snapshot_says_so_rather_than_building(tmp_path):
    with pytest.raises(FileNotFoundError, match="no network"):
        verify_source(STATS_SOURCE, tmp_path)


def test_every_source_records_licence_and_attribution():
    """Redistributable is a condition, and the condition travels with the spec.

    No date is recorded for either source: the sha256 is what identifies a
    snapshot, and a retrieval date would be both a weaker identifier and a
    timeline marker this repository does not carry.
    """
    for source_id, meta in SOURCES.items():
        for field in ("url", "licence", "licence_url", "attribution", "sha256"):
            assert meta.get(field), (source_id, field)
        assert not any("utc" in key or "date" in key for key in meta), (source_id, sorted(meta))


# --- the partition --------------------------------------------------------------


def test_the_three_pools_are_pairwise_disjoint():
    pools = entity_pools(DEFAULT_SOURCES_DIR)
    union = [key for members in pools.values() for key in members]
    assert len(union) == len(set(union))
    assert set(pools) == set(POOL_NAMES)


def test_the_pools_cover_exactly_the_shared_keys():
    pools = entity_pools(DEFAULT_SOURCES_DIR)
    union = {key for members in pools.values() for key in members}
    assert union == set(canonical_keys(DEFAULT_SOURCES_DIR))


def test_the_pools_are_evenly_sized():
    sizes = sorted(len(members) for members in entity_pools(DEFAULT_SOURCES_DIR).values())
    assert sizes[-1] - sizes[0] <= 1
    assert sum(sizes) >= 100


def test_each_pool_maps_to_one_split():
    assert set(SPLIT_FOR_POOL) == set(POOL_NAMES)
    assert len(set(SPLIT_FOR_POOL.values())) == len(POOL_NAMES)


# --- one case's table -----------------------------------------------------------


def _case(pool: str = "pool_a", index: int = 0):
    return next(c for c in real_recovery_cases(1) if c.pool == pool and c.case_id.endswith(f"{index:03d}"))


def test_a_subset_is_reproducible():
    spec = _case().data_table
    assert build_real_table(spec, DEFAULT_SOURCES_DIR).equals(
        build_real_table(spec, DEFAULT_SOURCES_DIR))


def test_a_subset_holds_exactly_the_entities_it_names():
    spec = _case().geo_table
    frame = build_real_table(spec, DEFAULT_SOURCES_DIR)
    assert sorted(frame["ADM0_A3"].astype(str)) == sorted(spec.entity_ids)
    assert len(frame) == len(spec.entity_ids)


def test_row_order_follows_the_spec_not_the_publisher():
    spec = _case().geo_table
    frame = build_real_table(spec, DEFAULT_SOURCES_DIR)
    assert list(frame["ADM0_A3"].astype(str)) == list(spec.entity_ids)


def test_a_subset_is_drawn_only_from_its_own_pool():
    pools = entity_pools(DEFAULT_SOURCES_DIR)
    for case in real_recovery_cases(1):
        for spec in (case.data_table, case.geo_table):
            assert set(spec.entity_ids) <= set(pools[case.pool])


def test_cases_in_one_pool_do_not_all_share_a_subset():
    subsets = {case.data_table.entity_ids for case in real_recovery_cases(4)
               if case.pool == "pool_a"}
    assert len(subsets) > 1


def test_a_spec_naming_an_absent_entity_is_refused():
    case = _case()
    broken = RealTableSpec(handle="x", source=GEO_SOURCE, kind="geo", pool=case.pool,
                           entity_ids=("ZZZ",), columns=("NAME_EN", "ADM0_A3"))
    with pytest.raises(ValueError, match="absent from the snapshot"):
        build_real_table(broken, DEFAULT_SOURCES_DIR)


def test_a_spec_declaring_a_column_the_snapshot_lacks_is_refused():
    case = _case()
    broken = RealTableSpec(handle="x", source=STATS_SOURCE, kind="stats", pool=case.pool,
                           entity_ids=case.data_table.entity_ids,
                           columns=("country", "not_a_real_column"))
    with pytest.raises(ValueError, match="the spec and the source disagree"):
        build_real_table(broken, DEFAULT_SOURCES_DIR)


def test_the_declared_columns_are_the_publishers_own_names():
    """Renaming `countryiso3code` would erase the mismatch the task is about."""
    case = _case()
    assert case.data_key == "countryiso3code"
    assert case.geo_key == "ADM0_A3"
    assert case.data_key != case.geo_key


# --- the mistakes ---------------------------------------------------------------


def test_the_rejected_wrong_key_is_offered_nowhere():
    """`date` merges successfully against a string key, so using it as a mistake
    would put a fabricated difficulty in the data."""
    for case in real_recovery_cases(1):
        for pair in case.wrong_pairs:
            assert "date" not in pair
        for kind in ("wrong_role", "value_as_key"):
            arguments = build_real_recovery_plan(case, kind)[2][1]
            assert arguments["statkey"] != "date"


def test_each_kind_puts_a_different_value_in_the_wrong_call():
    case = _case()
    stats_of = {kind: build_real_recovery_plan(case, kind)[2][1]
                for kind in ("wrong_role", "value_as_key", "missing_argument")}
    assert stats_of["wrong_role"]["statkey"] == case.geo_key
    assert stats_of["value_as_key"]["statkey"] == case.value_column
    assert "geokey" not in stats_of["missing_argument"]
    assert stats_of["wrong_role"]["statkey"] != stats_of["value_as_key"]["statkey"]


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown error kind"):
        build_real_recovery_plan(_case(), "not-a-kind")


def test_the_recovery_call_is_the_correct_pair():
    _load_data, _load_geo, _wrong, recovery = build_real_recovery_plan(_case(), "wrong_role")
    assert recovery[1]["statkey"] == "countryiso3code"
    assert recovery[1]["geokey"] == "ADM0_A3"


# --- provenance -----------------------------------------------------------------


def test_every_spec_records_where_it_came_from_and_under_what_terms():
    spec = _case().data_table.as_dict()
    assert spec["source_id"] == STATS_SOURCE
    assert spec["source_sha256"].startswith("sha256:")
    assert spec["source_licence"]
    assert spec["subset_row_ids"]
    assert spec["canonical_key"] == "iso3"
    assert spec["generator_version"] == GENERATOR_VERSION


def test_a_spec_round_trips_through_its_recorded_form():
    spec = _case().geo_table
    assert RealTableSpec.from_dict(spec.as_dict()) == spec


# --- the smoke candidate, if it has been built ----------------------------------


def _smoke_or_skip():
    if not (SMOKE / "metadata.json").is_file():
        pytest.skip("real-data smoke candidate has not been built in this checkout")
    return [json.loads(line) for line in
            (SMOKE / "samples.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_smoke_candidate_keeps_the_pools_apart():
    rows = _smoke_or_skip()
    by_split: dict[str, set[str]] = {}
    for row in rows:
        entities = {entity
                    for spec in row["source_provenance"]["generator"]["specs"]
                    for entity in spec["subset_row_ids"]}
        by_split.setdefault(row["lineage"]["intended_split"], set()).update(entities)
    splits = sorted(by_split)
    for left in range(len(splits)):
        for right in range(left + 1, len(splits)):
            assert not by_split[splits[left]] & by_split[splits[right]], (splits[left], splits[right])


def test_every_smoke_trace_holds_the_failure_its_lineage_declares():
    """Read the observation, not the label: `KeyError` is a wrong key and
    `ValueError ... merge on str and float64` is a real column used as one."""
    rows = _smoke_or_skip()
    expected = {"wrong_role": "KeyError",
                "value_as_key": "merge on str and float64",
                "missing_argument": "missing 1 required positional argument"}
    for row in rows:
        kind = row["lineage"]["recovery_error_kind"]
        failed = [step for step in row["steps"] if step["failed"]]
        assert len(failed) == 1, row["sample_id"]
        assert expected[kind] in failed[0]["observation_raw"], (row["sample_id"], kind)
        assert row["steps"][-1]["failed"] is False


# --- the formal candidate and its runner ----------------------------------------

CANDIDATE = ROOT / "evals" / "candidates" / "real-48"


def _candidate_or_skip():
    if not (CANDIDATE / "manifest.json").is_file():
        pytest.skip("formal real-data candidate has not been built in this checkout")
    return CANDIDATE


def test_the_formal_candidate_loads_and_validates():

    manifest, samples, split = _load()
    assert manifest["training_release"] is False
    assert manifest["gpu_runs"] == 0
    assert len(samples) == 48
    assert set(manifest["counts"].values()) == {16}
    assert len(split["assignments"]) == len(samples)


def _load():
    from scripts.run_real import load_candidate_bundle

    return load_candidate_bundle(_candidate_or_skip())


def test_the_runner_refuses_a_tampered_candidate(tmp_path):
    """The manifest's hashes exist to catch this, so it has to actually catch it."""
    import shutil

    from scripts.run_real import verify_candidate

    target = tmp_path / "candidate"
    shutil.copytree(_candidate_or_skip(), target)
    with (target / "samples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"sample_id": "f2:forged"}) + "\n")
    with pytest.raises(ValueError, match="artifact changed or missing"):
        verify_candidate(target)


def _rewrite_candidate(tmp_path, mutate):
    """Copy the candidate, apply `mutate`, and re-hash so the tamper check passes.

    Without the re-hash every tamper test would stop at "artifact changed" and
    would prove nothing about the gate it names -- the check would look green
    while never running.
    """
    import hashlib
    import shutil

    target = tmp_path / "candidate"
    shutil.copytree(_candidate_or_skip(), target)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in
               (target / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    mutate(samples, manifest)
    (target / "samples.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    # The digest and the artifact hashes are the *first* two gates, so a tamper
    # test that leaves them stale stops there and never reaches the gate it
    # names. Refreshing them is what makes the later assertions mean anything.
    from scripts.baseline_state import samples_digest

    manifest["samples_digest"] = samples_digest(samples)
    manifest["artifact_hashes"] = {
        name: "sha256:" + hashlib.sha256((target / name).read_bytes()).hexdigest()
        for name in manifest["artifact_hashes"]}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                          encoding="utf-8")
    return target


def test_the_runner_refuses_an_entity_that_appears_in_two_splits(tmp_path):
    """The one failure an accuracy number cannot show.

    Caught two ways, and the first one fires here: a split is *defined* by its
    pool, so a foreign entity trips "outside its pool" before it can trip
    "appears in two splits". The cross-split branch is the second line of
    defence, for a candidate whose pool-to-split mapping was itself edited --
    `test_the_runner_refuses_a_split_that_disagrees_with_the_report` covers the
    mapping. Asserting a specific message here would be asserting which of two
    correct gates happened to run first.
    """
    from scripts.run_real import load_candidate_bundle

    def borrow(samples, _manifest):
        stolen = next(s["source_provenance"]["generator"]["specs"][0]["subset_row_ids"][0]
                      for s in samples if s["lineage"]["intended_split"] == "test")
        for sample in samples:
            if sample["lineage"]["intended_split"] == "train":
                sample["source_provenance"]["generator"]["specs"][0]["subset_row_ids"][0] = stolen
                return
        raise AssertionError("no train sample to tamper with")

    with pytest.raises(ValueError, match="outside pool_a|appears in"):
        load_candidate_bundle(_rewrite_candidate(tmp_path, borrow))


def test_the_runner_refuses_a_source_hash_that_is_not_the_recorded_one(tmp_path):
    from scripts.run_real import load_candidate_bundle

    def forge(samples, _manifest):
        samples[0]["source_provenance"]["generator"]["specs"][0]["source_sha256"] = "sha256:deadbeef"

    with pytest.raises(ValueError, match="not the recorded snapshot"):
        load_candidate_bundle(_rewrite_candidate(tmp_path, forge))


def test_the_runner_refuses_a_split_that_disagrees_with_the_report(tmp_path):
    from scripts.run_real import load_candidate_bundle

    def relabel(samples, _manifest):
        samples[0]["lineage"]["intended_split"] = "test"

    with pytest.raises(ValueError, match="split report and lineage disagree"):
        load_candidate_bundle(_rewrite_candidate(tmp_path, relabel))


def test_the_runner_demands_an_explicit_bundle_and_run_root():
    """The synthetic runner's defaults name its own run root; a second experiment
    inheriting them is how results land in the wrong directory."""
    import subprocess
    import sys

    for missing in (["--run-root", "/tmp/x"], ["--bundle-dir", str(CANDIDATE)]):
        proc = subprocess.run(
            [sys.executable, "scripts/run_real.py", "--stage", "base",
             *missing],
            cwd=ROOT, capture_output=True, text=True, check=False)
        assert proc.returncode != 0
        assert "is required" in proc.stderr


def test_real_task_metadata_keeps_the_scorer_version():
    from scripts.run_real import task_metadata
    from strata.train.score import SCORER_VERSION

    _manifest, samples, _split = _load()
    metadata = task_metadata(
        samples,
        model="/model",
        tool_python="/python",
        source_manifest={"task_bundle": "fixed"},
    )

    assert metadata["scorer_version"] == SCORER_VERSION


def test_real_runner_derives_a_training_ready_split_without_mutating_candidate():
    from scripts.run_real import training_split_report

    candidate_split = {
        "policy": "by_entity_pool",
        "assignments": {"sample-a": "train", "sample-b": "validation"},
    }
    derived = training_split_report(candidate_split)

    assert candidate_split.get("training_ready") is None
    assert derived["training_ready"] is True
    assert derived["derived_from"] == "candidate split_report.json"
    assert derived["assignments"] == candidate_split["assignments"]


def test_the_tool_server_builds_the_real_profile_and_measures_a_fill_rate():
    from scripts.tool_server import build_real_tools, real_fill_rate

    case = _case()
    specs = [case.data_table.as_dict(), case.geo_table.as_dict()]
    tools, state = build_real_tools(specs, str(DEFAULT_SOURCES_DIR))
    assert "load_real_table" in tools and "load_data" not in tools
    tools["load_real_table"].execute(handle=case.geo_table.handle,
                                     output_dataframe_name="geo_frame")
    assert len(state["data_store"]["geo_frame"]) == len(case.geo_table.entity_ids)

    correct = real_fill_rate(specs, str(DEFAULT_SOURCES_DIR), "stats_frame", "geo_frame",
                             case.value_column, case.data_table.handle,
                             case.geo_table.handle, (case.data_key, case.geo_key))
    wrong = real_fill_rate(specs, str(DEFAULT_SOURCES_DIR), "stats_frame", "geo_frame",
                           case.value_column, case.data_table.handle,
                           case.geo_table.handle, case.wrong_pairs[0])
    assert correct >= 0.5
    assert wrong == 0.0
