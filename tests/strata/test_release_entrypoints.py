import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_real_candidate_rebuilds_with_recorded_snapshot_hashes(tmp_path):
    from scripts.build_real_candidate import build_candidate
    from scripts.run_real import load_candidate_bundle
    from strata.train.real_data import DEFAULT_SOURCES_DIR

    output = tmp_path / "real"
    build_candidate(output, DEFAULT_SOURCES_DIR, 16)
    manifest, samples, split = load_candidate_bundle(output)
    from scripts.baseline_state import samples_digest

    assert manifest["samples_digest"] == samples_digest(samples)
    assert all(
        spec["source_sha256"].startswith("sha256:")
        for sample in samples
        for spec in sample["source_provenance"]["generator"]["specs"]
    )
    assert manifest["counts"] == {"train": 16, "validation": 16, "test": 16}
    assert len(samples) == len(split["assignments"]) == 48


@pytest.mark.parametrize("stage", ["base", "sft"])
def test_real_wrapper_checks_candidate_without_model_or_lab_paths(tmp_path, stage):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STRATA_")}
    environment["STRATA_PYTHON"] = sys.executable
    environment["STRATA_RUN_ROOT"] = str(tmp_path / "run")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / f"run_real_{stage}.sh"), "--check-only"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["samples"] == 48
    assert report["counts"] == {"train": 16, "validation": 16, "test": 16}
    assert report["gpu_model_loaded"] is False
    assert report["model_present"] is False


def test_offline_package_defaults_preserve_sources_without_geobenchx(tmp_path):
    output = tmp_path / "offline"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "package_real.py"),
         "--out", str(output), "--geobenchx-root", str(tmp_path / "absent")],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive) as handle:
        assert "offline/project/evals/sources/ne_110m_admin_0_attributes.json" in handle.getnames()
    manifest = json.loads((output / "bundle_manifest.json").read_text())
    assert manifest["candidate_counts"] == {"train": 16, "validation": 16, "test": 16}
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STRATA_") and key != "GEOBENCHX_ROOT"}
    environment["STRATA_PYTHON"] = sys.executable
    environment["STRATA_RUN_ROOT"] = str(tmp_path / "run")
    check = subprocess.run(
        ["bash", str(output / "project" / "scripts" / "run_real_base.sh"), "--check-only"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60, check=False)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["samples"] == 48
