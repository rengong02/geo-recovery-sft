"""Freeze the F2 recovery candidate and platform task for upload."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_state import digest_mapping, file_digest, write_json_atomic
from scripts.run_recovery import DEFAULT_BUNDLE, load_candidate_bundle


def create_bundle(output: Path, candidate: Path, geobenchx_root: Path) -> dict:
    candidate = candidate.resolve()
    geobenchx_root = geobenchx_root.resolve()
    archive = output.with_suffix(".tar.gz")
    if output.exists() or archive.exists():
        raise FileExistsError("bundle/archive already exists; choose a new output name")
    manifest, _samples, _split = load_candidate_bundle(candidate)
    paths: dict[str, Path] = {}
    for directory in ("corecoder", "strata", "scripts"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                paths["project/" + str(path.relative_to(ROOT))] = path
    for name in sorted({
        "samples.jsonl", "build_report.json", "replay_report.json", "split_report.json",
        "manifest.json", "audit.json", "audit_candidates.py",
    }):
        source = candidate / name
        if source.is_file():
            paths[f"project/evals/candidates/{candidate.name}/{name}"] = source
    for name in ("pyproject.toml", "README.md", "LICENSE", "requirements-train.txt"):
        paths["project/" + name] = ROOT / name
    for path in sorted((geobenchx_root / "geobenchx").rglob("*.py")):
        paths["GeoBenchX/" + str(path.relative_to(geobenchx_root))] = path

    for relative, source in paths.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    files = {name: file_digest(output / name) for name in sorted(paths)}
    bundle_manifest = {
        "kind": "strata-f2-recovery-platform-task",
        "bundle_schema_version": 1,
        "bundle_id": output.name,
        "bundle_digest": digest_mapping(files),
        "files": files,
        "candidate_manifest": manifest,
        "recipe": "base144; recovery SFT train48 steps48 lr5e-6 seed7; validation48; test held out",
        "excluded": ["lab_ssh_key", ".env", "models", "venvs", "historical_results", "GeoBenchX/data"],
    }
    write_json_atomic(output / "bundle_manifest.json", bundle_manifest)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    return {
        "bundle": str(output.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256": file_digest(archive),
        "files": len(files),
        "bundle_digest": bundle_manifest["bundle_digest"],
        "candidate_samples_digest": manifest["samples_digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--geobenchx-root", type=Path, default=ROOT.parent / "GeoBenchX")
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.out, args.candidate_dir, args.geobenchx_root),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
