"""Create a frozen offline bundle for the real-data recovery task."""

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
from scripts.run_real import load_candidate_bundle
from strata.train.real_data import SOURCES, source_path, verify_source

CODE_DIRECTORIES = ("corecoder", "strata", "scripts")
CANDIDATE_FILES = (
    "audit.json",
    "build_report.json",
    "manifest.json",
    "metadata.json",
    "replay_report.json",
    "samples.jsonl",
    "split_report.json",
)


def _write_geo_attribute_fallback(source: Path, destination: Path) -> None:
    """Materialise only the table columns needed when geopandas is unavailable."""
    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "packaging the real-data task requires a local geopandas environment "
            "to create the offline attribute fallback") from exc

    frame = gpd.read_file(f"zip://{source.resolve()}")[
        ["NAME_EN", "ADM0_A3", "POP_EST", "geometry"]].copy()
    records = frame.drop(columns=["geometry"]).to_dict(orient="records")
    for record, geometry in zip(records, frame["geometry"], strict=True):
        record["geometry"] = geometry.wkt if geometry is not None else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_file(paths: dict[str, Path], relative: str, source: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required bundle file is missing: {source}")
    paths[relative] = source


def create_bundle(output: Path, candidate: Path, sources_dir: Path,
                  geobenchx_root: Path) -> dict:
    candidate = candidate.resolve()
    sources_dir = sources_dir.resolve()
    geobenchx_root = geobenchx_root.resolve()
    archive = output.with_suffix(".tar.gz")
    if output.exists() or archive.exists():
        raise FileExistsError("bundle/archive already exists; choose a new output name")

    manifest, _samples, _split = load_candidate_bundle(candidate)
    for source_id in SOURCES:
        verify_source(source_id, sources_dir)
    paths: dict[str, Path] = {}
    for directory in CODE_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                _copy_file(paths, "project/" + str(path.relative_to(ROOT)), path)

    for name in CANDIDATE_FILES:
        _copy_file(
            paths,
            f"project/evals/candidates/{candidate.name}/{name}",
            candidate / name,
        )

    for source_id, source in SOURCES.items():
        source_file = source_path(source_id, sources_dir)
        _copy_file(
            paths,
            "project/evals/sources/" + source_file.name,
            source_file,
        )

    geo_fallback = output.parent / ("." + output.name + ".ne_110m_admin_0_attributes.json")
    _write_geo_attribute_fallback(
        source_path("naturalearth_110m_admin_0_countries", sources_dir), geo_fallback)
    paths["project/evals/sources/ne_110m_admin_0_attributes.json"] = geo_fallback

    for name in ("pyproject.toml", "README.md", "LICENSE", "requirements-train.txt"):
        _copy_file(paths, "project/" + name, ROOT / name)

    for path in sorted((geobenchx_root / "geobenchx").rglob("*.py")):
        _copy_file(paths, "GeoBenchX/" + str(path.relative_to(geobenchx_root)), path)

    for relative, source in paths.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    geo_fallback.unlink()

    files = {name: file_digest(output / name) for name in sorted(paths)}
    bundle_manifest = {
        "kind": "strata-f2-real-recovery-platform-task",
        "bundle_schema_version": 1,
        "bundle_id": output.name,
        "bundle_digest": digest_mapping(files),
        "files": files,
        "candidate_id": manifest["candidate_id"],
        "candidate_samples_digest": manifest["samples_digest"],
        "candidate_counts": manifest["counts"],
        "source_snapshots": {
            source_id: {
                "filename": source["filename"],
                "sha256": "sha256:" + source["sha256"],
                "licence": source["licence"],
            }
            for source_id, source in SOURCES.items()
        },
        "recipe": "base48; recovery SFT train16 steps48 lr5e-6 seed7; validation16; test held out",
        "excluded": ["lab_ssh_key", ".env", "models", "venvs", "historical_results",
                     "GeoBenchX/data"],
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
        "source_snapshots": {
            source_id: "sha256:" + source["sha256"] for source_id, source in SOURCES.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path,
                        default=ROOT / "evals" / "candidates" / "real-48")
    parser.add_argument("--sources-dir", type=Path,
                        default=ROOT / "evals" / "sources")
    parser.add_argument("--geobenchx-root", type=Path, default=ROOT.parent / "GeoBenchX")
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.out, args.candidate_dir, args.sources_dir,
                                   args.geobenchx_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
