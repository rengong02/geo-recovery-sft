"""Independent shape-OOD F2 candidates with explicit split intent."""

from __future__ import annotations

from .synth import SynthJoinCase, SynthTableSpec
from .synth_families import F2_COVERAGE, F2_SHAPES

SHAPE_OOD_SPLITS = {
    "shared_root": "train",
    "case_only": "validation",
    "disjoint_root": "test",
}
SHAPE_OOD_GENERATOR_VERSION = "synth/f2-shape-ood-v1"
RECOVERY_SHAPE_OOD_SPLITS = dict(SHAPE_OOD_SPLITS)
RECOVERY_SHAPE_OOD_GENERATOR_VERSION = "synth/f2-shape-ood-recovery-v1"
_SHAPE_OOD_PREFIX = "synth/shape_ood"
_RECOVERY_SHAPE_OOD_PREFIX = "synth/recovery_shape_ood"
_SEED_BASE = 20_000
_RECOVERY_SEED_BASE = 30_000


def _family_cases(
    count_per_cell: int,
    *,
    seed_base: int,
    family_prefix: str,
    case_prefix: str,
    handle_prefix: str,
    world_prefix: str,
    generator_version: str,
) -> list[SynthJoinCase]:
    """Build F2 cells whose worlds are unique across every emitted sample.

    Unlike the paired diagnostic panel, full and partial cases do not share a
    world. The only intended reuse is the content schema within a shape, and
    both coverages are assigned to that shape's one split.
    """
    if count_per_cell < 1:
        raise ValueError("count_per_cell must be positive")

    cases: list[SynthJoinCase] = []
    world_index = 0
    for shape, (data_name, data_key, geo_name, geo_key) in F2_SHAPES.items():
        for coverage in F2_COVERAGE:
            for index in range(count_per_cell):
                seed = seed_base + world_index * 17
                world = f"{world_prefix}_{world_index:03d}"
                data_handle = f"{handle_prefix}_data_{world_index:03d}"
                geo_handle = f"{handle_prefix}_geo_{world_index:03d}"
                rows = 30 + (seed % 13)
                geo_rows = max(2, round(rows * F2_COVERAGE[coverage]))
                family = f"{family_prefix}/{shape}__{coverage}"
                data_table = SynthTableSpec(
                    handle=data_handle,
                    columns=(data_name, data_key, "population_2021"),
                    n_rows=rows,
                    seed=seed,
                    kind="stats",
                    id_column=data_key,
                    variant_family=family,
                    entity_namespace=world,
                )
                geo_table = SynthTableSpec(
                    handle=geo_handle,
                    columns=(geo_name, geo_key),
                    n_rows=geo_rows,
                    seed=seed,
                    kind="regions",
                    id_column=geo_key,
                    variant_family=family,
                    entity_namespace=world,
                )
                cases.append(SynthJoinCase(
                    case_id=f"{case_prefix}_{shape}__{coverage}_{index:03d}",
                    data_table=data_table,
                    geo_table=geo_table,
                    data_frame="stats_frame",
                    geo_frame="geo_frame",
                    data_key=data_key,
                    geo_key=geo_key,
                    value_column="population_2021",
                    question=(f"Join '{data_handle}' to '{geo_handle}'. "
                              "The two frames do not use the same column name for the key."),
                    wrong_pairs=[(data_name, geo_key), ("population_2021", geo_key)],
                    variant_family=family,
                    generator_module="strata.train.shape_ood",
                    generator_version=generator_version,
                ))
                world_index += 1
    return cases


def shape_ood_family_cases(count_per_cell: int = 24) -> list[SynthJoinCase]:
    return _family_cases(
        count_per_cell,
        seed_base=_SEED_BASE,
        family_prefix=_SHAPE_OOD_PREFIX,
        case_prefix="shape_ood",
        handle_prefix="shape_ood",
        world_prefix="shape_ood_world",
        generator_version=SHAPE_OOD_GENERATOR_VERSION,
    )


def recovery_shape_ood_family_cases(count_per_cell: int = 24) -> list[SynthJoinCase]:
    return _family_cases(
        count_per_cell,
        seed_base=_RECOVERY_SEED_BASE,
        family_prefix=_RECOVERY_SHAPE_OOD_PREFIX,
        case_prefix="recovery_shape_ood",
        handle_prefix="recovery_shape_ood",
        world_prefix="recovery_shape_ood_world",
        generator_version=RECOVERY_SHAPE_OOD_GENERATOR_VERSION,
    )


def _assign_family_splits(samples: list[dict], prefix: str,
                          split_policy: dict[str, str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "").strip()
        family = str((sample.get("lineage") or {}).get("generator_family") or "")
        if not sample_id or not family.startswith(prefix + "/"):
            raise ValueError(f"unmapped shape-OOD sample {sample_id!r}")
        shape = family[len(prefix) + 1:].split("__", 1)[0]
        if shape not in split_policy:
            raise ValueError(f"unmapped shape-OOD shape {shape!r} in {sample_id!r}")
        if sample_id in assignments:
            raise ValueError(f"duplicate shape-OOD sample {sample_id!r}")
        assignments[sample_id] = split_policy[shape]
    return assignments


def assign_shape_ood_splits(samples: list[dict]) -> dict[str, str]:
    """Assign every generated sample by shape, failing closed on unknown data."""
    return _assign_family_splits(samples, _SHAPE_OOD_PREFIX, SHAPE_OOD_SPLITS)


def assign_recovery_shape_ood_splits(samples: list[dict]) -> dict[str, str]:
    """Assign recovery samples by shape, independently from the old candidate."""
    return _assign_family_splits(
        samples, _RECOVERY_SHAPE_OOD_PREFIX, RECOVERY_SHAPE_OOD_SPLITS)
