"""Task worlds built from real, published GIS tables.

The synthetic line can name its own worlds: each case generates its entities, so
"these rows were never in training" is true by construction. Real tables do not
offer that. The entities are countries, they exist once, and every case drawn
from the same table would overlap every other one.

So the entities are partitioned instead. The canonical key space -- the country
codes present in *both* sources -- is split into three disjoint pools, one per
split, and a case only ever draws from its own split's pool. Whether the pools
are disjoint is then a property anyone can check on the emitted candidate rather
than a promise in a docstring.

Two consequences are load-bearing and are recorded rather than glossed:

* **a case's frames are subsets of the real tables, not the tables.** Row ids
  travel with the spec so the subset is reproducible and so nothing downstream
  can describe the frame as "the Natural Earth table".
* **the licence travels too.** These snapshots are redistributable (World Bank
  CC BY 4.0, Natural Earth public domain) but attribution is a condition, not a
  courtesy, so it is part of the spec that reaches the record.

The failure kinds are the same three as the synthetic line, and they are
executed rather than written down -- see `build_real_recovery_plan`. One
candidate wrong key that *looked* plausible (`date`) merges successfully and is
therefore not offered as a mistake anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

LOAD_TOOL_NAME = "load_real_table"
SOURCE_KIND = "real_source_snapshot"
GENERATOR_MODULE = "strata.train.real_data"
GENERATOR_VERSION = "real/wdi-ne-v1"
MANIFEST_VERSION = "real-data-recovery-manifest-v1"
CANONICAL_KEY = "iso3"

# Recorded from the published sources, not from a catalogue page: url, the licence
# as stated by the publisher, the publisher's own version string, and the hash of
# the bytes that were used. The hash is what identifies a snapshot -- a source
# whose hash changes is a different source -- so no retrieval date is kept here.
# A date would be a timeline marker in a repository that carries none, and it
# would be the weaker identifier anyway.
SOURCES: dict[str, dict] = {
    "worldbank_wdi_sp_pop_totl_2021": {
        "publisher": "World Bank",
        "dataset": "World Development Indicators, SP.POP.TOTL (Population, total), 2021",
        "url": ("https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
                "?format=json&per_page=400&date=2021"),
        "licence": "CC BY 4.0",
        "licence_url": "https://data.worldbank.org/summary-terms-of-use",
        "attribution": ("World Bank. World Development Indicators. "
                        "Licensed under CC BY 4.0."),
        "sha256": "05c43fe97cbfae4edfb5fb016c606978bdeba4454a5de2ee8ab9c350071a9164",
        "filename": "wb_sp_pop_totl_2021.json",
        "format": "worldbank_api_json",
    },
    "naturalearth_110m_admin_0_countries": {
        "publisher": "Natural Earth",
        "dataset": "Admin 0 countries, 1:110m",
        "url": ("https://naturalearth.s3.amazonaws.com/110m_cultural/"
                "ne_110m_admin_0_countries.zip"),
        "licence": "public domain",
        "licence_url": "https://www.naturalearthdata.com/about/terms-of-use/",
        "attribution": "Made with Natural Earth. Free vector and raster map data.",
        "source_version": "5.1.1",
        "sha256": "0f243aeac8ac6cf26f0417285b0bd33ac47f1b5bdb719fd3e0df37d03ea37110",
        "filename": "ne_110m_admin_0_countries.zip",
        "format": "naturalearth_shapefile_zip",
    },
}

STATS_SOURCE = "worldbank_wdi_sp_pop_totl_2021"
GEO_SOURCE = "naturalearth_110m_admin_0_countries"
STATS_COLUMNS = ("country", "countryiso3code", "date", "population_total")
GEO_COLUMNS = ("NAME_EN", "ADM0_A3", "POP_EST")
GEO_ATTRIBUTES_FALLBACK = "ne_110m_admin_0_attributes.json"
POOL_NAMES = ("pool_a", "pool_b", "pool_c")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_DIR = REPO_ROOT / "evals" / "sources"

_CACHE: dict[str, object] = {}


def source_path(source_id: str, sources_dir: Path) -> Path:
    return Path(sources_dir) / SOURCES[source_id]["filename"]


def verify_source(source_id: str, sources_dir: Path) -> str:
    """Hash the snapshot on disk and require the recorded value.

    Checked rather than assumed because a silently different input would make
    every downstream record describe a dataset nobody used. The failure is loud
    on purpose: `sha256:unavailable` would let a wrong file through.
    """
    path = source_path(source_id, sources_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"real-data snapshot missing: {path}. The server has no network, so the "
            f"snapshot has to travel with the candidate that uses it.")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    expected = "sha256:" + SOURCES[source_id]["sha256"]
    if digest != expected:
        raise ValueError(
            f"{source_id}: snapshot hash is {digest}, recorded is {expected}. "
            f"A different file is a different source; refusing to build on it.")
    return digest


def _read_source(source_id: str, sources_dir: Path):
    """Load one snapshot as a frame, cached per process.

    Column names are left as the publisher wrote them. Renaming `countryiso3code`
    to something tidy would erase the very mismatch this task family is about.
    """
    key = f"{source_id}:{sources_dir}"
    if key in _CACHE:
        return _CACHE[key]

    import pandas as pd

    verify_source(source_id, sources_dir)
    path = source_path(source_id, sources_dir)
    source = SOURCES[source_id]

    if source["format"] == "worldbank_api_json":
        payload = json.loads(path.read_text(encoding="utf-8"))[1]
        frame = pd.DataFrame([
            {"country": row["country"]["value"],
             "countryiso3code": row["countryiso3code"],
             "date": row["date"],
             "population_total": row["value"]}
            for row in payload
        ])
        # The API mixes real countries with regional aggregates ("Africa Eastern
        # and Southern", code AFE). They have no geometry anywhere, so leaving
        # them in would inflate the stats side with rows that can never join --
        # a mismatch that looks like a key problem and is not one.
        frame = frame[frame["countryiso3code"].astype(str).str.fullmatch(r"[A-Z]{3}")]
        frame = frame[frame["population_total"].notna()]
        frame = frame.drop_duplicates(subset=["countryiso3code"], keep="first")
        frame = frame.reset_index(drop=True)
    elif source["format"] == "naturalearth_shapefile_zip":
        try:
            import geopandas as gpd
        except ModuleNotFoundError:
            fallback = path.with_name(GEO_ATTRIBUTES_FALLBACK)
            if not fallback.is_file():
                raise ModuleNotFoundError(
                    "geopandas is unavailable and the packaged Natural Earth attribute "
                    f"fallback is missing: {fallback}")
            frame = pd.DataFrame(json.loads(fallback.read_text(encoding="utf-8")))
        else:
            frame = gpd.read_file(f"zip://{path.resolve()}")[
                ["NAME_EN", "ADM0_A3", "POP_EST", "geometry"]].copy()
        frame["ADM0_A3"] = frame["ADM0_A3"].astype(str)
        # `ISO_A3` carries the literal "-99" on five rows in this release, which
        # is a real defect of the real data. `ADM0_A3` has none, so it is the
        # key; `ISO_A3` stays available as a genuinely confusing alternative.
        frame = frame[frame["ADM0_A3"].str.fullmatch(r"[A-Z]{3}")]
        frame = frame.reset_index(drop=True)
    else:
        raise ValueError(f"unknown snapshot format {source['format']!r}")

    _CACHE[key] = frame
    return frame


def canonical_keys(sources_dir: Path) -> tuple[str, ...]:
    """Country codes present in both sources, in a stable order."""
    stats = _read_source(STATS_SOURCE, sources_dir)
    geo = _read_source(GEO_SOURCE, sources_dir)
    shared = set(stats["countryiso3code"].astype(str)) & set(geo["ADM0_A3"].astype(str))
    return tuple(sorted(shared))


def entity_pools(sources_dir: Path) -> dict[str, tuple[str, ...]]:
    """The canonical keys split into one disjoint pool per split.

    Sorted first, so the partition does not depend on the order the API happened
    to return rows in. The boundary is recorded with the candidate, and anyone
    can recompute it from the two snapshots.
    """
    keys = canonical_keys(sources_dir)
    if len(keys) < len(POOL_NAMES) * 2:
        raise ValueError(f"only {len(keys)} shared keys; too few to pool")
    size, remainder = divmod(len(keys), len(POOL_NAMES))
    pools: dict[str, tuple[str, ...]] = {}
    start = 0
    for index, name in enumerate(POOL_NAMES):
        stop = start + size + (1 if index < remainder else 0)
        pools[name] = keys[start:stop]
        start = stop
    return pools


@dataclass(frozen=True)
class RealTableSpec:
    """One case's view of one real snapshot: a named, reproducible subset."""

    handle: str
    source: str
    kind: str                       # "stats" | "geo"
    pool: str
    entity_ids: tuple[str, ...]
    columns: tuple[str, ...]

    def as_dict(self) -> dict:
        """Everything needed to rebuild the frame, plus what it came from.

        That includes the licence and the publisher's own identifier: the point
        of a provenance block is that it answers "where did this come from and
        under what terms", not only "which handle was it".
        """
        source = SOURCES[self.source]
        return {
            "handle": self.handle,
            "source_id": self.source,
            "source_url": source["url"],
            "source_sha256": "sha256:" + source["sha256"],
            "source_licence": source["licence"],
            "source_attribution": source["attribution"],
            "source_version": source.get("source_version", ""),
            "kind": self.kind,
            "pool": self.pool,
            "subset_row_ids": list(self.entity_ids),
            "subset_size": len(self.entity_ids),
            "columns": list(self.columns),
            "canonical_key": CANONICAL_KEY,
            "generator_version": GENERATOR_VERSION,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> RealTableSpec:
        return cls(
            handle=str(raw["handle"]),
            source=str(raw["source_id"]),
            kind=str(raw["kind"]),
            pool=str(raw["pool"]),
            entity_ids=tuple(str(value) for value in raw["subset_row_ids"]),
            columns=tuple(str(value) for value in raw["columns"]),
        )


def build_real_table(spec: RealTableSpec, sources_dir: Path):
    """The subset this spec names, with only the declared columns."""
    frame = _read_source(spec.source, sources_dir)
    key_column = "countryiso3code" if spec.kind == "stats" else "ADM0_A3"
    absent = [column for column in spec.columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{spec.handle}: declares columns {absent} that the snapshot "
                         f"does not have; the spec and the source disagree")
    wanted = set(spec.entity_ids)
    selected = frame[frame[key_column].astype(str).isin(wanted)].copy()
    missing = wanted - set(selected[key_column].astype(str))
    if missing:
        raise ValueError(f"{spec.handle}: subset names {len(missing)} entities absent from "
                         f"the snapshot, e.g. {sorted(missing)[:3]}")
    # Ordered by the spec rather than by the snapshot -- two runs over the same
    # subset must give the same frame, whatever order the publisher shipped.
    selected["__order"] = selected[key_column].astype(str).map(
        {entity: index for index, entity in enumerate(spec.entity_ids)})
    selected = selected.sort_values("__order").drop(columns="__order")
    columns = list(spec.columns)
    if spec.kind == "geo":
        columns = columns + ["geometry"]
    return selected[columns].reset_index(drop=True)


class RealLoadTool:
    """Puts a declared real subset into the store, and describes what it stored.

    The registry is the whole spec list, matching the synthetic loader: a loader
    bound to one table answers for one table and errors on the rest with a
    message that reads like bad data.
    """

    def __init__(self, specs: list[RealTableSpec], sources_dir: Path):
        self.registry = {spec.handle: spec for spec in specs}
        self.sources_dir = Path(sources_dir)
        self.name = LOAD_TOOL_NAME
        self.description = ("Load a published GIS table into the workspace and describe "
                            "it. Available handles: " + ", ".join(sorted(self.registry)))
        self.parameters = {
            "type": "object",
            "properties": {"handle": {"type": "string"},
                           "output_dataframe_name": {"type": "string"}},
            "required": ["handle", "output_dataframe_name"],
        }
        self.state: dict = {}

    def execute(self, handle: str = "", output_dataframe_name: str = "", **kwargs) -> str:
        spec = self.registry.get(handle)
        if spec is None:
            return (f"Error: handle {handle!r} is not in this workspace; "
                    f"available: {', '.join(sorted(self.registry))}")
        if not output_dataframe_name:
            return "Error: output_dataframe_name is required"

        frame = build_real_table(spec, self.sources_dir)
        self.state["data_store"][output_dataframe_name] = frame

        from .executor import render_column_names

        return (f"Loaded table '{spec.handle}' as '{output_dataframe_name}'.\n"
                f"Rows: {len(frame)}  Columns: {len(frame.columns)}\n"
                f"Column names (JSON): {render_column_names(frame.columns)}\n"
                f"Missing cells: {int(frame.isna().sum().sum())}")

    def bind(self, state: dict) -> RealLoadTool:
        self.state = state
        return self


def fresh_real_tools(specs: list[RealTableSpec], state: dict, sources_dir: Path) -> dict:
    """The real-data workspace: one snapshot loader plus the benchmark's own tools.

    The four tabular tools are the benchmark's unmodified code, for the same
    reason as on the synthetic side: what has to be independent is the data, not
    the arithmetic, and a re-implementation would drift from what evaluation runs.
    """
    from .slim_tools import synthetic_tabular_tools

    tools = synthetic_tabular_tools(state)
    loader = RealLoadTool(list(specs), sources_dir).bind(state)
    tools[loader.name] = loader
    return tools


@dataclass
class RealJoinCase:
    """An F2 task over two published tables with differently named keys."""

    case_id: str
    pool: str
    split: str
    data_table: RealTableSpec
    geo_table: RealTableSpec
    data_frame: str
    geo_frame: str
    data_key: str
    geo_key: str
    value_column: str
    question: str
    wrong_pairs: list[tuple[str, str]] = field(default_factory=list)
    variant_family: str = "f2_real"
    generator_module: str = GENERATOR_MODULE
    generator_version: str = GENERATOR_VERSION

    @property
    def source_lineage(self) -> str:
        return (f"real:{SOURCES[STATS_SOURCE]['sha256'][:12]}"
                f"+{SOURCES[GEO_SOURCE]['sha256'][:12]}:{self.pool}:{self.case_id}")


SPLIT_FOR_POOL = {"pool_a": "train", "pool_b": "validation", "pool_c": "test"}


def real_recovery_cases(count_per_pool: int = 6, sources_dir: Path | None = None,
                        subset_span: tuple[int, int] = (12, 18)) -> list[RealJoinCase]:
    """One case per (pool, index): a distinct subset from a distinct entity pool."""
    sources_dir = Path(sources_dir) if sources_dir is not None else DEFAULT_SOURCES_DIR
    pools = entity_pools(sources_dir)
    cases: list[RealJoinCase] = []
    for pool in POOL_NAMES:
        members = pools[pool]
        for index in range(count_per_pool):
            size = min(len(members), subset_span[0] + (index % (subset_span[1] - subset_span[0] + 1)))
            start = (index * size) % max(1, len(members) - size + 1)
            subset = tuple(members[start:start + size])
            assert subset, (pool, index)
            data_handle = f"real_{pool}_stats_{index:03d}"
            geo_handle = f"real_{pool}_geo_{index:03d}"
            family = f"f2_real/{pool}"
            data_table = RealTableSpec(handle=data_handle, source=STATS_SOURCE,
                                       kind="stats", pool=pool, entity_ids=subset,
                                       columns=STATS_COLUMNS)
            geo_table = RealTableSpec(handle=geo_handle, source=GEO_SOURCE,
                                      kind="geo", pool=pool, entity_ids=subset,
                                      columns=GEO_COLUMNS)
            cases.append(RealJoinCase(
                case_id=f"real_{pool}_{index:03d}",
                pool=pool, split=SPLIT_FOR_POOL[pool],
                data_table=data_table, geo_table=geo_table,
                data_frame="stats_frame", geo_frame="geo_frame",
                data_key="countryiso3code", geo_key="ADM0_A3",
                value_column="population_total",
                question=(f"Join '{data_handle}' to '{geo_handle}'. "
                          "The two frames do not use the same column name for the key."),
                wrong_pairs=[("country", "ADM0_A3"), ("population_total", "ADM0_A3")],
                variant_family=family,
            ))
    return cases


def build_real_recovery_plan(case: RealJoinCase, error_kind: str = "wrong_role",
                             ) -> list[tuple[str, dict]]:
    """Load both frames, take one real error, then recover correctly.

    The same three kinds as the synthetic line, and the same rule: the mistake is
    executed, never described. `date` is deliberately absent as a wrong key --
    measured, it merges successfully, so offering it would put a mistake in the
    data that is not a mistake.
    """
    from .sample import RECOVERY_ERROR_KINDS

    if error_kind not in RECOVERY_ERROR_KINDS:
        raise ValueError(f"unknown error kind {error_kind!r}; known: "
                         f"{tuple(sorted(RECOVERY_ERROR_KINDS))}")
    wrong = {"dataframe_name": case.data_frame, "geodataframe_name": case.geo_frame,
             "statkey": case.data_key, "geokey": case.geo_key,
             "output_dataframe_name": "failed_merge"}
    if error_kind == "wrong_role":
        wrong["statkey"] = case.geo_key
    elif error_kind == "value_as_key":
        wrong["statkey"] = case.value_column
    else:
        wrong.pop("geokey")
    return [
        (LOAD_TOOL_NAME, {"handle": case.data_table.handle,
                          "output_dataframe_name": case.data_frame}),
        (LOAD_TOOL_NAME, {"handle": case.geo_table.handle,
                          "output_dataframe_name": case.geo_frame}),
        ("merge_dataframes", wrong),
        ("merge_dataframes", {"dataframe_name": case.data_frame,
                              "geodataframe_name": case.geo_frame,
                              "statkey": case.data_key, "geokey": case.geo_key,
                              "output_dataframe_name": "merged"}),
    ]


def real_termination(case: RealJoinCase) -> dict:
    from .sample import TERMINATION_SCHEMA_VERSION

    return {"status": "done", "schema_version": TERMINATION_SCHEMA_VERSION,
            "join_keys": [case.data_key, case.geo_key]}


def check_real_f2_gold(case: RealJoinCase, tools: dict, state: dict):
    """The right key pair must fill, and every offered wrong pair must not.

    Deliberately a near-copy of the synthetic gold check rather than a shared
    helper. The two lines answer the same question about different artefacts,
    and the synthetic one is the check that produced a published result; a
    refactor that touched it to serve this one would put that result behind an
    edit made for a different experiment.
    """
    from .reference import MIN_CORRECT_FILL_RATE, MIN_FILL_MARGIN, JoinCheck, fill_rate

    for table, output in ((case.data_table, case.data_frame),
                          (case.geo_table, case.geo_frame)):
        tools[LOAD_TOOL_NAME].execute(handle=table.handle, output_dataframe_name=output)

    def merged_fill(pair: tuple[str, str], name: str) -> float:
        tools["merge_dataframes"].execute(
            dataframe_name=case.data_frame, geodataframe_name=case.geo_frame,
            statkey=pair[0], geokey=pair[1], output_dataframe_name=name)
        return fill_rate((state.get("data_store") or {}).get(name), case.value_column)

    problems: list[str] = []
    correct = merged_fill((case.data_key, case.geo_key), "gold_merge")
    wrong = [merged_fill(pair, f"gold_bad_{index}")
             for index, pair in enumerate(case.wrong_pairs)]
    if correct < MIN_CORRECT_FILL_RATE:
        problems.append(f"correct pair only fills {correct:.1%}")
    if wrong and correct - max(wrong) < MIN_FILL_MARGIN:
        problems.append(f"best wrong pair fills {max(wrong):.1%}")
    return JoinCheck(case.case_id, not problems, correct_fill_rate=correct,
                     wrong_fill_rates=wrong, problems=problems)


def build_real_recovery_sample(case: RealJoinCase, tools: dict, state: dict,
                               sources_dir: Path, seed: int = 0,
                               error_kind: str = "wrong_role") -> dict:
    from .build import _assemble

    return _assemble(
        case, build_real_recovery_plan(case, error_kind), real_termination(case),
        lambda _store: {"join_keys": [case.data_key, case.geo_key],
                        "value_column": case.value_column},
        family="F2", generator_family=case.variant_family,
        frames=[case.data_frame, case.geo_frame], tools=tools, state=state,
        seed=seed, geobenchx_root=None, source_kind=SOURCE_KIND,
        generator_module=case.generator_module,
        generator_version=case.generator_version,
        lineage_extra={"recovery_error_kind": error_kind, "entity_pool": case.pool,
                       "intended_split": case.split, "canonical_key": CANONICAL_KEY})


def build_real_recovery_batch(cases: list[RealJoinCase], sources_dir: Path,
                              ) -> tuple[list[dict], list[dict]]:
    """Build and validate every case, failing closed on a trace without an error."""
    from ..geobench.tools import benchmark_state
    from .sample import RECOVERY_ERROR_KINDS, validate_sample

    samples: list[dict] = []
    report: list[dict] = []
    kinds = tuple(sorted(RECOVERY_ERROR_KINDS))
    for seed, case in enumerate(cases):
        kind = kinds[seed % len(kinds)]
        state = benchmark_state()
        tools = fresh_real_tools([case.data_table, case.geo_table], state, sources_dir)
        gold = check_real_f2_gold(case, tools, state)
        sample = build_real_recovery_sample(case, tools, state, sources_dir,
                                            seed=seed, error_kind=kind)
        verdict = validate_sample(sample)

        failed = [index for index, step in enumerate(sample["steps"]) if step["failed"]]
        if len(failed) != 1:
            raise ValueError(
                f"{sample['sample_id']}: expected exactly one real failed call for "
                f"{kind!r}, found {len(failed)}; refusing rather than shipping a "
                "trace with no error in it")
        row = {"sample_id": sample["sample_id"], "gold_ok": gold.ok,
               "gold_problems": gold.problems, "validate_ok": verdict.ok,
               "blockers": verdict.blockers, "recovery_error_kind": kind,
               "entity_pool": case.pool, "split": case.split,
               "failed_steps": len(failed),
               "recovered": len(sample["steps"]) > failed[0] + 1
                            and not sample["steps"][failed[0] + 1]["failed"]}
        row.update(gold.as_dict())
        report.append(row)
        samples.append(sample)
    return samples, report
