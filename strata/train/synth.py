"""Generate independent task tables and execute recovery trajectories with local tabular tools."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from .sample import RECOVERY_ERROR_KINDS, TERMINATION_SCHEMA_VERSION

#: The synthetic loader's name. Chosen so a reader sees immediately that this
#: sample's data was generated rather than taken from the benchmark catalog.
LOAD_TOOL_NAME = "load_synthetic_table"
SYNTH_GENERATOR_MODULE = "strata.train.synth"
SYNTH_GENERATOR_VERSION = "synth/v1"


@dataclass(frozen=True)
class SynthTableSpec:
    """A generated table: its schema, its size, and what its columns mean."""

    handle: str
    columns: tuple[str, ...]
    n_rows: int
    seed: int
    kind: str = "stats"          # "stats" | "regions"
    id_column: str | None = None
    variant_family: str = "synth/v1"
    entity_namespace: str | None = None

    def as_dict(self) -> dict:
        result = {"handle": self.handle, "columns": list(self.columns),
                  "n_rows": self.n_rows, "seed": self.seed, "kind": self.kind,
                  "id_column": self.id_column, "variant_family": self.variant_family}
        if self.entity_namespace is not None:
            result["entity_namespace"] = self.entity_namespace
        return result


def _entity_rows(n: int, seed: int, namespace: str | None = None) -> list[tuple[str, str]]:
    """`(code, name)` pairs from a seeded generator.

    Names and codes are invented here rather than taken from any gazetteer, so
    nothing in the training set traces back to a published dataset.
    """
    rng = random.Random(seed)
    syllables = ["ar", "bel", "cor", "dan", "el", "fal", "gor", "hal", "ir",
                 "jun", "kal", "lom", "mar", "nor", "oth", "pel"]
    rows: list[tuple[str, str]] = []
    for i in range(n):
        code = f"R{i:03d}"
        name = f"{rng.choice(syllables).capitalize()}{rng.choice(syllables)}"
        if namespace:
            code = f"{namespace}:{code}"
            name = f"{namespace}:{name}"
        rows.append((code, name if name else f"Region{i:03d}"))
    return rows


def build_table(spec: SynthTableSpec) -> pd.DataFrame:
    """Materialise a spec. Deterministic: the same spec gives the same frame, so
    a sample built from it replays to itself."""
    rng = random.Random(spec.seed)
    entities = _entity_rows(spec.n_rows, spec.seed, spec.entity_namespace)
    data: dict[str, list] = {column: [] for column in spec.columns}

    for index, (code, name) in enumerate(entities):
        for column in spec.columns:
            lowered = column.lower()
            if (spec.id_column and column == spec.id_column
                    or "code" in lowered or lowered in ("iso_a3", "iso3")):
                data[column].append(code)
            elif ("name" in lowered or "economy" in lowered or "country" == lowered
                  or lowered in {"region", "district", "zone"}):
                data[column].append(name)
            elif "indicator" in lowered:
                data[column].append(f"Indicator {spec.seed % 7}")
            else:
                # A value column: seeded, with a few holes so a join that
                # misses rows is visible as missing values rather than as a
                # coincidentally-full column.
                data[column].append(
                    None if rng.random() < 0.05 else round(rng.uniform(0, 1000), 4))
    return pd.DataFrame(data)


class SyntheticLoadTool:
    """Puts a generated table into the store under a name, and describes it.

    Holds the **whole registry**, not one spec: a loader bound to a single table
    works for exactly that table and silently fails for every other one. That
    was this class's first shape, and it rejected 11 of 12 cases with a handle
    error that looked like bad data.

    The description is built from the frame that was actually stored, not from
    the spec, so a spec that disagreed with its generator would be visible.
    """

    def __init__(self, specs: list[SynthTableSpec]):
        self.registry = {spec.handle: spec for spec in specs}
        self.name = LOAD_TOOL_NAME
        self.description = ("Load a generated table into the workspace and describe "
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

        frame = build_table(spec)
        self.state["data_store"][output_dataframe_name] = frame

        from .executor import render_column_names

        nulls = int(frame.isna().sum().sum())
        return (f"Loaded generated table '{spec.handle}' as "
                f"'{output_dataframe_name}'.\n"
                f"Rows: {len(frame)}  Columns: {len(frame.columns)}\n"
                f"Column names (JSON): {render_column_names(frame.columns)}\n"
                f"Non-null cells: {int(frame.notna().sum().sum())} of "
                f"{frame.size} (missing: {nulls})")

    def bind(self, state: dict) -> SyntheticLoadTool:
        self.state = state
        return self


def fresh_synth_tools(specs: list[SynthTableSpec], state: dict) -> dict:
    """Create an isolated workspace with a generated-table loader and four local tabular tools."""
    from .slim_tools import synthetic_tabular_tools

    tools = synthetic_tabular_tools(state)
    loader = SyntheticLoadTool(list(specs)).bind(state)
    tools[loader.name] = loader
    return tools


@dataclass
class SynthCase:
    """An F1 task over a generated table."""

    case_id: str
    table: SynthTableSpec
    role: str
    question: str
    correct_column: str
    wrong_guesses: list[str] = field(default_factory=list)
    variant_family: str = "f1_synth"
    generator_module: str = SYNTH_GENERATOR_MODULE
    generator_version: str = SYNTH_GENERATOR_VERSION

    @property
    def source_lineage(self) -> str:
        return f"synthetic:{self.variant_family}:{self.case_id}"


@dataclass
class SynthJoinCase:
    """An F2 task over two generated tables with differently named keys."""

    case_id: str
    data_table: SynthTableSpec
    geo_table: SynthTableSpec
    data_frame: str
    geo_frame: str
    data_key: str
    geo_key: str
    value_column: str
    question: str
    wrong_pairs: list[tuple[str, str]] = field(default_factory=list)
    variant_family: str = "f2_synth"
    generator_module: str = SYNTH_GENERATOR_MODULE
    generator_version: str = SYNTH_GENERATOR_VERSION

    @property
    def source_lineage(self) -> str:
        return f"synthetic:{self.variant_family}:{self.case_id}"


def _synth_f1_case(seed: int, variant: str) -> SynthCase:
    """One generated F1 case, with the schema shape chosen by `variant`.

    Three schema styles, because a model that only works when the column is
    called `Country Name` has learned a string, not a role.
    """
    styles = {
        "plain": {"name": "Country Name", "code": "Country Code",
                  "value": "GDP (current US$)", "per_capita": "GDP per capita (current US$)"},
        "iso": {"name": "NAME_EN", "code": "ISO_A3",
                "value": "GDP (current US$)", "per_capita": "GDP per capita (current US$)"},
        "wdi": {"name": "Economy", "code": "Code",
                "value": "GDP (current US$)", "per_capita": "GDP per capita (current US$)"},
    }[variant]
    columns = (styles["name"], styles["code"], "Indicator Name",
               styles["value"], styles["per_capita"], "2021")
    table = SynthTableSpec(handle=f"syn_{variant}_{seed:03d}", columns=columns,
                           n_rows=40 + seed, seed=seed, kind="stats",
                           variant_family=f"synth/{variant}")

    role = "gdp_per_capita" if seed % 2 == 0 else "country_name"
    correct = styles["per_capita"] if role == "gdp_per_capita" else styles["name"]

    # A wrong guess is a name that is **not in the table**, so the trace it
    # produces is a real failure. Near-miss columns like `GDP (current US$)` are
    # deliberately in the table -- they are what the *scorer* must refuse, and
    # putting them here would emit a "failure" that succeeded.
    candidates = ("GDP per capita", "GDPPC", "Per Capita GDP", "gdp_per_capita",
                  "NAME_EN", "Country", "ADMIN", "SOVEREIGNT", "name", "Economy",
                  "ISO3", "ISO_A3", "code", "country")
    wrong = [name for name in candidates if name not in columns][:3]

    question = (f"Which column of the frame '{table.handle}' holds "
                + ("GDP per capita" if role == "gdp_per_capita" else "the country name") + "?")
    return SynthCase(case_id=f"{variant}_{seed:03d}", table=table, role=role,
                     question=question, correct_column=correct, wrong_guesses=wrong,
                     variant_family=f"f1_synth/{variant}")


def f1_synth_cases(count: int = 12) -> list[SynthCase]:
    """A spread of generated F1 cases across the three schema styles."""
    variants = ("plain", "iso", "wdi")
    return [_synth_f1_case(seed, variants[seed % len(variants)]) for seed in range(count)]


def _synth_f2_case(seed: int, variant: str) -> SynthJoinCase:
    styles = {
        "alpha": {"data_name": "Region", "data_key": "region_id",
                   "geo_name": "NAME_EN", "geo_key": "region_code"},
        "beta": {"data_name": "Area Name", "data_key": "area_code",
                 "geo_name": "ZONE_NAME", "geo_key": "zone_id"},
        "gamma": {"data_name": "District", "data_key": "district_ref",
                   "geo_name": "ADMIN_NAME", "geo_key": "district_code"},
    }[variant]
    data_table = SynthTableSpec(
        handle=f"syn_f2_data_{variant}_{seed:03d}",
        columns=(styles["data_name"], styles["data_key"], "population_2021"),
        n_rows=40 + seed,
        seed=1000 + seed,
        kind="stats",
        id_column=styles["data_key"],
        variant_family=f"f2_synth/{variant}",
    )
    geo_table = SynthTableSpec(
        handle=f"syn_f2_geo_{variant}_{seed:03d}",
        columns=(styles["geo_name"], styles["geo_key"], "zone_type"),
        n_rows=40 + seed,
        seed=2000 + seed,
        kind="regions",
        id_column=styles["geo_key"],
        variant_family=f"f2_synth/{variant}",
    )
    return SynthJoinCase(
        case_id=f"{variant}_{seed:03d}",
        data_table=data_table,
        geo_table=geo_table,
        data_frame="stats",
        geo_frame="regions",
        data_key=styles["data_key"],
        geo_key=styles["geo_key"],
        value_column="population_2021",
        question=(f"Join the statistics frame '{data_table.handle}' to the regions frame "
                  f"'{geo_table.handle}' using the matching region key columns."),
        wrong_pairs=[(styles["data_name"], styles["geo_key"]),
                     (styles["data_key"], styles["geo_name"])],
        variant_family=f"f2_synth/{variant}",
    )


def f2_synth_cases(count: int = 6) -> list[SynthJoinCase]:
    variants = ("alpha", "beta", "gamma")
    return [_synth_f2_case(seed, variants[seed % len(variants)]) for seed in range(count)]


# --- wiring the generated cases into samples ----------------------------------


def build_synth_f1_plan(case: SynthCase) -> list[tuple[str, dict]]:
    """Load the generated table, fail on names that are not there, land on the right one."""
    plan: list[tuple[str, dict]] = [
        (LOAD_TOOL_NAME, {"handle": case.table.handle, "output_dataframe_name": "frame"})]
    for guess in case.wrong_guesses:
        plan.append(("get_unique_values", {"dataframe_name": "frame", "column": guess}))
    plan.append(("get_unique_values",
                 {"dataframe_name": "frame", "column": case.correct_column}))
    return plan


def build_synth_f2_plan(case: SynthJoinCase) -> list[tuple[str, dict]]:
    plan: list[tuple[str, dict]] = [
        (LOAD_TOOL_NAME, {"handle": case.data_table.handle,
                          "output_dataframe_name": case.data_frame}),
        (LOAD_TOOL_NAME, {"handle": case.geo_table.handle,
                          "output_dataframe_name": case.geo_frame}),
    ]
    for index, (left, right) in enumerate(case.wrong_pairs):
        plan.append(("merge_dataframes", {
            "dataframe_name": case.data_frame,
            "geodataframe_name": case.geo_frame,
            "statkey": left,
            "geokey": right,
            "output_dataframe_name": f"bad_merge_{index}",
        }))
    plan.append(("merge_dataframes", {
        "dataframe_name": case.data_frame,
        "geodataframe_name": case.geo_frame,
        "statkey": case.data_key,
        "geokey": case.geo_key,
        "output_dataframe_name": "merged",
    }))
    return plan


#: The wrong calls a recovery trace may open with. Each is a mistake the model
#: was **observed** making, and each is produced by really executing it -- the
#: error in the sample is the tool's own output, never written by hand.
#:
#: The first version of this used `f"{geo_key}__missing"`, which was wrong twice
#: over: the error was a `KeyError` (a name that does not exist) while the model
#: actually fails with a `ValueError` (a name that *does* exist and holds the
#: wrong type), and the invented name announced itself -- a solver could learn
#: "if the name ends in `__missing`, pick another" without understanding keys at
#: all. Measured on that bundle: 96/96 errors carried the suffix, and every
#: wrong `statkey` was a valid column, so the trace taught the wrong lesson.
#:
#: None of these name themselves as wrong:
#:   * `wrong_role`  -- the other side's key name, which reads as a real column
#:   * `value_as_key`-- a genuine column that holds values, not keys
#:   * `missing_argument` -- a required parameter left out
ERROR_KINDS = tuple(sorted(RECOVERY_ERROR_KINDS))


def _wrong_merge(case, kind: str) -> dict:
    call = {"dataframe_name": case.data_frame, "geodataframe_name": case.geo_frame,
            "statkey": case.data_key, "geokey": case.geo_key,
            "output_dataframe_name": "failed_merge"}
    if kind == "wrong_role":
        # The geo side's key name, used on the stats side where it does not exist.
        # Plausible to a reader and to a model: it *is* a key name, just the
        # other frame's.
        call["statkey"] = case.geo_key
    elif kind == "value_as_key":
        # A real column holding numbers, so the merge fails on dtype rather than
        # on lookup -- the failure actually observed at inference.
        call["statkey"] = case.value_column
    else:
        call.pop("geokey")
    return call


def build_synth_f2_recovery_plan(case: SynthJoinCase, error_kind: str = "wrong_role",
                                 ) -> list[tuple[str, dict]]:
    """Load both frames, take **one real tool error**, then recover correctly.

    The mistake is executed, not described: whether it fails, and how, is the
    tool's answer. A written-out error would be a claim about a result nobody saw.
    """
    if error_kind not in ERROR_KINDS:
        raise ValueError(f"unknown error kind {error_kind!r}; known: {ERROR_KINDS}")
    return [
        (LOAD_TOOL_NAME, {"handle": case.data_table.handle,
                          "output_dataframe_name": case.data_frame}),
        (LOAD_TOOL_NAME, {"handle": case.geo_table.handle,
                          "output_dataframe_name": case.geo_frame}),
        ("merge_dataframes", _wrong_merge(case, error_kind)),
        ("merge_dataframes", {
            "dataframe_name": case.data_frame,
            "geodataframe_name": case.geo_frame,
            "statkey": case.data_key,
            "geokey": case.geo_key,
            "output_dataframe_name": "merged",
        }),
    ]


def synth_f1_termination(case: SynthCase) -> dict:
    return {"status": "done", "selected_column": case.correct_column, "schema_version": TERMINATION_SCHEMA_VERSION}


def check_synth_f1_gold(case: SynthCase, tools: dict, state: dict):
    """The same checks as the benchmark-catalog reference, against a generated table.

    Kept here rather than reusing `check_f1_gold` because that one is written
    against a case that names a catalog dataset, and adapting it would mean the
    synthetic path silently depends on the benchmark's shape.
    """
    from .reference import GoldCheck, confusable_columns

    tools[LOAD_TOOL_NAME].execute(handle=case.table.handle, output_dataframe_name="frame")
    frame = (state.get("data_store") or {}).get("frame")
    if frame is None:
        return GoldCheck(case.case_id, False, problems=["generated table not loaded"])

    columns = [str(c) for c in frame.columns]
    problems: list[str] = []
    if case.correct_column not in columns:
        problems.append(f"correct column {case.correct_column!r} is not in the table")
    for guess in case.wrong_guesses:
        if guess in columns:
            problems.append(f"wrong guess {guess!r} is a real column")
    if len(columns) != len(set(columns)):
        problems.append("the generated table has duplicate column names")

    return GoldCheck(case_id=case.case_id, ok=not problems, columns=columns,
                     confusable=confusable_columns(case.correct_column, columns),
                     problems=problems)


def build_synth_f1_sample(case: SynthCase, tools: dict, state: dict,
                          seed: int = 0, geobenchx_root=None) -> dict:
    from .build import _assemble

    return _assemble(
        case, build_synth_f1_plan(case), synth_f1_termination(case),
        lambda store: {
            "role": case.role, "column": case.correct_column,
            "frame_columns": list(store.get("frame").columns
                                   if store.get("frame") is not None else []),
        },
        # The case's own family, not one label for every synthetic case: these
        # ARE different generators, and a single shared label is what made the
        # split report flag cross-split leakage on data that had none.
        family="F1", generator_family=case.variant_family, frames=["frame"],
        tools=tools, state=state, seed=seed, geobenchx_root=geobenchx_root,
        source_kind="synthetic_generator", generator_module=case.generator_module,
        generator_version=case.generator_version)


def synth_f2_termination(case: SynthJoinCase) -> dict:
    return {"status": "done", "join_keys": [case.data_key, case.geo_key],
            "schema_version": TERMINATION_SCHEMA_VERSION}


def check_synth_f2_gold(case: SynthJoinCase, tools: dict, state: dict):
    from .reference import MIN_CORRECT_FILL_RATE, MIN_FILL_MARGIN, JoinCheck, fill_rate

    for table, output in ((case.data_table, case.data_frame),
                          (case.geo_table, case.geo_frame)):
        tools[LOAD_TOOL_NAME].execute(handle=table.handle,
                                      output_dataframe_name=output)

    def merged_fill(pair: tuple[str, str], name: str) -> float:
        tools["merge_dataframes"].execute(
            dataframe_name=case.data_frame,
            geodataframe_name=case.geo_frame,
            statkey=pair[0],
            geokey=pair[1],
            output_dataframe_name=name,
        )
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


def build_synth_f2_sample(case: SynthJoinCase, tools: dict, state: dict,
                          seed: int = 0, geobenchx_root=None) -> dict:
    from .build import _assemble

    return _assemble(
        case, build_synth_f2_plan(case), synth_f2_termination(case),
        lambda _store: {"join_keys": [case.data_key, case.geo_key],
                        "value_column": case.value_column},
        family="F2", generator_family=case.variant_family,
        frames=[case.data_frame, case.geo_frame], tools=tools, state=state,
        seed=seed, geobenchx_root=geobenchx_root,
        source_kind="synthetic_generator", generator_module=case.generator_module,
        generator_version=case.generator_version)


def build_synth_f2_recovery_sample(case: SynthJoinCase, tools: dict, state: dict,
                                   seed: int = 0, geobenchx_root=None,
                                   error_kind: str = "wrong_role") -> dict:
    from .build import _assemble

    return _assemble(
        case, build_synth_f2_recovery_plan(case, error_kind),
        synth_f2_termination(case),
        lambda _store: {"join_keys": [case.data_key, case.geo_key],
                        "value_column": case.value_column},
        family="F2", generator_family=case.variant_family,
        frames=[case.data_frame, case.geo_frame], tools=tools, state=state,
        seed=seed, geobenchx_root=geobenchx_root,
        source_kind="synthetic_generator", generator_module=case.generator_module,
        generator_version=case.generator_version,
        # Recorded as provenance so the kind can be audited and counted. It is
        # deliberately NOT placed in system/user/messages: a prompt that names
        # the error the trace is about to make has given away the answer.
        lineage_extra={"recovery_error_kind": error_kind})


def build_synth_f1_batch(cases: list[SynthCase], geobenchx_root=None):
    """Build every generated case, checking gold and validation on each.

    The tools factory rebuilds the loader registry per case so the workspace
    holds every handle the batch might use, while the frames themselves stay
    per-case -- reusing one state would let one case's table be visible to
    another.
    """
    from ..geobench.tools import benchmark_state
    from .sample import validate_sample

    samples: list[dict] = []
    report: list[dict] = []

    for seed, case in enumerate(cases):
        state = benchmark_state()
        tools = fresh_synth_tools([case.table], state)
        gold = check_synth_f1_gold(case, tools, state)
        sample = build_synth_f1_sample(case, tools, state, seed=seed,
                                       geobenchx_root=geobenchx_root)
        verdict = validate_sample(sample)
        row = {"sample_id": sample["sample_id"], "gold_ok": gold.ok,
               "gold_problems": gold.problems, "validate_ok": verdict.ok,
               "blockers": verdict.blockers,
               "failed_steps": sum(1 for s in sample["steps"] if s["failed"])}
        row.update(gold.as_dict())
        report.append(row)
        samples.append(sample)

    return samples, report


def build_synth_f2_batch(cases: list[SynthJoinCase], geobenchx_root=None):
    from ..geobench.tools import benchmark_state
    from .sample import validate_sample

    samples: list[dict] = []
    report: list[dict] = []
    for seed, case in enumerate(cases):
        state = benchmark_state()
        tools = fresh_synth_tools([case.data_table, case.geo_table], state)
        gold = check_synth_f2_gold(case, tools, state)
        sample = build_synth_f2_sample(case, tools, state, seed=seed,
                                       geobenchx_root=geobenchx_root)
        verdict = validate_sample(sample)
        row = {"sample_id": sample["sample_id"], "gold_ok": gold.ok,
               "gold_problems": gold.problems, "validate_ok": verdict.ok,
               "blockers": verdict.blockers,
               "failed_steps": sum(1 for step in sample["steps"] if step["failed"])}
        row.update(gold.as_dict())
        report.append(row)
        samples.append(sample)
    return samples, report


def build_synth_f2_recovery_batch(cases: list[SynthJoinCase], geobenchx_root=None):
    from ..geobench.tools import benchmark_state
    from .sample import validate_sample

    samples: list[dict] = []
    report: list[dict] = []
    for seed, case in enumerate(cases):
        # Even thirds by a stable rule. Not "the closest kind to the current
        # failure": mixing prevalence with error type would put two variables in
        # one column, and the first round could not say which one moved.
        kind = ERROR_KINDS[seed % len(ERROR_KINDS)]
        state = benchmark_state()
        tools = fresh_synth_tools([case.data_table, case.geo_table], state)
        gold = check_synth_f2_gold(case, tools, state)
        sample = build_synth_f2_recovery_sample(case, tools, state, seed=seed,
                                                geobenchx_root=geobenchx_root,
                                                error_kind=kind)
        verdict = validate_sample(sample)

        steps = sample["steps"]
        failed = [i for i, step in enumerate(steps) if step["failed"]]
        # Fail closed. A recovery trace whose "wrong" call succeeded is not a
        # harder case, it is a trace with no error in it -- and it would be
        # trained on as if it taught recovery. `wrong_role` is the kind this can
        # happen to: the other frame's key name is a legal column whenever the
        # two shapes share one.
        if len(failed) != 1:
            raise ValueError(
                f"{sample['sample_id']}: expected exactly one failed call for "
                f"{kind!r}, found {len(failed)}; refusing rather than shipping a "
                "trace with no error in it")
        recovered = len(steps) > failed[0] + 1 and not steps[failed[0] + 1]["failed"]

        row = {"sample_id": sample["sample_id"], "gold_ok": gold.ok,
               "gold_problems": gold.problems, "validate_ok": verdict.ok,
               "blockers": verdict.blockers,
               "recovery_error_kind": kind,
               "recovered": recovered,
               "failed_steps": len(failed)}
        row.update(gold.as_dict())
        report.append(row)
        samples.append(sample)
    return samples, report
