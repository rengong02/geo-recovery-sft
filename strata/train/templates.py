"""Task templates, and the plans they generate.

A template declares a *semantic role* -- "the column holding GDP per capita" --
and the column names a solver might plausibly try first. The plan it emits
deliberately includes a wrong guess before the right one, because the failure
trace is the point: p2c showed the model guessing `NAME`, `ADMIN`, `SOVEREIGNT`,
`Country` on a frame whose real column is `NAME_EN`, six times, without
recovering. A dataset of only-successful traces cannot support any claim about
error recovery, so the recovery has to be in the data.

The wrong guesses are not invented for the template. They are the names that
were actually observed, which is what makes them worth training against.

Plans name tools and arguments only. What the calls *return* is never written
here -- that comes from execution, and only from execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .sample import TERMINATION_SCHEMA_VERSION


@dataclass(frozen=True)
class F1Case:
    """One field-grounding task: find the column with a given meaning."""

    case_id: str
    loader: str          # "load_data" or "load_geodata"
    dataset: str
    frame: str
    role: str            # semantic role, e.g. "country_name"
    question: str
    correct_column: str
    wrong_guesses: list[str] = field(default_factory=list)
    variant_family: str = "f1_column_role"

    @property
    def source_lineage(self) -> str:
        return f"synthetic:{self.variant_family}:{self.case_id}"


def _loader_args(case: F1Case) -> dict:
    if case.loader == "load_geodata":
        return {"geodataset": case.dataset, "output_geodataframe_name": case.frame}
    return {"dataset": case.dataset, "output_dataframe_name": case.frame}


#: The first batch of F1 cases.
#:
#: **Provenance of the wrong guesses differs by case, and the report should say
#: so.** `countries_name` uses the six names p2c's solver actually tried
#: (`NAME`, `ADMIN`, `name`, `SOVEREIGNT`, `Country`, `country`); those four are
#: the observed ones. The World Bank cases use names that are *plausible* for
#: those frames but were **not** observed failing -- they widen the family
#: without pretending to be evidence of a real failure.
#:
#: Either way the guess is executed for real before it enters a sample, so an
#: unobserved-but-plausible guess that happens to be a real column is caught by
#: the reference check rather than shipped.
F1_CASES: list[F1Case] = [
    F1Case(
        case_id="countries_name",
        loader="load_geodata",
        dataset="Countries",
        frame="countries",
        role="country_name",
        question="Which column of the countries frame holds the English country name?",
        correct_column="NAME_EN",
        wrong_guesses=["NAME", "ADMIN", "SOVEREIGNT", "Country"],
        variant_family="f1_column_role/geodata",
    ),
    F1Case(
        case_id="wdi_country_name",
        loader="load_data",
        dataset="GDP per capita (current US$)",
        frame="gdp_pc",
        role="country_name",
        question="Which column of the GDP per capita table holds the country name?",
        correct_column="Country Name",
        wrong_guesses=["Country", "NAME", "name", "Economy"],
        variant_family="f1_column_role/wdi",
    ),
    F1Case(
        case_id="wdi_country_code",
        loader="load_data",
        dataset="GDP per capita (current US$)",
        frame="gdp_pc",
        role="country_code",
        question="Which column of the GDP per capita table holds the ISO3 country code?",
        correct_column="Country Code",
        wrong_guesses=["ISO_A3", "code", "Code", "ISO3"],
        variant_family="f1_column_role/wdi",
    ),
    F1Case(
        case_id="forest_country_name",
        loader="load_data",
        dataset="Forest area (% of land area)",
        frame="forest",
        role="country_name",
        question="Which column of the forest table holds the country name?",
        correct_column="Country Name",
        wrong_guesses=["Country", "NAME_EN", "country"],
        variant_family="f1_column_role/wdi",
    ),
]


def build_f1_plan(case: F1Case) -> list[tuple[str, dict]]:
    """Load, fail once on a plausible wrong name, then land on the right one.

    The wrong guess is included only when the template declares one -- a case
    with no observed failure teaches no recovery, and pretending otherwise
    would put a fabricated difficulty in the data.
    """
    plan: list[tuple[str, dict]] = [(case.loader, _loader_args(case))]
    for guess in case.wrong_guesses:
        plan.append(("get_unique_values",
                     {"dataframe_name": case.frame, "column": guess}))
    plan.append(("get_unique_values",
                 {"dataframe_name": case.frame, "column": case.correct_column}))
    return plan


def f1_termination(case: F1Case) -> dict:
    return {"status": "done", "selected_column": case.correct_column,
            "schema_version": TERMINATION_SCHEMA_VERSION}


# --- F2: join-key alignment ---------------------------------------------------


@dataclass(frozen=True)
class F2Case:
    """One join task: two frames whose key columns do not share a name."""

    case_id: str
    data_dataset: str
    data_frame: str
    data_key: str          # the statistical frame's key column
    geo_dataset: str
    geo_frame: str
    geo_key: str           # the geodata frame's key column
    value_column: str      # the column whose fill rate reveals whether it matched
    question: str
    wrong_pairs: list[tuple[str, str]] = field(default_factory=list)
    variant_family: str = "f2_join_key"

    @property
    def source_lineage(self) -> str:
        return f"synthetic:{self.variant_family}:{self.case_id}"


#: Why the gold is a *match rate* and not a row count, measured on the real data:
#:
#:     Country Code <-> ISO_A3   -> 251 rows, 186 filled (74.1%)
#:     Country Name <-> ISO_A3   -> 251 rows,   0 filled ( 0.0%)
#:
#: `merge_dataframes` is a **left** join, so both runs produce the identical
#: 251x96 frame. A check on shape or row count cannot tell them apart at all --
#: the correct key pair must also preserve meaningful value matches.
F2_CASES: list[F2Case] = [
    F2Case(
        case_id="gdp_country_code",
        data_dataset="GDP per capita (current US$)",
        data_frame="gdp_pc",
        data_key="Country Code",
        geo_dataset="Countries",
        geo_frame="countries",
        geo_key="ISO_A3",
        value_column="2023",
        question=("Join the GDP per capita table to the countries frame. "
                  "The two frames do not use the same column name for the country key."),
        wrong_pairs=[("Country Name", "ISO_A3"), ("Country Code", "NAME_EN")],
    ),
    F2Case(
        case_id="forest_country_code",
        data_dataset="Forest area (% of land area)",
        data_frame="forest",
        data_key="Country Code",
        geo_dataset="Countries",
        geo_frame="countries",
        geo_key="ISO_A3",
        value_column="2021",
        question=("Join the forest-area table to the countries frame. "
                  "The two frames do not use the same column name for the country key."),
        wrong_pairs=[("Country Name", "ISO_A3")],
    ),
]


def build_f2_plan(case: F2Case) -> list[tuple[str, dict]]:
    """Load both sides, try a plausible wrong pair, then land on the right one.

    The wrong pair runs for real. A join on mismatched keys does not raise -- it
    returns a frame that *looks* fine and is empty where the data should be, so
    an unexecuted "wrong attempt" would be a claim about a result nobody saw.
    """
    plan: list[tuple[str, dict]] = [
        ("load_data", {"dataset": case.data_dataset, "output_dataframe_name": case.data_frame}),
        ("load_geodata", {"geodataset": case.geo_dataset,
                          "output_geodataframe_name": case.geo_frame}),
    ]
    for index, (left, right) in enumerate(case.wrong_pairs):
        plan.append(("merge_dataframes", {
            "dataframe_name": case.data_frame, "geodataframe_name": case.geo_frame,
            "statkey": left, "geokey": right,
            "output_dataframe_name": f"bad_merge_{index}"}))
    plan.append(("merge_dataframes", {
        "dataframe_name": case.data_frame, "geodataframe_name": case.geo_frame,
        "statkey": case.data_key, "geokey": case.geo_key,
        "output_dataframe_name": "merged"}))
    return plan


def f2_termination(case: F2Case) -> dict:
    return {"status": "done", "join_keys": [case.data_key, case.geo_key],
            "schema_version": TERMINATION_SCHEMA_VERSION}
