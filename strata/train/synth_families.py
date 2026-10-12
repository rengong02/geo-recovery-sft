"""Generator families beyond the first three schema styles.

Codex's ruling: growing the set by re-seeding the same three templates is not
growth. A model that has seen `Country Name` / `NAME_EN` / `Economy` as three
spellings of one idea has seen one composition; a fourth spelling adds a row,
not a case. What generalises is the *shape* of the problem, so each family here
differs in what makes the answer hard:

| family | what is hard about it |
|---|---|
| `year_role` | the answer is a year column among many near-identical ones |
| `unit_role` | several real columns differ only by a unit suffix |
| `naming_convention` | snake_case / CamelCase / spaced spellings of the same role |
| `dense_near_miss` | **four** columns share tokens with the gold |
| `unambiguous` | no near-miss at all -- the control that says a failure is real |

The control matters as much as the hard ones. If every case is a trap, a model
that fails tells you nothing about whether it can do the easy thing.

The `unambiguous` family is also the one that keeps the *scorer* honest: if it
ever refuses a case with only one plausible answer, the scorer is wrong, not the
model.
"""

from __future__ import annotations

from .synth import SynthCase, SynthJoinCase, SynthTableSpec

FAMILY_VERSION = "synth/v2"


def _spec(handle: str, columns: tuple[str, ...], seed: int, family: str) -> SynthTableSpec:
    return SynthTableSpec(handle=handle, columns=columns, n_rows=30 + (seed % 17),
                          seed=seed, kind="stats", variant_family=family)


#: Each composition maps a seed to `(columns, role, correct, wrong_candidates)`.
#: Data rather than six near-identical functions: the differences between the
#: families are the *contents of this table*, and writing them as code hid that.
def _country_role(seed: int):
    columns = ("Country Name", "Country Code", "Indicator Name",
               "GDP (current US$)", "GDP per capita (current US$)", "2021")
    return columns, "gdp_per_capita", "GDP per capita (current US$)", (
        "GDP per capita", "GDPPC", "Per Capita GDP", "GDP")


def _year_role(seed: int):
    year = 1990 + (seed % 15)
    columns = ("Country Name", "Country Code", f"{year}", f"{year - 1}", f"{year + 1}",
               f"{year - 2}", "Indicator Name")
    return columns, f"forest_area_{year}", f"{year}", (
        str(year - 1), str(year + 1), f"{year}a", str(year))


def _unit_role(seed: int):
    columns = ("Country Name", "Forest area (% of land area)",
               "Forest area (sq. km)", "Forest area (1000 ha)",
               "Land area (sq. km)", "2021")
    return columns, "forest_area_km2", "Forest area (sq. km)", (
        "Forest area", "forest_km2", "Forest", "Area (sq. km)")


def _naming_convention(seed: int):
    conventions = (
        ("economy_name", "economy_code", "indicator_name",
         "gdp_per_capita_usd", "gdp_usd", "year_2021"),
        ("economyName", "economyCode", "indicatorName",
         "gdpPerCapitaUsd", "gdpUsd", "year2021"),
        ("Economy Name", "Economy Code", "Indicator Name",
         "GDP per capita USD", "GDP USD", "Year 2021"),
    )
    columns = conventions[seed % len(conventions)]
    return columns, "gdp_per_capita", columns[3], (
        "gdp_per_capita", "GDP per capita", "per_capita", "gdp_usd")


def _dense_near_miss(seed: int):
    columns = ("Country Name", "Country Code", "Country ISO",
               "GDP per capita (current US$)", "GDP per capita (constant 2015 US$)",
               "GDP (current US$)", "GNI per capita (current US$)")
    return columns, "gdp_per_capita_current", "GDP per capita (current US$)", (
        "GDP per capita", "GDP per capita current", "GNI per capita", "GDP per capita US$")


def _unambiguous(seed: int):
    """No near-miss. The only column that can answer is the right one."""
    columns = ("Country Name", "Country Code", "Population, total",
               "Surface area (sq. km)", "2021")
    return columns, "population_total", "Population, total", (
        "population", "Population", "total_population", "pop")


def _qualifier_constant(seed: int):
    """Two columns agree on the head noun and differ only in the qualifier.

    `GDP per capita (current US$)` and `GDP per capita (constant 2015 US$)` are
    both real, both about GDP per capita, and only one answers the question
    asked. A model that matches on the head noun gets this wrong half the time
    and cannot tell which half.
    """
    columns = ("Country Name", "Country Code", "GDP per capita (current US$)",
               "GDP per capita (constant 2015 US$)", "GDP (current US$)", "2021")
    return columns, "gdp_per_capita_constant", "GDP per capita (constant 2015 US$)", (
        "GDP per capita (current US$)", "GDP per capita", "GDP (constant 2015 US$)",
        "GDP per capita constant")


def _qualifier_unit(seed: int):
    """Same idea with units: the answer is decided by which unit is asked for."""
    columns = ("Country Name", "Forest area (% of land area)",
               "Forest area (sq. km)", "Forest area (1000 ha)",
               "Land area (sq. km)", "2021")
    return columns, "forest_area_share", "Forest area (% of land area)", (
        "Forest area (sq. km)", "Forest area (1000 ha)", "Forest share", "forest_pct")


F1_COMPOSITIONS = {
    "qualifier_constant": _qualifier_constant,
    "qualifier_unit": _qualifier_unit,
    "country_role": _country_role,
    "year_role": _year_role,
    "unit_role": _unit_role,
    "naming_convention": _naming_convention,
    "dense_near_miss": _dense_near_miss,
    "unambiguous": _unambiguous,
}

_ROLE_QUESTION = {
    "country_role": "GDP per capita",
    "year_role": "the value for the target year",
    "unit_role": "forest area in square kilometres",
    "naming_convention": "GDP per capita",
    "dense_near_miss": "GDP per capita in current US dollars",
    "unambiguous": "the total population",
    "qualifier_constant": "GDP per capita measured in constant 2015 US dollars",
    "qualifier_unit": "forest area as a share of land area",
}


def f1_family_cases(family: str, count: int = 24) -> list[SynthCase]:
    """`count` cases from one composition, each a different seed."""
    compose = F1_COMPOSITIONS[family]
    cases: list[SynthCase] = []
    for index in range(count):
        seed = index * 7 + len(family)
        columns, role, correct, wrong_pool = compose(seed)
        table = _spec(f"syn_{family}_{index:03d}", columns, seed, f"synth/{family}")
        # Only names the table does not actually have: a "wrong" guess that
        # resolved would emit a failure that succeeded.
        wrong = [name for name in wrong_pool if name not in columns][:3]
        role_question = _ROLE_QUESTION[family]
        if family == "year_role":
            role_question = f"the value for year {correct}"
        cases.append(SynthCase(
            case_id=f"{family}_{index:03d}", table=table, role=role,
            question=(f"Which column of the frame '{table.handle}' holds "
                      f"{role_question}?"),
            correct_column=correct, wrong_guesses=wrong,
            variant_family=f"f1_synth/{family}",
            generator_module="strata.train.synth_families",
            generator_version=FAMILY_VERSION))
    return cases


def all_f1_family_cases(count_per_family: int = 24) -> list[SynthCase]:
    return [case for family in F1_COMPOSITIONS
            for case in f1_family_cases(family, count_per_family)]


# --- F2 compositions ----------------------------------------------------------

#: What varies here is the **naming and the convention**, not the value space.
#:
#: The first draft of these families had one side keyed by an id and the other
#: by a name. That join does not exist: the two columns hold values from
#: different spaces, so no row matches and the "correct" pair scores zero. A
#: generated case whose gold pair cannot match is not a hard case, it is a
#: broken one -- and the fill-rate reference is what would have caught it.
#:
#: So both sides always key on the *same kind of value* (an entity code, or an
#: entity name); what changes is what those columns are called, and whether the
#: two frames spell the convention the same way.

#: The trailing flag says whether that key holds entity **codes** (True) or
#: entity **names** (False). `build_table` fills a column from a name heuristic
#: unless told otherwise, so a key named `region_id` would otherwise be filled
#: with random values and match nothing.
def _code_to_code(seed: int):
    return (("Region", "region_id", "population_2021", True),
            ("NAME_EN", "region_code", "geometry", True))


def _name_to_name(seed: int):
    return (("Area", "area_name", "population_2021", False),
            ("ADMIN_NAME", "admin_name", "geometry", False))


def _convention_mismatch(seed: int):
    return (("District", "DISTRICT_ID", "population_2021", True),
            ("district_name", "district_code", "geometry", True))


F2_SHAPES = {
    "shared_root": ("Region", "region_id", "NAME_EN", "region_code"),
    "case_only": ("Region", "REGION_ID", "NAME_EN", "region_code"),
    "disjoint_root": ("Region", "zone_id", "NAME_EN", "area_code"),
}

#: Multipliers on the geo frame's row count, **not** on how many of its rows
#: match. `merge_dataframes` is a left join that preserves every geo row, so a
#: geo frame *smaller* than the data frame still matches completely -- measured:
#: 2/3 the rows gave a 0.9565 fill, not the intended two thirds. What makes a
#: correct join incomplete is geo rows the data does not contain.
F2_COVERAGE = {
    "full": 1.0,     # the geo frame covers exactly the entities the data has
    "partial": 1.5,  # it carries extra entities, so a third of the joins miss
}


def f2_family_cases(family: str, count: int = 24) -> list[SynthJoinCase]:
    """Build one cell of a paired lexical-shape/coverage diagnostic panel.

    ``family`` is ``<shape>__<coverage>`` in audit metadata, not model-visible
    handles. Matching indices share the seed, data rows and neutral handles.
    Different cells therefore represent interventions on the same world, not
    independent examples to scatter across a training split.
    """
    shape, _, coverage = family.partition("__")
    if shape not in F2_SHAPES:
        raise ValueError(f"unknown F2 shape {shape!r}; known: {sorted(F2_SHAPES)}")
    if coverage not in F2_COVERAGE:
        raise ValueError(f"unknown F2 coverage {coverage!r}; known: {sorted(F2_COVERAGE)}")

    data_name, data_key, geo_name, geo_key = F2_SHAPES[shape]
    value_column = "population_2021"

    cases: list[SynthJoinCase] = []
    for index in range(count):
        seed = 7000 + index * 5
        rows = 30 + (seed % 13)
        geo_rows = max(2, round(rows * F2_COVERAGE[coverage]))
        data_table = SynthTableSpec(
            handle=f"syn_f2_data_{index:03d}",
            columns=(data_name, data_key, value_column), n_rows=rows,
            seed=seed, kind="stats", id_column=data_key,
            variant_family=f"synth/f2/{family}")
        geo_table = SynthTableSpec(
            handle=f"syn_f2_geo_{index:03d}",
            columns=(geo_name, geo_key), n_rows=geo_rows,
            seed=seed, kind="regions", id_column=geo_key,
            variant_family=f"synth/f2/{family}")

        # Only pairs that genuinely fail. In a name-keyed shape the cross pairing
        # can be a valid join, so listing it would be a false claim about the
        # data -- and the fill-rate reference refuses a case whose "wrong" pair
        # outperforms the right one.
        wrong = [(data_name, geo_key), (value_column, geo_key)]
        cases.append(SynthJoinCase(
            case_id=f"{family}_{index:03d}", data_table=data_table, geo_table=geo_table,
            data_frame="stats_frame", geo_frame="geo_frame",
            data_key=data_key, geo_key=geo_key, value_column=value_column,
            question=(f"Join '{data_table.handle}' to '{geo_table.handle}'. "
                      "The two frames do not use the same column name for the key."),
            wrong_pairs=wrong, variant_family=f"f2_synth/{family}",
            generator_module="strata.train.synth_families",
            generator_version="synth/f2-factorial-v1"))
    return cases


def all_f2_family_cases(count_per_family: int = 24) -> list[SynthJoinCase]:
    return [case for shape in F2_SHAPES for coverage in F2_COVERAGE
            for case in f2_family_cases(f"{shape}__{coverage}", count_per_family)]
