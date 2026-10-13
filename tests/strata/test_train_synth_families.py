"""The expanded generator families, and the two ways the first draft was wrong.

Both were caught by the reference check rather than by inspection, which is the
point of having one:

* **a key column filled with the wrong kind of value.** `build_table` chooses a
  column's contents from its *name*, so a key called `region_id` was filled with
  random numbers and matched nothing. The fix is to say explicitly which column
  holds the entity codes -- `id_column` -- instead of relying on the spelling.
* **a "wrong" key pair that actually matched.** In a name-keyed family both key
  columns hold entity names from the same seed, so pairing one frame's name
  column against the other's is a *valid join*. Declaring it wrong made every
  case in that family unsatisfiable, and the reference refused all twelve.
"""

from __future__ import annotations

from strata.train.reference import MIN_CORRECT_FILL_RATE, MIN_FILL_MARGIN
from strata.train.schema_groups import content_schema_fingerprint
from strata.train.synth import build_synth_f1_batch, build_synth_f2_batch
from strata.train.synth_families import (
    F1_COMPOSITIONS,
    F2_COVERAGE,
    F2_SHAPES,
    all_f1_family_cases,
    all_f2_family_cases,
    f1_family_cases,
    f2_family_cases,
)

F1_CASES = all_f1_family_cases(3)
F2_CASES = all_f2_family_cases(3)


# --- the shapes ---------------------------------------------------------------


def test_there_is_more_than_one_composition_per_family():
    """Generator families must vary structure as well as random seed."""
    assert len(F1_COMPOSITIONS) >= 5
    assert len(F2_SHAPES) >= 3 and len(F2_COVERAGE) >= 2


def test_the_control_family_has_no_near_miss():
    """If every case is a trap, a model that fails says nothing about whether it
    can do the easy thing."""
    columns, _role, correct, _wrong = F1_COMPOSITIONS["unambiguous"](0)
    from strata.train.reference import confusable_columns

    assert confusable_columns(correct, list(columns)) == []


def test_the_hard_family_has_several_near_misses():
    columns, _role, correct, _wrong = F1_COMPOSITIONS["dense_near_miss"](0)
    from strata.train.reference import confusable_columns

    assert len(confusable_columns(correct, list(columns))) >= 2


def test_every_family_produces_cases_whose_gold_column_exists():
    for family in F1_COMPOSITIONS:
        for case in f1_family_cases(family, 3):
            assert case.correct_column in case.table.columns, (family, case.case_id)
            assert not [g for g in case.wrong_guesses if g in case.table.columns], case.case_id


# --- the join families --------------------------------------------------------


def test_a_key_column_holds_codes_when_the_family_says_so():
    """`id_column` is the fix for the name-heuristic filling a key with values.

    Without it the "correct" pair matched 0% of rows and every case failed --
    the reference caught it on all thirty-six.
    """
    case = f2_family_cases("shared_root__full", 1)[0]
    assert case.data_table.id_column == case.data_key
    assert case.geo_table.id_column == case.geo_key


def test_the_name_family_does_not_declare_a_matching_pair_as_wrong():
    """The other failure, and the subtler one.

    In a name-keyed family the cross pairing is a real join. Listing it as a
    wrong answer is a false statement about the data, and the reference refuses
    a case whose wrong pair outperforms the right one.
    """
    case = f2_family_cases("case_only__full", 1)[0]
    other_frame_names = {case.geo_table.columns[0], case.geo_table.columns[1]}
    for left, right in case.wrong_pairs:
        assert not (left == case.data_key and right in other_frame_names
                    and right != case.geo_key), (left, right)


def test_every_join_family_separates_right_from_wrong():
    """The gold itself: the correct pair fills materially more than any wrong one."""
    _, report = build_synth_f2_batch(F2_CASES)
    for row in report:
        assert row["gold_ok"], (row["sample_id"], row["gold_problems"])
        assert row["correct_fill_rate"] >= MIN_CORRECT_FILL_RATE
        assert row["correct_fill_rate"] - max(row["wrong_fill_rates"] or [0.0]) >= MIN_FILL_MARGIN


def test_the_coverage_axis_makes_the_right_join_incomplete():
    """`partial` adds entities outside the data frame, so the correct pair is
    materially incomplete -- which is the property that distinguishes the axis
    from `full`, and the reason a checker expecting a full match would be wrong.
    """
    case = f2_family_cases("case_only__partial", 1)[0]
    _, report = build_synth_f2_batch([case])

    # The geo frame is *larger*: `merge_dataframes` is a left join that keeps
    # every geo row, so incompleteness can only come from geo entities the data
    # does not have. A smaller geo frame still matched completely (0.9565).
    assert case.geo_table.n_rows > case.data_table.n_rows
    assert MIN_CORRECT_FILL_RATE <= report[0]["correct_fill_rate"] < 0.95


def test_the_two_axes_are_independent():
    """The same shape at both coverages; the same coverage at two shapes.

    v2 bundled them, so `convention_mismatch` and `partial_overlap` were byte-
    identical and the 24/24-versus-0/24 gap could not be assigned to either.
    """
    full = f2_family_cases("case_only__full", 1)[0]
    partial = f2_family_cases("case_only__partial", 1)[0]
    other = f2_family_cases("disjoint_root__full", 1)[0]

    assert full.data_table.columns == partial.data_table.columns
    assert full.data_table.columns != other.data_table.columns


def test_shape_axis_keeps_non_key_columns_fixed():
    cases = [f2_family_cases(f"{shape}__full", 1)[0] for shape in F2_SHAPES]

    assert {case.data_table.columns[0] for case in cases} == {"Region"}
    assert {case.geo_table.columns[0] for case in cases} == {"NAME_EN"}
    assert {case.data_table.columns[2] for case in cases} == {"population_2021"}
    assert {case.geo_table.columns[1] for case in cases} == {
        "region_code", "area_code"
    }


def test_shape_axis_changes_one_lexical_factor_at_a_time():
    shared = f2_family_cases("shared_root__full", 1)[0]
    case_mixed = f2_family_cases("case_only__full", 1)[0]
    disjoint = f2_family_cases("disjoint_root__full", 1)[0]

    assert shared.data_key.lower() == case_mixed.data_key.lower()
    assert shared.geo_key.lower() == case_mixed.geo_key.lower()
    assert shared.data_key != case_mixed.data_key
    assert shared.geo_key == case_mixed.geo_key
    assert shared.data_key[-3:].lower() == case_mixed.data_key[-3:].lower()
    assert shared.geo_key[-5:].lower() == case_mixed.geo_key[-5:].lower()
    assert shared.data_key.lower() != disjoint.data_key.lower()
    assert shared.geo_key.lower() != disjoint.geo_key.lower()


def test_content_schema_groups_coverage_but_separates_shapes():
    def fingerprint(case):
        return content_schema_fingerprint({
            "source_provenance": {
                "source_kind": "synthetic_generator",
                "generator": {
                    "specs": [
                        case.data_table.as_dict(),
                        case.geo_table.as_dict(),
                    ]
                },
            }
        })

    shared_full = fingerprint(f2_family_cases("shared_root__full", 1)[0])
    shared_partial = fingerprint(f2_family_cases("shared_root__partial", 1)[0])
    case_only = fingerprint(f2_family_cases("case_only__full", 1)[0])
    disjoint = fingerprint(f2_family_cases("disjoint_root__full", 1)[0])

    assert shared_full == shared_partial
    assert len({shared_full, case_only, disjoint}) == 3


def test_factorial_cells_share_seed_rows_and_neutral_handles():
    cases = all_f2_family_cases(3)

    for index in range(3):
        block = [case for case in cases if case.case_id.endswith(f"_{index:03d}")]
        assert len(block) == 6
        assert len({case.data_table.seed for case in block}) == 1
        assert len({case.geo_table.seed for case in block}) == 1
        assert len({case.data_table.n_rows for case in block}) == 1
        assert len({case.data_table.handle for case in block}) == 1
        assert len({case.geo_table.handle for case in block}) == 1
        assert len({case.question for case in block}) == 1
        for case in block:
            for label in (*F2_SHAPES, *F2_COVERAGE):
                assert label not in case.question


def test_factorial_coverage_keeps_data_values_and_geo_prefix_fixed():
    from strata.train.synth import build_table

    for shape in F2_SHAPES:
        full = f2_family_cases(f"{shape}__full", 3)
        partial = f2_family_cases(f"{shape}__partial", 3)
        for full_case, partial_case in zip(full, partial):
            full_data = build_table(full_case.data_table)
            partial_data = build_table(partial_case.data_table)
            assert full_data.equals(partial_data)
            full_geo = build_table(full_case.geo_table)
            partial_geo = build_table(partial_case.geo_table)
            assert full_geo.equals(partial_geo.iloc[:len(full_geo)])


def test_factorial_cases_record_their_actual_generator_version():
    for case in all_f2_family_cases(1):
        assert case.generator_module == "strata.train.synth_families"
        assert case.generator_version == "synth/f2-factorial-v1"


# --- end to end ---------------------------------------------------------------


def test_the_expanded_set_builds_and_validates():
    f1_samples, f1_report = build_synth_f1_batch(F1_CASES)
    f2_samples, f2_report = build_synth_f2_batch(F2_CASES)

    assert len(f1_samples) == len(F1_CASES) and len(f2_samples) == len(F2_CASES)
    for row in f1_report + f2_report:
        assert row["gold_ok"], (row["sample_id"], row["gold_problems"])
        assert row["validate_ok"], (row["sample_id"], row["blockers"])


def test_each_sample_names_its_own_generator_family():
    """One label for every synthetic sample is what made the split report flag
    cross-split leakage on data that did not have any."""
    samples, _ = build_synth_f1_batch(F1_CASES)
    families = {s["lineage"]["generator_family"] for s in samples}
    assert len(families) == len({c.variant_family for c in F1_CASES}) > 1
