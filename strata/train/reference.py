"""The reference answer, computed from the data rather than asserted by the template.

Separate from the scorer on purpose. A single function that both produces the
label and grades against it grades itself: any mistake in the role mapping
becomes invisible, because the same mistake is on both sides.

So the split is:

* the template **declares** the role -- "country_name means this column";
* this module **checks the declaration against the loaded frame** -- the column
  really exists, the wrong guesses really do not, and the near-miss columns that
  a scorer must refuse are derived from the real column list rather than listed
  by hand;
* the scorer, later, grades a model's attempt against what this produced.

The check that matters most is the second one. A template whose "wrong" guess
turns out to be a real column would emit a failure trace that is not a failure
-- and the resulting sample would teach that a correct call is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .templates import F1Case, _loader_args

#: Tokens too common to indicate a role on their own. Without this, "Country
#: Code" and "Country Name" look like near-misses of each other because they
#: share "country" -- which is true and also not useful, since every column in
#: a country table shares it.
STOPWORDS = frozenset({"the", "of", "and", "per", "in", "a", "to"})


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-zA-Z]+", str(name).lower())
            if t and t not in STOPWORDS}


@dataclass
class GoldCheck:
    case_id: str
    ok: bool
    columns: list[str] = field(default_factory=list)
    confusable: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"case_id": self.case_id, "ok": self.ok, "problems": self.problems,
                "confusable": self.confusable, "n_columns": len(self.columns)}


def _columns_of(frame) -> list[str]:
    return [str(c) for c in getattr(frame, "columns", [])]


def confusable_columns(correct: str, columns: list[str]) -> list[str]:
    """Real columns that share a meaningful token with the right answer.

    These are what makes F1 more than "did the name exist". `GDP per capita
    (current US$)` and `GDP (current US$)` both exist in a World Bank frame, and
    choosing the wrong one is a *correct-looking* answer -- present, well-formed,
    and semantically wrong. A scorer that only checks existence passes it.
    """
    target = _tokens(correct)
    out = []
    for column in columns:
        if column == correct:
            continue
        shared = target & _tokens(column)
        if shared:
            out.append(column)
    return sorted(out)


def check_f1_gold(case: F1Case, tools: dict, state: dict) -> GoldCheck:
    """Load the frame for real and test the template's claims against it."""
    loader = tools.get(case.loader)
    if loader is None:
        return GoldCheck(case.case_id, False, problems=[f"no tool {case.loader!r}"])

    loader.execute(**_loader_args(case))
    frame = (state.get("data_store") or {}).get(case.frame)
    if frame is None:
        return GoldCheck(case.case_id, False, problems=["frame not loaded"])

    columns = _columns_of(frame)
    problems: list[str] = []

    if case.correct_column not in columns:
        problems.append(f"correct column {case.correct_column!r} is not in the frame")

    for guess in case.wrong_guesses:
        if guess in columns:
            # Not a cosmetic problem: this template would emit a "failure" that
            # succeeded, teaching the model that a working call is an error.
            problems.append(f"wrong guess {guess!r} is a real column")

    return GoldCheck(
        case_id=case.case_id,
        ok=not problems,
        columns=columns,
        confusable=confusable_columns(case.correct_column, columns),
        problems=problems,
    )


# --- F2: the gold is a match rate ---------------------------------------------

#: The correct key pair must fill at least this share of the value column.
#: Measured on the real data: the right pair fills 74.1%, the wrong ones 0%.
MIN_CORRECT_FILL_RATE = 0.5

#: ...and must beat every wrong pair by at least this much. Without the margin,
#: a case where both pairs happen to match would pass, and the sample would
#: teach a distinction that does not exist in the data.
MIN_FILL_MARGIN = 0.2


def fill_rate(frame, column: str) -> float:
    """Share of rows where `column` has a value after a left join.

    This is the discriminator because row count is not one. `merge_dataframes`
    is a left join, so the wrong key pair returns the same 251x96 frame as the
    right one -- only the *contents* differ, and only in whether they are empty.
    """
    if frame is None or column not in getattr(frame, "columns", []):
        return 0.0
    try:
        return float(frame[column].notna().mean())
    except (TypeError, ValueError):
        return 0.0


@dataclass
class JoinCheck:
    case_id: str
    ok: bool
    correct_fill_rate: float = 0.0
    wrong_fill_rates: list[float] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"case_id": self.case_id, "ok": self.ok,
                "correct_fill_rate": round(self.correct_fill_rate, 4),
                "wrong_fill_rates": [round(r, 4) for r in self.wrong_fill_rates],
                "problems": self.problems}


def check_f2_gold(case, tools: dict, state: dict) -> JoinCheck:
    """Load both sides, run the right join and every declared wrong one, and
    compare how much of the value column actually survived."""
    problems: list[str] = []

    for name, args in ((case.data_frame, {"dataset": case.data_dataset,
                                          "output_dataframe_name": case.data_frame}),
                       (case.geo_frame, {"geodataset": case.geo_dataset,
                                         "output_geodataframe_name": case.geo_frame})):
        tool = "load_geodata" if "geodataset" in args else "load_data"
        if tools.get(tool) is None:
            return JoinCheck(case.case_id, False, problems=[f"no tool {tool!r}"])
        tools[tool].execute(**args)

    def merged_fill(pair: tuple[str, str], name: str) -> float:
        tools["merge_dataframes"].execute(
            dataframe_name=case.data_frame, geodataframe_name=case.geo_frame,
            statkey=pair[0], geokey=pair[1], output_dataframe_name=name)
        return fill_rate((state.get("data_store") or {}).get(name), case.value_column)

    correct = merged_fill((case.data_key, case.geo_key), "gold_merge")
    wrong = [merged_fill(pair, f"gold_bad_{i}")
             for i, pair in enumerate(case.wrong_pairs)]

    if correct < MIN_CORRECT_FILL_RATE:
        problems.append(
            f"the declared correct key pair {case.data_key!r}/{case.geo_key!r} only fills "
            f"{correct:.1%} of {case.value_column!r} -- the 'right' answer does not work")
    if wrong and correct - max(wrong) < MIN_FILL_MARGIN:
        problems.append(
            f"the best wrong pair fills {max(wrong):.1%} against the correct "
            f"{correct:.1%} -- too close to teach a distinction")

    return JoinCheck(case.case_id, not problems, correct_fill_rate=correct,
                     wrong_fill_rates=wrong, problems=problems)
