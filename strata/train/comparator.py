"""V2: compare a stored observation against a fresh replay of the same call.

Repeated calls can differ in timestamps, workspace paths, and rendered-image
bytes. Comparison uses exact structural checks and normalized observations:

| layer | what | how |
|---|---|---|
| structural | call id, tool name, arguments, failed flag | **exact** -- a difference here is a real difference |
| observation | the tool's text | normalised, then exact |
| tolerance | floats | rounded to `FLOAT_DECIMALS`, a number written down rather than implied |
| normalised | ISO timestamps, temp paths | replaced with placeholders |
| manifest | base64 payloads | compared by **length**, not content |

Base64 is deliberately **not** matched by a generic "long token" rule: a SHA256
digest is a 64-character run of hex and would be swallowed by one, which would
make two genuinely different tool digests compare equal. A payload counts as
base64 here only if it is long *and* contains something outside lowercase hex.

Every difference is reported with its layer. "Not equal" alone cannot be acted
on -- the reader needs to know whether the arguments changed or only the clock.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

#: Bump when the rules below change. Samples carry the version they were built
#: with, so a rule change shows up as a data change instead of a silent shift in
#: what "equal" means.
COMPARATOR_VERSION = "c1"

#: Decimal places used when comparing floats. This *is* the tolerance: an
#: absolute one, applied at this precision.
FLOAT_DECIMALS = 6

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_TEMP_PATH = re.compile(r"/(?:private/)?(?:tmp|var/folders)/[^/\s'\"]+")
_LONG_RUN = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")
_FLOAT = re.compile(r"(?<![\w.])-?\d+\.\d+")


def _looks_like_base64(text: str) -> bool:
    """Long, and not simply hexadecimal.

    A digest is 64 hex characters and would otherwise be replaced by a
    length-only placeholder -- making two different digests compare equal. That
    is the failure mode this predicate exists to prevent.
    """
    return any(c not in "0123456789abcdef" for c in text)


def _round_float(literal: str) -> str:
    try:
        value = float(literal)
    except ValueError:
        return literal
    return f"{round(value, FLOAT_DECIMALS):.{FLOAT_DECIMALS}f}"


def normalise(text: str) -> str:
    """Fold away the differences two correct runs may legitimately have."""
    out = _TIMESTAMP.sub("<TS>", str(text))
    out = _TEMP_PATH.sub("<TMP>", out)
    out = _LONG_RUN.sub(
        lambda m: f"<B64:{len(m.group(0))}>" if _looks_like_base64(m.group(0)) else m.group(0),
        out)
    return _FLOAT.sub(lambda m: _round_float(m.group(0)), out)


@dataclass
class Diff:
    layer: str
    field: str
    expected: str
    actual: str

    def as_dict(self) -> dict:
        return {"layer": self.layer, "field": self.field,
                "expected": self.expected, "actual": self.actual}


@dataclass
class Comparison:
    call_id: str
    equal: bool
    diffs: list[Diff] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"call_id": self.call_id, "equal": self.equal,
                "diffs": [d.as_dict() for d in self.diffs]}


def compare_observations(expected: str, actual: str) -> list[Diff]:
    left, right = normalise(expected), normalise(actual)
    if left == right:
        return []
    return [Diff(layer="observation", field="observation",
                 expected=left[:400], actual=right[:400])]


def compare_steps(expected: dict, actual: dict) -> Comparison:
    call_id = str(expected.get("call_id") or actual.get("call_id") or "?")
    diffs: list[Diff] = []

    for field_name in ("call_id", "tool"):
        if expected.get(field_name) != actual.get(field_name):
            diffs.append(Diff(layer="structural", field=field_name,
                              expected=str(expected.get(field_name)),
                              actual=str(actual.get(field_name))))

    if json.dumps(expected.get("arguments"), sort_keys=True, ensure_ascii=False) != \
            json.dumps(actual.get("arguments"), sort_keys=True, ensure_ascii=False):
        diffs.append(Diff(layer="structural", field="arguments",
                          expected=json.dumps(expected.get("arguments"), ensure_ascii=False),
                          actual=json.dumps(actual.get("arguments"), ensure_ascii=False)))

    if bool(expected.get("failed")) != bool(actual.get("failed")):
        diffs.append(Diff(layer="structural", field="failed",
                          expected=str(bool(expected.get("failed"))),
                          actual=str(bool(actual.get("failed")))))

    diffs.extend(compare_observations(str(expected.get("observation_raw") or ""),
                                      str(actual.get("observation_raw") or "")))

    return Comparison(call_id=call_id, equal=not diffs, diffs=diffs)


def compare_executions(expected_steps: list[dict], actual_steps: list[dict]) -> dict:
    """A report, not a boolean. The caller needs to see which step and which
    layer, or the only possible response is to re-run and hope."""
    def index_steps(steps: list[dict]) -> tuple[dict[str, dict], list[str]]:
        indexed: dict[str, dict] = {}
        duplicates: list[str] = []
        for step in steps:
            call_id = str(step.get("call_id"))
            if call_id in indexed and call_id not in duplicates:
                duplicates.append(call_id)
            indexed[call_id] = step
        return indexed, duplicates

    expected_by_id, duplicate_expected = index_steps(expected_steps)
    actual_by_id, duplicate_actual = index_steps(actual_steps)

    missing = sorted(set(expected_by_id) - set(actual_by_id))
    extra = sorted(set(actual_by_id) - set(expected_by_id))

    diffs: list[dict] = []
    for call_id in sorted(set(expected_by_id) & set(actual_by_id)):
        comparison = compare_steps(expected_by_id[call_id], actual_by_id[call_id])
        for diff in comparison.diffs:
            entry = diff.as_dict()
            entry["call_id"] = call_id
            diffs.append(entry)

    equal = not diffs and not missing and not extra and not duplicate_expected and not duplicate_actual
    return {
        "comparator_version": COMPARATOR_VERSION,
        "float_decimals": FLOAT_DECIMALS,
        "equal": equal,
        "steps_compared": len(set(expected_by_id) & set(actual_by_id)),
        "missing": missing,
        "extra": extra,
        "duplicate_expected": duplicate_expected,
        "duplicate_actual": duplicate_actual,
        "diffs": diffs,
    }
