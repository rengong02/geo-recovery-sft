"""A record may only be exported for training when it is actually trainable.

The p2c audit produced 18 candidates and marked every one `train_eligible=false`,
but that was a hand-written field in the artifact -- a rule living in prose and
one JSON value, with nothing that would catch a record wrongly marked true. The
the export requirements are checked mechanically:

1. the trace carries a complete, aligned observation for every tool call;
2. its semantics are not under review;
3. it does not come from the reserved evaluation cluster.

The point is the disagreement case: a record that *says* it is eligible while a
gate blocks it must be refused, loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strata.geobench.export import (
    BLOCKER_INCOMPLETE_OBSERVATIONS,
    BLOCKER_MISSING_TASK_ID,
    BLOCKER_OPEN_ISSUES,
    BLOCKER_RESERVED_TASK,
    BLOCKER_SEMANTICS_UNREVIEWED,
    RESERVED_TASK_IDS,
    audit_records,
    validate_record,
)

RESERVED = next(iter(sorted(RESERVED_TASK_IDS)))
ROOT = Path(__file__).resolve().parents[2]


def _record(**overrides) -> dict:
    """A record that passes every gate, so each test can spoil exactly one thing."""
    record = {
        "record_id": "p2c:head_tail:TASK_X",
        "task_id": "TASK_X",
        "train_eligible": True,
        "tool_call_plan": [
            {"function_name": "load_data", "arguments": {"dataset": "Net migration"}},
            {"function_name": "make_choropleth_map", "arguments": {"mappingkey": "2023"}},
        ],
        "observations": ["Loaded data. 266 rows.", "Map created."],
        "review": {"planning_status": "pass", "answer_status": "verified", "issues": []},
    }
    record.update(overrides)
    return record


# --- the gates ----------------------------------------------------------------


def test_a_complete_record_passes():
    verdict = validate_record(_record(), RESERVED_TASK_IDS)
    assert verdict.eligible, verdict.blockers
    assert verdict.blockers == []


def test_a_trace_without_observations_is_refused():
    """The plan is the model's calls; without what came back it is not a trace.

    This is the actual state of all 18 p2c candidates, which is why they cannot
    become SFT data by adding a flag.
    """
    verdict = validate_record(_record(observations=None), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_INCOMPLETE_OBSERVATIONS in verdict.blockers


def test_observations_must_align_with_the_plan():
    """Fewer observations than calls means some step has no outcome recorded."""
    verdict = validate_record(_record(observations=["only one"]), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_INCOMPLETE_OBSERVATIONS in verdict.blockers


def test_an_empty_observation_is_not_an_observation():
    verdict = validate_record(
        _record(observations=["Loaded data. 266 rows.", "   "]), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_INCOMPLETE_OBSERVATIONS in verdict.blockers


def test_a_record_from_the_reserved_cluster_is_refused():
    """Benchmark tasks stay in the evaluation pool.

    Training on them and then reporting a score on the same tasks measures
    memorisation, not capability.
    """
    verdict = validate_record(_record(task_id=RESERVED), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_RESERVED_TASK in verdict.blockers


def test_a_record_without_task_identity_is_refused():
    verdict = validate_record(_record(task_id=None), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_MISSING_TASK_ID in verdict.blockers


def test_a_whitespace_only_task_identity_is_refused():
    verdict = validate_record(_record(task_id="   "), RESERVED_TASK_IDS)
    assert not verdict.eligible
    assert BLOCKER_MISSING_TASK_ID in verdict.blockers


def test_semantics_under_review_is_refused():
    for review in (
        {"planning_status": "needs_review", "answer_status": "verified", "issues": []},
        {"planning_status": "pass", "answer_status": "not_end_to_end_verified", "issues": []},
        {"planning_status": "pass", "answer_status": "verified",
         "issues": ["date disclosure wrong"]},
    ):
        verdict = validate_record(_record(review=review), RESERVED_TASK_IDS)
        assert not verdict.eligible, review
        assert any(b.startswith(BLOCKER_SEMANTICS_UNREVIEWED) or b == BLOCKER_OPEN_ISSUES
                   for b in verdict.blockers), (review, verdict.blockers)


def test_every_blocker_that_applies_is_reported():
    """Not just the first one: the reader needs the whole list to fix the record."""
    verdict = validate_record(
        _record(task_id=RESERVED, observations=None,
                review={"planning_status": "needs_review", "answer_status": "x",
                        "issues": ["a"]}),
        RESERVED_TASK_IDS)
    assert BLOCKER_INCOMPLETE_OBSERVATIONS in verdict.blockers
    assert BLOCKER_RESERVED_TASK in verdict.blockers
    assert BLOCKER_OPEN_ISSUES in verdict.blockers


# --- the audit over a file ----------------------------------------------------


def test_audit_reports_the_disagreement_rather_than_the_field(tmp_path):
    """The field is not the gate. A record claiming eligible while blocked is a finding."""
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _record(),                                                   # passes
        _record(record_id="liar", task_id=RESERVED),                 # claims true, blocked
        _record(record_id="clean", train_eligible=False),            # blocked by nothing
    ]) + "\n", encoding="utf-8")

    report = audit_records(path, RESERVED_TASK_IDS)

    assert report["records"] == 3
    assert report["gate_eligible"] == 2
    assert report["claimed_eligible"] == 2
    # The one that matters: it says true, the gate says no.
    assert [r["record_id"] for r in report["wrongly_claimed"]] == ["liar"]
    assert report["wrongly_claimed"][0]["blockers"] == [BLOCKER_RESERVED_TASK]


def test_the_real_audit_candidates_are_all_refused():
    """The 18 p2c candidates: every gate blocks them, and they already say false.

    Asserted against the artifact rather than a fixture, because the claim being
    checked is about the actual file the project is carrying.
    """
    path = (ROOT / "evals" / "reviews" / "head-tail" / "audited_candidates.jsonl")
    if not path.exists():
        pytest.skip(f"audit artifact not present: {path}")

    report = audit_records(path, RESERVED_TASK_IDS)

    assert report["records"] == 18
    assert report["gate_eligible"] == 0
    assert report["wrongly_claimed"] == []
