"""The gate a record must pass before it may be exported for training.

The p2c audit shipped 18 candidates, each carrying `train_eligible=false` and a
prose `exclusion_reasons`. That is a statement of intent, not a mechanism: it is
one hand-written field, and nothing would have caught a record that said `true`
while failing a gate. The gates themselves are mechanical, so they live here.

Three, from the round that asked for this:

1. **Complete, aligned observations.** A `tool_call_plan` is the model's calls;
   it is not a trace. Without what each call returned there is nothing to learn
   from, and inventing the missing replies would be fabricating data.
2. **Semantics not under review.** A plan can be well-formed and still disclose
   the wrong thing -- one p2c candidate reported a date range copied from a
   mislabelled dataset name. Unreviewed is not the same as fine.
3. **Not from the reserved evaluation cluster.** Scoring on the tasks that
   produced the training data measures memorisation.

`validate_record` reports **every** blocker rather than the first, because
fixing a record one refusal at a time is worse than seeing the whole list.

Nothing here writes anything: `audit_records` reads a file of candidates and
returns verdicts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..provenance import RESERVED_TASK_IDS

#: The only answer status that clears the semantics gate. `not_end_to_end_verified`
#: is the status most of the audit records carry, and it is not a pass -- it says
#: the number was never checked against the data, which is precisely the open
#: question the gate exists to catch.
ACCEPTED_ANSWER_STATUSES: frozenset[str] = frozenset({"verified"})

BLOCKER_INCOMPLETE_OBSERVATIONS = "incomplete_observations"
BLOCKER_MISSING_TASK_ID = "missing_task_id"
BLOCKER_RESERVED_TASK = "reserved_eval_task"
BLOCKER_OPEN_ISSUES = "open_review_issues"
BLOCKER_SEMANTICS_UNREVIEWED = "semantics_unreviewed"


@dataclass
class Verdict:
    record_id: str
    eligible: bool
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"record_id": self.record_id, "eligible": self.eligible,
                "blockers": self.blockers}


def task_id_of(record: dict) -> str:
    """Where a record's task id lives, tolerating both layouts in the wild.

    The audit artifact nests it under `source`; a hand-built record puts it at
    the top level. Whitespace-only values are not task identities.
    """
    candidates = (
        record.get("task_id"),
        (record.get("source") or {}).get("task_ID"),
    )
    for candidate in candidates:
        normalized = str(candidate).strip() if candidate is not None else ""
        if normalized:
            return normalized
    return ""


def observations_are_complete(record: dict) -> bool:
    """One non-empty observation per planned call.

    Counted, not just presence-checked: a trace that recorded two outcomes for
    four calls has holes, and a hole is where a fabricated step would go.
    """
    observations = record.get("observations")
    if not isinstance(observations, list):
        return False
    plan = record.get("tool_call_plan") or []
    if len(observations) != len(plan):
        return False
    return all(isinstance(o, str) and o.strip() for o in observations)


def semantics_blockers(record: dict) -> list[str]:
    review = record.get("review") or {}
    blockers: list[str] = []

    planning = review.get("planning_status")
    if planning != "pass":
        blockers.append(f"{BLOCKER_SEMANTICS_UNREVIEWED}:planning_status={planning}")

    answer = review.get("answer_status")
    if answer not in ACCEPTED_ANSWER_STATUSES:
        blockers.append(f"{BLOCKER_SEMANTICS_UNREVIEWED}:answer_status={answer}")

    if review.get("issues"):
        blockers.append(BLOCKER_OPEN_ISSUES)

    return blockers


def validate_record(record: dict, reserved: Iterable[str] = RESERVED_TASK_IDS) -> Verdict:
    blockers: list[str] = []
    task_id = task_id_of(record)

    if not observations_are_complete(record):
        blockers.append(BLOCKER_INCOMPLETE_OBSERVATIONS)

    if not task_id:
        blockers.append(BLOCKER_MISSING_TASK_ID)
    elif task_id in set(reserved):
        blockers.append(BLOCKER_RESERVED_TASK)

    blockers.extend(semantics_blockers(record))

    return Verdict(record_id=str(record.get("record_id", "?")),
                   eligible=not blockers, blockers=blockers)


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit_records(path: Path, reserved: Iterable[str] = RESERVED_TASK_IDS) -> dict:
    """Verdicts for a candidate file, plus the disagreements.

    `wrongly_claimed` is the finding this exists for: a record asserting
    `train_eligible` while a gate refuses it. A record that claims `false` and
    is refused agrees with the gate and is not news.
    """
    reserved = set(reserved)
    verdicts = [validate_record(r, reserved) for r in read_records(path)]

    claimed = sum(1 for r in read_records(path) if r.get("train_eligible") is True)
    wrongly = [v.as_dict() for v, r in zip(verdicts, read_records(path))
               if r.get("train_eligible") is True and not v.eligible]

    return {
        "source": str(path),
        "records": len(verdicts),
        "gate_eligible": sum(1 for v in verdicts if v.eligible),
        "claimed_eligible": claimed,
        "wrongly_claimed": wrongly,
        "verdicts": [v.as_dict() for v in verdicts],
    }
