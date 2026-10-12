#!/usr/bin/env python
"""Decide which candidate records may leave for training, and refuse the rest.

Offline and read-only unless `--export` is given. The gate itself lives in
`strata.geobench.export`; this is the command-line face of it.

Exits non-zero when a record *claims* `train_eligible` while a gate refuses it,
so this can run as a check rather than only as a report. A file where every
record already says `false` and every gate agrees is a quiet success -- which is
the state of the first audit pass, and worth being able to confirm cheaply.

    python scripts/validate_export.py \
        --candidates evals/reviews/p2c_20261004/audited_candidates.jsonl
    python scripts/validate_export.py --candidates ... \
        --export evals/training/eligible.jsonl --report evals/training/gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata.geobench.export import RESERVED_TASK_IDS, audit_records, read_records, validate_record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, required=True,
                    help="JSONL of candidate records")
    ap.add_argument("--report", type=Path, default=None, help="write the full audit as JSON")
    ap.add_argument("--export", type=Path, default=None,
                    help="write only the gate-eligible records here; nothing is written "
                         "when the gate refuses everything")
    ap.add_argument("--reserved-from", type=Path, default=None,
                    help="JSON list of reserved task ids, overriding the built-in nine")
    args = ap.parse_args()

    if not args.candidates.exists():
        print(f"no such file: {args.candidates}", file=sys.stderr)
        return 2

    reserved = set(RESERVED_TASK_IDS)
    if args.reserved_from is not None:
        reserved = set(json.loads(args.reserved_from.read_text(encoding="utf-8")))

    report = audit_records(args.candidates, reserved)

    print(f"candidates : {report['records']}")
    print(f"gate says  : {report['gate_eligible']} eligible, "
          f"{report['records'] - report['gate_eligible']} refused")
    print(f"file says  : {report['claimed_eligible']} marked train_eligible=true")
    print(f"reserved   : {len(reserved)} task ids held back from training")

    reasons = Counter(b.split(":")[0] for v in report["verdicts"] for b in v["blockers"])
    if reasons:
        print("\nwhy records were refused (a record can hit several):")
        for reason, count in reasons.most_common():
            print(f"  {count:>4}  {reason}")

    if report["wrongly_claimed"]:
        print(f"\n⚠️  {len(report['wrongly_claimed'])} record(s) claim train_eligible "
              "while a gate refuses them:", file=sys.stderr)
        for item in report["wrongly_claimed"]:
            print(f"    {item['record_id']}: {', '.join(item['blockers'])}", file=sys.stderr)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"\nwrote {args.report}")

    if args.export is not None:
        eligible = [r for r in read_records(args.candidates) if validate_record(r, reserved).eligible]
        if not eligible:
            print("export: nothing eligible, so nothing was written")
        else:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in eligible) + "\n",
                encoding="utf-8")
            print(f"export: wrote {len(eligible)} record(s) to {args.export}")

    return 1 if report["wrongly_claimed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
