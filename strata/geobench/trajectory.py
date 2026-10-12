"""Recording a run as a GeoBenchX trajectory, and writing it where the judge reads.

GeoBenchX scores `Solution(steps=[Step(function_name, arguments), ...])`. The
result the tool returned is not part of it: `get_solution_code()` renders each
step as `function_name(k='v')` and the judge compares that text against the
reference trajectory. Two consequences worth stating plainly:

* the natural-language answer does not affect the score, and
* a run that "looks right" but never called the tools scores zero.

So the recorder is the load-bearing part of the adapter, not a log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corecoder.tool_errors import is_error_result


@dataclass
class Step:
    function_name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict:
        return {"function_name": self.function_name, "arguments": self.arguments, "comment": None}


@dataclass
class Trajectory:
    """The tool calls a single task produced, in order."""

    task_id: str
    steps: list[Step] = field(default_factory=list)
    #: Why the turn ended, so a truncated run is never scored as a finished one.
    stop_reason: str = "completed"
    #: What the model said at the end. The judge never sees this, but a report
    #: that claims a run "reached the answer" needs it as evidence.
    final_response: str = ""
    #: (tool name, result) for every call whose result was an error string.
    tool_errors: list[tuple[str, str]] = field(default_factory=list)
    #: Names the model asked for that the profile does not expose, in order.
    #:
    #: Kept out of `steps` on purpose -- the judge must not see a call that was
    #: never made -- but not thrown away either. Without this, "the model never
    #: needed the other twelve tools" is unfalsifiable: every time it reached for
    #: one, the evidence was discarded. Reported as counts, never scored.
    unexposed: list[str] = field(default_factory=list)
    #: Identical calls that failed, by call key -> how many times.
    #:
    #: The deterministic cost of trial and error, and it costs nothing to
    #: measure because it never depends on the judge. This is the number P1 has
    #: to move: on the 918547 case the model guessed one column name per call,
    #: nine times, because nothing told it which names existed.
    repeat_failures: dict[str, int] = field(default_factory=dict)
    #: (tool, seconds late, deadline_mode) for every call that returned past the
    #: turn's deadline.
    #:
    #: GeoBenchX's own tools are **cooperative**: they are in-process functions
    #: and nothing outside can stop one mid-call. Its profile excludes `bash`
    #: and `agent`, so this list is the honest statement of where the deadline
    #: does and does not hold -- not a claim that all tools are bounded.
    deadline_overruns: list[tuple[str, float, str]] = field(default_factory=list)
    #: Where this run's raw tool results were stored, when evidence was on.
    #:
    #: Diagnostics only. `steps` is what the judge reads and it is unchanged:
    #: an artifact the harness filed away is not a tool call the model made.
    evidence_refs: list[dict] = field(default_factory=list)
    #: How often the evidence block engaged and how much it saved, for this task.
    #: Empty when the arm was off -- which is what makes a paired comparison
    #: possible without reading the artifacts.
    evidence_stats: dict = field(default_factory=dict)
    #: Wall-clock seconds for the turn.
    wall_seconds: float = 0.0
    #: Tokens this task spent (prompt + completion, measured as a delta).
    tokens: int = 0

    def record(self, function_name: str, arguments: dict[str, Any]) -> None:
        self.steps.append(Step(function_name, dict(arguments)))

    def record_result(self, function_name: str, result: str) -> None:
        # The same predicate the retry counter uses. Two spellings of "is this a
        # failure" would let the recorder and the counter disagree, and the
        # counts are the whole evidence base for P1.
        if is_error_result(result):
            self.tool_errors.append((function_name, result[:500]))

    def as_solution(self) -> dict:
        return {"steps": [s.as_dict() for s in self.steps]}

    def wrote_anything(self) -> bool:
        return bool(self.steps)


def write_submission(
    trajectories: list[Trajectory],
    out_path: Path,
    tasks: list[dict],
    model_label: str,
) -> Path:
    """Write a GeoBenchX task set with `generated_solution` filled in.

    The input `tasks` are GeoBenchX's own task dicts, copied through unchanged
    except for the field the judge reads -- the scorer expects the original
    fields (including the reference solutions) to still be there.
    """
    by_id = {t.task_id: t for t in trajectories}
    out: list[dict] = []
    for task in tasks:
        entry = dict(task)
        traj = by_id.get(entry.get("task_ID"))
        entry["generated_solution"] = traj.as_solution() if traj else {"steps": []}
        entry["match_reasoning_LLM"] = None
        entry["match_score_LLM"] = None
        entry["submission_stop_reason"] = traj.stop_reason if traj else "missing"
        out.append(entry)

    payload = {"metadata": {"model": model_label}, "tasks": out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_reference_tasks(geobenchx_root: Path) -> list[dict]:
    """Read GeoBenchX's own task file, so a submission carries its fields through."""
    path = geobenchx_root / "benchmark_set" / "tasks_and_reference_solutions.json"
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]
