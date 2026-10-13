#!/usr/bin/env python
"""V4: measure the untrained model, before anything is fine-tuned.

Without this number the later "improvement" is unfalsifiable -- there is nothing
to improve from. So this runs the base checkpoint over the built samples with the
same tool manifest they were built with, executes whatever it asks for against a
real tool server, and scores the result.

Two environment facts shape the design:

* the interpreter with torch has no geopandas here, so tools run in a separate
  process over a pipe (`scripts/tool_server.py`) rather than by installing the
  GIS stack into the model's environment;
* an untrained 0.6B is not expected to emit well-formed tool calls. That is not
  a reason to make the harness lenient -- it is the measurement. Every parsing
  strategy that had to be tried is recorded, and a sample the model never
  finished is reported as a format failure rather than counted as a wrong answer.

Greedy decoding; numeric reproducibility still depends on the execution environment.

    python scripts/run_baseline.py \
        --samples evals/candidates/real-48/samples.jsonl \
        --out evals/runs/a800-baseline-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def real_sources_dir() -> str:
    """Where the real-data snapshots live.

    Overridable, and defaulted rather than hard-coded, because a candidate built
    elsewhere carries its own copy and the server that runs it has no network to
    fetch one from.
    """
    return os.environ.get("STRATA_REAL_SOURCES_DIR") or str(
        ROOT / "evals" / "sources")

from scripts.baseline_state import (
    append_progress_row,
    coverage_for,
    exclusive_lock,
    load_progress,
    progress_metadata,
    read_jsonl_samples,
    runtime_source_manifest,
    write_json_atomic,
)
from scripts.baseline_state import samples_digest as compute_samples_digest

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def extract_tool_call(text: str) -> tuple[dict | None, str | None]:
    """Pull one tool call out of the model's text, and say how.

    The strategy is recorded because "the model called a tool" and "the regex
    found something that looked like a tool call" are different claims, and only
    one of them is about the model.
    """
    for match in TOOL_CALL_BLOCK.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("name"):
            return payload, "tool_call_block"

    for match in JSON_OBJECT.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("name"):
            return payload, "bare_json_call"
        if payload.get("status"):
            return {"__termination__": payload}, "bare_json_termination"
    return None, None


def extract_termination(text: str) -> tuple[dict | None, str | None]:
    for match in JSON_OBJECT.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("status"), str):
            return payload, "json_status"
    return None, None


def append_tool_turn(messages: list[dict], call_id: str, call: dict,
                     observation: str) -> list[dict]:
    """Append a standard assistant tool call and its paired observation."""
    return [
        *messages,
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call.get("arguments") or {},
            },
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": observation},
    ]


#: What each kind of evaluation is, in the report's own words.
#:
#: Hardcoded before, and that is how a *probe* checkpoint came to be described as
#: "the untrained checkpoint" -- the string said `untrained` because it was
#: written for the base run and never varied. A report whose self-description
#: does not depend on what it measured cannot be trusted about what it measured.
EVALUATION_KINDS = {
    "base": {
        "what": "V4 baseline: the untrained checkpoint on the built samples",
        "not": ("not a fine-tuned result, and not a stable accuracy estimate "
                "at this n"),
    },
    "sft-probe": {
        # Must not contain the word the base description turns on: a reader
        # grepping for how a checkpoint was produced would otherwise match this
        # one and conclude the probe run measured a base model.
        "what": ("SFT probe checkpoint evaluated on built samples: a training "
                 "smoke test, not a capability measurement"),
        "not": ("exploratory small-sample result from a short probe run; not a "
                "final accuracy estimate, and not comparable to a full training "
                "run"),
    },
}


def evaluation_description(kind: str = "base") -> dict:
    """`{"what", "not"}` for the kind of evaluation being written.

    Raises rather than defaulting: labelling a probe run as the base run is the
    mistake this exists to prevent, and a silent fallback would reintroduce it
    the moment someone passes a typo.
    """
    try:
        return dict(EVALUATION_KINDS[kind])
    except KeyError:
        raise ValueError(
            f"unknown evaluation kind {kind!r}; known: {sorted(EVALUATION_KINDS)}"
        ) from None


def samples_digest(samples: list[dict]) -> str:
    return compute_samples_digest(samples)


def select_device(torch_module, requested: str) -> str:
    """Choose an accelerator explicitly, or the fastest available default."""
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(f"requested device {requested!r} is unavailable")
        return requested

    if requested == "mps":
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("requested device 'mps' is unavailable")
        return requested

    if requested == "cpu":
        return requested
    raise ValueError(f"unsupported device {requested!r}")


def select_attention_implementation(device: str, requested: str) -> str:
    """Use an attention backend compatible with the selected device."""
    if requested != "auto":
        return requested
    return "eager" if device == "mps" else "sdpa"


def render_prompt(tokenizer, messages: list[dict], tools: list[dict] | None) -> tuple[str, str]:
    """Render a prompt and report whether native tool schemas were accepted."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        try:
            return tokenizer.apply_chat_template(messages, tools=tools, **kwargs), "native_tools"
        except (TypeError, ValueError):
            # Some tokenizer revisions expose no `tools` template variable. The
            # sample's textual manifest remains a usable fallback, but the
            # report must say that the native schema was not actually used.
            pass
    return tokenizer.apply_chat_template(messages, **kwargs), "text_manifest"


def load_checkpoint(model_loader, checkpoint: str, attention_implementation: str):
    return model_loader.from_pretrained(
        checkpoint, torch_dtype="auto", attn_implementation=attention_implementation)


def requires_catalog_rates(samples: list[dict]) -> bool:
    return any(
        (sample.get("source_provenance") or {}).get("source_kind")
        != "synthetic_generator"
        for sample in samples
    )


class ToolServer:
    """The GIS tools, in the interpreter that has them."""

    def __init__(self, python: str | None = None):
        # The GIS environment is intentionally separate from the model
        # environment.  Keep the local default useful, but make the server
        # interpreter portable to another machine.
        python = python or os.getenv("STRATA_TOOL_PYTHON") or sys.executable
        self.process = subprocess.Popen(
            [python, str(ROOT / "scripts" / "tool_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.next_id = 0

    def _request(self, request: dict) -> dict:
        self.next_id += 1
        request = {**request, "id": self.next_id}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return {"observation": "Error: tool server closed", "failed": True}
        return json.loads(line)

    def call(self, tool: str, arguments: dict) -> dict:
        return self._request({"tool": tool, "arguments": arguments})

    def configure_for_sample(self, sample: dict) -> None:
        source_kind = (sample.get("source_provenance") or {}).get("source_kind")
        if source_kind == "synthetic_generator":
            generator = (sample.get("source_provenance") or {}).get("generator") or {}
            response = self._request({"op": "configure_synthetic",
                                      "specs": generator.get("specs") or []})
            expected = "synthetic"
        elif source_kind == "real_source_snapshot":
            generator = (sample.get("source_provenance") or {}).get("generator") or {}
            # The snapshots are read by the tool process, so it needs the
            # directory as well as the specs. Defaulted rather than hard-coded so
            # a candidate built elsewhere can point at its own copy.
            response = self._request({"op": "configure_real",
                                      "specs": generator.get("specs") or [],
                                      "sources_dir": real_sources_dir()})
            expected = "real"
        else:
            response = self._request({"op": "configure_catalog"})
            expected = "catalog"
        if response.get("failed") or response.get("configured") != expected:
            raise RuntimeError(response.get("observation", "tool profile setup failed"))

    def _join_fill_rate(self, sample: dict, pair: tuple[str, str], kind: str) -> float:
        """Ask the tool process how much of the value column survives a key pair.

        One implementation for both provenance kinds: they differ in the op name,
        the loader and whether a snapshot directory is needed, and in nothing
        else. Two copies would drift, and the fill rate is what the scorer reads.
        """
        loader = ("load_synthetic_table" if kind == "synthetic_generator"
                  else "load_real_table")
        provenance = sample.get("source_provenance") or {}
        generator = provenance.get("generator") or {}
        specs = generator.get("specs") or []
        loads = []
        for message in sample.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != loader:
                    continue
                loads.append(function.get("arguments") or {})
        if len(loads) < 2 or not specs:
            raise ValueError(f"{kind} F2 sample is missing two loader calls or specs")
        request = {
            "op": ("synthetic_fill_rate" if kind == "synthetic_generator"
                   else "real_fill_rate"),
            "specs": specs,
            "data_frame": loads[0].get("output_dataframe_name"),
            "geo_frame": loads[1].get("output_dataframe_name"),
            "data_handle": loads[0].get("handle"),
            "geo_handle": loads[1].get("handle"),
            "value_column": (sample.get("gold") or {}).get("value_column"),
            "statkey": pair[0],
            "geokey": pair[1],
        }
        if kind == "real_source_snapshot":
            request["sources_dir"] = real_sources_dir()
        response = self._request(request)
        if response.get("failed"):
            raise RuntimeError(response.get("observation", f"{kind} fill-rate failed"))
        return float(response.get("fill_rate", 0.0))

    def synthetic_fill_rate(self, sample: dict, pair: tuple[str, str]) -> float:
        return self._join_fill_rate(sample, pair, "synthetic_generator")

    def real_fill_rate(self, sample: dict, pair: tuple[str, str]) -> float:
        return self._join_fill_rate(sample, pair, "real_source_snapshot")

    def fill_rates(self) -> dict:
        """The F2 match rates, measured by the process that owns geopandas."""
        request = {"id": "rates", "op": "fill_rates"}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return {}
        return json.loads(line).get("fill_rates") or {}

    def close(self) -> None:
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - shutting down, nothing to report
            self.process.kill()


def run_one(model, tokenizer, sample: dict, server: ToolServer, max_turns: int,
            max_new_tokens: int) -> dict:
    """One task, driven to a termination or to the turn limit."""
    import torch

    messages = list(sample["messages"][:2])  # system + user, as built
    calls: list[dict] = []
    # What the *tool* returned, not what the model said about it. Without this
    # the two are indistinguishable in the report: an error the model wrote into
    # its own termination reads the same as an error the tool actually produced,
    # and a causal claim built on that difference cannot be checked. One was
    # drawn from `termination.error_message` alone, and could not be.
    tool_observations: list[dict] = []
    parse_strategies: list[str] = []
    raw_turns: list[str] = []
    prompt_modes: list[str] = []
    termination = None
    termination_strategy = None
    tools = sample.get("tools") or None

    for _turn in range(max_turns):
        prompt, prompt_mode = render_prompt(tokenizer, messages, tools)
        prompt_modes.append(prompt_mode)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        raw_turns.append(text)

        call, strategy = extract_tool_call(text)
        if call is None:
            termination, termination_strategy = extract_termination(text)
            break

        parse_strategies.append(strategy)
        if "__termination__" in call:
            termination, termination_strategy = call["__termination__"], strategy
            break

        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        call_id = f"call_{len(calls) + 1}"
        parsed_call = {"name": str(call.get("name")), "arguments": arguments}
        calls.append({"id": call_id, **parsed_call})
        result = server.call(parsed_call["name"], arguments)
        tool_observations.append({
            "call_id": call_id,
            "tool": parsed_call["name"],
            "arguments": arguments,
            "observation": result.get("observation", ""),
            "failed": bool(result.get("failed")),
        })
        messages = append_tool_turn(
            messages, call_id, parsed_call, result.get("observation", ""))

    return {
        "attempt": {"termination": termination, "calls": calls},
        "parse_strategies": parse_strategies,
        "termination_strategy": termination_strategy,
        "termination_keys": sorted(termination) if isinstance(termination, dict) else None,
        "prompt_modes": prompt_modes,
        # The raw text is kept because without it a format failure is
        # unattributable: "the model cannot emit a tool call" and "the model was
        # cut off before it did" produce the same outcome code and are different
        # findings. The first baseline run made exactly that mistake.
        "raw_turns": raw_turns,
        # Kept so a later claim about *why* a run failed can be checked against
        # what the tools actually returned rather than against the model's
        # account of it.
        "tool_observations": tool_observations,
        "turns": len(calls),
    }


def evaluate_sample(model, tokenizer, sample: dict, server: ToolServer, rates: dict,
                    max_turns: int, max_new_tokens: int) -> dict:
    from strata.train.score import score_attempt

    server.configure_for_sample(sample)
    run = run_one(model, tokenizer, sample, server, max_turns, max_new_tokens)
    case_id = sample["sample_id"].split(":", 1)[-1]
    source_kind = (sample.get("source_provenance") or {}).get("source_kind")
    if source_kind == "synthetic_generator":
        sample_fill_rate = lambda pair: server.synthetic_fill_rate(sample, pair)
    elif source_kind == "real_source_snapshot":
        sample_fill_rate = lambda pair: server.real_fill_rate(sample, pair)
    else:
        sample_fill_rate = lambda pair: rates.get(f"{case_id}|{pair[0]}|{pair[1]}", 0.0)
    outcome, detail = score_attempt(
        sample, run["attempt"],
        frame_columns=list(sample.get("gold", {}).get("frame_columns") or []),
        fill_rate=sample_fill_rate)
    return {"sample_id": sample["sample_id"], "family": sample["family"],
            "outcome": outcome, "detail": detail, **run}


def run_pending_samples(samples: list[dict], rows: list[dict], evaluate, progress: Path,
                        deadline: float | None = None) -> str:
    measured = {row["sample_id"] for row in rows}
    for sample in samples:
        if sample["sample_id"] in measured:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            return "budget_exhausted"
        row = evaluate(sample)
        if row.get("sample_id") != sample["sample_id"]:
            raise ValueError("evaluator returned the wrong sample_id")
        append_progress_row(progress, row)
        rows.append(row)
        measured.add(sample["sample_id"])
        print(f"  {row['outcome']:26s} {sample['sample_id']:24s} turns={row.get('turns')}", flush=True)
    return "complete"


def run_resumable(args) -> int:
    from strata.train.score import SCORER_VERSION, summarise

    started = time.monotonic()
    samples = read_jsonl_samples(args.samples)
    if args.limit:
        samples = samples[:args.limit]
    progress = args.progress or args.out.with_suffix(".progress.jsonl")
    tool_python = args.tool_python or os.getenv("STRATA_TOOL_PYTHON") or sys.executable
    source_manifest = runtime_source_manifest(ROOT, args.model, tool_python)
    if args.source_manifest:
        source_manifest["task_bundle"] = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    metadata = progress_metadata(
        samples, model=args.model, evaluation_kind=args.evaluation_kind, device=args.device,
        attention_implementation=args.attn_implementation, max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens, source_manifest=source_manifest)
    metadata["scorer_version"] = SCORER_VERSION
    with exclusive_lock(progress.with_suffix(progress.suffix + ".lock")):
        if not args.resume and (args.out.exists() or progress.exists()):
            raise FileExistsError("output/progress already exists; use --resume or a new output path")
        if args.resume and args.out.exists() and not progress.exists():
            raise ValueError("cannot resume a legacy report without its progress journal")
        _, rows = load_progress(progress, metadata)
        device = args.device
        attention = args.attn_implementation
        if args.out.exists():
            previous = json.loads(args.out.read_text(encoding="utf-8"))
            if previous.get("metadata") != metadata:
                raise ValueError("report metadata mismatch")
            device = previous.get("device", device)
            attention = previous.get("attention_implementation", attention)
        stop_reason = "interrupted"
        server = None
        try:
            if coverage_for(samples, rows)["complete"]:
                stop_reason = "complete"
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device = select_device(torch, args.device)
                attention = select_attention_implementation(device, args.attn_implementation)
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = load_checkpoint(AutoModelForCausalLM, args.model, attention)
                model.to(device)
                model.eval()
                server = ToolServer(tool_python)
                rates = server.fill_rates() if requires_catalog_rates(samples) else {}
                deadline = started + args.budget_seconds if args.budget_seconds else None
                stop_reason = run_pending_samples(
                    samples, rows,
                    lambda sample: evaluate_sample(model, tokenizer, sample, server, rates,
                                                   args.max_turns, args.max_new_tokens),
                    progress, deadline)
        finally:
            if server is not None:
                server.close()
            report = {
                **evaluation_description(args.evaluation_kind),
                **{key: value for key, value in metadata.items() if key != "sample_ids"},
                "metadata": metadata,
                "device": device,
                "attention_implementation": attention,
                "progress_file": str(progress),
                "stop_reason": stop_reason,
                "coverage": coverage_for(samples, rows),
                "n": len(rows),
                "summary": summarise([row["outcome"] for row in rows]),
                "rows": rows,
            }
            write_json_atomic(args.out, report)
        print(f"\nsummary: {json.dumps(report['summary'], ensure_ascii=False)}")
        print(f"coverage: {len(rows)}/{len(samples)}; wrote {args.out}")
        return 0 if report["coverage"]["complete"] else 75


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, mps, cuda, or a CUDA device such as cuda:0")
    ap.add_argument("--attn-implementation", default="auto",
                    choices=("auto", "eager", "sdpa", "flash_attention_2"),
                    help="attention backend; auto selects eager on MPS and SDPA elsewhere")
    ap.add_argument("--checkpoint-kind", "--evaluation-kind", dest="evaluation_kind",
                    choices=sorted(EVALUATION_KINDS), default="base",
                    help="what this run measures; decides how the report describes "
                         "itself, so a probe checkpoint is never written up as the "
                         "untrained baseline")
    ap.add_argument("--tool-python", default=None,
                    help="Python interpreter for the GIS tool server; defaults to "
                         "$STRATA_TOOL_PYTHON or the model interpreter")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="256 was the first setting and it silently truncated every\n                         run before the model finished emitting anything parseable --\n                         the resulting 6/6 format failure was an artefact of the budget")
    ap.add_argument("--progress", type=Path, help="durable JSONL journal; defaults next to --out")
    ap.add_argument("--resume", action="store_true", help="resume only when data/weights/config/source hashes match")
    ap.add_argument("--budget-seconds", type=float, default=0,
                    help="soft wall-clock budget checked between samples; 0 means unlimited")
    ap.add_argument("--source-manifest", type=Path, help="frozen task-bundle identity to bind to the journal")
    args = ap.parse_args()
    if args.limit < 0 or args.max_turns < 1 or args.max_new_tokens < 1 or args.budget_seconds < 0:
        ap.error("limit/budget must be nonnegative and turn/token limits positive")
    return run_resumable(args)


if __name__ == "__main__":
    raise SystemExit(main())
