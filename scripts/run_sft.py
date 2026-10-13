#!/usr/bin/env python3
"""Run full-parameter supervised fine-tuning on validated tool-call trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def digest_rows(rows: list[dict]) -> str:
    normalized = []
    for row in rows:
        record = dict(row)
        if isinstance(record.get("steps"), list):
            record["steps"] = [{key: value for key, value in step.items() if key != "seconds"}
                               for step in record["steps"]]
        normalized.append(record)
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in normalized
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gold_only_messages(row: dict) -> list[dict]:
    messages = row["messages"]
    kept = list(messages[:2])
    step_index = 0
    index = 2
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            kept.append(message)
            index += 1
            continue
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise ValueError(f"unpaired tool call in {row['sample_id']}")
        step = row["steps"][step_index]
        keep = step["tool"] == "load_synthetic_table" or step_index == len(row["steps"]) - 1
        if keep:
            kept.extend(messages[index:index + 2])
        step_index += 1
        index += 2
    if step_index != len(row["steps"]):
        raise ValueError(f"step/message mismatch in {row['sample_id']}")
    if kept[-1].get("role") != "assistant" or kept[-1].get("tool_calls"):
        raise ValueError(f"gold-only trajectory did not terminate in {row['sample_id']}")
    return kept


def _supervised_assistant_segments(row: dict, messages: list[dict], trajectory: str) -> list[bool]:
    assistant_segments = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if trajectory in {"gold_only", "full"}:
        return [True] * len(assistant_segments)
    if trajectory != "recovery":
        raise ValueError(f"unknown trajectory policy: {trajectory}")

    supervised: list[bool] = []
    step_index = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if message.get("tool_calls"):
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                raise ValueError(f"unpaired tool call in {row['sample_id']}")
            if step_index >= len(row["steps"]):
                raise ValueError(f"step/message mismatch in {row['sample_id']}")
            supervised.append(not bool(row["steps"][step_index].get("failed")))
            step_index += 1
        else:
            if index != len(messages) - 1:
                raise ValueError(f"non-terminal assistant message in {row['sample_id']}")
            supervised.append(True)

    if step_index != len(row["steps"]):
        raise ValueError(f"step/message mismatch in {row['sample_id']}")
    if not supervised or not supervised[-1]:
        raise ValueError(f"recovery trajectory did not terminate successfully in {row['sample_id']}")
    return supervised


def choose_rows(rows: list[dict], split_report: dict, split: str, limit: int,
                seed: int) -> list[dict]:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    assignments = split_report["assignments"]
    candidates = [row for row in rows if assignments.get(row["sample_id"]) == split]
    random.Random(seed).shuffle(candidates)
    if limit:
        candidates = candidates[:limit]
    return candidates


def encode_row(tokenizer, row: dict, trajectory: str, max_length: int) -> dict:
    messages = gold_only_messages(row) if trajectory == "gold_only" else row["messages"]
    supervised_segments = _supervised_assistant_segments(row, messages, trajectory)
    tools = row["tools"]

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{row['sample_id']} has {len(input_ids)} tokens, over max_length={max_length}"
        )
    labels = [-100] * len(input_ids)
    marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    marker_positions: list[tuple[int, int]] = []
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start < 0:
            break
        end = text.find(end_marker, start + len(marker))
        if end < 0:
            raise ValueError(f"unterminated assistant segment in {row['sample_id']}")
        marker_positions.append((start + len(marker), end + len(end_marker)))
        search_from = end + len(end_marker)
    assistant_segments = sum(1 for message in messages if message.get("role") == "assistant")
    if len(marker_positions) != assistant_segments:
        raise ValueError(
            f"assistant marker count mismatch in {row['sample_id']}: "
            f"markers={len(marker_positions)} messages={assistant_segments}"
        )
    if len(supervised_segments) != len(marker_positions):
        raise ValueError(
            f"assistant supervision count mismatch in {row['sample_id']}: "
            f"segments={len(supervised_segments)} markers={len(marker_positions)}"
        )
    for segment_index, (span_start, span_end) in enumerate(marker_positions):
        if not supervised_segments[segment_index]:
            continue
        covered = 0
        for index, (token_start, token_end) in enumerate(offsets):
            if span_start <= token_start < token_end <= span_end:
                labels[index] = input_ids[index]
                covered += 1
        if covered == 0:
            raise ValueError(f"empty assistant token span in {row['sample_id']}")
    if not any(label != -100 for label in labels):
        raise ValueError(f"no assistant labels in {row['sample_id']}")
    return {
        "sample_id": row["sample_id"],
        "input_ids": input_ids,
        "labels": labels,
        "assistant_segments": assistant_segments,
        "supervised_assistant_segments": sum(supervised_segments),
    }


def collate(batch: list[dict], pad_token_id: int):
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", choices=("gold_only", "recovery", "full"),
                        default="gold_only")
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.max_length < 1:
        parser.error("steps, learning-rate and max-length must be positive")
    if args.output.exists():
        parser.error("output must be a new directory; existing checkpoints are never overwritten")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
    identifiers = [row["sample_id"] for row in rows]
    if len(set(identifiers)) != len(rows) or set(identifiers) != set(split_report["assignments"]):
        raise ValueError("split report must cover each unique sample exactly once")
    if not split_report.get("training_ready"):
        raise ValueError("split report is not training_ready")
    train_rows = choose_rows(rows, split_report, "train", args.train_limit, args.seed)
    eval_rows = choose_rows(rows, split_report, args.eval_split, args.eval_limit, args.seed + 1)
    if not train_rows or not eval_rows:
        raise ValueError("both train and evaluation probe sets must be non-empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    train_examples = [encode_row(tokenizer, row, args.trajectory, args.max_length)
                      for row in train_rows]
    for row in eval_rows:
        encode_row(tokenizer, row, args.trajectory, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    step_sample_ids: list[str] = []
    steps = args.steps

    for step in range(steps):
        example = train_examples[step % len(train_examples)]
        batch = collate([example], pad_token_id)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        step_sample_ids.append(example["sample_id"])
        print(f"step={step + 1}/{steps} sample={example['sample_id']} "
              f"tokens={int(batch['attention_mask'].sum())} loss={loss_value:.6f} "
              f"grad_norm={float(grad_norm):.6f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)
    manifest = {
        "what": "dependency-free small SFT probe",
        "not": "not a tuned result or a final training recipe",
        "base_model": str(args.model),
        "trajectory": args.trajectory,
        "device": args.device,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "method": "full_parameter_sft; not LoRA or QLoRA",
        "label_policy": (
            "assistant_body_and_im_end_only; context_and_headers_masked; "
            "failed_recovery_tool_calls_masked"
            if args.trajectory == "recovery"
            else "assistant_body_and_im_end_only; context_and_headers_masked"
        ),
        "train_count": len(train_rows),
        "seen_train_count": len(set(step_sample_ids)),
        "step_sample_ids": step_sample_ids,
        "eval_split": args.eval_split,
        "eval_count": len(eval_rows),
        "steps": steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "losses": losses,
        "source_samples_digest": digest_rows(rows),
        "train_sample_digest": digest_rows(train_rows),
        "eval_sample_digest": digest_rows(eval_rows),
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "eval_sample_ids": [row["sample_id"] for row in eval_rows],
    }
    (args.output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "eval_samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in eval_rows) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
