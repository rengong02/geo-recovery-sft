# Results — F2 recovery, paired error-kind ablation (r2)

Every number below was read out of the frozen report JSON in `evals/runs/`, not
from prose. The reports were pulled from the server and **verified against the server's own
`sha256sum`** before being used here (§5). Pulls are read-only; nothing was written on the server.

## 0. What the experiment asks

r1 trained on a recovery trace whose failure was produced by an invented, self-labelling key
(`geokey="region_code__missing"`). Its SFT scored `0/48` on validation — **worse than base**.
The measured inference-time failure is different: `ValueError: merge on str and float64 columns`,
from using a real numeric column as a join key.

r2 asks one question: **if the failure class in the training data is swapped for the real one —
and nothing else changes — does recovery SFT work?**

"Nothing else" is enforced, not asserted. The builder refuses to emit r2 unless, against r1:
sample ids identical, worlds identical, dataset digests identical, initial system/user prompts
identical, and `recovery_error_kind` changed in all 144 samples. Measured at build time:

```
shared_sample_ids 144 | identical_outside_changed_factor true | initial_prompts_identical true
changed_error_kinds 144 | divergent_samples [] | prompt_divergent_samples [] | problems []
```

r2 is therefore **not** an independent new-world candidate. It is r1 with one factor swapped,
and the manifest says so (`world_policy=same_world_paired_ablation`).

## 1. Result

| experiment | split | model | seed | n | correct | acc | answered | fmt fail | no-term | reported err |
|---|---|---|---|---|---|---|---|---|---|---|
| `r2-base-144` | all 144 | Qwen3-0.6B (base) | — | 144 | **0** | 0.000 | 2 | 77 | 51 | 65 |
| `r2-sft-seed7-val` | validation | r2 SFT ckpt | 7 | 48 | **48** | 1.000 | 48 | 0 | 0 | 0 |
| `r2-sft-seed7-test` | test | r2 SFT ckpt | 7 | 48 | **42** | 0.875 | 42 | 6 | 6 | 0 |
| `r2-sft-seed7-test-max12` | test | r2 SFT ckpt | 7 | 48 | **42** | 0.875 | 42 | 6 | 6 | 0 |
| `r2-sft-seed17-val` | validation | seed17 SFT ckpt | 17 | 48 | **48** | 1.000 | 48 | 0 | 0 | 0 |
| `r2-sft-seed17-test` | test | seed17 SFT ckpt | 17 | 48 | **48** | 1.000 | 48 | 0 | 0 | 0 |

Paired controls, read from their own reports (§5) rather than quoted:

| control | n | correct | answered | fmt fail | no-term | reported err |
|---|---|---|---|---|---|---|
| `r1-base-144` | 144 | 0 | 2 | 77 | 51 | 65 |
| `r1-sft-seed7-val` | 48 | 0 | 0 | 48 | 48 | 0 |
| `r2-base-144` | 144 | 0 | 2 | 77 | 51 | 65 |

The two Base rows are not merely equal in aggregate — re-checked sample by sample: **shared ids 144,
outcome differences 0**. r1's SFT validation answered nothing at all (48/48 `no_termination`), and
r1/r2 SFT validation ran the identical 48 samples. So the difference between the two SFT runs cannot
be attributed to their starting points or to a different validation set.

Trajectory telemetry (tool observations, continuation after failure, repeated calls), measured
from the same reports:

| experiment | tool calls | failed observations | continued after failure | duplicate calls |
|---|---|---|---|---|
| `r2-base-144` | 163 | 144 | 31 / 144 | 41 |
| `r2-sft-seed7-val` | 192 | 48 | 48 / 48 | 0 |
| `r2-sft-seed7-test` | 202 | 64 | 46 / 48 | 18 |
| `r2-sft-seed7-test-max12` | 238 | 100 | 46 / 48 | 54 |
| `r2-sft-seed17-val` | 192 | 48 | 48 / 48 | 0 |
| `r2-sft-seed17-test` | 158 | 14 | 14 / 48 | 0 |

Base success is not near-miss: `0/144` has a 95% Wilson upper bound of **2.60%**; of the 144,
only 2 even terminated with an answer, and both were wrong key pairs.

## 2. The denominators — read this before quoting any number

**"192" counts runs, not tasks. There are 96 distinct samples.**

| | distinct samples | runs |
|---|---|---|
| validation | 48 | 48 × 2 seeds = 96 |
| test | 48 | 48 × 2 seeds = 96 |
| **total** | **96** | **192** |

Checked directly against the split report: `train/validation/test = 48/48/48`, the validation and
test reports cover exactly their named split, and `validation ∩ test = ∅`.

So the honest sentences are:

- validation: **48 samples, run twice — 96/96 correct**, 95% Wilson CI **[96.15%, 100%]**
- held-out test: **48 samples, run twice — 90/96 correct**, 95% Wilson CI **[87.03%, 97.10%]**
- seed spread is real: seed7 test `42/48 = 87.5%` [75.30%, 94.14%], seed17 test `48/48` [92.59%, 100%]

and the dishonest ones are "192 tasks", "93.75% accuracy", or any figure quoted without its
denominator, seed and synthetic-data nature.

### 2.1 A digest caveat found while building this table

`samples_digest` is **order-sensitive** — it hashes the sample records in list order. The two
seeds' validation runs used the same 48 samples in a *different order*, so their digests differ
(`b3934a96…` vs `20961785…`) even though the id sets are identical. The test runs happened to
share an order, so their digests match (`7808f8e1…`).

Consequence: equal digests imply equal set *and* order; equal id sets do **not** imply equal
digests. Compare id sets when the question is "same samples", digests when it is "same file".

## 3. The failure that remains, and the diagnosis that it is not a budget problem

All 6 test failures under seed7 are `no_termination` and **all 6 are in `disjoint_root__full`**
(that family: 18/24 correct, 95% Wilson [55.10%, 88.00%]). The model loads both frames, picks
`statkey=population_2021` — a real numeric column — with `geokey=area_code`, gets
`ValueError: merge on str and float64 columns`, then repeats the identical call until the turn
budget runs out.

`test_report_maxturns12.json` raises `max_turns` 6 → 12 on the same frozen checkpoint and the same
test split, and the result is **byte-for-byte the same outcome**: `42/48`, same 6 failures, same
`no_termination`. Calls go 202 → 238, failed observations 64 → 100, duplicates 18 → 54 — more
attempts, zero new successes. Deterministic repetition, not a budget shortfall. No further
turn-budget experiments were run.

Reproducing this pair is the strongest single piece of evidence in the project: the same model,
same checkpoint, same samples, one changed parameter, and the metric does not move while the
effort doubles.

## 4. Recipe and provenance

| field | value |
|---|---|
| candidate | `evals/candidates/recovery-ablation`, revision r2 |
| `samples_digest` | `sha256:67a295a1bfb6f946d912ffb40fc6067f4ad4e0147a985b21c5dc7a0fa00cf0a6` |
| split | train 48 / validation 48 / test 48, explicit, by shape |
| generator | `strata.train.shape_ood` / `synth/f2-shape-ood-recovery-v1` |
| trajectory | `recovery` |
| label policy | `failed_tool_call_context_only; successful_recovery_and_termination_supervised` |
| error kinds | `missing_argument` 48 / `value_as_key` 48 / `wrong_role` 48; `16/16/16` in every split |
| training | full-parameter BF16 (not LoRA/GRPO), 48 steps, `lr=5e-6`, 48 train rows, `max_length=4096` |
| base model | `<lab-root>/models/Qwen3-0.6B` |
| decoding | greedy (`do_sample=false`), `max_new_tokens=1024`, `max_turns=6` (12 in the diagnostic) |
| scorer | `v2.1-status-schema` — deterministic, no judge model |
| env | python 3.12.8, torch 2.5.1+cu118, transformers 4.51.3, pandas 3.0.6, numpy 1.26.4 |
| archive | `sha256:fb7a3079eff25f8e469e46316d33db0570ff1951314dcfadd2e5b588c3811720` |

The archive hash is a **frozen-file integrity check, not a version id** — archives embed
wall-clock `steps[].seconds`, so re-running the builder does not reproduce it. Compare
`samples_digest` across machines. (`samples_digest` strips that field by design.)

## 5. Raw reports

| file | sha256 (of the copy here) |
|---|---|
| `evals/runs/a800-baseline/qwen3-0.6b_a800_baseline.json` | `a909f5612fe04dda…` |
| `evals/runs/a800-baseline/qwen3-0.6b_a800_probe.json` | `39f3eb6f41a6c907…` |
| `evals/runs/a800-baseline-report.json` | `60c828b3396a08c1…` |
| `evals/runs/real-seed7/base/report.json` | `98f86682ee932bd4…` |
| `evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/checkpoint/probe_manifest.json` | `d1caaf80b2b39193…` |
| `evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/recipe.json` | `a682e1181d8c777e…` |
| `evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/test_report.json` | `858c8afd6be2b0f0…` |
| `evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/test_samples.jsonl` | `a0590495282b05d1…` |
| `evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/validation_report.json` | `d1df5378b9d11314…` |
| `evals/runs/real-seed7/split_report_sft.json` | `20ac763699ad7a50…` |
| `evals/runs/recovery-ablation/base/report.json` | `e57fd1c7f15d9af0…` |
| `evals/runs/recovery-ablation/sft_seed7/recipe.json` | `e8a10b20c515b748…` |
| `evals/runs/recovery-ablation/sft_seed7/test_report.json` | `afd6cf6bbd35d669…` |
| `evals/runs/recovery-ablation/sft_seed7/test_report_maxturns12.json` | `e5179c116412577a…` |
| `evals/runs/recovery-ablation/sft_seed7/validation_report.json` | `a73c036dcfd01d20…` |
| `evals/runs/recovery-ablation-seed17/sft_seed17/test_report.json` | `bd8fd474f2bb190b…` |
| `evals/runs/recovery-ablation-seed17/sft_seed17/validation_report.json` | `f335aefc08227b53…` |
| `evals/runs/recovery-baseline/base/report.json` | `ed72e4e044793af8…` |
| `evals/runs/recovery-baseline/sft_seed7/recipe.json` | `e092c3415c7fa6c1…` |
| `evals/runs/recovery-baseline/sft_seed7/validation_report.json` | `53801e5b97b419a7…` |

These are the only backing for every figure in this file; nothing here is quoted
from a summary or from memory.

Each was re-hashed locally against the server's `sha256sum` **before** any number was copied
out of it, and every number was read from those bytes. Two edits were applied to the copies
afterwards: the lab machine's absolute path root is replaced by `<lab-root>`
(`scripts/redact_lab_paths.py`, applied to the whole tree), and the fields that recorded when a
run happened were dropped so the repository carries no project timeline. The hashes above are of
the copies as they are here, which is why they are labelled as the hash of the copy rather than
as a server checksum. The `recipe.json` entries list no absolute path and are unchanged.

## 6. Limits

- **Synthetic data.** All 96 samples are generated tables. This does not yet satisfy the "real data"
  requirement. A separate **18-trace CPU smoke candidate** over published tables now exists
  (`evals/candidates/real-smoke/`) but it is a mechanism proof with no GPU run behind it, and
  its numbers must never be merged with the ones here.
- **96 distinct samples**, two seeds. Enough for the paired comparison this experiment was designed
  for; not enough to claim a stable rate. The seed spread (87.5% vs 100%) is reported, not smoothed.
- **`disjoint_root` generalisation is unsolved** — 6/6 failures, and more turns do not help.
- **0.6B parameters, one task family.** Nothing here generalises to GIS agents in general, and no
  claim in this file should be quoted as if it did.
