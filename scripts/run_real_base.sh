#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${STRATA_REPO:-$(cd "$script_dir/.." && pwd)}"
package_root="${STRATA_PACKAGE_ROOT:-$(cd "$repo/.." && pwd)}"
python="${STRATA_PYTHON:-<lab-root>/envs/strata-model312/bin/python}"
tool_python="${STRATA_TOOL_PYTHON:-$python}"
model="${STRATA_MODEL:-<lab-root>/models/Qwen3-0.6B}"
bundle="${STRATA_BUNDLE_DIR:-$repo/evals/train/f2_real_20261011}"
run_root="${STRATA_RUN_ROOT:-<lab-root>/experiments/f2_real_20261011_seed7}"
budget="${BUDGET_SECONDS:-0}"

test -x "$python"
test -f "$repo/scripts/run_real.py"
test -f "$bundle/manifest.json"

check_only=false
for argument in "$@"; do
  if [ "$argument" = "--check-only" ]; then
    check_only=true
  fi
done
if [ "$check_only" != true ]; then
  test -d "$model"
fi

mkdir -p "$run_root"
cd "$repo"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export GEOBENCHX_ROOT="${GEOBENCHX_ROOT:-$package_root/GeoBenchX}"
export STRATA_REAL_SOURCES_DIR="${STRATA_REAL_SOURCES_DIR:-$bundle/../real_sources}"
export STRATA_TOOL_PYTHON="$tool_python"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$python" -u scripts/run_real.py \
  --stage base \
  --bundle-dir "$bundle" \
  --run-root "$run_root" \
  --model "$model" \
  --budget-seconds "$budget" \
  "$@"
