# geo-recovery-sft

geo-recovery-sft 是面向地理工具调用的轻量模型训练项目，提供恢复轨迹构建、数据校验、监督微调和多轮工具评测。模型读取表结构与工具错误观测，选择正确的连接键，并输出结构化结果。

## 方法

训练数据覆盖三类由工具实际执行产生的错误：

| 错误类型 | 错误调用 | 工具观测 |
|---|---|---|
| `wrong_role` | 使用另一张表的字段作为当前表的键 | `KeyError` |
| `value_as_key` | 将数值列作为连接键 | 列类型不匹配的 `ValueError` |
| `missing_argument` | 省略必需参数 | `TypeError` |

失败调用与工具观测保留为上下文；失败调用不参与监督，训练标签只覆盖正确的 assistant 调用和终止结果。样本写盘前检查真实错误、正确结果、独立重放、数据来源和 split。

```mermaid
flowchart LR
    Data["合成表 / 公开快照"] --> Trace["真实工具执行与恢复轨迹"]
    Trace --> Gate["来源、结果、重放与 split 校验"]
    Gate --> Train["Qwen3-0.6B SFT"]
    Train --> Eval["多轮工具调用评测"]
    Eval --> Report["逐样本结果与报告"]
```

合成数据按 schema 家族划分 train、validation 和 test；真实数据使用 World Bank WDI 和 Natural Earth 快照，将共有国家键分成三个互不相交的实体池。配对消融保持 world、样本标识、数据和初始提示一致，只改变训练轨迹中的错误类型。

## 结果

| 数据线 | 独立样本 | Base 正确数 | SFT 正确数 |
|---|---:|---:|---:|
| 合成 validation | 48 | 0/48 | seed 7：48/48；seed 17：48/48 |
| 合成 held-out test | 48 | 0/48 | seed 7：42/48；seed 17：48/48 |
| 真实 validation | 16 | 0/16 | seed 7：16/16 |
| 真实 held-out test | 16 | 0/16 | seed 7：16/16 |

合成 test 为同一组 48 个样本的两次运行，合计 90/96 次正确（93.75%）。真实数据上的正确轨迹为直接正确调用，支持窄域 schema 和连接键学习的结论。详细配置、消融、报告链接与限制见 [RESULTS.md](RESULTS.md)。

## 安装

CPU 数据构建与校验无需 GPU、模型权重或外部 GeoBenchX checkout：

```bash
git clone https://github.com/rengong02/geo-recovery-sft.git
cd geo-recovery-sft
uv sync --extra dev
source .venv/bin/activate
uv pip install --python "$(command -v python)" -e '.[dev]' numpy pandas geopandas
```

使用 CUDA 训练时，在对应环境中安装训练依赖：

```bash
uv pip install --python "$(command -v python)" -e '.[dev]' -r requirements-train.txt
```

GPU 配置必须支持 BF16，模型目录需包含 Qwen3-0.6B 权重与 tokenizer。CUDA 环境中的 PyTorch 安装按对应 CUDA 版本配置。模型不随仓库分发，runner 使用本地文件运行。

## 数据构建与检查

重建合成配对候选和 48 条真实数据候选；输出目录需要为空或尚不存在：

```bash
python scripts/build_recovery_candidate.py --out-dir outputs/candidates/recovery
python scripts/build_real_candidate.py --count-per-pool 16 \
  --out-dir outputs/candidates/real
```

对仓库内的冻结真实数据候选做 CPU 预检：

```bash
python scripts/run_real.py --stage base --check-only \
  --bundle-dir evals/candidates/real-48 --run-root outputs/real-check
```

预检检查候选摘要、实体池与工具执行；`--check-only` 不加载模型。真实数据快照的 URL、许可和 SHA256 见 [real_data.py](strata/train/real_data.py)。

## 训练与评测

在安装好的 CUDA 环境中设置模型与输出目录，依次运行 Base 与 SFT；两阶段使用同一个输出目录：

```bash
export STRATA_PYTHON="$(command -v python)"
export STRATA_MODEL="$PWD/models/Qwen3-0.6B"
export STRATA_RUN_ROOT="$PWD/outputs/real"
bash scripts/run_real_base.sh
bash scripts/run_real_sft.sh
```

真实数据线使用 16 个 train 样本训练 48 步，每条暴露三次；学习率为 `5e-6`，seed 为 7。SFT 阶段生成 checkpoint 并评测 validation。先按 split 生成 test 输入文件，再单独评测：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from scripts.run_sft import choose_rows

root = Path(os.environ["STRATA_RUN_ROOT"])
candidate = Path("evals/candidates/real-48")
samples = [json.loads(row) for row in (candidate / "samples.jsonl").read_text().splitlines()]
split = json.loads((candidate / "split_report.json").read_text())
selected = choose_rows(samples, split, "test", 0, 7)
(root / "test_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
PY
```

```bash
python scripts/run_baseline.py \
  --samples "$STRATA_RUN_ROOT/test_samples.jsonl" \
  --model "$STRATA_RUN_ROOT/sft_recovery16_epoch1_lr5e-6/checkpoint" \
  --evaluation-kind sft-probe --device cuda --attn-implementation sdpa \
  --max-turns 6 --max-new-tokens 1024 \
  --out "$STRATA_RUN_ROOT/sft_recovery16_epoch1_lr5e-6/test_report.json"
```

合成恢复线使用 `scripts/run_recovery.py` 的 `--stage base` 与 `--stage sft`，候选路径为 `evals/candidates/recovery-ablation`。训练 checkpoint、recipe 和报告应放在新的专用 run root。

## 离线任务包

无外网服务器可使用携带源码、候选和真实快照的任务包：

```bash
python scripts/package_real.py --out outputs/real-offline
```

解压后在 `project/` 下运行对应脚本。包中提供 Natural Earth 的属性表 fallback；模型和 Python 环境需单独准备。

## 测试与目录

```bash
python -m pytest tests -q
```

```text
corecoder/        模型交互和工具执行运行时
strata/train/     样本构建、标签、评分和重放
scripts/          SFT、评测、校验和离线打包
evals/candidates/ 冻结候选和 manifest
evals/runs/       冻结结果和训练配置
evals/sources/    公开数据快照
tests/            数据管线与运行时测试
```

## 来源与许可

本项目基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 的 MIT 运行时构建，领域训练代码位于 `strata/` 和 `scripts/`；与 [StrataAgent](https://github.com/rengong02/StrataAgent) 共用工具协议，各仓库维护独立代码副本。

代码采用 [MIT](LICENSE)。World Bank WDI 快照按 CC BY 4.0 提供，Natural Earth 为 public domain，来源 attribution 和文件 SHA256 随数据配置保留。
