# Results

本页记录冻结 candidate 的训练和评测结果。每个数字均来自 `evals/runs/` 下的 JSON 报告，分母按独立样本和重复运行分别说明。

## 1. 合成 paired ablation

candidate 包含 train、validation、test 各 48 条样本，三类错误各 48 条。ablation 与 control 共享 sample id、world、数据摘要和初始提示，只改变 `recovery_error_kind`，从而隔离错误类型这一数据因素。

| 运行 | 样本 / 运行次数 | 正确 | 准确率 |
|---|---:|---:|---:|
| Base | 144 | 0 | 0% |
| SFT validation，seed 7 | 48 | 48 | 100% |
| SFT held-out test，seed 7 | 48 | 42 | 87.5% |
| SFT validation，seed 17 | 48 | 48 | 100% |
| SFT held-out test，seed 17 | 48 | 48 | 100% |

两个 seed 在同一组 48 个 test 样本上运行，合计 90/96 次运行正确。这个合计值用于描述 seed 结果，不能当作 96 个独立 test 任务。validation 的合计为 96/96 次运行正确。

## 2. 真实数据线

真实 candidate 包含 48 条样本，train、validation、test 各 16 条。World Bank WDI 与 Natural Earth 的共有国家键被分到三个互不相交的实体池；每条样本记录来源、快照 SHA256、许可证、实体池和实体标识。

| 运行 | 样本 | 正确 | 格式失败 | 报告错误 |
|---|---:|---:|---:|---:|
| Base | 48 | 0 | 18 | 29 |
| Recovery SFT validation | 16 | 16 | 0 | 0 |
| Recovery SFT held-out test | 16 | 16 | 0 | 0 |

真实 validation 和 test 的正确轨迹都是直接正确调用，不表示模型在推理期先产生错误后完成恢复。

## 3. 训练配置

- 模型：Qwen3-0.6B。
- 方法：full-parameter BF16 SFT，48 steps，learning rate `5e-6`，合成 seed 7/17、真实 seed 7，最大序列长度 4096。
- 训练目标：屏蔽系统提示、用户问题、工具上下文和失败工具调用，只监督正确恢复与合法终止。
- 解码：greedy，`max_new_tokens=1024`，`max_turns=6`。
- 评分：`v2.1-status-schema`，不依赖 judge model。
- 报告目录：`evals/runs/recovery-ablation/`、`evals/runs/recovery-ablation-seed17/`、`evals/runs/real-seed7/`。

## 4. 失败分析

seed 7 test 的 6 个失败全部来自 `disjoint_root` family。模型在错误观测后重复选择数值列作为连接键，导致类型不匹配，并重复同一工具调用。把同一冻结 checkpoint 的最大回合数从 6 调到 12 后，结果仍为 42/48，说明增加回合不能解决该泛化问题。

## 5. 限制

- 合成线只有 96 个不同样本，两个 seed 的合计是重复运行数，不是独立任务数。
- 真实线只覆盖窄 schema 和 join-key 任务，不能外推到通用 GIS Agent 或广义在线恢复能力。
- 结果依赖训练数据分布、工具 schema、prompt、模型和解码设置。
配对 control 使用单一缺失键错误轨迹，其 seed 7 validation 为 0/48；三类错误轨迹的 seed 7 validation 为 48/48。两者共用 144 个 world 和初始提示，Base 的逐样本结果一致。该对照限定于当前配方与任务分布。

## 原始报告索引

下表 SHA256 标识仓库内的报告文件，链接可查看逐样本结果与运行配置。

| 报告 | SHA256 |
|---|---|
| [真实 Base](evals/runs/real-seed7/base/report.json) | `98f86682ee932bd485de456f10b0703fdb9a976be73319fc324d54834b071d38` |
| [真实 SFT test](evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/test_report.json) | `858c8afd6be2b0f03721abcfa18f8495a7a34e00e327f456e63f1970ff6b145b` |
| [真实 SFT validation](evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/validation_report.json) | `d1df5378b9d113145eda982438f5400614aafc949f63e1d8c38882a94d358dd8` |
| [合成 Base](evals/runs/recovery-ablation/base/report.json) | `e57fd1c7f15d9af0d2d683d94cb414ef61c44249dc2a98bce7fdeeee478f12b0` |
| [合成 SFT seed 7 test](evals/runs/recovery-ablation/sft_seed7/test_report.json) | `afd6cf6bbd35d669796ac8edf1248a0649b217a4d0a96bb4a15d9166eb9cd94d` |
| [合成 SFT seed 7 validation](evals/runs/recovery-ablation/sft_seed7/validation_report.json) | `a73c036dcfd01d20afaaa332f64fe3ae8758ac9eca23cb0a415839e6f4a0ff07` |
| [合成 SFT seed 17 test](evals/runs/recovery-ablation-seed17/sft_seed17/test_report.json) | `bd8fd474f2bb190ba3e5bca29312f872729cdf1f639ae9101504580e896e5855` |
| [合成 SFT seed 17 validation](evals/runs/recovery-ablation-seed17/sft_seed17/validation_report.json) | `f335aefc08227b5398f31c2b445cb1f13a12186e8de163c46b9e18413af08e5c` |
| [12 回合诊断](evals/runs/recovery-ablation/sft_seed7/test_report_maxturns12.json) | `e5179c116412577ae0393b27c2ce2371b664af7070c12d910ca2fb86f027c318` |
| [配对 control Base](evals/runs/recovery-baseline/base/report.json) | `ed72e4e044793af8a8a44bba060d8a1c0a1de5e4fb6f11f90112a3d30c969fb3` |
| [配对 control SFT validation](evals/runs/recovery-baseline/sft_seed7/validation_report.json) | `53801e5b97b419a71c0e5b97286d580698e42c83329637673a847c63da4b5679` |

真实数据的冻结报告使用输入摘要 `sha256:79e62b311240654dd7abcab577b1b27a9dcad61fcd59e2d025b023f7d253d185`；仓库候选的 manifest 使用 `sha256:e6794199c3684a1fbd1f77fcd8f7fcf9e302c9eec11dab64d37d0d4355159509`。两者的任务、实体、工具观测和 gold 相同，候选摘要差异来自来源获取时间字段的省略；报告保留运行时输入身份。

重建候选会记录当前工具源码摘要，源码变化时 `samples_digest` 会相应变化。相同样本集合还需要核对 sample id、split 和数据摘要；归档文件的字节哈希不用于识别跨环境重建的数据。
