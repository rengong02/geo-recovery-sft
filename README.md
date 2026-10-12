# geo-recovery-sft

训练一个小模型**读完工具报错并改对参数**：地理工具调用的恢复能力，从数据构造到评测的一条管线。

被训的模型是 Qwen3-0.6B，但主要工作量在另外两件事上——数据构造的正确性，以及结论归因的干净程度。
让"模型学会了恢复"这句话能被复核，而不是只能被相信。

## 起点是一次错误的结论

第一版恢复数据里的"错误调用"是人为构造的：把 join key 写成 `f"{key}__missing"`。
它确实会失败，但有两个问题。

- **它自我标注。** 求解器只要学会"名字以 `__missing` 结尾就换一个"就能通过，
  学到的是字符串匹配，不是键的语义。
- **它不是推理期真正发生的失败。** 实测推理期的失败（192/192）是
  `ValueError: merge on str and float64`——把真实数值列当成了 join key，
  而第一版数据里一条这种样本都没有。

那一版 SFT 在 validation 上是 `0/48`，比不微调还差。整条数据管线据此重做。

## 做法

**三类真实错误，全部由工具执行产生。** `wrong_role`（用另一张表的键名，`KeyError`）、
`value_as_key`（用真实数值列，`ValueError: merge on str and float64`）、
`missing_argument`（省略必需参数，`TypeError`）。错误类别只写入 lineage 与 manifest，
不进入 prompt，有测试专门断言这一点。

**失败调用只作 context，不参与监督。** 监督的是失败之后的正确 merge 和合法 termination——
要教的是"看到错误后改对"，不是复现那一次错误。

**配对消融隔离变量。** 新旧两版数据的 sample id、world、dataset digest 与初始 system/user prompt
完全一致，只有错误类别改变，且由构建门禁强制：不满足就拒绝产出。
两版 Base 在 144 条上逐样本 outcome 完全一致，所以两次 SFT 的差异只能归因到数据。

**门禁在写盘前跑完。** gold、validator、fresh replay、split、schema、kind 分布、
failed-step kind 对齐、与基线的配对性，任一条不过则整包拒绝。

**真实数据子线。** 接 World Bank WDI（CC BY 4.0）与 Natural Earth（public domain）两张公开表。
真实实体是国家、天然共享，无法沿用合成线的"每个 case 一个 world"，
所以把两表共有的 canonical key 切成三个互不相交的实体池，每个 case 只能取自所属 split 的池。
快照随仓库携带，构建时逐个校验 SHA256，哈希不符即拒绝构建。

## 结果

| | Base | SFT |
|---|---|---|
| 合成配对 validation（48 条样本 × 2 seed） | 0/48 | 96/96 |
| 合成配对 held-out test（48 条 × 2 seed） | — | 90/96 |
| 真实数据 validation / held-out test | 0/48 | 16/16 ／ 16/16 |

单 seed 分别是 42/48 与 48/48，方差如实保留。
另外用一组对照说明剩余失败不是回合预算问题：同一个冻结 checkpoint、同一个 test split，
`max_turns` 从 6 提到 12，结果不变而重复调用显著增加。

完整表格、区间、分母与每个数字的原始报告哈希见 [RESULTS.md](RESULTS.md)。
**两条线的数字不得合并**；真实线只作窄任务 schema/join-key 学习的评估。

## 跑起来

```bash
python -m pytest tests -q          # 618 passed, 1 skipped
```

本仓库不依赖 GeoBenchX，缺了它测试照样全绿。

```bash
# 合成线的配对消融候选（门禁含与基线的配对性校验）
python scripts/build_recovery_candidate.py --out-dir <新目录>

# 真实数据线候选（快照随仓库携带，构建时校验 SHA256）
python scripts/build_real_candidate.py --out-dir <新目录>

# 干跑：校验候选、实体池、源哈希与工具档案；不加载模型、不占 GPU
python scripts/run_real.py --stage base --check-only \
    --bundle-dir evals/candidates/real-48 --run-root <专用目录>
```

`run_real.py` 的 bundle 与 run root **没有默认值**，两个都必须显式给：
合成线的 runner 默认指向它自己的 run root，第二次实验沿用就会把结果写进第一次的目录。

`--check-only` 不需要模型 checkpoint，所以干跑在任何机器上都能做。

## 仓库结构

```
strata/train/      样本管线：真实执行 → 校验记录、独立 gold、比较器、重放适配器
strata/geobench/   工具层与训练数据导出校验
scripts/           构建、打包、运行与门禁
tests/             测试
evals/candidates/  冻结的候选（manifest 与四份报告）
evals/runs/        跑出来的冻结报告，RESULTS.md 的数字出自这里
evals/sources/     真实数据线的两张公开表快照
```

仓库不含任何研发时间节点的叙事：目录与文件名只说明用途，冻结报告里也不记录运行时间；
数据快照由 SHA256 标识，而不是"什么时候下载的"。

## 局限

- 合成线只有 **96 条不同样本、两个 seed**；真实线只有 48 条、单 seed。够支撑配对消融要回答的问题，
  不足以声称稳定性能。`disjoint_root` 家族的泛化问题未解决：6/6 失败都在那里，加大回合预算无效。
- 真实线的正确轨迹是**直接正确**的，没有展示模型在推理期先犯错再恢复；
  它支持的是窄域 schema/键选择的学会，不是广义在线恢复能力。
- 0.6B、单一任务族，不外推到通用 GIS Agent。

## 来源与许可

- 运行时是 [CoreCoder](https://github.com/he-yufeng/CoreCoder)（MIT）的引擎，vendor 在此，
  因为模型正是跑在它上面调用工具的；`corecoder/` 里对 harness 的改动见
  [StrataAgent](https://github.com/rengong02/StrataAgent)。
- 真实数据来源：World Bank WDI（CC BY 4.0）、Natural Earth（public domain），
  URL、许可证与 SHA256 记录在 `strata/train/real_data.py` 的 `SOURCES`。
