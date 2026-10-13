# Real-data evaluation reports

这里保存真实数据 candidate 的冻结报告，模型 checkpoint 和训练环境不进入仓库。

| 阶段 | 样本 | 正确 | 准确率 | 格式失败 | 报告错误 |
|---|---:|---:|---:|---:|---:|
| Base | 48 | 0 | 0% | 18 | 29 |
| Recovery SFT validation | 16 | 16 | 100% | 0 | 0 |
| Recovery SFT held-out test | 16 | 16 | 100% | 0 | 0 |

## 配置

- 模型：`Qwen3-0.6B`
- 训练：full-parameter BF16，48 steps，`lr=5e-6`，seed 7
- 训练集：16 条真实数据样本，每条暴露三次
- validation/test：互不相交的实体池，各 16 条
- 仓库 candidate digest：`sha256:e6794199c3684a1fbd1f77fcd8f7fcf9e302c9eec11dab64d37d0d4355159509`
- 运行输入 digest：`sha256:79e62b311240654dd7abcab577b1b27a9dcad61fcd59e2d025b023f7d253d185`，对应相同任务内容及来源获取时间元数据

validation 和 test 中的正确行都是直接正确的工具调用轨迹，没有测量推理期先失败再恢复。结果只支持窄 schema/key-selection 任务上的学习结论。
