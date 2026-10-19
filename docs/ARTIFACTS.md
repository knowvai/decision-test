# 测试材料目录

所有压缩包位于 [`artifacts/`](../artifacts/)。目录名保留原实验标识，`run-001--模型名.tar.gz` 是该次运行中一个模型的记录，不是另一轮测试。
[`publication.json`](../manifests/publication.json) 列出每个包的校验值、内部文件、原始 SHA256 和发布 SHA256。

所有历史文档、源码和压缩包内路径均使用独立评测名称。发布哈希对应当前副本，原始哈希仅用于来源追溯；名称修改不影响逐条预测、耗时或标签。

## 正式结果

| 实验 | 原始材料 | 结果与执行记录 |
| --- | --- | --- |
| 中文分类 | [classification-20261015](../artifacts/classification-20261015/) | [最终结果](research/2026-10-16-decision-classification-final-results.md)、[初始执行](research/2026-10-15-decision-classification-execution.md)、[Qwen3.8 非思考补充](research/2026-10-16-qwen38-classification-supplement.md)、[思考对照](research/2026-10-16-qwen38-thinking-comparison.md) |
| 答案证据筛选 | [evidence-20261018](../artifacts/evidence-20261018/) | [HTML](research/2026-10-19-answer-evidence-summary.html)、[全部条件](research/2026-10-19-answer-evidence-results.md)、[执行与审计修订](research/2026-10-18-answer-evidence-execution.md) |
| reranker 独立部署 | [passage-reranker-20261017](../artifacts/passage-reranker-20261017/) | [部署说明与资源](research/2026-10-17-passage-reranker-deployment.md)、[Top10 结果附录](research/2026-10-19-answer-evidence-reranker-appendix.html) |

分类正式成绩来自 `run-001/002/003/006/007/008`，每个模型配置只取其对应的 2,974 条正式记录。
`run-004/005` 是失败的冷启动修复尝试，没有正式响应；不能删除，也不能算作额外测试样本。
`source`、`source-r2` 至 `source-r8` 保留各次执行时的源码，不能用最终版本覆盖早期版本。

答案证据筛选实际执行版本是 `source-r3`，正式运行是 `run-001`。`source-r1/r2` 只做过准备。
`preparation-r3` 包含冻结文件、完整输入覆盖、历史保全、测试验收和启动记录。
`offline-audit-20261019-online-variation-v1` 是显式修订跨调用 reranker 一致性要求后的独立审计，原严格审计退出信息在执行记录及终态验证 JSON 中保留。

## 补充实验

| 内容 | 原始材料 | 结果 |
| --- | --- | --- |
| HTTP 容量预检 | [capacity-20261016](../artifacts/capacity-20261016/) | [容量预检](research/2026-10-16-decision-capacity-preflight.md) |
| 旧相关性筛选及原生 choice 重算 | [passage-20261017](../artifacts/passage-20261017/) | [阈值版](research/2026-10-17-passage-filtering-results.md)、[choice 重算](research/2026-10-17-passage-choice-results.md) |
| 旧相关性 noul 对照 | [passage-noul-20261017](../artifacts/passage-noul-20261017/) | [结果](research/2026-10-17-passage-noul-results.md) |
| D3 提示与阈值调优 | [d3-tuning-20261019](../artifacts/d3-tuning-20261019/) | [结果](research/2026-10-19-d3-tuning-results.md) |
| D3 运行时并发参数 | [d3-runtime-20261019](../artifacts/d3-runtime-20261019/) | [结果](research/2026-10-19-d3-runtime-results.md) |
| Clef 运行时并发参数 | [clef-runtime-20261019](../artifacts/clef-runtime-20261019/) | [结果](research/2026-10-19-clef-runtime-results.md) |
| Qwen 并发诊断 | [qwen3-diagnostic-20261017](../artifacts/qwen3-diagnostic-20261017/) | [结果](research/2026-10-17-qwen3-capacity-diagnosis.md) |
| Vela 并发诊断 | [vela-diagnostic-20261017](../artifacts/vela-diagnostic-20261017/) | [结果](research/2026-10-17-vela-capacity-diagnosis.md) |
| D3 官方参考实现诊断尝试 | [初版](../artifacts/evidence-reference-20261019/)、[r2](../artifacts/evidence-reference-20261019-r2/)、[r3](../artifacts/evidence-reference-20261019-r3/) | 未完成数值对照，不能当作参考实现等价性证明；保留协议、失败状态及尝试记录 |

以上轮次的问题、长度、标签或参数可能不同。重复调用不增加独立样本量，预热和工程阶段不进入主场景正式质量表。

## 部署与来源

- [`acceptance-20261015`](../artifacts/acceptance-20261015/)：模型接入、鉴权、长输入和部署验收。
- [`preflight-20261015`](../artifacts/preflight-20261015/)：实际使用的 Jeva 和 vllm-srun 源码、依赖定义与构建上下文。
- [`framework-engine-install-v1`](../artifacts/framework-engine-install-v1/)、[`framework-sglang-fetch-v1`](../artifacts/framework-sglang-fetch-v1/)、[`framework-qwen-fetch-v1`](../artifacts/framework-qwen-fetch-v1/)：镜像引用、摘要及派生构建记录，不含镜像二进制。
- [`model-provenance`](../artifacts/model-provenance/)：现存模型元数据、发布校验表和许可文本，不含权重。

## 文件含义

| 文件 | 内容 |
| --- | --- |
| `responses.private.jsonl` | 分类逐条预测、输入/请求哈希、原生输出和耗时。`.private` 是历史文件名，不表示仓库包含凭据 |
| `requests.jsonl`、`intents.jsonl` | 片段逐次请求结果、评分或概率、时间、哈希及调用意图记录 |
| `chains.jsonl` | 一个问题的全部筛选结束后的保留片段 ID、成功状态和真实总耗时 |
| `*.labels.private.jsonl` | 原标签及原始样本 ID，不包含模型生成的伪标签 |
| `deployment.json`、`container.json` | 当次引擎、镜像、参数、资源、容器身份及非密钥环境设置 |
| `protocol*.json`、`frozen.json` | 当次协议、输入/源码哈希及计划范围 |
| `audit*.json`、`summary.json` | 当次检查或汇总；必须结合对应版本和执行记录阅读 |
| `*.recipe.json.gz` | 去掉数据集原文后的结构与文本哈希，用于从官方数据重建逐字节一致的输入文件 |

运行日志可能含环境信息，因此不公开原始 `.log`；其文件哈希及未发布原因列入清单。测试失败的状态、审计结论和解释仍在公开材料中。
