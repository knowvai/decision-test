# 脚本索引

## 主测试

| 用途 | 入口 | 备注 |
| --- | --- | --- |
| 分类数据准备和执行 | [`public_decision_benchmark.py`](../evaluation/runner/scripts/public_decision_benchmark.py) | 主执行子命令是 `local-classification`；`prepare` 保留历史 v1/v3 构造逻辑 |
| 答案证据数据构造与执行 | [`decision_evidence_benchmark.py`](../evaluation/runner/scripts/decision_evidence_benchmark.py) | `prepare` 从固定原始来源构造数据；`run` 是 choice/noul × direct/Top10 的冻结协议入口 |
| 传输、公共请求及统计复用 | [`decision_passage_benchmark.py`](../evaluation/runner/scripts/decision_passage_benchmark.py)、[`decision_capacity_preflight.py`](../evaluation/runner/scripts/decision_capacity_preflight.py) | 新证据测试复用其传输与统计，不应另写一个算法近似的 runner |
| 原生 choice/noul 解码 | [`deployment_probe.py`](../deploy/docker/decision-eval/deployment_probe.py) | 模型家族适配、完整输入检查；probe 会实际调用模型，不能用于离线复核 |
| 独立答案证据审计 | [`2026-10-18-evidence-audit.py`](research/2026-10-18-evidence-audit.py) | 历史执行必须显式 `--revision r3`；修订审计使用 `--allow-online-reranker-variation`。冻结未修订版在 `source-r3` 包内 |
| 分类独立审计 | 分类归档中的 `audit-20261015/`、`qwen38-supplement-20261016/`、`qwen38-thinking-20261016/` | 每轮的 `classification-audit.py` 与结果一起保存 |
| 片段 HTML | [`2026-10-19-answer-evidence-summary.py`](research/2026-10-19-answer-evidence-summary.py) | `--view direct` 是主报告，`--view reranker` 是附录 |

`decision_rag_worker.py`、`replay_decision_rag.py` 和 `src/evaluation/retrieval_eval.py` 保留为历史导入、AST 提取和评分依赖。
主测试没有调用产品服务。旧 `run` 子命令中依赖原产品的路径不属于本独立仓库支持的执行入口。

## 数据集与准备

- [`tools/download_data.py`](../tools/download_data.py)：下载选定的公开来源并核对 SHA256。
- [`tools/reconstruct_inputs.py`](../tools/reconstruct_inputs.py)：按原记录顺序、结构和文本哈希重建历史输入，核对完整文件哈希。
- [`2026-10-18-evidence-fetch.py`](research/2026-10-18-evidence-fetch.py)：历史 T2Ranking 全量语料分块下载工具；保留原实现，包含原机器 CA 文件路径，跨系统优先用新的下载入口。
- [`2026-10-18-evidence-native-coverage.py`](research/2026-10-18-evidence-native-coverage.py)、[`length-preflight.py`](research/2026-10-18-evidence-length-preflight.py)：原生编码和完整输入覆盖检查，不以截断换取覆盖。
- [`2026-10-18-evidence-prepare.py`](research/2026-10-18-evidence-prepare.py)：旧主机的冻结与保全流程，会读取 Docker 和既有运行目录；不是通用数据下载器。
- [`2026-10-19-d3-tuning.py`](research/2026-10-19-d3-tuning.py)：独立开发/留出划分及调优统计。

## 补充测试

| 用途 | 脚本 |
| --- | --- |
| 旧相关性 noul | [`decision_noul_validation.py`](../evaluation/runner/scripts/decision_noul_validation.py) 及 `2026-10-17-passage-noul-*` |
| D3 质量调优 | `docs/research/2026-10-19-d3-tuning-{run,audit,report}.py` |
| D3 并发参数 | `docs/research/2026-10-19-d3-runtime-{run,audit,report,diagnostics}.py` |
| Clef 并发参数 | `docs/research/2026-10-19-clef-runtime-{run,audit,report}.py` |
| 低分离线分析 | `docs/research/2026-10-19-d3-evidence-diagnosis.py`、`2026-10-19-d3-review-verification.py` |
| 官方参考实现尝试 | `docs/research/2026-10-19-evidence-reference-{host,check}.py`，另有各次实际源码包 |

## 本次归档工具

| 工具 | 是否访问网络或调用模型 |
| --- | --- |
| [`verify.py`](../tools/verify.py) | 只读离线校验与重算，不调用模型 |
| [`materialize.py`](../tools/materialize.py) | 仅解包到新目录，不执行归档内代码 |
| [`test_offline.py`](../tools/test_offline.py) | 禁止网络连接，执行模拟接口的回归测试 |
| [`export_bundle.py`](../tools/export_bundle.py) | 从已收集的本地材料生成公开副本，记录原始/发布哈希，不连接服务器 |
| [`build_model_index.py`](../tools/build_model_index.py) | 从已有部署、版本锁和运行时注册表生成索引，不加载权重 |
| [`normalize_publication.py`](../tools/normalize_publication.py) | 按外部规则整理文件及压缩包内命名，保留来源哈希和固定种子计算字节，不调用模型 |

历史审计包含绝对路径、前序保全和当时镜像检查，不能简单改一个根目录就声称完整复现。公开归档的可移植核对入口是 `tools/verify.py`。
