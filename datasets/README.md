# 数据集与重建

## 来源

| 数据 | 固定版本 | 用途 |
| --- | --- | --- |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) v1.1，zh-CN | Parquet revision `ed58ac423a2f4121720918bf5301577edce4ffd3`；原 loader revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740` | 主分类使用全部 2,974 条 test；后续调优另用 train/validation |
| [THUIR/T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking) | `2a369a430a70979223f1b9a41b1919774d46b432` | 新答案证据任务；使用原始问题、全文和 L0–L3 标签 |
| [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking) | `76631901a18387f85eaa53e5450019b87ad58ef9` | 旧相关性任务；新证据任务只用其问题集合排除重叠 |

下载 URL、文件名和 SHA256 在 [`sources.json`](sources.json)。MASSIVE 为 CC-BY-4.0；T2Ranking 上游标注 Apache-2.0；C-MTEB 派生数据卡未明确单独许可，因此本仓库不再分发语料原文。

## 下载与恢复原输入

```sh
.venv/bin/python tools/download_data.py --all --output downloads
.venv/bin/python tools/materialize.py --output work
.venv/bin/python tools/reconstruct_inputs.py \
  --sources-dir downloads --output work/evidence
```

完整 T2Ranking `collection.tsv` 为 3,659,243,528 字节，其他文件较小。下载器先校验固定哈希再接受文件，已有同名文件也要校验；不会用最新版本替换旧版本。
只想检查下载器可先运行 `--sources massive-test`，不会顺带下载完整语料。

重建工具按 [`recipes/`](recipes/) 恢复原输入文件的样本顺序、字段、全文和 JSON 序列化，最终 SHA256 必须与历史文件一致。
它不重新抽样、不修改标签、不裁剪文本。标签、协议、合成预热输入已在相应归档中，不需要推理来恢复。
`--experiment evidence-20261018` 可只恢复一个实验；`--sources-dir` 可以重复传入已有下载目录。

## 原构造代码

- 分类：[public_decision_benchmark.py](../evaluation/runner/scripts/public_decision_benchmark.py) 的 `prepare_massive`、`prepare`。保留重复输入、原始 18 类标签及固定顺序。
- 答案证据：[decision_evidence_benchmark.py](../evaluation/runner/scripts/decision_evidence_benchmark.py) 的 `source_selection`、`select_queries`、`prepare`。种子 `answer-evidence-v1`；排除旧数据中的 6,129 个规范化问题；L2/L3 为正，L0/L1 为负。
- 调优：[2026-10-19-d3-tuning.py](../docs/research/2026-10-19-d3-tuning.py)。开发集和留出集独立于主测试。

种子名称是公开别名。参与哈希排序与随机抽样的原始 UTF-8 字节以十六进制保存在[固定种子清单](../manifests/seed-material.json)，代码使用相同计算字节；名称清理不重新抽样。协议中的显示名称会不同于历史文件，原始协议哈希仍保留在发布清单中。

也可以直接从固定来源重新构造新证据数据：

```sh
.venv/bin/python evaluation/runner/scripts/decision_evidence_benchmark.py \
  prepare --source downloads --output work/new-evidence-dataset
```

新构造的协议会记录新的本地路径，协议文件不一定与历史协议逐字节相同；恢复历史文件请使用上面的 recipe 入口。样本内容和标签不得因此改变。

## 规模与排除

| 阶段 | 原始问题 / 片段 | 所有模型共同完整覆盖的问题 / 片段 |
| --- | --- | --- |
| 工程 | 12 / 177 | 12 / 177 |
| 验证 | 20 / 327 | 20 / 327 |
| 正式 | 200 / 3,105 | 197 / 3,054 |
| 独立无证据诊断 | 40 / 557 | 38 / 517 |

正式共同集：924 个正例、2,130 个负例；183 个问题有标注证据，14 个问题没有。
3 个正式问题因完整输入无法被所有模型处理而统一排除，共 51 个片段，其中 11 个正例。原始分母和排除理由保留在[共同覆盖记录](../docs/research/2026-10-18-evidence-common-coverage.json)中，不算成正确拒绝。

旧相关性标签、新答案证据标签和后续调优划分不能混用。任何公开数据标签都不等于逐条事实正确性裁决。
