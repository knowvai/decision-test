# Decision 模型测试材料

整理 2026-10-15 至 2026-10-19 的中文分类、答案证据片段筛选，以及相关的部署验收、并发和参数对照实验。
这里保存的是已有测试材料，没有为本次整理重新调用模型或修改得分。

## 文章

[开源 Decision 模型实测](docs/articles/2026-10-20-decision-agentic-rag/README.md)：结合两项测试讨论 Decision 模型在 Agentic RAG 中的路由选择与片段筛选，包含完整结果表、指标说明和流程动图。

## 两个主场景

| 场景 | 数据与规模 | 查看结果 |
| --- | --- | --- |
| 中文分类 | MASSIVE v1.1 中文测试集，2,974 条、18 类；15 款模型、16 个配置，共 47,584 条正式记录 | [HTML 汇总](docs/research/2026-10-16-decision-classification-summary.html) · [Markdown 结果](docs/research/2026-10-16-decision-classification-final-results.md) |
| 答案证据片段筛选 | T2Ranking 原始四级标签，L2/L3 为正例；共同测试 197 个问题、3,054 个片段；15 款模型、28 个直接筛选模式 | [HTML 汇总](docs/research/2026-10-19-answer-evidence-summary.html) · [完整结果](docs/research/2026-10-19-answer-evidence-results.md) |

片段筛选的 reranker Top10 对照单独放在[附录](docs/research/2026-10-19-answer-evidence-reranker-appendix.html)，不混入主 HTML 排名。
包含工程、验证、正式和独立无证据诊断阶段，直接筛选与 Top10 合计 **195,580 次调用、14,952 条完整筛选记录**。
GitHub 文件页不直接渲染 HTML；克隆仓库后可直接用浏览器打开，Markdown 可在线阅读。

## 从哪里找

| 材料 | 位置 |
| --- | --- |
| 所有轮次、正式与补充结果的对应关系 | [材料目录](docs/ARTIFACTS.md) |
| 测试脚本、评分器、审计与报告脚本 | [脚本索引](docs/SCRIPTS.md) |
| 模型部署代码、Dockerfile、原生适配及版本记录 | [部署说明](docs/DEPLOYMENT.md)、[模型版本索引](manifests/models.json) |
| 逐条原始响应、请求、耗时、标签、部署记录、冻结源码 | [按轮次分包的原始材料](artifacts/) |
| 数据集下载地址、固定版本、SHA256、构造与重建方法 | [数据集说明](datasets/README.md) |
| 逐文件原始/发布哈希、脱敏与未发布项 | [发布清单](manifests/publication.json) |
| 本次整理的复核结果 | [验证记录](docs/VALIDATION.md) |

原始文件较多，按实验和模型压缩保存；单个归档不依赖 Git LFS，也不需要另找外部下载附件。
测试代码统一放在 `evaluation/runner/`，部署代码放在 `deploy/docker/decision-eval/`。
文档、源码、容器标签和压缩包内部路径都使用独立评测命名；历史源码是经过名称清理的发布副本，原始/发布 SHA256 分别保留，不能把两者混称为逐字节相同。
固定抽样和统计种子的计算字节见 [`seed-material.json`](manifests/seed-material.json)，只调整显示名称，不改变抽样、预测或得分。

## 离线核对

Python 3.12，无需 GPU，也不需要模型凭据：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/verify.py
.venv/bin/python tools/test_offline.py -q
```

`verify.py` 校验归档和逐文件哈希，从原始记录重算两个主场景的质量点估计与耗时，核对记录数量及片段请求的串行时间区间。
它不重新生成 bootstrap 置信区间，也不冒充原主机环境审计。`test_offline.py` 禁止网络连接，测试使用模拟接口，不调用模型。

需要解包、重新生成报告或查看冻结源码时：

```sh
.venv/bin/python tools/materialize.py --output work
.venv/bin/python work/docs/research/2026-10-19-answer-evidence-summary.py --replace
```

`work` 必须是新目录。解包后，历史运行目录位于 `work/evidence/`；数据集原文按[数据集说明](datasets/README.md)另行下载并重建。

## 阅读边界

- 分类、旧相关性筛选、新答案证据筛选、开发集调优和并发测试分开存放，不能把它们合并成同一张排行榜。
- 片段分数按原始公开标签计算；没有人工重新标注，也没有把 F1 当作分类准确率。
- 原严格审计的失败记录和允许在线 reranker 排序波动后的修订审计都保留，没有覆盖旧结论。
- 服务器地址和本地用户目录做了脱敏；密钥、环境文件、生成的鉴权配置、运行缓存和模型权重不入库。第三方数据原文通过固定版本下载，不在本仓库重新分发。
- 11 个模型/重排器条目有明确的上游版本引用，其余 5 个的版本缺口在[部署说明](docs/DEPLOYMENT.md)中列出。不将“可以离线核对结果”写成“任意机器都能一键复现全部 GPU 实验”。

第三方模型、数据集和运行时分别遵循各自条款，见[来源与许可](THIRD_PARTY.md)。
