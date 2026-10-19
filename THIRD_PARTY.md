# 来源与许可

本仓库整理测试代码、逐条输出和记录，不转授第三方模型、数据集或引擎的许可，也不为原有项目整体授予新许可证。

## 数据

- Amazon MASSIVE，由 AmazonScience 发布；CC-BY-4.0。固定来源见 [`datasets/sources.json`](datasets/sources.json)，仓库不分发语料原文。
- THUIR T2Ranking，由 THUIR 发布；上游数据卡标注 Apache-2.0。标签定义参见 [T2Ranking 论文](https://arxiv.org/abs/2304.03679)。
- C-MTEB/T2Reranking 是派生版本，数据卡没有明确单独的再分发许可。本仓库保留 ID、哈希、标签和构造方式，使用者从固定上游下载原文。
- 数据集中的文本是待评测材料，不是仓库维护者对其中事实、意见或链接的背书。

## 引擎和模型

- vLLM Semantic Router Model Runtime：[`vllm-project/semantic-router`](https://github.com/vllm-project/semantic-router)，Apache-2.0。归档包含实际使用的 `vllm-srun` 0.2.0 源码子目录；该子目录原拷贝没有根 LICENSE，本次从官方项目补入 Apache-2.0 文本，不能把它当成历史模型执行文件。
- Jeva：[`LiQiiiii/Jeva`](https://github.com/LiQiiiii/Jeva)，原源码内 LICENSE、NOTICE 和 CITATION 随归档保留。
- vLLM、SGLang、Laya 及各模型权重遵循各自上游条款。版本索引见 [`manifests/models.json`](manifests/models.json)，已有许可/归属文本见 [`artifacts/model-provenance/`](artifacts/model-provenance/)。
- Vela 等模型的发布包可能包含额外的 distribution/licensing 状态说明；不能仅凭底座或运行时使用 Apache-2.0 就推断所有权重适用同一许可。

权重、完整语料、镜像二进制和私有运行凭据均未提交。外部下载与使用仍需由使用者确认相应上游条款。
