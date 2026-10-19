# 模型部署与复现条件

本页是独立仓库的部署材料入口。历史脚本和 README 保留当时状态；涉及旧队列“正在运行”的说明是历史记录，不是现在的操作指令。
本次整理没有新建模型容器、替换权重、重启服务或发送探针推理。

公开脚本的容器所有权标签统一为 `evaluation.decision.acceptance`。这是独立评测环境的命名空间，未对原服务器上的任何容器改名或执行操作。历史部署记录也已做名称归一，原始文件哈希保留用于追溯。

## 使用过的运行方式

| 模型 | 实际运行方式 | 主测试资源 |
| --- | --- | --- |
| D3-edge / mini / flash，Vela 2.0 四种规模 | 官方 `vllm-srun` 0.2.0 原生 worker | 单张 A100-PCIE-40GB |
| Laya Multilingual | Laya 0.4.3 官方 HTTP/batch API | 单卡 |
| Jeva-4B | 作者 SDK + 薄 HTTP 适配；原生 GPU 执行串行 | 单卡 |
| NeoHorse-Jev-4B | 作者 vLLM pooling + CPUHead | 单卡，CPUHead 计入耗时 |
| Clef-Flash | 专用 joint-head SGLang 路径 | 单卡 |
| PPLX-Decider-v1.1-27B | 专用 noncausal/readout SGLang 路径 | 双卡 TP=2 |
| JEV-27B | 作者服务 + vLLM 0.31.0 eager + 决策 LoRA | 双卡 TP=2 |
| 两款 Qwen LLM | vLLM 0.31.0，BF16；分类使用结构化输出 | 双卡 TP=2 |
| Qwen3-Reranker-8B | vLLM 0.31.0 pooling，官方模板 | 另一台主机单卡，和 embedding 共享 GPU |

精筛通常为 8 CPU、96 GiB 容器内存预算；NeoHorse 的 CPU 预算分给 pooling 与 CPUHead。
reranker 为 4 CPU、32 GiB、8192 上下文、GPU memory utilization 0.80、seq16/tokens16384、prefix cache off。
具体以每模型当次 `deployment.json` 为准。两台主机不是等资源对照，共享 embedding 流量也不属于评测并发数。

## 代码与构建材料

代码目录：[`deploy/docker/decision-eval/`](../deploy/docker/decision-eval/)。

| 文件 | 用途 |
| --- | --- |
| `launch_reviewed.py` | 原生部署入口，记录镜像/命令/参数；只替换带评测所有权标签的容器，检查端口和 GPU 占用 |
| `server.py`、`jeva_backend.py`、`neohorse_backend.py`、`laya_http.py` | 薄接口及完整输入、队列、就绪检查 |
| `Dockerfile`、`Dockerfile.decision-v3`、`Dockerfile.preflight`、`Dockerfile.acceptance` | Python/SDK/worker 镜像的逐层构建材料 |
| `Dockerfile.sglang`、`patch_sglang_length.py` | SGLang 专用路径及严格长度检查 |
| `Dockerfile.vllm-lora`、`patch_vllm_lora.py` | JEV 装载时的可选内存拷贝和一致性检查，不改变前向或替换数值 |
| `qwen3_reranker.jinja`、`qwen_reranker.py` | reranker 模板和适配依赖 |
| `nginx.native.conf.template` | 鉴权代理模板；生成后带密钥的配置不入库 |
| `*-lock.json`、`download_models.py` | 已有权重获取和逐文件验证机制 |

实际 Jeva/vllm-srun 源码和构建目录在 [`preflight-20261015`](../artifacts/preflight-20261015/) 中，保留第三方许可。
Dockerfile 需要的 wheel 不入库：Laya 0.4.3 SHA256 为 `1cd9c33ab0b2b12f0122e6bf643fb4ead70309983ffb68f8db854d7a8f412d50`；
Linux CPython 3.12 / CUDA 12 / Torch 2.9 的 `causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE` wheel SHA256 为 `506b7a7c7d89790340b99bac53be5a292237d71d4ac9f1ef1ac5ad98cdcc30c1`。
需要从相应上游取得匹配文件后放入原构建上下文，不能换个同名包就认作相同环境。

已记录的基础 OCI 引用：

```text
vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
lmsysorg/sglang@sha256:b2d574138e22e7b54d9ae26c84670c6e5a13a5b55da7c31bf4a5503866f9aa9b
```

派生镜像的 image ID、补丁哈希和当次参数在原部署记录中；Dockerfile 里的历史本地标签不是可直接从 Docker Hub 拉取的公共镜像。
本仓库未分发 GPU 镜像 tar，未在另一台 GPU 主机重建全部镜像。

## 权重版本

[`models.json`](../manifests/models.json) 汇总 15 个精筛模型和 1 个 reranker。
其中 D3 三款、Vela 四款、Laya、Clef、Qwen3.8 和 reranker 共 11 项有明确的上游 revision；Vela 的版本来自实际安装运行时的固定注册表，不是当前网上最新版。

**当前收集材料里未找到不可变上游 revision 的 5 项：Jeva-4B、NeoHorse-Jev-4B、JEV-27B、PPLX-Decider-v1.1-27B、Qwen3-30B-A3B-Instruct-2507。**
它们的实际部署记录、现存模型发布清单/校验表（有则收录）和原生代码仍已归档。
这几项不能默认下载 `main` 再声称权重与历史测试逐字节相同；严格重跑前须补齐来源或与原保存权重核对。这不影响从已有逐条输出重算本次分数。

## 新机器执行顺序

1. 先离线校验、解包，并按固定来源重建数据。使用新的专用评测目录，不把输出指向归档。
2. 按许可证和版本索引取得权重、原生运行时代码及 wheel；先核对文件哈希，再构建派生镜像。
3. 在新目录配置独立的 `DECISION_EVAL_API_KEY`，权限设为仅所有者可读；使用可信网络或隧道。不要提交 `.env` 或生成的代理配置。
4. 使用 `launch_reviewed.py` 部署指定模型，按原协议验收。该脚本和 `deployment_probe.py` 有真实副作用，不能把它们当离线检查工具执行。
5. 分类使用 `public_decision_benchmark.py local-classification`；答案证据使用 `decision_evidence_benchmark.py run`。先用 `--help` 核对必需参数，重新冻结新主机的运行清单。所有输出写入新 run 目录。
6. 对新运行独立审计。新主机、新依赖或新权重结果必须标成新实验，不覆盖原始记录。

历史 `compose*.yml`、`smoke_test.py`、`initialize_runtime.py` 是保留的构建/早期接口材料，不代表主测试的最终名单和部署参数。
主测试不是 Ollama 测试；仓库里的部署调研不能当作已经测过 Ollama 的性能结果。
