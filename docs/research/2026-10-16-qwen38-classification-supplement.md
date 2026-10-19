# Qwen3.8-27B 分类补充测试

日期：2026-10-16。状态：06:41:00（Asia/Shanghai）run-007 完成，06:53:28 通过全量离线审计。
新增 2,974 条全部有效，Accuracy 88.03%、macro-F1 87.04%；旧 14 款的 98 个证据文件不变。
新增[机器可读审计及配对区间](2026-10-16-qwen38-classification-audit.json)，不覆盖原审计。
唯一方法入口为[总方案](../plans/2026-10-14-decision-benchmark-plan.md)。本次只追加一款通用
LLM，不重跑或改写已审计的 14 款，不启动候选筛选或产品集成。

## 官方版本

- 仓库：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)。
- 固定 revision：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，Apache-2.0。
- 完整仓库 32 文件、55,586,114,863 字节，18 个权重分片，包含官方 tokenizer、chat template、
  config、index 和视觉处理配置；不是 FP8/社区量化版。架构为 `Qwen3_5ForConditionalGeneration`。
- 官方默认开启 thinking，本补充试验显式设 `enable_thinking=false`，与旧 Qwen 的直接结构化
  分类口径对照。不是评测其默认 xhigh 思考能力，也不根据正式成绩调提示。

## 代理清理与下载

已在测试机复核旧 Qwen3-30B 的全部 28 个源文件：大小与官方仓库元数据一致（魔搭特有的
`.gitattributes` 对照魔搭），16 分片齐全、JSON/index 可解析，非权重文件另比对 SHA256。
测试机仅多一个历史 `.download.pid`；无模型文件差异。旧下载 PID 均不存在，下载/同步锁
可独占获取，无模型文件被进程打开；节点监控的主机根目录挂载不当作模型使用。

04:19:15 上海时间，删除代理机 `staging/qwen3-30b-a3b-instruct-2507` 冗余副本
61,084,263,662 字节（约 56.9 GiB）。可用空间由 74,859,859,968 增至 135,943,606,272
字节。之前六款的临时副本已自动释放；保留测试机权重、旧下载日志/manifest、Docker 镜像、
其他数据及业务服务。没有额外全量 SHA256。

复用旧 acquisition 脚本快照，仅把任务根目录改为脚本所在目录；不修改旧快照。
新目录 `staging/acquire-qwen38-20261016` 保留 `manifest.json`、`acquire.py`、
`proxy-cleanup-verification.json`、`proxy-cleanup-result.json`、`events.jsonl` 和日志。
清单 SHA256：`cc6beac782c3acfcb7b9eddd82cfb0ecd6afe41182cd4378c5ee783aea1b0cba`。

04:19:29 启动 `decision-acquire-qwen38-20261016.service`，初始 runner PID 1869317，
下载子进程 1869398，仅供定位，每次须核对真实进程/锁。使用已有 huggingface_hub 0.36.0
经代理下载官方固定 revision，禁用 Xet，八路续传。源/目标均保留 5 GiB；最多四路逐文件
rsync 内网同步至测试机 `staging/qwen3.8-27b`，完成逐文件大小、JSON/index 和忽略目录
时间戳的 dry-run 后才释放本次临时副本。正常字节增长不重启、不重复下载。

## 补充协议与验收

预备 `source-r7` / `run-007`，仍在既有 `classification-20261015` 根目录和共享锁下。
在原 runner 增加显式 `--extend-roster --after-run run-006`：只有末尾序列完成、协议前序
哈希及数据身份匹配时才能追加；禁止任何前序计划内模型，避免重测旧模型或恢复 Intern。
新增来源链接与旧 14 款成绩分开保留，不覆盖 run-001/002/003/006 及失败修复记录。

测试机现有 benchmark venv 的相关测试 **61 passed**；默认不允许扩名单，显式追加拒绝
重复模型、前序哈希/数据/完成列表变化、循环引用、证据缺失和未完成序列。真实旧协议链
读取到 14 个已完成模型。原审计列出的 **98 个文件哈希全部保持**，提示、LLM 调用、
逐题计时、评分及配对区间共 11 个函数与 source-r6 的 AST 一致。准备证据与准确的启动
命令保存在测试机 `classification-20261015/qwen38-supplement-20261016/preparation.json`；
启动前准备记录的状态为 `prepared_not_started`，不把工具测试通过当作新模型推理验收。

- 同一 MASSIVE v1.1 zh-CN/test，2974 条、18 类，重复输入保留。
- 复用原 LangChain strict JSON schema、提示和标签，temperature=0、max_tokens=64，
  一条在途、每题一次、无重试/回退/截断；300 秒单题上限、2400 秒冷启动上限。
- 同一双 A100-PCIE-40GB、BF16/TP2、8 CPU、8192 context、16 max sequences、16384
  batched tokens、关闭 prefix cache。实际复用 `decision-eval-vllm:v0.31.0-pinned`，
  ID `c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`。
- 新模型专属 `--language-model-only --default-chat-template-kwargs '{"enable_thinking": false}'`。
  镜像源码已确认有 Qwen3.5 架构及这些参数，Transformers 5.17.0；这不代替实机装载验收。
  不使用 JEV 的 LoRA 装载补丁，不量化、不 offload、不改变模型 forward。
- 下载/同步完成前保留最后 PPLX 服务。后续只替换本任务拥有的评测容器，先检查所有输入
  token 长度、鉴权、非思考模板及九条独立合成预热，再运行正式记录。
- 完成后核对记录/ID/输入哈希/源码/部署身份，重算 Accuracy、macro-F1、有效输出率、
  成功请求平均延迟和 P99，与旧 Qwen 按同题配对比较；双卡性能单列。

## 下载完成与冷启动预检

05:39:36 上海时间完成下载校验，05:56:13 完成全部 32 个文件的内网同步与大小、JSON/index、
rsync dry-run（无差异）检查，05:56:27 释放本次代理临时副本。期间两次网络连接断开，
已有脚本在子进程退出后保留断点续传，第三次下载子进程正常完成；没有重复文件写者。
完成事件副本保存在测试机补充证据目录的 `acquisition-events.jsonl`。

05:58:55 重新核对目标全部 32 文件、55,586,114,863 字节及 18 分片索引；非权重文件与
官方 Git blob/LFS 元数据哈希一致，未对全量权重额外计算 SHA256。原审计 98 文件与
source-r7 的准备哈希再次通过；证据 `target-verification.json`。

现有镜像内实际 tokenizer 渲染相同 domain_messages：正式输入 327 至 367 token，
三种合成预热输入为 327/342/390 token，均含完整模板且另预留 64 输出 token，远低于
8192 上限。显式非思考模板包含已闭合的空 think 区块。初次计数误把 Transformers 5
返回的映射字段数当作 token 数，已标记无效并单独保留，正式推理前改为对完整渲染文本
直接 encode 计数；有效结果见 `token-length-preflight.json`，未改变模型或评测输入。

05:59:48 仅替换本任务拥有的评测容器，开始 Qwen3.8 双卡冷启动；原 PPLX 测试成绩及
权重保留。`decision-qwen38-preflight-20261016.service` 使用原 deployment_probe、
`--pool 1` 进行合成运行验收，不是并发测试或正式分类；日志为补充证据目录 `probe.log`。
初始日志识别 Qwen3.5 架构和 text-only 模式。06:08:51 运行验收完成：健康接口授权请求
200，缺失、错误及大小写变化的凭据均 401；6 条独立合成请求全部 200、合法 JSON、
finish_reason=stop，无 reasoning/reasoning_content，输出 7/12/8/12/8/12 token；
超长请求 400 拒绝。仅使用 pool=1，没有并发测试。验收结果已复制到补充证据目录
`accepted-probe.json`、`accepted-preflight-deployment.json`，摘要为 `runtime-acceptance.json`。

## 补充队列启动

06:18:25 再次核对原 98 个文件、source-r7、目标大小及固定镜像，确认无既有分类进程、
run-007 不存在且共享锁空闲后，启动 `decision-classification-20261015-r7.service`。
实际初始 PID 582745，持有 `classification.lock`；结果 `run-007`，日志 `runner-r7.log`。
只追加 `qwen3.8-27b`，前序为 run-006，使用显式 `--extend-roster`。

原 runner 重新冷启动已验收模型并完成 9 条独立合成预热，再开始正式 2974 条。
启动时 status 为 `warmup_and_classification`，该状态同时覆盖等待就绪，监控结合
warmup/response 文件和日志判断推进。原 14 款没有重测，没有修改任何旧成绩。

## 完成审计

06:41:00 全部正式记录落盘，systemd 正常退出（ExecMainStatus=0、MainPID=0），
共享分类锁释放。06:53:28 完成以下离线核验，不补发模型请求：

- 2,974 个 ID 唯一且顺序、输入哈希与冻结数据完全匹配；2,921 个不同输入，重复记录保留。
- 9 条独立合成预热的输入、调用哈希、模型身份和结果一致，未混入正式成绩或时延。
- 全部正式记录有效，各有一次结构化调用，调用输入哈希、reported_model 和预测一致；
  无超时、请求失败或无效输出。2,720 条输出 7 token，254 条输出 8 token，没有达到 64 token 上限。
- 重算整个 summary，并独立核对正确数、macro-F1、成功请求平均延迟及线性插值 P99；
  失败仍保留在质量分母，合法但分类错误的请求仍计入成功请求延迟。
- 前序链 run-001/002/003/006/007 匹配，合计 15 款、44,610 条正式记录，没有重复模型；
  run-004/005 仍为零正式记录的历史修复尝试。
- 原审计 SHA256 及其 98 个文件在审计前后全部不变；source-r7 准备哈希、协议及部署源码
  哈希匹配。原 11 个核心函数及 classification_schema 的 AST 与 source-r6 一致。
- 核对固定 revision 清单、32 文件大小、18 分片索引和官方非权重哈希，不额外全量权重 SHA256。
  当前 owned 容器的镜像、entrypoint 加 command、只读模型/源码挂载、双卡及 8 CPU 配额匹配。
  最后的 Qwen 模型及代理容器保持运行，不停止业务服务。

审计脚本及结果另存于测试机 `qwen38-supplement-20261016/classification-audit.py` 和
`classification-audit.json`。结果 SHA256：
`5156b7ffe6d406d5a8a34f8dcbdfbadd5e716ad8f4485274f444edf89a14dbc7`。
本地机器可读副本与远端一致，原 14 款审计未覆盖。

## 双卡 LLM 对照

两者均为双 A100-PCIE-40GB、BF16/TP2、8 CPU、同一 vLLM 镜像、相同题目/提示/评分，
每次只有一条请求在途。新 Qwen 使用非思考文本模式；这不是并发吞吐比较。

| 模型 | 正确 / 总数 | Accuracy | macro-F1 | 有效率 | 成功请求平均 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-Instruct-2507（旧 run-001） | 2489 / 2974 | 83.69% | 82.92% | 100% | 145.37 ms | 155.55 ms |
| Qwen3.8-27B（新增 run-007） | 2618 / 2974 | 88.03% | 87.04% | 100% | 355.49 ms | 380.56 ms |

新模型净多答对 129 题：纠正旧模型 185 题，同时在旧模型答对的 56 题上答错；
两者都正确 2,433 题、都错误 300 题。Accuracy 提升 **4.34 个百分点**，macro-F1
提升 **4.12 个百分点**；平均请求时延约为旧模型 **2.45 倍**。两款返回均有效，所以
这里的错误是与数据集标签不一致的分类判断，不是接口失败或 JSON 解析失败。

复用原 classification_interval，以 2,921 个相同输入簇按金标签集合分层进行 2,000 次
配对 bootstrap，簇内保留全部原始记录。新减旧的 95% 区间：

- Accuracy：+3.36 至 +5.33 个百分点。
- macro-F1：+3.02 至 +5.22 个百分点。

本次固定数据与协议下，两项提升的区间均不跨零。没有据成绩调整提示、阈值或挑选运行，
也不把这一个公开语料上的结果称为通用模型能力榜单。

## 适用边界

本次是非思考文本分类，不代表 Qwen3.8 默认思考模式、视觉能力或其他任务的表现。
公开数据可能存在训练暴露；单轮、单条在途没有覆盖并发吞吐或长期可靠性。没有连续
资源遥测，因此不重建历史峰值显存或声称单位 GPU 吞吐优势。

正式记录保留结构化解析结果与 token usage，未保留原始 finish_reason/reasoning 字段；
这些字段只在独立合成验收中逐条检查，不能把合成验收扩称为逐条正式原文检查。
没有改动正式记录格式，也没有为补证重发正式题目。测试及审计已完成，保留最后 Qwen 服务；
不启动 B 场景，不恢复排除的模型，不接入产品。
