# Decision 部署整改与运行验收

日期：2026-10-15。范围：14 个 Decision 候选和两个本地基线的隔离评测服务。
范围说明：以上是当时实际验收的 16 项，不是当前待测数量。用户随后移除 Intern，当前为 13 个 Decision + 2 个基线；分类已启动，片段筛选尚未启动，以[当前方案](../plans/2026-10-14-decision-benchmark-plan.md)和[分类执行记录](2026-10-15-decision-classification-execution.md)为准。
本文承接[部署 review](../archive/decision-eval/pre-2026-10-15/research/2026-10-15-decision-deployment-review.md)，不修改产品服务、不跑正式质量榜。
结论：16 项均已有通过本轮合成运行验收的配置；这不是“16 项均已证明最快”，也不是生产 SLA 验收。

## 已完成的整改

- D3 三款及 Vela 四款改用已固定版本的官方 `vllm-srun` worker，不沿用旧参考镜像。
- Jeva、Intern 复用已有队列实现有界接入：256 个等待位、300 秒请求期限，队满 429、超时 504。GPU 仍串行；响应明确记录 `cross_request_batching=false`，不把 HTTP 并发称为 GPU 合批。
- Laya 使用 0.4.3 官方 HTTP app 和 `/v1/systemone/batch`。只加载指定 multilingual 权重，通过官方预测 hook 复用完整输入校验，拒绝 state、instruction、option 截断；不启用受 1024 长度限制的 fast 路径。
- NeoHorse 使用作者的 vLLM pooling 启动器和常驻 CPUHead，通过薄 SystemOne 适配合批提交编码行、按原索引回填。没有重写模型 forward、普通 score 替代决策头或排除 CPU/HTTP 开销。
- JEV 显式关闭 prefix cache，固定作者 `single`/`thinking=off` 模式及实际通过的 vLLM eager 配置。D3/Vela 禁用结果缓存和 shared context；其他引擎保留专用适配要求的缓存/分块限制。
- 所有外部路径经过既有独立 bearer 鉴权，原生端口仅绑定 loopback；使用用户要求的 HTTP，不增加 TLS。Nginx 使用大小写敏感的锚定匹配，避免普通 map 忽略 key 大小写。凭据只保存在远端私有文件。模型只读挂载，缓存独立，去除 capabilities/只读根文件系统。
- 新启动器只替换携带指定归属标签的验收容器，拒绝端口冲突和其他 GPU 计算进程。旧评测容器和模型均保留，未操作业务服务、驱动或原始数据。

## 实际部署边界

硬件为双 A100-PCIE-40GB，PHB 互连，无 NVLink。每次只驻留一个评测模型；
单卡统一 GPU 0，27B 和普通 Qwen LLM 用 TP=2。总 CPU 配额为 8，NeoHorse 分为主干 6、CPUHead 2。
当前调度值是可验收的起始配置，**不是已经证明最快的配置**。

| 路径 | 本轮调度配置 | 注意事项 |
| --- | --- | --- |
| D3/Vela worker | 最大 16 序列、16384 token，队列 256 | 实际合批查看 worker metrics，不能用上限代替实际 batch |
| Jeva/Intern SDK | 单 GPU forward，有界等待队列 | 参考服务性能单列；Jeva 的 option batch 不是跨请求合批 |
| Laya 官方 API | 本轮原生 batch 请求指定 8 | 49 条独立 states 一次提交，允许底层多批；无截断 |
| NeoHorse | 作者固定 128 序列、32768 token、显存比例 0.4、eager；薄适配最多合入 8 条记录 | 不受统一启动器的 `--seqs/--tokens` 参数控制；保留作者原始启动逻辑 |
| Clef/PPLX SGLang | 最大 16 running requests、16384 prefill token | 非因果/联合决策头所需限制保留，不能强开缓存或 CUDA graph |
| JEV/Qwen vLLM | 最大 16 序列、16384 token，prefix cache 关闭 | 单卡和双卡分组报告；使用原生 API |

## 逐模型验收

验收为独立合成中文题：18 选项分类、4 个单条二元判断、49 个候选整池、鉴权、
返回完整性和单条/合批数值对比。Qwen reranker 只验原生整池重排，不做分类。
这些检查不代表准确率评测、生产 SLA 或完整上下文边界压测。

| 模型 | 当前结果 | 运行方式 |
| --- | --- | --- |
| Vela 2.0-0.3B | 通过合成题和整池检查 | 官方 worker，保留默认 FP32 策略 |
| Vela 2.0-0.8B | 通过 | 官方 worker |
| Vela 2.0-4B | 通过 | 官方 worker |
| Vela 2.0-9B | 通过 | 官方 worker |
| D3-edge | 通过 | 官方 worker |
| D3-mini | 通过 | 官方 worker |
| D3-flash | 通过 | 官方 worker |
| Jeva-4B | 通过；内核 doctor 严格检查通过 | 作者加速 SDK + 有界排队 |
| Intern-Decision-4B | 通过 | 作者 HF backend + 有界排队 |
| Laya Multilingual | 整池通过；预算超限实际返回 422 | 官方 API + 无截断 hook |
| NeoHorse-Jev-4B | 通过 | 作者 vLLM + CPUHead + 薄接口 |
| Clef-Flash | 通过 | 官方适配 SGLang，TP=1 |
| Qwen3-Reranker-8B | 通过原生整池与单条检查 | vLLM pooling，官方模板与 yes/no readout |
| JEV-27B | eager 路径分类、整池及长度拒绝通过；编译路径未通过 | 作者 vLLM 服务 + 决策 LoRA，TP=2，eager |
| PPLX-Decider-v1.1-27B | 通过分类、整池与超长输入拒绝检查 | 专用 SGLang 适配，TP=2 |
| Qwen3-30B-A3B-Instruct-2507 | TP=2 结构化分类、49 条判断及超长拒绝通过 | vLLM，TP=2，原生结构化输出预检 |

已完成的前十款单条/整池返回概率经离线复核均有效，四条配对题未发生类别翻转。
最大绝对概率差：Vela-0.3B 约 4.57e-7，其他 Vela 为 0；
D3-edge/mini/flash 分别约 0.003791/0.000393/0.000309；Jeva、Intern 为 0，Laya 为 0.0072。
NeoHorse 最大差约 8.58e-6，四条配对题无类别翻转，实际薄适配合入批次出现 2、7、8 条记录。
Clef 最大差约 0.002629，PPLX 约 1.87e-5，均无配对类别翻转；Qwen reranker 的单条/整池分数最大差约 6.88e-5。
JEV eager 最大差约 0.002974，四条配对题无翻转；18 类及每条二元判断均为一个模型请求，18 类明确返回 `adaptation=single`。
这只是小样本一致性检查，不能推断正式集所有决策不变。
D3/Vela 的原生 `golden.status=unverified`、CUDA `validated=false` 仍如实保留，未伪造官方认证。

## 证据与复现

实现入口：[启动器](../../deploy/docker/decision-eval/launch_reviewed.py)、
[合成验收](../../deploy/docker/decision-eval/deployment_probe.py)、
[服务说明](../../deploy/docker/decision-eval/README.md)。
远端评测根目录下的 `acceptance-20261015/evidence/` 保存逐模型 `.deployment.json`、
`.probe.json`、带时间戳的尝试记录及 `.container.log`；不含 API key。
最终逐项汇总为 `acceptance-summary.json`，核对 16/16 配置通过；JEV 的 eager 与失败编译尝试分别留档。
结束时保留的是带鉴权的 JEV eager TP=2 服务，公共端口 30889，健康和服务信息接口复查为 200；
其余模型是轮换验收并停止，不是同时常驻。正式测试仍由同一启动器逐个切换。

镜像：vLLM `c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`；
SGLang `a682ef469e9900ec254bb6d98e9c6052d0e0c727dcb005b4ea1a229b77424fae`；
最新 SDK/worker 派生镜像 `34427a5062234b04a3702dabad9a7fa08598295c074db880d3429ea99b856523`。
早期 worker 轮次使用该轮记录中的派生镜像；随后补入 Laya hook/NeoHorse HTTP 客户端，未改变官方 worker 内核。
正式评测须重新冻结最终镜像和源码身份，不能用一个最终 hash 追溯替代所有旧尝试。

22 个镜像环境 CPU-only 针对性测试全部通过，覆盖队列、过载、超时、合批映射、Laya hook、Jeva 输出、概率合法性及鉴权配置。
真实代理的 `/health`、`/v1/models`、`/v1/systemone` 共 9 组无 key、错误 key、大小写变动 key 检查均为 401。
另一台机器通过内网访问公共入口得到 401，访问两条原生端口被拒绝连接；原生端口未向内网开放。
真实模型验收不采用 CPU fallback、不重算权重 SHA256，也没有暗中缩 batch 或截短原文。
PPLX 对编码后 18104 token 的请求返回 400，普通 Qwen 也明确拒绝超出 8192 context 的请求。
Laya 以有效的低预算 `max_len=16` 触发真实预测 hook，返回 422；这验证了拒绝路径，不代表已压测其完整 8192 上下文。
普通 Qwen 的预检只检查原生结构化输出，没有伪造概率，也不代替正式 runner 的 LangChain 客户端验收。

冷启动读盘较慢，验收控制器在保留 Clef 已加载进度的情况下延长等待窗口，没有重载模型。
后续 vLLM/SGLang 启动配置启用框架自带 checkpoint prefetch、每 rank 两个预读线程；
只影响加载阶段，不改 checkpoint、量化、推理头或计时边界。早先已完成启动的配置以各自记录为准，
不能把新增预读取推断为正式推理加速。

### JEV 编译路径的失败证据

后续状态修正：同日正式队列的 eager 冷启动仍出现 NaN，已通过独立拷贝实验定位到本机
部分 pinned CPU LoRA 缓冲区的主机到 GPU 传输异常。因此下面是历史验收现象，不能解释为
已证明编译器是根因或 eager 单独就能保证正确。仅禁用 pinning 的派生镜像也在 run-004
预热失败；当前第二版加入独立 CPU 副本和 GPU 传输逐值校验，实机验收状态以执行记录为准。
前向、BF16、TP=2、eager 不变，证据和续跑状态见[分类执行记录](2026-10-15-decision-classification-execution.md)。

本轮 JEV 在 TP=2、8192 context、16 sequences/16384 tokens、prefix cache 关闭时，
BF16 权重装载及 LoRA 注册成功，但分类及二元请求返回 400：
`Out of range float values are not JSON compliant: nan`。
没有将 NaN 替换为零或均匀概率，没有更换权重、启用 CPU offload 或把失败计为分类错误。
`jev-27b.compiled-failed.*` 保存该次配置、原始响应和容器日志。
随后保持相同权重、精度、TP、输入、调度及缓存策略，只启用 vLLM `--enforce-eager`，
实际分类、49 候选整池和超长输入拒绝全部通过。这是当时单次验收证据，不代表后续冷启动稳定；
不自动尝试编译后失败再回退，也不把 eager 和编译模式混成一条速度结果。
该历史验收未定位到具体上游算子，也未证明 eager 是稳定规避路径，不声称修好了 vLLM 内核。
两张卡的原生 P2P 不可用，vLLM/SGLang 均退至 NCCL 通信；这是通信后端选择，不是 CPU 模型推理。
PPLX 已在同一硬件 TP=2 实际通过，不能据 JEV 的失败概括成“双卡机器不能运行 27B”。

## 尚未完成的正式评测前置项

1. 按[冻结方案](../plans/2026-10-14-decision-benchmark-plan.md)使用独立 12 个工程池、每模型最多三组配置择优；本轮合成题冷启动/JIT/耗时不能进入性能表。
2. 用独立 20 池做筛选阈值校准；LLM 分类复用既有 LangChain 结构化输出，筛选的评分或结构化判断路径另行冻结。
3. 分类已经按新协议启动，正式 200 池筛选尚未运行；分类客户端已有单独回归和执行记录，不重复列为部署待办。完整结果审计前不发布“已提速多少倍”或“可替换产品 reranker”的结论。
4. 修改工程配置后重新核对输入、输出及运行身份；本轮未做完整上下文边界、4 池并发、生产许可或长期稳定性验收。
