# Qwen3-30B-A3B HTTP 并发性能诊断

日期：2026-10-17（上海时间）。范围：用户授权临时切换评测服务，使用独立合成输入诊断 n1/n5，结束后恢复 Qwen3.8 思考服务。不是冻结容量测试重跑，不是正式质量测试，也没有实施部署优化。

## 结论

**已定位到引擎执行阶段，尤其是解码过程随并发增长而变慢；没有证据支持“并发上限或 CPU 配额设得太保守”。** 当前硬件/后端组合确有未调优项，但尚未通过内核级 profile 或相应 A/B 实验分离 TP 通信与 MoE 内核各自的贡献，不能宣布 GPU 算力已到极限，也不能宣称某个启动开关就能解决。

原容量测试 Qwen3 的 n5 全部成功，mean 512.28 ms、P95 673.01 ms、9.713 RPS；n1 为 184.77 ms、249.74 ms、5.412 RPS。即延迟约 2.77 倍、吞吐约 1.79 倍，不是 HTTP 承载失败。20% 延迟增幅不是冻结的通过线。

## 基线与边界

调用链：独立合成输入 → 无重试 closed-loop HTTP 客户端 → 鉴权代理 → vLLM → TP=2 的 Qwen3 MoE → 严格 JSON 校验。没有修改请求协议或在下游补偿引擎行为。

- 服务：vLLM 0.31.0；镜像 `sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`。
- 原设置：BF16、TP=2、context=8192、max-num-seqs=32、max-num-batched-tokens=32768、GPU memory utilization=0.90、CPU=8、OMP_NUM_THREADS=2。
- CUDA graphs 已启用并成功捕获；不是 eager-only。prefix caching 按冻结协议关闭。
- 客户端连接池 20，代理 worker_connections=2048，限速 5000r/s、burst=256 nodelay；未见 n5 admission 瓶颈。
- 硬件：两张 A100-PCIE-40GB，GPU 间拓扑 PHB，无 NVLink；P2P read 查询跨卡为 CNS（Chipset not supported）。
- A3B 指每 token 激活参数量，不是全部权重占用。BF16 权重约 56.87 GiB，不能原样改 TP=1 放入一张 40GB 卡。

## 独立对照

复用冻结 launcher、请求构造、解析和计时函数，未修改冻结源码。三组依次执行：原设置、仅 CPU 8→16（不重启模型）、CPU 恢复 8 且仅 batched-token 参数 32768→2048（重启模型）。

每组 60 条独立合成中文输入，query+document 为 67–1054 字符；每档预热 20 条 × 2 轮；测量顺序 n1/n5/n5/n1，每块 60 条。合计 240 预热、720 测量，全部有效，无重试，每个请求一次调用。不是原 199 条工程池、20 条验证池或正式 200 条池。

各组每档合并 120 条测量；输入平均 312.73 tokens、输出平均 10.20 tokens，三组及两档相同。温度 0，最大输出 64，全部 stop、非思考、严格 JSON。合并 RPS 使用有效请求数除以两块 wall time 之和，P95 从原始记录重算。

| 设置 | n1 mean / P95 ms | n1 RPS | n5 mean / P95 ms | n5 RPS |
| --- | ---: | ---: | ---: | ---: |
| 原设置：CPU 8 / tokens 32768 | 168.25 / 207.99 | 5.943 | 432.05 / 549.31 | 11.271 |
| CPU 16 / tokens 32768 | 169.20 / 208.39 | 5.910 | 426.97 / 524.17 | 11.426 |
| CPU 8 / tokens 2048 | 170.82 / 207.74 | 5.854 | 444.11 / 581.01 | 10.964 |

CPU 加倍的 n5 mean 下降约 1.18%，RPS 增加约 1.37%，不构成明显改善。tokens=2048 没有表现出收益；本次小幅变慢不应推广为对所有负载的稳定退化。顺序不是跨条件随机化，只有两轮短测，且额外 GPU 频率采样仅覆盖第三组。

## 时间花在哪里

下表为原设置的服务端 Prometheus histogram sum/count 增量均值；每块服务端请求计数均为 60，与客户端一致。

| 指标 | n1 ms | n5 ms |
| --- | ---: | ---: |
| 客户端端到端 | 168.25 | 432.05 |
| 服务端端到端 | 164.82 | 427.78 |
| 引擎初始排队 | 0.148 | 0.035 |
| 预填充阶段 | 87.45 | 129.86 |
| 解码阶段 | 72.83 | 264.35 |
| 首 token | 92.29 | 163.55 |
| 每输出 token 时间 | 7.92 | 28.75 |

主要新增时延出现在解码阶段，而不是 HTTP 代理或初始入队。这里的 decode 是服务端阶段墙钟时间，包含运行中的调度/交错影响，不是纯 GPU kernel 时间；queue 接近零不代表调度完全没有开销。各阶段与端到端之间还存在调度、预处理和交付等差额，不能把这些指标简单强行相加为完整 profile。

三组全部测量块的 CPU throttled periods/time 增量均为 0；原设置平均使用约 3.10 核，增加到 16 核没有解决解码变慢。所有块 preemption 增量为 0。原设置 KV 容量 78,240 tokens；tokens=2048 后变为 134,496 tokens，但也没有改善 n5。因此 32 序列上限、当前短输入的 KV 容量不是本次瓶颈；这不保证 32 条满 8192 上下文也能同时驻留。

第三组测量阶段每张卡采到 n1 23 个、n5 11 个频率样本，全部为 SM 1410 MHz、memory 1215 MHz，clocks event reasons=0；功率上限为默认 250W，未修改频率或功率。不能把启动阶段显示的 default application clock 765 MHz 当作实际推理被锁在 765 MHz。

## 已确认的后端限制

以下来自本次诊断容器启动日志，不是借用前一天验收日志：

1. 两张卡缺少可用 GPU P2P，vLLM 明确关闭 custom all-reduce，TP/EP communicator 只选择 `PYNCCL`。这是实际通信路径限制，但 PyNCCL 并不意味着改为 CPU 计算，现有数据也没有证明 PCIe 带宽被打满。
2. 使用 `TRITON Unquantized MoE` / `TritonExperts`，缺少 `E=128,N=384,device_name=NVIDIA_A100-PCIE-40GB.json` 调优文件，明确回退到 default MoE config。已确认没有对应调优配置，不等于已量化它造成了多少损失。
3. CUDA graphs 已捕获，冷启动/装载未计入测量；n5 两轮表现接近。没有证据把稳定的 n5 变慢解释为一次性 JIT。

这套部署更准确的描述是“使用通用后端与不利的双卡互联，尚未证明经过针对性优化”，而不是“已经测到 Qwen3/A100 的生产容量上限”。GPU 活动率也不能证明 Tensor Core 算力饱和。

## 下一步优先级

先对同样独立合成输入做 CUDA/NCCL 时间线或 operator profile，分离小批量 MoE kernel、collective 与调度间隙；再用官方支持的 A100/MoE tuning 或兼容后端做单变量对照。当前证据不支持继续盲目增加 max-num-seqs、CPU 或 KV 预算。

单卡量化、不同 TP/PP 拓扑或更好的互联属于新的部署配置实验；量化还需要单独核对判断变化，不能混入冻结容量结果，也不能把单卡 TP=1 作为当前 BF16 权重的直接替换。

## 证据与恢复

远端目录：`/data/sdb/decision-model-eval/qwen3-diagnostic-20261017`。包含协议、逐请求元数据、逐块统计、原始指标快照、CPU cgroup 增量、GPU dmon、频率采样及脱敏启动日志。未打印私有原文、密钥或思考内容。

本地执行器：[diagnostic.py](2026-10-17-qwen3-capacity-diagnostic.py)；离线核验器：[diagnostic-audit.py](2026-10-17-qwen3-capacity-diagnostic-audit.py)。

上海 02:05:02 状态变为 `complete_restored`；02:05:33 独立复核确认 systemd 正常退出（ExecMainStatus=0、MainPID=0），共享 classification.lock 无持有者。原 Qwen3.8 思考服务的 command/image/CPU 配额与原值一致，带鉴权 health=200、无鉴权=401；恢复过程没有发送推理请求。

完整离线核验通过：720 测量、240 预热全部有效；ID/input/request hash、一次调用、实际在途峰值、服务端计数和显式单变量配置检查通过。原最终审计保护的 1011 个文件 SHA 全部未变，冻结测量、源码及分类证据未覆盖。

机器可读结果：[diagnostic-audit.json](2026-10-17-qwen3-capacity-diagnostic-audit.json)，SHA256 `fd0af450af36e4fd46917ce47068960378e02f279d1a2f8742413957a8feffec`。远端原容量审计 SHA256 仍为 `1268349d65461980fd1b8c53bf5b6f074f1d0cf9e602c645d2b04296ff8ae084`。本次未改生产代码、驱动、功率/频率设置、镜像或模型权重，没有部署“优化后配置”。
