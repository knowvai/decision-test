# Vela-0.8B 单卡并发诊断

时间：2026-10-17 上海 02:25–02:33。用户授权临时切换隔离评测服务，排查单卡模型 n5 吞吐近乎不变的原因。没有修改模型、镜像、运行时源码、精度、驱动或 GPU 功率/频率设置，没有重跑冻结测量。

## 结论

**当前 Vela 解码模型的原生运行时会合并调度批次，但不同完整输入前缀随后仍被拆成串行的底层调用。** 这是独立片段负载几乎没有 n5 吞吐收益的明确机制；不能把该曲线解释成 A100 已经达到算力上限。它也不是把 HTTP 或 max-num-seqs 上限错误设为 1。

实际使用 `vllm-srun 0.2.0` 的 `native` PyTorch 引擎，不是 Qwen 的 vLLM 0.31.0 生成服务。日志确认 cuda:0、bfloat16、max_num_seqs=32、max_num_batched_tokens=32768、batch_window_ms=2、shared context off、approximate numerics；live command/image 与原 run-001 Vela-0.8B 部署记录完全一致。

## 调用链与证据

冻结切片：独立合成输入 → 原请求映射（require_full_input）→ 原鉴权代理 → 原生调度器 → Vela2Model → DecoderMember → NativeEngineModel.tree → 原判断头与校准 → 原响应校验。

本次增加的唯一请求选项是官方 `options.return_meta=true`，只获取身份和时间元数据。服务执行代码未打补丁，结果缓存仍为 0，全程一次 HTTP 调用、无重试、无截断或 CPU 回退。

原镜像中源代码的关键路径：

1. `scheduler/batching.py:344` 调用 coalesced，把队列中的多条请求组为调度批次；这些批次标记为非 exact。
2. `scheduler/scheduler.py:436` 对该批次调用 `model.run_approximate(items)`。
3. `families/vela2/family.py:457` 对 DecoderMember 固定进入 `member.run(items, packed=True)`。启动参数 `--no-enable-shared-context` 不会关闭这一内部分支。
4. `families/vela2/members.py:334` 按 `tuple(tree.prefix)` 分组，`members.py:351` 循环调用底层 `engine_model.tree()`。前缀包含完整输入部分，不是只比较公共 system prompt。
5. `engines/native/engine.py:271` 的 packed 路径进一步调用单前缀 `_run_tree`；它并没有把多个不同前缀变成一次 rows/forest 调用。

对原类运行无模型、无 GPU 推理的调度测试，以假的底层引擎仅记录调用次数：

| 输入与路径 | 原方法产生的底层调用 |
| --- | --- |
| 5 个不同前缀，packed=True | 5 次，每次 1 个前缀 |
| 5 个相同前缀，packed=True | 1 次，1 个前缀、5 个问题块 |
| 5 个不同前缀，packed=False | 1 次，5 个前缀的 rows 布局 |

最后一行只证明已有代码能构造该调用，不是该路径的真实性能或数值一致性验证。本次没有切换 HTTP 服务到 rows 模式。

## 真实 HTTP 对照

沿用既有诊断生命周期、冻结 launcher、请求映射、计时和解析函数。新增独立诊断目录，使用三个各含 60 条的合成池：

- mixed-distinct：60 个不同输入，67–1054 字符，平均 358.83 渲染 tokens。
- fixed-distinct：60 个等长不同输入，仅记录编号不同；每条 286 字符、288 渲染 tokens。
- fixed-shared：60 个不同 case ID，但输入完全相同；每条同为 286 字符、288 渲染 tokens。这是机制对照，不模拟独立片段业务。

每池 n1/n5 各 20 条预热 × 2 轮；测量顺序 n1/n5/n5/n1，每块 60 条。合计 240 预热、720 测量，全部有效，HTTP 200，无重试。原 199 条工程池、20 条验证池与正式 200 条池均未用于这次诊断。

| 输入池 | n1 mean / P95 ms | n1 RPS | n5 mean / P95 ms | n5 RPS |
| --- | ---: | ---: | ---: | ---: |
| 不同长度、不同输入 | 81.44 / 83.72 | 12.278 | 383.22 / 402.63 | 12.744 |
| 等长、不同输入 | 81.65 / 84.26 | 12.247 | 385.07 / 401.20 | 12.727 |
| 等长、完全相同输入 | 82.14 / 84.75 | 12.174 | 162.26 / 167.70 | 30.313 |

各格合并两轮 120 条，P95 从原始记录重算，RPS 为有效请求数除以块 wall time 之和。固定长度对照控制了长度和 token 数；条件顺序未随机化，只有两轮短测，不能据此报告生产 SLA。

## 批次内部发生了什么

使用原生 Prometheus histogram 的 sum/count 增量，而不是凭 HTTP 并发数推断 GPU batch：

| 等长池指标 | 不同输入 n1 | 不同输入 n5 | 相同输入 n5 |
| --- | ---: | ---: | ---: |
| 调度批次平均条数 | 1.00 | 2.50 | 2.50 |
| 每批处理含 readout 的均值 ms | 76.47 | 193.80 | 79.88 |
| 请求等待首次调度的均值 ms | 2.23 | 177.34 | 76.78 |

不同输入时，每批 2.5 条的耗时约为单条的 2.53 倍，吞吐基本不变；相同前缀时，相同平均批次大小的耗时接近单条。因此，代码中的前缀分组串行执行与真实请求的时间、吞吐变化一致。底层调用次数测试是无 GPU 的结构验证；没有冒充实际 CUDA trace。

**指标命名陷阱：** 该版本响应中的 `meta.queue_ms` 是接收至提交的时间；`meta.compute_ms` 是提交至结果完成的时间，包含调度等待。源码见 `runtime.py:1267` 和 `runtime.py:1288`。本次 n5 不同输入的这两个字段约为 0.75 / 379.02 ms，不能据此声称“排队几乎为零、纯 GPU 计算 379 ms”。真正的调度排队值是上表的 Prometheus queue 指标。

## 资源观察及边界

所有测量块 CPU cgroup throttled periods/time 增量为 0；总 CPU 使用约 0.96–1.00 核，未碰到 8 核配额。固定不同输入池的 GPU0 一秒采样平均 SM 活动率约 25.8%、显存活动率约 1.7%、功率约 41W；GPU1 空闲。采样覆盖预热和块间间隙，不是纯测量区间的算力利用率，更不是 FLOPS profile。

本轮 GPU0 pclk 采样为 765 MHz；这与前一次 Qwen 诊断测量期间的 1410 MHz 不同。未修改频率，也未做锁频 A/B，不能把低频视为已证明的根因或据此估算提速倍数。单线程 CPU 发射、许多小算子、运行时路径和频率各自的开销仍需 profile 分离。当前证据不支持“GPU 算力打满”的结论，但也不证明没有其他硬件/CPU 限制。

结论只对本次 Vela-0.8B 实测成立。Vela-4B/9B 源码共用 DecoderMember，具有同一分组路径，但本次没有分别运行它们。Vela-0.3B 是 EncoderMember，不能套用本结论；D3、Laya、Jeva、Qwen 也不能被统一归因。

## 后续方向

优先验证原生 rows/forest 路径对不同输入的真实合批，在单独配置中检查性能、显存、完整输入覆盖和判断/概率差异；必要时再做 CPU/CUDA 时间线分析。不能只增加 seq、线程或 HTTP admission，也不能把相同输入控制组的 30.3 RPS 当成独立业务片段容量。

这轮完成的是根因诊断，不是修复上线。没有覆盖原镜像、源码或测量，也没有偷偷将服务改成优化后的执行路径。

## 证据与恢复

远端目录：`/data/sdb/decision-model-eval/vela-diagnostic-20261017`。上海 02:31:27 标记 `complete_restored`，02:33:04 独立检查 systemd 正常退出、MainPID=0、共享锁无持有者。原 Qwen3.8 思考服务 command/image/CPU 配额一致，带鉴权 health=200、无鉴权=401；恢复未发送推理请求。

离线核验通过 ID/input/request hash、一次调用、在途峰值、服务端请求计数、完整记录和来源身份。原容量审计保护的 1011 个文件、上一轮 Qwen 诊断的 73 个文件及其审计文件均未变。151 个原生运行时 Python 文件记录 SHA，关键 7 个文件保留只读内容快照。

- [诊断执行器](2026-10-17-vela-capacity-diagnostic.py)
- [离线核验器](2026-10-17-vela-capacity-diagnostic-audit.py)
- [机器可读审计](2026-10-17-vela-capacity-diagnostic-audit.json)，SHA256 `408c42c28609a5dfce875eb017ddeed6c94289877709ee590266ed566593bf5d`
- [原生代码与无推理调用测试](2026-10-17-vela-runtime-source.json)
