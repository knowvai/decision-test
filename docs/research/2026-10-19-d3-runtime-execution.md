# D3 Runtime 参数实验执行记录

方案：[冻结参数矩阵](../plans/2026-10-19-d3-runtime-parameters.md)。
用户在上一轮 pilot 完成后明确要求继续实测 runtime 参数，授权限于两款 D3
的隔离评测，不是生产上线、旧全模型测试重跑或额外模型扩展。

## 准备与边界

- 复用上一轮的数据协议，SHA256
  `3a223a53f3d2b0a96046ab26d308b157c6a0fd8f652073acc9a12ae2a09faf17`。
  两个任务各 128 条固定性能输入，改回原 baseline 提示和原生 choice；
  不与上一轮 concise/阈值配置的绝对延迟直接相减。
- 复用且逐 ID/hash 检查上一轮原生 CPU 编码预检；本轮所用输入全部小于
  最小显式预算 8192 token。未截断，未新增输入或更换权重。
- 检查固定镜像内 vllm-srun 0.2.0 的 CLI/runtime/Decision3/scheduler 源码。
  只新增 launcher 的显式 `worker_auto_budget` 开关，auto 时省略 seq/token
  参数以允许原生设备默认值及启动校准。历史默认不变；没有引擎或 kernel 补丁。
- 复用上一轮 Session，仅增加可选 measurement 注入。新 adapter 在第一次
  已观察失败后阻止新请求，正在途请求按原截止时间排空，未发请求单列。
- 测试命令最初引用了新快照中未包含的 `test_deployment_probe.py`，pytest 在
  收集前退出，没有运行推理。保留 `tests.xml`，改为实际快照中的既有测试目录，
  **最终 113 项通过，6.21 秒**，保留 `tests-final.xml`。
- 新冻结源码 50 文件，历史保护清单 2,615 文件；包括完整上一轮 pilot 的
  源码、数据、结果、审计与运行记录，不包括临时编译缓存。未修改历史证据。
- 前置检查共享锁空闲；测试机 GPU 进程均归原 acceptance 模型容器所有。
  代理机三只保护容器的原 ID/PID/StartedAt 不变、running、无 OOM/重启。

## 启动

上海 **2026-10-19 10:23:12** 启动唯一服务
`decision-d3-runtime-20261019.service`，`Restart=no`、`RemainAfterExit=yes`。
初始 MainPID **1729068**，已核对真实命令及共享 classification.lock 的 FLOCK
持有者一致。运行根 `/data/sdb/decision-model-eval/d3-runtime-20261019`。

原模型 ID
`d18a58afabef1dfb624378ed7a6365c603ff39a8320dcb9e12678fa261d7b3e5`
及原 proxy ID
`56b73142e97098100daad87383967c4e709ddebdf99ac020a71aef80cfe6d514`
已停止并改名为 `*-held-d3runtime` 保留，未删除。正常完成后恢复原容器；
PID 和启动时间会改变，模型、镜像、命令和名称必须恢复。

两模型各九次部署（八种配置及末尾复测基线），每次使用独立 runtime 目录。
计划 432 块、41,472 HTTP 调用，其中预热 4,608、实测 36,864。
原生初始化和 auto 校准另有内部 GPU 工作，不伪称包含在 HTTP 数内。
实际完成数与独立审计结果待终态追加，当前不能将计划当作完成结果。

## 审计准备

离线审计器位于冻结 source 外，不在运行中修改已冻结源码。使用 Python
audit hook 禁止网络及子进程；Prometheus 使用标准 parser。额外 8 项纯离线
测试通过，覆盖按 ID 配对、离线限制、重复样本计数及 CPU 计数器单调性。
未因审计增加推理调用。终态前仅做已完成块的格式/解析核验，不发布中途排名。

## 完成、恢复与外部核对

- 上海 **11:33:51** 最后一块写完。18 次部署各 2,304 次推理调用，两模型
  各 20,736 次；共 432 块、41,472 完整换行记录、74,258,877 字节 JSONL。
  逐行检查全部 valid / HTTP 200，无失败、超时、未尝试或 HTTP 重试。
- 上海 **11:34:31** 进入原服务恢复；**11:39:24** 写入 complete。
  systemd 保留为 `active/exited`（显式 RemainAfterExit=yes），MainPID=0，
  ExecMainStatus=0，ExecStart 的真实执行记录显示 code=exited/status=0。
  共享 classification.lock 无 FLOCK 持有者，不将 active/exited 误读为仍在推理。
- 原模型、proxy 均恢复原 ID/名称/镜像/命令并通过既有鉴权就绪检查。
  模型 PID 1782824，启动 UTC `2026-10-19T03:34:30.946240687Z`；proxy PID
  1782919，启动 UTC `2026-10-19T03:34:31.216588789Z`。均 running，
  OOM=false、RestartCount=0；这些评测服务的启动时间改变是事前声明的恢复行为。
- 从本地 SSH 再查代理机三保护服务，ID/PID/StartedAt 与事前基线完全一致：
  embedding PID 2661389，reranker PID 1980450，proxy PID 1981420；均运行且无重启/OOM。
  本轮没有向它们发推理或修改它们。

## 独立审计

上海 **11:40:39**，首次完整离线审计通过，输出新文件
`offline-audit/results-001.json`。执行的是 source 外的
`offline-audit/2026-10-19-d3-runtime-audit-v2.py`；早期 parser 预检查副本保留。
Prometheus parser 从本地既有纯 Python 包复制到独立审计目录，未安装到运行环境，
逐源码哈希记入审计输出。审计 hook 禁止网络和子进程。

核对完整行、计划分母、逐调用一次、input/request hash、原生完整 token 数、
模型身份/精度/原生读出、实际 seq/token 解析、并发区间及块间不重叠；重算端到端
wall/百分位、指标增量和 CPU/GPU 观测。2,615 个历史文件和 50 个冻结 source
文件 SHA256 均不变，原容器恢复记录通过。审计不要求批处理结果逐位相同；
实际判断变化照实进入结果，不将真实波动误判成调用或协议失败。

- 完整审计 JSON SHA256：
  `6404c6985452059f0a4aa39afe392ea2e5bc7474345a9cbc8e97f1e5942c88b4`
- 实际审计器 SHA256：
  `be38169e0c5f8b62820d43923179d2bb8dc0a3aa207638f4c549c1eec721313b`
- 补充离线诊断 JSON SHA256：
  `8bfd119ae50fff42240d6b87ef150b8ef17a1b0d478f95506cb5360b9d491911`

补充诊断只读取已审计文件，所有输入文件再次核对原审计 hash，新增推理 0。
定位了 mini 两个接近 0.5 的片段在高并发下翻转，以及部分分类首轮的原生 compute
秒级长尾；没有修改数据、追加预热或重跑。没有足够细的 trace 将长尾唯一归因到
JIT、图捕获或具体 kernel。预注册筛选规则未在看过结果后放宽。

## 结果与交付

[结论与全部表格](2026-10-19-d3-runtime-results.md)、
[完整审计](2026-10-19-d3-runtime-results.json)、
[只读补充诊断](2026-10-19-d3-runtime-diagnostics.json)。

auto 在部分分类高并发格子降低了本轮长尾，是后续复验候选；首轮慢请求、两轮差异
和基线漂移使其不能被称为普遍稳态收益。flash seq8/n1 虽机械通过筛选，但 n1
未触及 seq8 上限，不能据此证明限制参数的因果收益。片段筛选没有明显参数收益，
提高并发增加尾延迟；mini 2 个边界样本出现变化，未验证准确率提升。
保留原服务，不做生产配置调整，没有继续启动新实验或创建跟踪自动化。
