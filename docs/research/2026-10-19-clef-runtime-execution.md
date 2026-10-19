# Clef-Flash Runtime 参数实验执行记录

方案：[参数矩阵](../plans/2026-10-19-clef-runtime-parameters.md)。

## 准备与启动

- 用户明确授权 Clef-Flash 参数与并发实测。复用既有启动器、原生请求/解析、
  闭环计时和失败后关闭新请求的逻辑，不启动旧全模型测试。
- 唯一启动器改动是默认关闭的 SGLang metrics 开关；Session 增加显式原生
  family 与完整 token 数校验，worker 默认保持原样。未改生产默认配置。
- 本地 37 项针对性测试通过；远端冻结快照现有测试目录 31 项通过。
  远端集合较小，不将两者相加为独立测试数。
- CPU 原生编码预检 256 条通过，最长 3762 tokens，实际守卫上限 8186。
  完整 query/片段、分类输入、提示与 token 编码均不截断；没有预检 GPU 推理。
- 冻结 53 个源码文件，保护历史 3681 个文件；数据沿用 D3 pilot 固定性能子集。
  复用样本用于性能/输出稳定性，不充当新留出集或完整池质量测试。
- 事前共享锁空闲，GPU 计算进程均归原 acceptance 模型所有。代理机三只保护
  服务 ID/PID/启动时间与历史基线完全一致，running、无 OOM/重启。
- 原 acceptance 模型与代理容器停止后分别改名为 `*-held-clefruntime` 保留。
  正常完成将恢复原 ID、名称、镜像、命令；PID/启动时间会变化。
- 上海 **2026-10-19 14:45:27** 启动唯一
  `decision-clef-runtime-20261019.service`，`Restart=no`、`RemainAfterExit=yes`。
  实际 MainPID **1813382** 与共享 classification.lock 的 FLOCK 持有者一致。
  运行根 `/data/sdb/decision-model-eval/clef-runtime-20261019`。
- 计划 7 部署、196 块、19712 次 HTTP 调用，其中 1792 预热、17920 实测。
  当前为计划，最终实际覆盖与审计结果另行追加。

## 边界

只测 SGLang 参数，不换 Ollama、不量化、不改权重/阈值/提示、不调用 reranker。
单 A100 GPU0、BF16/TP1、8 CPU、96GiB RAM；无更多硬件或结果缓存。
客户端并发 1/8/16/32/64；实际 GPU 合批不能用并发设置值替代。
全部参数组启用相同 metrics，新增可观测性成本包含在本轮测量内。
首个失败阻止新请求，保留已在途请求与未尝试分母，不自动修复或重跑。

## 完成与恢复

- 上海15:24核对全部196块、19712条完整换行记录，均valid/HTTP200；
  1792次预热、17920次实测，零请求失败/超时/未尝试。
- **15:24:44**进入原服务恢复，**15:28:34**完成。
  systemd终态`active/exited`，MainPID=0、Result=success、ExecMainStatus=0，
  Restart=no，共享classification.lock无FLOCK持有者。不是仍在运行推理。
- 原模型ID `d18a58afabef1dfb624378ed7a6365c603ff39a8320dcb9e12678fa261d7b3e5`
  和proxy ID `56b73142e97098100daad87383967c4e709ddebdf99ac020a71aef80cfe6d514`
  的名称、镜像、命令已恢复。模型PID1833114、proxy PID1833212，启动分别为
  UTC `2026-10-19T07:24:44.486383162Z`、`2026-10-19T07:24:44.782689686Z`；
  均running、无OOM/重启。PID/启动时间变化是事先声明的恢复过程。
- 本地已有SSH复核代理机三保护服务，原ID/PID/StartedAt均不变：
  embedding PID2661389、reranker PID1980450、proxy PID1981420，均running、
  无OOM/重启。本轮没有调用它们推理或修改它们。

## 独立离线审计

- **15:29:45**首次完整审计通过，新增
  `offline-audit/results-001.json`，SHA256
  `1f2e68b9b88c77db308478c26093ced19c93dcc2525d9d02dc7df24f12b79288`。
- 审计器为source外的`2026-10-19-clef-runtime-audit-v4.py`；此前准备版本保留，
  没有在运行中改冻结源码。审计hook禁止网络/子进程，逐行复算、核对模型原生
  token数、真实服务端解析参数和请求计数器增量，3681历史文件及53冻结源码未变。
- Prometheus使用此前审计已有的标准parser，没有改运行环境依赖。
  审计开发时发现本地缺parser，独立临时目录复用既有包；随后修正测试fixture的
  counter TYPE名称。新增11项离线审计测试通过，连同相关回归共48项本地通过。
  这些是审计准备问题，不是模型推理失败或重跑。
- counters按Prometheus类型解析，不将名称以`_count`结尾的gauge误当单调计数器。
  每块SGLang请求数、e2e计数、prompt token总量与逐请求记录完全一致，生成token为0。
  原生启动就绪检查/编译与正式HTTP调用分开；采样GPU值不是精确峰值。
- 原生SGLang响应不提供D3式逐请求CUDA meta。GPU路径由固定镜像、原生joint head、
  GPU0部署、实际配置和GPU观测建立，不能伪称逐响应含GPU证明或已获得实际batch直方图。
- 只读复核全部片段性能记录的ID并集，1个近0.5输入出现25个重复翻转事件；
  分类无变化。保留实际波动，不以输出必须逐位相同作为审计成功条件。

## 交付

[结论与全部表格](2026-10-19-clef-runtime-results.md)、
[完整独立审计](2026-10-19-clef-runtime-results.json)。
表格由离线renderer生成，结论与边界样本解释在生成后补充。
没有发现通用的服务端调参吞吐收益；客户端8并发是当前延迟/吞吐的实用起点。
参数上限调高没有明显收益，seq8呈现部分尾延迟/资源取舍，不作为生产扩容结论。
未改生产默认值、未新增自动化、未自动追加后续测试。
