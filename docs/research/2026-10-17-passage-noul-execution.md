# 原生 Noul 验证执行记录

## 范围

用户明确选择全部支持 noul 的既有 Decision 模型，验证池 20 个 / 333 对；不是新正式评测。协议见 `2026-10-17-passage-noul-protocol.md`。原正式集已完成原生 choice 离线重算，见 `2026-10-17-passage-choice-results.md`；历史阈值证据未覆盖。

## 准备与启动

- 新根：测试机 `/data/sdb/decision-model-eval/passage-noul-20261017`，独立 `source-r1/`、`runtime-r1/`、`preparation-r1/`、`run-001/`。旧运行、数据和源码只读，1418 个历史文件纳入保全清单。
- 13 配置均有模型卡/源码 noul 支持证据，文件 SHA 在 `preparation-r1/support.json`。原分类、容量、四链路入口源码 SHA 完全一致；新增验证入口复用原 transport/readiness/launcher/statistics，原生适配仅新增 noul 及共用 wire 映射。
- 本机 Python 缺 pytest；目标机既有评测 venv 运行 127 项测试最终全部通过（3.84 秒）。最初归档遗漏已有离线测试依赖，出现 122 通过/5 缺文件失败、随后 124 通过/3 缺文件失败；补入三个旧冻结依赖后通过，没有修改被测历史函数。`test-supplement.json` 曾提前记录 127，已用独立 `test-final.json` 明确纠正并保留旧记录。上述过程均在任何新推理之前。
- 原四链路 unit `inactive/dead`、Result success、退出码 0；共享锁无人持有。GPU 采样为 0% 利用率，原 Qwen3.8 专属评测容器仍驻留。新测试获授权后仅由原 launcher 轮换带 acceptance=true 标签的专属容器；代理机服务不操作。
- 2026-10-17 上海 **18:45:13** 启动 `decision-passage-noul-20261017-r1.service`，Restart=no。真实 PID **1075495**、命令为新 `decision_noul_validation.py`，持有既有 classification.lock。未启动旧 local-passage/classification/capacity 命令。
- 18:48 左右，D3-edge、Vela-2.0-0.3B 已各完成 333 有效验证请求及模型内审计，下一模型装载中。这只是进度，不代表全轮完成或质量排名。

## 结果状态

2026-10-17 上海 **20:05:07** 全部完成，**20:06:15** 独立离线审计通过。13 配置各 333 条，共 4329 条唯一有效请求，0 失败/超时，无重试/重启/续跑；20 池的 ID、完整输入/request 哈希、原生 readout、部署/镜像与全局 n=1 区间均核验通过。没有新增合成/choice/正式/reranker 推理。

systemd MainPID=0、inactive/dead、Result success、ExecMainStatus=0；瞬态 unit 已 GC（LoadState=not-found），另核对 unit journal 的 Deactivated successfully，不能仅凭未加载 unit 的默认字段推定正常退出。共享锁已释放。1418 个历史文件 / 355333380 字节保全通过。

最终 PPLX 模型仍运行：容器 `8374099973cd6a008f074b59384a4794ef985b8de82141e7a8544aaf54dddb03`，PID 1111860，启动 `2026-10-17T11:52:14.766877532Z`，与该模型验收身份一致。代理机 embedding/reranker/proxy 的完整 ID、PID、启动时间及 running 状态前后核对不变，没有探针推理或服务改动。旧自动跟踪未恢复，本轮未创建额外自动化。

见 [完整模式对照表](2026-10-17-passage-noul-results.md) 与 [独立审计 JSON](2026-10-17-passage-noul-results.json)。11/13 配置的候选 Accuracy 点估计提升，但池级 F1 只有 Laya 与 D3-mini 提升。Clef 的 F1 为 42.53% -> 41.97%，D3-flash 为 52.90% -> 51.21%，两者配对差值 95% 区间跨零，不能声称换 noul 改善了它们的整体筛选效果。20 池比较未做多重比较校正，也未产生新的正式集结论。

正式集二选一重算还用独立的预测/标签投影交叉复算了 15 配置直接精筛及验证基线，共 30 组 P/R/F1 和 TP/FP/FN/TN，全部吻合；这项检查新增推理为 0。
