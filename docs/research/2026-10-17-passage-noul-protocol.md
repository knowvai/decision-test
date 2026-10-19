# 原生 Noul 验证对照协议

## 授权与边界

2026-10-17 用户要求片段筛选主结果按原生二选一统计，不使用验证集校准阈值，并明确选择“验证池覆盖全部支持 noul 的 Decision 模型”。原 15 配置 / 30 条件已独立离线重算，见 `2026-10-17-passage-choice-results.md`。旧阈值运行及其审计不改写。

本次只对既有 13 个 Decision 配置检查原生 noul 支持并做一次验证，排除两款 LLM；不增加已移除的 Intern，不调用 reranker。输入限定冻结的 20 验证池、333 个完整 query/document 对，116 正例 / 217 负例。沿用 C-MTEB 的检索相关性标签，包括相关但不能单独回答问题的片段，不改成“充分回答”的新任务。

## 冻结变化

上游：`dataset-six-mask-v1/validation.inputs.private.jsonl`，数据 protocol SHA256 `b4b03b04cac155e27c38c984b24d2ac31b71d89f2b8c8d05d35fa27f54b78a40`。

目标阶段：复用 `decision_capacity_preflight.QUESTIONS` 中的同一问题 ID 和 instructions；仅 type 改为 noul，移除 choice criteria。原模型、权重、部署预算、完整输入保护、非思考配置不变。各原生实现可能有不同的 noul 头或内置校准；不修改作者实现，不称之为只换输出字段。

下游：true 映射 relevant，false 映射 irrelevant。若原生返回 choice/boolean，直接用原生类别；若仅返回 P(true)，固定取 true/false 概率较大者，相等取 false。概率比较在数学上等价于固定的 0.5 决策边界，但不是由验证标签挑选的 F1 阈值，更没有第二次置信度过滤。保留原生数值和 readout 来源，不伪造布尔返回。

基线：读取 `run-001/<model>/requests.jsonl` 中已有的 333 条 validation 原生 choice；不重发 choice。以 case_id / fragment_id / input_sha256 严格配对。验证集曾被旧运行用于校准，不能把这次模式选择结论当作独立正式集泛化结论。

## 执行与失败

- 使用新的源码、运行时和结果目录；复用权重只读挂载。持有同一 classification.lock 后，原启动器仅轮换带有 `evaluation.decision.acceptance=true` 的专属测试容器。保留最终测试模型，不恢复旧 Qwen，也不动代理机 reranker/embedding。
- 复用既有 SerialHTTP、readiness、launcher、输入哈希与统计函数；独立验证入口是必要的，因为原入口强制执行阈值校准和正式四链路，不能用它替代验证专测。不修改产品代码或分类/容量逻辑。
- 全局 HTTP n=1、固定池及候选顺序、无重试、无自动续跑；启动/JIT 2400 秒，单请求硬超时 300 秒。任何不合法响应或不明确终态立即停止，不自动修复/重启/转用 choice。
- 静态源码/模型卡核对 noul 支持和 true 方向；每条验证响应同时做类型、范围、完整输入和非思考契约检查，不另发额外合成探针。不支持及未执行明确列出，不冒称成功覆盖。
- 只读取标签进行离线统计，不用标签修改提示、请求、决策边界或部署。新请求上限为 13 * 333 = 4329；实际覆盖以落盘记录为准。
- 前后保全旧运行、数据、源码与历史审计；记录新源文件、镜像、部署、原生输出、请求和响应哈希、唯一 call_id、n1 时间区间以及所有失败。

## 报告

每模型报告验证池宏 P/R/F1 与固定 2000 次池级 bootstrap 95% 区间、候选 Accuracy、TP/FP/FN/TN、空选择池、模式翻转数量及配对差值区间。失败池仍占 20 池分母，失败请求不计为正确；成功子集另列。给出有效/尝试/失败/超时数、请求 mean/P95/P99 和全尝试耗时。新 noul 与历史 choice 的时间不是同轮交替实测，不能归因于模式的纯速度提升。

不宣称 noul 天然更准，也不由 20 池对照宣称最优模式。不启动新正式评测或产品集成。
