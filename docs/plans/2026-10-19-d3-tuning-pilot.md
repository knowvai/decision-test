# D3-mini / D3-flash 调优试验

## 授权与边界

2026-10-19 用户授权调研并实测两款 D3 的质量与效率优化。此轮仅测试
D3-mini、D3-flash，不扩展到 Clef，不修改生产业务、权重、精度、模型原生
计算或旧测试证据。属于隔离评测实验，不是生产缺陷修复或上线验收。

基线流程：冻结公开输入与标签 -> 原生完整输入编码 -> 鉴权 HTTP
`/v1/systemone` -> vllm-srun 0.2.0 native Decision3 -> 原生概率与 choice
校验 -> 独立统计。输入全文不截断、无重试、无结果缓存、不调用 reranker。
复用 public_decision_benchmark / decision_capacity_preflight / deployment_probe
的权威构造、校验、计时和统计；新编排单列，不能修改旧固定全模型协议。

## 研究依据

- 固定下载模型的官方 README、decision_config、MODEL_MANIFEST。
- 固定 readiness 镜像内 `vllm-srun serve --help` 及
  `families/decision3/family.py`、`engines/native/fast.py`。
- 两者均为 Qwen3.5 text backbone + 原生 FP32 readout，支持跨请求合批。
- 不将模型包 SDK 的 AMD 特有 D3_FAST 开关当作本次 NVIDIA worker 优化。
- 历史容量测试 n20/n1 吞吐：mini 1.53、flash 1.23；它是旧相关性短输入
  负载，不能当作本轮答案证据质量或新容量成绩。

## 数据与质量方案

- 固定 MASSIVE 原 revision 的中文 train 用于开发，validation 用于留出。
  按类别各取最多 20 / 40 条（不足则全取），去掉归一化重复输入，并排除旧 test
  全部输入。保留原类别和原文；均衡抽样结果不等同原始分布 Accuracy。
- 原 T2Ranking 固定版本重新选 16 个开发池、40 个留出池，排除旧 C-MTEB
  全部查询与 evidence-v1 全部阶段查询。查询、片段 ID、文本哈希留档；
  开发/留出之间不共享片段。L2/L3 为正，完整池，不挑难例或重标标签。
- 对两个任务分别冻结 baseline / concise / explicit 三个通用提示版本。
  不使用逐题例外或测试集示例。每款模型仅在开发集选择配置。
- 分类以开发集 macro-F1 最大且 Accuracy 不低于 baseline 为选择规则；
  并列保留 baseline。片段在 3 提示 x 原生 choice 或阈值
  {0.35, 0.45, 0.50, 0.55, 0.65} 中选择池宏 F1 最大且召回不低于 baseline
  的配置。阈值只在已有原生概率上离线计算，不增加推理。并列保留 baseline。
- 留出只比较 baseline 与开发集冻结方案；不根据留出改提示、阈值或重选。
  报 P/R/F1、漏光率、空池拒绝、Accuracy、macro-F1、配对 95% 区间。
  小规模试验只用于发现候选收益，不能宣称生产泛化或训练微调收益。

## 效率方案

质量统一冻结为各模型开发集选出的方案，固定同一张 A100、8 CPU、96 GiB
主机内存限制、原镜像和 BF16/FP32 readout、无缓存。三档部署：

1. baseline：threads=2、seq32、tokens32768、batch window=2 ms。
2. thread1：仅改 threads=1（OMP_NUM_THREADS 同步为 1）。
3. compact：仅改 seq8、tokens8192，threads=2、window=2 ms。

每个任务从开发集预先哈希选择 128 个完整输入作为固定负载；分开报告。
每档先做两轮 16 条长度分层预热，再对 n=1/4/8/16 各测两轮，第二轮反序。
单请求硬超时 300 秒，装载允许 2400 秒。记录逐调用输入/请求/响应哈希、
原生预测和概率、时序、模型元数据、服务 metrics 与 cgroup CPU；不记录密钥。
RPS 分子仅有效请求，分母实测整块 wall；延迟包含 HTTP 排队，不包含客户端
等待槽位。此为闭环有限负载，不是开放到达率、长期 SLA 或生产容量。
每档均对照 n1 语义质量和决策变化。不同运行时块顺序带来的热身/缓存影响
须披露；不挑最快一轮，不据留出成绩重选质量配置。

## 安全、审计与恢复

- 启动前确认旧 r3 已退出、共享 classification.lock 空闲，取得非阻塞锁。
- 仅临时停止并改名保留原两只 acceptance 容器；不删除原容器。新部署使用
  新运行根及私有配置，模型路径只读复用。结束时移除本轮容器并恢复原名、
  原容器、原镜像服务；明确披露隔离评测服务启动时间变化。
- 不触碰代理机 embedding、reranker、proxy；前后从本地验证身份不变。
- 旧证据 hash 保护，输出独占创建；预检失败不发推理；推理异常停止后续档位，
  保留失败与未尝试分母，不自动重跑。传输不明先确认 GPU 无在途再恢复服务。
- 独立离线重算全部汇总、选择与区间；完整行与计划核对，区分重复性能负载
  和独立质量样本。运行期间不改冻结源码。
