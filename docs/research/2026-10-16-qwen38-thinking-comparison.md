# Qwen3.8 思考与非思考分类对照

日期：2026-10-16。状态：run-008 于 10:30:18（Asia/Shanghai）完成，2,974 条正式记录全部
通过独立离线审计。两种配置均已完成；旧成绩不变，最后的思考模型服务保留。
方法入口为[总方案](../plans/2026-10-14-decision-benchmark-plan.md)，非思考基线为
[run-007 补充记录](2026-10-16-qwen38-classification-supplement.md)。

## 授权与条件

用户要求两种模式作对比。原 run-007 明确使用 enable_thinking=false；官方模型支持开关，
默认开启思考、reasoning_effort=xhigh，也支持 medium/low。本次补测默认 xhigh 思考深度，
不额外测试另外两档，不按正式成绩选择档位。

保留已有非思考 2,974 条及审计，不重跑。新配置名 qwen3.8-27b-thinking，共享相同权重路径、
served-model 名称和固定 revision；这是同一模型的第二个配置，不是第 16 款独立模型。
不下载、不清理已有模型，不启动 B 场景或产品集成。

| 条件 | 非思考 run-007 | 思考 run-008 |
|---|---|---|
| enable_thinking | false | true |
| reasoning_effort | 不适用 | xhigh |
| 总输出预算 | 64 token | 4096 token，含思考与最终答案 |
| 思考解析 | 不需要 | vLLM 原生 qwen3 parser |
| 权重/镜像 | 固定官方 revision / vLLM 0.31.0 | 相同 |
| 硬件 | 双 A100-PCIE-40GB、BF16/TP2、8 CPU | 相同 |
| context / seq / batched tokens | 8192 / 16 / 16384，prefix cache 关闭 | 相同 |
| 请求 | 原提示、strict JSON schema、温度 0、单条在途 | 相同 |
| 超时与失败 | 300 秒，每题一次，无重试、回退或截断 | 相同，length stop 计失败 |

4096 是本次预先冻结的思考加答案预算，不把旧版 64 token 限制直接套到思考过程。
如正式记录触及上限或超时，保留失败及质量分母，不重试、不根据成绩提高预算。
最终必须同时报告截断/超时率、有效率、质量、完整请求延迟和输出 token 成本。
温度仍取原协议的 0，而非官方推荐的思考采样温度 1.0；因此不是厂商默认配置成绩。

## 实现与保护

沿用 reviewed launcher 和 local-classification，只增加显式配置映射；底层模型路径相同。
新模型容器使用 language-model-only、enable_thinking=true、reasoning_effort=xhigh、
reasoning-parser=qwen3。没有修改 forward、权重、量化、offload、JEV 补丁或镜像。

当前 LangChain 0.3.35 丢弃供应商 reasoning/reasoning_content 字段，故在同一 HTTP 客户端
增加仅对本配置启用的 response observer。记录请求/响应哈希、模型、用量、结束原因、
思考与答案的字符数/哈希，不另发请求，不打印或保存思考原文。观测开销包含在端到端
计时中；日志文件写入仍在计时之后。默认非思考请求格式不变。

冻结源码 source-r8、输出 run-008、共享 classification.lock。显式
--models qwen3.8-27b-thinking --after-run run-007 --extend-roster，只接续新配置。
任何既有 run-008 或同名 unit 都阻止再次启动。预检持锁，验收通过才启动正式独立
systemd unit 并释放锁交接；正式 runner 仍重新装载和做 9 条独立合成预热。

原审计 SHA256 与 98 个 artifact、新增非思考审计 SHA256 与其 45 个 artifact 集合均再次
通过（两个集合有重叠，不相加当作独立文件数）。source-r7 不改动。domain_messages、
classification_schema、llm_classify、评分、分位数、配对区间等 9 个函数 AST 不变。
变更仅涉及显式输出预算、观测/失败元数据和配置协议记录。

## 准备与实机门槛

测试机 benchmark venv：67 passed，包含默认请求不变、思考预算/模板、一次请求、
reasoning/reasoning_content 元数据、length stop 保留失败而无重试、旧部署配置不受影响。
实际 tokenizer 对完整思考模板计数：正式题 363～403 token，三种预热 363/378/426 token，
另留 4096 输出均低于 8192。模板末尾为尚未闭合的 think 区块，不是空的已闭合区块。

测试机根目录 classification-20261015/qwen38-thinking-20261016 保存 preparation.json、
token-length-preflight.json、preflight.log 和预检证据；准确正式命令及全部准备源码哈希
已记录。qwen38-thinking-driver.py 只编排原 launcher/runner，不另实现分类调用。

07:47:29 启动 decision-qwen38-thinking-preflight-20261016.service，07:53:04 完成
9 条独立合成验收。最终 JSON 全部合法，三种输入的思考字符数分别为 139/270/150，
总输出 68/103/70 token，三轮一致；全部 finish_reason=stop，没有达到 4096 上限。
健康接口授权返回 200，缺失/错误/大小写变化凭据均 401。预检目录的
responses.private.jsonl 是空文件，没有发正式题。证据 runtime-acceptance.json 与
preflight/warmup.private.json 保留，后者 SHA256 为
`b7884cd2af1ba8065e345880c0e7d1c59f9401b118ba6873c79b2673c8f7bc31`。

通过以上验收并再次核对旧证据及源码后，自动启动 decision-classification-20261015-r8.service，
初始 PID 602986（只供定位，监控须核对真实进程）。run-008 前序为 run-007，日志 runner-r8.log，
07:53:06 进入 warmup_and_classification。该状态也覆盖等待服务就绪；执行期间结合真实
进程、锁、响应行数/字节、预热文件及容器活动确认推进，没有重复启动或推理。
10:30:18 正常退出，ExecMainStatus=0、MainPID=0，共享锁已释放；模型与代理仍运行。

## 全量结果

同一 MASSIVE 2,974 条、18 类试卷，双 A100、单条在途。下表延迟包含完整思考、最终
答案及校验，排除冷启动、9 条独立预热和文件写入；思考模式另含元数据观测开销。

| 指标 | 非思考 run-007 | 思考 xhigh run-008 |
|---|---:|---:|
| 正确 / 总记录 | 2618 / 2974 | 2606 / 2974 |
| Accuracy | 88.0296% | 87.6261% |
| macro-F1 | 87.0448% | 86.8793% |
| 有效输出率 | 100% | 100% |
| 成功平均延迟 | 355.49 ms | 3124.02 ms |
| 成功 P99 延迟 | 380.56 ms | 12069.45 ms |
| 平均总输出 token | 7.0854 | 115.4479 |
| 全部总输出 token | 21,072 | 343,342 |

思考模式平均延迟为非思考的 **8.79 倍**，平均总输出 token 为 **16.29 倍**。其输出
token 最小/中位数/P99/最大为 34/96/466.27/1518，全部低于冻结的 4096 上限。
供应商报告的 reasoning token 共 319,351，平均 107.381；其余输出共 23,991，平均
8.067，后者可能含协议分隔符，不能全部当成答案正文 token。完整正式请求耗时合计
约 154.85 分钟，不是并发吞吐或连续资源利用率测量。

思考模式失败、超时、length finish、输出触顶、缺失 transport、空思考均为 **0**。
2,974 条全部 finish_reason=stop，确实产生非空思考，不是仅设置了开关却未实际思考。

## 配对比较

复用原 classification_interval：2,921 个相同输入簇、按 gold-label-set 分层、2,000 次
配对 bootstrap，保留重复记录原权重。以下差值均为“思考减非思考”，单位为百分点。

| 指标 | 点估计 | 95% 配对区间 |
|---|---:|---:|
| Accuracy | -0.4035 | [-1.1127, +0.2373] |
| macro-F1 | -0.1655 | [-0.9808, +0.6095] |

两者都正确 2,555 条、都错误 305 条；思考纠正非思考错误 51 条，同时把非思考正确的
63 条变错，净少答对 12 条。两个区间都跨零，**本次没有观察到思考模式的明确质量
收益，也不足以断言其质量显著更差或两者等价**。

就这项短文本 18 类分类及冻结协议，非思考是更合适的延迟/成本取舍；这个结论不外推
到复杂推理、其他数据集、medium/low 档位或产品路由，也不构成整体能力榜单。

## 审计与保留

新证据为 [独立审计 JSON](2026-10-16-qwen38-thinking-audit.json)，SHA256：
`dbb8cc70448bfb6372e1f513d35947324bd7e700d6957d333d74d9ee7230e8cf`。
测试机 `classification-20261015/qwen38-thinking-20261016/classification-audit.py` 保存
审计脚本，同目录 `classification-audit.json` 保存原件。审计没有请求模型接口。

- 2,974 个 ID、顺序、输入哈希、一次调用/transport 映射与 run-007 完全对齐；2,921 输入簇不变。
- 使用冻结 LangChain binding 在禁止 HTTP 的客户端上重建请求 JSON，全部请求与消息哈希匹配，覆盖原 strict schema、提示、模型、温度、4096 预算和 xhigh 开关。
- 正式及独立合成预热的模型身份、用量、结束原因、非空思考均核验；三种预热重复三轮，不计正式成绩和时延。
- 重算整份 summary，并独立复算 Accuracy、macro-F1、有效率、成功平均及 P99；失败始终保留在 2,974 的质量分母中。
- source-r8 准备/协议/部署哈希、9 个核心函数 AST、前序 run-007、镜像、实际容器命令/挂载、双卡/8 CPU 核验通过。
- 固定 revision 的 32 文件大小、18 分片索引及官方非权重文件 hash 再次通过；没有额外全量权重 SHA256。
- 原审计 SHA256 及其 98 个 artifact、run-007 审计 SHA256 及其 45 个 artifact 在审计前后均不变；两集合有重叠。旧 run-004/005 仍为零正式记录的历史尝试。

累计是 **15 款独立模型、16 个测试配置、47,584 条正式记录**，不是 16 款独立模型。
没有覆盖旧审计、重测旧模型、启动 B 场景或接入产品；最后思考服务不自动停止。

适用边界：单次固定公开试卷可能存在训练暴露；两配置输出预算不同，均为温度 0，
不是厂商推荐随机采样成绩。没有并发压力、持续资源遥测或多次运行方差证据。
原始思考/响应正文没有保存，不能独立重算其正文哈希；已验证的请求重建哈希、解析
结果及元数据不等同于原文留档。token 细分采用供应商报告，不由字符数换算。
