# D3 证据筛选诊断与替代部署调研

核对日期：2026-10-19。对象：D3-mini、D3-flash；部署调研补充 Clef-Flash。

## 结论

**需要修正此前“低分主要说明模型筛选能力不足”的倾向性解释。** 本轮找到具体的标签语义冲突：原始标签为负例的片段，有的明确包含答案；原始标签为正例的片段，有的只有标题或仅命中词语。记录关联本身没有错，但“与 T2Ranking 标签一致”不能直接等同于“真实业务判断正确”。

这不是证明所有标签都错，也不是证明模型足够好。24 条抽样是刻意增加错误样本的诊断集，且由单个模型辅助阅查，不是独立双人人工金标。不能用它估计全量错标率、修正总体准确率或重排模型名次。

部署方面：D3 有官方 AutoModel、原生 HTTP server 和 Semantic Router worker/Engine 路线；Clef-Flash 已正式支持 Ollama 和专用 SGLang 路线。未找到可直接保留 D3 原生读出语义的 Ollama 支持。**不能把 Ollama 支持 Qwen3.5、支持其他决策模型，当成支持 D3。**

本轮没有新发被测模型推理、下载模型权重、启动/切换容器、变更生产配置或修改任何旧成绩。只读取既有实验、获取公开官方源码/文档，并新建离线诊断产物。本地仅隔离安装 Jinja2 3.1.6 用于模板验证。

## 核验范围

冻结流程：原始 query/passage/grade -> 完整 JSON state -> 已冻结的证据提示与原生 choice -> 已记录响应 -> L2/L3 正例评分。复用已有请求适配与数据校验助手，不另写推理入口。

| 检查 | 本轮结果 | 不代表什么 |
| --- | --- | --- |
| 独立留出集 | 40 池、643 对；210 正例、433 负例 | 不是此前 197 池正式集的重跑 |
| 来源重连 | 原始 query、完整 passage、qrels 等级逐对匹配 643/643；完整源文件 SHA 匹配 | 不证明人工原标签的语义必然正确 |
| 请求与输出 | 两模型各 643 条原记录的 ID/input/request hash、一次调用、有效 HTTP200、原生类别和概率一致 | 不证明神经网络输出等同官方参考实现 |
| 提示文本对照 | 643 个完整输入 × choice/noul，共 1,286 个文本对照全匹配 | 未重跑 tokenizer，也未核验 GPU 数值 |
| 模板版本 | 冻结 worker 格式、原包 `d3_format.py` 与 10 月 17 日新包格式一致；chat template 按原文件执行 | 格式一致不等于权重计算路径一致 |
| 人工裁决 | 本轮人类标注者 0 人；单次模型辅助盲阅 24 条 | 不可称作人工审计通过 |

论文明确将 L2/L3 定义为部分/充分满足信息需求，并用于二元检索正例。因此本轮没有发现“正负等级阈值整体选反”的问题。[原论文标注定义](https://ar5iv.labs.arxiv.org/html/2304.03679)

### 既有记录的补充统计

以下仍以原始 T2 标签为标准，仅作诊断，不是新的成绩或调参选择。

| 指标 | D3-mini | D3-flash |
| --- | ---: | ---: |
| TP / FP / FN / TN | 157 / 176 / 53 / 257 | 152 / 156 / 58 / 277 |
| 候选 Accuracy | 64.39% | 66.72% |
| 概率排序 ROC-AUC | 0.7460 | 0.7457 |
| Average Precision | 0.5775 | 0.5721 |
| 原标签负例且 P(有证据) >= 0.9 | 70 | 52 |
| 原标签正例且 P(有证据) <= 0.1 | 14 | 16 |

全拒绝 Accuracy 为 67.34%，正例比例为 32.66%。AUC/AP 表明相对这套标签存在区分信号，但不足以证明业务可用；高置信度与标签冲突也可能来自模型错误或标签问题，不能直接当作模型过度自信的定论。

两模型的共同结果：共同 TP 149、FP 149、FN 50、TN 250，彼此分歧 45。错误高度重叠，但两模型共用家族/任务格式，不能把它们当成两个独立裁判。

## 盲阅发现

抽样种子 `evidence-diagnosis-v1`，按 ID 哈希固定选择：共同 FP 8、共同 FN 8、模型分歧 4、共同 TP/TN 各 2。审阅时只展示完整 query/passage 和匿名编号；先保存判定及理由，再读取原等级/模型输出对照表。没有截断、改写模型输入或让被测模型重新判断。

结果：**16 条初判与二元原标签冲突，3 条边界不明确，5 条一致。** 这是错误富集诊断样本，且单次模型辅助判断也可能出错。尤其“部分答案”的范围、导航式短查询、历史背景、文章事实错误和仅有猜测的内容，需要人类按业务标准裁决。

报告不粘贴输入全文。以下摘要用于定位，原始 ID、输入 hash 和全部判定保存在新 JSON 中。

| 编号 / 原始 ID | 原标签 | 盲阅摘要 | 诊断意义 |
| --- | --- | --- | --- |
| R18 / t2q:19617 / t2p:33503 | L0 | 片段明确同时给出所问诗句、作品名及正文，初判有证据 | 具体的“原负例可能含答案”反例；两模型均保留 |
| R13 / t2q:22628 / t2p:635942 | L0 | 明确列出所问地方美食榜单的第九、十项，初判部分答案 | 不能把这类统计 FP 全部解释为乱收片段 |
| R06 / t2q:17412 / t2p:204599 | L2 | 只有法律咨询的标题和入口，没有子女抚养判断规则 | “承诺提供答案”与“已经提供证据”不同；两模型均拒绝 |
| R03 / t2q:11749 / t2p:309641 | L2 | 请求猜谜，正文为贵金属首饰分类，没有谜底联系 | 词语重合不能直接满足信息需求 |
| R21 / t2q:16311 / t2p:482046 | L2 | 请求识别俗名对应物种，正文是给宠物起名的方法 | 同词不同意图；两模型均拒绝 |
| R04 / t2q:22628 / t2p:635940 | L2 | 有当地菜品及说明，初判能提供部分答案；两模型均拒绝 | 仍存在值得检查的真实漏判候选，不能全为模型开脱 |
| R15 / t2q:7548 / t2p:230085 | L1 | 仅报告某 app 连接异常，没有解决办法，且目标 app 未对应 | mini 保留、flash 拒绝；条件/对象对齐仍需检查 |

其中部分摘要涉及医学或安全文本：这里只判断文本是否提供所问信息，不验证其中陈述的医学/安全正确性，也不把这些文本作为行动建议。这正是“答案证据代理标签”与“可安全用于最终回答的事实”之间另一层差异。

### 当前可以与不可以得出的结论

- 可以：原始数据连接、请求编排、标签转换和文本提示格式在本轮检查中未发现错位；标签与证据口径已有具体冲突候选。
- 可以：此前的分数应称为对固定 T2 代理标签的表现，不能直接称为 产品 实际业务正确率。
- 不可以：根据 24 条阅查宣称全量标签错误比例、模型真实准确率、已达到上线条件或哪款绝对最好。
- 不可以：断言 A100 runtime 完全无问题。原验收的四条配对题主要检查同引擎单条/批量一致性，不是代表性业务集上的官方参考实现对照。

## 部署方式

| 模型 | 路线 | 支持证据与适用性 | 本机实测状态 |
| --- | --- | --- | --- |
| D3-mini / D3-flash | `vllm-srun` 原生 worker | 当前路线；支持调度、合批、typed decision；不是通用 `vllm serve` 的别名 | 既有两轮已测 |
| D3-mini / D3-flash | Transformers `AutoModel(..., trust_remote_code=True).system_one(...)` | 官方入口，加载专用 runtime；保留 BF16 backbone、FP32 readout；适合作为参考对照 | 本轮仅核验源码/提示，未跑 A100 对照 |
| D3-mini / D3-flash | 包内 `d3_server.py` | 官方 `/v1/systemone`；锁内一次处理一个请求；`batch_size` 是单请求问题数/forward，不是多请求并发度 | 未新部署，不预设更快 |
| D3-mini / D3-flash | Semantic Router Engine mode | 在 worker 前增加管理/路由/部署层，可用引擎参数 | 不等于更换底层判定算法，未新增实测 |
| D3-mini / D3-flash | Ollama / 原版 llama.cpp | 核对 Ollama v0.40.6 与当前主线、llama.cpp 当前主线，未找到 D3 专用格式和 255-way readout 适配 | 不能声称支持；未尝试硬转 GGUF |
| D3-mini / D3-flash | 通用 vLLM / SGLang | Qwen3.5 底座支持不等于 D3 支持；需要非因果 full attention、last pooling、专用读出与正确编码 | 未找到可直接采用的官方 D3 服务指南；未做自定义移植 |
| Clef-Flash | 官方 Transformers | `load_release_model` / `joint_schema_model.py`，保留 joint schema head | 可作参考路线；未新增实测 |
| Clef-Flash | 专用 SGLang | 官方模型卡提供明确部署方法；现有测试已用此路线 | 既有测试已测 |
| Clef-Flash | Ollama | 官方文档/模型库和源码均确认；`/v1/systemone`，Clef 支持要求 >=0.35.1 | 有实际可用发布路线，但本轮未拉取/部署/测吞吐 |
| Clef-Flash | llama.cpp | 当前源码有 `conversion/clef.py`、`src/models/clef.cpp`、SystemOne server 及 joint-head 加载 | 实现存在，不等于本机数值/性能已验证 |

### Ollama 的关键限制

本轮核对的最新稳定版为 **v0.40.6**，发布于 2026-10-16；固定源码 revision `2476d5ead197abe04629651028f83d48dc75d2b2`。不要套用旧版本“只能聊天”的印象，也不要将当前主线新特性当作稳定版已有能力。

- 官方模型库列出 `clef-flash:9b-q8_0`、`9b-mlx-bf16`、`9b-mxfp8` 等。`latest` 为可变标签；Q8 与我们现有 BF16 不是等精度对比，MLX 也不是 A100 路线。需要固定具体模型清单/digest。
- v0.40.6 `images.go` 对 GGUF `decision.type` 移除 completion 能力，`sched.go` 对非 completion 模型设 `numParallel=1`，GGUF SystemOne 经该调度器进入 runner。因此该路径不能靠设置 `OLLAMA_NUM_PARALLEL=8` 就得到八路模型执行；请求排队与实际并行要分开实测。不要将此结论扩展到所有 Ollama 模型、所有后端或未来版本。
- `OLLAMA_MAX_QUEUE` 管队列容量，不代表 GPU 合批；增加它不会自动提高计算吞吐。
- SystemOne 本地文本与 schema 上限 64 KiB，完整请求 32 MiB，输入还必须满足实际加载 context；不支持视频/streaming/生成控制参数。标称模型库 256K context 不会取消 API 字节限制。
- 不能照搬当前 worker 的 `options.return_meta` / `require_full_input` 扩展字段；适配时必须验证 Ollama 自己的完整输入拒绝行为与返回协议。
- 本地端点默认不要求 API key；如未来开放到网络，必须沿用独立受保护代理，不能裸露端口，也不能把业务私有片段发往其云服务。

### D3 参考实现与当前包

两款固定旧 revision：mini `a475367de0d8133a3c1a2a0dc12303d89cfa2ab4`，flash `521525effa79469d899a570726547dd7f00da3af`。

10 月 17 日新包分别为 mini `bd0090dd758d32ba59ef5c749910a8e87fc4391f`、flash `13dc1cc8ef23caefa765907e1d764751e5629f7d`。本轮比较官方 manifest：backbone、readout、tokenizer、chat template 的 identity 文件 hash 与旧包一致；变化包含文档、`d3_format.py`、`d3_runtime.py`、`d3_server.py` 和依赖。不能把更新包误称为训练出更准的新权重，也不能无验证直接升级。

官方数值一致性记录针对 CPU 与 AMD MI325X，包含完整 mask、原生 tokenizer、BF16/FP32 读出及 kernel 设置要求；上游 CUDA accelerator 仍显式 `validated=False`。这不是证明 CUDA 算错，但意味着“已跑通”不能取代在 A100 上对官方 AutoModel 的数值验证。

官方新文档中的 mini 151 RPS、2.7 倍等数字来自 MI325X、短输入和特定并发/冷暖条件，不能当成咱们 A100 长片段负载的承诺。官方量化记录也显示有些 int4 路径省显存却更慢；量化是单独的质量/效率变量。

## 建议的下一步

1. **先裁决标注。** 两位人类按统一的“可部分回答”标准，独立复核本轮样本；分歧保留。再增加按原分布随机抽取的对照样本，才可估计标签问题的规模。建立独立业务金标集，不覆盖 T2 原标签。模型辅助阅查样本不再充当后续调优的未见测试集。
2. **再验证 D3 部署一致性。** 新建小规模独立诊断，固定旧权重、提示、完整输入与 BF16/FP32，A100 上比较官方 AutoModel 和当前 worker。覆盖强置信错误、边界概率、长短输入，核对 token IDs、类别、概率差异及 n=1/n>1。不能仅测四个简单题，也不强求跨批逐 bit 相等。
3. **Clef 再做框架对照。** 官方参考/SGLang 与 Ollama 独立比较。先厘清 BF16/Q8 的区别、完整输入覆盖和真实并发路径，再比较同成功样本的 wall、RPS、P95/P99、显存、失败/拒绝率；量化收益另报。

本轮交付的是诊断和部署可行性调查，以上人类裁决和 GPU 对照尚未执行。未自动开始新一轮测试。

## 产物与复现

- [离线统计及源哈希](2026-10-19-d3-evidence-diagnosis.json)
- [先于揭盲保存的判定与理由](2026-10-19-d3-evidence-blind-review.json)
- [全部 24 条 ID/等级/模型判断/阅查对照及模板验证](2026-10-19-d3-review-verification.json)
- [统计与抽样脚本](2026-10-19-d3-evidence-diagnosis.py)、[揭盲连接与模板验证脚本](2026-10-19-d3-review-verification.py)

脚本禁止网络连接、子进程及 shell；仅新建输出，不提供推理/服务入口。原统计 SHA256 `1cf6fbe3fff6b316ba6fb1aefd610687b8fcad04db73bb7698f2ecf7b0745778`；盲阅包 SHA256 `ab927d214471a7720671bbd7da70b5ac8f98bbfecd63dfa9055c370fc8fb0b34`；对照验证 JSON SHA256 `28c3fed1a538aaa4d5d250593689a1241e8f19cb78ac6de5d534c70b768aa043`。包含全文的盲阅包仅保留于本地临时诊断目录，不写入仓库报告。

核验完成：1,286 个模板文本对照；4 个含并列概率的 AUC/AP 定义测试；643 对完整原文/等级重连与两模型记录核对；24 条无重复阅查连接。均不增加 GPU 推理。仓库基线 `37070e98d32ff92ef153095a42b8fdf746bd1c0e`，工作区既有未提交改动原样保留。

## 官方资料

1. [D3-mini 固定包与 AutoModel 入口](https://huggingface.co/vllm-sr/d3-mini/blob/a475367de0d8133a3c1a2a0dc12303d89cfa2ab4/README.md)，[D3-flash 固定包](https://huggingface.co/vllm-sr/d3-flash/blob/521525effa79469d899a570726547dd7f00da3af/README.md)。
2. [D3 原生模型加载及读出说明](https://huggingface.co/vllm-sr/d3-mini/blob/a475367de0d8133a3c1a2a0dc12303d89cfa2ab4/modeling_d3.py)，[新版 serving 测试边界](https://huggingface.co/vllm-sr/d3-mini/blob/bd0090dd758d32ba59ef5c749910a8e87fc4391f/README.md)。
3. [vllm-srun 官方说明](https://github.com/vllm-project/semantic-router/blob/36b5409e4c62cda965552b8ea862a12c43a63d44/src/model-runtime/README.md)，[CPU/MI325X parity](https://github.com/vllm-project/semantic-router/blob/36b5409e4c62cda965552b8ea862a12c43a63d44/src/model-runtime/docs/records/decision3-parity.md)，[CUDA 状态](https://github.com/vllm-project/semantic-router/blob/36b5409e4c62cda965552b8ea862a12c43a63d44/src/model-runtime/vllm_srun/accel/cuda.py)。
4. [Clef-Flash 官方 Transformers/SGLang](https://huggingface.co/Cloudflare/clef-flash)，[Ollama v0.40.6 决策指南](https://github.com/ollama/ollama/blob/v0.40.6/docs/capabilities/decision.mdx)，[Clef-Flash 标签与格式](https://ollama.com/library/clef-flash/tags)。
5. [Ollama SystemOne 协议](https://docs.ollama.com/api/systemone)，[v0.40.6 调度代码](https://github.com/ollama/ollama/blob/v0.40.6/server/sched.go)，[能力检测](https://github.com/ollama/ollama/blob/v0.40.6/server/images.go)，[端点长度检查](https://github.com/ollama/ollama/blob/v0.40.6/server/routes.go)。
6. [llama.cpp Clef 专用读出实现](https://github.com/ggml-org/llama.cpp/blob/31288c9c66e59377f7788d920678a565bb89dd37/src/models/clef.cpp)，[vLLM 支持模型与自定义模型约束](https://docs.vllm.ai/en/latest/models/supported_models.html)。
