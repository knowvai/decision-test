# 开源 Decision 模型实测

![开源 Decision 模型实测](assets/00-cover.png)

[TypeSafe Jev](https://docs.typesafe.ai/concepts/system-one) 的出现，让「把高频决策交给专用模型」成为一个可选项——而在 Agentic RAG 中，这类判断与决策至关重要。

Decision 模型（决策模型）主要面向分类、路由、条件判定和评分等任务，通常返回选项概率或结构化判断，供应用据此决定后续动作。Jev 就属于这一类模型。

这类模型将自然语言描述的条件与结构化评分结合，使不同的分类与判断任务能够采用相似的调用方式。路由选择可以采用概率最高的路径，证据筛选可以依据候选匹配概率和预设规则决定保留哪些片段。通用 LLM 通过结构化输出也能完成这些工作，因此有必要比较专用模型与通用 LLM 的判断质量和请求延迟。

在 Agentic RAG 中，可以由专用模型承担路由选择和检索片段筛选，Agent 负责规划、信息综合与回答。私有化项目要求数据不出内网，这类判断必须本地完成，不能调用外部 Decision API。

本文以两款通用 LLM 为对照，部署 13 款开源 Decision 模型，测试其中文分类与证据筛选能力，并记录判断质量与端到端请求耗时。

## Agentic RAG 中的决策分工

Agentic RAG 中，Agent 根据任务制定计划、调用工具，并依据已有材料决定继续检索或生成回答。这一过程既包含跨步骤推理，也包含重复执行的分类与筛选。

规划需要完整的任务上下文；路由选择与片段筛选的输入范围相对有限，输出格式也可以预先约定，适合由独立的决策模型服务处理。

**路由选择。** 输入包括用户问题、候选路径及其适用条件。Decision 模型对候选路径评分，上层程序据此将请求分派至对应流程。计划生成、工具参数构造与多步执行仍由 Agent 负责。

**检索后的片段筛选。** 根据用户问题检索得到候选片段后，Decision 模型评估片段是否提供答案证据。程序据此保留或排除片段，再将筛选结果交给 Agent，由其组织回答或继续检索。

![路由选择：根据用户问题与候选路径说明，选择对应流程。片段筛选：用户问题经检索得到候选片段，筛选后由 Agent 决定组织回答或继续检索。](assets/01-architecture-motion.gif)

*图 1：路由选择与片段筛选的功能流程。路径 B 与片段保留状态仅作示意，不代表实测结果。*

路由与筛选独立部署后，可以分别选择模型、记录质量指标，并按请求量配置资源。这些决策服务也可与回答生成服务分别配置：片段筛选通常涉及多次简短判断，回答生成则需要较长的文本输出。

片段评分用于筛选候选材料，Agent 则结合完整上下文判断材料是否足以回答问题，并处理跨文档互补和证据冲突。

[TypeSafe 官方的 RAG 示例](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) 采用了类似分工：在检索与生成之间，分别判断片段的相关性、可用证据和冲突等属性，再组织回答所需的材料。本次测试聚焦其中的答案证据二分类。

## Decision 模型的用法

以 TypeSafe 的 System One 为例，调用时用 `state` 提供待判断的上下文，在 `questions` 中描述问题与判断标准。Choice、Noul、Score 是同一个 `POST /v1/systemone` 接口下的三种提问方式，可以在一次请求中组合使用，并非三种思考模式。

### Choice：从给定选项中选择

[Choice](https://docs.typesafe.ai/primitives/choice) 适合类别判断和路由选择。除了上下文与问题，还需要提供选项及各自的含义。例如，为用户问题选择处理路径时，可以列出“知识库检索”“结构化数据查询”“无需检索”，并说明每条路径适用的情况。

接口返回各选项的概率，以及概率最高的选中项 `choice`，程序据此分派请求。选项之间不要求有高低顺序；关键是把它们的区别描述清楚。

### Noul：判断一个条件是否成立

[Noul](https://docs.typesafe.ai/primitives/noul) 用于是非判断，例如“这段内容是否提供了回答问题所需的证据”。它不需要列出一组选项，也可以补充 `true`、`false` 的判定标准。

返回的 `noul` 是 0 到 1 之间的成立概率，由程序依据阈值决定是否保留片段。这里的 0.8 表示模型认为命题成立的概率为 80%，并不是片段覆盖了 80% 的答案。因此，Noul 适合判断“有没有”，不能直接用来衡量“有多少”。

### Score：按有序等级评价

[Score](https://docs.typesafe.ai/primitives/score) 适合程度判断，需要按顺序描述各个等级。例如，评价材料对答案的覆盖程度，可以依次定义“没有答案信息”“提供部分答案”“完整覆盖答案要点”三个等级。

接口返回各等级的概率，以及等级编号按概率加权得到的 `score`。三个等级从 0 编号到 2，分数也在 0 到 2 之间，可以出现 1.5 这样的中间值。它表示在所定义量表上的位置，不是命题成立的概率，可用于内容质量评价或排序。

Choice 和 Score 还返回 `confidence`，概括概率分布的集中程度；它与选项概率、等级分数各有含义，不能直接当作模型判断的正确率。

![System One 的三种提问方式：Choice 返回选中项与选项概率，Noul 返回命题成立概率，Score 返回分级分数与等级概率。](assets/02-modes.png)

*图 2：三种提问方式的输入要求与输出含义。*

## 私有化部署与测试范围

在要求数据不出内网的私有化项目中，问题分类、检索、片段筛选和回答生成都要在内网完成。开源 Decision 模型成为了满足这一需求的方案选择。

测试覆盖 D3、Vela 系列，以及 Clef-Flash、Laya、NeoHorse-Jev、Jeva、AutoTrust JEV-27B 和 PPLX-Decider，共 13 款模型。它们的规模、训练方法和推理实现不同，本文以“Decision”概括其任务类型，不将它们视为同一种架构。

其中，[AutoTrust JEV-27B](https://huggingface.co/autotrust/JEV-27B) 是独立团队发布的开放权重模型，与 TypeSafe Jev 并非同一产品。**本文未测试 TypeSafe Jev API**，下文结果均来自本地部署的开源模型及 LLM 对照。

两项测试分别考察：

- **中文分类**：将每条输入归入 18 个领域类别之一，评估路由所需的类别判断能力。
- **片段筛选**：根据问题与检索片段评估答案证据，统计保留、误收和漏选情况。

分类测试对应路由中的基础判断，不包含完整的工具选择与任务规划。片段筛选比较 choice 与 noul：前者提供“有答案证据”和“无答案证据”两个选项，后者直接判断“是否提供答案证据”。同一个任务采用不同的表达和判定方式，结果可能不同，这也是同时测试两者的原因。

下文使用各开源模型的原生接口，具体返回格式并不完全相同。片段筛选共测试 28 个模式：13 款 Decision 模型各测 choice、noul，两款 LLM 各测非思考的 choice。**Score 未纳入本轮测试。**

## 分类：准确率与请求延迟

分类测试采用 MASSIVE v1.1 中文测试集，共 2,974 条输入、18 个类别。下表列出全部 16 个配置，含 Qwen3.8 的思考与非思考对照；每个配置均完成全部 2,974 条输入的测试。

表中的两项质量指标分别衡量整体正确率与各类别的分类表现：

- **Accuracy（准确率）**：全部测试输入中，类别判断正确的比例。它反映整体分类效果，样本较多的类别对结果影响更大。
- **macro-F1（宏平均 F1）**：先为每个类别计算 F1，再对 18 个类别等权取平均。对某一类别，精确率衡量“被分到该类的输入中，有多少确实属于该类”，召回率衡量“本属该类的输入中，有多少被正确识别”；F1 综合这两项指标。等权平均使样本较少的类别同样重要，避免其表现被大类别掩盖。

两项质量指标均以百分比表示，越高越好。平均延迟与 P99 按有效请求统计，单位为毫秒，越低越好。

**表 1：中文分类结果 · 16 个配置**

<table><thead><tr><th align="center">分组</th><th align="left">模型 / 配置</th><th align="right">Accuracy (%)</th><th align="right">macro-F1 (%)</th><th align="right">平均延迟 (ms)</th><th align="right">P99 (ms)</th></tr>
</thead><tbody><tr><th rowspan="4" scope="rowgroup" align="center">小于 1B</th><td align="left">D3-edge</td><td align="right">82.72</td><td align="right">82.78</td><td align="right">26.64</td><td align="right">27.83</td></tr>
<tr><td align="left">Vela 2.0-0.8B</td><td align="right">81.67</td><td align="right">81.64</td><td align="right">84.03</td><td align="right">87.28</td></tr>
<tr><td align="left">Vela 2.0-0.3B</td><td align="right">77.91</td><td align="right">77.81</td><td align="right">15.37</td><td align="right">16.45</td></tr>
<tr><td align="left">Laya Multilingual</td><td align="right">61.23</td><td align="right">60.85</td><td align="right">29.31</td><td align="right">31.19</td></tr>
</tbody><tbody><tr><th rowspan="4" scope="rowgroup" align="center">4B</th><td align="left">D3-mini</td><td align="right">88.77</td><td align="right">88.04</td><td align="right">69.92</td><td align="right">72.83</td></tr>
<tr><td align="left">Vela 2.0-4B</td><td align="right">86.99</td><td align="right">86.98</td><td align="right">111.94</td><td align="right">115.08</td></tr>
<tr><td align="left">NeoHorse-Jev-4B</td><td align="right">85.14</td><td align="right">84.98</td><td align="right">596.67</td><td align="right">626.26</td></tr>
<tr><td align="left">Jeva-4B（串行参考）</td><td align="right">80.60</td><td align="right">79.74</td><td align="right">205.62</td><td align="right">212.62</td></tr>
</tbody><tbody><tr><th rowspan="3" scope="rowgroup" align="center">9B</th><td align="left">Clef-Flash</td><td align="right">91.69</td><td align="right">91.36</td><td align="right">71.84</td><td align="right">76.05</td></tr>
<tr><td align="left">D3-flash</td><td align="right">89.64</td><td align="right">89.35</td><td align="right">71.20</td><td align="right">73.74</td></tr>
<tr><td align="left">Vela 2.0-9B</td><td align="right">86.79</td><td align="right">87.22</td><td align="right">112.73</td><td align="right">116.87</td></tr>
</tbody><tbody><tr><th rowspan="2" scope="rowgroup" align="center">27B</th><td align="left">PPLX-Decider-v1.1-27B</td><td align="right">91.90</td><td align="right">91.43</td><td align="right">204.08</td><td align="right">230.25</td></tr>
<tr><td align="left">JEV-27B</td><td align="right">85.74</td><td align="right">85.21</td><td align="right">277.24</td><td align="right">288.28</td></tr>
</tbody><tbody><tr><th rowspan="3" scope="rowgroup" align="center">LLM 基线</th><td align="left">Qwen3-30B-A3B-Instruct-2507</td><td align="right">83.69</td><td align="right">82.92</td><td align="right">145.37</td><td align="right">155.55</td></tr>
<tr><td align="left">Qwen3.8-27B 非思考</td><td align="right">88.03</td><td align="right">87.04</td><td align="right">355.49</td><td align="right">380.56</td></tr>
<tr><td align="left">Qwen3.8-27B 思考 xhigh</td><td align="right">87.63</td><td align="right">86.88</td><td align="right">3124.02</td><td align="right">12069.45</td></tr>
</tbody></table>

Clef-Flash 在单张 A100 上取得 **91.69%** 的准确率，平均请求延迟为 **71.84 ms**。双卡部署的 PPLX-Decider 准确率为 91.90%，正确分类的输入比 Clef-Flash 多 6 条，平均延迟为 204.08 ms。在上述部署配置下，Clef-Flash 使用更少的 GPU 达到了接近的分类质量。

D3-mini 的准确率为 **88.77%**，平均延迟为 **69.92 ms**；D3-flash 分别为 **89.64%** 和 **71.20 ms**。两者延迟接近，D3-flash 的准确率略高。D3-mini 则提供了一个 4B 规模的选择。

Qwen3.8 开启思考后，平均延迟从 355.49 ms 增至 3,124.02 ms，约为原来的 8.8 倍，本次测试未显示出明确的分类质量提升。

延迟按单次请求测量，包含通信与解析。由于部署方式不同，此处比较的是实际配置表现，无法据此推算并发吞吐量。

## 片段筛选：证据保留与噪声过滤

测试采用 T2Ranking。它按照片段对问题的信息需求满足程度划分等级：

- L0 表示不匹配；
- L1 表示主题相关，但不满足信息需求；
- L2 表示相关，并部分满足信息需求；
- L3 表示满足信息需求，且明确包含答案。

本次测试将 L2/L3 作为正例，L0/L1 作为负例。为区分答案保留与无效内容过滤，下文分别记录四个等级的结果。

正式测试采用测试集中的 **197 个问题、3,054 个片段**，含正例 924 个、负例 2,130 个；本轮以“完整问题 + 单个完整片段”为一次评测输入，不截断内容。两款 LLM 使用同一份结构化输出提示词模板作答。

choice 直接采用接口选中的选项；noul 优先采用接口返回的是非判断结果。若 noul 仅返回数值型成立概率，则比较 true（有答案证据）与 false（无答案证据）的概率，选择概率较高的一项，同分记为 false。

评价重点是：**保住答案的同时，能过滤多少无效内容。** 表 2 将这两方面分开呈现：

- **L3 保留率**：181 个 L3 片段中，被保留下来的比例，衡量直接答案片段的保留情况。
- **L2 保留率**：743 个 L2 片段中，被保留下来的比例，反映部分答案信息的保留情况。
- **L1 过滤率、L0 过滤率**：分别在 451 个 L1 和 1,679 个 L0 片段中，被排除的比例，衡量对主题相关但无效内容、以及不匹配内容的过滤效果。
- **有 L3 问题命中率**：101 个原本含 L3 的问题中，筛选后至少保留一条 L3 的问题比例。其余 96 个问题没有 L3，但仍可能有可用的 L2，不应直接视为没有答案信息。
- **每题保留片段数**：保留总数除以全部 197 个问题。筛选前平均每题有 15.50 个片段；保留数量需与答案保留情况一起比较，并非越少越好。

前五项比例均为越高越好。分级保留率与过滤率按各等级的全部片段计算，并非保留结果中的构成比例；问题命中率按问题计算。表格保留全部 28 个模式，并按模型规模分组，不按某一项指标重新排名。

表中同时列出平均延迟、P99 和每个问题平均耗时。前两项对应一次片段判断，后一项对应筛完该问题全部候选片段的实测总耗时，单位均为毫秒，包含通信、推理和结果检查。

**表 2：分级筛选与耗时 · 28 个模式**

<table><thead><tr><th scope="col" align="center">分组</th><th scope="col" align="left">模型 / 模式</th><th scope="col" align="right">L3 保留率<br>(%)</th><th scope="col" align="right">L2 保留率<br>(%)</th><th scope="col" align="right">L1 过滤率<br>(%)</th><th scope="col" align="right">L0 过滤率<br>(%)</th><th scope="col" align="right">有 L3 问题<br>命中率 (%)</th><th scope="col" align="right">每题保留<br>片段数</th><th scope="col" align="right">平均延迟<br>(ms)</th><th scope="col" align="right">P99<br>(ms)</th><th scope="col" align="right">每个问题<br>平均耗时 (ms)</th></tr>
</thead><tbody><tr><th scope="rowgroup" rowspan="8" align="center">小于 1B</th><td align="left">D3-edge<br>choice</td><td align="right">62.98</td><td align="right">52.22</td><td align="right">62.53</td><td align="right">76.35</td><td align="right">72.28</td><td align="right">5.42</td><td align="right">29.88</td><td align="right">54.46</td><td align="right">584.97</td></tr>
<tr><td align="left">D3-edge<br>noul</td><td align="right">58.56</td><td align="right">50.34</td><td align="right">63.41</td><td align="right">78.14</td><td align="right">66.34</td><td align="right">5.14</td><td align="right">29.91</td><td align="right">57.44</td><td align="right">578.32</td></tr>
<tr><td align="left">Vela 2.0-0.3B<br>choice</td><td align="right">86.74</td><td align="right">90.58</td><td align="right">14.19</td><td align="right">25.91</td><td align="right">92.08</td><td align="right">12.49</td><td align="right">22.68</td><td align="right">100.69</td><td align="right">454.74</td></tr>
<tr><td align="left">Vela 2.0-0.3B<br>noul</td><td align="right">17.68</td><td align="right">15.48</td><td align="right">90.91</td><td align="right">92.26</td><td align="right">24.75</td><td align="right">1.61</td><td align="right">22.90</td><td align="right">101.99</td><td align="right">464.47</td></tr>
<tr><td align="left">Vela 2.0-0.8B<br>choice</td><td align="right">96.69</td><td align="right">91.66</td><td align="right">9.31</td><td align="right">30.55</td><td align="right">98.02</td><td align="right">12.34</td><td align="right">85.45</td><td align="right">95.40</td><td align="right">1448.31</td></tr>
<tr><td align="left">Vela 2.0-0.8B<br>noul</td><td align="right">88.95</td><td align="right">81.56</td><td align="right">24.83</td><td align="right">46.75</td><td align="right">92.08</td><td align="right">10.15</td><td align="right">85.50</td><td align="right">95.76</td><td align="right">1459.89</td></tr>
<tr><td align="left">Laya Multilingual<br>choice</td><td align="right">55.80</td><td align="right">42.93</td><td align="right">65.85</td><td align="right">81.42</td><td align="right">67.33</td><td align="right">4.50</td><td align="right">28.02</td><td align="right">56.10</td><td align="right">557.98</td></tr>
<tr><td align="left">Laya Multilingual<br>noul</td><td align="right">68.51</td><td align="right">55.59</td><td align="right">49.89</td><td align="right">66.29</td><td align="right">77.23</td><td align="right">6.75</td><td align="right">28.05</td><td align="right">57.01</td><td align="right">556.07</td></tr>
</tbody><tbody><tr><th scope="rowgroup" rowspan="8" align="center">4B</th><td align="left">D3-mini<br>choice</td><td align="right">87.29</td><td align="right">72.95</td><td align="right">47.45</td><td align="right">67.96</td><td align="right">85.15</td><td align="right">7.49</td><td align="right">77.82</td><td align="right">243.82</td><td align="right">1339.72</td></tr>
<tr><td align="left">D3-mini<br>noul</td><td align="right">86.74</td><td align="right">72.54</td><td align="right">47.01</td><td align="right">68.02</td><td align="right">85.15</td><td align="right">7.47</td><td align="right">77.83</td><td align="right">243.84</td><td align="right">1332.78</td></tr>
<tr><td align="left">Vela 2.0-4B<br>choice</td><td align="right">87.85</td><td align="right">79.14</td><td align="right">36.14</td><td align="right">58.01</td><td align="right">89.11</td><td align="right">8.83</td><td align="right">117.37</td><td align="right">248.23</td><td align="right">1955.82</td></tr>
<tr><td align="left">Vela 2.0-4B<br>noul</td><td align="right">84.53</td><td align="right">72.01</td><td align="right">49.22</td><td align="right">64.74</td><td align="right">87.13</td><td align="right">7.66</td><td align="right">117.48</td><td align="right">249.40</td><td align="right">1957.61</td></tr>
<tr><td align="left">NeoHorse-Jev-4B<br>choice</td><td align="right">91.16</td><td align="right">79.41</td><td align="right">39.69</td><td align="right">55.57</td><td align="right">91.09</td><td align="right">9.00</td><td align="right">1066.52</td><td align="right">6624.51</td><td align="right">16683.04</td></tr>
<tr><td align="left">NeoHorse-Jev-4B<br>noul</td><td align="right">91.71</td><td align="right">77.52</td><td align="right">38.36</td><td align="right">55.69</td><td align="right">93.07</td><td align="right">8.95</td><td align="right">1046.17</td><td align="right">6645.67</td><td align="right">16370.33</td></tr>
<tr><td align="left">Jeva-4B<br>choice</td><td align="right">40.33</td><td align="right">31.76</td><td align="right">86.47</td><td align="right">93.39</td><td align="right">46.53</td><td align="right">2.44</td><td align="right">145.54</td><td align="right">294.01</td><td align="right">2403.99</td></tr>
<tr><td align="left">Jeva-4B<br>noul</td><td align="right">41.44</td><td align="right">32.71</td><td align="right">85.14</td><td align="right">92.02</td><td align="right">47.52</td><td align="right">2.63</td><td align="right">145.84</td><td align="right">294.90</td><td align="right">2409.99</td></tr>
</tbody><tbody><tr><th scope="rowgroup" rowspan="6" align="center">9B</th><td align="left">D3-flash<br>choice</td><td align="right">85.08</td><td align="right">70.66</td><td align="right">49.67</td><td align="right">69.68</td><td align="right">84.16</td><td align="right">7.18</td><td align="right">89.43</td><td align="right">377.26</td><td align="right">1536.60</td></tr>
<tr><td align="left">D3-flash<br>noul</td><td align="right">83.98</td><td align="right">69.45</td><td align="right">51.00</td><td align="right">70.52</td><td align="right">84.16</td><td align="right">7.03</td><td align="right">89.21</td><td align="right">373.72</td><td align="right">1523.43</td></tr>
<tr><td align="left">Clef-Flash<br>choice</td><td align="right">83.98</td><td align="right">66.08</td><td align="right">52.77</td><td align="right">71.89</td><td align="right">81.19</td><td align="right">6.74</td><td align="right">83.50</td><td align="right">304.81</td><td align="right">1423.45</td></tr>
<tr><td align="left">Clef-Flash<br>noul</td><td align="right">78.45</td><td align="right">58.68</td><td align="right">58.76</td><td align="right">75.04</td><td align="right">76.24</td><td align="right">6.01</td><td align="right">83.74</td><td align="right">303.77</td><td align="right">1431.40</td></tr>
<tr><td align="left">Vela 2.0-9B<br>choice</td><td align="right">91.16</td><td align="right">76.18</td><td align="right">41.69</td><td align="right">59.98</td><td align="right">92.08</td><td align="right">8.46</td><td align="right">124.76</td><td align="right">382.91</td><td align="right">2065.92</td></tr>
<tr><td align="left">Vela 2.0-9B<br>noul</td><td align="right">87.85</td><td align="right">70.12</td><td align="right">50.11</td><td align="right">65.99</td><td align="right">89.11</td><td align="right">7.49</td><td align="right">124.98</td><td align="right">384.46</td><td align="right">2085.35</td></tr>
</tbody><tbody><tr><th scope="rowgroup" rowspan="4" align="center">27B</th><td align="left">JEV-27B<br>choice</td><td align="right">93.37</td><td align="right">80.22</td><td align="right">40.13</td><td align="right">58.37</td><td align="right">92.08</td><td align="right">8.80</td><td align="right">395.69</td><td align="right">1801.48</td><td align="right">6283.99</td></tr>
<tr><td align="left">JEV-27B<br>noul</td><td align="right">92.27</td><td align="right">78.87</td><td align="right">42.79</td><td align="right">61.41</td><td align="right">91.09</td><td align="right">8.42</td><td align="right">394.04</td><td align="right">1799.46</td><td align="right">6262.86</td></tr>
<tr><td align="left">PPLX-Decider-v1.1-27B<br>choice</td><td align="right">93.37</td><td align="right">77.93</td><td align="right">42.57</td><td align="right">65.34</td><td align="right">91.09</td><td align="right">8.07</td><td align="right">329.76</td><td align="right">1684.17</td><td align="right">5269.62</td></tr>
<tr><td align="left">PPLX-Decider-v1.1-27B<br>noul</td><td align="right">91.71</td><td align="right">76.72</td><td align="right">44.79</td><td align="right">68.37</td><td align="right">90.10</td><td align="right">7.70</td><td align="right">327.10</td><td align="right">1659.26</td><td align="right">5223.53</td></tr>
</tbody><tbody><tr><th scope="rowgroup" rowspan="2" align="center">LLM 基线</th><td align="left">Qwen3-30B-A3B-Instruct-2507<br>非思考 / choice</td><td align="right">98.34</td><td align="right">96.10</td><td align="right">4.43</td><td align="right">7.50</td><td align="right">99.01</td><td align="right">14.60</td><td align="right">209.58</td><td align="right">594.72</td><td align="right">3388.97</td></tr>
<tr><td align="left">Qwen3.8-27B<br>非思考 / choice</td><td align="right">72.93</td><td align="right">60.03</td><td align="right">63.86</td><td align="right">77.78</td><td align="right">71.29</td><td align="right">5.65</td><td align="right">646.47</td><td align="right">1962.09</td><td align="right">10311.25</td></tr>
</tbody></table>

从 choice 模式的结果看，**D3-mini 在答案保留、过滤效果和耗时之间取得了较好的平衡**。它保留了 87.29% 的 L3 片段，过滤了 63.62% 的 L0/L1，平均每题候选从 15.50 条减至 7.49 条，单次请求的平均延迟为 77.82 ms。

Clef-Flash 的过滤更严格，L0/L1 过滤率为 67.84%，但 L3 保留率也降至 83.98%，平均请求延迟为 83.50 ms。D3-flash 的两项质量指标介于 D3-mini 和 Clef-Flash 之间，平均请求延迟为 89.43 ms，略高于前两者。在这项测试中，更大的模型规模没有带来更好的综合表现。

PPLX-Decider 保留了更多答案片段，L3 保留率达到 93.37%；不过，它采用双卡部署，平均请求延迟为 329.76 ms，约为 D3-mini 的 4.2 倍。其优势在于减少答案遗漏，而不是筛选速度或过滤效果。

Qwen3-30B-A3B 虽然 L3 保留率达到 98.34%，平均每题仍保留 14.60 条，实际排除的内容很少。相比之下，D3-mini 和 Clef-Flash 在保留大部分答案片段的同时，也过滤了更多无效内容。

## 测试说明

两项测试均使用 A100-PCIE-40GB。27B 模型及 LLM 基线采用双卡 TP=2，其余使用单卡，并采用各自对应的部署引擎。
