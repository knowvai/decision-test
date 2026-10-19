"""Build a standalone HTML summary from the completed audit; no inference or rescoring."""
import argparse
import base64
import hashlib
from html import escape
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_SHA256 = "ad5840e5c8458d86ebbddaa3165b504616a3b49b28ad2d3dae6602820c7461ed"
GROUPS = [
    ("小于 1B", [("d3-edge", "D3-edge"), ("vela-2.0-0.3b", "Vela 2.0-0.3B"),
                ("vela-2.0-0.8b", "Vela 2.0-0.8B"), ("laya-multilingual", "Laya Multilingual")]),
    ("4B", [("d3-mini", "D3-mini"), ("vela-2.0-4b", "Vela 2.0-4B"),
            ("neohorse-jev-4b", "NeoHorse-Jev-4B"), ("jeva-4b", "Jeva-4B")]),
    ("9B", [("d3-flash", "D3-flash"), ("clef-flash", "Clef-Flash"), ("vela-2.0-9b", "Vela 2.0-9B")]),
    ("27B", [("jev-27b", "JEV-27B"), ("pplx-decider-v1.1-27b", "PPLX-Decider-v1.1-27B")]),
    ("LLM 基线", [("qwen3-30b-a3b-instruct-2507", "Qwen3-30B-A3B-Instruct-2507"),
                  ("qwen3.8-27b", "Qwen3.8-27B")]),
]


class ReferenceStyle(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "style":
            self.inside = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


def configurations(data):
    for group, models in GROUPS:
        for model, name in models:
            for mode in ("choice", "noul"):
                if mode + ".direct" in data["models"][model]["phases"]["formal"]:
                    yield group, model, name, mode


def phase(data, model, mode, arm):
    return data["models"][model]["phases"]["formal"][mode + "." + arm]


def pair(data, model, mode):
    key = f"{model}/{mode}.direct -> {model}/{mode}.top10"
    return data["paired_comparisons"][key]["formal"]


def number(value, scale=1, signed=False):
    return f"{value * scale:+.2f}" if signed else f"{value * scale:.2f}"


def interval(value, signed=False):
    return (number(value["point"], 100, signed) +
            f'<span class="ci">[{number(value["low"], 100)}, {number(value["high"], 100)}]</span>')


def model_label(name, mode, model):
    label = "非思考 / choice" if model.startswith("qwen") else mode
    return f'{escape(name)}<span class="mode">{escape(label)}</span>'


def colored(value, text):
    style = "positive" if value > 0 else "negative" if value < 0 else ""
    return f'<span class="{style}">{text}</span>'


def table(headers, rows, table_id, label):
    out = [f'<div class="table-wrap" role="region" aria-label="{escape(label)}" tabindex="0">',
           f'<table id="{table_id}"><thead><tr>']
    out += [f'<th scope="col">{h}</th>' for h in headers]
    out.append('</tr></thead>')
    for group, _ in GROUPS:
        selected = [r for r in rows if r[0] == group]
        if not selected:
            continue
        out.append('<tbody>')
        for i, (_, key, cells) in enumerate(selected):
            out.append(f'<tr data-key="{escape(key, quote=True)}">')
            if i == 0:
                out.append(f'<th scope="rowgroup" rowspan="{len(selected)}">{escape(group)}</th>')
            out.extend(f'<td>{cell}</td>' for cell in cells)
            out.append('</tr>')
        out.append('</tbody>')
    return "\n".join(out + ['</table></div>'])


def time_chart(data, configs):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = [f"{name} / {mode}" for _, _, name, mode in configs]
    values = [100 * (1 - 1 / pair(data, model, mode)["ratio_of_summed_paired_times"])
              for _, model, _, mode in configs]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, 8.6), dpi=150)
    ax.barh(range(len(values)), values, color=["#23735e" if v > 0 else "#a63a45" for v in values], height=.64)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(min(values) - 24, max(values) + 19)
    ax.axvline(0, color="#667277", linewidth=.8)
    ax.set_xlabel("Total chain wall time saved (%) · Top10 vs. direct", labelpad=12)
    ax.xaxis.grid(True, color="#e7ebed")
    ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(value + (1.7 if value >= 0 else -1.7), i, f"{value:+.2f}%",
                va="center", ha="left" if value >= 0 else "right", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.subplots_adjust(left=.32, right=.98, top=.99, bottom=.08)
    image = BytesIO()
    fig.savefig(image, format="png", facecolor="white", metadata={"Software": "Answer-evidence audit summary"})
    plt.close(fig)
    return base64.b64encode(image.getvalue()).decode("ascii")


def render(data, css, view="direct", chart_image=None):
    assert view in {"direct", "reranker"}
    appendix = view == "reranker"
    assert data["passed"] and data["audit_policy"]["allow_online_reranker_variation"]
    configs = list(configurations(data))
    assert len(data["models"]) == 15 and len(configs) == 28 and data["calls"] == 195580
    assert set(data["models"]) == {m for _, models in GROUPS for m, _ in models}
    direct_rows, top_rows, extra_rows = [], [], []
    fast, improved = 0, 0
    for group, model, name, mode in configs:
        direct, top = (phase(data, model, mode, arm) for arm in ("direct", "top10"))
        delta = pair(data, model, mode)
        assert delta["paired_valid_pools"] == 197
        for report in (direct, top):
            assert report["requests"]["valid"] == report["requests"]["attempted"] == report["planned_requests"]
            assert report["requests"]["failed"] == report["requests"]["timeouts"] == 0
            assert report["quality"]["valid_pools"] == 197
        label = model_label(name, mode, model)
        key = model + "/" + mode
        q = direct["quality"]["positive_pool_macro"]
        request = direct["stages"]["refine"]["all_attempt_latency_ms"]
        direct_rows.append((group, key, [label, number(q["precision"]["point"], 100), number(q["recall"]["point"], 100),
            interval(q["f1"]), number(request["mean_ms"]), number(request["p99_ms"]), number(direct["pool_wall_ms"]["mean_ms"])]))
        q = top["quality"]["positive_pool_macro"]
        saved = 1 - 1 / delta["ratio_of_summed_paired_times"]
        fast += saved > 0
        improved += delta["f1"]["point"] > 0
        top_rows.append((group, key, [label, number(q["precision"]["point"], 100), number(q["recall"]["point"], 100),
            interval(q["f1"]), colored(delta["f1"]["point"], interval(delta["f1"], True)),
            number(top["pool_wall_ms"]["mean_ms"]), colored(saved, number(saved, 100, True) + "%"),
            number(delta["ratio_of_summed_paired_times"])]))
        original = data["original_denominator_zero_credit"][model]["formal"]
        both = lambda fn: " / ".join(number(fn(v), 100) for v in (direct, top))
        adjusted = " / ".join(number(original[mode + "." + arm]["coverage_adjusted_zero_credit"]["positive_pool_macro"]["f1"]["point"], 100)
                              for arm in ("direct", "top10"))
        extra_cells = [label, both(lambda v: v["quality"]["candidate_accuracy"]),
            both(lambda v: v["quality"]["correct_empty_rate"]["point"]), both(lambda v: v["quality"]["missed_all_evidence_rate"]["point"]),
            adjusted, str(top["positive_omissions"]["reranker_pruned_positive"]), str(top["positive_omissions"]["refiner_rejected_positive"])]
        if not appendix:
            quality = direct["quality"]
            extra_cells = [label, number(quality["candidate_accuracy"], 100),
                number(quality["correct_empty_rate"]["point"], 100),
                number(quality["missed_all_evidence_rate"]["point"], 100),
                number(original[mode + ".direct"]["coverage_adjusted_zero_credit"]["positive_pool_macro"]["f1"]["point"], 100),
                str(quality["counts"].get("fp", 0)), str(quality["counts"].get("fn", 0))]
        extra_rows.append((group, key, extra_cells))
    direct_table = table(["分组", "模型 / 模式", "Precision<br>(%)", "Recall<br>(%)", "F1 (%)<br><small>[95% 置信区间]</small>",
                          "平均延迟<br>(ms)", "P99<br>(ms)", "每个问题<br>平均耗时 (ms)"], direct_rows, "direct-table", "片段筛选结果，28 个模式")
    top_table = table(["分组", "模型 / 模式", "Precision<br>(%)", "Recall<br>(%)", "F1 (%)<br><small>[95% CI]</small>",
                       "ΔF1 (pp)<br><small>[95% CI]</small>", "每个问题<br>平均耗时 (ms)", "整链耗时<br>节省", "速度比"], top_rows, "top10-table", "在线 Top10 后筛选，28 个模式")
    error_headers = ["Top10 裁掉<br>正例数", "精筛拒绝<br>正例数"] if appendix else ["误收 FP", "漏收 FN"]
    extra_table = table(["分组", "模型 / 模式", "Accuracy<br>(%)", "无证据问题<br>全拒绝率 (%)", "有证据问题<br>全部漏选率 (%)", "计入未覆盖问题<br>F1 (%)",
                         *error_headers], extra_rows, "extra-table", "误收、漏选与未覆盖问题统计")
    mini = phase(data, "d3-mini", "choice", "direct")
    pplx = phase(data, "pplx-decider-v1.1-27b", "noul", "direct")
    qwen = phase(data, "qwen3-30b-a3b-instruct-2507", "choice", "direct")
    qwen_top = phase(data, "qwen3-30b-a3b-instruct-2507", "choice", "top10")
    qwen_delta = pair(data, "qwen3-30b-a3b-instruct-2507", "choice")
    baseline_f1 = data["baselines"]["formal"]["all_keep"]["positive_pool_macro"]["f1"]["point"]
    qmetric = lambda r, k: number(r["quality"]["positive_pool_macro"][k]["point"], 100)
    laya = [phase(data, "laya-multilingual", mode, "direct") for mode in ("choice", "noul")]
    vela = [phase(data, "vela-2.0-0.3b", mode, "direct") for mode in ("choice", "noul")]
    empty_correct = [sum(c["correct_empty"] for c in phase(data, model, "choice", "direct")["quality"]["cases"])
                     for model in ("d3-mini", "d3-flash", "clef-flash")]
    title = "附录：固定 Top10 流程测试对照" if appendix else "B 场景：片段精筛测试结果汇总"
    scope = "28 个精筛模式 / 56 个链路条件" if appendix else "28 个直接精筛模式"
    comparison_intro = '''<p><strong>Top10 后筛选</strong>先对一个问题的全部候选片段评分，再对前10个片段精筛。
    裁掉的标注正例也计为漏选。本次只测试固定Top10，所有请求依次执行，没有比较其他TopN或并发配置。</p>
    <p><a href="2026-10-19-answer-evidence-summary.html">返回片段筛选主报告</a></p>''' if appendix else ""
    top_section = f'''<section id="top10" aria-labelledby="top10-title">
    <h2 id="top10-title">二、在线 Reranker Top10 后筛选</h2>
    <p>每模式正式执行 197 次在线 rerank + 1,954 次精筛，全部有效；精筛次数减少 36.02%，但另有排序成本。
    ΔF1 为 Top10 减直接筛选，单位为百分点（pp）；耗时节省为 1 − Top10 总时长 / 直接总时长，
    <strong>正值才表示节省</strong>。速度比 = 直接总时长 / Top10 总时长，大于 1 才是加速。两种方法均完整测试了相同的197个问题。</p>
    {top_table}</section>''' if appendix else ""
    comparison_conclusion = f'''<li><strong>固定 Top10 在本次串行负载下没有普遍收益。</strong>
    28 个配对中，F1 点估计 {improved} 个上升、{28-improved} 个下降，Recall 全部下降；
    整链耗时 {fast} 个缩短、{28-fast} 个变长。每模式有207～208个标注正例在排序阶段被裁掉。</li>''' if appendix else f'''<li>
    <strong>误收比漏选更多。</strong>D3-mini 的 choice 模式保留了{mini['quality']['counts']['tp']}个标注正例，
    同时误收{mini['quality']['counts']['fp']}个负例，漏选{mini['quality']['counts']['fn']}个正例。
    保留的片段中有一半以上是标注负例，这是精确率较低的直接原因。
    其片段判断准确率为{number(mini['quality']['candidate_accuracy'], 100)}%，与按问题平均的F1不是同一指标。</li>'''
    qwen_comparison = f'''加入 Top10 后 F1 为 {qmetric(qwen_top, 'f1')}%、Recall 为 {qmetric(qwen_top, 'recall')}%，
    整链耗时节省 {number(1-1/qwen_delta['ratio_of_summed_paired_times'], 100)}%。''' if appendix else ""
    figure = ""
    if appendix:
        chart = chart_image if chart_image is not None else time_chart(data, configs)
        figure = f'''<figure><div class="chart-wrap" role="region" aria-label="28 个模式的整链耗时节省图" tabindex="0">
        <img width="1800" height="1290" src="data:image/png;base64,{chart}"
        alt="Top10 对比直接筛选的整链耗时节省，28 个模式中10个缩短、18个变长；数值见前方表格。"></div>
        <figcaption>完整筛选相同197个问题的实测总时长。正值表示节省，负值表示变慢；两种方法使用的总硬件资源不同。</figcaption></figure>'''
    direct_reports = [c for m in data["models"].values() for conditions in m["phases"].values()
                      for name, c in conditions.items() if name.endswith(".direct")]
    direct_calls = sum(c["requests"]["attempted"] for c in direct_reports)
    execution = ("完整实验共195,580次有效调用，其中正式测试145,740次。" if appendix else
                 f"直接筛选共{direct_calls:,}次有效调用，其中正式测试85,512次，其余为工程检查、验证和额外38个无证据问题的测试。")
    variation_note = '''<p>同一输入的在线分数允许自然波动；每次仍按当次分数及原始顺序同分规则选出 Top10。
    对比各测试阶段同一问题的首次调用，共274次仅顺序变化、14次入选片段变化，后者涉及6个问题；正式测试为8次、涉及4个问题。
    原版跨调用一致性审计失败记录保留；修订经用户确认，未重跑推理、修改输出或调参。
    不同模式的入选片段可能不同，因此这里比较的是两步筛选的整体结果。</p>''' if appendix else ""
    resource_note = '''reranker额外使用代理机A100并与业务embedding共享GPU；BF16/TP1、8192上下文、
    0.80显存预算、4 CPU/32 GiB、prefix cache off。业务流量不属评测n=1，不宣称等总资源加速。''' if appendix else ""
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="audit-sha256" content="{AUDIT_SHA256}">
  <title>{title}</title>
  <style>
{css}
    main {{ max-width: 1320px; }}
    h1, h2, h3 {{ overflow-wrap: anywhere; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; font-weight: 650; }}
    h3 {{ font-size: 16px; }}
    p {{ line-height: 1.8; }}
    strong {{ color: #202529; }}
    a {{ color: #235a83; text-underline-offset: 3px; }}
    a:hover {{ color: #143e60; }}
    :focus-visible {{ outline: 2px solid #235a83; outline-offset: 3px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 0 24px; border-block: 1px solid #d5dbde; margin: 20px 0; }}
    nav a {{ display: inline-flex; min-height: 44px; align-items: center; }}
    table {{ min-width: 1020px; font-size: 14px; }}
    th, td {{ padding: 9px 11px; }}
    tbody td {{ white-space: nowrap; }}
    tbody td:first-of-type {{ min-width: 190px; max-width: 260px; white-space: normal; }}
    tbody th {{ width: 82px; }}
    tbody tr:hover td {{ background: #f7f9fa; }}
    small, .ci, .mode {{ color: #586367; font-size: 12px; font-weight: 400; }}
    .ci, .mode {{ display: block; line-height: 1.6; }}
    .mode {{ font-size: 13px; }}
    .positive {{ color: #206b56; }}
    .negative {{ color: #9d3341; }}
    .note {{ border-left: 3px solid #849794; padding-left: 14px; }}
    .results li {{ margin-bottom: 14px; }}
    figure {{ margin: 24px 0; }}
    .chart-wrap {{ overflow-x: auto; }}
    figure img {{ display: block; width: 100%; min-width: 860px; height: auto; }}
    figcaption {{ color: #596267; margin-top: 8px; line-height: 1.7; }}
    details {{ border-block: 1px solid #d5dbde; margin-top: 28px; padding-bottom: 4px; }}
    summary {{ padding: 14px 0; font-size: 16px; font-weight: 650; cursor: pointer; }}
    footer {{ margin: 32px 0 18px; padding-top: 16px; border-top: 1px solid #d5dbde; color: #596267; }}
    footer p {{ overflow-wrap: anywhere; font-size: 13px; }}
    @media (max-width: 600px) {{ nav {{ column-gap: 20px; }} th, td {{ padding: 8px 10px; }} }}
    @media print {{ nav, details, .chart-wrap {{ display: none; }} table {{ min-width: 0; font-size: 10px; }}
      .ci, .mode, small {{ font-size: 9px; }} tbody td:first-of-type {{ min-width: 130px; }}
      .table-wrap {{ overflow: visible; }} h2 {{ break-after: avoid; }} a {{ color: inherit; }} }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>2026-10-19 · 15 款模型 / {scope} · 每模式 3,054 条正式数据</p>
  <p>本次使用 <strong>T2Ranking 数据集中的197个问题、3,054个候选片段</strong>。
  每次输入一个问题和一个完整片段，判断片段是否提供了能够部分或完整回答问题的信息。
  按原始标签，L2/L3为有证据（924条），L0/L1为无证据（2,130条）。
  13款Decision模型分别测试choice、noul两种模式，两款LLM只测试非思考的choice模式；所有请求依次执行，不重试、不截断、不调整判定阈值。</p>
  <p><strong>choice（选项选择）</strong>：明确给出“有答案证据”和“无答案证据”两个选项，由模型选择。
  <strong>noul（是/否判断）</strong>：不给这两个自定义选项，而是使用模型原生的true/false判断，
  true对应有答案证据，false对应无答案证据；部分模型返回的是true的概率。</p>
  <p><strong>为什么比较两种模式？</strong>同一款模型使用相同的问题、完整片段和判断说明，
  换一种判定方式，结果和耗时也可能不同。本次同时测试两种模式，是为了比较各模型用哪种方式筛得更准、耗时更少。
  choice和noul不是两款模型，也不是思考与非思考的区别。</p>
  <p><strong>Precision（精确率）</strong>是保留片段中标注为有证据的比例；
  <strong>Recall（召回率）</strong>是所有标注证据中被保留的比例；<strong>F1</strong>是精确率与召回率的调和平均。
  三项指标先按每个问题计算，再对183个有标注证据的问题取平均。另14个问题没有标注证据，单独统计。
  方括号内为F1的95%置信区间。</p>
  <p>耗时单位为毫秒（ms）。<strong>平均延迟、P99</strong>对应一次片段判断，P99表示约99%的请求耗时不超过该值；
  <strong>每个问题平均耗时</strong>指筛完该问题全部候选片段的总用时再取平均。
  耗时包含通信、推理和结果检查，不含模型加载。
  27B模型及两款LLM使用两张A100-PCIE-40GB（TP=2），其余模型使用单卡。</p>
  {comparison_intro}
  <section id="direct" aria-label="直接片段筛选结果">
    {'<h2>一、直接筛选</h2>' if appendix else ''}
    {direct_table}
  </section>
  {top_section}
  <section id="conclusions" class="results" aria-labelledby="conclusions-title">
    <h2 id="conclusions-title">测试结果总结</h2>
    <ol>
      <li><strong>PPLX 的 F1 最高，D3-mini 耗时更短。</strong>
      PPLX 的noul模式F1为{qmetric(pplx, 'f1')}%，D3-mini的choice模式为{qmetric(mini, 'f1')}%。
      完成一个问题全部片段的筛选，平均分别需要{number(pplx['pool_wall_ms']['mean_ms'])} ms和{number(mini['pool_wall_ms']['mean_ms'])} ms。
      D3-mini使用单卡，PPLX使用双卡。两者F1接近，目前不能认定差异显著。</li>
      {comparison_conclusion}
      <li><strong>Qwen3-30B 召回率高，但没有滤掉足够多的负例。</strong>Recall为{qmetric(qwen, 'recall')}%，
      Precision为{qmetric(qwen, 'precision')}%，F1为{qmetric(qwen, 'f1')}%。
      不做筛选、全部保留的F1为{number(baseline_f1, 100)}%，与它的结果接近。{qwen_comparison}</li>
      <li><strong>noul并不总比choice好。</strong>Laya改用noul后，F1从{qmetric(laya[0], 'f1')}%升至{qmetric(laya[1], 'f1')}%；
      Vela 0.3B则从{qmetric(vela[0], 'f1')}%降至{qmetric(vela[1], 'f1')}%。模式选择需要分别看各模型的结果。</li>
      <li><strong>没有可用证据时，模型仍容易保留片段。</strong>在14个全部标为无证据的问题中，
      D3-mini、D3-flash、Clef-Flash的choice模式分别只有{empty_correct[0]}、{empty_correct[1]}、{empty_correct[2]}个问题做到了全部拒绝。
      这部分只有14个问题，结果单独列在补充表中。</li>
    </ol>
    {figure}
  </section>
  <details id="supplement">
    <summary>补充数据与测试说明</summary>
    <p>{'前四个数值列按“直接 / Top10”排列。' if appendix else ''}Accuracy是3,054个片段中判断正确的比例。
    “全拒绝率”统计14个无证据问题中，有多少没有保留任何片段；“全部漏选率”统计183个有证据问题中，有多少没有保留任何正例。
    {'最后两列分别是Top10裁掉、精筛拒绝的正例数。' if appendix else 'FP是误收的负例数，FN是漏选的正例数。'}</p>
    <p>原定测试200个问题，其中3个问题因部分片段过长、无法被所有模型完整处理而统一排除，共51条片段（含11个正例）。
    “计入未覆盖问题F1”将未覆盖的有证据问题按0分计入平均值；主表只比较完整测试的197个问题。
    所有得分均按T2Ranking原始标签计算，没有重新人工标注。</p>
    <p>noul优先采用模型原生的是/否结果；仅返回概率时，选择true、false中概率较大的一项，相等时判为无证据。
    这一规则事先固定，没有根据测试结果调整阈值。</p>
    {extra_table}
  <section id="audit" aria-labelledby="audit-title">
    <h2 id="audit-title">测试记录</h2>
    <p>{execution}
    已纳入测试的数据均完成调用，失败和超时为0。逐请求检查及修订后的独立审计已通过，原审计退出和修订记录仍保留。</p>
    {variation_note}
    <p>单次请求计时不含调用前后写入记录的时间；每个问题的总耗时包含{'排序、' if appendix else ''}全部精筛请求及其记录写入，
    不含模型加载、问题之间的服务检查和问题汇总记录写入。
    置信区间按问题重复抽样计算，未作多重比较校正。{resource_note}</p>
  </section>
  </details>
  <footer>
    <p>来源：<a href="2026-10-19-answer-evidence-results.md">完整结果表与各阶段统计</a> ·
    <a href="2026-10-19-answer-evidence-results.json.gz">独立审计 JSON</a> ·
    <a href="2026-10-18-answer-evidence-execution.md">执行记录与审计修订</a> ·
    <a href="2026-10-19-answer-evidence-reranker-appendix.html">Reranker 测试附录</a> ·
    <a href="2026-10-16-decision-classification-summary.html">分类场景汇总</a></p>
  </footer>
</main>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "2026-10-19-answer-evidence-results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "2026-10-19-answer-evidence-summary.html")
    parser.add_argument("--view", choices=("direct", "reranker"), default="direct")
    parser.add_argument("--replace", action="store_true", help="Replace only an existing generated HTML report")
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == AUDIT_SHA256, "Unexpected audit snapshot"
    style = ReferenceStyle()
    style.feed((ROOT / "2026-10-16-decision-classification-summary.html").read_text())
    assert style.parts
    assert args.output.suffix == ".html", "Only generated HTML output is supported"
    page = render(json.loads(raw), "\n".join(style.parts), args.view)
    with args.output.open("w" if args.replace else "x", encoding="utf-8") as stream:
        stream.write(page)
    print(json.dumps({"output": str(args.output), "view": args.view, "conditions": 56 if args.view == "reranker" else 28,
                      "new_inference_calls": 0, "audit_sha256": AUDIT_SHA256}))


if __name__ == "__main__":
    main()
