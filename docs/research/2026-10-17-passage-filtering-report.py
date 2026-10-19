"""Render public tables from the independent offline audit, without remote access."""
import argparse
import json
from pathlib import Path


def ci(value, scale=100, digits=2):
    return f'{value["point"] * scale:.{digits}f} [{value["low"] * scale:.{digits}f}, {value["high"] * scale:.{digits}f}]'


def render(data):
    assert data["passed"] and data["proxy"]["passed"]
    models = data["models"]
    lines = ["# 相关片段筛选四链路：完整结果与独立审计", "",
             "## 结论与范围", "",
             "2026-10-17 上海 06:25:14 启动，15:16:40 完成。15 个精筛配置、30 个链路条件全部完成独立离线复核。",
             "正式共 97,815 次 HTTP 请求，加工程 3,390 次、验证 4,995 次，总计 106,200 次；全部有效，失败与超时均为 0。6,000 条正式链路全部有效。",
             "主分母为共同修订数据：200 池、3,188 个完整候选、1,437 正例及 1,751 负例。仅移除预先批准的 6 个片段（1 正例、5 负例），原始 3,194 候选及旧证据不变。",
             "B1/B2 是全候选 Decision/非思考 LLM；B3/B4 是在线 Qwen3-Reranker-8B Top29 后的同一精筛器。不同模型之间不混合引擎、资源和质量构造总冠军。", ""]

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        lines.append("")

    ranked = sorted(((v["all_pools"]["arms"][v["all_pools"]["paired"]["direct"]]["quality"]["f1"]["point"], k)
                     for k, v in models.items()), reverse=True)
    lines.append(f"直接精筛的最高 F1 点估计为 `{ranked[0][1]}`：{ranked[0][0] * 100:.2f}%。这只是本轮点估计，不是显著胜出或生产推荐。各模型的区间、延迟及资源如下。")
    ratios = [v["all_pools"]["paired"]["ratio_of_summed_paired_times"] for v in models.values()]
    lines.append(f"全 200 池的前置链路实测时间比（直接/前置）范围 {min(ratios):.3f}–{max(ratios):.3f}；大于 1 才表示前置更快，不能用减少的候选数直接声称加速。")
    first = next(iter(models.values()))
    retain = first["all_pools"]["retention"]
    lines.extend([f"Top29 成员在全部 15 个配置间一致：3,188 → 3,133 候选，只减少 {55 / 3188 * 100:.2f}% 的精筛调用；剪去 8 个正例与 47 个负例。池均正例保留率 {retain['macro_positive_retention'] * 100:.3f}%，候选加权保留率 {1429 / 1437 * 100:.3f}%。", "",
                  "95% 区间采用冻结统计函数的池级非参数 bootstrap，2,000 次，固定随机种子；同模型前后差值使用相同池配对。区间包含零不表示等效；小切片区间和 P99 仅作描述，不做多重比较后的显著排名。", "",
                  "## 表一：30 条件质量与实测整链路", "",
                  "质量单位为 %，方括号为 95% 区间；延迟单位为秒。所有行成功率均为 200/200（100%）。质量按所有 200 池等权平均，不对保留子集重设 Recall 分母。", ""])
    rows = []
    for name, model in models.items():
        for arm, value in model["all_pools"]["arms"].items():
            q, t = value["quality"], value["all_chain_latency_ms"]
            rows.append([name, arm, ci(q["precision"]), ci(q["recall"]), ci(q["f1"]),
                         f'{t["mean_ms"]/1000:.3f}', f'{t["p95_ms"]/1000:.3f}', f'{t["p99_ms"]/1000:.3f}'])
    table(["精筛配置", "链路", "Precision [95% CI]", "Recall [95% CI]", "F1 [95% CI]", "mean s", "P95 s", "P99 s"], rows)
    lines.extend(["## 表二：同模型前置收益与损失", "",
                  "Δ 为前置减直接，质量单位为百分点；节省为直接减前置的池均实测 wall time，单位为秒，正值才是节省。比值为同一成功配对池的直接总时间/前置总时间，不是池比值的均值。每行 200 个成功配对池。", ""])
    rows = []
    for name, model in models.items():
        pair = model["all_pools"]["paired"]
        q = pair["quality_delta"]
        rows.append([name, ci(q["precision"]), ci(q["recall"]), ci(q["f1"]),
                     ci(pair["time_saved_ms"], .001, 3), f'{pair["ratio_of_summed_paired_times"]:.3f}',
                     model["retained_pair_decision_changes"]])
    table(["精筛配置", "ΔPrecision pp", "ΔRecall pp", "ΔF1 pp", "节省 s [95% CI]", "时间比", "保留项判断变化数"], rows)
    lines.extend(["保留项判断变化数比较同一候选在直接/后置精筛中的最终判断，不把重复调用的数值或生成差异归因于裁剪。裁剪损失仍以原修订标签计入 FN。", "",
                  "## 表三：Top29 保留率与池规模", "",
                  "切片按掩码前原始池候选数划分；保留率的正例分母仍采用共同修订标签。成员和损失在全部配置间一致，表内只计一次，不能把同一 200 池重复 15 次当作独立样本。", ""])
    rows = []
    for label, report in [("全部", first["all_pools"])] + list(first["slices"].items()):
        r = report["retention"]
        n = r["all_pools"]
        rows.append([label, n, r["original_candidates"], r["revised_candidates"], r["retained_candidates"],
                     f'{r["revised_candidates"]/n:.3f} → {r["retained_candidates"]/n:.3f}',
                     f'{r["macro_positive_retention"]*100:.3f}%',
                     f'{r["pruned_positive_candidates"]}/{r["positive_denominator_on_valid_pools"]}'])
    table(["原池切片", "池数", "原候选", "修订候选", "Top29 候选", "池均候选前→后", "池均正例保留率", "剪去正例/正例"], rows)
    lines.extend(["### 原池切片质量", "", "每个切片均报告全部池，不按成功或质量筛选。质量为 % [95% CI]；小切片不用于确定性排名。", ""])
    rows = []
    for name, model in models.items():
        for label, report in model["slices"].items():
            for arm, value in report["arms"].items():
                q, t = value["quality"], value["all_chain_latency_ms"]
                rows.append([name, label, arm, value["pools"], ci(q["precision"]), ci(q["recall"]), ci(q["f1"]), f'{t["mean_ms"]/1000:.3f}'])
    table(["精筛配置", "原池切片", "链路", "池数", "Precision", "Recall", "F1", "整链路 mean s"], rows)
    lines.extend(["### 原池切片配对时间", "", "每行使用该切片相同成功池的实测时间。reranker mean/P95 为该配置实际在线请求，不复用首个模型计时。", ""])
    rows = []
    for name, model in models.items():
        for label, report in model["slices"].items():
            pair = report["paired"]
            t = report["stages"][pair["pipeline"] + ":rerank"]["all_attempt_latency_ms"]
            rows.append([name, label, pair["timed_successful_pairs"], ci(pair["quality_delta"]["f1"]),
                         ci(pair["time_saved_ms"], .001, 3), f'{pair["ratio_of_summed_paired_times"]:.3f}',
                         f'{t["mean_ms"]/1000:.3f}/{t["p95_ms"]/1000:.3f}'])
    table(["精筛配置", "原池切片", "配对池", "ΔF1 pp", "节省 s [95% CI]", "时间比", "rerank mean/P95 s"], rows)
    lines.extend(["## 表四：n=1 单请求阶段基线", "",
                  "阶段请求单位不可混比：refine 是一个问题加一个完整候选，rerank 是一个问题加该池全部修订候选。直接 3,188 对、Top29 子集 3,133 对分别统计。全尝试均值本轮等于有效均值（全部有效），全尝试总耗时为各请求 elapsed_ms 的和，不是整链路实测。", ""])
    rows = []
    for name, model in models.items():
        for stage, value in model["all_pools"]["stages"].items():
            t = value["all_attempt_latency_ms"]
            rows.append([name, stage, f'{value["valid"]}/{value["attempted"]}', f'{value["failed"]}/{value["timeouts"]}',
                         f'{t["mean_ms"]:.2f}', f'{t["p95_ms"]:.2f}', f'{t["p99_ms"]:.2f}',
                         f'{value["all_attempt_total_ms"]/1000:.3f}'])
    table(["精筛配置", "链路:阶段", "有效/尝试", "失败/超时", "有效及全尝试 mean ms", "P95 ms", "P99 ms", "全尝试总 s"], rows)
    lines.extend(["## 工程、验证与冻结阈值", "",
                  "每模型先完成 12 工程池的 226 个完整对，再完成独立 20 验证池的 333 对，随后冻结阈值并一次执行正式两链路。阈值最大化验证 macro F1，同分取 Precision 再取更高阈值，使用 >=；两 LLM 使用严格 JSON 硬判断，不构造概率。下列工程/验证均为有效/尝试，全尝试 mean/P95/P99 与总时间完整记录于审计 JSON。", ""])
    rows = []
    for name, model in models.items():
        rule = model["threshold"]
        rows.append([name, f'{rule["threshold"]:.9g}' if rule["kind"] == "threshold" else "hard_choice",
                     "226/226", "333/333", f'{model["engineering"]["all_attempt_total_ms"]/1000:.3f}',
                     f'{model["validation"]["all_attempt_total_ms"]/1000:.3f}'])
    table(["精筛配置", "冻结规则", "工程", "验证", "工程总 s", "验证总 s"], rows)
    lines.extend(["## 日志与计时边界", "",
                  "整链路 wall_ms 是 runner 在每条在线链路开始前至完成选集/元数据构造时的实测计时，包含网络、鉴权、reranker、精筛和每请求意图/结果的同步落盘。未用阶段和替代。",
                  "该计时不含随后链路行自身的 fsync、每池后的容器身份检查、模型装载/JIT、工程、验证或最终审计；不是整轮运行耗时。单请求 elapsed_ms 从构造/哈希到 HTTP/解析/校验，扣除意图落盘且不含结果写盘。网络/鉴权没有独立计时器，不能拆成虚构的纯 GPU 时间。",
                  "以下日志为每池内所有请求的意图与结果写盘耗时；占比为该臂日志总时间/实测链路总时间。", ""])
    rows = []
    for name, model in models.items():
        for arm, value in model["all_pools"]["arms"].items():
            journal = value["request_journal_ms"]
            rows.append([name, arm, f'{journal["mean_ms"]:.2f}', f'{journal["p95_ms"]:.2f}',
                         f'{journal["total_ms"]/1000:.3f}', f'{journal["total_ms"]/value["all_attempt_total_ms"]*100:.2f}%',
                         f'{value["all_attempt_total_ms"]/1000:.3f}'])
    table(["精筛配置", "链路", "日志 mean ms", "日志 P95 ms", "日志总 s", "日志占 wall", "链路全尝试总 s"], rows)
    lines.extend(["## 部署与资源", "",
                  "精筛器每配置前后链路共享同一容器身份、镜像、提示、阈值和预算；全局评测 HTTP 峰值 1，无池/阶段请求重叠。工程与验证在正式之前完成，正式不是冷启动测试。",
                  "测试机为双 A100-PCIE-40GB；每个精筛服务限制 8 CPU / 96GiB RAM，TP 见表。原生后端内部调度不等于 HTTP 并发；不能把不同引擎性能归为模型本体因果优势。", ""])
    rows = []
    for name, model in models.items():
        dep = model["deployment"]
        s = dep["scheduling"]
        rows.append([name, dep["family"], dep["tp"], f'{s.get("max_sequences")}/{s.get("max_batched_tokens")}',
                     dep["image_id"].removeprefix("sha256:")[:12]])
    table(["精筛配置", "后端 family", "TP", "原生 sequences/tokens", "镜像 SHA 前12位"], rows)
    lines.extend(["`worker` 为 vllm-srun 原生服务；`sglang` 为 SGLang；`jev` 为作者 serve_decide + LoRA；`neohorse` 为作者 launch.py + adapter；`sdk` 为 Jeva SDK；`laya` 为原生 Laya 适配；`llm` 为原生 vLLM。None 表示不适用该统一批调度字段，不表示无限资源。完整命令、镜像 SHA、模型路径、资源、容器与 token usage 分布均保存在审计 JSON。",
                  "LLM 均为非思考、temperature=0、max_tokens=64、strict JSON；全部精筛请求 reasoning_chars=0 且 finish_reason=stop。原生 vLLM prefix cache 关闭，worker result cache=0/shared context关闭；其他后端依冻结原生命令，不额外假设缓存隔离。",
                  "前置 reranker 另占代理机单 A100 40GB，与业务 embedding 共享：原生 vLLM 0.31.0 pooling、BF16/TP1、8192 context、0.80 显存预算、seq16/tokens16384、4 CPU / 32GiB RAM、prefix cache关闭。未扩到16384 context，未改85%。",
                  "业务 embedding 的完整 ID、PID、启动时间均与起始基线相同；reranker 与代理身份未变，无重启或OOM。共享业务流量不属于评测 n=1，无法从本轮隔离其干扰。额外主机/GPU与跨机网络成本已包含于观测条件，不能称等总资源加速，也不能由 n=1 外推生产并发或 产品 端到端效果。", "",
                  "## 审计与保全", "",
                  "- systemd 最终 inactive/dead，Result=success、ExecMainStatus=0、MainPID=0、Restart=no；共享 FLOCK 已释放。status 的 completed_pools=199 是最后一次零基索引残留，不能解释成少一池：独立核对每配置两臂各200个唯一池和400条链路。",
                  "- 全部15份原 audit.json 的文件哈希复核通过，并重新执行冻结 audit_model 的纯离线审计。精确重建调用ID、输入/请求hash、完整输入与Top29覆盖、阈值、预测集合、质量/区间及时间统计；全部106200调用唯一，无重试。",
                  "- 同模型用单调时钟区间检查无重叠，跨模型用记录时间及请求持续时长复核间隔；链路请求分组、池顺序、奇偶先后、保留候选原顺序均一致。链路日志未保存独立绝对起止时间，链路串行还依赖冻结执行源码与连续请求消费序列，不伪称有不存在的独立链路时间戳。",
                  "- 六项掩码由原始输入/标签重新应用后与修订数据逐对象完全一致；工程/验证/正式池ID与query隔离。历史保全1163文件/140647988字节全部hash一致，代理机保全文件也通过。",
                  "- frozen.json、startup.json、reranker-frozen.json、preservation.json、数据/运行协议与只读源码重新校验；39个分类函数AST和capacity源码未变。118项测试为启动前已通过的冻结证据，本次未重测或发额外推理。",
                  "- 离线审计只读取原证据，新结果单独保存；执行前后源文件与证据hash未变。审计程序的函数总数与Docker命令表示假设经只读核对纠正（39个函数全部不变；Entrypoint与Cmd合并匹配），不是测试执行失败，未改原runner或原audit。",
                  "- 失败分母规则仍适用：若链路失败则该池质量为0并保留失败状态；本轮实际全部有效，没有用剔除失败获得成绩。",
                  "- 最终Qwen3.8模型、reranker与业务embedding继续保留；不自动开展下一阶段。", "",
                  "证据：[独立审计 JSON](2026-10-17-passage-filtering-audit.json)、[只读审计程序](2026-10-17-passage-filtering-audit.py)、[表格生成程序](2026-10-17-passage-filtering-report.py)、[执行记录](2026-10-17-passage-filtering-execution.md)、[方法](../plans/2026-10-14-decision-benchmark-plan.md)。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(render(json.loads(args.audit.read_text())))
