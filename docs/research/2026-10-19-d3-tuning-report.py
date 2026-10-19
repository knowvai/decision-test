"""Render the independently audited D3 pilot as a factual Markdown report."""
import argparse
import hashlib
import json
from pathlib import Path


def pct(value):
    return f"{value*100:.2f}"


def ci(value):
    return f"{value['point']*100:+.2f} [{value['low']*100:+.2f}, {value['high']*100:+.2f}]"


def metrics_table(data, root):
    from prometheus_client.parser import text_string_to_metric_families

    def parse(text):
        return {(sample.name, tuple(sorted(sample.labels.items()))): sample.value
                for family in text_string_to_metric_families(text) for sample in family.samples}

    out = ["## 合批与排队诊断", "",
           "以下是已保存 Prometheus 计数器的块前后增量，使用标准 parser 解析；本地副本 SHA256 与审计记录一致。",
           "展示 n8/n16 的两轮合计。模型 forward 含 readout；排队是每任务首次 forward 前等待，不等同 HTTP 总延迟。",
           "这些是解释线索，不单独证明某个底层机制导致了收益或退化。", "",
           "| 模型 | 部署 | 任务 | n | 平均每批条数 | 平均 forward ms/批 | 平均排队 ms/请求 |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for model, result in data["models"].items():
        for profile, tasks in result["performance"].items():
            for task in tasks:
                for n in (8, 16):
                    totals = {}
                    names = ("vllm_srun_batch_rows_count", "vllm_srun_batch_rows_sum",
                             "vllm_srun_forward_duration_seconds_sum", "vllm_srun_queue_duration_seconds_count",
                             "vllm_srun_queue_duration_seconds_sum", "vllm_srun_requests_total")
                    for repeat in (1, 2):
                        relative = f"{model}/{profile}/{task}-n{n}-r{repeat}.json"
                        raw = (root / relative).read_bytes()
                        assert hashlib.sha256(raw).hexdigest() == data["artifact_sha256"][relative]
                        meta = json.loads(raw)
                        before, after = (parse(meta[key]) for key in ("metrics_before", "metrics_after"))
                        delta = {name: sum(value-before.get(key, 0) for key, value in after.items() if key[0] == name)
                                 for name in names}
                        assert delta["vllm_srun_requests_total"] == meta["requests"]
                        for name, value in delta.items():
                            assert value >= 0
                            totals[name] = totals.get(name, 0) + value
                    batches = totals["vllm_srun_batch_rows_count"]
                    rows = totals["vllm_srun_batch_rows_sum"] / batches
                    forward = 1000 * totals["vllm_srun_forward_duration_seconds_sum"] / batches
                    queue = 1000 * totals["vllm_srun_queue_duration_seconds_sum"] / totals["vllm_srun_queue_duration_seconds_count"]
                    out.append(f"| {model} | {profile} | {task} | {n} | {rows:.2f} | {forward:.2f} | {queue:.2f} |")
    return out + [""]


def render(data, metrics_root=None):
    assert data["passed"]
    out = ["# D3-mini / D3-flash 调优试验结果", "",
           "本轮是独立新样本上的小规模试验，不是旧测试重跑，也没有修改模型权重、精度或原生计算。",
           "提示与阈值只由开发集选择，留出集不参与调优；仅直接筛选，不调用 reranker。", "",
           f"独立审计通过：{data['total_calls']:,} 次有效调用，{data['blocks']} 个完整块，零失败/超时；"
           f"{data['preserved_files']} 个旧证据文件保持不变。调用按阶段：`{json.dumps(data['calls'])}`。", "",
           "## 数据", "",
           "分类开发集来自固定版本 MASSIVE 中文 train，留出来自 validation；排除旧 test 和跨划分归一化重复。",
           "每类开发最多 20、留出最多 40，接近类别均衡，不等同旧测试的自然类别分布。",
           "证据使用固定 T2Ranking 原始全文与等级，排除旧测试查询，开发和留出不共享片段文本。",
           "L2/L3 为证据正例，L0/L1 为负例；这只是代理标签，不证明事实正确或最终回答质量。", "",
           "| 划分 | 分类条数 | 证据池 | 证据对 | L0/L1/L2/L3 | 空池 |",
           "| --- | ---: | ---: | ---: | --- | ---: |"]
    for split, value in data["dataset"].items():
        grades = "/".join(str(value["evidence_grades"].get(str(i), value["evidence_grades"].get(i,0))) for i in range(4))
        out.append(f"| {split} | {value['classification']} | {value['evidence_pools']} | {value['evidence_pairs']} | {grades} | {value['empty_pools']} |")
    out += ["", "## 开发集选择", "", "所有候选在推理前冻结；并列保留 baseline。",
            "分类要求 Accuracy 不低于 baseline，再最大化 macro-F1；证据要求池宏 Recall 不低于 baseline，再最大化池宏 F1。",
            "阈值只对已有原生概率离线计算，没有为阈值扫描增加请求。", "",
            "原生概率未做业务校准，阈值只适用于本轮待复验的候选方案。", "",
            "| 模型 | 任务 | 提示 | 判定方式 | 开发集主指标 % |",
            "| --- | --- | --- | --- | ---: |"]
    for model, result in data["models"].items():
        for task, q in result["quality"].items():
            s = q["selection"]
            value = s["quality"]["macro_f1"] if task == "classification" else s["quality"]["positive_pool_macro"]["f1"]["point"]
            threshold = "原生 choice" if s["threshold"] is None else f"P(有证据) >= {s['threshold']}"
            out.append(f"| {model} | {task} | {s['variant']} | {threshold} | {pct(value)} |")
    out += ["", "## 独立留出：分类", "",
            "差值为开发集选择方案减 baseline，单位百分点；95% 区间按类别分层配对 bootstrap。", "",
            "| 模型 | Accuracy 基线→方案 % | macro-F1 基线→方案 % | ΔAccuracy [95% CI] | Δmacro-F1 [95% CI] |",
            "| --- | --- | --- | --- | --- |"]
    for model, result in data["models"].items():
        q = result["quality"]["classification"]
        b, s, delta = q["baseline"], q["selected"], q["paired_delta"]["metrics"]
        out.append(f"| {model} | {pct(b['accuracy'])} → {pct(s['accuracy'])} | {pct(b['macro_f1'])} → {pct(s['macro_f1'])} | {ci(delta['accuracy'])} | {ci(delta['macro_f1'])} |")
    out += ["", "## 独立留出：答案证据", "",
            "P/R/F1 是有证据池的池级宏平均，不是候选微平均；差异区间按问题池配对抽样。", "",
            "| 模型 | 方案 | P % | R % | F1 % | 漏光率 % | 空池全拒绝 % | 保留片段 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, result in data["models"].items():
        for name in ("baseline", "selected"):
            q = result["quality"]["evidence"][name]
            v = q["positive_pool_macro"]
            empty = pct(q["correct_empty_rate"]["point"]) if q["correct_empty_rate"] else "N/A"
            out.append(f"| {model} | {name} | {pct(v['precision']['point'])} | {pct(v['recall']['point'])} | {pct(v['f1']['point'])} | {pct(q['missed_all_evidence_rate']['point'])} | {empty} | {q['kept_candidates']} |")
    out += ["", "| 模型 | ΔP pp [95% CI] | ΔR pp [95% CI] | ΔF1 pp [95% CI] |", "| --- | --- | --- | --- |"]
    for model, result in data["models"].items():
        d = result["quality"]["evidence"]["paired_delta"]
        out.append(f"| {model} | {ci(d['precision'])} | {ci(d['recall'])} | {ci(d['f1'])} |")
    out += ["", "空池仅 2 个，不能据此认定拒绝能力稳定。多个候选是在开发集选择，留出结果照实报告，不将开发集收益替代泛化收益。",
            "当选中 baseline 提示时，阈值前后复用同一批原生响应；没有伪称为两次独立模型运行。", "",
            "## 并发与运行时", "",
            "单张 A100-PCIE-40GB、8 CPU、96 GiB 容器上限，固定 readiness 镜像、BF16 backbone / FP32 readout。",
            "三个部署均独立装载后预热；模型间反转两个候选部署的次序，但不是随机多次冷启动实验。",
            "- baseline：threads2、seq32、tokens32768、batch window 2 ms。",
            "- thread1：仅 threads 与 OMP_NUM_THREADS 改为 1。",
            "- compact：threads2、seq8、tokens8192、batch window 2 ms。",
            "", "每任务 128 个开发集完整输入，每档两轮，第二轮并发档位反序；每格共 256 次尝试，独立输入仍只有 128 个。",
            "性能子集最长：分类 433 token，片段 3,765 token。完整质量留出含更长输入，不能据此声称超长片段容量。",
            "RPS=有效请求/两轮真实 wall 之和；P95/P99 对合并请求计算，不平均百分位。延迟包含服务端排队与鉴权 HTTP，",
            "不含客户端等待空闲槽位和冷启动。是闭环短程压力，不是开放流量、长期 SLA 或业务容量承诺。", ""]
    for task in ("classification", "evidence"):
        out += [f"### {task}", "", "| 模型 | 部署 | n | RPS | RPS 两轮 | Mean ms | P95 ms | P99 ms | 相同 n 吞吐比 | 样本 Accuracy % | 对基线同档判断变化 /256 |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for model, result in data["models"].items():
            for profile, tasks in result["performance"].items():
                for n, v in tasks[task].items():
                    rounds = " / ".join(f"{r['rps']:.2f}" for r in v["rounds"])
                    out.append(f"| {model} | {profile} | {n} | {v['rps']:.2f} | {rounds} | {v['mean_ms']:.2f} | {v['p95_ms']:.2f} | {v['p99_ms']:.2f} | {v['rps_ratio_vs_baseline_same_n']:.3f} | {pct(v['accuracy'])} | {v['changes_vs_baseline_same_n']} |")
        out.append("")
    if metrics_root is not None:
        out += metrics_table(data, metrics_root)
    out += ["## 审计与解释边界", "",
            "逐文件检查实际换行记录、输入/请求哈希、原生概率读出、一次调用、块内在途上限与块间无重叠，独立重算统计和开发集选择。",
            "检查选择文件早于留出请求、源文件不变、旧证据不变、原评测容器 ID/镜像/命令恢复。",
            "响应保留原生 answers/usage/meta 与完整响应哈希，未保留 HTTP 原始字节；可以重验原生答案，但不声称重建原始响应字节。",
            "代理机三只保护服务身份由外部 SSH 单独前后核验，不由纯离线审计证明。",
            "未改权重、未量化、未增加 CPU/GPU 资源、未使用结果缓存；没有运行 Clef 或重跑旧报告。",
            "公开数据是否曾进入模型训练未知；本轮也不是 产品 线上真实业务质量验收。未修改生产模型绑定。",
            "每档仅两轮，不能把请求数当作独立的性能试验次数，也不能将小幅吞吐差异直接认定为稳定提升。",
            "运行时收益目前只在固定性能子集验证，若采用新部署还需要完整长输入质量回归与更长压力测试。",
            "", "相关文件：", "",
            "- [实验方案](../plans/2026-10-19-d3-tuning-pilot.md)",
            "- [执行记录](2026-10-19-d3-tuning-execution.md)",
            "- [独立审计与完整数据](2026-10-19-d3-tuning-results.json)",
            "- [官方 D3-mini](https://huggingface.co/vllm-sr/d3-mini)",
            "- [官方 D3-flash](https://huggingface.co/vllm-sr/d3-flash)", ""]
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics-root", type=Path)
    args = p.parse_args()
    document = render(json.loads(args.input.read_text()), args.metrics_root)
    with args.output.open("x") as stream:
        stream.write(document)
