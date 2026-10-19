"""Offline independent verification/reporting for the answer-evidence run."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys


def verify_original_sources(dataset_path, source_path):
    """Independently rejoin original IDs, grades and text before runtime freezing."""
    protocol = json.loads((dataset_path / "protocol.json").read_text())
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    for name, expected in protocol["source_sha256"].items():
        assert digest(source_path / name) == expected, name
    csv.field_size_limit(1024 * 1024 * 1024)
    with (source_path / "queries.dev.tsv").open() as stream:
        queries = {r["qid"]: r["text"] for r in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)}
    graded = defaultdict(dict)
    with (source_path / "qrels.dev.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            assert row["pid"] not in graded[row["qid"]]
            graded[row["qid"]][row["pid"]] = int(row["rel"])
    expected_texts, totals, all_qids = {}, {}, set()
    for phase, selected in protocol["split_qids"].items():
        pools = [json.loads(line) for line in (dataset_path / (phase + ".inputs.private.jsonl")).read_text().splitlines()]
        labels = {r["case_id"]: r for r in map(json.loads, (dataset_path / (phase + ".labels.private.jsonl")).read_text().splitlines())}
        assert [p["source_qid"] for p in pools] == selected
        grades = Counter()
        conflicts = 0
        for pool in pools:
            qid = pool["source_qid"]
            assert qid not in all_qids and pool["query"] == queries[qid]
            all_qids.add(qid)
            label = labels[pool["case_id"]]
            assert label["grades"] == {"t2p:" + pid: g for pid, g in graded[qid].items()}
            assert set(label["positive_ids"]) == {"t2p:" + pid for pid, g in graded[qid].items() if g in (2, 3)}
            assert {c["source_pid"] for c in pool["candidates"]} == set(graded[qid])
            binary_by_text = defaultdict(set)
            for candidate in pool["candidates"]:
                pid = candidate["source_pid"]
                assert candidate["fragment_id"] == "t2p:" + pid
                if pid in expected_texts:
                    assert expected_texts[pid] == candidate["content"]
                expected_texts[pid] = candidate["content"]
                grades[graded[qid][pid]] += 1
                binary_by_text[candidate["content"]].add(graded[qid][pid] >= 2)
            conflicts += sum(len(v) > 1 for v in binary_by_text.values())
        assert not conflicts, "Identical query/text has conflicting evidence labels"
        totals[phase] = {"pools": len(pools), "pairs": sum(grades.values()), "grades": dict(grades),
                         "conflicting_identical_text_labels": conflicts}
    found = set()
    with (source_path / "collection.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            pid = row["pid"]
            if pid in expected_texts:
                assert pid not in found and row["text"] == expected_texts[pid]
                found.add(pid)
    assert found == set(expected_texts)
    return {"passed": True, "dataset_protocol_sha256": digest(dataset_path / "protocol.json"),
            "original_sources_sha256": protocol["source_sha256"], "unique_passage_ids": len(found),
            "source_ids_grades_and_full_text_rejoined": True, "splits": totals,
            "script_sha256": digest(Path(__file__))}


def audit_online_rankings(rows, allow_variation=False):
    """Describe cross-call variation; per-call Top10 is verified by audit_model."""
    first, events, phases = {}, [], defaultdict(Counter)
    changed_pools, order_pools, membership_pools = set(), set(), set()
    for row in rows:
        key = row["phase"], row["case_id"]
        counts = phases[row["phase"]]
        counts["calls"] += 1
        if key not in first:
            first[key] = row
            counts["pools"] += 1
            continue
        baseline = first[key]
        for field in ("input_sha256", "request_sha256"):
            assert row[field] == baseline[field], "Reranker input/request changed: " + field
        assert [s["fragment_id"] for s in row["scores"]] == [s["fragment_id"] for s in baseline["scores"]]
        counts["comparisons_to_first"] += 1
        counts["score_changed_calls"] += row["scores"] != baseline["scores"]
        if row["retained_ids"] == baseline["retained_ids"]:
            counts["same_ordered_top10"] += 1
            continue
        assert allow_variation, "Online Top10 membership changed"
        before, after = set(baseline["retained_ids"]), set(row["retained_ids"])
        kind = "order_only" if before == after else "membership_changed"
        counts[kind] += 1
        changed_pools.add(key)
        (order_pools if before == after else membership_pools).add(key)
        events.append({"phase": key[0], "case_id": key[1], "kind": kind,
            "baseline_call_id": baseline["call_id"], "call_id": row["call_id"],
            "baseline_model": baseline["model"], "model": row["model"],
            "baseline_arm": baseline["arm"], "arm": row["arm"],
            "input_sha256": row["input_sha256"], "request_sha256": row["request_sha256"],
            "baseline_retained_ids": baseline["retained_ids"], "retained_ids": row["retained_ids"],
            "removed_ids": sorted(before - after), "added_ids": sorted(after - before)})
    for phase, counts in phases.items():
        for name, keys in (("changed_pools", changed_pools), ("order_only_pools", order_pools),
                           ("membership_changed_pools", membership_pools)):
            counts[name] = sum(key[0] == phase for key in keys)
    totals = Counter()
    for counts in phases.values():
        totals.update(counts)
    return {"policy": "per_request_top10_with_observed_variation" if allow_variation else "strict_cross_call_identity",
            "comparison_baseline": "first chronological call per phase/case across the frozen model order",
            "counts": dict(totals), "phases": {k: dict(v) for k, v in phases.items()}, "events": events}


def report_md(result):
    pct = lambda v: f"{100 * v:.2f}"
    interval = lambda v: "N/A" if v is None else f"{pct(v['point'])} [{pct(v['low'])}, {pct(v['high'])}]"
    lines = ["# 答案证据筛选：原生二选一", "",
        "正例为原始 T2Ranking L2/L3，负例为 L0/L1；完整片段、原生二选一、无校准阈值。直接筛选与在线 reranker Top10 配对。",
        "原始主测试 200 个新问题；超出共同完整输入覆盖的整池单列，不替换、不计正确拒绝。额外无证据池不混入主测试。", "",
        f"共同覆盖：{json.dumps(result['common_coverage']['covered'], ensure_ascii=False)}。", "",
        f"实际请求 {result['calls']}；本版审计通过：{result['passed']}。原始标签/划分来源、请求哈希、原生结果读出、全局 n=1 和历史保全均已复核。", "",
        "## 主测试质量", "",
        "P/R/F1 只在有 L2/L3 的问题上作池级宏平均，含 95% 配对池级 bootstrap 区间。无证据池另报正确全拒绝率。", "",
        "| 模型 / 模式 | Precision % | Recall % | F1 % | Accuracy % | 漏光证据 % | 无证据正确全拒绝 % | 保留片段 |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |"]
    for model, value in result["models"].items():
        for mode, phase in value["phases"]["formal"].items():
            q = phase["quality"]
            m = q["positive_pool_macro"]
            lines.append(f"| {model} / {mode} | {interval(m['precision'])} | {interval(m['recall'])} | {interval(m['f1'])} | {pct(q['candidate_accuracy'])} | {interval(q['missed_all_evidence_rate'])} | {interval(q['correct_empty_rate'])} | {q['kept_candidates']} |")
    variation = result.get("online_reranker_variation")
    if variation:
        lines += ["", "## 在线排序波动与审计口径", "",
            "冻结原版审计要求同一池的跨调用 Top10 有序列表完全一致，曾因此退出，未生成最终报告。",
            "用户在运行结束后认可独立在线调用的自然波动；本版显式选择放宽跨调用一致性，不改原版或推理证据。",
            "逐请求的完整输入、请求哈希、当次分数稳定排序 Top10、原顺序精筛、原始池计分仍须全部通过。",
            "以下均相对每个阶段/池第一次实际调用；分数变化不一定改变 Top10，顺序变化不一定改变成员。",
            "原因未由本次观测确定。不是重试或额外复测；跨模式 Top10 比较是实测整条链路对比，不是固定候选下的精筛器因果差异。", "",
            "| 阶段 | 调用 | 池 | 对首次比较 | 分数变化调用 | 仅排序变化调用 | 成员变化调用 | 成员变化池 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for phase, counts in variation["phases"].items():
            lines.append(f"| {phase} | " + " | ".join(str(counts.get(k, 0)) for k in
                ("calls", "pools", "comparisons_to_first", "score_changed_calls", "order_only", "membership_changed", "membership_changed_pools")) + " |")
        lines += ["", "审计模式：`" + variation["policy"] + "`；详细调用 ID、哈希、成员差异见同名 JSON。"]
    if result.get("original_denominator_zero_credit"):
        lines += ["", "## 原始分母覆盖调整", "",
            "主测试保留原始 200 池/3105 对分母，覆盖 197 池/3054 对。排除的 3 池/51 对（11 正例）未发推理，不进入语义混淆矩阵。",
            "以下 P/R/F1 对未覆盖的有证据池给零分，accuracy 对未覆盖候选给零分；不是把它们伪造成普通 FN/TN。", "",
            "| 模型 / 模式 | 覆盖调整 P % [95% CI] | 覆盖调整 R % [95% CI] | 覆盖调整 F1 % [95% CI] | 覆盖调整 accuracy % | 未评分候选 |",
            "| --- | --- | --- | --- | --- | ---: |"]
        for model, phases in result["original_denominator_zero_credit"].items():
            for arm, q in phases["formal"].items():
                adjusted = q["coverage_adjusted_zero_credit"]
                lines.append(f"| {model} / {arm} | " + " | ".join(interval(adjusted["positive_pool_macro"][k])
                    for k in ("precision", "recall", "f1")) + f" | {pct(adjusted['candidate_accuracy'])} | {q['counts'].get('unscored', 0)} |")
    lines += ["", "## 无模型基线", "", "| 规则 | Precision % | Recall % | F1 % | 无证据正确全拒绝 % | Accuracy % |",
              "| --- | --- | --- | --- | --- | --- |"]
    for mode, q in result["baselines"]["formal"].items():
        m = q["positive_pool_macro"]
        lines.append(f"| {mode} | {interval(m['precision'])} | {interval(m['recall'])} | {interval(m['f1'])} | {interval(q['correct_empty_rate'])} | {pct(q['candidate_accuracy'])} |")
    lines += ["", "## 配对比较", "", "质量差值为后者减前者；耗时节省为前者减后者。未做多重比较校正。", "",
              "| 比较 | 主测试 ΔF1 pp [95% CI] | 主测试 ΔRecall pp [95% CI] | 整链路节省 ms [95% CI] / 前者与后者总时长比 | 同成功池 / 总时长节省 % |",
              "| --- | --- | --- | --- | --- |"]
    for name, phases in result["paired_comparisons"].items():
        delta = phases["formal"]
        t = delta["time_saved_ms"]
        timing = f"{t['point']:.2f} [{t['low']:.2f}, {t['high']:.2f}] / {delta['ratio_of_summed_paired_times']:.4f}"
        lines.append(f"| {name} | {interval(delta['f1'])} | {interval(delta['recall'])} | {timing} | {delta['paired_valid_pools']} / {pct(1 - 1 / delta['ratio_of_summed_paired_times'])} |")
    lines += ["", "## 主测试耗时", "", "整池为实测 wall，包含网络/鉴权及请求落盘；排除整池记录落盘和池间身份检查。请求耗时扣除 intent 落盘、排除结果落盘；二者不是同一时钟。", "",
              "| 模型 / 模式 | 有效 / 尝试 / 计划 | 请求 mean / P95 / P99 ms | 整池 mean / P95 / P99 ms | 全尝试请求耗时 s |",
              "| --- | --- | --- | --- | ---: |"]
    for model, value in result["models"].items():
        for mode, phase in value["phases"]["formal"].items():
            rs, wall = phase["requests"], phase["pool_wall_ms"]
            latency = rs["all_attempt_latency_ms"]
            triple = lambda t: " / ".join(f"{t[k]:.2f}" for k in ("mean_ms", "p95_ms", "p99_ms"))
            lines.append(f"| {model} / {mode} | {rs['valid']} / {rs['attempted']} / {phase['planned_requests']} | {triple(latency)} | {triple(wall)} | {phase['all_attempt_elapsed_ms'] / 1000:.3f} |")
    lines += ["", "## 等级错误", "", "L0/L1 列为误留率，L2/L3 列为保留率，均使用对应等级全部候选为分母。", "",
              "| 模型 / 模式 | L0 误留 % | L1 误留 % | L2 保留 % | L3 保留 % | TP / FP / FN / TN |",
              "| --- | --- | --- | --- | --- | --- |"]
    for model, value in result["models"].items():
        for mode, phase in value["phases"]["formal"].items():
            q = phase["quality"]
            rates = [pct(g.get("accepted", 0) / g["planned"]) if g.get("planned", 0) else "N/A" for g in q["grades"].values()]
            confusion = " / ".join(str(q["counts"].get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
            lines.append(f"| {model} / {mode} | " + " | ".join(rates) + f" | {confusion} |")
    lines += ["", "## 裁剪与精筛漏判", "", "| 模型 / 模式 | reranker 裁掉正例 | 精筛拒绝正例 | 总 FN | reranker-only P / R / F1 % [95% CI] |",
              "| --- | ---: | ---: | ---: | --- |"]
    for model, value in result["models"].items():
        for arm, phase in value["phases"]["formal"].items():
            omissions = phase.get("positive_omissions")
            if omissions:
                quality = phase["reranker_only"]["positive_pool_macro"]
                lines.append(f"| {model} / {arm} | {omissions['reranker_pruned_positive']} | {omissions['refiner_rejected_positive']} | {phase['quality']['counts'].get('fn', 0)} | " +
                    " / ".join(interval(quality[k]) for k in ("precision", "recall", "f1")) + " |")
    lines += ["", "## 候选数切片", "", "共同主测试的 <=10 与 >10 切片。P/R/F1 仍仅对有证据池宏平均，整池耗时覆盖切片所有池。", "",
              "| 模型 / 模式 / 切片 | 池数 | P / R / F1 % [95% CI] | 整池 mean / P95 / P99 ms |",
              "| --- | ---: | --- | --- |"]
    triple = lambda t: " / ".join("N/A" if t[k] is None else f"{t[k]:.2f}" for k in ("mean_ms", "p95_ms", "p99_ms"))
    for model, value in result["models"].items():
        for arm, phase in value["phases"]["formal"].items():
            for name, subset in phase["slices"].items():
                q = subset["quality"]
                if q is not None:
                    lines.append(f"| {model} / {arm} / {name} | {q['pools']} | " + " / ".join(interval(q["positive_pool_macro"][k])
                        for k in ("precision", "recall", "f1")) + f" | {triple(subset['pool_wall_ms'])} |")
    lines += ["", "## 各阶段实际覆盖与耗时", "", "未尝试列仅指共同覆盖计划内请求；前置覆盖排除另外保留原始分母。成功率是链路成功率，不是语义正确率。", "",
              "| 模型 / 阶段 / 模式 | 有效 / 尝试 / 计划 | 失败 / 超时 / 未尝试 | 链路成功 % | 请求 mean / P95 / P99 ms | 整池 mean / P95 / P99 ms | 全尝试请求耗时 s |",
              "| --- | --- | --- | --- | --- | --- | ---: |"]
    for model, value in result["models"].items():
        for phase_name, arms in value["phases"].items():
            for arm, phase in arms.items():
                rs = phase["requests"]
                lines.append(f"| {model} / {phase_name} / {arm} | {rs['valid']} / {rs['attempted']} / {phase['planned_requests']} | {rs['failed']} / {rs['timeouts']} / {phase['planned_requests'] - rs['attempted']} | {pct(phase['chain_success_rate'])} | {triple(rs['all_attempt_latency_ms'])} | {triple(phase['pool_wall_ms'])} | {phase['all_attempt_elapsed_ms'] / 1000:.3f} |")
    lines += ["", "## 各阶段 Rerank 与 Refine 请求", "", "各阶段请求耗时不是整链 wall 的替代。全尝试耗时由请求数乘全尝试均值还原。", "",
              "| 模型 / 阶段 / 模式 / 请求类型 | 有效 / 尝试 | 失败 / 超时 | mean / P95 / P99 ms | 全尝试耗时 s |",
              "| --- | --- | --- | --- | ---: |"]
    for model, value in result["models"].items():
        for phase_name, arms in value["phases"].items():
            for arm, phase in arms.items():
                for stage, rs in phase["stages"].items():
                    latency = rs["all_attempt_latency_ms"]
                    elapsed = rs["attempted"] * latency["mean_ms"] / 1000 if rs["attempted"] else 0
                    lines.append(f"| {model} / {phase_name} / {arm} / {stage} | {rs['valid']} / {rs['attempted']} | {rs['failed']} / {rs['timeouts']} | {triple(latency)} | {elapsed:.3f} |")
    lines += ["", "## 独立无证据诊断", "", "不混入主测试。共同覆盖 38 池；原始分母 40 池，覆盖排除 2 池不算成功空预测。", "",
              "| 模型 / 模式 | 正确全拒绝 % [95% CI] | 原始分母覆盖调整 % [95% CI] | 误留负例 / 完整候选 |",
              "| --- | --- | --- | --- |"]
    for model, value in result["models"].items():
        for arm, phase in value["phases"].get("no_evidence", {}).items():
            q = phase["quality"]
            adjusted = result["original_denominator_zero_credit"][model]["no_evidence"][arm]["coverage_adjusted_zero_credit"]
            lines.append(f"| {model} / {arm} | {interval(q['correct_empty_rate'])} | {interval(adjusted['correct_empty_rate'])} | {q['counts'].get('fp', 0)} / {q['counts']['planned']} |")
    lines += ["", "## 边界", "",
        "- 原始四级标签是答案证据价值的公开代理，不是逐条事实核验或来源可靠性标注。没有评测最终生成答案，不能声称 RAG 回答正确率提高。",
        "- 有证据池与无证据池的指标分别计算；不能把正例池的空结果当正确，也不能把负例池的正确拒绝当漏判。",
        "- 同一部署中模式/链路使用冻结均衡顺序；不同模型仍顺序部署。reranker 使用额外主机资源，与业务 embedding 共享 GPU；不是等总资源加速或生产容量。",
        "- 工程、验证、主测试、额外无证据池的完整统计、失败数、字符缩减及模式差值在 JSON；字符缩减不是 token 节省。",
        "- 没有在本轮输出上调提示、阈值或选择 TopN；所有推理记录和旧结果保留。", ""]
    return "\n".join(lines)


def paired(e, left, right, left_chains, right_chains):
    assert [r["case_id"] for r in left] == [r["case_id"] for r in right]
    out = {}
    for metric in ("precision", "recall", "f1", "correct_empty", "exact_match"):
        pairs = [(a, b) for a, b in zip(left, right) if
            (bool(a["gold_count"]) if metric in ("precision", "recall", "f1") else
             not a["gold_count"] if metric == "correct_empty" else True)]
        out[metric] = e.b.bootstrap_mean([b[metric] - a[metric] for a, b in pairs]) if pairs else None
    a = {r["case_id"]: r["wall_ms"] for r in left_chains if r["valid"]}
    b = {r["case_id"]: r["wall_ms"] for r in right_chains if r["valid"]}
    ids = sorted(a.keys() & b.keys())
    out.update(paired_valid_pools=len(ids), time_saved_ms=e.b.bootstrap_mean([a[i] - b[i] for i in ids]),
        ratio_of_summed_paired_times=sum(a[i] for i in ids) / sum(b[i] for i in ids),
        per_pool_speed_ratio=e.b.bootstrap_mean([a[i] / b[i] for i in ids]))
    return out


def main(args):
    if args.verify_data_only:
        result = verify_original_sources(args.root / "dataset-v1", args.original_source)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result))
        return
    source = args.root / ("source-" + args.revision)
    preparation = args.root / ("preparation-" + args.revision)
    sys.path.insert(0, str(source / "evaluation/runner/scripts"))
    sys.path.insert(0, str(source / "deploy/docker/decision-eval"))
    import decision_evidence_benchmark as e

    def offline(event, _args):
        if event in ("socket.connect", "subprocess.Popen"):
            raise RuntimeError("Offline audit cannot issue network calls or launch processes")

    sys.addaudithook(offline)
    run = args.root / "run-001"
    original_data, dataset = e.load_dataset(args.root / "dataset-v1")
    protocol = json.loads((run / "protocol.json").read_text())
    frozen = json.loads((preparation / "frozen.json").read_text())
    coverage = preparation / "coverage.json"
    assert e.sha(coverage) == frozen["coverage_sha256"] == protocol["coverage_sha256"]
    data, unsupported = e.load_coverage(coverage, original_data, frozen["dataset_protocol_sha256"])
    assert json.loads((run / "status.json").read_text())["stage"] == "complete"
    assert protocol["models"] == list(e.p.ROSTER) and protocol["conditions"] == 56 and protocol["top_n"] == 10
    assert protocol["questions"] == {m: e.questions(m) for m in ("choice", "noul")}
    assert protocol["dataset_protocol"] == dataset
    assert protocol["http_concurrency"] == 1 and protocol["retries"] == 0
    assert e.sha(preparation / "frozen.json") == protocol["frozen_sha256"]
    preserved = json.loads((preparation / "preservation.json").read_text())
    provenance = json.loads((preparation / "data-provenance.json").read_text())
    assert provenance["passed"] and provenance["source_ids_grades_and_full_text_rejoined"]
    assert provenance["dataset_protocol_sha256"] == frozen["dataset_protocol_sha256"]
    assert e.sha(preparation / "data-provenance.json") == frozen["data_provenance_sha256"]
    assert e.sha(preparation / "reranker-frozen.json") == frozen["reranker_evidence_sha256"] == protocol["reranker_evidence_sha256"]
    e.p.check_artifacts(preserved)
    e.p.check_artifacts(protocol["source_sha256"])
    assert e.sha(args.root / "dataset-v1/protocol.json") == frozen["dataset_protocol_sha256"]
    result = {"passed": True, "audited_at": e.p.now(), "models": {}, "baselines": {},
              "dataset": dataset, "data_provenance": provenance, "paired_comparisons": {}, "calls": 0,
              "common_coverage": json.loads((run / "common-coverage.json").read_text()),
              "semantic_error_ledger": [], "original_denominator_zero_credit": {},
              "preserved_files": len(preserved), "script_sha256": e.sha(__file__)}
    result["audit_policy"] = {"revision": args.revision, "allow_online_reranker_variation": args.allow_online_reranker_variation,
        "frozen_audit_sha256": e.sha(source / "docs/research/2026-10-18-evidence-audit.py"),
        "per_request_verifier": "unchanged frozen decision_evidence_benchmark.audit_model"}
    seen_calls, end, rerank_rows, all_chains = set(), 0, [], {}
    for model in e.p.ROSTER:
        folder = run / model
        recorded = json.loads((folder / "audit.json").read_text())
        rebuilt = e.audit_model(folder, model, data)
        assert rebuilt == recorded, "Model audit differs: " + model
        deployment = json.loads((folder / "deployment.json").read_text())
        original = json.loads((args.baseline / model / "deployment.json").read_text())
        for key in ("model", "served_model", "family", "image_id", "tp", "cpu_budget", "host_memory_limit_gib", "command", "scheduling", "env"):
            assert deployment[key] == original[key], (model, key)
        rows = e.b.read_jsonl(folder / "requests.jsonl")
        all_chains[model] = e.b.read_jsonl(folder / "chains.jsonl")
        result["original_denominator_zero_credit"][model] = {}
        assert rebuilt["first_started_seconds"] >= end
        end = rebuilt["last_ended_seconds"]
        for row in rows:
            assert row["call_id"] not in seen_calls
            seen_calls.add(row["call_id"])
            if row["stage"] == "rerank":
                rerank_rows.append(row)
        # Cross-check candidate confusion directly from original graded labels.
        for phase, (pools, labels) in data.items():
            result["original_denominator_zero_credit"][model][phase] = {}
            truth = {(r["case_id"], fid): grade >= 2 for r in labels for fid, grade in r["grades"].items()}
            for mode in e.conditions(model):
                group = [r for r in rows if r["phase"] == phase and r["arm"] == mode]
                group = [r for pool in pools for r in e.scored_rows(pool,
                    [r for r in group if r["case_id"] == pool["case_id"]])]
                counts = Counter()
                for row in group:
                    gold = truth[(row["case_id"], row["fragment_id"])]
                    pred = row["prediction"] == "relevant"
                    counts["tp" if gold and pred else "fn" if gold else "fp" if pred else "tn"] += 1
                quality = rebuilt["phases"][phase][mode]["quality"]
                assert sum(counts.values()) == len(truth) == quality["counts"]["planned"]
                assert all(quality["counts"].get(k, 0) == counts[k] for k in ("tp", "fp", "fn", "tn"))
                original_pools, original_labels = original_data[phase]
                result["original_denominator_zero_credit"][model][phase][mode] = e.quality(original_pools, original_labels, group)
                grades = {(r["case_id"], fid): grade for r in labels for fid, grade in r["grades"].items()}
                original_inputs = {(pool["case_id"], c["fragment_id"]): e.b.digest(e.p.pair_state(pool, c))
                    for pool in pools for c in pool["candidates"]}
                ranks = {r["case_id"]: r for r in rows if r["phase"] == phase and r["arm"] == mode and r["stage"] == "rerank"}
                for row in group:
                    key = row["case_id"], row["fragment_id"]
                    gold, chosen = grades[key] >= 2, row["prediction"] == "relevant"
                    if gold == chosen:
                        continue
                    pruned = row.get("decision_source") == "reranker_pruned"
                    source = ranks[row["case_id"]] if pruned else row
                    result["semantic_error_ledger"].append({"model": model, "phase": phase, "arm": mode,
                        "stage": "rerank" if pruned else "refine", "case_id": key[0], "fragment_id": key[1],
                        "grade": grades[key], "reason_code": "reranker_pruned_positive" if pruned else
                        "refiner_rejected_positive" if gold else "refiner_accepted_negative",
                        "input_sha256": original_inputs[key], "originating_call_id": source["call_id"],
                        "request_sha256": source["request_sha256"], "execution_failure": False})
        result["models"][model] = rebuilt
        result["calls"] += rebuilt["calls"]
    result["online_reranker_variation"] = audit_online_rankings(rerank_rows, args.allow_online_reranker_variation)
    comparisons = []
    for model in e.p.ROSTER:
        comparisons += [(model, mode + ".direct", model, mode + ".top10") for mode in e.modes(model)]
        if len(e.modes(model)) == 2:
            comparisons += [(model, "choice." + arm, model, "noul." + arm) for arm in e.ARMS]
            comparisons += [(model, mode + "." + arm, llm, "choice." + arm)
                for mode in e.modes(model) for arm in e.ARMS for llm in e.p.ROSTER[-2:]]
    for lm, la, rm, ra in comparisons:
        name = f"{lm}/{la} -> {rm}/{ra}"
        result["paired_comparisons"][name] = {phase: paired(e,
            result["models"][lm]["phases"][phase][la]["quality"]["cases"],
            result["models"][rm]["phases"][phase][ra]["quality"]["cases"],
            [r for r in all_chains[lm] if r["phase"] == phase and r["arm"] == la],
            [r for r in all_chains[rm] if r["phase"] == phase and r["arm"] == ra]) for phase in data}
    for phase, (pools, labels) in data.items():
        result["baselines"][phase] = {}
        for name, prediction in (("all_keep", "relevant"), ("all_reject", "irrelevant")):
            rows = [{"case_id": pool["case_id"], "fragment_id": c["fragment_id"], "status": "valid", "prediction": prediction}
                    for pool in pools for c in pool["candidates"]]
            result["baselines"][phase][name] = e.quality(pools, labels, rows)
    assert result["calls"] == frozen["planned_calls_total"] == len(seen_calls)
    e.p.check_artifacts(preserved)
    e.p.check_artifacts(protocol["source_sha256"])
    e.p.save(args.output, result)
    with args.output.with_suffix(".md").open("x") as stream:
        stream.write(report_md(result))
    print(json.dumps({"passed": True, "models": len(result["models"]), "conditions": 56,
                      "calls": result["calls"], "preserved_files": len(preserved)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", default="r2")
    parser.add_argument("--allow-online-reranker-variation", action="store_true",
                        help="Record cross-call variation; retain frozen per-request Top10 verification")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--verify-data-only", action="store_true")
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
