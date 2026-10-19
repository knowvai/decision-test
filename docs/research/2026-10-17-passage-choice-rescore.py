"""Re-score frozen passage records using native choice only; never infer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rescore(args):
    def deny_network(event, _args):
        if event == "socket.connect":
            raise RuntimeError("This is an offline rescore, not an inference runner")

    sys.dont_write_bytecode = True
    sys.addaudithook(deny_network)
    root, prior = args.root, read(args.audit)
    assert prior["passed"]
    source = root / "source-r1"
    scripts = source / "evaluation/runner/scripts"
    for name in ("public_decision_benchmark.py", "decision_capacity_preflight.py", "decision_passage_benchmark.py"):
        assert sha(scripts / name) == prior["source_sha256"]["evaluation/runner/scripts/" + name]
    sys.path.insert(0, str(scripts))
    import decision_passage_benchmark as p
    b = p.benchmark
    data, run = root / "dataset-six-mask-v1", root / "run-001"
    assert sha(data / "protocol.json") == prior["dataset_protocol_sha256"]
    evidence = {data / name: value for name, value in read(data / "protocol.json")["files"].items()}
    evidence[data / "protocol.json"] = prior["dataset_protocol_sha256"]
    evidence[run / "protocol.json"] = prior["run_protocol_sha256"]
    for model, old in prior["models"].items():
        evidence.update({run / model / name: value for name, value in old["audit"]["artifact_sha256"].items()})
    assert all(sha(path) == expected for path, expected in evidence.items())
    pools = b.read_jsonl(data / "formal.inputs.private.jsonl")
    labels = b.read_jsonl(data / "formal.labels.private.jsonl")
    validation = b.read_jsonl(data / "validation.inputs.private.jsonl")
    validation_labels = b.read_jsonl(data / "validation.labels.private.jsonl")
    p.validate_pools(pools, labels)
    p.validate_pools(validation, validation_labels)
    gold = {r["case_id"]: set(r["positive_ids"]) for r in labels + validation_labels}
    candidates = {r["case_id"]: {c["fragment_id"] for c in r["candidates"]} for r in pools + validation}
    removed = Counter(r["case_id"] for r in read(data / "exclusions.json")["pairs"])
    slices = {"original_le29": [], "original_gt29": []}
    for pool in pools:
        n = len(pool["candidates"]) + removed[pool["case_id"]]
        slices["original_le29" if n <= 29 else "original_gt29"].append(pool["case_id"])
    assert [len(slices[k]) for k in slices] == [194, 6]

    def case_result(cid, selected, valid=True):
        positive, all_ids = gold[cid], candidates[cid]
        assert selected <= all_ids
        return {**p.set_metrics(positive, selected, valid), "case_id": cid,
                "tp": len(selected & positive), "fp": len(selected - positive),
                "fn": len(positive - selected), "tn": len(all_ids - (selected | positive)),
                "selected": len(selected), "empty": int(not selected)}

    def summary(values):
        count = Counter()
        for value in values:
            count.update({k: value[k] for k in ("tp", "fp", "fn", "tn", "empty", "selected")})
        return {"pools": len(values), "quality": {k: b.bootstrap_mean([r[k] for r in values]) for k in ("precision", "recall", "f1")},
                "counts": dict(count), "candidate_accuracy": (count["tp"] + count["tn"]) / sum(count[k] for k in ("tp", "fp", "fn", "tn")),
                "mean_selected": count["selected"] / len(values)}

    result = {"at": datetime.now(timezone.utc).isoformat(), "kind": "offline_native_choice_rescore", "passed": False,
              "rule": "status valid and stored prediction == relevant; no calibrated score threshold",
              "new_inference_calls": 0, "old_audit_sha256": sha(args.audit), "script_sha256": sha(__file__),
              "dataset": prior["dataset"], "old_run_request_totals": prior["totals"],
              "timing_scope": "Original online run only. No new native-choice chain wall time was measured.",
              "baseline_select_all": summary([case_result(r["case_id"], candidates[r["case_id"]]) for r in pools]),
              "models": {}}
    for model in p.ROSTER:
        old = prior["models"][model]
        records = b.read_jsonl(run / model / "requests.jsonl")
        chains = b.read_jsonl(run / model / "chains.jsonl")
        by_call = {r["call_id"]: r for r in records}
        assert len(records) == len(by_call) == 7080 and len(chains) == 400
        assert all(r["status"] == "valid" for r in records + chains)
        arms, old_values, changes = {}, {}, {}
        for chain in chains:
            cid, arm = chain["case_id"], chain["arm"]
            requests = [by_call[k] for k in chain["request_ids"]]
            refine = [r for r in requests if r["stage"] == "refine"]
            assert all(r["phase"] == "formal" and r["case_id"] == cid and r["arm"] == arm for r in requests)
            assert all(r["prediction"] in ("relevant", "irrelevant") for r in refine)
            seen = {r["fragment_id"] for r in refine}
            assert len(seen) == len(refine)
            assert seen == (candidates[cid] if arm in ("B1", "B2") else set(chain["retained_ids"]))
            native = p.selected(refine, {"kind": "hard_choice"})
            historical = set(chain["selected_ids"])
            assert historical == p.selected(refine, old["threshold"])
            arms.setdefault(arm, []).append(case_result(cid, native))
            old_values.setdefault(arm, []).append(case_result(cid, historical))
            removed_ids, added_ids = historical - native, native - historical
            changes.setdefault(arm, Counter()).update(
                changed_pools=int(native != historical), threshold_only=len(removed_ids), native_only=len(added_ids),
                threshold_only_positive=len(removed_ids & gold[cid]), threshold_only_negative=len(removed_ids - gold[cid]),
                native_only_positive=len(added_ids & gold[cid]), native_only_negative=len(added_ids - gold[cid]))
        model_result = {"arms": {}, "validation": None, "historical_rule": old["threshold"],
                        "historical_online_timing": {arm: {k: v[k] for k in ("successful_chain_latency_ms", "all_chain_latency_ms", "success_rate")} for arm, v in old["all_pools"]["arms"].items()}}
        for arm, values in arms.items():
            previous = old_values[arm]
            for metric in ("precision", "recall", "f1"):
                assert abs(statistics.mean(v[metric] for v in previous) - old["all_pools"]["arms"][arm]["quality"][metric]["point"]) < 1e-12
            model_result["arms"][arm] = {**summary(values), "successes": 200, "failures": 0,
                "selection_changes": dict(changes[arm]), "vs_historical_threshold_delta": {
                    k: b.bootstrap_mean([a[k] - z[k] for a, z in zip(values, previous)]) for k in ("precision", "recall", "f1")},
                "slices": {name: summary([v for v in values if v["case_id"] in ids]) for name, ids in slices.items()}}
            if old["threshold"]["kind"] == "hard_choice":
                assert changes[arm]["changed_pools"] == 0
        direct, pipeline = sorted(arms)
        downstream = {v["case_id"]: v for v in arms[pipeline]}
        model_result["pipeline_minus_direct"] = {
            k: b.bootstrap_mean([downstream[v["case_id"]][k] - v[k] for v in arms[direct]]) for k in ("precision", "recall", "f1")}
        vals = [r for r in records if r["phase"] == "validation"]
        assert len(vals) == 333 and all(r["stage"] == "refine" for r in vals)
        model_result["validation"] = summary([case_result(pool["case_id"], p.selected(
            [r for r in vals if r["case_id"] == pool["case_id"]], {"kind": "hard_choice"})) for pool in validation])
        result["models"][model] = model_result
        print(json.dumps({"rescored": model, "formal_chains": 400}), flush=True)
    assert all(sha(path) == expected for path, expected in evidence.items())
    result.update(passed=True, evidence_files_verified_unchanged=len(evidence),
                  classification_rule="18-way native choice; LLM strict JSON class; no passage threshold calibration")
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


def render(args):
    data = read(args.audit)
    assert data["passed"] and data["new_inference_calls"] == 0
    def ci(v):
        return f'{v["point"]*100:.2f} [{v["low"]*100:.2f}, {v["high"]*100:.2f}]'
    lines = ["# 片段筛选主结果：原生二选一", "",
             "按用户最新要求，所有精筛器仅依据既有请求记录的原生 `prediction == relevant` 保留片段，不使用验证集阈值，不重算 argmax 来替换原生 choice。LLM 同样直接用其返回类别。", "",
             "这是对已完成运行的离线重算：新增推理 0 次，15 配置、30 条件、6000 条有效链路；共同正式分母仍是 200 池、3188 候选、1437 正例、1751 负例。B3/B4 沿用各自实际在线 reranker 的 Top29 成员，裁剪正例仍计入 FN。旧原始记录、阈值报告和审计保留不变。", "",
             "P/R/F1 为池级宏平均，95% CI 复用冻结函数与固定种子的 2000 次池级 bootstrap。Accuracy 为候选加权 (TP+TN)/3188，与宏平均指标分母不同。此报告没有重新实测计时，原运行时间仅在 JSON 的 historical_online_timing 中保留，不能称为新 choice 方案的整链路实测。", "",
             "## 30 条件主表", "",
             "| 模型 | 链路 | Precision % [95% CI] | Recall % [95% CI] | F1 % [95% CI] | 候选 Accuracy % | 空选择池 |",
             "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for model, value in data["models"].items():
        for arm, row in value["arms"].items():
            q = row["quality"]
            lines.append(f'| {model} | {arm} | {ci(q["precision"])} | {ci(q["recall"])} | {ci(q["f1"])} | {row["candidate_accuracy"]*100:.2f} | {row["counts"]["empty"]} |')
    lines += ["", "## 直接筛选的口径变化", "", "以下变化仅来自选择规则改变，不是新模型推理。阈值额外保留指旧阈值保留、原生 choice 拒绝的候选；正负例按冻结标签计算。", "",
              "| 模型 | 原生 TP / FP / FN / TN | 阈值额外保留正例 / 负例 | 原生额外保留 | ΔPrecision pp | ΔRecall pp | ΔF1 pp |",
              "| --- | --- | --- | ---: | --- | --- | --- |"]
    for model, value in data["models"].items():
        arm = next(a for a in value["arms"] if a in ("B1", "B2"))
        r = value["arms"][arm]
        c, change, delta = r["counts"], r["selection_changes"], r["vs_historical_threshold_delta"]
        lines.append(f'| {model} | {c["tp"]} / {c["fp"]} / {c["fn"]} / {c["tn"]} | {change["threshold_only_positive"]} / {change["threshold_only_negative"]} | {change["native_only"]} | {ci(delta["precision"])} | {ci(delta["recall"])} | {ci(delta["f1"])} |')
    lines += ["", "## 审计与边界", "",
              f'- {data["evidence_files_verified_unchanged"]} 个源证据文件重算前后 SHA256 一致，所有原阈值成绩重建匹配；两款 LLM 的选择集合和成绩未变。',
              "- 固定数据的六项掩码、原候选/标签、Top29 成员与前后两次真实精筛调用不变；不以直接精筛结果替代后置精筛结果。",
              "- 原池 <=29 与 >29 切片、验证池原生 choice 基线、同模型前后差值及原阈值差值区间均在 JSON 中。空集 P/R/F1 仍记 0，不筛除不利池。",
              "- 原生 choice 不等于最佳可达质量，也不保证 F1 高于经过验证集优化的规则。结论只针对当前提示、模式、数据和引擎，不能外推生产并发或下游答案正确率。",
              "- noul 是后续独立验证对照，不由本次离线重算生成，不将其结果拼入本轮正式表。",
              "", "证据：`2026-10-17-passage-choice-results.json`；重算脚本：`2026-10-17-passage-choice-rescore.py`；历史阈值版：`2026-10-17-passage-filtering-results.md`。", ""]
    with args.output.open("x") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rescore", "render"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rescore(args) if args.mode == "rescore" else render(args)
