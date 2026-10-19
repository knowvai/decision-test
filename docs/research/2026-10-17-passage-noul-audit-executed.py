"""Independent offline evidence audit and report; never issue inference."""
from datetime import datetime, timedelta, timezone
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args):
    def offline(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Offline audit cannot make network requests")
    sys.dont_write_bytecode = True
    sys.addaudithook(offline)
    root, old = args.root, args.old
    source, run = root / "source-r1", root / "run-001"
    sys.path.insert(0, str(source / "evaluation/runner/scripts"))
    sys.path.insert(0, str(source / "deploy/docker/decision-eval"))
    import decision_noul_validation as n
    from deployment_probe import native_question_request, native_noul_result
    from launch_reviewed import MODELS
    frozen = read(root / "preparation-r1/frozen.json")
    for name, expected in frozen["sources"].items():
        assert sha(source / name) == expected
    supplement = read(root / "preparation-r1/test-final.json")
    for name, expected in supplement["supplementary_test_dependencies"].items():
        assert sha(source / name) == expected
    protocol = read(run / "protocol.json")
    n.p.check_artifacts(protocol["source_sha256"])
    preserved = read(root / "preparation-r1/preservation.json")
    assert sha(root / "preparation-r1/preservation.json") == protocol["preservation_sha256"]
    n.p.check_artifacts(preserved)
    support = read(root / "preparation-r1/support.json")
    assert sha(root / "preparation-r1/support.json") == protocol["support_sha256"]
    n.p.check_artifacts(support["source_sha256"])
    state = read(run / "status.json")
    assert state["stage"] in ("complete", "failed"), "Do not audit a changing run"
    unit = subprocess.check_output(["systemctl", "show", "decision-passage-noul-20261017-r1.service",
        "-p", "MainPID", "-p", "LoadState", "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "ExecMainStatus", "-p", "Restart"], text=True)
    service = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line)
    assert service["MainPID"] == "0" and service["ActiveState"] in ("inactive", "failed")
    journal = subprocess.check_output(["journalctl", "-u", "decision-passage-noul-20261017-r1.service",
        "--no-pager", "-o", "json"], text=True)
    journal = [{k: row.get(k) for k in ("__REALTIME_TIMESTAMP", "MESSAGE", "UNIT_RESULT", "EXIT_CODE", "EXIT_STATUS")}
               for line in journal.splitlines() if line for row in [json.loads(line)]]
    locks = subprocess.check_output(["lslocks", "-o", "PID,TYPE,MODE,PATH"], text=True)
    assert not any("classification.lock" in line for line in locks.splitlines())
    pools, labels = n.load_dataset(SimpleNamespace(dataset=old / "dataset-six-mask-v1"))
    expected = n.jobs(pools)
    all_ids, previous_end = set(), None
    report = {"at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
        "state": state, "service": service, "unit_journal": journal, "shared_lock_released": True,
        "complete": state["stage"] == "complete",
        "protocol_sha256": sha(run / "protocol.json"), "preserved_files": len(preserved),
        "preserved_bytes": sum(Path(path).stat().st_size for path in preserved),
        "old_evidence_unchanged": True, "source_unchanged": True, "tests": supplement["tests"],
        "dataset": {"pools": 20, "candidates": 333, "positives": 116, "negatives": 217},
        "models": {}, "total_requests": 0, "total_valid": 0, "total_failed": 0, "total_timeouts": 0}
    for model in n.MODELS:
        folder = run / model
        if not (folder / "requests.jsonl").exists():
            report["models"][model] = {"status": "not_executed", "planned_validation_pairs": 333}
            continue
        raw = (folder / "requests.jsonl").read_bytes()
        assert not raw or raw.endswith(b"\n")
        rows = n.b.read_jsonl(folder / "requests.jsonl")
        intents = n.b.read_jsonl(folder / "intents.jsonl")
        assert len(rows) == len(intents) <= 333
        family, _, served = MODELS[model]
        original_deployment = read(old / "run-001" / model / "deployment.json")
        deployment = read(folder / "deployment.json")
        for field in ("model", "served_model", "family", "image_id", "tp", "cpu_budget", "host_memory_limit_gib", "command", "scheduling", "env"):
            assert deployment[field] == original_deployment[field]
        last_end = 0
        for row, intent, (pool, candidate) in zip(rows, intents, expected):
            identity = {"model": model, "phase": "validation", "arm": "noul", "stage": "refine",
                "case_id": pool["case_id"], "fragment_id": candidate["fragment_id"]}
            state_input = n.p.pair_state(pool, candidate)
            body = n.payload(served, state_input)
            path, wire = native_question_request(family, body)
            assert all(row[k] == v for k,v in identity.items())
            assert row["call_id"] == intent["call_id"] == n.b.digest(identity)
            assert row["input_sha256"] == intent["input_sha256"] == n.b.digest(state_input)
            assert row["request_sha256"] == intent["request_sha256"] == n.b.digest(wire)
            assert row["call_id"] not in all_ids and row["calls"] == 1 and row["path"] == path
            all_ids.add(row["call_id"])
            assert row["inflight_at_start"] == 1 and row["started_seconds"] >= last_end
            last_end = row["ended_seconds"]
            start = datetime.fromisoformat(row["at"])
            if previous_end is not None:
                assert start >= previous_end - timedelta(milliseconds=1)
            previous_end = start + timedelta(seconds=row["ended_seconds"]-row["started_seconds"])
            if row["status"] == "valid":
                decoded = native_noul_result(family, body, row["native_response"])
                assert row["prediction"] == decoded["prediction"] and row["probabilities"] == decoded["probabilities"]
        stats = n.p.request_stats(rows)
        report["total_requests"] += len(rows)
        report["total_valid"] += stats["valid"]
        report["total_failed"] += stats["failed"]
        report["total_timeouts"] += stats["timeouts"]
        if len(rows) == 333 and stats["valid"] == 333:
            value = n.audit_model(folder, model, pools, labels, old / "run-001")
            saved = read(folder / "audit.json")
            assert all(value[k] == v for k,v in saved.items() if k != "evidence_sha256")
            for name, digest in saved["evidence_sha256"].items():
                assert sha(folder / name) == digest
            value["status"] = "complete"
            value["binary_probability_ties"] = sum(r["probabilities"]["true"] == r["probabilities"]["false"] for r in rows)
            value["returned_scalar_half"] = sum(type(r.get("noul")) in (int, float) and r["noul"] == .5 for r in rows)
        else:
            value = {"status": "incomplete", "requests": stats, "noul": n.quality(pools, labels, rows),
                "unattempted": 333-len(rows), "last_failure": {k: rows[-1].get(k) for k in ("error_type", "http_status", "sequence_halted", "server_terminal_confirmed")} if rows else None}
        report["models"][model] = value
    if report["complete"]:
        assert report["total_requests"] == report["total_valid"] == 4329
        assert len(state["completed"]) == 13 and service["Result"] == "success" and service["ExecMainStatus"] == "0"
        assert any("Deactivated successfully" in (row["MESSAGE"] or "") for row in journal)
        report["final_container"] = n.p.container_identity()
        assert report["final_container"] == read(run / n.MODELS[-1] / "container.json")
    n.p.check_artifacts(preserved)
    report.update(passed=True, unique_calls=True, global_nonoverlap=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "complete", "total_requests", "total_valid", "total_failed")}))


def render(args):
    data = read(args.input)
    assert data["passed"]
    def point(row, field):
        return f'{row["quality"][field]["point"]*100:.2f}'
    def ci(v):
        return f'{v["point"]*100:.2f} [{v["low"]*100:.2f}, {v["high"]*100:.2f}]'
    lines = ["# 原生 Noul 与 Choice：验证池对照", "",
        "20 验证池 / 333 完整候选 / 116 正例、217 负例。保持检索相关性含义和同一提示，只改变原生模式；choice 来自旧运行，没有重发，也没有重跑正式集。", "",
        "原生类别直接采用；仅返回 P(true) 的实现使用固定二类 argmax，相等为 false，不使用验证标签校准阈值。P/R/F1 为池级宏平均；差值为 noul-choice，区间为固定 2000 次配对池级 bootstrap。Accuracy 是候选加权，不能与 Precision 混称。", "",
        f'实际 {data["total_requests"]} 请求，{data["total_valid"]} 有效、{data["total_failed"]} 失败、{data["total_timeouts"]} 超时。全轮完成：{data["complete"]}。', "",
        "| 模型 | choice P / R / F1 % | noul P / R / F1 % | ΔF1 pp [95% CI] | choice / noul Accuracy % | 判断翻转 |",
        "| --- | --- | --- | --- | --- | ---: |"]
    for model, row in data["models"].items():
        if row["status"] != "complete":
            lines.append(f'| {model} | {row["status"]} | 未完整执行，不作模式优劣结论 | | | |')
            continue
        a,z = row["noul"],row["choice"]
        lines.append(f'| {model} | {" / ".join(point(z,k) for k in ("precision","recall","f1"))} | {" / ".join(point(a,k) for k in ("precision","recall","f1"))} | {ci(row["paired_delta"]["f1"])} | {z["candidate_accuracy"]*100:.2f} / {a["candidate_accuracy"]*100:.2f} | {row["flips"]} |')
    lines += ["", "## 原生 Noul 明细", "", "| 模型 | Precision % [95% CI] | Recall % [95% CI] | F1 % [95% CI] | TP / FP / FN / TN | 空选择池 | 概率平局 |", "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for model,row in data["models"].items():
        if "noul" not in row:
            continue
        q,c = row["noul"]["quality"],row["noul"]["counts"]
        lines.append(f'| {model} | {ci(q["precision"])} | {ci(q["recall"])} | {ci(q["f1"])} | {" / ".join(str(c.get(k,0)) for k in ("tp","fp","fn","tn"))} | {c.get("empty_pools",0)} | {row.get("binary_probability_ties","NA")} |')
    lines += ["", "## 请求时间与覆盖", "", "计时沿用既有请求计时器，含构造/哈希/网络/鉴权/解析；扣除 intent 日志写入，排除结果落盘。历史 choice 与新 noul 不是同轮交替性能实验。", "", "| 模型 | 有效 / 尝试 / 计划 | 失败 / 超时 | noul mean / P95 / P99 ms | 全尝试耗时 ms |", "| --- | --- | --- | --- | ---: |"]
    for model,row in data["models"].items():
        if "requests" not in row:
            continue
        s=row["requests"]
        t=s["all_attempt_latency_ms"]
        latency = " / ".join(f'{t[k]:.2f}' if t[k] is not None else "NA" for k in ("mean_ms", "p95_ms", "p99_ms"))
        total = f'{t["mean_ms"]*s["attempted"]:.2f}' if t["mean_ms"] is not None else "NA"
        lines.append(f'| {model} | {s["valid"]} / {s["attempted"]} / 333 | {s["failed"]} / {s["timeouts"]} | {latency} | {total} |')
    lines += ["", "## 审计与局限", "", f'- {data["preserved_files"]} 个历史证据文件 / {data["preserved_bytes"]} 字节保全通过；新源码、模型支持源码、部署/镜像、输入/request 哈希、唯一调用和全局 n=1 区间核验通过。',
        "- JSON 保留各项区间、配对差值、readout 来源、历史 choice 请求时间及所有未执行/失败覆盖；失败池不从 20 池分母删除，失败请求不计为正确。",
        "- 此验证集曾被历史运行用于阈值校准。本轮没有调参，但模式选择不能冒称独立正式集泛化验证；20 池较小，13 模型比较未做多重比较校正。",
        "- 原生 noul 的内部实现/作者校准可能与 choice 不同，不能笼统称为同一头只换字段，也不能据此宣称最佳模式或生产容量。",
        "- 不改代理机 reranker/embedding，不自动恢复旧模型，不启动新正式推理。", "",
        "完整证据：`2026-10-17-passage-noul-results.json`；协议：`2026-10-17-passage-noul-protocol.md`；正式集二选一表：`2026-10-17-passage-choice-results.md`。", ""]
    with args.output.open("x") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "render"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args) if args.mode == "audit" else render(args)
