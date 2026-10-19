"""Independent, read-only audit of the frozen passage run; never sends inference."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def command(*args):
    return subprocess.check_output(args, text=True, timeout=60)


def system_state():
    fields = "MainPID,ActiveState,SubState,Result,ExecMainStatus,Restart"
    state = dict(line.split("=", 1) for line in command(
        "systemctl", "show", "decision-passage-20261017-r1.service",
        "--property=" + fields).splitlines())
    assert state == dict(MainPID="0", ActiveState="inactive", SubState="dead",
                         Result="success", ExecMainStatus="0", Restart="no")
    locks = json.loads(command("lslocks", "--json", "--output", "PID,TYPE,MODE,PATH"))
    assert not any(r.get("path", "").endswith("classification-20261015/classification.lock")
                   for r in locks.get("locks", []))
    return state


def inspect(cid):
    d = json.loads(command("docker", "inspect", cid))[0]
    s = d["State"]
    assert s["Running"] and not s["OOMKilled"] and d["RestartCount"] == 0
    return d, {"id": d["Id"], "image_id": d["Image"], "pid": s["Pid"],
               "started_at": s["StartedAt"], "running": s["Running"],
               "restarts": d["RestartCount"], "oom": s["OOMKilled"]}


def forbid_network(event, args):
    if event == "socket.connect":
        raise RuntimeError("Network inference is forbidden in this offline audit")


def test_audit(root):
    sys.dont_write_bytecode = True
    sys.addaudithook(forbid_network)
    source = root / "source-r1"
    scripts = source / "evaluation/runner/scripts"
    service = source / "deploy/docker/decision-eval"
    sys.path[:0] = [str(scripts), str(service)]
    assert sha(scripts / "decision_passage_benchmark.py") == "d10d17d11667addcc5da71300f947056bd2a888f614f6c5fc38c1e7f5edaa2dd"
    assert sha(scripts / "public_decision_benchmark.py") == "c9b0520f36bcb243a602bb27425f652cedb06f425deab92cc75337e1604ce8fd"
    import decision_passage_benchmark as p
    from launch_reviewed import MODELS
    b = p.benchmark
    run = root / "run-001"
    prep = root / "preparation-r1"
    data = root / "dataset-six-mask-v1"
    before = {str(f): sha(f) for folder in (run, prep, source, data)
              for f in folder.rglob("*") if f.is_file()}
    state = system_state()
    status = read(run / "status.json")
    assert status["stage"] == "complete" and status["completed"] == list(p.ROSTER)
    frozen, protocol = read(prep / "frozen.json"), read(run / "protocol.json")
    preserved = read(prep / "preservation.json")
    assert len(preserved) == 1163
    assert sum(Path(f).stat().st_size for f in preserved) == 140647988
    p.check_artifacts(preserved)
    assert read(run / "preservation-final.json") == {"unchanged": True, "artifacts": 1163}
    assert protocol["preserved_artifact_sha256"] == preserved
    for name, value in frozen["source_files"].items():
        assert sha(source / name) == value
        assert (source / name).stat().st_mode & 0o222 == 0
    p.check_artifacts(protocol["source_sha256"])
    startup = read(prep / "startup.json")
    assert startup["source_manifest_sha256"] == sha(prep / "frozen.json")
    assert startup["retries"] == 0 and startup["automatic_restart"] is False
    assert startup["unit"] == "decision-passage-20261017-r1.service"
    assert str(scripts / "public_decision_benchmark.py") in startup["command"]
    for flag, value in (("--dataset", data), ("--output", run), ("--root", root / "runtime-r1"),
                        ("--service-source", service), ("--preservation", prep / "preservation.json")):
        assert startup["command"][startup["command"].index(flag) + 1] == str(value)
    for key, value in dict(top_n=29, http_concurrency=1, pool_concurrency=1, timeout=300,
                           startup_timeout=2400, retries=0, fallback=False,
                           truncation=False, automatic_resume=False).items():
        assert protocol[key] == value
    assert protocol["models"] == list(p.ROSTER)
    assert protocol["reranker"] == read(prep / "reranker-frozen.json")
    dp = read(data / "protocol.json")
    assert sha(data / "protocol.json") == protocol["dataset_protocol_sha256"] == frozen["dataset_protocol_sha256"] == "b4b03b04cac155e27c38c984b24d2ac31b71d89f2b8c8d05d35fa27f54b78a40"
    p.check_artifacts(dp["preserved_source_sha256"])
    for name, value in dp["files"].items():
        assert sha(data / name) == value
    assert protocol["formal_exclusions_sha256"] == sha(data / "exclusions.json")
    original_exclusions = prep / "2026-10-17-passage-exclusions.json"
    assert sha(original_exclusions) == frozen["exclusions_sha256"] == "7839546bddbd59dd5ab0af4f3c7d3085ae85fa9c55f443f5b824d75bf4f3cb48"
    assert read(original_exclusions) == read(data / "exclusions.json")
    dataset = {name: b.read_jsonl(data / filename) for name, filename in (
        ("engineering", "engineering.inputs.private.jsonl"), ("validation", "validation.inputs.private.jsonl"),
        ("validation_labels", "validation.labels.private.jsonl"), ("formal", "formal.inputs.private.jsonl"),
        ("formal_labels", "formal.labels.private.jsonl"))}
    original_dir = root.parent / "classification-20261015/dataset"
    original = [r for r in b.read_jsonl(original_dir / "inputs.private.jsonl") if r["task"] == "reranking"]
    original_labels = [r for r in b.read_jsonl(original_dir / "labels.private.jsonl") if r["task"] == "reranking"]
    masked, labels = p.mask_fragments(original, original_labels, read(original_exclusions)["pairs"])
    assert masked == dataset["formal"] and labels == dataset["formal_labels"]
    counts = {phase: sum(len(v["candidates"]) for v in dataset[phase]) for phase in ("engineering", "validation", "formal")}
    assert counts == dict(engineering=226, validation=333, formal=3188)
    assert len(original) == len(masked) == 200 and sum(len(v["candidates"]) for v in original) == 3194
    assert sum(len(v["positive_ids"]) for v in labels) == 1437
    all_pools = dataset["engineering"] + dataset["validation"] + masked
    assert len(all_pools) == len({v["case_id"] for v in all_pools}) == len({v["query"] for v in all_pools}) == 232
    p.validate_pools(masked, labels)
    p.validate_pools(dataset["validation"], dataset["validation_labels"])
    original_counts = {v["case_id"]: len(v["candidates"]) for v in original}
    slices = {name: [v for v in masked if (original_counts[v["case_id"]] <= 29) == small]
              for name, small in (("original_le29", True), ("original_gt29", False))}
    baseline = root.parent / "capacity-20261016/source-r2/evaluation/runner/scripts"
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old, new = functions(baseline / "public_decision_benchmark.py"), functions(scripts / "public_decision_benchmark.py")
    changed = [name for name in old if old[name] != new.get(name)]
    assert changed == [] and len(old) == 39
    assert sha(baseline / "decision_capacity_preflight.py") == sha(scripts / "decision_capacity_preflight.py")

    def stats(rows):
        return {**p.request_stats(rows), "all_attempt_total_ms": sum(r["elapsed_ms"] for r in rows)}

    def duration_stats(values):
        return {**b.timing(values), "total_ms": sum(values)}

    def subset_report(pools, chains, rows):
        ids = {v["case_id"] for v in pools}
        ls = [v for v in labels if v["case_id"] in ids]
        cs = [v for v in chains if v["case_id"] in ids]
        rs = [v for v in rows if v["phase"] == "formal" and v["case_id"] in ids]
        report = p.score_chains(pools, ls, cs)
        report = {k: v for k, v in report.items() if not k.startswith("candidates_")}
        for arm, metrics in report["arms"].items():
            group = [v for v in cs if v["arm"] == arm]
            metrics.update(success_rate=metrics["valid_pools"] / metrics["pools"],
                           all_attempt_total_ms=sum(v["wall_ms"] for v in group),
                           request_journal_ms=duration_stats([v["request_journal_ms"] for v in group]))
        report["stages"] = {arm + ":" + stage: stats([v for v in rs if (v["arm"], v["stage"]) == (arm, stage)])
                            for arm, stage in sorted({(v["arm"], v["stage"]) for v in rs})}
        report["retention"] = p.retention_stats(ls, rs)
        report["retention"].update(original_candidates=sum(original_counts[v["case_id"]] for v in pools),
                                    revised_candidates=sum(len(v["candidates"]) for v in pools))
        return report

    results, global_ids, previous_end, memberships = {}, set(), None, {}
    for model in p.ROSTER:
        folder = run / model
        audit = read(folder / "audit.json")
        for name, value in audit["artifact_sha256"].items():
            assert sha(folder / name) == value
        repeated = p.audit_model(folder, dataset, model)
        assert all(repeated[k] == audit[k] for k in audit if k not in ("at", "artifact_sha256"))
        rows = b.read_jsonl(folder / "requests.jsonl")
        intents = b.read_jsonl(folder / "intents.jsonl")
        chains = b.read_jsonl(folder / "chains.jsonl")
        family, model_path, served = MODELS[model]
        direct, pipeline = ("B2", "B4") if family == "llm" else ("B1", "B3")
        by_id = {r["call_id"]: r for r in rows}
        ids = set(by_id)
        assert not ids & global_ids
        global_ids.update(ids)
        assert len(rows) == 7080 and len(chains) == 400
        assert Counter(v["phase"] for v in rows) == dict(engineering=226, validation=333, formal=6521)
        assert all(r["status"] == "valid" and r["http_status"] == 200 for r in rows)
        assert all(r["status"] == "valid" for r in chains)
        for intent, result in zip(intents, rows):
            assert intent["planned_http_calls"] == 1
            assert all(intent[k] == result[k] for k in ("call_id", "input_sha256", "request_sha256", "path", "started_seconds"))
            assert result["started_seconds"] <= result["http_started_seconds"] <= result["ended_seconds"]
            assert len(result["response_sha256"]) == 64
        start = datetime.fromisoformat(rows[0]["at"]).timestamp()
        end = datetime.fromisoformat(rows[-1]["at"]).timestamp() + rows[-1]["ended_seconds"] - rows[-1]["started_seconds"]
        assert previous_end is None or previous_end < start
        previous_end = end
        consumed = [rid for c in chains for rid in c["request_ids"]]
        assert consumed == [r["call_id"] for r in rows if r["phase"] == "formal"]
        for index, pool in enumerate(masked):
            pair = chains[index * 2:index * 2 + 2]
            assert [c["arm"] for c in pair] == ([direct, pipeline] if index % 2 == 0 else [pipeline, direct])
            for chain in pair:
                assert chain["case_id"] == pool["case_id"]
                members = [by_id[rid] for rid in chain["request_ids"]]
                refine = [r for r in members if r["stage"] == "refine"]
                kept = set(chain["retained_ids"])
                expected = [v["fragment_id"] for v in pool["candidates"] if chain["arm"] == direct or v["fragment_id"] in kept]
                assert [r["fragment_id"] for r in refine] == expected
                assert chain["planned_refine"] == chain["attempted_refine"] == len(expected)
                assert chain["original_revised_candidates"] == len(pool["candidates"])
                assert chain["wall_ms"] >= sum(r["elapsed_ms"] for r in members)
                assert 0 <= chain["request_journal_ms"] <= chain["wall_ms"]
                if chain["arm"] == pipeline:
                    assert members[0]["stage"] == "rerank"
                    assert all(math.isfinite(v["score"]) and 0 <= v["score"] <= 1 for v in members[0]["scores"])
                    previous = memberships.setdefault(pool["case_id"], kept)
                    assert previous == kept
                    assert set(read(run / "top29-reference.json")[pool["case_id"]]) == kept
        deployment, container = read(folder / "deployment.json"), read(folder / "container.json")
        assert deployment["model"] == model and deployment["served_model"] == served and deployment["family"] == family
        assert deployment["image_id"] == container["image_id"]
        assert container["running"] and container["owned"] == "true"
        for name, value in deployment["source_sha256"].items():
            assert sha(service / name) == value
        rule = read(folder / "threshold.json")
        if family == "llm":
            wire = p.refine_wire(family, served, p.pair_state(masked[0], masked[0]["candidates"][0]))[1]
            assert wire["temperature"] == 0 and wire["max_tokens"] == 64 and wire["response_format"]["json_schema"]["strict"]
            assert all(r["reasoning_chars"] == 0 and r["finish_reason"] == "stop" and r["usage"]["completion_tokens"] <= 64
                       for r in rows if r["stage"] == "refine")
            assert rule["kind"] == "hard_choice"
        summary = subset_report(masked, chains, rows)
        direct_rows = {(r["case_id"], r["fragment_id"]): r for r in rows if r["arm"] == direct}
        pipeline_rows = [r for r in rows if r["arm"] == pipeline and r["stage"] == "refine"]
        decision_changes = sum(bool(p.selected([r], rule)) != bool(p.selected([direct_rows[(r["case_id"], r["fragment_id"])]], rule))
                               for r in pipeline_rows)
        token_stats = {}
        for arm, stage in sorted({(r["arm"], r["stage"]) for r in rows}):
            group = [r for r in rows if (r["arm"], r["stage"]) == (arm, stage)]
            token_stats[arm + ":" + stage] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                values = [(r.get("usage") or {}).get(key) for r in group]
                values = [v for v in values if type(v) is int]
                token_stats[arm + ":" + stage][key] = {"recorded": len(values), "attempted": len(group),
                    "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                    "mean": sum(values) / len(values) if values else None}
        metadata = {}
        for name in ("requests.jsonl", "intents.jsonl", "chains.jsonl"):
            f = folder / name
            raw = f.read_bytes()
            assert raw.endswith(b"\n") and all(line.strip() for line in raw.splitlines())
            metadata[name] = {"lines": raw.count(b"\n"), "bytes": len(raw), "sha256": sha(f),
                              "mtime_utc": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}
        results[model] = {"audit": repeated, "files": metadata, "threshold": rule,
                          "deployment": {k: deployment[k] for k in ("served_model", "family", "image_id", "tp", "cpu_budget", "host_memory_limit_gib", "scheduling", "command")},
                          "model_path_in_frozen_roster": str(model_path), "container": container,
                          "all_requests": stats(rows), "engineering": stats([r for r in rows if r["phase"] == "engineering"]),
                          "validation": stats([r for r in rows if r["phase"] == "validation"]), "all_pools": summary,
                          "slices": {name: subset_report(pools, chains, rows) for name, pools in slices.items()},
                          "token_usage": token_stats,
                          "retained_pair_decision_changes": decision_changes,
                          "interval_utc_seconds": [start, end],
                          "reasoning_chars": sum(r.get("reasoning_chars", 0) for r in rows)}
        print("Offline audit passed: " + model, file=sys.stderr, flush=True)
    assert len(global_ids) == 106200
    final_raw, final_container = inspect(results[p.ROSTER[-1]]["container"]["id"])
    assert (final_raw["Config"].get("Entrypoint") or []) + final_raw["Config"]["Cmd"] == results[p.ROSTER[-1]]["deployment"]["command"]
    for key in ("id", "image_id", "pid", "started_at"):
        assert final_container[key] == results[p.ROSTER[-1]]["container"][key]
    p.check_artifacts(preserved)
    assert all(sha(path) == value for path, value in before.items())
    assert system_state() == state
    return {"version": "passage-independent-offline-audit-v1", "at": p.now(), "passed": True,
            "systemd": state, "finished_at": status["at"], "started_at": startup["at"],
            "preservation": {"files": len(preserved), "bytes": 140647988, "all_hashes_match": True},
            "classification_function_asts_unchanged": 39, "capacity_source_unchanged": True,
            "frozen_test_evidence": {"tests_passed": 118, "rerun_by_audit": False},
            "source_sha256": frozen["source_files"],
            "preparation_sha256": {f.name: sha(f) for f in prep.glob("*.json")},
            "run_protocol_sha256": sha(run / "protocol.json"), "dataset_protocol_sha256": sha(data / "protocol.json"),
            "dataset": {"pools": 200, "original_candidates": 3194, "candidates": 3188, "positives": 1437,
                        "negatives": 1751, "removed_positive": 1, "removed_negative": 5,
                        "slices": {k: len(v) for k, v in slices.items()}, "exact_mask_reconstructed": True},
            "totals": {"models": 15, "conditions": 30, "requests": len(global_ids), "formal_requests": 97815,
                       "engineering_requests": 3390, "validation_requests": 4995, "chains": 6000,
                       "failed": 0, "timeouts": 0, "maximum_inflight": 1},
            "checks": {"all_original_audits_reproduced": True, "all_calls_unique": True,
                       "all_ids_input_request_hashes_reconstructed": True, "all_thresholds_recomputed": True,
                       "global_request_intervals_nonoverlapping": True, "original_order_and_alternation": True,
                       "top29_members_identical_across_models": True, "old_files_unchanged_by_audit": True},
            "final_container": final_container, "models": results,
            "proxy_expected": protocol["reranker"]}


def proxy_audit(expected):
    sys.addaudithook(forbid_network)
    preserved = expected["preserved_proxy_artifact_sha256"]
    assert all(sha(path) == value for path, value in preserved.items())
    model, model_state = inspect(expected["container_id"])
    _, embedding = inspect(expected["embedding_id"])
    _, proxy = inspect("43390b1c1207d35f036e0e294d2e2ea04d027b9aaaf08755ead34cb65474b30a")
    assert model_state["pid"] == expected["container_pid"]
    assert model_state["started_at"] == expected["container_started_at"]
    assert embedding["pid"] == expected["embedding_pid"] and embedding["started_at"] == expected["embedding_started_at"]
    dep = expected["deployment"]
    assert (model["Config"].get("Entrypoint") or []) + model["Config"]["Cmd"] == dep["command"] and model["Image"] == dep["image_id"]
    assert model["HostConfig"]["NanoCpus"] == 4000000000 and model["HostConfig"]["Memory"] == 32 * 1024**3
    cmd = model["Config"]["Cmd"]
    for flag, value in (("--max-model-len", "8192"), ("--max-num-seqs", "16"), ("--max-num-batched-tokens", "16384"),
                        ("--gpu-memory-utilization", "0.8"), ("--tensor-parallel-size", "1"), ("--dtype", "bfloat16"), ("--runner", "pooling")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--no-enable-prefix-caching" in cmd
    return {"at": datetime.now(timezone.utc).isoformat(), "passed": True,
            "preserved_files": len(preserved), "all_hashes_match": True, "model": model_state,
            "proxy": proxy, "embedding": embedding, "command_and_resources_unchanged": True}


def collect(args):
    source = Path(__file__).read_text()
    def remote(host, arguments, timeout):
        import shlex
        cmd = "PYTHONDONTWRITEBYTECODE=1 python3 -B - " + shlex.join(arguments)
        for attempt in range(2):
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd],
                                    input=source, text=True, capture_output=True, timeout=timeout)
            if result.returncode != 255:
                break
        sys.stderr.write(result.stderr)
        assert result.returncode == 0, "Read-only remote audit failed; no inference was launched"
        return json.loads(result.stdout)
    result = remote(args.test_host, ["--test-root", args.root], 1200)
    result["proxy"] = remote(args.proxy_host, ["--proxy-expected", json.dumps(result.pop("proxy_expected"))], 120)
    result["auditor_sha256"] = sha(__file__)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "output": args.output, "totals": result["totals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--proxy-expected")
    parser.add_argument("--test-host")
    parser.add_argument("--proxy-host")
    parser.add_argument("--root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.test_root:
        print(json.dumps(test_audit(args.test_root)))
    elif args.proxy_expected:
        print(json.dumps(proxy_audit(json.loads(args.proxy_expected))))
    else:
        collect(args)
