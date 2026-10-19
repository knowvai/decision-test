"""Independent offline audit. Reject network/subprocess use and create a new output."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
from zoneinfo import ZoneInfo


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offline(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
        raise RuntimeError("Offline audit forbids network and subprocesses")


def parse_metrics(text):
    from prometheus_client.parser import text_string_to_metric_families
    return {(sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in text_string_to_metric_families(text) for sample in family.samples}


def metrics_delta(meta):
    before, after = (parse_metrics(meta[key]) for key in ("metrics_before", "metrics_after"))
    names = ("vllm_srun_batch_rows_count", "vllm_srun_batch_rows_sum",
             "vllm_srun_forward_duration_seconds_sum", "vllm_srun_queue_duration_seconds_count",
             "vllm_srun_queue_duration_seconds_sum", "vllm_srun_requests_total")
    delta = {name: sum(value - before.get(key, 0) for key, value in after.items() if key[0] == name)
             for name in names}
    assert all(value >= 0 for value in delta.values())
    assert delta["vllm_srun_requests_total"] == meta["requests"]
    assert delta["vllm_srun_batch_rows_count"] > 0
    assert delta["vllm_srun_batch_rows_sum"] == meta["requests"]
    return delta


def cpu_delta(meta):
    before, after = meta["cpu_before"], meta["cpu_after"]
    assert before["quota"] == after["quota"]
    stats = [{k: int(v) for k, v in (line.split() for line in item["stats"].splitlines())}
             for item in (before, after)]
    delta = {k: stats[1][k] - value for k, value in stats[0].items()}
    assert all(v >= 0 for v in delta.values())
    seconds = ((after["usage_ns"] - before["usage_ns"]) / 1e9 if "usage_ns" in before
               else delta["usage_usec"] / 1e6)
    return {"cpu_seconds": seconds, "periods": delta["nr_periods"], "throttled_periods": delta["nr_throttled"]}


def compare(rows, reference):
    assert rows.keys() == reference.keys()
    return {"decision_changes": sum(r["prediction"] != reference[k]["prediction"] for k, r in rows.items()),
            "max_probability_delta": max(abs(v - reference[k]["probabilities"][label])
                                         for k, r in rows.items() for label, v in r["probabilities"].items())}


def quality(rows, jobs, task):
    gold = {(j["case_id"], j["fragment_id"]): j["gold"] for j in jobs}
    out = {"unique_inputs": len(gold), "accuracy": sum(row["prediction"] == gold[row["case_id"], row["fragment_id"]]
                                                     for row in rows) / len(rows)}
    if task == "evidence":
        tp = sum(r["prediction"] == "relevant" and gold[r["case_id"], r["fragment_id"]] == "relevant" for r in rows)
        fp = sum(r["prediction"] == "relevant" and gold[r["case_id"], r["fragment_id"]] != "relevant" for r in rows)
        fn = sum(r["prediction"] != "relevant" and gold[r["case_id"], r["fragment_id"]] == "relevant" for r in rows)
        out.update(tp=tp, fp=fp, fn=fn, tn=len(rows)-tp-fp-fn,
                   precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                   f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return out


def main(args):
    sys.dont_write_bytecode = True
    sys.addaudithook(offline)
    r = load("frozen_runtime", args.root / "source/docs/research/2026-10-19-d3-runtime-run.py")
    t = r.t
    native = r.p.native_choice_result
    data = args.root.parent / "d3-tuning-20261019/dataset"
    protocol = t.read_data(data)
    prep = json.loads((args.root / "preparation.json").read_text())
    run = json.loads((args.root / "run-protocol.json").read_text())
    assert prep["profiles"] == r.PROFILES and prep["levels"] == list(r.LEVELS)
    assert run["preparation_sha256"] == t.sha(args.root / "preparation.json")
    assert t.sha(data / "protocol.json") == prep["dataset_sha256"] == r.DATA_SHA
    assert run["retries"] == 0 and run["fail_closed"] is True
    assert run["profile_order"] == {m: r.profile_order(m) for m in t.MODELS}
    assert json.loads((args.root / "status.json").read_text())["stage"] == "complete"
    t.capacity.check_artifacts(prep["source_sha256"])
    preserved = json.loads((args.root / "preservation.json").read_text())
    t.capacity.check_artifacts(preserved)
    restored = json.loads((args.root / "restoration.json").read_text())
    assert restored["before"] == run["original"] == prep["original"]
    assert len(restored["after"]) == 2
    for a, b in zip(restored["before"], restored["after"]):
        assert all(a[k] == b[k] for k in ("id", "name", "image", "command"))
        assert b["running"] and not b["oom"] and b["restart_count"] == a["restart_count"] == 0
    hashes, spans, counts, seen_files = {}, [], Counter(), set()
    model_hashes = {model: json.loads((args.root.parent / "models" / model / "MODEL_MANIFEST.json").read_text())
                    ["identity"]["model_sha256"] for model in t.MODELS}
    token_counts = {(model, row["task"], row["case_id"], row["fragment_id"]): row["tokens"]
                    for model, preflight in prep["preflights"].items() for row in preflight["rows"]}

    def read(path):
        hashes[str(path.relative_to(args.root))] = t.sha(path)
        return json.loads(path.read_text())

    def audit_block(folder, name, jobs, question, n, task, phase, repeat):
        path = folder / (task + "-" + name + ".jsonl")
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == len(jobs)
        rows = t.read_rows(path)
        meta = read(path.with_suffix(".json"))
        hashes[str(path.relative_to(args.root))] = t.sha(path)
        seen_files.add(path)
        assert len(rows) == len(jobs) == meta["requests"] == meta["valid"]
        assert meta["model"] == folder.parent.name and meta["task"] == task and meta["phase"] == phase
        assert meta["concurrency"] == n and meta["repeat"] == repeat and meta["failed"] == 0
        assert meta["question"] == question and meta["variant"] == "baseline" and meta["threshold"] is None
        began, ended = (datetime.fromisoformat(meta[key]) for key in ("started_at_utc", "ended_at_utc"))
        assert 0 < meta["wall_seconds"] <= (ended - began).total_seconds()
        events = []
        for i, (row, job) in enumerate(zip(rows, jobs)):
            assert row["sequence"] == i and row["status"] == "valid" and row["http_status"] == 200 and row["calls"] == 1
            assert all(row[k] == job[k] for k in ("case_id", "fragment_id", "input_sha256", "chars"))
            assert row["input_sha256"] == t.benchmark.digest(job["state"])
            assert began <= datetime.fromisoformat(row["started_at_utc"]) <= ended
            _, body = t.wire(meta["model"], job, question)
            body["options"] = {"return_meta": True}
            assert row["request_sha256"] == t.benchmark.digest(body)
            prediction, probabilities = native("worker", body, row["response"])
            assert prediction == row["prediction"] and probabilities == row["probabilities"]
            native_meta = row["response"]["meta"]
            assert native_meta["device"] == "cuda:0" and native_meta["accelerator"] == "cuda"
            assert native_meta["engine"] == "native" and native_meta["dtype"] == "bfloat16"
            assert native_meta["numerics"] == "approximate" and native_meta["model_id"] == meta["model"]
            assert native_meta["model_sha256"] == model_hashes[meta["model"]]
            assert row["response"]["usage"]["input_tokens"] == token_counts[meta["model"], task, job["case_id"], job["fragment_id"]]
            assert row["response"]["usage"]["output_tokens"] == 0
            assert re.fullmatch(r"[0-9a-f]{64}", row["response_sha256"])
            assert 0 <= row["started_seconds"] < row["ended_seconds"] <= meta["wall_seconds"]
            assert abs(row["elapsed_ms"] - (row["ended_seconds"]-row["started_seconds"])*1000) < 2
            events.extend(((row["started_seconds"], 1), (row["ended_seconds"], -1)))
        active, peak = 0, 0
        for _, delta in sorted(events):
            active += delta
            peak = max(peak, active)
            assert 0 <= active <= n
        assert active == 0 and peak == meta["peak_inflight"] == min(n, len(rows))
        summary = t.capacity.block_summary(rows, {k: meta[k] for k in ("wall_seconds", "peak_inflight", "requests", "concurrency")})
        assert all(meta[k] == v for k, v in summary.items())
        assert meta["p99_ms"] == t.benchmark.percentile([x["elapsed_ms"] for x in rows], .99)
        spans.append((began, ended))
        counts[phase] += len(rows)
        return rows, meta, metrics_delta(meta), cpu_delta(meta)

    report = {"models": {}, "passed": False, "plan": "2026-10-19-d3-runtime-parameters.md",
              "dataset_sha256": r.DATA_SHA, "profiles": r.PROFILES, "levels": r.LEVELS,
              "limitations": ["reused 128 inputs/task, not unseen quality data", "closed-loop short blocks, not production SLA",
                              "one cold deployment/alternative; two within-process rounds", "sampled GPU values, not exact peaks",
                              "runtime startup/calibration is extra internal GPU work", "live internal coalesced budget unobserved"]}
    for model in t.MODELS:
        performance, deployment_info, row_maps = {}, {}, {}
        pre = prep["preflights"][model]
        assert t.sha(pre["source"]) == pre["sha256"]
        native_pre = json.loads(Path(pre["source"]).read_text())
        assert native_pre["manifest_sha256"] == t.sha(args.root.parent / "models" / model / "MODEL_MANIFEST.json")
        for task in protocol["variants"]:
            jobs = r.workload(data, protocol, task)
            encoded = {(x["case_id"], x["fragment_id"]): x for x in pre["rows"] if x["task"] == task}
            assert len(encoded) == len(jobs)
            assert all(encoded[j["case_id"], j["fragment_id"]]["input_sha256"] == j["input_sha256"] and
                       0 < encoded[j["case_id"], j["fragment_id"]]["tokens"] <= 8192 for j in jobs)
        for profile, config in r.PROFILES.items():
            folder = args.root / model / profile
            deploy = read(folder / "deployment.json")
            initial, terminal = (read(folder / name) for name in ("identity.json", "terminal-identity.json"))
            assert initial == terminal
            assert terminal["running"] and not terminal["oom"] and terminal["restart_count"] == 0
            assert deploy["model"] == deploy["served_model"] == model and deploy["image_id"] == terminal["image"] == r.IMAGE
            assert deploy["tp"] == 1 and deploy["cpu_budget"] == 8 and deploy["host_memory_limit_gib"] == 96
            cmd = deploy["command"]
            assert terminal["command"] == cmd
            assert cmd[:3] == ["vllm-srun", "serve", "/model"]
            flags = {"--device": "cuda:0", "--dtype": "auto", "--served-model-name": model,
                     "--threads": "2", "--batch-window-ms": str(config["batch_window_ms"]),
                     "--result-cache-entries": "0", "--load-attempts": "1", "--max-queue": "256"}
            for flag, value in flags.items():
                assert cmd.count(flag) == 1 and cmd[cmd.index(flag)+1] == value
            assert "--offline" in cmd and "--no-enable-shared-context" in cmd
            assert deploy["env"]["OMP_NUM_THREADS"] == "2"
            resolved = read(folder / "resolved-limits.json")
            match = re.search(r"max_num_seqs (\d+), max_num_batched_tokens (\d+) from (.*?), batch_window_ms ([\d.]+), dtype (\w+)", resolved["serving"][0])
            assert match and float(match[4]) == config["batch_window_ms"]
            seqs, tokens = int(match[1]), int(match[2])
            if profile == "auto":
                assert "--max-num-seqs" not in cmd and "--max-num-batched-tokens" not in cmd
                assert resolved["auto_requested"] and "calibrated" in match[3]
            else:
                assert cmd[cmd.index("--max-num-seqs")+1] == str(config["seqs"]) == str(seqs)
                assert cmd[cmd.index("--max-num-batched-tokens")+1] == str(config["tokens"]) == str(tokens)
            read(folder / "model-cards.json")
            gpu_path = folder / "gpu.csv"
            hashes[str(gpu_path.relative_to(args.root))] = t.sha(gpu_path)
            gpu = []
            with gpu_path.open() as stream:
                for timestamp, index, utilization, memory, power in csv.reader(stream):
                    if index.strip() == "0":
                        when = datetime.strptime(timestamp.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                        gpu.append((when, float(utilization), float(memory), float(power)))
            assert gpu
            deployment_info[profile] = {"resolved_seqs": seqs, "resolved_tokens": tokens, "source": match[3],
                                        "dtype": match[5], "gpu_samples": len(gpu),
                                        "loading_seconds": (datetime.fromtimestamp((folder/"identity.json").stat().st_mtime, timezone.utc) -
                                                            datetime.fromisoformat(initial["started_at"][:26]+"+00:00")).total_seconds()}
            performance[profile] = {}
            for task, variants in protocol["variants"].items():
                jobs = r.workload(data, protocol, task)
                measured = {}
                for name, selected, n, phase, repeat in r.blocks(jobs):
                    result = audit_block(folder, name, selected, variants["baseline"], n, task, phase, repeat)
                    if phase == "performance":
                        measured[n, repeat] = result
                        row_maps[profile, task, n, repeat] = {(x["case_id"], x["fragment_id"]): x for x in result[0]}
                levels = {}
                for n in r.LEVELS:
                    rows = [x for repeat in (1, 2) for x in measured[n, repeat][0]]
                    wall = sum(measured[n, repeat][1]["wall_seconds"] for repeat in (1, 2))
                    deltas = {key: sum(measured[n, repeat][2][key] for repeat in (1, 2)) for key in measured[n, 1][2]}
                    batches = deltas["vllm_srun_batch_rows_count"]
                    samples = [x for x in gpu if any(datetime.fromisoformat(measured[n, rep][1]["started_at_utc"]) <= x[0] <=
                                                      datetime.fromisoformat(measured[n, rep][1]["ended_at_utc"]) for rep in (1, 2))]
                    cpu = {k: sum(measured[n, rep][3][k] for rep in (1, 2)) for k in measured[n, 1][3]}
                    levels[str(n)] = {"valid": len(rows), "attempted": len(rows), "failed": 0, "timeout": 0, "not_attempted": 0,
                        "wall_seconds": wall, "rps": len(rows)/wall, **t.benchmark.timing([x["elapsed_ms"] for x in rows]),
                        "p99_ms": t.benchmark.percentile([x["elapsed_ms"] for x in rows], .99),
                        "quality": quality(rows, jobs, task), "metrics": deltas,
                        "batch_rows_mean": deltas["vllm_srun_batch_rows_sum"]/batches,
                        "forward_ms_mean": 1000*deltas["vllm_srun_forward_duration_seconds_sum"]/batches,
                        "queue_ms_mean": 1000*deltas["vllm_srun_queue_duration_seconds_sum"]/deltas["vllm_srun_queue_duration_seconds_count"],
                        "cpu_cores_mean": cpu["cpu_seconds"]/wall,
                        "cpu_throttled_period_fraction": cpu["throttled_periods"]/cpu["periods"] if cpu["periods"] else 0,
                        "gpu_samples": len(samples), "gpu_util_mean": statistics.mean(x[1] for x in samples) if samples else None,
                        "gpu_memory_mib_max_sampled": max((x[2] for x in samples), default=None),
                        "rounds": [{"rps": 128/measured[n, rep][1]["wall_seconds"],
                                    "p95_ms": measured[n, rep][1]["successful_latency_ms"]["p95_ms"]} for rep in (1, 2)]}
                    assert 1 <= levels[str(n)]["batch_rows_mean"] <= seqs
                performance[profile][task] = levels
        for profile, tasks in performance.items():
            for task, levels in tasks.items():
                for n, value in levels.items():
                    base = performance["baseline"][task][n]
                    end = performance["baseline-end"][task][n]
                    pairs = [compare(row_maps[profile, task, int(n), rep], row_maps["baseline", task, int(n), rep]) for rep in (1, 2)]
                    own = [compare(row_maps[profile, task, int(n), rep], row_maps[profile, task, 1, rep]) for rep in (1, 2)]
                    value.update(changes_vs_baseline=sum(x["decision_changes"] for x in pairs),
                                 max_probability_delta_vs_baseline=max(x["max_probability_delta"] for x in pairs),
                                 changes_vs_n1=sum(x["decision_changes"] for x in own),
                                 rps_ratio=value["rps"]/base["rps"], p95_ratio=value["p95_ms"]/base["p95_ms"],
                                 baseline_end_rps_drift=end["rps"]/base["rps"]-1)
                    value["promising_candidate"] = (profile not in ("baseline", "baseline-end") and
                        all(a["rps"] >= 1.05*b["rps"] for a, b in zip(value["rounds"], base["rounds"])) and
                        value["p95_ratio"] <= 1.1 and value["changes_vs_baseline"] == 0 and
                        abs(value["baseline_end_rps_drift"]) <= .1)
        report["models"][model] = {"performance": performance, "deployments": deployment_info}
    assert seen_files == {x for model in t.MODELS for x in (args.root/model).glob("*/*.jsonl")}
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert len(seen_files) == prep["planned_blocks"] == 432
    assert dict(counts) == {"warmup": 4608, "performance": 36864}
    assert sum(counts.values()) == prep["planned_http_calls"] == 41472
    for name in ("preparation.json", "run-protocol.json", "preservation.json", "restoration.json", "status.json", "tests-final.xml"):
        hashes[name] = t.sha(args.root/name)
    import prometheus_client
    parser_root = Path(prometheus_client.__file__).parent
    report.update(passed=True, total_calls=sum(counts.values()), calls=dict(counts), blocks=len(seen_files),
                  preserved_files=len(preserved), frozen_files=len(prep["source_sha256"]), artifact_sha256=hashes,
                  audited_at_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=t.sha(Path(__file__)),
                  parser_source_sha256={str(p.relative_to(parser_root)): t.sha(p) for p in parser_root.rglob("*.py")},
                  restoration=restored)
    t.save(args.output, report)
    print(json.dumps({k: report[k] for k in ("passed", "total_calls", "calls", "blocks", "preserved_files", "frozen_files")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
