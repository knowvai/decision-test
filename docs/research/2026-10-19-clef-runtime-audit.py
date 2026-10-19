"""Independent, network-free Clef audit and complete performance tables."""
import argparse
from collections import Counter, defaultdict
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


helper = load("d3_audit_helpers", Path(__file__).with_name("2026-10-19-d3-runtime-audit.py"))


def values(tree, key):
    if isinstance(tree, dict):
        return ([tree[key]] if key in tree else []) + [item for v in tree.values() for item in values(v, key)]
    if isinstance(tree, list):
        return [item for v in tree for item in values(v, key)]
    return []


def check_intervals(rows, n, wall, reported_peak):
    events = []
    for row in rows:
        assert 0 <= row["started_seconds"] < row["ended_seconds"] <= wall
        assert abs(row["elapsed_ms"] - (row["ended_seconds"] - row["started_seconds"]) * 1000) < 2
        events.extend(((row["started_seconds"], 1), (row["ended_seconds"], -1)))
    active, peak = 0, 0
    for _, delta in sorted(events):
        active += delta
        peak = max(active, peak)
        assert 0 <= active <= n
    assert active == 0 and peak == reported_peak == min(n, len(rows))


def metrics_delta(meta):
    from prometheus_client.parser import text_string_to_metric_families

    def cumulative(text):
        return {(s.name, tuple(sorted(s.labels.items()))): s.value
                for family in text_string_to_metric_families(text)
                if family.type in ("counter", "histogram", "summary")
                for s in family.samples if s.name.endswith(("_count", "_sum", "_total"))}

    before, after = (cumulative(meta[k]) for k in ("metrics_before", "metrics_after"))
    totals = defaultdict(float)
    for key, value in after.items():
        if key[0].endswith(("_count", "_sum", "_total")):
            delta = value - before.get(key, 0)
            assert delta >= -1e-6, "Counter decreased within deployment"
            totals[key[0]] += delta
    return dict(totals)


def compare(rows, reference):
    result = helper.compare(rows, reference)
    result["unique_inputs_changed"] = len({key[:2] for key, row in rows.items()
                                          if row["prediction"] != reference[key]["prediction"]})
    return result


def main(args):
    sys.dont_write_bytecode = True
    sys.addaudithook(helper.offline)
    c = load("frozen_clef", args.root / "source/docs/research/2026-10-19-clef-runtime-run.py")
    t = c.t
    sys.path.insert(0, str(args.root.parent / "d3-runtime-20261019/offline-audit"))
    data = args.root.parent / "d3-tuning-20261019/dataset"
    protocol = t.read_data(data)
    read_json = lambda path: json.loads(path.read_text())
    prep, run, pre = [read_json(args.root / name) for name in ("preparation.json", "run-protocol.json", "preflight.json")]
    assert prep["profiles"] == c.PROFILES and prep["levels"] == list(c.LEVELS)
    assert run["profile_order"] == list(c.PROFILES) and run["retries"] == 0 and run["fail_closed"]
    assert run["preparation_sha256"] == t.sha(args.root / "preparation.json")
    assert prep["dataset_sha256"] == t.sha(data / "protocol.json") == c.r.DATA_SHA
    assert prep["preflight_sha256"] == t.sha(args.root / "preflight.json")
    assert pre["inference_calls"] == 0 and len(pre["rows"]) == 256 and all(x["supported"] for x in pre["rows"])
    assert c.model_inventory(args.root.parent) == prep["model_inventory"]
    assert read_json(args.root / "status.json")["stage"] == "complete"
    preserved = read_json(args.root / "preservation.json")
    t.capacity.check_artifacts(preserved)
    t.capacity.check_artifacts(prep["source_sha256"])
    restored = read_json(args.root / "restoration.json")
    assert restored["before"] == prep["original"] == run["original"]
    for a, b in zip(restored["before"], restored["after"]):
        assert all(a[k] == b[k] for k in ("id", "name", "image", "command"))
        assert b["running"] and not b["oom"] and a["restart_count"] == b["restart_count"] == 0
    tokens = {(x["task"], x["case_id"], x["fragment_id"]): x for x in pre["rows"]}
    hashes, counts, spans, seen_files = {}, Counter(), [], set()
    results, deployments, maps = {}, {}, {}

    def read(path):
        hashes[str(path.relative_to(args.root))] = t.sha(path)
        return read_json(path)

    def audit_block(folder, name, jobs, question, n, phase, task, repeat):
        path = folder / (task + "-" + name + ".jsonl")
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == len(jobs)
        hashes[str(path.relative_to(args.root))] = t.sha(path)
        seen_files.add(path)
        rows, meta = t.read_rows(path), read(path.with_suffix(".json"))
        assert len(rows) == len(jobs) == meta["requests"] == meta["valid"] and meta["failed"] == 0
        assert meta["model"] == c.MODEL and meta["task"] == task and meta["phase"] == phase
        assert meta["concurrency"] == n and meta["repeat"] == repeat and meta["question"] == question
        assert meta["variant"] == "baseline" and meta["threshold"] is None
        began, ended = (datetime.fromisoformat(meta[k]) for k in ("started_at_utc", "ended_at_utc"))
        assert 0 < meta["wall_seconds"] <= (ended - began).total_seconds()
        for i, (row, job) in enumerate(zip(rows, jobs)):
            assert row["sequence"] == i and row["status"] == "valid" and row["http_status"] == 200 and row["calls"] == 1
            assert all(row[k] == job[k] for k in ("case_id", "fragment_id", "input_sha256", "chars"))
            assert row["input_sha256"] == t.benchmark.digest(job["state"])
            token = tokens[task, job["case_id"], job["fragment_id"]]
            assert token["input_sha256"] == job["input_sha256"] and 0 < token["tokens"] <= pre["limit"] <= 8192
            _, body = c.p.native_choice_request("sglang", {"model": c.MODEL, "state": job["state"], "questions": question})
            assert row["request_sha256"] == t.benchmark.digest(body)
            pred, prob = c.p.native_choice_result("sglang", body, row["response"])
            assert pred == row["prediction"] and prob == row["probabilities"]
            assert row["response"]["usage"] == {"input_tokens": token["tokens"], "output_tokens": 0}
            assert re.fullmatch(r"[0-9a-f]{64}", row["response_sha256"])
            assert began <= datetime.fromisoformat(row["started_at_utc"]) <= ended
        check_intervals(rows, n, meta["wall_seconds"], meta["peak_inflight"])
        summary = t.capacity.block_summary(rows, {k: meta[k] for k in ("wall_seconds", "peak_inflight", "requests", "concurrency")})
        assert all(meta[k] == v for k, v in summary.items())
        assert meta["p99_ms"] == t.benchmark.percentile([x["elapsed_ms"] for x in rows], .99)
        spans.append((began, ended))
        counts[phase] += len(rows)
        metrics = metrics_delta(meta)
        assert metrics["sglang:num_requests_total"] == len(rows)
        assert metrics["sglang:e2e_request_latency_seconds_count"] == len(rows)
        assert metrics["sglang:prompt_tokens_total"] == sum(row["response"]["usage"]["input_tokens"] for row in rows)
        assert metrics.get("sglang:generation_tokens_total", 0) == 0
        return rows, meta, helper.cpu_delta(meta), metrics

    for profile, config in c.PROFILES.items():
        folder = args.root / c.MODEL / profile
        deploy, start, end, settings, startup = [read(folder / name) for name in (
            "deployment.json", "identity.json", "terminal-identity.json", "resolved-settings.json", "startup.json")]
        assert start == end and end["running"] and not end["oom"] and end["restart_count"] == 0
        assert deploy["image_id"] == end["image"] == c.IMAGE
        assert deploy["model"] == deploy["served_model"] == c.MODEL and deploy["family"] == "sglang"
        assert deploy["tp"] == 1 and deploy["cpu_budget"] == 8 and deploy["host_memory_limit_gib"] == 96
        cmd = deploy["command"]
        assert end["command"] == cmd and cmd[:3] == ["python3", "-m", "sglang.launch_server"]
        for flag, value in {"--max-running-requests": config["seqs"], "--max-prefill-tokens": config["tokens"],
                            "--dtype": "bfloat16", "--tp-size": 1, "--context-length": 8192,
                            "--mem-fraction-static": .85, "--host": "127.0.0.1", "--port": 30888}.items():
            assert cmd.count(flag) == 1 and cmd[cmd.index(flag) + 1] == str(value)
        assert "--enable-metrics" in cmd and deploy["env"]["OMP_NUM_THREADS"] == "2"
        graph_configs = values(settings, "cuda_graph_config")
        assert graph_configs and all(g[phase]["backend"] == "disabled" for g in graph_configs for phase in ("decode", "prefill"))
        for key, expected in {"max_running_requests": config["seqs"], "max_prefill_tokens": config["tokens"],
                              "is_embedding": True, "disable_radix_cache": True, "chunked_prefill_size": -1,
                              "enable_metrics": True, "dtype": "bfloat16", "tp_size": 1}.items():
            observed = values(settings, key)
            assert observed and all(x == expected for x in observed), (profile, key, observed)
        read(folder / "model-cards.json")
        gpu_path = folder / "gpu.csv"
        hashes[str(gpu_path.relative_to(args.root))] = t.sha(gpu_path)
        gpu = [[v.strip() for v in row] for row in csv.reader(gpu_path.open()) if len(row) == 5]
        assert gpu and any(row[1] == "0" and float(row[3]) > 1000 for row in gpu)
        gpu_samples = [(datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("Asia/Shanghai")),
                        float(row[2]), float(row[3])) for row in gpu if row[1] == "0"]
        deployments[profile] = {"settings": settings, "loading_seconds": startup["loading_seconds"],
            "image": end["image"], "max_sampled_gpu0_memory_mib": max(float(row[3]) for row in gpu if row[1] == "0")}
        results[profile] = {}
        for task in protocol["variants"]:
            jobs = c.r.workload(data, protocol, task)
            groups = defaultdict(list)
            for name, selected, n, phase, repeat in c.blocks(jobs):
                rows, meta, cpu, metric = audit_block(folder, name, selected, protocol["variants"][task]["baseline"],
                                                     n, phase, task, repeat)
                if phase == "performance":
                    groups[n].append((rows, meta, cpu, metric))
            results[profile][task] = {}
            for n, blocks in groups.items():
                rows = [row for rows, *_ in blocks for row in rows]
                wall = sum(meta["wall_seconds"] for _, meta, *_ in blocks)
                row_map = {(row["case_id"], row["fragment_id"], meta["repeat"]): row
                           for rows, meta, *_ in blocks for row in rows}
                assert len(row_map) == 256
                maps[profile, task, n] = row_map
                metric = Counter()
                for *_, delta in blocks:
                    metric.update(delta)
                windows = [(datetime.fromisoformat(meta["started_at_utc"]), datetime.fromisoformat(meta["ended_at_utc"]))
                           for _, meta, *_ in blocks]
                samples = [sample for sample in gpu_samples if any(began <= sample[0] <= ended for began, ended in windows)]
                results[profile][task][str(n)] = {
                    "attempted": len(rows), "valid": len(rows), "failed": 0, "timeouts": 0, "not_attempted": 0,
                    "wall_seconds": wall, "rps": len(rows) / wall,
                    "mean_ms": statistics.mean(row["elapsed_ms"] for row in rows),
                    "p95_ms": t.benchmark.percentile([row["elapsed_ms"] for row in rows], .95),
                    "p99_ms": t.benchmark.percentile([row["elapsed_ms"] for row in rows], .99),
                    "all_attempt_elapsed_ms": sum(row["elapsed_ms"] for row in rows),
                    "rounds": [{"repeat": meta["repeat"], "rps": len(rows) / meta["wall_seconds"],
                                "p95_ms": t.benchmark.percentile([x["elapsed_ms"] for x in rows], .95)}
                               for rows, meta, *_ in blocks],
                    "cpu_cores_mean": sum(cpu["cpu_seconds"] for _, _, cpu, _ in blocks) / wall,
                    "gpu_util_mean_sampled": statistics.mean(x[1] for x in samples) if samples else None,
                    "gpu_memory_mib_max_sampled": max((x[2] for x in samples), default=None),
                    "gpu_samples": len(samples),
                    "quality": helper.quality(rows, jobs, task), "metric_deltas": dict(metric)}
    for profile, tasks in results.items():
        for task, levels in tasks.items():
            for n, result in levels.items():
                base = results["baseline"][task][n]
                closing = results["baseline-end"][task][n]
                result["vs_baseline"] = compare(maps[profile, task, int(n)], maps["baseline", task, int(n)])
                result["vs_n1"] = compare(maps[profile, task, int(n)], maps[profile, task, 1])
                result["rps_ratio"] = result["rps"] / base["rps"]
                result["p95_ratio"] = result["p95_ms"] / base["p95_ms"]
                result["baseline_end_rps_drift"] = closing["rps"] / base["rps"] - 1
                result["promising_candidate"] = (
                    profile not in ("baseline", "baseline-end") and
                    all(a["rps"] >= b["rps"] * 1.05 for a, b in zip(result["rounds"], base["rounds"])) and
                    result["p95_ratio"] <= 1.1 and result["vs_baseline"]["decision_changes"] == 0 and
                    result["vs_n1"]["decision_changes"] == 0 and abs(result["baseline_end_rps_drift"]) <= .1)
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert set((args.root / c.MODEL).glob("*/*.jsonl")) == seen_files
    assert sum(counts.values()) == c.CALLS and len(seen_files) == c.BLOCKS
    import prometheus_client
    parser_sources = {str(path): t.sha(path) for path in Path(prometheus_client.__file__).parent.rglob("*.py")}
    output = {"passed": True, "model": c.MODEL, "image": c.IMAGE, "at_utc": datetime.now(timezone.utc).isoformat(),
              "profiles": c.PROFILES, "levels": c.LEVELS, "calls": counts, "total_calls": sum(counts.values()),
              "blocks": len(seen_files), "preserved_files": len(preserved), "frozen_files": len(prep["source_sha256"]),
              "performance": results, "deployments": deployments, "artifact_sha256": hashes,
              "auditor_sha256": t.sha(__file__), "helper_sha256": t.sha(helper.__file__),
              "prometheus_parser_sources": parser_sources,
              "limitations": ["256 unique reused development inputs, not unseen quality evaluation",
                              "closed-loop short blocks, not production SLA or maximum sustainable capacity",
                              "one cold deployment/candidate; two within-process rounds",
                              "GPU sampling does not measure exact peaks; native batch telemetry may be unavailable",
                              "GPU execution established by deployed native CUDA model and resource observations, not per-response CUDA metadata"]}
    t.save(args.output, output)
    print(json.dumps({k: output[k] for k in ("passed", "total_calls", "blocks", "calls", "preserved_files", "frozen_files")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
