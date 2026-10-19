"""Read-only, no-network audit of the D3 pilot; writes only a new report."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys


def check_restoration(original, restoration):
    assert restoration["checked"] and restoration["before"] == original
    assert len(restoration["after"]) == len(original) == 2
    for before, after in zip(original, restoration["after"]):
        for key in ("id", "name", "image", "command"):
            assert before[key] == after[key], key
        assert after["running"] and not after["oom"] and after["pid"] > 0
        assert before["restart_count"] == after["restart_count"] == 0


def check_deployment(deploy, model, config, image):
    assert deploy["model"] == deploy["served_model"] == model and deploy["family"] == "worker"
    assert deploy["tp"] == 1 and deploy["cpu_budget"] == 8 and deploy["host_memory_limit_gib"] == 96
    assert deploy["image_id"] == image
    command = deploy["command"]
    assert command[:3] == ["vllm-srun", "serve", "/model"]
    expected = {"--device": "cuda:0", "--dtype": "auto", "--served-model-name": model,
                "--threads": str(config["worker_threads"]), "--max-num-seqs": str(config["seqs"]),
                "--max-num-batched-tokens": str(config["tokens"]),
                "--batch-window-ms": str(config["batch_window_ms"]),
                "--result-cache-entries": "0", "--load-attempts": "1"}
    for flag, value in expected.items():
        assert command.count(flag) == 1 and command[command.index(flag)+1] == value, flag
    assert "--no-enable-shared-context" in command and "--offline" in command
    assert deploy["env"]["OMP_NUM_THREADS"] == str(config["worker_threads"])


def main(args):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("d3_tuning", args.root / "source/docs/research/2026-10-19-d3-tuning.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    from deployment_probe import native_choice_result
    data = args.root / "dataset"
    protocol = t.read_data(data)
    run = json.loads((args.root / "run-protocol.json").read_text())
    assert run["models"] == list(t.MODELS) and run["profiles"] == t.PROFILES
    assert run["levels"] == list(t.LEVELS) and run["retries"] == 0
    assert json.loads((args.root / "status.json").read_text())["stage"] == "complete"
    assert t.sha(data / "protocol.json") == run["dataset_sha256"]
    t.capacity.check_artifacts(run["source_sha256"])
    preservation = json.loads((args.root / "preservation.json").read_text())
    t.capacity.check_artifacts(preservation)
    restoration = json.loads((args.root / "restoration.json").read_text())
    check_restoration(run["original_service"], restoration)
    sources, blocks, counts, spans = {}, {}, Counter(), []

    def read_block(folder, name, task, split, question, phase, n, subset=None):
        path = folder / (name + ".jsonl")
        meta_path = folder / (name + ".json")
        assert path.read_bytes().endswith(b"\n")
        rows = t.read_rows(path)
        meta = json.loads(meta_path.read_text())
        assert meta["model"] == folder.parent.name and meta["task"] == task
        jobs = t.read_rows(data / f"{task}-{split}.private.jsonl")
        if subset is not None:
            jobs = subset
        assert len(rows) == len(jobs) == meta["requests"] == meta["valid"]
        assert meta["failed"] == 0 and meta["phase"] == phase and meta["concurrency"] == n
        assert meta["question"] == question
        events = []
        began, ended = (datetime.fromisoformat(meta[key]) for key in ("started_at_utc", "ended_at_utc"))
        assert 0 < meta["wall_seconds"] <= (ended-began).total_seconds()
        for index, (row, job) in enumerate(zip(rows, jobs)):
            assert row["sequence"] == index and row["status"] == "valid" and row["http_status"] == 200
            assert row["calls"] == 1
            for key in ("case_id", "fragment_id", "input_sha256", "chars"):
                assert row[key] == job[key]
            assert row["input_sha256"] == t.benchmark.digest(job["state"])
            assert began <= datetime.fromisoformat(row["started_at_utc"]) <= ended
            _, body = t.wire(meta["model"], job, question)
            body["options"] = {"return_meta": True}
            assert row["request_sha256"] == t.benchmark.digest(body)
            choice, probabilities = native_choice_result("worker", body, row["response"])
            assert choice == row["prediction"] and probabilities == row["probabilities"]
            assert row["response"]["meta"]["device"].startswith("cuda")
            assert 0 <= row["started_seconds"] < row["ended_seconds"] <= meta["wall_seconds"]
            assert abs(row["elapsed_ms"] - (row["ended_seconds"] - row["started_seconds"])*1000) < 2
            events.extend(((row["started_seconds"], 1), (row["ended_seconds"], -1)))
        active, peak = 0, 0
        for _, delta in sorted(events):
            active += delta
            peak = max(peak, active)
            assert 0 <= active <= n
        assert active == 0 and peak == meta["peak_inflight"] == min(n, len(rows))
        expected = t.capacity.block_summary(rows, {k: meta[k] for k in
                       ("wall_seconds", "peak_inflight", "requests", "concurrency")})
        for key, value in expected.items():
            assert meta[key] == value, key
        assert meta["p99_ms"] == t.benchmark.percentile([r["elapsed_ms"] for r in rows], .99)
        sources[str(path.relative_to(args.root))] = t.sha(path)
        sources[str(meta_path.relative_to(args.root))] = t.sha(meta_path)
        spans.append((began, ended))
        counts[phase] += len(rows)
        blocks[str(path)] = (rows, meta)
        return rows, meta

    report = {"models": {}, "dataset": protocol["summary"], "protocol_sha256": run["dataset_sha256"],
              "limitations": ["pilot; selection on development only", "closed-loop finite workload, not production SLA",
                              "two repetitions are not independent quality samples",
                              "T2Ranking proxy labels, not final RAG answer correctness",
                              "quality and performance deployment comparisons remain separate"]}
    for model in t.MODELS:
        model_root, quality = args.root / model, {}
        folder = model_root / "quality"
        quality_deployment = json.loads((folder / "deployment.json").read_text())
        check_deployment(quality_deployment, model, t.PROFILES["baseline"],
                         "sha256:f3ccfb750e184c66e3aee485d64b7ed19cfc0ce83d73650b8b92b3463a3a66b4")
        preflight = json.loads((args.root / f"preflight-{model}.json").read_text())
        assert preflight["manifest_sha256"] == t.sha(args.root.parent / "models" / model / "MODEL_MANIFEST.json")
        assert all(r["supported"] for r in preflight["rows"]) and preflight["inference_calls"] == 0
        expected_encoding = {(task, split, variant, job["case_id"], job["fragment_id"]): job["input_sha256"]
                             for task, variants in protocol["variants"].items()
                             for split in ("dev", "holdout") for variant in variants
                             for job in t.read_rows(data / f"{task}-{split}.private.jsonl")}
        encoded = {(row["task"], row["split"], row["variant"], row["case_id"], row["fragment_id"]): row["input_sha256"]
                   for row in preflight["rows"]}
        assert encoded == expected_encoding and len(encoded) == len(preflight["rows"])
        assert all(0 < row["tokens"] <= min(preflight["limit"], 32768) for row in preflight["rows"])
        for task, variants in protocol["variants"].items():
            jobs = t.read_rows(data / f"{task}-dev.private.jsonl")
            for repeat in (1, 2):
                read_block(folder, f"{task}-warm-r{repeat}", task, "dev", variants["baseline"], "warmup", 1, jobs[:16])
            by_variant = {variant: read_block(folder, f"{task}-dev-{variant}", task, "dev", question, "dev", 1)[0]
                          for variant, question in variants.items()}
            chosen = t.selection(data, task, by_variant)
            assert chosen == json.loads((folder / f"{task}-selection.json").read_text())
            # The choice must be written before any holdout request is submitted.
            chosen_mtime = (folder / f"{task}-selection.json").stat().st_mtime
            held, held_timing = {}, {}
            for variant in dict.fromkeys(("baseline", chosen["variant"])):
                rows, meta = read_block(folder, f"{task}-holdout-{variant}", task, "holdout", variants[variant], "holdout", 1)
                assert chosen_mtime <= datetime.fromisoformat(meta["started_at_utc"]).timestamp()
                held[variant] = rows
                held_timing[variant] = {"wall_seconds": meta["wall_seconds"],
                                        **t.benchmark.timing([r["elapsed_ms"] for r in rows]),
                                        "p99_ms": meta["p99_ms"]}
            baseline = t.score(data, task, "holdout", held["baseline"])
            selected = t.score(data, task, "holdout", held[chosen["variant"]], chosen["threshold"])
            if task == "classification":
                labels = {j["case_id"]: j["gold"] for j in t.read_rows(data / "classification-holdout.private.jsonl")}
                predictions = lambda rs: {1: {r["case_id"]: r["prediction"] for r in rs}}
                delta = t.benchmark.classification_interval(labels, predictions(held[chosen["variant"]]), predictions(held["baseline"]))
            else:
                left = {c["case_id"]: c for c in baseline["cases"] if c["gold_count"]}
                right = {c["case_id"]: c for c in selected["cases"] if c["gold_count"]}
                delta = {key: t.benchmark.bootstrap_mean([right[c][key]-left[c][key] for c in left])
                         for key in ("precision", "recall", "f1", "missed_all")}
            quality[task] = {"selection": chosen, "baseline": baseline, "selected": selected, "paired_delta": delta,
                             "timing_by_variant": held_timing,
                             "same_response_reused": chosen["variant"] == "baseline"}

        performance, baseline_predictions = {}, {}
        for profile in t.PROFILES:
            folder = model_root / profile
            deploy = json.loads((folder / "deployment.json").read_text())
            check_deployment(deploy, model, t.PROFILES[profile], quality_deployment["image_id"])
            terminal = json.loads((folder / "terminal-identity.json").read_text())
            initial = json.loads((folder / "identity.json").read_text())
            assert terminal["id"] == initial["id"] and terminal["image"] == deploy["image_id"]
            assert terminal["command"] == initial["command"] and terminal["started_at"] == initial["started_at"]
            assert terminal["running"] and not terminal["oom"] and terminal["restart_count"] == 0
            performance[profile] = {}
            for task, choices in protocol["variants"].items():
                selected = quality[task]["selection"]
                question = choices[selected["variant"]]
                keys = {(r["case_id"], r["fragment_id"]) for r in protocol["performance"][task]}
                jobs = [j for j in t.read_rows(data / f"{task}-dev.private.jsonl") if (j["case_id"], j["fragment_id"]) in keys]
                ordered = sorted(jobs, key=lambda r: r["chars"])
                warm = [ordered[i*(len(ordered)-1)//15] for i in range(16)]
                for n in t.LEVELS:
                    for repeat in (1, 2):
                        read_block(folder, f"{task}-warm-n{n}-r{repeat}", task, "dev", question, "warmup", n, warm)
                measured = {(n,r): read_block(folder, f"{task}-n{n}-r{r}", task, "dev", question, "performance", n, jobs)
                            for n in t.LEVELS for r in (1, 2)}
                levels = {}
                for n in t.LEVELS:
                    rows = [row for r in (1, 2) for row in measured[n,r][0]]
                    wall = sum(measured[n,r][1]["wall_seconds"] for r in (1, 2))
                    adjusted = t.threshold_rows(rows, selected["threshold"])
                    gold = {(j["case_id"], j["fragment_id"]): j["gold"] for j in jobs}
                    reference = [row for r in (1, 2) for row in measured[1,r][0]]
                    reference = t.threshold_rows(reference, selected["threshold"])
                    if profile == "baseline":
                        baseline_predictions[task,n] = adjusted
                    cross_reference = baseline_predictions[task,n]
                    levels[str(n)] = {"valid": len(rows), "failed": 0, "wall_seconds": wall,
                        "rps": len(rows)/wall, **t.benchmark.timing([r["elapsed_ms"] for r in rows]),
                        "p99_ms": t.benchmark.percentile([r["elapsed_ms"] for r in rows], .99),
                        "accuracy": sum(r["prediction"] == gold[r["case_id"],r["fragment_id"]] for r in adjusted)/len(rows),
                        "changes_vs_same_round_n1": sum(a["prediction"] != b["prediction"] for a,b in zip(adjusted, reference)),
                        "changes_vs_baseline_same_n": sum(a["prediction"] != b["prediction"] for a,b in zip(adjusted, cross_reference)),
                        "peak_inflight": max(measured[n,r][1]["peak_inflight"] for r in (1,2)),
                        "rounds": [{"rps": len(jobs)/measured[n,r][1]["wall_seconds"],
                                    "p95_ms": measured[n,r][1]["successful_latency_ms"]["p95_ms"]} for r in (1,2)]}
                performance[profile][task] = levels
        for profile in performance:
            for task, levels in performance[profile].items():
                for n, value in levels.items():
                    ref = performance["baseline"][task][n]
                    value["rps_ratio_vs_baseline_same_n"] = value["rps"]/ref["rps"]
                    value["p95_ratio_vs_baseline_same_n"] = value["p95_ms"]/ref["p95_ms"]
        report["models"][model] = {"quality": quality, "performance": performance}
    actual = {str(p) for model in t.MODELS for p in (args.root / model).glob("*/*.jsonl")}
    assert actual == set(blocks), "Unaccounted or missing call files"
    spans.sort()
    assert all(left[1] <= right[0] for left,right in zip(spans,spans[1:])), "Overlapping experiment blocks"
    controls = [args.root / name for name in ("run-protocol.json", "preservation.json", "restoration.json", "status.json")]
    controls += [data / "protocol.json"]
    for model in t.MODELS:
        controls.append(args.root / f"preflight-{model}.json")
        for folder in (args.root / model).iterdir():
            controls.extend(folder.glob("*selection.json"))
            controls.extend(folder / name for name in ("deployment.json", "identity.json", "terminal-identity.json")
                            if (folder / name).exists())
    sources.update({str(path.relative_to(args.root)): t.sha(path) for path in controls})
    report.update(passed=True, calls=dict(counts), total_calls=sum(counts.values()), blocks=len(blocks),
                  preserved_files=len(preservation), artifact_sha256=sources,
                  audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  restoration=restoration, audit_source_sha256=t.sha(Path(__file__)))
    t.save(args.output, report)
    print(json.dumps({"passed": True, "calls": report["calls"], "total_calls": report["total_calls"],
                      "blocks": len(blocks), "preserved_files": len(preservation)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
