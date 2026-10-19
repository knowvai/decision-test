"""Capacity-only extension of public_decision_benchmark; no product imports."""
from __future__ import annotations

import asyncio
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import public_decision_benchmark as benchmark


LEVELS = (1, 5, 10, 20)
ORDERS = ((1, 5, 10, 20), (20, 10, 5, 1), (5, 1, 20, 10))
QUESTIONS = {"relevance": {
    "type": "choice",
    "instructions": "判断片段内容与用户问题是否具有检索相关性。只判断相关性，不要求该片段单独完整回答问题。",
    "criteria": {"relevant": "相关", "irrelevant": "不相关"},
}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def workload(pools, max_chars):
    jobs, excluded = [], []
    for pool in pools:
        for candidate in pool["candidates"]:
            state = {"query": pool["query"], "document": candidate["content"]}
            row = {"case_id": pool["case_id"], "fragment_id": candidate["fragment_id"],
                   "state": state, "input_sha256": benchmark.digest(state),
                   "chars": len(state["query"]) + len(state["document"])}
            (jobs if row["chars"] <= max_chars else excluded).append(row)
    jobs.sort(key=lambda row: benchmark.selection_hash(row["case_id"] + ":" + row["fragment_id"]))
    assert len({(r["case_id"], r["fragment_id"]) for r in jobs}) == len(jobs)
    return jobs, [{k: v for k, v in row.items() if k != "state"} for row in excluded]


def prepare(args):
    import pyarrow.parquet as parquet

    os.umask(0o077)
    parent = json.loads((args.dataset / "dataset-protocol.private.json").read_text())
    for name, field in (("inputs.private.jsonl", "inputs_sha256"),
                        ("labels.private.jsonl", "labels_sha256")):
        assert sha(args.dataset / name) == parent[field]
    assert sha(args.source) == benchmark.DATA_SOURCES["t2"]["sha256"]
    source = parquet.read_table(args.source).to_pylist()
    excluded_ids = set(parent["excluded_historical_pool_ids"])
    assert len(excluded_ids) == 32
    formal = [r for r in benchmark.read_jsonl(args.dataset / "inputs.private.jsonl")
              if r["task"] == "reranking"]
    recreated, _, _ = benchmark.prepare_t2(source, 200, excluded_ids)
    assert benchmark.digest(formal) == benchmark.digest(recreated)
    pools, labels, _ = benchmark.prepare_t2(source, 32, ["t2:" + str(i) for i in range(len(source))
                                                       if "t2:" + str(i) not in excluded_ids])
    assert {r["case_id"] for r in pools} == excluded_ids
    assert not {r["query"] for r in pools} & {r["query"] for r in formal}
    assert not {benchmark.digest(r["candidates"]) for r in pools} & {
        benchmark.digest(r["candidates"]) for r in formal}
    pools.sort(key=lambda row: benchmark.selection_hash("capacity-engineering:" + row["case_id"]))
    engineering, validation = pools[:12], pools[12:]
    jobs, excluded = workload(engineering, args.max_chars)
    assert len(jobs) >= 100, "Insufficient engineering requests for sustained load"
    args.output.mkdir(parents=True, exist_ok=False)
    save_rows(args.output / "inputs.private.jsonl", jobs)
    save_rows(args.output / "engineering-pools.private.jsonl", engineering)
    save_rows(args.output / "validation-pools.private.jsonl", validation)
    save_rows(args.output / "validation-labels.private.jsonl", [r for r in labels if
              r["case_id"] in {p["case_id"] for p in validation}])
    save(args.output / "excluded.json", excluded)
    protocol = {"version": "passage-capacity-engineering-v1", "source": benchmark.DATA_SOURCES["t2"],
                "parent_protocol_sha256": sha(args.dataset / "dataset-protocol.private.json"),
                "formal_inputs_sha256": parent["inputs_sha256"], "formal_pools": len(formal),
                "engineering_ids": [r["case_id"] for r in engineering],
                "validation_ids": [r["case_id"] for r in validation],
                "isolation_checked": ["case_id", "query", "complete_candidate_pool"],
                "max_chars_selection_only": args.max_chars, "requests": len(jobs),
                "excluded_pairs": len(excluded), "characters": {
                    "min": min(r["chars"] for r in jobs), "max": max(r["chars"] for r in jobs),
                    "mean": statistics.mean(r["chars"] for r in jobs),
                    "p95": benchmark.percentile([r["chars"] for r in jobs], .95)},
                "questions": QUESTIONS, "labels_sent": False, "truncation": False,
                "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    save(args.output / "protocol.json", protocol)
    print(json.dumps({"prepared": len(jobs), "engineering_pools": 12,
                      "validation_pools": 20, "excluded_pairs": len(excluded)}))


def llm_body(model, state, profile=None, questions=None):
    questions = QUESTIONS if questions is None else questions
    question = next(iter(questions.values()))
    schema = {"type": "object", "properties": {"choice": {
        "type": "string", "enum": list(question["criteria"])}},
        "required": ["choice"], "additionalProperties": False}
    body = {"model": model, "temperature": 0, "max_tokens": 4096 if profile else 64,
            "messages": [{"role": "system", "content": question["instructions"] +
                          "只返回符合给定 JSON schema 的判断。"},
                         {"role": "user", "content": json.dumps(state, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "passage_relevance", "strict": True, "schema": schema}}}
    if profile:
        body["chat_template_kwargs"] = {k: profile[k] for k in ("enable_thinking", "reasoning_effort")}
    elif model == "Qwen3.8-27B":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def llm_result(raw, model, profile=None):
    assert raw["model"] == model and len(raw["choices"]) == 1
    choice = raw["choices"][0]
    assert choice["finish_reason"] == "stop", "Incomplete generation"
    message = choice["message"]
    parsed = json.loads(message["content"])
    assert set(parsed) == {"choice"} and parsed["choice"] in QUESTIONS["relevance"]["criteria"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if profile:
        assert reasoning, "Missing thinking output"
    else:
        assert not reasoning, "Unexpected thinking output"
    return {"prediction": parsed["choice"], "usage": raw.get("usage"),
            "finish_reason": choice["finish_reason"], "reasoning_chars": len(reasoning),
            "reasoning_sha256": benchmark.digest(reasoning)}


async def measure(jobs, infer, concurrency, timeout):
    """Closed-loop workers; no barrier between requests or pools, no retry."""
    iterator = iter(enumerate(jobs))
    records = [None] * len(jobs)
    active, peak = 0, 0
    began = time.perf_counter()

    async def worker():
        nonlocal active, peak
        for index, job in iterator:
            start = time.perf_counter()
            active += 1
            peak = max(peak, active)
            row = {k: job[k] for k in ("case_id", "fragment_id", "input_sha256", "chars")}
            row.update(sequence=index, started_seconds=start - began, inflight_at_start=active)
            try:
                result = await asyncio.wait_for(infer(job), timeout)
                row.update(result, status="valid")
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__)
                row.update(getattr(error, "capacity_metadata", {}))
            finally:
                active -= 1
                row["elapsed_ms"] = (time.perf_counter() - start) * 1000
                row["ended_seconds"] = time.perf_counter() - began
                records[index] = row

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    seconds = time.perf_counter() - began
    assert all(records)
    return records, {"wall_seconds": seconds, "peak_inflight": peak,
                     "requests": len(records), "concurrency": concurrency}


def block_summary(records, block):
    valid = [r for r in records if r["status"] == "valid"]
    return {**block, "valid": len(valid), "failed": len(records) - len(valid),
            "successful_latency_ms": benchmark.timing([r["elapsed_ms"] for r in valid]),
            "all_attempt_latency_ms": benchmark.timing([r["elapsed_ms"] for r in records]),
            "effective_requests_per_second": len(valid) / block["wall_seconds"],
            "failure_types": dict(Counter(r["error_type"] for r in records if r["status"] != "valid"))}


def summarize(blocks):
    reports = {}
    for n in LEVELS:
        selected = [b for b in blocks if b["concurrency"] == n]
        rows = [r for b in selected for r in b["records"]]
        reports[str(n)] = block_summary(rows, {"wall_seconds": sum(b["wall_seconds"] for b in selected),
                    "requests": len(rows), "concurrency": n,
                    "rounds": [block_summary(b["records"], {k:v for k,v in b.items() if k != "records"})
                               for b in selected]})
        reference = {(r["case_id"], r["fragment_id"], b["repeat"]): r for b in blocks
                     if b["concurrency"] == 1 for r in b["records"]}
        pairs = [(r, reference[(r["case_id"], r["fragment_id"], b["repeat"])])
                 for b in selected for r in b["records"]]
        successes = [(a,b) for a,b in pairs if a["status"] == b["status"] == "valid"]
        reports[str(n)]["paired_valid_with_n1"] = len(successes)
        reports[str(n)]["decision_changes_vs_n1"] = sum(a["prediction"] != b["prediction"] for a,b in successes)
        reports[str(n)]["max_probability_delta_vs_n1"] = max((
            max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
            for a,b in successes if "probabilities" in a and "probabilities" in b), default=None)
    first = {(r["case_id"], r["fragment_id"]): r for b in blocks
             if b["concurrency"] == 1 and b["repeat"] == 1 for r in b["records"]}
    changes, paired = 0, 0
    for b in blocks:
        if b["concurrency"] == 1 and b["repeat"] != 1:
            for r in b["records"]:
                before = first[(r["case_id"], r["fragment_id"])]
                if before["status"] == r["status"] == "valid":
                    paired += 1
                    changes += before["prediction"] != r["prediction"]
    for report in reports.values():
        report["throughput_ratio_to_n1"] = (report["effective_requests_per_second"] /
            reports["1"]["effective_requests_per_second"] if reports["1"]["valid"] else None)
        for metric in ("mean_ms", "p95_ms"):
            base = reports["1"]["successful_latency_ms"].get(metric)
            value = report["successful_latency_ms"].get(metric)
            report[metric + "_ratio_to_n1"] = value / base if base and value is not None else None
    return {"levels": reports, "n1_repeat_consistency": {"paired_valid": paired, "changes": changes}}


async def execute_model(args, model, key, jobs, folder, status_callback):
    import httpx
    from deployment_probe import native_choice_request, native_choice_result
    from launch_reviewed import MODELS, CLASSIFICATION_PROFILES

    family, _, served = MODELS[model]
    profile = CLASSIFICATION_PROFILES.get(model)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
    transport = httpx.AsyncHTTPTransport(retries=0, limits=limits)
    async with httpx.AsyncClient(transport=transport, timeout=args.timeout, trust_env=False,
                                follow_redirects=False) as http:
        headers = {"Authorization": "Bearer " + key}
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                response = await http.get(args.base_url + "/health", headers=headers, timeout=5)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            state = subprocess.check_output(["docker", "inspect", "decision-acceptance-model",
                "--format", "{{.State.Status}}"], text=True).strip()
            if state != "running" or time.monotonic() >= deadline:
                raise RuntimeError("Native runtime did not become ready")
            await asyncio.sleep(3)
        auth = []
        for credential in (None, "incorrect-evaluation-key", key.swapcase()):
            if credential == key:
                continue
            response = await http.get(args.base_url + "/health", timeout=5,
                headers={} if credential is None else {"Authorization": "Bearer " + credential})
            auth.append(response.status_code)
        save(folder / "auth.json", {"ready": 200, "unauthorized": auth})
        assert all(code == 401 for code in auth)

        async def infer(job):
            if family == "llm":
                path, body = "/v1/chat/completions", llm_body(served, job["state"], profile)
            else:
                payload = {"model": served, "state": job["state"], "questions": QUESTIONS}
                path, body = native_choice_request(family, payload)
            metadata = {"request_sha256": benchmark.digest(body), "path": path, "calls": 1}
            try:
                response = await http.post(args.base_url + path, json=body, headers=headers)
                metadata.update(http_status=response.status_code,
                                response_sha256=hashlib.sha256(response.content).hexdigest())
                response.raise_for_status()
                raw = response.json()
                if family == "llm":
                    metadata.update(usage=raw.get("usage"), finish_reason=raw.get("choices", [{}])[0].get("finish_reason"))
                    return {**metadata, **llm_result(raw, served, profile)}
                prediction, probabilities = native_choice_result(family, payload, raw)
                return {**metadata, "prediction": prediction, "probabilities": probabilities,
                        "execution": raw.get("execution"), "usage": raw.get("usage")}
            except Exception as error:
                error.capacity_metadata = metadata
                raise

        ordered = sorted(jobs, key=lambda r:r["chars"])
        warmup = [ordered[i * (len(ordered)-1) // 19] for i in range(20)]
        for n in LEVELS:
            for repeat in (1, 2):
                status_callback("warmup", concurrency=n, repeat=repeat)
                records, block = await measure(warmup, infer, n, args.timeout)
                save_rows(folder / f"warmup-n{n}-r{repeat}.jsonl", records)
                save(folder / f"warmup-n{n}-r{repeat}.json", block_summary(records, block))
                assert all(r["status"] == "valid" for r in records), "Warmup failed; measured workload not started"
        blocks = []
        for repeat, order in enumerate(ORDERS, 1):
            for n in order:
                if (args.output / "STOP").exists():
                    raise RuntimeError("Capacity stop marker present")
                status_callback("measuring", concurrency=n, repeat=repeat)
                records, block = await measure(jobs, infer, n, args.timeout)
                block["repeat"] = repeat
                save_rows(folder / f"responses-n{n}-r{repeat}.jsonl", records)
                save(folder / f"block-n{n}-r{repeat}.json", block_summary(records, block))
                blocks.append({**block, "records": records})
                if any("Timeout" in r.get("error_type", "") for r in records):
                    raise RuntimeError("Timeout recorded; stop before another load level")
        save(folder / "summary.json", {"model": model, **summarize(blocks)})


def check_artifacts(manifest):
    for filename, expected in manifest.items():
        assert sha(filename) == expected, "Preserved artifact changed: " + filename


def capacity_predecessor(directory, output, models, protocol):
    """Resume only a failed, unmeasured suffix; preserve every predecessor file."""
    directory = directory.resolve(strict=True)
    assert directory.parent == output.resolve().parent and directory != output.resolve()
    old = json.loads((directory / "protocol.json").read_text())
    state = json.loads((directory / "status.json").read_text())
    assert state["stage"] == "failed", "Predecessor must have stopped"
    assert state["models"] == old["models"]
    done = state["completed"]
    assert len(done) == len(set(done)) and done == old["models"][:len(done)]
    assert models and models == old["models"][len(done):], "Resume only the exact uncompleted suffix"
    assert state["model"] == models[0], "Only the active unmeasured configuration may be resumed"
    for field in ("version", "dataset_protocol_sha256", "concurrency_orders", "questions",
                  "requests_per_block", "timeout", "startup_timeout", "max_sequences",
                  "max_batched_tokens", "retries", "fallback", "truncation", "llm", "timer", "load",
                  "formal_classification", "formal_passage_scoring"):
        assert benchmark.digest(old[field]) == benchmark.digest(protocol[field]), "Changed capacity contract: " + field
    check_artifacts(old["source_sha256"])
    check_artifacts(old["preserved_artifact_sha256"])
    old_module = next(Path(p) for p in old["source_sha256"] if Path(p).name == Path(__file__).name)

    def measurement_ast(path):
        names = {"workload", "llm_body", "llm_result", "measure", "block_summary", "summarize", "execute_model"}
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}

    assert len(measurement_ast(old_module)) == 7
    assert measurement_ast(old_module) == measurement_ast(Path(__file__)), "Measurement functions changed"
    for name in ("public_decision_benchmark.py", "deployment_probe.py"):
        old_hash = next(h for p,h in old["source_sha256"].items() if Path(p).name == name)
        new_hash = next(h for p,h in protocol["source_sha256"].items() if Path(p).name == name)
        assert new_hash == old_hash, "Native request mapping or benchmark changed"
    expected = {f"responses-n{n}-r{repeat}.jsonl" for repeat, order in enumerate(ORDERS, 1) for n in order}
    for model in done:
        folder = directory / model
        assert {p.name for p in folder.glob("responses-*")} == expected, "Incomplete predecessor output"
        summary = json.loads((folder / "summary.json").read_text())
        assert (folder / "deployment.json").is_file()
        for name in expected:
            assert len(benchmark.read_jsonl(folder / name)) == old["requests_per_block"]
        assert all(summary["levels"][str(n)]["requests"] == old["requests_per_block"] * len(ORDERS)
                   for n in LEVELS)
    for model in models:
        folder = directory / model
        assert not list(folder.glob("responses-*")), "Measured requests must not be replayed"
        assert not list(folder.glob("block-*")) and not (folder / "summary.json").exists()
    assert (directory / models[0] / "deployment.json").is_file(), "Missing failed deployment evidence"
    preserved = {**old["preserved_artifact_sha256"], **old["source_sha256"],
                 **{str(p): sha(p) for p in directory.rglob("*") if p.is_file()}}
    return {"directory": str(directory), "protocol_sha256": sha(directory / "protocol.json"),
            "completed": done, "unstarted_recovery": {"model": models[0], "measured_requests": 0},
            "measurement_functions_ast_unchanged": True}, preserved


def run(args):
    import fcntl

    os.umask(0o077)
    sys.path.insert(0, str(args.service_source.resolve(strict=True)))
    from launch_reviewed import MODELS, launch, read_key

    args.base_url = benchmark.local_endpoint(args.base_url)
    assert urlsplit(args.base_url).port == args.port
    assert len(args.models) == len(set(args.models))
    assert all(m in MODELS and MODELS[m][0] != "reranker" and m != "intern-4b" for m in args.models)
    assert args.models[-1] in {"qwen3.8-27b", "qwen3.8-27b-thinking"}
    data_protocol = json.loads((args.dataset / "protocol.json").read_text())
    for name, expected in data_protocol["files"].items():
        assert sha(args.dataset / name) == expected
    jobs = benchmark.read_jsonl(args.dataset / "inputs.private.jsonl")
    assert len(jobs) == data_protocol["requests"]
    assert all(r["input_sha256"] == benchmark.digest(r["state"]) for r in jobs)
    artifacts = {}
    for path in args.prior_audits:
        for name, expected in json.loads(path.read_text())["artifact_sha256"].items():
            assert name not in artifacts or artifacts[name] == expected
            artifacts[name] = expected
        artifacts[str(path)] = sha(path)
    check_artifacts(artifacts)
    source_paths = [Path(__file__), Path(benchmark.__file__), *args.service_source.glob("*.py"),
                    args.service_source / "nginx.native.conf.template"]
    sources = {str(path.resolve()): sha(path) for path in source_paths}
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"version": "pointwise-http-capacity-v1",
            "dataset_protocol_sha256": sha(args.dataset / "protocol.json"), "models": args.models,
            "concurrency_orders": ORDERS, "questions": QUESTIONS, "requests_per_block": len(jobs),
            "timeout": args.timeout, "startup_timeout": args.startup_timeout,
            "max_sequences": 32, "max_batched_tokens": 32768, "source_sha256": sources,
            "preserved_artifact_sha256": artifacts, "retries": 0, "fallback": False,
            "truncation": False, "formal_classification": False, "formal_passage_scoring": False,
            "llm": {"method": "strict_json_schema", "temperature": 0,
                    "nonthinking_max_tokens": 64, "thinking_max_tokens": 4096},
            "timer": "construct/send through output validation; no client slot wait or file IO",
            "load": "closed-loop stream, one complete query-document per HTTP request",
            "limitations": ["length-selected engineering subset", "not global-optimum deployment proof",
                            "not open-loop SLA or maximum capacity", "no formal quality ranking"]}
        if getattr(args, "after_run", None):
            link, preserved = capacity_predecessor(args.after_run, args.output, args.models, protocol)
            assert all(name not in artifacts or artifacts[name] == value for name,value in preserved.items())
            artifacts.update(preserved)
            protocol["predecessor"] = link
        if getattr(args, "laya_max_concurrent", None) is not None:
            assert "laya-multilingual" in args.models and 1 <= args.laya_max_concurrent <= 256
            protocol["deployment_overrides"] = {"laya-multilingual": {
                "http_admission_limit": args.laya_max_concurrent, "gpu_execution_unchanged": True}}
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "protocol.json", protocol)
        state = {"pid": os.getpid(), "models": args.models, "completed": []}

        def status(stage, **details):
            state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
            benchmark.write_json(args.output / "status.json", state)
            print(json.dumps({k:v for k,v in state.items() if k != "models"}), flush=True)

        status("starting")
        try:
            key = read_key(args.env_file)
            for model in args.models:
                if (args.output / "STOP").exists():
                    raise RuntimeError("Capacity stop marker present")
                check_artifacts(sources)
                check_artifacts(artifacts)
                folder = args.output / model
                folder.mkdir()
                status("loading", model=model, concurrency=None, repeat=None)
                launch(SimpleNamespace(root=args.root, model=model, env_file=args.env_file,
                    native_port=args.port-1, public_port=args.port, seqs=32, tokens=32768, enforce_eager=False,
                    laya_max_concurrent=(getattr(args, "laya_max_concurrent", None)
                                         if model == "laya-multilingual" else None)))
                identity = json.loads((args.root / "acceptance-20261015/evidence" /
                                       (model + ".deployment.json")).read_text())
                save(folder / "deployment.json", identity)
                asyncio.run(execute_model(args, model, key, jobs, folder, status))
                check_artifacts(artifacts)
                state["completed"].append(model)
                status("model_complete")
            check_artifacts(sources)
            check_artifacts(artifacts)
            save(args.output / "preservation-final.json", {"unchanged": True, "artifacts": len(artifacts)})
            status("complete", model=None, concurrency=None, repeat=None)
        except BaseException as error:
            status("failed", error_type=type(error).__name__)
            raise


def register(subparsers):
    parser = subparsers.add_parser("prepare-capacity", help="Prepare isolated engineering data; no inference")
    for name in ("source", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.set_defaults(function=prepare)
    parser = subparsers.add_parser("local-capacity", help="Independent HTTP capacity preflight, not formal scoring")
    for name in ("root", "service-source", "dataset", "env-file", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-audits", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--after-run", type=Path, help="Preserve a failed predecessor and resume only its unmeasured suffix")
    parser.add_argument("--laya-max-concurrent", type=int, help="Explicit Laya-only HTTP admission override")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--port", type=int, default=30889)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=2400)
    parser.set_defaults(function=run)
