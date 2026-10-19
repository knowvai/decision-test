"""Frozen public-data evaluation; never imported by the serving application."""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import random
import shlex
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


SEED = "public-cn-v1"
DOMAINS = {
    "social": "社交平台：发帖或查询社交消息",
    "transport": "交通出行：车票、打车、交通状况或交通信息",
    "calendar": "日历日程：添加、删除或查询日程",
    "play": "播放或启动：播放音乐、广播、播客、有声书或启动游戏",
    "news": "新闻：查询新闻资讯",
    "datetime": "日期时间：查询时间日期或换算时区",
    "recommendation": "推荐：推荐活动、地点或电影",
    "email": "电子邮件：发送或查询邮件、添加或查询联系人",
    "iot": "智能家居：控制灯光、插座、清洁设备或咖啡机",
    "general": "通用互动：问候、闲聊或讲笑话",
    "audio": "音量控制：增减音量、设置音量或静音",
    "lists": "清单：创建、添加、删除或查询列表项目",
    "qa": "知识问答：事实、定义、数学、汇率或股票信息",
    "cooking": "烹饪：查询食谱或烹饪信息",
    "takeaway": "外卖：查询或订购外卖",
    "music": "音乐信息与偏好：查询、设置、喜欢或不喜欢音乐，不含播放",
    "alarm": "闹钟：设置、删除或查询闹钟",
    "weather": "天气：查询天气情况",
}
DOMAIN_QUESTIONS = {"domain": {
    "type": "choice",
    "instructions": "判断用户这句话所属的任务领域，选择最符合实际请求的一类。只分类，不执行请求。",
    "criteria": DOMAINS,
}}
DATA_SOURCES = {
    "massive": {
        "repository": "AmazonScience/massive", "version": "v1.1", "config": "zh-CN", "split": "test",
        "revision": "ed58ac423a2f4121720918bf5301577edce4ffd3",
        "loader_revision": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "filename": "massive-zh-CN-test.parquet",
        "sha256": "7eca92a47e6bffeec3fc7de751948b82904680353a42eef7e8c6fdbe89adb2db",
        "license": "CC-BY-4.0",
    },
    "t2": {
        "repository": "C-MTEB/T2Reranking", "split": "dev",
        "revision": "76631901a18387f85eaa53e5450019b87ad58ef9",
        "filename": "t2reranking-dev.parquet",
        "sha256": "ceaf1ba1a4777dbb7aecad86fb8aa9cae7a678af6d711d5529c238e89bb6adff",
        "labels": "supplied positive/negative lists, not original graded qrels",
        "license": "derivative card unspecified; original THUIR/T2Ranking Apache-2.0; no redistribution",
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def selection_hash(value):
    return hashlib.sha256((bytes.fromhex("6b6e6f77762d7075626c69632d636e2d7631").decode("utf-8") + ":" + value).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_massive(rows, per_domain):
    domains = list(DOMAINS)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate MASSIVE identities"
    inputs, labels = [], []
    for index, domain in enumerate(domains):
        group = [r for r in rows if r["scenario"] == index]
        assert group and (per_domain is None or len(group) >= per_domain > 0), "Insufficient domain examples"
        selected = sorted(group, key=lambda r: selection_hash(domain + ":" + r["id"]))[:per_domain]
        for row in selected:
            assert row["locale"] == "zh-CN" and row["partition"] == "test" and row["utt"].strip()
            case_id = "massive:" + row["id"]
            inputs.append({"case_id": case_id, "task": "classification", "state": {"utterance": row["utt"]}})
            labels.append({"case_id": case_id, "task": "classification", "gold": domain,
                           "source_id": row["id"], "source_intent": row["intent"]})
    inputs.sort(key=lambda r: selection_hash(r["case_id"]))
    return inputs, labels


def prepare_t2(rows, count, excluded_ids=()):
    assert len({r["query"] for r in rows}) == len(rows), "Duplicate T2 queries"
    eligible = []
    for index, row in enumerate(rows):
        texts = row["positive"] + row["negative"]
        if ("t2:" + str(index) not in excluded_ids and row["positive"] and row["negative"]
                and all(isinstance(s, str) and s.strip() for s in [row["query"]] + texts)
                and len(set(texts)) == len(texts)):
            eligible.append((index, row))
    assert 0 < count <= len(eligible)
    selected = sorted(eligible, key=lambda pair: selection_hash(pair[1]["query"]))[:count]
    inputs, labels = [], []
    for index, row in selected:
        case_id = "t2:" + str(index)
        candidates = [{"fragment_id": hashlib.sha256(s.encode()).hexdigest(), "content": s}
                      for s in row["positive"] + row["negative"]]
        candidates.sort(key=lambda c: selection_hash(row["query"] + ":" + c["content"]))
        inputs.append({"case_id": case_id, "task": "reranking", "query": row["query"], "candidates": candidates})
        labels.append({"case_id": case_id, "task": "reranking", "source_row": index,
                       "positive_ids": [hashlib.sha256(s.encode()).hexdigest() for s in row["positive"]]})
    return inputs, labels, len(eligible)


def reused_sources():
    root = Path(__file__).parent
    source = (root / "decision_rag_worker.py").read_text()
    tree = ast.parse(source)
    names = {"digest", "validate_decision", "RecordingDecisionClient", "PointwiseDecisionReranker", "ListwiseDecisionReranker"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    definitions = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    replay = ast.parse((root / "replay_decision_rag.py").read_text())
    score_assignment = next(n for n in ast.walk(replay) if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "score_questions" for t in n.targets))
    return definitions, ast.literal_eval(score_assignment.value)


def baseline_sources():
    source = Path(__file__).with_name("decision_rag_worker.py").read_text()
    names = {"telemetry", "bundle_info"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    return ast.unparse(ast.Module(body=nodes, type_ignores=[]))


def prepare(args):
    import pyarrow.parquet as parquet

    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    tables = {}
    for key, source in DATA_SOURCES.items():
        path = args.downloads / source["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], "Public source hash mismatch"
        tables[key] = parquet.read_table(path)
    features = json.loads(tables["massive"].schema.metadata[b"huggingface"])["info"]["features"]
    assert features["scenario"]["names"] == list(DOMAINS), "Official label order mismatch"
    v3 = getattr(args, "version", "v1") == "v3"
    excluded = []
    if v3:
        assert args.exclude_dataset is not None, "v3 requires the historical smoke dataset"
        old_protocol = json.loads((args.exclude_dataset / "dataset-protocol.private.json").read_text())
        old_inputs = args.exclude_dataset / "inputs.private.jsonl"
        assert hashlib.sha256(old_inputs.read_bytes()).hexdigest() == old_protocol["inputs_sha256"]
        excluded = [r["case_id"] for r in read_jsonl(old_inputs) if r["task"] == "reranking"]
        assert len(excluded) == len(set(excluded)) == 32
    route, route_labels = prepare_massive(tables["massive"].to_pylist(), None if v3 else args.per_domain)
    rank, rank_labels, eligible = prepare_t2(tables["t2"].to_pylist(), 200 if v3 else args.pools, excluded)
    if v3:
        assert len(route) == 2974, "Expected the complete official Chinese test split"
    inputs, labels = route + rank, route_labels + rank_labels
    for filename, records in [("inputs.private.jsonl", inputs), ("labels.private.jsonl", labels)]:
        (args.output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    definitions, score_questions = reused_sources()
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "sources": DATA_SOURCES,
        "seed": SEED, "selection": "lowest SHA-256, before inference; class-balanced domains and uniform eligible pools",
        "classification_cases": len(route), "per_domain": None if v3 else args.per_domain,
        "version": "v3" if v3 else "v1", "excluded_historical_pool_ids": excluded,
        "rerank_pools": len(rank), "rerank_pairs": sum(len(r["candidates"]) for r in rank),
        "t2_total_pools": len(tables["t2"]), "t2_eligible_pools": eligible,
        "t2_exclusions": "missing class, empty text, duplicate passage string; no length or model-score exclusions",
        "domain_questions": DOMAIN_QUESTIONS, "score_questions": score_questions,
        "score_mapping": "expected ordinal grade / 3; evaluation labels are binary",
        "repeats": args.repeats, "timeout_seconds": 15, "pacing_seconds": 0 if v3 else 0.21, "retries": 0,
        "pool_timeout_seconds": 300,
        "confidence_gate": False, "fallback": False, "truncation": False,
        "warmup": "first classification plus one first-pool passage; excluded from scoring/timing",
        "public_training_exposure": "unknown", "publication": "aggregates only; no source redistribution",
        "inputs_sha256": hashlib.sha256((args.output / "inputs.private.jsonl").read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256((args.output / "labels.private.jsonl").read_bytes()).hexdigest(),
        "reused_definitions_sha256": hashlib.sha256(definitions.encode()).hexdigest(),
        "scorer_sha256": hashlib.sha256((Path(__file__).parents[1] / "src/evaluation/retrieval_eval.py").read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_definitions_sha256": hashlib.sha256(baseline_sources().encode()).hexdigest(),
        "intervals": {"samples": 2000, "confidence": 0.95,
                      "classification": "gold-label-set-stratified paired identical-input clusters",
                      "reranking": "paired query clusters; all repeats retained together; failures score zero"},
        "llm_baseline": {"factory": "deployed ServiceUtils.create_understanding_llm_service",
                         "temperature": 0, "max_tokens": 64, "retries": 0, "timeout_seconds": 15,
                         "structured_output_method": "deployed ConfigUtils.get_request_config",
                         "output": "strictly validated single domain ID; no generated confidence or fake server timer"},
    }
    if v3:
        input_groups = {}
        gold = {r["case_id"]: r["gold"] for r in route_labels}
        for record in route:
            input_groups.setdefault(digest(record["state"]), []).append(record["case_id"])
        protocol.update(classification_distinct_inputs=len(input_groups),
                        classification_duplicate_input_groups=sum(len(v) > 1 for v in input_groups.values()),
                        classification_mixed_label_input_groups=sum(len({gold[k] for k in v}) > 1 for v in input_groups.values()))
        preflight = [{"case_id": "synthetic:route", "task": "classification", "state": {"utterance": "hello"}},
                     {"case_id": "synthetic:rank", "task": "reranking", "query": "Which city is the capital of China?",
                      "candidates": [{"fragment_id": "synthetic:" + str(i), "content": text} for i, text in enumerate([
                          "Beijing is the capital of China.", "Paris is the capital of France.",
                          "A Python list can contain several values.", "This document is about a warranty period.",
                          "Beijing is the political center and capital of China.", "Planets orbit the sun.",
                          "An apple is a fruit.", "Neutral context. " * 80, "London is the capital of the UK."])]}]
        preflight.append({"case_id": "synthetic:long-rank", "task": "reranking", "query": preflight[1]["query"],
                          "candidates": [{"fragment_id": "synthetic:long:" + str(i),
                                          "content": "Neutral document: " + "This is a neutral statement. " * 1000}
                                         for i in range(8)]})
        (args.output / "preflight.inputs.private.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preflight))
        protocol.update(selection="all official MASSIVE test cases; lowest SHA-256 eligible T2 pools excluding old 32",
                        warmup="separate synthetic inputs, no formal cases or labels",
                        preflight_sha256=hashlib.sha256((args.output / "preflight.inputs.private.jsonl").read_bytes()).hexdigest(),
                        rerank_transport="one full pool per HTTP request; independent native scoring; no pacing")
    write_json(args.output / "dataset-protocol.private.json", protocol)
    print(json.dumps({k: protocol[k] for k in ["classification_cases", "rerank_pools", "rerank_pairs", "t2_eligible_pools"]}))


def domain_messages(state, questions):
    question = questions["domain"]
    return [
        {"role": "system", "content": question["instructions"] + "\n" +
         json.dumps({"criteria": question["criteria"], "output": {"domain": "one criterion ID"}}, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
    ]


async def llm_classify(client, state, questions, active, timeout):
    messages = domain_messages(state, questions)
    call = {"input_sha256": digest({"messages": messages}), "kind": "structured_llm_classification"}
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(client.ainvoke(messages), timeout=timeout)
        parsed = response.get("parsed")
        assert not response.get("parsing_error") and isinstance(parsed, dict)
        assert set(parsed) == {"domain"} and parsed["domain"] in questions["domain"]["criteria"]
        raw = response["raw"]
        usage = getattr(raw, "usage_metadata", None) or {}
        metadata = getattr(raw, "response_metadata", None) or {}
        call.update(status="valid", result={"prediction": parsed["domain"], "usage": usage,
                    "reported_model": metadata.get("model_name", metadata.get("model")),
                    "latency_ms": None})
        return parsed["domain"]
    except Exception as exc:
        call.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        call["client_elapsed_ms"] = (time.perf_counter() - started) * 1000
        active.setdefault("decision_calls", []).append(call)


async def prepare_llm_baseline(data):
    from src.api.main import SendMessageRequest, _resolve_platform_conversation_model, _build_conversation_rag_request
    from src.agents.utils.service_utils import ServiceUtils
    from src.agents.utils.config_utils import ConfigUtils
    from src.services.platform_config import get_platform_configuration_client
    from src.config.settings import DecomposeRAGConfig

    config = data["llm_baseline"]
    namespace = {"urlsplit": urlsplit}
    exec(compile(data["baseline_definitions"], "reused_baseline_metadata", "exec"), namespace)
    resolver = get_platform_configuration_client(DecomposeRAGConfig())

    async def binding():
        bundle = await asyncio.to_thread(resolver.resolve_conversation_models, config["tenant_id"],
                                        "model.binding.conversation.generation", None, use_rerank=True, force_refresh=True)
        return namespace["bundle_info"](bundle)

    before = await binding()
    assert before == config["expected_models"], "Original model binding drift"
    request = SendMessageRequest.model_validate({"message": "public classification preparation", "use_cache": False,
                                                "stream": False, "temperature": 0, "max_tokens": 64})
    resolved = await _resolve_platform_conversation_model(config["tenant_id"], request)
    rag = _build_conversation_rag_request(resolved)
    client = ServiceUtils.create_understanding_llm_service(rag)
    # Use the canonical model factory, with this experiment's zero-retry deadline.
    client.max_retries = 0
    client.request_timeout = data["timeout_seconds"]
    client.temperature = 0
    client.max_tokens = 64
    retries = {name: getattr(getattr(client, name, None), "max_retries", None)
               for name in ["root_client", "root_async_client"]}
    assert all(type(value) is int and value == 0 for value in retries.values()), "Baseline SDK retries must already be disabled"
    selected = before["understanding"] or before["generation"]
    assert client.model_name == selected["name"], "Resolved baseline model differs"
    schema = {"name": "public_domain_classification", "description": data["domain_questions"]["domain"]["instructions"],
              "parameters": {"type": "object", "properties": {"domain": {"type": "string", "enum": list(DOMAINS)}},
                             "required": ["domain"], "additionalProperties": False}}
    method = ConfigUtils.get_request_config(rag, DecomposeRAGConfig())["structured_output_method"]
    data["llm_execution"] = {"structured_output_method": method, "temperature": 0, "max_tokens": 64,
                             "deadline_seconds": data["timeout_seconds"], "sdk_max_retries": retries}
    structured = client.with_structured_output(schema, method=method, include_raw=True)
    return structured, binding, before


def inference_jobs(inputs, repeats, warmup=False):
    if warmup:
        route = next(r for r in inputs if r["task"] == "classification")
        rank = next(r for r in inputs if r["task"] == "reranking")
        inputs = [route, {**rank, "candidates": rank["candidates"][:1]}]
        repeats = 1
    jobs = []
    for repeat in range(1, repeats + 1):
        for record in inputs:
            allowed = ({"case_id", "task", "state"} if record["task"] == "classification"
                       else {"case_id", "task", "query", "candidates"})
            assert set(record) == allowed, "Unexpected input metadata or label"
            if record["task"] == "reranking":
                assert all(set(c) == {"fragment_id", "content"} for c in record["candidates"])
            jobs.append({**record, "repeat": repeat})
    return jobs


async def native_run(data):
    """Only this function runs inside the test Supervisor's isolated interpreter."""
    logging.disable(logging.CRITICAL)
    from src.services.decision_model import DecisionModelClient, DecisionModelError

    llm_mode = data.get("classifier_kind") == "llm"
    allowed_ips = set()
    if llm_mode:
        for host in data["llm_baseline"]["approved_public_hosts"]:
            allowed_ips.update(info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM))
        allowed = None
    else:
        url = urlsplit(data["endpoint"])
        ip = ipaddress.ip_address(url.hostname)
        assert url.scheme == "http" and ip.is_private and not ip.is_loopback
        allowed = (str(ip), url.port)
    destinations, blocked = set(), set()
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def checked(address):
        if not isinstance(address, tuple):
            return
        approved = address[:2] == allowed
        if llm_mode:
            ips = ([str(ipaddress.ip_address(address[0]))] if address[0] in allowed_ips else
                   [info[4][0] for info in socket.getaddrinfo(address[0], address[1], type=socket.SOCK_STREAM)])
            approved = bool(ips) and all(ipaddress.ip_address(ip).is_private or ipaddress.ip_address(ip).is_loopback
                                         or ip in allowed_ips for ip in ips)
        if not approved:
            blocked.add(str(address[:2]))
            raise RuntimeError("Unapproved public-benchmark egress")
        destinations.add(address[:2])

    def connect(sock, address):
        checked(address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        checked(address)
        return original_connect_ex(sock, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in data["source_paths"]}
    namespace = {"asyncio": asyncio, "asdict": asdict, "hashlib": hashlib, "json": json,
                 "math": math, "time": time, "DecisionModelClient": DecisionModelClient,
                 "DecisionModelError": DecisionModelError, "DATA": {"decision_pacing_seconds": data["pacing_seconds"]},
                 "active": {}, "SCORE_QUESTIONS": data["score_questions"]}
    exec(compile(data["reused_definitions"], "reused_decision_eval", "exec"), namespace)
    client = None if llm_mode else namespace["RecordingDecisionClient"](api_key=data["api_key"], endpoint=data["endpoint"],
                                                  model=data["model"], timeout_seconds=data["timeout_seconds"])
    reranker = None if llm_mode else namespace["PointwiseDecisionReranker"](client)
    listwise = None if llm_mode else namespace["ListwiseDecisionReranker"](
        client, data.get("listwise_max_options", 32), data.get("listwise_max_state_chars", 6000)
    )
    baseline, binding, before, after = None, None, None, None
    cancelled = False
    pool_client = None
    if data.get("batch_transport") and data.get("rerank_transport") != "systemone_concurrent":
        import httpx

        pool_client = httpx.AsyncClient(timeout=data["pool_timeout_seconds"], trust_env=False,
                                       headers={"Authorization": "Bearer " + data["api_key"]})
    try:
        if llm_mode:
            baseline, binding, before = await prepare_llm_baseline(data)
        for job in data["jobs"]:
            if data.get("stop_file") and Path(data["stop_file"]).exists():
                cancelled = True
                break
            namespace["active"].clear()
            started = time.perf_counter()
            record = {"case_id": job["case_id"], "task": job["task"], "repeat": job["repeat"],
                      "input_sha256": digest({k: v for k, v in job.items() if k != "repeat"})}
            try:
                if job["task"] == "classification":
                    if llm_mode:
                        record["prediction"] = await llm_classify(baseline, job["state"], data["domain_questions"],
                                                                 namespace["active"], data["timeout_seconds"])
                    else:
                        result = await client.decide(state=job["state"], questions=data["domain_questions"])
                        record["prediction"] = result.answers["domain"].choice
                else:
                    if data.get("rerank_transport") == "systemone_concurrent":
                        scores = await concurrent_rerank(reranker, data, job)
                    elif data.get("rerank_transport") == "listwise_decision":
                        scores = await listwise.rerank(job["query"], [SimpleNamespace(**c) for c in job["candidates"]])
                    elif pool_client is None:
                        scores = await reranker.rerank(job["query"], [SimpleNamespace(**c) for c in job["candidates"]])
                    else:
                        scores = await rerank_pool(pool_client, data, job, namespace["active"])
                    assert len(scores) == len(job["candidates"]) and all(math.isfinite(s) for s in scores)
                    record["scores"] = [{"fragment_id": c["fragment_id"], "score": s}
                                        for c, s in zip(job["candidates"], scores)]
                record["status"] = "valid"
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
            if data.get("rerank_transport") == "systemone_concurrent":
                for call in namespace["active"].get("decision_calls", []):
                    if call.get("result") is not None:
                        call["result"]["latency_ms"] = None  # SGLang does not report this timer.
            record.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                          calls=namespace["active"].get("decision_calls", []))
            print("BENCHMARK_PUBLIC " + json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        if binding is not None:
            after = await binding()
        if pool_client is not None:
            await pool_client.aclose()
        unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in source_hashes.items())
        print("BENCHMARK_PUBLIC_MANIFEST " + json.dumps({"source_sha256": source_hashes, "source_unchanged": unchanged,
              "network_destinations": sorted(destinations), "blocked_destinations": sorted(blocked),
              "models_before": before, "models_after": after,
              "cancelled": cancelled, "llm_execution": data.get("llm_execution"),
              "bindings_unchanged": not llm_mode or (before is not None and before == after),
              "scope": "native_public_benchmark_no_router_no_fallback_no_application_writes"}), flush=True)
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex


async def concurrent_rerank(reranker, data, job):
    concurrency = (len(job["candidates"])
                   if data.get("pool_concurrency") == "full_pool"
                   else data["batch_size"])
    semaphore = asyncio.Semaphore(concurrency)

    async def score(candidate):
        async with semaphore:
            return (await reranker.rerank(job["query"], [SimpleNamespace(**candidate)]))[0]

    # Drain the entire pool even when one independent request fails. No outstanding
    # work may spill into the next pool's timer, and no partial result is accepted.
    results = await asyncio.gather(*(score(c) for c in job["candidates"]), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def rerank_body(data, job):
    body = {"model": data["model"], "query": job["query"],
            "documents": [c["content"] for c in job["candidates"]]}
    if data.get("rerank_transport", "native_batch") == "vllm":
        assert data["ranker_kind"] == "qwen"
        body.update(top_n=len(body["documents"]), max_tokens_per_query=0,
                    max_tokens_per_doc=0, truncate_prompt_tokens=None)
        return body
    body["batch_size"] = data["batch_size"]
    if data["ranker_kind"] == "decision":
        body["questions"] = data["score_questions"]
    return body


async def rerank_pool(client, data, job, active):
    body = rerank_body(data, job)
    url = urlsplit(data["endpoint"])._replace(path="/v1/rerank", query="", fragment="").geturl()
    call = {"input_sha256": digest(body), "documents": len(body["documents"]), "kind": "full_pool_http"}
    started = time.perf_counter()
    try:
        response = await client.post(url, json=body)
        call["http_status"] = response.status_code
        response.raise_for_status()
        result = response.json()
        rows = result["results"]
        assert result["model"] == data["model"]
        assert len(rows) == len(body["documents"])
        assert all(type(r["index"]) is int for r in rows)
        assert sorted(r["index"] for r in rows) == list(range(len(rows)))
        if data.get("rerank_transport", "native_batch") == "vllm":
            rows = sorted(rows, key=lambda row: row["index"])
            call["execution"] = {"engine": "vllm", "scheduling": "dynamic",
                                 "forward_calls": None, "actual_batch_sizes": None}
        else:
            assert [r["index"] for r in rows] == list(range(len(rows)))
            execution = result["execution"]
            assert execution["batch_size"] == data["batch_size"] and not execution["truncated"] and not execution["cpu_fallback"]
            assert len(execution["input_tokens_per_document"]) == len(rows)
            assert execution["forward_calls"] == math.ceil(len(rows) / data["batch_size"])
        scores = [r["relevance_score"] for r in rows]
        assert all(type(s) in (float, int) and math.isfinite(s) and 0 <= s <= 1 for s in scores)
        call.update(status="valid", result=result)
        return scores
    except Exception as exc:
        call.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        call["client_elapsed_ms"] = (time.perf_counter() - started) * 1000
        active.setdefault("decision_calls", []).append(call)


def run_candidate(args):
    import configparser

    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    protocol = json.loads((args.dataset / "dataset-protocol.private.json").read_text())
    for filename, key in [("inputs.private.jsonl", "inputs_sha256"), ("labels.private.jsonl", "labels_sha256")]:
        assert hashlib.sha256((args.dataset / filename).read_bytes()).hexdigest() == protocol[key]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol["adapter_sha256"], "Frozen adapter changed"
    definitions, score_questions = reused_sources()
    assert hashlib.sha256(definitions.encode()).hexdigest() == protocol["reused_definitions_sha256"]
    assert score_questions == protocol["score_questions"]
    settings = configparser.ConfigParser(interpolation=None)
    settings.optionxform = str
    settings.read_string("[evaluation]\n" + args.access.read_text())
    key = shlex.split(settings["evaluation"]["DECISION_EVAL_API_KEY"])[0]
    v3 = protocol.get("version") == "v3"
    selected = read_jsonl(args.dataset / "inputs.private.jsonl")
    if v3 and args.warmup:
        path = args.dataset / "preflight.inputs.private.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["preflight_sha256"]
        selected = read_jsonl(path)
    tasks = getattr(args, "tasks", ["classification", "reranking"])
    selected = [r for r in selected if r["task"] in tasks]
    if getattr(args, "limit", None):
        selected = selected[:args.limit]
    assert selected and (getattr(args, "ranker_kind", "decision") != "qwen" or tasks == ["reranking"])
    jobs = inference_jobs(selected, 1 if args.warmup else protocol["repeats"], args.warmup and not v3)
    data = {"api_key": key, "endpoint": args.endpoint, "model": args.model, "jobs": jobs,
            "reused_definitions": definitions, "domain_questions": protocol["domain_questions"],
            "score_questions": score_questions, "pacing_seconds": protocol["pacing_seconds"],
            "timeout_seconds": protocol["timeout_seconds"], "source_paths": ["src/services/decision_model.py"]}
    data["rerank_transport"] = getattr(args, "rerank_transport", "native_batch")
    data.update(batch_transport=(v3 and data["rerank_transport"] != "listwise_decision"), pool_timeout_seconds=protocol.get("pool_timeout_seconds", 300),
                batch_size=getattr(args, "batch_size", 1), ranker_kind=getattr(args, "ranker_kind", "decision"),
                pool_concurrency=("full_pool" if getattr(args, "pool_concurrency", "bounded") == "full_pool" else "bounded"))
    if data["rerank_transport"] == "listwise_decision":
        limits = {"d3-edge": 255, "d3-mini": 255, "d3-flash": 255,
                  "clef-flash": 255, "Intern-Decision-4B": 62, "laya-multilingual": 20}
        data["listwise_max_options"] = limits.get(args.model, 32)
        # Start with the whole pool; a deterministic 413/422-only splitter
        # discovers the model's actual no-truncation boundary.
        data["listwise_max_state_chars"] = 900000
    if data["rerank_transport"] == "vllm":
        assert v3 and data["ranker_kind"] == "qwen" and tasks == ["reranking"]
    elif data["rerank_transport"] == "systemone_concurrent":
        assert v3 and data["ranker_kind"] == "decision" and data["pacing_seconds"] == 0
        assert data["pool_concurrency"] in {"bounded", "full_pool"}
    stop_file = getattr(args, "stop_file", None)
    assert stop_file is None or stop_file.is_absolute(), "Stop marker must be an absolute remote path"
    data["stop_file"] = str(stop_file) if stop_file is not None else None
    data["classifier_kind"] = getattr(args, "classifier_kind", "decision")
    if data["classifier_kind"] == "llm":
        assert tasks == ["classification"] and args.llm_baseline_config is not None
        data["llm_baseline"] = json.loads(args.llm_baseline_config.read_text())
        data["source_paths"] = sorted(set(data["llm_baseline"]["source_paths"]) |
                                      {"src/agents/utils/config_utils.py", "src/config/settings.py"})
        data["baseline_definitions"] = baseline_sources()
        assert hashlib.sha256(data["baseline_definitions"].encode()).hexdigest() == protocol["baseline_definitions_sha256"]
        data["batch_transport"] = False
    else:
        assert args.endpoint is not None, "A decision endpoint is required"
    frozen = {"model": args.model, "endpoint": args.endpoint, "jobs": len(jobs), "labels_sent": False,
              "warmup": args.warmup, "dataset_protocol_sha256": digest(protocol),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    frozen.update(tasks=tasks, batch_size=data["batch_size"], ranker_kind=data["ranker_kind"],
                  rerank_transport=data["rerank_transport"],
                  pool_concurrency=data["pool_concurrency"],
                  classifier_kind=data["classifier_kind"],
                  stop_file=data["stop_file"],
                  measurement_role=getattr(args, "measurement_role", "formal"))
    write_json(args.output / "run-protocol.private.json", frozen)
    worker = ("import asyncio,types,sys,os\n"
              "os.environ.update(LANGFUSE_ENABLED='false', LANGFUSE_TRACING_ENABLED='false', LANGCHAIN_TRACING_V2='false')\n"
              "module=types.ModuleType('public_decision_benchmark');sys.modules[module.__name__]=module\n"
              + "exec(compile(" + repr(Path(__file__).read_text()) + ", 'public_decision_benchmark', 'exec'), module.__dict__)\n"
              + "asyncio.run(module.native_run(" + repr(data) + "))\n")
    count, manifest = 0, None
    with (args.output / "ssh-stderr.private.txt").open("w") as errors, subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", args.host,
                           "docker exec -i " + shlex.quote(args.container) + " python -"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True) as proc:
        try:
            proc.stdin.write(worker)
            proc.stdin.close()
        except BrokenPipeError:
            proc.stdin.close()
            diagnostic = proc.stdout.read()
            proc.wait()
            raise RuntimeError("remote benchmark worker exited before execution: " + diagnostic[-4000:]) from None
        with (args.output / "responses.private.jsonl").open("w") as stream:
            for line in proc.stdout:
                if line.startswith("BENCHMARK_PUBLIC "):
                    record = json.loads(line.removeprefix("BENCHMARK_PUBLIC "))
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
                    if count % 30 == 0 or count == len(jobs) or record["status"] == "failed":
                        print(json.dumps({"model": args.model, "completed": count, "total": len(jobs),
                                          "task": record["task"], "status": record["status"],
                                          "elapsed_ms": round(record["elapsed_ms"], 1)}), flush=True)
                elif line.startswith("BENCHMARK_PUBLIC_MANIFEST "):
                    manifest = json.loads(line.removeprefix("BENCHMARK_PUBLIC_MANIFEST "))
        code = proc.wait()
    if manifest is not None:
        manifest.update(returncode=code, completed=count, total=len(jobs), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "manifest.private.json", manifest)
    assert code == 0 and manifest is not None and count == len(jobs), "Incomplete run; retain partial output"
    assert manifest["source_unchanged"] and not manifest["blocked_destinations"], "Preservation check failed"
    assert manifest.get("bindings_unchanged", True), "Original model binding drift"


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timing(values):
    return {"n": len(values), "mean_ms": statistics.mean(values) if values else None,
            "p50_ms": percentile(values, 0.5), "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99)}


def local_endpoint(base_url):
    url = urlsplit(base_url)
    address = ipaddress.ip_address(url.hostname)
    assert (url.scheme == "http" and (address.is_private or address.is_loopback)
            and not url.username and not url.password and not url.query and not url.fragment
            and url.path in ("", "/") and url.port), "Expected an explicit private HTTP endpoint"
    return base_url.rstrip("/")


def classification_schema(questions):
    return {"title": "public_domain_classification", "type": "object",
            "description": questions["domain"]["instructions"],
            "properties": {"domain": {"type": "string", "enum": list(questions["domain"]["criteria"])}},
            "required": ["domain"], "additionalProperties": False}


def local_llm_client(base_url, model, key, http_client, questions, timeout, profile=None):
    from langchain_openai import ChatOpenAI

    options = ({"extra_body": {"chat_template_kwargs": {
        name: profile[name] for name in ("enable_thinking", "reasoning_effort")}}} if profile else {})
    client = ChatOpenAI(base_url=local_endpoint(base_url) + "/v1", api_key=key, model=model,
                        temperature=0, max_tokens=profile["max_tokens"] if profile else 64,
                        max_retries=0, timeout=timeout, http_async_client=http_client, **options)
    assert client.root_async_client.max_retries == 0
    return client.with_structured_output(classification_schema(questions), method="json_schema",
                                         strict=True, include_raw=True)


async def observe_classification_response(response, captured):
    """Observe provider metadata lost by LangChain without retaining private text."""
    if response.request.method != "POST" or response.request.url.path != "/v1/chat/completions":
        return
    await response.aread()
    request = json.loads(response.request.content)
    record = {"http_status": response.status_code, "request_sha256": digest(request),
              "messages_sha256": digest({"messages": request["messages"]}),
              "response_sha256": hashlib.sha256(response.content).hexdigest(),
              "request_options": {k: request[k] for k in ("model", "temperature", "max_tokens",
                                  "max_completion_tokens", "chat_template_kwargs") if k in request}}
    captured.append(record)
    try:
        body = response.json()
    except ValueError:
        record["invalid_json"] = True
        return
    record.update(reported_model=body.get("model"), usage=body.get("usage"), choices=[])
    for choice in body.get("choices", []):
        message = choice.get("message") or {}
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        content = message.get("content") or ""
        record["choices"].append({"finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len(reasoning), "reasoning_sha256": digest(reasoning),
            "content_chars": len(content), "content_sha256": digest(content)})


async def local_classification_records(jobs, infer, output, model, stop_file, timeout):
    """One in-flight task; failed attempts remain in the same immutable stream."""
    with output.open("x") as stream:
        for index, job in enumerate(jobs, 1):
            if stop_file.exists():
                raise RuntimeError("Classification stop marker present; partial output retained")
            started = time.perf_counter()
            record = {"case_id": job["case_id"], "task": "classification", "repeat": 1,
                      "input_sha256": digest(job)}
            try:
                result = await asyncio.wait_for(infer(job["state"]), timeout=timeout)
                assert result["prediction"] in DOMAINS
                record.update(result, status="valid")
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                if hasattr(error, "classification_transport"):
                    record["transport"] = error.classification_transport
                response = getattr(error, "response", None)
                if response is not None:
                    record["http_status"] = response.status_code
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 100 == 0 or index == len(jobs) or record["status"] == "failed":
                print(json.dumps({"model": model, "completed": index, "total": len(jobs),
                                  "status": record["status"]}), flush=True)


async def local_classification_model(args, family, served_model, key, jobs, questions, warmup, output):
    import httpx
    from deployment_probe import native_choice_request, native_choice_result

    base = local_endpoint(args.base_url)
    profile = getattr(args, "llm_profile", None)
    captured = []

    async def observe(response):
        await observe_classification_response(response, captured)

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False, follow_redirects=False,
                                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                                transport=httpx.AsyncHTTPTransport(retries=0),
                                event_hooks={"response": [observe]} if profile else None) as http:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                response = await http.get(base + "/health", headers={"Authorization": "Bearer " + key}, timeout=5)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            state = subprocess.check_output(["docker", "inspect", "decision-acceptance-model",
                                             "--format", "{{.State.Status}}"], text=True).strip()
            if state != "running" or time.monotonic() > deadline:
                raise RuntimeError("Model readiness failed; startup is not a scored attempt")
            await asyncio.sleep(3)

        llm = (local_llm_client(base, served_model, key, http, questions, args.timeout, profile)
               if family == "llm" else None)

        async def infer(state):
            if llm is not None:
                active = {}
                captured.clear()
                try:
                    prediction = await llm_classify(llm, state, questions, active, args.timeout)
                    if profile:
                        assert len(captured) == 1 and captured[0]["http_status"] == 200
                        assert len(captured[0]["choices"]) == 1
                        assert captured[0]["choices"][0]["finish_reason"] == "stop", "Incomplete model output"
                except Exception as error:
                    if profile:
                        error.classification_transport = list(captured)
                    raise
                return {"prediction": prediction, "calls": active["decision_calls"],
                        **({"transport": list(captured)} if profile else {})}
            payload = {"model": served_model, "state": state, "questions": questions}
            path, body = native_choice_request(family, payload)
            response = await http.post(base + path, json=body, headers={"Authorization": "Bearer " + key})
            response.raise_for_status()
            raw = response.json()
            prediction, probabilities = native_choice_result(family, payload, raw)
            return {"prediction": prediction, "probabilities": probabilities,
                    "request_sha256": digest(body), "native_response": raw}

        warmup_records = []
        for _ in range(3):
            for job in warmup:
                started = time.perf_counter()
                result = await asyncio.wait_for(infer(job["state"]), timeout=args.timeout)
                warmup_records.append({"input": job, "result": result,
                                       "elapsed_ms": (time.perf_counter() - started) * 1000})
        write_json(output / "warmup.private.json", warmup_records)
        await local_classification_records(jobs, infer, output / "responses.private.jsonl",
                                           served_model, args.output / "STOP", args.timeout)


def local_classification_summary(inputs, labels, records):
    gold = {row["case_id"]: row["gold"] for row in labels if row["task"] == "classification"}
    assert len(records) == len(gold) == len(inputs)
    assert len({row["case_id"] for row in records}) == len(gold)
    originals = {row["case_id"]: row for row in inputs}
    assert set(originals) == set(gold)
    for row in records:
        assert row["input_sha256"] == digest(originals[row["case_id"]])
    predictions = {row["case_id"]: row["prediction"] for row in records if row["status"] == "valid"}
    valid = [row["elapsed_ms"] for row in records if row["status"] == "valid"]
    return {**classification_metrics(gold, predictions), "records": len(records),
            "valid_output_rate": len(valid) / len(records),
            "successful_task_latency": timing(valid),
            "all_attempt_latency": timing([row["elapsed_ms"] for row in records]),
            "failures": dict(Counter(row["error_type"] for row in records if row["status"] != "valid"))}


def classification_roster_history(directory, models, reference):
    """Validate the immutable predecessor chain before adding new model names."""
    visited, planned, completed = set(), set(), set()
    link = {"directory": str(directory), "protocol_sha256": digest(reference)}
    while link:
        current = Path(link["directory"]).resolve(strict=True)
        assert current.parent == directory.resolve().parent and current not in visited, "Invalid predecessor path"
        visited.add(current)
        prior = json.loads((current / "protocol.private.json").read_text())
        status = json.loads((current / "status.json").read_text())
        assert digest(prior) == link["protocol_sha256"], "Predecessor protocol changed"
        for field in ("parent_dataset_protocol", "inputs_sha256", "labels_sha256", "records", "domain_questions"):
            assert prior[field] == reference[field], "Predecessor dataset changed"
        done = status["completed"]
        if "completed" in link:
            assert done == link["completed"], "Predecessor completion changed"
        assert len(done) == len(set(done)) and set(done).issubset(prior["models"])
        assert not completed.intersection(done), "Duplicate completed models in chain"
        for model in done:
            folder = current / model
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["records"] == reference["records"]
            assert all((folder / name).is_file() for name in ("deployment.json", "responses.private.jsonl"))
        completed.update(done)
        planned.update(prior["models"])
        link = prior.get("predecessor")
    assert models and not planned.intersection(models), "Roster extension must contain only new models"
    return {"completed_models": sorted(completed),
            "parent_dataset_protocol": reference["parent_dataset_protocol"],
            "inputs_sha256": reference["inputs_sha256"], "labels_sha256": reference["labels_sha256"]}


def classification_predecessor(directory, models, resume_unstarted=False, extend_roster=False):
    status = json.loads((directory / "status.json").read_text())
    prior = json.loads((directory / "protocol.private.json").read_text())
    completed = status["completed"]
    assert completed and not set(completed).intersection(models), "Do not repeat completed models"
    extension = None
    if extend_roster:
        assert not resume_unstarted and status["state"] == "complete", "Roster extension requires a complete sequence"
        assert set(completed) == set(prior["models"]), "Sequence has pending models"
        extension = classification_roster_history(directory, models, prior)
    unstarted = None
    if resume_unstarted:
        active = status.get("active")
        assert status["state"] == "failed" and active in prior["models"] and active not in completed
        assert models and models[0] == active, "Resume the unstarted model before later models"
        pending = directory / active
        assert (pending / "deployment.json").is_file(), "Missing failed deployment evidence"
        assert not any((pending / name).exists() for name in ("responses.private.jsonl", "summary.json")), \
            "Formal attempts must not be replayed"
        unstarted = {"model": active, "error_type": status.get("error_type"), "formal_attempts": 0}
    assert unstarted is not None or status["state"] == "complete" or (
        (directory / "STOP").exists() and status["active"] in completed
    ), "Previous sequence did not finish its active model"
    for model in completed:
        summary = json.loads((directory / model / "summary.json").read_text())
        assert summary["records"] == prior["records"]
    assert extension is not None or set(models).issubset(set(prior["models"]) - set(completed))
    return {"directory": str(directory), "protocol_sha256": digest(prior), "completed": completed,
            **({"roster_extension": extension} if extension is not None else {}),
            "unstarted_recovery": unstarted,
            "excluded": [model for model in prior["models"] if model not in completed and model not in models]}


def run_local_classification(args):
    """Extend the existing benchmark with an isolated, sequential deployment mode."""
    import fcntl
    import importlib.metadata

    os.umask(0o077)
    for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGFUSE_ENABLED", "LANGFUSE_TRACING_ENABLED"):
        os.environ[name] = "false"
    sys.path.insert(0, str(args.service_source.resolve(strict=True)))
    from launch_reviewed import CLASSIFICATION_PROFILES, MODELS, launch, read_key

    local_endpoint(args.base_url)
    assert urlsplit(args.base_url).port == args.port
    assert len(args.models) == len(set(args.models))
    assert all(model in MODELS and MODELS[model][0] != "reranker" for model in args.models)
    profiles = [CLASSIFICATION_PROFILES[m] for m in args.models if m in CLASSIFICATION_PROFILES]
    assert not profiles or len(args.models) == 1, "Run a comparison profile in its own sequence"
    args.llm_profile = profiles[0] if profiles else None
    protocol = json.loads((args.dataset / "dataset-protocol.private.json").read_text())
    for filename, field in (("inputs.private.jsonl", "inputs_sha256"), ("labels.private.jsonl", "labels_sha256")):
        assert hashlib.sha256((args.dataset / filename).read_bytes()).hexdigest() == protocol[field]
    inputs = [r for r in read_jsonl(args.dataset / "inputs.private.jsonl") if r["task"] == "classification"]
    inference_jobs(inputs, 1)
    assert len(inputs) == protocol["classification_cases"] == 2974
    assert len({digest(row["state"]) for row in inputs}) == protocol["classification_distinct_inputs"] == 2921
    assert protocol["domain_questions"] == DOMAIN_QUESTIONS
    labels = read_jsonl(args.dataset / "labels.private.jsonl")
    warmup_path = args.dataset / "preflight.inputs.private.jsonl"
    assert hashlib.sha256(warmup_path.read_bytes()).hexdigest() == protocol["preflight_sha256"]
    synthetic = next(row for row in read_jsonl(warmup_path) if row["task"] == "classification")
    warmup = [{**synthetic, "case_id": "synthetic:classification:" + str(length),
               "state": {"utterance": synthetic["state"]["utterance"] * length}} for length in (1, 16, 64)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.output.parent / "classification.lock").open("a")
    assert not args.resume_unstarted or args.after_run, "Unstarted recovery requires a predecessor"
    assert not args.extend_roster or args.after_run, "Roster extension requires a predecessor"
    if args.after_run:
        assert args.after_run.resolve().parent == args.output.resolve().parent
        print(json.dumps({"waiting_for_sequence": str(args.after_run)}), flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.after_run else fcntl.LOCK_NB))
    predecessor = classification_predecessor(args.after_run, args.models, args.resume_unstarted,
                                             args.extend_roster) if args.after_run else None
    if args.extend_roster:
        history = predecessor["roster_extension"]
        assert history["parent_dataset_protocol"] == digest(protocol), "Extension dataset changed"
        assert all(history[field] == protocol[field] for field in ("inputs_sha256", "labels_sha256"))
    args.output.mkdir(mode=0o700, exist_ok=False)
    code_paths = [Path(__file__), args.service_source / "deployment_probe.py", args.service_source / "launch_reviewed.py"]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    new_protocol = {"version": "classification-local-20261015", "parent_dataset_protocol": digest(protocol),
                    "inputs_sha256": protocol["inputs_sha256"], "labels_sha256": protocol["labels_sha256"],
                    "models": args.models, "records": len(inputs), "domain_questions": DOMAIN_QUESTIONS,
                    "base_url": args.base_url, "concurrency": 1, "repeats": 1, "timeout_seconds": args.timeout,
                    "retries": 0, "fallback": False, "truncation": False, "labels_sent": False,
                    "warmup": "3 independent synthetic lengths, 3 rounds; excluded",
                    "main_latency": ["successful_task_mean_ms", "successful_task_p99_ms"],
                    "timer": "before request construction through output validation; no file I/O",
                    "source_sha256": source_hashes, "predecessor": predecessor,
                    "llm": {"method": "json_schema", "strict": True, "temperature": 0, "max_tokens": 64},
                    "client_packages": {name: importlib.metadata.version(name) for name in
                                        ("httpx", "langchain-openai", "langchain-core", "openai")}}
    if args.llm_profile:
        new_protocol["llm"].update(max_tokens=args.llm_profile["max_tokens"], profile=args.llm_profile,
                                   transport_observer="hashes, usage, finish reasons and reasoning lengths; no private text")
    write_json(args.output / "protocol.private.json", new_protocol)
    write_json(args.output / "client-packages.json", {
        dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()})
    key = read_key(args.env_file)
    completed = []
    status = {"pid": os.getpid(), "state": "starting", "completed": completed, "models": args.models}

    def save_status():
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "status.json", status)

    save_status()
    try:
        for model in args.models:
            if (args.output / "STOP").exists():
                raise RuntimeError("Classification stop marker present")
            status.update(state="loading", active=model)
            save_status()
            directory = args.output / model
            directory.mkdir()
            launch(SimpleNamespace(root=args.root, model=model, env_file=args.env_file, native_port=args.port - 1,
                                   public_port=args.port, seqs=16, tokens=16384, enforce_eager=False))
            identity = json.loads((args.root / "acceptance-20261015/evidence" / (model + ".deployment.json")).read_text())
            write_json(directory / "deployment.json", identity)
            status["state"] = "warmup_and_classification"
            save_status()
            asyncio.run(local_classification_model(args, MODELS[model][0], MODELS[model][2], key,
                                                   inputs, DOMAIN_QUESTIONS, warmup, directory))
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in source_hashes.items())
            summary = local_classification_summary(inputs, labels, read_jsonl(directory / "responses.private.jsonl"))
            write_json(directory / "summary.json", {"model": model, **summary})
            completed.append(model)
            save_status()
            print(json.dumps({"model_complete": model, "accuracy": summary["accuracy"],
                              "valid_output_rate": summary["valid_output_rate"],
                              "latency": summary["successful_task_latency"]}), flush=True)
        status.update(state="complete", active=None)
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        save_status()
        lock.close()


def classification_metrics(labels, predictions):
    classes = list(DOMAINS)
    confusion = {g: dict.fromkeys(classes + ["failed"], 0) for g in classes}
    for case_id, gold in labels.items():
        predicted = predictions.get(case_id)
        confusion[gold][predicted if predicted in classes else "failed"] += 1
    return classification_from_confusion(confusion, len(labels))


def classification_from_confusion(confusion, count):
    classes = list(DOMAINS)
    f1 = []
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in classes if g != c)
        fn = sum(confusion[c][p] for p in classes + ["failed"] if p != c)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"accuracy": sum(confusion[c][c] for c in classes) / count,
            "macro_f1": statistics.mean(f1), "confusion": confusion}


def bootstrap_mean(values, samples=2000):
    assert values and all(math.isfinite(v) for v in values)
    rng = random.Random(bytes.fromhex("6b6e6f77762d7075626c69632d76332d6369").decode("utf-8"))
    estimates = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"independent_cases": len(values), "point": statistics.mean(values), "confidence": 0.95,
            "low": percentile(estimates, 0.025), "high": percentile(estimates, 0.975), "samples": samples}


def classification_interval(labels, left, right=None, samples=2000, case_groups=None):
    """Keep identical inputs and all their repeated requests in one paired cluster."""
    case_groups = case_groups if case_groups is not None else {k: k for k in labels}
    assert case_groups.keys() == labels.keys()
    clusters, strata = {}, {}
    for case, group in case_groups.items():
        clusters.setdefault(group, []).append(case)
    for cases in clusters.values():
        strata.setdefault(tuple(sorted({labels[k] for k in cases})), []).append(cases)
    groups = list(strata.values())
    rng = random.Random(bytes.fromhex("6b6e6f77762d7075626c69632d76332d636c6173732d6369").decode("utf-8"))

    def score(predictions, weights):
        per_repeat = []
        for repeat in sorted(predictions):
            confusion = {g: dict.fromkeys(list(DOMAINS) + ["failed"], 0) for g in DOMAINS}
            for case, weight in weights.items():
                predicted = predictions[repeat].get(case)
                confusion[labels[case]][predicted if predicted in DOMAINS else "failed"] += weight
            per_repeat.append(classification_from_confusion(confusion, sum(weights.values())))
        return {metric: statistics.mean(r[metric] for r in per_repeat) for metric in ["accuracy", "macro_f1"]}

    def estimate(weights):
        result = score(left, weights)
        if right is not None:
            reference = score(right, weights)
            result = {k: v - reference[k] for k, v in result.items()}
        return result

    point = estimate(dict.fromkeys(labels, 1))
    draws = [estimate(Counter(case for group in groups for cluster in rng.choices(group, k=len(group)) for case in cluster))
             for _ in range(samples)]
    return {"independent_cases": len(clusters), "source_cases": len(labels), "confidence": 0.95, "samples": samples,
            "sampling": "gold-label-set-stratified identical-input clusters; original cases and repeats stay together",
            "metrics": {k: {"point": v, "low": percentile([r[k] for r in draws], 0.025),
                            "high": percentile([r[k] for r in draws], 0.975)} for k, v in point.items()}}


def rank_case_values(labels, records, repeats):
    values = {}
    indexed = {(r["case_id"], r["repeat"]): r for r in records}
    for case, label in labels.items():
        outcomes = [rerank_metrics({case: label}, {case: indexed[(case, repeat)]}) for repeat in range(1, repeats + 1)]
        values[case] = {"ndcg5": statistics.mean(r["ndcg_at_k"]["5"] for r in outcomes),
                        "mrr": statistics.mean(r["mrr"] for r in outcomes),
                        "all_attempt_mean_ms": statistics.mean(indexed[(case, repeat)]["elapsed_ms"] for repeat in range(1, repeats + 1))}
    return values


def retrieval_module():
    path = Path(__file__).parents[1] / "src/evaluation/retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("public_retrieval_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rerank_metrics(labels, records):
    metrics = retrieval_module()
    cases = [metrics.EvalCase(case_id=case_id, query="",
                             expected_evidence=[metrics.Evidence(fragment_id=p) for p in row["positive_ids"]])
             for case_id, row in labels.items()]
    results = {}
    for case_id, record in records.items():
        scores = record.get("scores", []) if record["status"] == "valid" else []
        # Stable content identity breaks tied scores independently of the gold class.
        ranked = sorted(scores, key=lambda s: (-s["score"], s["fragment_id"]))
        results[case_id] = metrics.EvalRunResult(case_id=case_id, retrieved_items=[
            metrics.RetrievedItem(fragment_id=s["fragment_id"], rank=i + 1, order=i)
            for i, s in enumerate(ranked)])
    return metrics.evaluate_cases(cases, results, k_values=(1, 5, 10)).stage_metrics["final"].to_dict()


def repeat_agreement(records, case_ids, repeats):
    if repeats < 2 or not case_ids:
        return None
    grouped = {case_id: [r for r in records if r["case_id"] == case_id] for case_id in case_ids}
    stable = sum(len(rows) == repeats and all(r["status"] == "valid" for r in rows)
                 and len({r.get("prediction") for r in rows}) == 1 for rows in grouped.values())
    return stable / len(case_ids)


def summarize(args):
    protocol = json.loads((args.dataset / "dataset-protocol.private.json").read_text())
    for filename, key in [("inputs.private.jsonl", "inputs_sha256"), ("labels.private.jsonl", "labels_sha256")]:
        assert hashlib.sha256((args.dataset / filename).read_bytes()).hexdigest() == protocol[key]
    assert hashlib.sha256((Path(__file__).parents[1] / "src/evaluation/retrieval_eval.py").read_bytes()).hexdigest() == protocol["scorer_sha256"]
    inputs = {r["case_id"]: r for r in read_jsonl(args.dataset / "inputs.private.jsonl")}
    labels = read_jsonl(args.dataset / "labels.private.jsonl")
    route_labels = {r["case_id"]: r["gold"] for r in labels if r["task"] == "classification"}
    route_clusters = {case: digest(inputs[case]["state"]) for case in route_labels}
    rank_labels = {r["case_id"]: r for r in labels if r["task"] == "reranking"}
    reports, routes_by_model, ranks_by_model = {}, {}, {}
    for directory in args.runs:
        frozen = json.loads((directory / "run-protocol.private.json").read_text())
        assert not frozen["warmup"] and frozen["dataset_protocol_sha256"] == digest(protocol)
        assert frozen.get("measurement_role", "formal") == "formal", "Shared-load records are not formal results"
        records = read_jsonl(directory / "responses.private.jsonl")
        manifest = json.loads((directory / "manifest.private.json").read_text())
        assert manifest["source_unchanged"] and not manifest["blocked_destinations"] and manifest["returncode"] == 0
        tasks = frozen.get("tasks", ["classification", "reranking"])
        selected_inputs = {k: v for k, v in inputs.items() if v["task"] in tasks}
        expected = {(case_id, repeat) for case_id in selected_inputs for repeat in range(1, protocol["repeats"] + 1)}
        actual = [(r["case_id"], r["repeat"]) for r in records]
        assert len(actual) == len(set(actual)) and set(actual) == expected, "Missing or duplicate cases"
        for r in records:
            original = inputs[r["case_id"]]
            assert r["task"] == original["task"] and r["input_sha256"] == digest(original)
            if r["status"] == "valid":
                concurrent = frozen.get("rerank_transport") == "systemone_concurrent"
                if frozen.get("rerank_transport") == "listwise_decision" and r["task"] == "reranking":
                    assert len(r["calls"]) >= 1
                    if r["status"] == "valid":
                        assert any(c["status"] == "valid" for c in r["calls"])
                else:
                    expected_calls = (1 if r["task"] == "classification" or (protocol.get("version") == "v3" and not concurrent)
                                      else len(original["candidates"]))
                    assert len(r["calls"]) == expected_calls and all(c["status"] == "valid" for c in r["calls"])
                if r["task"] == "reranking":
                    assert len(r["scores"]) == len(original["candidates"]) and {s["fragment_id"] for s in r["scores"]} == {c["fragment_id"] for c in original["candidates"]}
                    if concurrent:
                        expected_hashes = Counter(digest({"state": {"query": original["query"], "passage": c["content"]},
                                                          "questions": protocol["score_questions"]})
                                                  for c in original["candidates"])
                        assert Counter(c["input_sha256"] for c in r["calls"]) == expected_hashes
                    elif protocol.get("version") == "v3" and frozen.get("rerank_transport") != "listwise_decision":
                        body = rerank_body({**frozen, "score_questions": protocol["score_questions"]}, original)
                        assert r["calls"][0]["input_sha256"] == digest(body)
        per_repeat = []
        for repeat in range(1, protocol["repeats"] + 1):
            selected = [r for r in records if r["repeat"] == repeat]
            routes = {r["case_id"]: r.get("prediction") for r in selected if r["task"] == "classification"}
            ranks = {r["case_id"]: r for r in selected if r["task"] == "reranking"}
            per_repeat.append({"repeat": repeat, "classification": classification_metrics(route_labels, routes) if "classification" in tasks else None,
                               "reranking": rerank_metrics(rank_labels, ranks) if "reranking" in tasks else None,
                               "reranking_valid_only": rerank_metrics(
                                   {k: v for k, v in rank_labels.items() if ranks[k]["status"] == "valid"},
                                   {k: v for k, v in ranks.items() if v["status"] == "valid"}
                               ) if any(r["status"] == "valid" for r in ranks.values()) else None})
        groups = {task: [r for r in records if r["task"] == task] for task in ["classification", "reranking"]}
        stats = {}
        for task, group in groups.items():
            calls = [c for r in group for c in r["calls"]]
            valid_calls = [c for c in calls if c["status"] == "valid"]
            tokens = [c["result"].get("input_tokens", c["result"].get("usage", {}).get("input_tokens",
                      c["result"].get("usage", {}).get("prompt_tokens"))) for c in valid_calls]
            tokens = [v for v in tokens if type(v) is int and v >= 0]
            stats[task] = {"jobs": len(group), "failed_jobs": sum(r["status"] != "valid" for r in group),
                           "calls": len(calls), "failed_calls": len(calls) - len(valid_calls),
                           "failed_http_statuses": dict(Counter(str(c.get("http_status", "none")) for c in calls if c["status"] != "valid")),
                           "client_all_calls": timing([c["client_elapsed_ms"] for c in calls]),
                           "client_valid_calls": timing([c["client_elapsed_ms"] for c in valid_calls]),
                           "server_valid_calls": timing([c["result"]["latency_ms"] for c in valid_calls
                                                         if type(c["result"].get("latency_ms")) in (int, float)]),
                           "whole_job_including_pacing": timing([r["elapsed_ms"] for r in group]),
                           "valid_input_tokens": {"reported_calls": len(tokens), "min": min(tokens, default=None), "max": max(tokens, default=None)}}
        report = {"model": frozen["model"], "classification_source_cases": len(route_labels) if "classification" in tasks else 0,
                  "independent_classification_cases": len(set(route_clusters.values())) if "classification" in tasks else 0,
                  "independent_rerank_pools": len(rank_labels) if "reranking" in tasks else 0, "repeats": protocol["repeats"],
                  "classification_repeat_agreement": repeat_agreement(groups["classification"], route_labels,
                                                                        protocol["repeats"]) if "classification" in tasks else None,
                  "rerank_transport": frozen.get("rerank_transport", "native_batch"),
                  "tasks": tasks, "measurement_role": frozen.get("measurement_role", "formal"),
                  "per_repeat": per_repeat, "timing_and_failures": stats, "audit_passed": True,
                  "source_sha256": manifest["source_sha256"]}
        if protocol.get("version") == "v3":
            if "classification" in tasks:
                routes_by_model[frozen["model"]] = {repeat: {r["case_id"]: r.get("prediction") for r in groups["classification"]
                                                            if r["repeat"] == repeat} for repeat in range(1, protocol["repeats"] + 1)}
                report["classification_interval"] = classification_interval(route_labels, routes_by_model[frozen["model"]],
                                                                            case_groups=route_clusters)
            if "reranking" in tasks:
                values = rank_case_values(rank_labels, groups["reranking"], protocol["repeats"])
                ranks_by_model[frozen["model"]] = values
                report["reranking_intervals"] = {metric: bootstrap_mean([row[metric] for row in values.values()])
                                                  for metric in ["ndcg5", "mrr", "all_attempt_mean_ms"]}
        reports[frozen["model"]] = report
    paired = {"classification": [], "reranking": []}
    for i, model in enumerate(routes_by_model):
        for reference in list(routes_by_model)[:i]:
            paired["classification"].append({"model": model, "reference": reference, "difference":
                classification_interval(route_labels, routes_by_model[model], routes_by_model[reference], case_groups=route_clusters)})
    for i, model in enumerate(ranks_by_model):
        for reference in list(ranks_by_model)[:i]:
            assert ranks_by_model[model].keys() == ranks_by_model[reference].keys()
            paired["reranking"].append({"model": model, "reference": reference, "difference": {
                metric: bootstrap_mean([values[metric] - ranks_by_model[reference][case][metric]
                                        for case, values in ranks_by_model[model].items()])
                for metric in ["ndcg5", "mrr", "all_attempt_mean_ms"]}})
    unranked = {case_id: {"status": "valid", "scores": [
        {"fragment_id": c["fragment_id"], "score": -index}
        for index, c in enumerate(inputs[case_id]["candidates"])]} for case_id in rank_labels}
    write_json(args.output, {"dataset_protocol_sha256": digest(protocol), "models": reports, "paired_intervals": paired,
                             "unranked_hash_shuffled_pool": rerank_metrics(rank_labels, unranked),
                             "limitations": ["public training exposure unknown", "dataset-specific proxy, not 产品 route accuracy", "binary derivative labels", "shared GPU host; consult environment gates before comparing speed"]})
    for model, report in reports.items():
        repeats = report["per_repeat"]
        print(json.dumps({"model": model,
                          "accuracy_mean": statistics.mean(r["classification"]["accuracy"] for r in repeats) if "classification" in report["tasks"] else None,
                          "macro_f1_mean": statistics.mean(r["classification"]["macro_f1"] for r in repeats) if "classification" in report["tasks"] else None,
                          "ndcg5_mean": statistics.mean(r["reranking"]["ndcg_at_k"]["5"] for r in repeats) if "reranking" in report["tasks"] else None,
                          "mrr_mean": statistics.mean(r["reranking"]["mrr"] for r in repeats) if "reranking" in report["tasks"] else None,
                          "classification_client": report["timing_and_failures"]["classification"]["client_valid_calls"],
                          "rerank_pool": report["timing_and_failures"]["reranking"]["whole_job_including_pacing"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare-capacity", "local-capacity"}:
        from decision_capacity_preflight import register
        register(sub)
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare-passage", "local-passage"}:
        from decision_passage_benchmark import register
        register(sub)
    p = sub.add_parser("local-classification", help="Sequential isolated models; no product bindings or load test")
    for name in ("root", "service-source", "dataset", "env-file", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--after-run", type=Path, help="Wait for the same sequence lock and preserve completed models")
    p.add_argument("--resume-unstarted", action="store_true", help="Explicitly resume a failed deployment before any formal attempts")
    p.add_argument("--extend-roster", action="store_true", help="Add only new models after a completed sequence, preserving its protocol chain")
    p.add_argument("--base-url", required=True)
    p.add_argument("--port", type=int, default=30889)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--startup-timeout", type=float, default=2400)
    p.set_defaults(function=run_local_classification)
    p = sub.add_parser("prepare")
    p.add_argument("--downloads", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-domain", type=int, default=20)
    p.add_argument("--pools", type=int, default=32)
    p.add_argument("--version", choices=["v1", "v3"], default="v1")
    p.add_argument("--exclude-dataset", type=Path)
    p.add_argument("--repeats", type=int, choices=[1, 3], default=3)
    p.set_defaults(function=prepare)
    p = sub.add_parser("run")
    for name in ["dataset", "access", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["host", "model"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--endpoint")
    p.add_argument("--container", default="llm_wiki_supervisor")
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--tasks", nargs="+", choices=["classification", "reranking"], default=["classification", "reranking"])
    p.add_argument("--ranker-kind", choices=["decision", "qwen"], default="decision")
    p.add_argument("--rerank-transport", choices=["native_batch", "vllm", "systemone_concurrent", "listwise_decision"], default="native_batch")
    p.add_argument("--pool-concurrency", choices=["bounded", "full_pool"], default="bounded",
                   help="For pointwise Decision scoring, start one request per candidate when full_pool is selected.")
    p.add_argument("--classifier-kind", choices=["decision", "llm"], default="decision")
    p.add_argument("--llm-baseline-config", type=Path)
    p.add_argument("--batch-size", type=int, choices=[1, 4, 8], default=1)
    p.add_argument("--measurement-role", choices=["functional_shared_load", "formal"], default="formal")
    p.add_argument("--limit", type=int)
    p.add_argument("--stop-file", type=Path)
    p.set_defaults(function=run_candidate)
    p = sub.add_parser("summarize")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)
