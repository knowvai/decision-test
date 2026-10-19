import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import decision_capacity_preflight as capacity


def jobs(count):
    return [{"case_id": "test:" + str(i), "fragment_id": str(i),
             "state": {"query": "synthetic query", "document": "synthetic passage " + str(i)},
             "input_sha256": str(i), "chars": 40} for i in range(count)]


@pytest.mark.parametrize("n", [1, 5, 10, 20])
def test_exact_concurrency_single_attempt_and_complete_records(n):
    calls, active, peak = [], 0, 0

    async def infer(job):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(job["case_id"])
        await asyncio.sleep(0.002)
        active -= 1
        return {"prediction": "relevant", "calls": 1}

    rows, block = asyncio.run(capacity.measure(jobs(47), infer, n, 10))
    assert peak == block["peak_inflight"] == n
    assert len(set(calls)) == len(calls) == len(rows) == 47
    assert all(r["status"] == "valid" and r["calls"] == 1 for r in rows)
    assert [r["sequence"] for r in rows] == list(range(47))
    assert all("state" not in r for r in rows)


def test_workers_refill_without_batch_barrier():
    events = []

    async def infer(job):
        i = int(job["fragment_id"])
        events.append(("start", i))
        await asyncio.sleep(0.03 if i == 0 else 0.001)
        events.append(("end", i))
        return {"prediction": "irrelevant"}

    asyncio.run(capacity.measure(jobs(6), infer, 2, 10))
    assert events.index(("start", 2)) < events.index(("end", 0))


def test_failures_retained_no_retry_and_throughput_uses_valid_only():
    calls = []

    async def infer(job):
        calls.append(job["case_id"])
        if job["fragment_id"] == "1":
            error = ValueError("private body must not be stored")
            error.capacity_metadata = {"http_status": 429, "calls": 1}
            raise error
        return {"prediction": "relevant"}

    rows, block = asyncio.run(capacity.measure(jobs(4), infer, 2, 10))
    assert len(calls) == 4
    assert rows[1]["status"] == "failed" and rows[1]["http_status"] == 429
    report = capacity.block_summary(rows, block)
    assert report["requests"] == 4 and report["valid"] == 3 and report["failed"] == 1
    assert report["successful_latency_ms"]["n"] == 3
    assert report["all_attempt_latency_ms"]["n"] == 4
    assert report["effective_requests_per_second"] == 3 / block["wall_seconds"]
    assert "private body" not in json.dumps(rows)


def test_timeout_retained_without_resubmission():
    calls = []

    async def infer(job):
        calls.append(job["case_id"])
        await asyncio.sleep(1)

    rows, _ = asyncio.run(capacity.measure(jobs(2), infer, 2, 0.002))
    assert len(calls) == 2 and all(r["error_type"] == "TimeoutError" for r in rows)


def test_length_selection_never_truncates_or_uses_labels():
    pools = [{"case_id": "p", "query": "Q", "positive_ids": ["b"], "candidates": [
        {"fragment_id": "a", "content": "short"}, {"fragment_id": "b", "content": "long" * 10}]}]
    selected, excluded = capacity.workload(pools, 10)
    assert len(selected) == len(excluded) == 1
    assert selected[0]["state"] == {"query": "Q", "document": "short"}
    assert "positive_ids" not in json.dumps(selected)
    assert excluded[0]["chars"] == 41 and "state" not in excluded[0]


def response(model="Qwen3.8-27B", reason="stop", reasoning=None):
    return {"model": model, "choices": [{"finish_reason": reason, "message": {
        "content": '{"choice":"relevant"}', "reasoning": reasoning}}],
        "usage": {"completion_tokens": 10, "prompt_tokens": 50}}


def test_llm_schema_modes_and_reasoning_metadata_only():
    profile = {"enable_thinking": True, "reasoning_effort": "xhigh"}
    body = capacity.llm_body("Qwen3.8-27B", jobs(1)[0]["state"])
    assert body["temperature"] == 0 and body["max_tokens"] == 64
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["strict"]
    body = capacity.llm_body("Qwen3.8-27B", jobs(1)[0]["state"], profile)
    assert body["max_tokens"] == 4096 and body["chat_template_kwargs"] == profile
    result = capacity.llm_result(response(reasoning="private reasoning"), "Qwen3.8-27B", profile)
    assert result["reasoning_chars"] == 17
    assert "private reasoning" not in json.dumps(result)


@pytest.mark.parametrize("raw,model,profile", [
    (response(reason="length"), "Qwen3.8-27B", None),
    (response(), "wrong-model", None),
    (response(), "Qwen3.8-27B", {"enable_thinking": True}),
    (response(reasoning="unexpected"), "Qwen3.8-27B", None),
])
def test_llm_rejects_incomplete_or_wrong_mode(raw, model, profile):
    with pytest.raises(AssertionError):
        capacity.llm_result(raw, model, profile)


def test_summary_pairs_by_identity_not_completion_order():
    blocks = []
    for repeat in range(1, 4):
        for n in capacity.LEVELS:
            rows = [{"case_id": "a", "fragment_id": "1", "status": "valid",
                     "prediction": "irrelevant" if n == 20 else "relevant", "elapsed_ms": 100*n}]
            blocks.append({"repeat": repeat, "concurrency": n, "wall_seconds": 1,
                           "requests": 1, "peak_inflight": 1, "records": rows})
    result = capacity.summarize(blocks)
    assert result["levels"]["20"]["decision_changes_vs_n1"] == 3
    assert result["levels"]["20"]["mean_ms_ratio_to_n1"] == 20
    assert result["levels"]["1"]["decision_changes_vs_n1"] == 0
    assert result["n1_repeat_consistency"] == {"paired_valid": 2, "changes": 0}


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "rows.jsonl"
    capacity.save_rows(path, [{"preserved": True}])
    with pytest.raises(FileExistsError):
        capacity.save_rows(path, [])
    assert json.loads(path.read_text()) == {"preserved": True}


def test_artifact_preservation_detects_changes(tmp_path):
    path = tmp_path / "artifact.json"
    capacity.save(path, {"ok": True})
    expected = capacity.sha(path)
    capacity.check_artifacts({str(path): expected})
    with pytest.raises(AssertionError):
        capacity.check_artifacts({str(path): "wrong hash"})


def test_native_mapping_is_reused_without_changing_single_pair_semantics():
    sys.path.insert(0, str(Path(__file__).parents[3] / "deploy/docker/decision-eval"))
    from deployment_probe import native_choice_request
    state = jobs(1)[0]["state"]
    payload = {"model": "model", "state": state, "questions": capacity.QUESTIONS}
    path, body = native_choice_request("worker", payload)
    assert path == "/v1/systemone" and body["state"] == state
    assert body["questions"]["relevance"]["require_full_input"] is True
    path, body = native_choice_request("jev", payload)
    assert json.loads(body["state"]) == state
    assert body["options"] == list(capacity.QUESTIONS["relevance"]["criteria"].values())
    assert body["strategy"] == "single" and body["thinking"] == "off"


def test_http_path_reuses_connections_and_records_no_private_text(tmp_path, monkeypatch):
    import httpx

    sys.path.insert(0, str(Path(__file__).parents[3] / "deploy/docker/decision-eval"))
    key = "test-only-key"
    active, peak, calls = 0, 0, []
    transport_options = []

    async def handle(request):
        nonlocal active, peak
        if request.method == "GET":
            return httpx.Response(200 if request.headers.get("authorization") == "Bearer " + key else 401)
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["response_format"]["json_schema"]["strict"]
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001)
        active -= 1
        return httpx.Response(200, json=response())

    def transport(**kwargs):
        transport_options.append(kwargs)
        return httpx.MockTransport(handle)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    monkeypatch.setattr(capacity, "ORDERS", ((1, 5, 10, 20),))
    args = SimpleNamespace(base_url="http://127.0.0.1:8000", timeout=10, startup_timeout=10, output=tmp_path)
    asyncio.run(capacity.execute_model(args, "qwen3.8-27b", key, jobs(47), tmp_path, lambda *a,**k:None))
    assert len(calls) == 2 * 4 * 20 + 4 * 47
    assert peak == 20 and transport_options[0]["retries"] == 0
    assert transport_options[0]["limits"].max_connections == 20
    rows = capacity.benchmark.read_jsonl(tmp_path / "responses-n20-r1.jsonl")
    assert len(rows) == 47 and all(r["calls"] == 1 and r["status"] == "valid" for r in rows)
    assert "synthetic passage" not in json.dumps(rows)
    assert key not in json.dumps(rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["levels"]["20"]["rounds"][0]["peak_inflight"] == 20


def predecessor_fixture(tmp_path):
    old = tmp_path / "run-001"
    old.mkdir()
    source = tmp_path / "old-source"
    source.mkdir()
    for name, text in (("decision_capacity_preflight.py", Path(capacity.__file__).read_text()),
                       ("public_decision_benchmark.py", "benchmark"), ("deployment_probe.py", "mapping")):
        (source / name).write_text(text)
    sources = {str(p): capacity.sha(p) for p in source.iterdir()}
    protocol = {"version": "pointwise-http-capacity-v1", "dataset_protocol_sha256": "frozen-data",
        "models": ["complete", "pending", "last"], "concurrency_orders": capacity.ORDERS,
        "questions": capacity.QUESTIONS, "requests_per_block": 1, "timeout": 300,
        "startup_timeout": 2400, "max_sequences": 32, "max_batched_tokens": 32768,
        "retries": 0, "fallback": False, "truncation": False, "llm": {"temperature": 0},
        "timer": "frozen", "load": "frozen", "formal_classification": False,
        "formal_passage_scoring": False, "source_sha256": sources, "preserved_artifact_sha256": {}}
    capacity.save(old / "protocol.json", protocol)
    capacity.save(old / "status.json", {"stage": "failed", "models": protocol["models"],
        "completed": ["complete"], "model": "pending"})
    for model in ("complete", "pending"):
        folder = old / model
        folder.mkdir()
        capacity.save(folder / "deployment.json", {"model": model})
    for repeat, order in enumerate(capacity.ORDERS, 1):
        for n in order:
            capacity.save_rows(old / "complete" / f"responses-n{n}-r{repeat}.jsonl", [{"immutable": True}])
    capacity.save(old / "complete/summary.json", {"levels": {str(n): {"requests": 3} for n in capacity.LEVELS}})
    capacity.save_rows(old / "pending/warmup-n20-r1.jsonl", [{"status": "failed", "http_status": 503}])
    return old, protocol


def test_continuation_preserves_completed_and_failed_warmup(tmp_path):
    old, protocol = predecessor_fixture(tmp_path)
    link, artifacts = capacity.capacity_predecessor(old, tmp_path / "run-002", ["pending", "last"], protocol)
    assert link["completed"] == ["complete"]
    assert link["unstarted_recovery"] == {"model": "pending", "measured_requests": 0}
    assert link["measurement_functions_ast_unchanged"]
    assert str(old / "pending/warmup-n20-r1.jsonl") in artifacts
    assert len([name for name in artifacts if "/complete/responses-" in name]) == 12
    capacity.check_artifacts(artifacts)


@pytest.mark.parametrize("models", [["complete", "pending", "last"], ["last"], ["last", "pending"]])
def test_continuation_rejects_replay_skip_or_reordering(tmp_path, models):
    old, protocol = predecessor_fixture(tmp_path)
    with pytest.raises(AssertionError):
        capacity.capacity_predecessor(old, tmp_path / "run-002", models, protocol)


@pytest.mark.parametrize("filename", ["responses-n1-r1.jsonl", "block-n1-r1.json", "summary.json"])
def test_continuation_rejects_any_measured_file_even_empty(tmp_path, filename):
    old, protocol = predecessor_fixture(tmp_path)
    (old / "pending" / filename).touch()
    with pytest.raises(AssertionError):
        capacity.capacity_predecessor(old, tmp_path / "run-002", ["pending", "last"], protocol)


@pytest.mark.parametrize("field,value", [("dataset_protocol_sha256", "changed"), ("timeout", 10),
                                        ("concurrency_orders", [[1, 10]]), ("questions", {})])
def test_continuation_rejects_changed_measurement_contract(tmp_path, field, value):
    old, protocol = predecessor_fixture(tmp_path)
    protocol[field] = value
    with pytest.raises(AssertionError):
        capacity.capacity_predecessor(old, tmp_path / "run-002", ["pending", "last"], protocol)


def test_continuation_rejects_changed_prior_source(tmp_path):
    old, protocol = predecessor_fixture(tmp_path)
    Path(next(iter(protocol["source_sha256"]))).write_text("tampered")
    with pytest.raises(AssertionError):
        capacity.capacity_predecessor(old, tmp_path / "run-002", ["pending", "last"], protocol)


def test_continuation_rejects_incomplete_predecessor(tmp_path):
    old, protocol = predecessor_fixture(tmp_path)
    (old / "complete/responses-n1-r1.jsonl").unlink()
    with pytest.raises(AssertionError):
        capacity.capacity_predecessor(old, tmp_path / "run-002", ["pending", "last"], protocol)
