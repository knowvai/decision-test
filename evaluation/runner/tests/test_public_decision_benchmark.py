import asyncio
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


path = Path(__file__).parents[1] / "scripts/public_decision_benchmark.py"
spec = importlib.util.spec_from_file_location("public_benchmark_under_test", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_timing_retains_p99_and_empty_sample_count():
    assert benchmark.timing([])["p99_ms"] is None
    assert benchmark.timing([100, 110, 120, 130, 1000])["mean_ms"] == 292
    assert benchmark.timing([100, 110, 120, 130, 1000])["p99_ms"] == pytest.approx(965.2)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8000", "http://8.8.8.8:8000",
                                "http://localhost:8000", "http://key@127.0.0.1:8000"])
def test_local_classifier_rejects_unapproved_endpoints(url):
    with pytest.raises((AssertionError, ValueError)):
        benchmark.local_endpoint(url)


def test_local_llm_uses_strict_schema_without_confidence_or_explanation():
    schema = benchmark.classification_schema(benchmark.DOMAIN_QUESTIONS)
    assert schema["required"] == ["domain"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["domain"]["enum"] == list(benchmark.DOMAINS)


@pytest.mark.parametrize("thinking", [False, True])
def test_local_langchain_wire_uses_json_schema_and_exactly_one_request(thinking):
    import httpx

    calls = []
    captured = []
    profile = {"enable_thinking": True, "reasoning_effort": "xhigh", "max_tokens": 4096} if thinking else None

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["temperature"] == 0
        assert body.get("max_tokens", body.get("max_completion_tokens")) == (4096 if thinking else 64)
        if thinking:
            assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
        else:
            assert "chat_template_kwargs" not in body
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "Qwen3-test", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"domain":"weather"}',
                "reasoning": "synthetic reasoning" if thinking else None}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}})

    async def run():
        async def observe(response):
            await benchmark.observe_classification_response(response, captured)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                     event_hooks={"response": [observe]} if thinking else None) as http:
            client = benchmark.local_llm_client("http://127.0.0.1:8000", "Qwen3-test", "test-only", http,
                                                benchmark.DOMAIN_QUESTIONS, 300, profile)
            active = {}
            result = await benchmark.llm_classify(client, {"utterance": "test"}, benchmark.DOMAIN_QUESTIONS, active, 300)
            assert result == "weather"
            assert active["decision_calls"][0]["result"]["usage"]["output_tokens"] == 8

    asyncio.run(run())
    assert len(calls) == 1
    if thinking:
        assert len(captured) == 1
        assert captured[0]["messages_sha256"] == benchmark.digest({"messages": calls[0]["messages"]})
        assert captured[0]["choices"][0]["reasoning_chars"] == len("synthetic reasoning")
        assert captured[0]["choices"][0]["finish_reason"] == "stop"
        assert "synthetic reasoning" not in json.dumps(captured)
    else:
        assert captured == []


def test_thinking_length_stop_is_failed_with_metadata_and_no_retry(tmp_path, monkeypatch):
    import httpx

    calls = []
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200)
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "Qwen-test", "choices": [{"index": 0, "finish_reason": "length" if len(calls) == 4 else "stop",
                "message": {"role": "assistant", "content": '{"domain":"weather"}', "reasoning_content": "test"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handle))
    args = SimpleNamespace(base_url="http://127.0.0.1:8000", timeout=300, startup_timeout=2400,
        output=tmp_path, llm_profile={"enable_thinking": True, "reasoning_effort": "xhigh", "max_tokens": 4096})
    jobs = [{"case_id": "test", "task": "classification", "state": {"utterance": "hello"}}]
    asyncio.run(benchmark.local_classification_model(args, "llm", "Qwen-test", "test-only", jobs,
                                                      benchmark.DOMAIN_QUESTIONS, jobs, tmp_path))
    rows = benchmark.read_jsonl(tmp_path / "responses.private.jsonl")
    assert len(calls) == 4 and len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["transport"][0]["choices"][0]["finish_reason"] == "length"
    assert rows[0]["transport"][0]["choices"][0]["reasoning_chars"] == 4
    assert len(json.loads((tmp_path / "warmup.private.json").read_text())) == 3


def test_local_classification_is_serial_records_failure_and_times_wrong_answers(tmp_path):
    jobs = [{"case_id": f"case:{i}", "task": "classification", "state": {"utterance": str(i)}} for i in range(3)]
    active, peak = 0, 0

    async def infer(state):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0.001)
        active -= 1
        if state["utterance"] == "1":
            raise ValueError("invalid output")
        return {"prediction": "weather"}

    output = tmp_path / "records.jsonl"
    asyncio.run(benchmark.local_classification_records(jobs, infer, output, "test", tmp_path / "STOP", 1))
    rows = benchmark.read_jsonl(output)
    assert peak == 1 and len(rows) == 3
    labels = [{"case_id": job["case_id"], "task": "classification", "gold": "news"} for job in jobs]
    result = benchmark.local_classification_summary(jobs, labels, rows)
    assert result["accuracy"] == 0
    assert result["valid_output_rate"] == 2 / 3
    assert result["successful_task_latency"]["n"] == 2
    assert result["all_attempt_latency"]["n"] == 3
    assert result["failures"] == {"ValueError": 1}


def test_local_classification_timeout_is_explicit_without_retry(tmp_path):
    called = 0

    async def infer(state):
        nonlocal called
        called += 1
        await asyncio.sleep(1)

    output = tmp_path / "records.jsonl"
    jobs = [{"case_id": "test", "task": "classification", "state": {"utterance": "hello"}}]
    asyncio.run(benchmark.local_classification_records(jobs, infer, output, "test", tmp_path / "STOP", 0.001))
    assert called == 1
    assert benchmark.read_jsonl(output)[0]["error_type"] == "TimeoutError"


def test_classification_continuation_excludes_only_pending_models(tmp_path):
    benchmark.write_json(tmp_path / "protocol.private.json", {"models": ["baseline", "omit", "next"], "records": 2974})
    benchmark.write_json(tmp_path / "status.json", {"state": "failed", "active": "baseline", "completed": ["baseline"]})
    (tmp_path / "baseline").mkdir()
    benchmark.write_json(tmp_path / "baseline/summary.json", {"records": 2974})
    (tmp_path / "STOP").symlink_to(tmp_path / "baseline/summary.json")
    assert benchmark.classification_predecessor(tmp_path, ["next"])["excluded"] == ["omit"]
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(tmp_path, ["baseline", "next"])
    benchmark.write_json(tmp_path / "status.json", {"state": "failed", "active": "omit", "completed": ["baseline"]})
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(tmp_path, ["next"])


@pytest.fixture
def failed_preflight(tmp_path):
    benchmark.write_json(tmp_path / "protocol.private.json", {"models": ["done", "pending", "later"], "records": 2974})
    benchmark.write_json(tmp_path / "status.json", {"state": "failed", "active": "pending",
                                                    "completed": ["done"], "error_type": "HTTPStatusError"})
    (tmp_path / "done").mkdir()
    (tmp_path / "pending").mkdir()
    benchmark.write_json(tmp_path / "done/summary.json", {"records": 2974})
    benchmark.write_json(tmp_path / "pending/deployment.json", {"model": "pending"})
    return tmp_path


def test_preflight_recovery_is_explicit_and_keeps_failed_predecessor(failed_preflight):
    status_before = (failed_preflight / "status.json").read_bytes()
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(failed_preflight, ["pending", "later"])
    result = benchmark.classification_predecessor(failed_preflight, ["pending", "later"], True)
    assert result["completed"] == ["done"] and result["excluded"] == []
    assert result["unstarted_recovery"] == {"model": "pending", "error_type": "HTTPStatusError", "formal_attempts": 0}
    assert (failed_preflight / "status.json").read_bytes() == status_before


@pytest.mark.parametrize("name", ["responses.private.jsonl", "summary.json"])
def test_preflight_recovery_rejects_even_empty_formal_artifacts(failed_preflight, name):
    (failed_preflight / "pending" / name).touch()
    with pytest.raises(AssertionError, match="Formal attempts"):
        benchmark.classification_predecessor(failed_preflight, ["pending", "later"], True)


@pytest.mark.parametrize("models", [["later"], ["later", "pending"], ["done", "pending"]])
def test_preflight_recovery_does_not_skip_active_or_repeat_completed(failed_preflight, models):
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(failed_preflight, models, True)


def test_preflight_recovery_requires_failed_deployment_evidence(failed_preflight):
    (failed_preflight / "pending/deployment.json").unlink()
    with pytest.raises(AssertionError, match="deployment evidence"):
        benchmark.classification_predecessor(failed_preflight, ["pending", "later"], True)


@pytest.fixture
def completed_classification_chain(tmp_path):
    parent = None
    for run, model in (("run-1", "first"), ("run-2", "second")):
        root = tmp_path / run
        (root / model).mkdir(parents=True)
        protocol = {"models": [model], "records": 1, "parent_dataset_protocol": "dataset",
                    "inputs_sha256": "inputs", "labels_sha256": "labels",
                    "domain_questions": benchmark.DOMAIN_QUESTIONS, "predecessor": parent}
        benchmark.write_json(root / "protocol.private.json", protocol)
        benchmark.write_json(root / "status.json", {"state": "complete", "active": None, "completed": [model]})
        benchmark.write_json(root / model / "summary.json", {"records": 1})
        benchmark.write_json(root / model / "deployment.json", {"model": model})
        (root / model / "responses.private.jsonl").write_text('{}\n')
        parent = {"directory": str(root), "protocol_sha256": benchmark.digest(protocol), "completed": [model]}
    return tmp_path / "run-2"


def test_roster_extension_requires_explicit_opt_in_and_retains_chain(completed_classification_chain):
    root = completed_classification_chain
    before = {str(p): p.read_bytes() for p in root.parent.rglob('*') if p.is_file()}
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(root, ["new"])
    result = benchmark.classification_predecessor(root, ["new"], extend_roster=True)
    assert result["roster_extension"]["completed_models"] == ["first", "second"]
    assert result["excluded"] == []
    assert before == {str(p): p.read_bytes() for p in root.parent.rglob('*') if p.is_file()}


@pytest.mark.parametrize("models", [["first"], ["second"], ["new", "first"]])
def test_roster_extension_cannot_repeat_any_ancestor(completed_classification_chain, models):
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(completed_classification_chain, models, extend_roster=True)


@pytest.mark.parametrize("mutation", ["hash", "completion", "dataset", "cycle", "missing_evidence", "incomplete", "recovery"])
def test_roster_extension_rejects_invalid_history(completed_classification_chain, mutation):
    root = completed_classification_chain
    prior = json.loads((root / "protocol.private.json").read_text())
    if mutation == "hash":
        prior["predecessor"]["protocol_sha256"] = "changed"
    elif mutation == "completion":
        prior["predecessor"]["completed"] = []
    elif mutation == "dataset":
        prior["inputs_sha256"] = "changed"
    elif mutation == "cycle":
        prior["predecessor"]["directory"] = str(root)
    elif mutation == "missing_evidence":
        (root.parent / "run-1/first/deployment.json").unlink()
    elif mutation == "incomplete":
        benchmark.write_json(root / "status.json", {"state": "failed", "active": "pending", "completed": ["second"]})
    benchmark.write_json(root / "protocol.private.json", prior)
    with pytest.raises(AssertionError):
        benchmark.classification_predecessor(root, ["new"], resume_unstarted=mutation == "recovery", extend_roster=True)


def test_massive_selection_is_balanced_deterministic_and_label_free():
    rows = [dict(id=str(i * 10 + j), scenario=i, intent=i, locale="zh-CN", partition="test", utt=f"request {i} {j}")
            for i in range(18) for j in range(4)]
    inputs, labels = benchmark.prepare_massive(rows, 2)
    assert (inputs, labels) == benchmark.prepare_massive(list(reversed(rows)), 2)
    assert len(inputs) == 36
    assert all(sum(r["gold"] == c for r in labels) == 2 for c in benchmark.DOMAINS)
    assert all(set(r) == {"case_id", "task", "state"} for r in inputs)
    assert all(set(r["state"]) == {"utterance"} for r in inputs)


def test_t2_keeps_whole_pools_and_rejects_ambiguous_or_missing_class():
    rows = [dict(query="q", positive=["a", "b"], negative=["c", "d"]),
            dict(query="ambiguous", positive=["x"], negative=["x"]),
            dict(query="empty", positive=["x"], negative=[])]
    inputs, labels, eligible = benchmark.prepare_t2(rows, 1)
    assert eligible == 1 and len(inputs[0]["candidates"]) == 4
    assert {c["content"] for c in inputs[0]["candidates"]} == {"a", "b", "c", "d"}
    swapped = [dict(query="q", positive=["c", "d"], negative=["a", "b"])]
    # Inference ordering and IDs must not encode which class a passage belongs to.
    assert benchmark.prepare_t2(swapped, 1)[0] == inputs
    assert len(labels[0]["positive_ids"]) == 2


def test_full_massive_keeps_natural_distribution_and_excludes_no_case():
    rows = [dict(id=str(i * 100 + j), scenario=i, intent=i, locale="zh-CN", partition="test", utt=f"request {i} {j}")
            for i in range(18) for j in range(i + 1)]
    inputs, labels = benchmark.prepare_massive(rows, None)
    assert len(inputs) == len(labels) == len(rows)
    assert (inputs, labels) == benchmark.prepare_massive(list(reversed(rows)), None)
    assert all(sum(r["gold"] == c for r in labels) == i + 1 for i, c in enumerate(benchmark.DOMAINS))


def test_t2_excludes_historical_pool_before_selection_and_keeps_full_text():
    rows = [dict(query=f"q{i}", positive=["positive " * 300], negative=["negative " * 300]) for i in range(5)]
    inputs, _, eligible = benchmark.prepare_t2(rows, 3, {"t2:0", "t2:1"})
    assert eligible == len(inputs) == 3
    assert {r["case_id"] for r in inputs} == {"t2:2", "t2:3", "t2:4"}
    assert all(len(r["candidates"]) == 2 for r in inputs)
    assert all(len(c["content"]) > 2000 for r in inputs for c in r["candidates"])


def test_full_pool_client_does_not_send_gold_and_rejects_partial_result():
    class Client:
        async def post(self, url, json):
            assert url == "http://10.0.0.1:8082/v1/rerank"
            assert set(json) == {"model", "query", "documents", "batch_size", "questions"}
            assert json["documents"] == ["full passage one", "full passage two"]
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"model": "test", "results": [{"index": 0, "relevance_score": 0.5}], "execution": {}})

    data = dict(endpoint="http://10.0.0.1:8082/v1/systemone", model="test", batch_size=4,
                ranker_kind="decision", score_questions={"relevance": {"type": "score", "criteria": ["low", "high"]}})
    job = dict(query="q", candidates=[dict(content="full passage one"), dict(content="full passage two")])
    active = {}
    with pytest.raises(AssertionError):
        asyncio.run(benchmark.rerank_pool(Client(), data, job, active))
    assert len(active["decision_calls"]) == 1 and active["decision_calls"][0]["status"] == "failed"


def test_inference_jobs_reject_labels_and_exclude_warmup_from_formal():
    inputs = [{"case_id": "a", "task": "classification", "state": {"utterance": "hello"}},
              {"case_id": "b", "task": "reranking", "query": "q", "candidates": [
                  {"fragment_id": "x", "content": "one"}, {"fragment_id": "y", "content": "two"}]}]
    assert len(benchmark.inference_jobs(inputs, 3)) == 6
    warmup = benchmark.inference_jobs(inputs, 3, True)
    assert len(warmup) == 2 and len(warmup[1]["candidates"]) == 1
    with pytest.raises(AssertionError):
        benchmark.inference_jobs([{**inputs[0], "gold": "general"}], 1)


@pytest.mark.parametrize("indexes", [[1, 0], [0, 0], [0], [False, 1]])
def test_vllm_restores_input_order_and_rejects_missing_or_duplicate_scores(indexes):
    class Client:
        async def post(self, url, json):
            assert url.endswith("/v1/rerank")
            assert "batch_size" not in json and "questions" not in json
            assert json["top_n"] == 2 and json["truncate_prompt_tokens"] is None
            assert json["max_tokens_per_query"] == json["max_tokens_per_doc"] == 0
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {
                "model": "test", "results": [{"index": i, "relevance_score": i * 0.8} for i in indexes]})

    data = dict(endpoint="http://10.0.0.1:8086/v1/systemone", model="test", batch_size=8,
                ranker_kind="qwen", rerank_transport="vllm")
    job = dict(query="query", candidates=[dict(content="one"), dict(content="two")])
    active = {}
    if indexes == [1, 0] and type(indexes[0]) is int:
        assert asyncio.run(benchmark.rerank_pool(Client(), data, job, active)) == [0, 0.8]
        assert active["decision_calls"][0]["execution"]["forward_calls"] is None
    else:
        with pytest.raises(AssertionError):
            asyncio.run(benchmark.rerank_pool(Client(), data, job, active))
        assert active["decision_calls"][0]["status"] == "failed"
    assert len(active["decision_calls"]) == 1


def test_repeat_agreement_is_unmeasured_for_single_pass_and_counts_failures():
    rows = [{"case_id": "a", "prediction": "weather", "status": "valid"}]
    assert benchmark.repeat_agreement(rows, ["a"], 1) is None
    assert benchmark.repeat_agreement(rows * 2, ["a"], 2) == 1
    assert benchmark.repeat_agreement(rows + [{**rows[0], "status": "failed"}], ["a"], 2) == 0


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_pool_is_bounded_independent_and_drained_before_return(fail):
    class Ranker:
        running = 0
        peak = 0
        finished = 0

        async def rerank(self, query, candidates):
            assert query == "q" and len(candidates) == 1
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.001)
            self.running -= 1
            self.finished += 1
            if fail and candidates[0].content == "0":
                raise ValueError("One candidate failed")
            return [int(candidates[0].content) / 10]

    ranker = Ranker()
    job = {"query": "q", "candidates": [{"content": str(i)} for i in range(5)]}
    if fail:
        with pytest.raises(ValueError):
            asyncio.run(benchmark.concurrent_rerank(ranker, {"batch_size": 2}, job))
    else:
        assert asyncio.run(benchmark.concurrent_rerank(ranker, {"batch_size": 2}, job)) == [0, 0.1, 0.2, 0.3, 0.4]
    assert ranker.finished == 5 and ranker.running == 0 and ranker.peak == 2


def test_concurrent_pool_full_pool_matches_candidate_count():
    class Ranker:
        running = 0
        peak = 0

        async def rerank(self, query, candidates):
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.001)
            self.running -= 1
            return [int(candidates[0].content)]

    ranker = Ranker()
    job = {"query": "q", "candidates": [{"content": str(i)} for i in range(5)]}
    assert asyncio.run(benchmark.concurrent_rerank(ranker, {"batch_size": 2, "pool_concurrency": "full_pool"}, job)) == [0, 1, 2, 3, 4]
    assert ranker.peak == 5


def test_failed_classifications_count_wrong():
    labels = {c: c for c in benchmark.DOMAINS}
    perfect = benchmark.classification_metrics(labels, labels)
    assert perfect["accuracy"] == perfect["macro_f1"] == 1
    failed = benchmark.classification_metrics(labels, {})
    assert failed["accuracy"] == failed["macro_f1"] == 0
    assert all(failed["confusion"][c]["failed"] == 1 for c in labels)


def test_existing_binary_retrieval_metrics_include_failed_pools_as_zero():
    labels = {"a": {"positive_ids": ["p"]}, "b": {"positive_ids": ["p"]}}
    records = {"a": {"status": "valid", "scores": [
        {"fragment_id": "p", "score": 1.0}, {"fragment_id": "n", "score": 0.0}]},
        "b": {"status": "failed", "scores": [{"fragment_id": "p", "score": 1.0}]}}
    metrics = benchmark.rerank_metrics(labels, records)
    assert metrics["ndcg_at_k"]["5"] == metrics["mrr"] == 0.5
    assert metrics["case_count"] == 2


@pytest.mark.parametrize("parsed", [None, {"domain": "not-a-domain"}, {"domain": "qa", "extra": True}])
def test_llm_baseline_rejects_illegal_output_without_retry(parsed):
    class Client:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            assert json.loads(messages[1]["content"]) == {"utterance": "test query"}
            assert json.loads(messages[0]["content"].split("\n", 1)[1])["criteria"] == benchmark.DOMAINS
            return {"parsed": parsed, "raw": SimpleNamespace(usage_metadata={}, response_metadata={})}

    client, active = Client(), {}
    with pytest.raises(AssertionError):
        asyncio.run(benchmark.llm_classify(client, {"utterance": "test query"}, benchmark.DOMAIN_QUESTIONS, active, 15))
    assert client.calls == 1 and active["decision_calls"][0]["status"] == "failed"


def test_llm_baseline_records_model_and_usage_without_inventing_server_latency():
    class Client:
        async def ainvoke(self, messages):
            return {"parsed": {"domain": "weather"}, "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 12, "output_tokens": 3}, response_metadata={"model_name": "pinned-report"})}

    active = {}
    result = asyncio.run(benchmark.llm_classify(Client(), {"utterance": "weather query"}, benchmark.DOMAIN_QUESTIONS, active, 15))
    assert result == "weather"
    assert active["decision_calls"][0]["result"]["latency_ms"] is None
    assert active["decision_calls"][0]["result"]["reported_model"] == "pinned-report"
    assert active["decision_calls"][0]["result"]["usage"]["input_tokens"] == 12


def test_llm_baseline_timeout_is_one_failed_attempt():
    class Client:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            await asyncio.sleep(1)

    client, active = Client(), {}
    with pytest.raises(TimeoutError):
        asyncio.run(benchmark.llm_classify(client, {"utterance": "query"}, benchmark.DOMAIN_QUESTIONS, active, 0.001))
    assert client.calls == 1 and active["decision_calls"][0]["status"] == "failed"


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
def test_llm_baseline_uses_authoritative_request_method_and_model_factory(method):
    expected = {"understanding": {"name": "current-understanding"}, "generation": None}
    rag = SimpleNamespace(structured_output_method=method)
    observed = {}

    class Client:
        model_name = "current-understanding"
        root_client = SimpleNamespace(max_retries=0)
        root_async_client = SimpleNamespace(max_retries=0)

        def with_structured_output(self, schema, **kwargs):
            observed.update(kwargs)
            assert schema["parameters"]["properties"]["domain"]["enum"] == list(benchmark.DOMAINS)
            return "structured-client"

    client = Client()

    async def resolve(tenant, request):
        assert tenant == "test-tenant" and request.temperature == 0 and request.max_tokens == 64
        return "resolved-request"

    def build(resolved):
        assert resolved == "resolved-request"
        return rag

    def create(request):
        assert request is rag
        return client

    def request_config(request, config):
        assert request is rag and config == "deployed-config"
        return {"structured_output_method": method}

    modules = {
        "src.api.main": SimpleNamespace(SendMessageRequest=SimpleNamespace(model_validate=lambda v: SimpleNamespace(**v)),
                                        _resolve_platform_conversation_model=resolve, _build_conversation_rag_request=build),
        "src.agents.utils.service_utils": SimpleNamespace(ServiceUtils=SimpleNamespace(create_understanding_llm_service=create)),
        "src.agents.utils.config_utils": SimpleNamespace(ConfigUtils=SimpleNamespace(get_request_config=request_config)),
        "src.services.platform_config": SimpleNamespace(get_platform_configuration_client=lambda config:
            SimpleNamespace(resolve_conversation_models=lambda *args, **kwargs: expected)),
        "src.config.settings": SimpleNamespace(DecomposeRAGConfig=lambda: "deployed-config"),
    }
    data = {"llm_baseline": {"tenant_id": "test-tenant", "expected_models": expected},
            "baseline_definitions": "def bundle_info(bundle):\n    return bundle\n",
            "domain_questions": benchmark.DOMAIN_QUESTIONS, "timeout_seconds": 15}
    with patch.dict(sys.modules, modules):
        structured, binding, before = asyncio.run(benchmark.prepare_llm_baseline(data))
        assert asyncio.run(binding()) == expected
    assert structured == "structured-client" and before == expected
    assert observed == {"method": method, "include_raw": True}
    assert (client.temperature, client.max_tokens, client.max_retries, client.request_timeout) == (0, 64, 0, 15)
    assert data["llm_execution"]["structured_output_method"] == method
    assert all(value == 0 for value in data["llm_execution"]["sdk_max_retries"].values())


def test_bootstrap_clusters_repeats_by_query_and_includes_failed_pool():
    labels = {"a": {"positive_ids": ["p"]}, "b": {"positive_ids": ["p"]}}
    records = [{"case_id": case, "repeat": repeat, "status": "valid" if case == "a" else "failed", "elapsed_ms": 10,
                "scores": [{"fragment_id": "p", "score": 1}, {"fragment_id": "n", "score": 0}]}
               for case in labels for repeat in [1, 2, 3]]
    values = benchmark.rank_case_values(labels, records, 3)
    interval = benchmark.bootstrap_mean([v["ndcg5"] for v in values.values()], samples=30)
    assert interval["independent_cases"] == 2 and interval["point"] == 0.5
    assert benchmark.bootstrap_mean([0.5, 0.5], samples=30)["low"] == 0.5
    assert interval == benchmark.bootstrap_mean([v["ndcg5"] for v in values.values()], samples=30)


def test_paired_classification_bootstrap_is_zero_for_identical_models():
    labels = {str(i): domain for i, domain in enumerate(benchmark.DOMAINS)}
    predictions = {r: labels for r in [1, 2, 3]}
    result = benchmark.classification_interval(labels, predictions, predictions, samples=30)
    assert result["independent_cases"] == 18
    assert all(v == 0 for row in result["metrics"].values() for v in row.values())


@pytest.mark.parametrize("mixed_labels", [False, True])
def test_classification_bootstrap_keeps_duplicate_inputs_in_one_cluster(monkeypatch, mixed_labels):
    from collections import Counter

    labels = {"a": "weather", "b": "music" if mixed_labels else "weather", "c": "weather"}
    predictions = {r: {"a": "weather", "b": "weather", "c": "music"} for r in [1, 2, 3]}
    groups = {"a": "same-input", "b": "same-input", "c": "different-input"}
    weights = []

    def observed_counter(values):
        result = Counter(values)
        weights.append(result)
        return result

    monkeypatch.setattr(benchmark, "Counter", observed_counter)
    result = benchmark.classification_interval(labels, predictions, samples=30, case_groups=groups)
    assert result["independent_cases"] == 2 and result["source_cases"] == 3
    assert len(weights) == 30 and all(w["a"] == w["b"] for w in weights)
    assert result["metrics"]["accuracy"]["point"] == (1 / 3 if mixed_labels else 2 / 3)


@dataclass
class Answer:
    answer_type: str
    choice: str = "general"
    confidence: float = 0.1
    noul: float | None = None
    probabilities: dict | None = None


@dataclass
class Result:
    answers: dict
    latency_ms: float = 1
    input_tokens: int = 10


def test_native_worker_uses_low_confidence_and_fails_partial_pool(tmp_path, capsys):
    class Error(RuntimeError):
        pass

    class Client:
        requests = []

        def __init__(self, **kwargs):
            pass

        async def decide(self, *, state, questions):
            self.requests.append(state)
            if state.get("passage") == "bad":
                raise Error("backend failed")
            qid, question = next(iter(questions.items()))
            keys = list(question["criteria"]) if question["type"] == "choice" else ["0", "1", "2", "3"]
            return Result({qid: Answer(question["type"], probabilities={k: 1 / len(keys) for k in keys})})

    definitions, score_questions = benchmark.reused_sources()
    source = tmp_path / "source.txt"
    source.write_text("unchanged")
    data = dict(endpoint="http://10.0.0.1:8082/v1/systemone", api_key="test", model="test", pacing_seconds=0,
                timeout_seconds=15, source_paths=[str(source)], reused_definitions=definitions,
                domain_questions=benchmark.DOMAIN_QUESTIONS, score_questions=score_questions,
                jobs=[dict(case_id="a", task="classification", repeat=1, state={"utterance": "hello"}),
                      dict(case_id="b", task="reranking", repeat=1, query="q", candidates=[
                          dict(fragment_id="p", content="ok"), dict(fragment_id="n", content="bad")])])
    module = SimpleNamespace(DecisionModelClient=Client, DecisionModelError=Error)
    with patch.dict(sys.modules, {"src.services.decision_model": module}):
        asyncio.run(benchmark.native_run(data))
    records = [json.loads(s.removeprefix("BENCHMARK_PUBLIC ")) for s in capsys.readouterr().out.splitlines()
               if s.startswith("BENCHMARK_PUBLIC ")]
    assert records[0]["status"] == "valid" and records[0]["prediction"] == "general"
    assert records[1]["status"] == "failed" and "scores" not in records[1]
    assert [r["status"] for r in records[1]["calls"]] == ["valid", "failed"]
    assert all(set(s) in [{"utterance"}, {"query", "passage"}] for s in Client.requests)


@pytest.mark.parametrize("stop_initially", [True, False])
def test_environment_stop_marker_prevents_new_native_requests(tmp_path, capsys, stop_initially):
    marker = tmp_path / "stop"
    if stop_initially:
        marker.touch()
    source = tmp_path / "source.txt"
    source.write_text("unchanged")
    definitions, questions = benchmark.reused_sources()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def decide(self, **kwargs):
            assert not marker.exists(), "No new inference after the environment stop marker"
            marker.touch()
            return Result({"domain": Answer("choice", probabilities={k: 1 / len(benchmark.DOMAINS)
                                                                       for k in benchmark.DOMAINS})})

    data = dict(endpoint="http://10.0.0.1:8082/v1/systemone", api_key="test", model="test", pacing_seconds=0,
                timeout_seconds=15, source_paths=[str(source)], reused_definitions=definitions,
                domain_questions=benchmark.DOMAIN_QUESTIONS, score_questions=questions, stop_file=str(marker),
                jobs=[dict(case_id=str(i), task="classification", repeat=1, state={"utterance": "hello"}) for i in range(2)])
    module = SimpleNamespace(DecisionModelClient=Client, DecisionModelError=RuntimeError)
    with patch.dict(sys.modules, {"src.services.decision_model": module}):
        asyncio.run(benchmark.native_run(data))
    output = capsys.readouterr().out.splitlines()
    assert sum(s.startswith("BENCHMARK_PUBLIC ") for s in output) == (0 if stop_initially else 1)
    manifest = json.loads(next(s.removeprefix("BENCHMARK_PUBLIC_MANIFEST ") for s in output
                               if s.startswith("BENCHMARK_PUBLIC_MANIFEST ")))
    assert manifest["cancelled"] and manifest["source_unchanged"]
