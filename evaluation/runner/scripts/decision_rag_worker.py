"""Isolated deployed-runtime evaluation worker; injected inputs stay outside Git.

Executed over SSH/stdin, never imported into the serving application. The caller
supplies an authorized snapshot and private endpoint configuration.
"""
import asyncio
from dataclasses import asdict
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import types
from urllib.parse import urlsplit


DATA = __EVALUATION_INPUT__
os.environ.update(LANGFUSE_ENABLED="false", LANGFUSE_TRACING_ENABLED="false", LANGCHAIN_TRACING_V2="false", CONVERSATION_CACHE_ENABLED="false")
logging.disable(logging.CRITICAL)
allowed_ips = set()
for host in DATA["approved_public_hosts"]:
    allowed_ips.update(info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM))
destinations, blocked = set(), set()
original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex


def permitted(address):
    if not isinstance(address, tuple):
        return
    host, port = address[:2]
    try:
        ips = [str(ipaddress.ip_address(host))]
    except ValueError:
        ips = [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
    if not ips or not all(ipaddress.ip_address(ip).is_private or ipaddress.ip_address(ip).is_loopback or ip in allowed_ips for ip in ips):
        blocked.add(str(host))
        raise RuntimeError("Unapproved evaluation egress")
    destinations.add((str(host), int(port)))


def connect(self, address):
    permitted(address)
    return original_connect(self, address)


def connect_ex(self, address):
    permitted(address)
    return original_connect_ex(self, address)


socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
from src.api.main import (SendMessageRequest, _resolve_platform_conversation_model,
                          _build_conversation_rag_request, _route_conversation_with_smart_router, agent_factory)
from src.models.schemas import Message
from src.services.decision_model import DecisionModelClient, DecisionModelError
from src.services.knowledge_retrieval.retrieval_core import KnowledgeRetrievalCore
from src.services.knowledge_retrieval.retrieval_contracts import FragmentEvidence
from src.services.knowledge_retrieval.runtime_adapters import build_knowledge_retrieval_core
from src.services.platform_config import get_platform_configuration_client
from src.config.settings import DecomposeRAGConfig
from src.agents.utils.service_utils import RerankerService

comparison = types.ModuleType("evaluation_comparison")
sys.modules[comparison.__name__] = comparison
exec(compile(__COMPARISON_SOURCE__, "evaluation_comparison.py", "exec"), comparison.__dict__)

active = {}
SCORE_QUESTIONS = DATA["score_questions"]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_decision(result, questions):
    if set(result.answers) != set(questions):
        raise DecisionModelError("Incomplete question coverage")
    for qid, q in questions.items():
        a = result.answers[qid]
        if a.answer_type != q["type"]:
            raise DecisionModelError("Wrong answer type")
        if q["type"] == "noul":
            if a.noul is None or not math.isfinite(a.noul) or not 0 <= a.noul <= 1:
                raise DecisionModelError("Invalid noul probability")
        else:
            keys = set(q["criteria"]) if q["type"] == "choice" else {str(i) for i in range(len(q["criteria"]))}
            if set(a.probabilities) != keys or not all(math.isfinite(p) and 0 <= p <= 1 for p in a.probabilities.values()) or abs(sum(a.probabilities.values()) - 1) > 0.005:
                raise DecisionModelError("Invalid probability coverage")
            if q["type"] == "choice" and (a.choice not in keys or a.confidence is None or not math.isfinite(a.confidence) or not 0 <= a.confidence <= 1):
                raise DecisionModelError("Invalid choice or confidence")


class RecordingDecisionClient(DecisionModelClient):
    async def decide(self, *, state, questions):
        # Match the serving client's connection lifecycle; isolate proxy rate pacing
        # from model/client time while including it in pool/whole-flow wall time.
        spacing = DATA.get("decision_pacing_seconds", 0.21)
        await asyncio.sleep(max(0, spacing - (time.perf_counter() - getattr(self, "_previous_started", 0))))
        self._previous_started = time.perf_counter()
        started = time.perf_counter()
        record = {"input_sha256": digest({"state": state, "questions": questions}), "question_ids": list(questions)}
        try:
            result = await super().decide(state=state, questions=questions)
            record["raw_result"] = asdict(result)
            validate_decision(result, questions)
            record.update(result=asdict(result), status="valid")
            return result
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error_message=str(exc)[:300], status="failed")
            response = getattr(exc.__cause__, "response", None)
            if response is not None:
                record["http_status"] = response.status_code
            raise
        finally:
            record["client_elapsed_ms"] = (time.perf_counter() - started) * 1000
            active.setdefault("decision_calls", []).append(record)


class PointwiseDecisionReranker:
    def __init__(self, client):
        self.client = client

    async def rerank(self, query, evidence):
        scores = []
        for item in evidence:
            result = await self.client.decide(state={"query": query, "passage": item.content}, questions=SCORE_QUESTIONS)
            a = result.answers["relevance"]
            scores.append(sum(int(k) * p for k, p in a.probabilities.items()) / 3)
        active.setdefault("raw_scores", []).append({"backend": "decision_pointwise", "fragment_ids": [i.fragment_id for i in evidence], "scores": scores})
        return scores


class ListwiseDecisionReranker:
    """Evaluate a candidate pool with native choice probabilities.

    Candidates are placed in the state and the choice options are stable IDs.
    This keeps the full passage text available to the model without relying on
    an option-description length limit (notably Laya's).  Pools that cannot fit
    one request are reduced by a deterministic top-3-per-group tournament;
    group boundaries depend only on input size, never on labels or answers.
    """

    def __init__(self, client, max_options=32, max_state_chars=6000):
        self.client = client
        self.max_options = int(max_options)
        self.max_state_chars = int(max_state_chars)

    @staticmethod
    def _question(candidate_ids):
        return {"relevance": {
            "type": "choice",
            "instructions": (
                "Which candidate passages most directly help answer the query? "
                "Choose the candidate ID with the strongest direct evidence. "
                "The probabilities are used to rank all candidates."
            ),
            "criteria": {candidate_id: candidate_id for candidate_id in candidate_ids},
        }}

    def _state_size(self, query, candidates):
        return len(json.dumps({
            "query": query,
            "candidates": [{"id": f"c{index}", "content": item.content}
                           for index, item in enumerate(candidates)],
        }, ensure_ascii=False, separators=(",", ":")))

    def _pack(self, query, candidates):
        groups, current = [], []
        for item in candidates:
            single = self._state_size(query, [item])
            if single > self.max_state_chars:
                raise DecisionModelError("candidate cannot fit the frozen listwise input budget")
            projected = self._state_size(query, current + [item])
            if current and (len(current) >= self.max_options or projected > self.max_state_chars):
                groups.append(current)
                current = []
                projected = single
            current.append(item)
        if current:
            groups.append(current)
        return groups

    async def _choose(self, query, candidates):
        keys = [f"c{index}" for index in range(len(candidates))]
        question = self._question(keys)
        result = await self.client.decide(
            state={"query": query, "candidates": [
                {"id": key, "content": item.content}
                for key, item in zip(keys, candidates)
            ]},
            questions=question,
        )
        answer = result.answers["relevance"]
        probabilities = {item.fragment_id: float(answer.probabilities[key])
                         for key, item in zip(keys, candidates)}
        return sorted(candidates, key=lambda item: (-probabilities[item.fragment_id], item.fragment_id)), probabilities

    @staticmethod
    def _can_split(exc):
        response = getattr(getattr(exc, "__cause__", None), "response", None)
        return ((response is not None and response.status_code in {400, 413, 422})
                or str(exc) == "Invalid probability coverage")

    async def _scored_groups(self, query, candidates):
        pending = list(self._pack(query, candidates))
        groups = []
        while pending:
            group = pending.pop(0)
            try:
                ranked, probabilities = await self._choose(query, group)
            except DecisionModelError as exc:
                if not self._can_split(exc) or len(group) < 2:
                    raise
                midpoint = len(group) // 2
                pending[0:0] = [group[:midpoint], group[midpoint:]]
                continue
            groups.append((ranked, probabilities))
        return groups

    async def _rank(self, query, candidates):
        scored_groups = await self._scored_groups(query, candidates)
        groups = [ranked for ranked, _ in scored_groups]
        if len(groups) == 1:
            return scored_groups[0]

        ranked_groups = []
        survivors = []
        for ranked, probabilities in scored_groups:
            ranked_groups.append((ranked, probabilities))
            survivors.extend(ranked[:3])

        if len(survivors) >= len(candidates):
            raise DecisionModelError("pool cannot be compared within the frozen listwise input budget")

        final_ranked, final_probabilities = await self._rank(query, survivors)
        final_ids = {item.fragment_id for item in final_ranked}
        remainder = []
        for group_index, (ranked, probabilities) in enumerate(ranked_groups):
            for local_rank, item in enumerate(ranked):
                if item.fragment_id not in final_ids:
                    remainder.append((item, -probabilities[item.fragment_id], group_index, local_rank))
        remainder.sort(key=lambda row: (row[1], row[2], row[3], row[0].fragment_id))
        return final_ranked + [row[0] for row in remainder], final_probabilities

    async def rerank(self, query, evidence):
        ranked, _ = await self._rank(query, evidence)
        # Scores are an ordering projection, not probabilities. Native
        # probabilities remain in the per-call audit records.
        position = {item.fragment_id: index for index, item in enumerate(ranked)}
        scores = [1.0 - position[item.fragment_id] / (len(ranked) + 1) for item in evidence]
        active.setdefault("raw_scores", []).append({
            "backend": "decision_listwise_choice",
            "fragment_ids": [item.fragment_id for item in evidence],
            "scores": scores,
        })
        return scores


class RecordingReranker:
    def __init__(self, delegate, stage):
        self.delegate, self.stage = delegate, stage

    async def rerank(self, query, evidence):
        started = time.perf_counter()
        try:
            scores = await self.delegate.rerank(query, evidence)
            active.setdefault("raw_scores", []).append({"backend": self.stage, "fragment_ids": [i.fragment_id for i in evidence], "scores": scores})
            return scores
        except Exception as exc:
            failure = {"stage": self.stage, "error_type": type(exc).__name__}
            if getattr(exc, "response", None) is not None:
                failure["http_status"] = exc.response.status_code
            active.setdefault("reranker_errors", []).append(failure)
            raise
        finally:
            active.setdefault("adapter_times", []).append({"stage": self.stage, "elapsed_ms": (time.perf_counter() - started) * 1000})


original_rerank = KnowledgeRetrievalCore._rerank_candidates
original_provider_rerank = RerankerService.rerank_with_scores


async def observed_provider_rerank(self, query, documents):
    record = {"document_count": len(documents), "top_k": self.top_k,
        "max_doc_chars": self.max_doc_chars, "truncated_document_count": sum(len(d) > self.max_doc_chars for d in documents)}
    try:
        rows = await original_provider_rerank(self, query, documents)
        record["returned_scores"] = len(rows)
        return rows
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        if getattr(exc, "response", None) is not None:
            record["http_status"] = exc.response.status_code
        raise
    finally:
        active.setdefault("original_provider_calls", []).append(record)


async def observed_rerank(self, *, query, candidates):
    identities = []
    for item in candidates:
        assert item.kb_id == DATA["kb_id"] and item.file_id in DATA["file_ids"], "Evidence scope drift"
        original = DATA["corpus_identities"].get(item.fragment_id)
        assert original and original["file_id"] == item.file_id and original["content_sha256"] == hashlib.sha256(item.content.strip().encode()).hexdigest(), "Evidence content drift"
        identities.append({"fragment_id": item.fragment_id, "content_sha256": original["content_sha256"], "base_score": item.score})
    started = time.perf_counter()
    old_primary, old_chat = self._reranker, self._chat_reranker
    if active["replace_reranker"]:
        self._reranker = PointwiseDecisionReranker(active["client"])
    if self._reranker is not None:
        self._reranker = RecordingReranker(self._reranker, "primary")
    if self._chat_reranker is not None:
        self._chat_reranker = RecordingReranker(self._chat_reranker, "chat_fallback")
    try:
        ranked = await original_rerank(self, query=query, candidates=candidates)
        active.setdefault("ranked_pools", []).append({"query": query, "input_candidates": identities,
            "input_sha256": digest(identities), "retained": [{"fragment_id": i.fragment_id, "score": i.score} for i in ranked],
            "threshold": self._rerank_threshold, "fallback_min": self._rerank_fallback_min,
            "elapsed_ms": (time.perf_counter() - started) * 1000})
        return ranked
    finally:
        self._reranker, self._chat_reranker = old_primary, old_chat


def telemetry(snapshot):
    info = snapshot.telemetry()
    info.pop("snapshot_age_seconds", None)
    url = urlsplit(snapshot.base_url)
    info.update(name=snapshot.name, endpoint_scheme=url.scheme, endpoint_host=url.hostname, endpoint_port=url.port, endpoint_path=url.path, embedding_dimension=snapshot.embedding_dimension)
    return info


def bundle_info(bundle):
    return {k: telemetry(getattr(bundle, k)) if getattr(bundle, k) is not None else None for k in ("generation", "understanding", "embedding", "rerank")}


async def run():
    agent = agent_factory.create_agent(use_checkpointer=False)
    agent.config.langfuse_enabled = False
    agent.config.conversation_cache_enabled = False
    config_client = get_platform_configuration_client(DecomposeRAGConfig())
    before = await asyncio.to_thread(config_client.resolve_conversation_models, DATA["tenant_id"], "model.binding.conversation.generation", None, use_rerank=True, force_refresh=True)
    models_before = bundle_info(before)
    assert models_before == DATA["expected_models"], "Original model binding drift"
    source_paths = DATA["source_paths"]
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    KnowledgeRetrievalCore._rerank_candidates = observed_rerank
    RerankerService.rerank_with_scores = observed_provider_rerank
    try:
        for job in DATA["jobs"]:
            active.clear()
            active.update(replace_reranker=job.get("replace_reranker", False))
            started = time.perf_counter()
            result = {k: v for k, v in job.items() if k not in {"case", "pool"}}
            case = job["case"]
            result["case_id"] = case["case_id"]
            try:
                candidate = job.get("candidate")
                if candidate:
                    active["client"] = RecordingDecisionClient(api_key=DATA["evaluation_api_key"], endpoint=candidate["endpoint"], model=candidate["model"], timeout_seconds=DATA["decision_timeout_seconds"])
                flags = {"decision_model_enabled": bool(job.get("replace_router")),
                    "decision_model_router_enabled": bool(job.get("replace_router")),
                    "decision_model_shadow_mode": False, "decision_model_min_confidence": 0.8}
                request = SendMessageRequest.model_validate({"message": case["query"], "use_cache": False,
                    "filters": {"all_of": [{"any_of": [{"tenant_id": DATA["tenant_id"]}]}, {"any_of": [{"kb_id": DATA["kb_id"]}]}, {"any_of": [{"file_id": f} for f in DATA["file_ids"]]}]},
                    "cache_scope": {"tenant_id": DATA["tenant_id"], "kb_ids": [DATA["kb_id"]], "file_ids": DATA["file_ids"]},
                    "request_config": flags, "retrieval_setting": {"include_debug_metadata": True}})
                t0 = time.perf_counter()
                resolved = await _resolve_platform_conversation_model(DATA["tenant_id"], request)
                rag = _build_conversation_rag_request(resolved)
                rag.messages = [Message.model_validate(m) for m in case["conversation_history"]] + [Message(role="user", content=case["query"])]
                result["resolve_ms"] = (time.perf_counter() - t0) * 1000
                timing = {}
                extra = {"timing": timing, "conversation_cache_enabled": False}
                if job.get("replace_router"):
                    extra["decision_model_client"] = active["client"]
                if job["task"] == "rank":
                    _, llm, embedding = agent._create_services(rag)
                    config = agent._get_request_config(rag)
                    reranker = agent._build_core_reranker(rag, request_config=config, retrieval_limit=5)
                    core = build_knowledge_retrieval_core(opensearch_client=agent._require_shared_opensearch_client(), embedding_runtime=embedding, reranker_service=reranker, chat_model=llm if reranker is not None else None, request_config=config)
                    candidates = [FragmentEvidence.model_validate({k: v for k, v in c.items() if k in FragmentEvidence.model_fields}) for c in job["pool"]["retrieved_items"]]
                    await core._rerank_candidates(query=job["pool"]["query"], candidates=candidates)
                    result["status"] = "ranked"
                else:
                    t0 = time.perf_counter()
                    routing = await _route_conversation_with_smart_router(agent, rag, trace_id="isolated-decision-evaluation", tenant_id=DATA["tenant_id"], extra_configurable=extra, conversation_messages=case["conversation_history"])
                    result.update(route_ms=(time.perf_counter() - t0) * 1000, routing=routing, timing=timing)
                    result["router_fast_path"] = routing.get("routing_reason") == "decision-model 高置信纯路由判断。"
                    result["status"] = "routed"
                    if job["task"] == "flow":
                        if routing.get("execution_mode") == "agent":
                            result["status"] = "outside_runtime_agent_requires_tool_execution"
                        elif routing.get("execution_mode") == "needs_confirmation":
                            result["status"] = "outside_runtime_confirmation_delivery_route_only"
                        else:
                            extra["routing_decision"] = routing
                            turn = comparison.TurnResult(endpoint=job["experiment"], question=case["query"], status_code=200)
                            pending, events = "", []
                            async def consume():
                                nonlocal pending
                                async for chunk in agent.process_query_stream(rag, trace_id="isolated-decision-evaluation", tenant_id=DATA["tenant_id"], extra_configurable=extra, use_all_messages=True):
                                    pending += chunk
                                    parsed, pending = comparison._decode_sse_data(pending)
                                    for event in parsed:
                                        events.append(event)
                                        comparison._apply_event(turn, event)
                                        if turn.first_event_ms is None:
                                            turn.first_event_ms = (time.perf_counter() - started) * 1000
                                        if turn.first_content_ms is None and any(turn._content_parts):
                                            turn.first_content_ms = (time.perf_counter() - started) * 1000
                            await asyncio.wait_for(consume(), timeout=DATA["flow_timeout_seconds"])
                            turn.elapsed_ms = (time.perf_counter() - started) * 1000
                            comparison._finalize_result(turn)
                            result.update(turn=asdict(turn), events=events, pending_sse_bytes=len(pending.encode()), status="completed" if turn.terminal_status in {"completed", "stop"} and turn.answer_chars and not turn.error else "failed_stream")
            except Exception as exc:
                result.update(status="error", error_type=type(exc).__name__, error_locations=[f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_tb(exc.__traceback__)])
            result.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                diagnostics={k: v for k, v in active.items() if k not in {"client"}}, labels_sent=False)
            print("BENCHMARK_EVAL " + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        KnowledgeRetrievalCore._rerank_candidates = original_rerank
        RerankerService.rerank_with_scores = original_provider_rerank
        after = await asyncio.to_thread(config_client.resolve_conversation_models, DATA["tenant_id"], "model.binding.conversation.generation", None, use_rerank=True, force_refresh=True)
        await agent_factory.close_search_backends()
        print("BENCHMARK_EVAL_MANIFEST " + json.dumps({"models_before": models_before, "models_after": bundle_info(after),
            "bindings_unchanged": models_before == bundle_info(after), "source_sha256": source_hashes,
            "source_unchanged": all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in source_hashes.items()),
            "network_destinations": sorted(destinations), "blocked_destinations": sorted(blocked),
            "checkpointer": False, "cache_enabled": False, "scope": "isolated_deployed_routing_to_answer_runtime_not_durable_turn_or_frontend"}), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
