"""Bounded evaluation HTTP transport for official text decision runtimes."""

from __future__ import annotations

import hmac
import json
import os
import queue
import sys
import threading
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr
    state: StrictStr | dict | list
    questions: dict[str, dict] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_questions(self):
        for qid, question in self.questions.items():
            if not qid or len(qid) > 128:
                raise ValueError("question IDs must be nonempty and at most 128 characters")
            if set(question) - {"type", "instructions", "criteria"}:
                raise ValueError("unsupported question field")
            if not isinstance(question.get("instructions", ""), str):
                raise ValueError("instructions must be text")
            kind = question.get("type")
            criteria = question.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError("choice needs 2 to 255 named criteria")
                if any(not key or not isinstance(value, str) for key, value in criteria.items()):
                    raise ValueError("choice criteria must have nonempty IDs and text descriptions")
            elif kind == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 32:
                    raise ValueError("score needs 2 to 32 ordered criteria")
                if any(not isinstance(value, str) for value in criteria):
                    raise ValueError("score criteria must be text")
            elif kind == "noul":
                if criteria is not None:
                    raise ValueError("noul does not take criteria")
            else:
                raise ValueError("type must be choice, noul or score")
        return self


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: StrictStr
    query: StrictStr = Field(min_length=1)
    documents: list[StrictStr] = Field(min_length=1, max_length=128)
    batch_size: Literal[1, 4, 8] = 8
    questions: dict[str, dict] | None = None

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value):
        if type(value) is not int:
            raise ValueError("Batch size must be an integer")
        return value

    @model_validator(mode="after")
    def validate_text(self):
        if not self.query.strip() or any(not doc.strip() for doc in self.documents):
            raise ValueError("Query and documents must be nonempty text")
        if self.questions is not None:
            DecisionRequest(model=self.model, state={}, questions=self.questions)
            if len(self.questions) != 1 or self.questions.get("relevance", {}).get("type") not in {"score", "choice"}:
                raise ValueError("Decision reranking requires one relevance score or choice question")
            if self.questions["relevance"]["type"] == "choice" and "relevant" not in self.questions["relevance"]["criteria"]:
                raise ValueError("Choice reranking requires a 'relevant' criterion")
        return self


class BatchQueueFull(RuntimeError):
    """The bounded SystemOne queue cannot accept another request."""


@dataclass
class _BatchItem:
    body: dict
    key: str
    enqueued_at: float
    completed: threading.Event
    result: dict | None = None
    error: BaseException | None = None
    expired: bool = False


def _batch_key(body):
    return json.dumps(
        {"model": body["model"], "questions": body["questions"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class SystemOneBatcher:
    """Coalesce compatible HTTP requests into one native runtime invocation."""

    def __init__(self, operation: Callable[[list[dict]], list[dict]], max_batch_size, wait_ms, queue_size,
                 native_batching=True, timeout_seconds=300):
        self.operation = operation
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_ms / 1000
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="systemone-batcher", daemon=True)
        self.pending = None
        self.native_batching = native_batching
        self.timeout_seconds = timeout_seconds

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.pending is not None:
            self.pending.error = RuntimeError("SystemOne batcher stopped")
            self.pending.completed.set()
            self.pending = None
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            item.error = RuntimeError("SystemOne batcher stopped")
            item.completed.set()

    def submit(self, body):
        if self.stop_event.is_set():
            raise RuntimeError("SystemOne batcher stopped")
        item = _BatchItem(body, _batch_key(body), time.perf_counter(), threading.Event())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            raise BatchQueueFull("SystemOne batch queue is full") from None
        if not item.completed.wait(timeout=self.timeout_seconds):
            item.expired = True
            raise TimeoutError("SystemOne request deadline exceeded")
        if item.error is not None:
            raise item.error
        return item.result

    def _next_item(self):
        if self.pending is not None:
            item, self.pending = self.pending, None
            return item
        while not self.stop_event.is_set():
            try:
                return self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _run(self):
        while not self.stop_event.is_set():
            first = self._next_item()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item.key == first.key:
                    batch.append(item)
                else:
                    self.pending = item
                    break
            started = time.perf_counter()
            batch = [item for item in batch if not item.expired]
            if not batch:
                continue
            try:
                results = self.operation([item.body for item in batch])
                if not isinstance(results, list) or len(results) != len(batch):
                    raise RuntimeError("Native SystemOne batch returned an invalid result count")
            except BaseException as exc:
                for item in batch:
                    item.error = exc
                    item.completed.set()
                continue
            finished = time.perf_counter()
            for item, result in zip(batch, results):
                if not isinstance(result, dict):
                    item.error = RuntimeError("Native SystemOne batch returned an invalid result")
                else:
                    result = dict(result)
                    execution = dict(result.get("execution") or {})
                    execution["systemone_batch"] = {
                        "batch_size": len(batch),
                        "max_batch_size": self.max_batch_size,
                        "queue_wait_ms": round((started - item.enqueued_at) * 1000, 3),
                        "coalescing_wait_ms": round((started - first.enqueued_at) * 1000, 3),
                        "native_batch_calls": int(self.native_batching),
                        "cross_request_batching": self.native_batching,
                        "runtime_calls": 1,
                    }
                    result["execution"] = execution
                    item.result = result
                item.completed.set()


def predict_d3_batch(runtime, bodies, max_length):
    from d3_runtime import product_answer

    prepared = []
    for body in bodies:
        item = runtime.prepare(body["state"], body["questions"])
        if item.errors:
            raise ValueError("Official D3 encoder rejected the record")
        for question_id in body["questions"]:
            if question_id not in item.sequences or question_id not in item.questions:
                raise ValueError("Official D3 encoder omitted a question")
            if len(item.sequences[question_id]) > max_length:
                raise ValueError("Input exceeds the evaluation token limit")
        prepared.append(item)

    sequences, option_counts, references = [], [], []
    for item in prepared:
        for question_id in item.questions:
            sequences.append(item.sequences[question_id])
            option_counts.append(len(item.questions[question_id].keys))
            references.append((item, question_id))
    probabilities = runtime.probabilities(sequences, option_counts)
    if len(probabilities) != len(references):
        raise RuntimeError("Incomplete official SystemOne batch")

    answers_by_body = [dict() for _ in bodies]
    probability_index = 0
    for answer_index, item in enumerate(prepared):
        for question_id in item.questions:
            answers_by_body[answer_index][question_id] = product_answer(
                item.questions[question_id], probabilities[probability_index]
            )
            probability_index += 1
    tokens = [sum(len(item.sequences[qid]) for qid in item.questions) for item in prepared]
    return [
        {"model": body["model"], "answers": answers, "usage": {"input_tokens": token_count, "output_tokens": 0}}
        for body, answers, token_count in zip(bodies, answers_by_body, tokens)
    ]


def predict_clef_batch(model, processor, bodies, max_length):
    from joint_schema_model import collate_records, encode_record, systemone_answer

    records = []
    tokens = []
    for body in bodies:
        record = encode_record(processor.tokenizer, body, max_length=max_length + 1, processor=processor)
        if len(record.input_ids) > max_length:
            raise ValueError("Input exceeds the evaluation token limit")
        records.append(record)
        tokens.append(len(record.input_ids))
    device = next(model.parameters()).device
    batch = collate_records(records, processor.tokenizer.pad_token_id, device)
    logits = model(batch)
    if len(logits) != len(records):
        raise RuntimeError("Incomplete official SystemOne batch")
    results = []
    for body, record, output, token_count in zip(bodies, records, logits, tokens):
        rows = [output] if getattr(output, "ndim", 1) == 1 else list(output)
        if len(rows) != len(record.questions) or len(rows) != len(body["questions"]):
            raise RuntimeError("Official Clef batch returned an invalid question count")
        answers = {}
        for question_id, question, encoded_question, row in zip(
            body["questions"], body["questions"].values(), record.questions, rows
        ):
            probabilities = row.float().softmax(-1).tolist()
            answers[question_id] = systemone_answer(
                question, dict(zip(encoded_question.option_ids, probabilities))
            )
        results.append({
            "model": body["model"],
            "answers": answers,
            "usage": {"input_tokens": token_count, "output_tokens": 0},
        })
    return results


def predict_laya_batch(agent, bodies, max_length):
    questions = bodies[0]["questions"]
    states = []
    for body in bodies:
        if body["questions"] != questions:
            raise ValueError("Compatible questions are required for native Laya batching")
        validate_laya_input(agent, body, max_length)
        states.append(body["state"])
    results = agent.predict_batch(
        states, questions, batch_size=len(states), max_len=max_length,
        head_max_len=max_length, sort_by_length=False,
    )
    if len(results) != len(bodies):
        raise RuntimeError("Incomplete official Laya batch")
    return [
        {"model": body["model"], "answers": result["answers"], "usage": result.get("usage", {})}
        for body, result in zip(bodies, results)
    ]


def predict_clef(model, processor, body, max_length):
    from joint_schema_model import encode_record, systemone

    # The official encoder truncates state. One extra token detects overflow
    # before inference while retaining the official schema/tokenization rules.
    encoded = encode_record(processor.tokenizer, body, max_length=max_length + 1, processor=processor)
    if len(encoded.input_ids) > max_length:
        raise ValueError("Input exceeds the evaluation token limit")
    return systemone(model, processor, body, max_length=max_length)


def validate_laya_input(agent, body, max_length, head_max_length=None):
    from laya.common import build_head, build_sequence, encode_text, render_options

    lengths = []
    head_max_length = max_length if head_max_length is None else head_max_length
    for question in body["questions"].values():
        internal = agent._to_internal(question)
        options = render_options(internal)
        tokenized = [encode_text(agent.tok, " " + option.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"] for option in options]
        if any(len(tokens) > 48 for tokens in tokenized):
            raise ValueError("Official Laya option limit would truncate criteria")
        instruction = encode_text(agent.tok, f'{internal["t"]} question: {internal["ins"]}'.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        head, markers, _ = build_head(agent.tok, internal, head_max_len=head_max_length)
        if len(head) != 3 + len(instruction) + sum(1 + len(tokens) for tokens in tokenized):
            raise ValueError("Official Laya head budget would truncate instructions")
        sequence, actual_markers, stats = build_sequence(agent.tok, body["state"], internal, max_len=max_length, head_max_len=head_max_length, return_truncation_stats=True)
        if stats["truncated"] or len(actual_markers) != len(markers):
            raise ValueError("Input exceeds the evaluation token limit")
        lengths.append(len(sequence))
    return lengths


def predict_laya(agent, body, max_length):
    validate_laya_input(agent, body, max_length)
    return agent.predict(body["state"], body["questions"], max_len=max_length, head_max_len=max_length)


def create_app(predict=None, settings=None, rerank=None, predict_batch=None):
    settings = os.environ if settings is None else settings
    api_key = settings.get("DECISION_EVAL_API_KEY", "")
    model_name = settings.get("MODEL_NAME", "")
    backend = settings.get("MODEL_BACKEND", "")
    try:
        max_inflight = int(settings.get("MAX_INFLIGHT_REQUESTS", "1"))
    except (TypeError, ValueError):
        raise RuntimeError("MAX_INFLIGHT_REQUESTS must be an integer") from None
    if not 1 <= max_inflight <= 128:
        raise RuntimeError("MAX_INFLIGHT_REQUESTS must be between 1 and 128")
    try:
        systemone_batch_size = int(settings.get("SYSTEMONE_BATCH_SIZE", "1"))
        systemone_batch_wait_ms = int(settings.get("SYSTEMONE_BATCH_WAIT_MS", "2"))
        systemone_queue_size = int(settings.get("SYSTEMONE_QUEUE_SIZE", "64"))
        systemone_timeout = int(settings.get("SYSTEMONE_TIMEOUT_SECONDS", "300"))
    except (TypeError, ValueError):
        raise RuntimeError("SystemOne batch settings must be integers") from None
    if not 1 <= systemone_batch_size <= 32:
        raise RuntimeError("SYSTEMONE_BATCH_SIZE must be between 1 and 32")
    if not 0 <= systemone_batch_wait_ms <= 1000:
        raise RuntimeError("SYSTEMONE_BATCH_WAIT_MS must be between 0 and 1000")
    if not 1 <= systemone_queue_size <= 1024:
        raise RuntimeError("SYSTEMONE_QUEUE_SIZE must be between 1 and 1024")
    if not 1 <= systemone_timeout <= 3600:
        raise RuntimeError("SYSTEMONE_TIMEOUT_SECONDS must be between 1 and 3600")
    queue_single = settings.get("SYSTEMONE_QUEUE_SINGLE_REQUESTS", "0")
    if queue_single not in {"0", "1"}:
        raise RuntimeError("SYSTEMONE_QUEUE_SINGLE_REQUESTS must be 0 or 1")
    # Keep the normal single-flight GPU guard, while allowing an explicitly
    # provisioned evaluation run to measure pointwise request concurrency.
    gate = threading.BoundedSemaphore(max_inflight)
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if (
            not api_key
            or credentials is None
            or not hmac.compare_digest(credentials.credentials.encode(), api_key.encode())
        ):
            raise HTTPException(401, "Valid evaluation API key required", headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app):
        batch_operation = predict_batch
        if len(api_key) < 32 or not model_name:
            raise RuntimeError("A model name and an evaluation API key of at least 32 characters are required")
        if predict is not None:
            app.state.predict = predict
            if batch_operation is None:
                batch_operation = getattr(predict, "batch", None)
        elif rerank is not None:
            app.state.rerank = rerank
        elif backend == "neohorse":
            from neohorse_backend import NeoHorseBackend
            import torch

            torch.set_num_threads(2)
            runtime = NeoHorseBackend(settings["MODEL_PATH"], settings["NEOHORSE_POOLING_URL"])
            app.state.predict = runtime
            batch_operation = runtime.batch
        else:
            import torch

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("A CUDA BF16 GPU is required")
            torch.set_num_threads(2)
            torch.cuda.set_per_process_memory_fraction(float(settings["GPU_MEMORY_FRACTION"]), 0)
            model_path = str(Path(settings["MODEL_PATH"]).resolve(strict=True))
            max_length = int(settings.get("MAX_INPUT_TOKENS", "2048"))
            sys.path.insert(0, model_path)
            if backend == "clef":
                from joint_schema_model import load_release_model

                model, processor = load_release_model(model_path, device="cuda", attn_implementation="sdpa")
                app.state.predict = lambda body: predict_clef(model, processor, body, max_length)
                batch_operation = lambda bodies: predict_clef_batch(model, processor, bodies, max_length)
                from decision_reranker import DecisionReranker

                app.state.rerank = DecisionReranker(backend, (model, processor), max_length)
            elif backend == "nox":
                from transformers import AutoModel

                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, device="cuda", threads=2)
                # The official backend exposes a hard encoding cap; lower it without
                # changing weights, calibration, batching or answer normalization.
                model.runtime.backend.cap = min(model.runtime.backend.cap, max_length)
                app.state.predict = lambda body: model.system_one(state=body["state"], questions=body["questions"])
            elif backend == "intern":
                from inference import DecisionEngine

                engine = DecisionEngine(checkpoint=model_path, device="cuda", max_length=max_length)
                app.state.predict = lambda body: engine.predict({"state": body["state"], "questions": body["questions"]})
                from decision_reranker import DecisionReranker

                app.state.rerank = DecisionReranker(backend, engine, max_length)
            elif backend == "laya":
                import laya

                agent = laya.load(model_path, device="cuda", fast=False, compile=False)
                app.state.predict = lambda body: predict_laya(agent, body, max_length)
                batch_operation = lambda bodies: predict_laya_batch(agent, bodies, max_length)
                from decision_reranker import DecisionReranker

                app.state.rerank = DecisionReranker(backend, agent, max_length, validate_laya_input)
            elif backend == "d3":
                from d3_runtime import D3
                from decision_reranker import DecisionReranker

                runtime = D3.from_pretrained(model_path, device="cuda:0", batch_size=1,
                                            local_files_only=True, verify="fast", max_length=max_length)
                app.state.predict = lambda body: runtime.system_one(state=body["state"], questions=body["questions"])
                batch_operation = lambda bodies: predict_d3_batch(runtime, bodies, max_length)
                app.state.rerank = DecisionReranker(backend, runtime, max_length)
            elif backend == "qwen_reranker":
                from qwen_reranker import QwenReranker

                app.state.rerank = QwenReranker(model_path, max_length)
            elif backend == "jeva":
                from jeva import Jeva
                from jeva_backend import predict_jeva

                runtime = Jeva.from_pretrained(
                    model_path, device="cuda", local_files_only=True, cache="none",
                    max_state_length=max_length, max_option_length=max_length,
                    option_chunk_size=int(settings.get("JEVA_OPTION_CHUNK_SIZE", "16")),
                    option_batch_tokens=int(settings.get("JEVA_OPTION_BATCH_TOKENS", "16384")),
                    truncation="error",
                )
                app.state.predict = lambda body: predict_jeva(runtime, body)
            else:
                raise RuntimeError("Unknown decision runtime")
            if batch_operation is not None:
                app.state.predict_batch = batch_operation
            print(f"Loaded {model_name} using {backend}; torch={torch.__version__}; max_input_tokens={max_length}", flush=True)
        if batch_operation is not None:
            app.state.predict_batch = batch_operation
            if systemone_batch_size > 1:
                app.state.systemone_batcher = SystemOneBatcher(
                    operation=lambda bodies: invoke_batch(batch_operation, bodies),
                    max_batch_size=systemone_batch_size,
                    wait_ms=systemone_batch_wait_ms,
                    queue_size=systemone_queue_size,
                    timeout_seconds=systemone_timeout,
                )
                app.state.systemone_batcher.start()
        if queue_single == "1" and not hasattr(app.state, "systemone_batcher") and hasattr(app.state, "predict"):
            app.state.systemone_batcher = SystemOneBatcher(
                operation=lambda bodies: [invoke(app.state.predict, bodies[0])],
                max_batch_size=1, wait_ms=0, queue_size=systemone_queue_size,
                native_batching=False, timeout_seconds=systemone_timeout,
            )
            app.state.systemone_batcher.start()
        try:
            yield
        finally:
            batcher = getattr(app.state, "systemone_batcher", None)
            if batcher is not None:
                batcher.close()

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(authorize)], docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        readiness = getattr(getattr(app.state, "predict", None), "is_ready", None)
        if readiness is not None and not readiness():
            raise HTTPException(503, "Model dependency is not ready")
        return {"status": "ok", "model": model_name, "backend": backend}

    def invoke(operation, body):
        # The batch worker is the sole owner of its native batch forward, but
        # it still shares the GPU with the optional reranker endpoint.
        if not gate.acquire(blocking=True, timeout=300):
            raise RuntimeError("Model is busy")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Warning: GPU memory exceeded.*", category=RuntimeWarning)
                try:
                    return operation(body)
                except RuntimeWarning:
                    raise RuntimeError("GPU out of memory; CPU fallback disabled") from None
        finally:
            gate.release()

    def invoke_batch(operation, bodies):
        return invoke(operation, bodies)

    def execute(request, operation, batchable=False):
        if request.model != model_name:
            raise HTTPException(422, "Model does not match this endpoint")
        try:
            started = time.perf_counter()
            body = request.model_dump()
            if isinstance(request, RerankRequest) and body["questions"] is None:
                body.pop("questions")
            batcher = getattr(app.state, "systemone_batcher", None) if batchable else None
            if batcher is not None:
                result = batcher.submit(body)
            else:
                if not gate.acquire(blocking=False):
                    raise HTTPException(429, "Model is busy", headers={"Retry-After": "1"})
                try:
                    # The SDK warns immediately before a CPU retry. Keep all
                    # evaluation operations GPU-only, including classification.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", message="Warning: GPU memory exceeded.*", category=RuntimeWarning)
                        try:
                            result = operation(body)
                        except RuntimeWarning:
                            raise RuntimeError("GPU out of memory; CPU fallback disabled") from None
                finally:
                    gate.release()
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return result
        except BatchQueueFull:
            raise HTTPException(429, "SystemOne batch queue is full", headers={"Retry-After": "1"}) from None
        except TimeoutError:
            raise HTTPException(504, "SystemOne request deadline exceeded") from None
        except (ValueError, TypeError, KeyError):
            raise HTTPException(422, "Request could not be encoded by the official runtime") from None
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                import torch

                torch.cuda.empty_cache()
                raise HTTPException(503, "GPU memory budget exceeded") from None
            raise HTTPException(500, "Official runtime inference failed") from None

    @app.post("/v1/systemone")
    @app.post("/v1/decisions")
    def decide(request: DecisionRequest):
        if not hasattr(app.state, "predict"):
            raise HTTPException(422, "This backend does not provide decisions")
        return execute(request, app.state.predict, batchable=True)

    @app.post("/v1/rerank")
    def rerank_pool(request: RerankRequest):
        if not hasattr(app.state, "rerank"):
            raise HTTPException(422, "This backend does not provide batch reranking")
        return execute(request, app.state.rerank)

    return app


app = create_app()
