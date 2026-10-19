"""Isolated, serial passage-filtering extension; never imported by product services."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import public_decision_benchmark as benchmark
from decision_capacity_preflight import QUESTIONS, check_artifacts, llm_body, llm_result, save, save_rows, sha


ROSTER = (
    "d3-edge", "vela-2.0-0.3b", "vela-2.0-0.8b", "laya-multilingual", "d3-mini",
    "vela-2.0-4b", "neohorse-jev-4b", "jeva-4b", "d3-flash", "clef-flash",
    "vela-2.0-9b", "jev-27b", "pplx-decider-v1.1-27b",
    "qwen3-30b-a3b-instruct-2507", "qwen3.8-27b",
)
TOP_N = 29


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_pools(pools, labels=None):
    assert len({p["case_id"] for p in pools}) == len(pools)
    gold = {r["case_id"]: r["positive_ids"] for r in labels} if labels is not None else None
    if gold is not None:
        assert len(gold) == len(labels) and set(gold) == {p["case_id"] for p in pools}
    for pool in pools:
        assert pool["task"] == "reranking" and isinstance(pool["query"], str) and pool["query"].strip()
        ids = [c["fragment_id"] for c in pool["candidates"]]
        assert ids and len(set(ids)) == len(ids)
        for candidate in pool["candidates"]:
            assert candidate["content"].strip()
            assert hashlib.sha256(candidate["content"].encode()).hexdigest() == candidate["fragment_id"]
        if gold is not None:
            assert gold[pool["case_id"]] and len(set(gold[pool["case_id"]])) == len(gold[pool["case_id"]])
            assert set(gold[pool["case_id"]]) <= set(ids)


def mask_fragments(pools, labels, exclusions):
    """Apply an explicit predeclared mask, not a length/label-based selection rule."""
    validate_pools(pools, labels)
    keys = {(r["case_id"], r["fragment_id"]) for r in exclusions}
    assert len(keys) == len(exclusions) and keys
    all_keys = {(p["case_id"], c["fragment_id"]) for p in pools for c in p["candidates"]}
    assert keys <= all_keys, "Unknown exclusion; refuse silent partial masking"
    revised = [{**p, "candidates": [c for c in p["candidates"]
               if (p["case_id"], c["fragment_id"]) not in keys]} for p in pools]
    new_labels = [{**r, "positive_ids": [f for f in r["positive_ids"]
                  if (r["case_id"], f) not in keys]} for r in labels]
    validate_pools(revised, new_labels)
    return revised, new_labels


def prepare(args):
    os.umask(0o077)
    original = json.loads((args.formal_dataset / "dataset-protocol.private.json").read_text())
    engineering = json.loads((args.engineering_dataset / "protocol.json").read_text())
    exclusion = json.loads(args.exclusions.read_text())
    sources = {str(args.exclusions.resolve()): sha(args.exclusions)}
    for name, field in (("inputs.private.jsonl", "inputs_sha256"), ("labels.private.jsonl", "labels_sha256")):
        path = args.formal_dataset / name
        assert sha(path) == original[field]
        sources[str(path.resolve())] = sha(path)
    assert original["inputs_sha256"] == engineering["formal_inputs_sha256"] == exclusion["original_inputs_sha256"]
    for name, expected in engineering["files"].items():
        path = args.engineering_dataset / name
        assert sha(path) == expected
        sources[str(path.resolve())] = expected
    for path in (args.formal_dataset / "dataset-protocol.private.json", args.engineering_dataset / "protocol.json"):
        sources[str(path.resolve())] = sha(path)
    pools = [r for r in benchmark.read_jsonl(args.formal_dataset / "inputs.private.jsonl") if r["task"] == "reranking"]
    labels = [r for r in benchmark.read_jsonl(args.formal_dataset / "labels.private.jsonl") if r["task"] == "reranking"]
    revised, gold = mask_fragments(pools, labels, exclusion["pairs"])
    assert len(pools) == exclusion["expected_original_pools"]
    assert sum(len(p["candidates"]) for p in pools) == exclusion["expected_original_candidates"]
    assert sum(len(p["candidates"]) for p in revised) == exclusion["expected_retained_candidates"]
    positives = sum(len(p["positive_ids"]) for p in gold)
    assert positives == exclusion["expected_retained_positives"]
    assert sum(len(p["positive_ids"]) for p in labels) - positives == exclusion["expected_removed_positives"]
    ep = benchmark.read_jsonl(args.engineering_dataset / "engineering-pools.private.jsonl")
    vp = benchmark.read_jsonl(args.engineering_dataset / "validation-pools.private.jsonl")
    vl = benchmark.read_jsonl(args.engineering_dataset / "validation-labels.private.jsonl")
    validate_pools(ep)
    validate_pools(vp, vl)
    assert len(ep) == 12 and len(vp) == 20
    all_pools = revised + ep + vp
    assert len({p["case_id"] for p in all_pools}) == len(all_pools)
    assert len({p["query"] for p in all_pools}) == len(all_pools)
    assert len({benchmark.digest(p["candidates"]) for p in all_pools}) == len(all_pools)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("formal.inputs.private.jsonl", revised), ("formal.labels.private.jsonl", gold),
                       ("engineering.inputs.private.jsonl", ep), ("validation.inputs.private.jsonl", vp),
                       ("validation.labels.private.jsonl", vl)):
        save_rows(args.output / name, rows)
    save(args.output / "exclusions.json", exclusion)
    protocol = {"version": "passage-six-fragment-mask-v1", "at": now(), "pools": len(revised),
        "candidates": sum(len(p["candidates"]) for p in revised), "positives": positives,
        "negatives": sum(len(p["candidates"]) for p in revised) - positives,
        "original_pools": len(pools), "original_candidates": sum(len(p["candidates"]) for p in pools),
        "excluded_pairs": len(exclusion["pairs"]), "top_n": TOP_N, "concurrency": 1,
        "preserved_source_sha256": sources,
        "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}
    save(args.output / "protocol.json", protocol)
    print(json.dumps({k: protocol[k] for k in ("pools", "candidates", "positives", "negatives", "excluded_pairs")}))


def pair_state(pool, candidate):
    return {"query": pool["query"], "document": candidate["content"]}


def refine_wire(family, served, state, questions=None):
    from deployment_probe import native_question_request
    questions = QUESTIONS if questions is None else questions
    if family == "llm":
        return "/v1/chat/completions", llm_body(served, state, questions=questions)
    return native_question_request(family, {"model": served, "state": state, "questions": questions})


def refine_result(family, served, state, raw, questions=None):
    from deployment_probe import native_choice_result, native_noul_result
    questions = QUESTIONS if questions is None else questions
    if family == "llm":
        return llm_result(raw, served)
    payload = {"model": served, "state": state, "questions": questions}
    if next(iter(questions.values()))["type"] == "noul":
        return native_noul_result(family, payload, raw)
    prediction, probabilities = native_choice_result(family, payload, raw)
    assert not (raw.get("usage") or {}).get("truncated_tokens", 0)
    return {"prediction": prediction, "probabilities": probabilities,
            "usage": raw.get("usage"), "execution": raw.get("execution")}


def rerank_result(pool, raw, top_n=TOP_N):
    assert type(top_n) is int and top_n > 0
    assert raw["model"] == "Qwen3-Reranker-8B"
    rows = raw["results"]
    count = len(pool["candidates"])
    assert all(type(r["index"]) is int for r in rows)
    assert sorted(r["index"] for r in rows) == list(range(count))
    assert all(type(r["relevance_score"]) in (int, float) and math.isfinite(r["relevance_score"])
               and 0 <= r["relevance_score"] <= 1 for r in rows)
    scores = {r["index"]: r["relevance_score"] for r in rows}
    order = sorted(range(count), key=lambda i: (-scores[i], i))
    return {"scores": [{"fragment_id": c["fragment_id"], "score": scores[i]}
                       for i, c in enumerate(pool["candidates"])],
            "retained_ids": [pool["candidates"][i]["fragment_id"] for i in order[:top_n]],
            "usage": raw.get("usage")}


def selected(records, rule):
    assert rule["kind"] in ("threshold", "hard_choice")
    return {r["fragment_id"] for r in records if r["status"] == "valid" and
            (r["probabilities"]["relevant"] >= rule["threshold"] if rule["kind"] == "threshold"
             else r["prediction"] == "relevant")}


def set_metrics(gold, prediction, valid=True):
    assert gold, "This protocol requires a positive in every retained pool"
    if not valid:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(gold & prediction)
    return {"precision": tp / len(prediction) if prediction else 0.0,
            "recall": tp / len(gold), "f1": 2 * tp / (len(gold) + len(prediction))}


def calibrate(pools, labels, records, continuous):
    grouped = {p["case_id"]: [r for r in records if r["case_id"] == p["case_id"]] for p in pools}
    gold = {r["case_id"]: set(r["positive_ids"]) for r in labels}
    for p in pools:
        assert {r["fragment_id"] for r in grouped[p["case_id"]]} == {c["fragment_id"] for c in p["candidates"]}
        assert len(grouped[p["case_id"]]) == len(p["candidates"])
    thresholds = [0.0, math.nextafter(1.0, math.inf)]
    if continuous:
        thresholds += [r["probabilities"]["relevant"] for r in records if r["status"] == "valid"]
    else:
        thresholds = [None]
    candidates = []
    for threshold in sorted(set(thresholds)) if continuous else thresholds:
        rule = {"kind": "threshold", "threshold": threshold} if continuous else {"kind": "hard_choice"}
        scores = [set_metrics(gold[p["case_id"]], selected(grouped[p["case_id"]], rule),
                  all(r["status"] == "valid" for r in grouped[p["case_id"]])) for p in pools]
        candidates.append({**rule, "macro_f1": statistics.mean(s["f1"] for s in scores),
                           "macro_precision": statistics.mean(s["precision"] for s in scores)})
    best = max(candidates, key=lambda r: (r["macro_f1"], r["macro_precision"], r.get("threshold", 0)))
    return {**best, "validation_pools": len(pools), "validation_requests": len(records),
            "valid_requests": sum(r["status"] == "valid" for r in records),
            "validation_records_sha256": benchmark.digest(records), "candidates_evaluated": len(candidates),
            "tie_break": "macro F1, then macro Precision, then higher threshold; score >= threshold"}


class HaltSequence(RuntimeError):
    """An uncertain server terminal state must never be followed by more inference."""


@contextmanager
def wall_deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError("HTTP wall deadline")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class SerialHTTP:
    def __init__(self, http, key, output, timeout=300, stop_file=None):
        self.http, self.key, self.output, self.timeout = http, key, output, timeout
        self.stop_file = stop_file or output / "STOP"
        self.halted = False
        self.journal_seconds = 0.0
        self.active = 0
        self.seen = set()
        self.origin = time.perf_counter()
        self.intent = (output / "intents.jsonl").open("x")
        self.results = (output / "requests.jsonl").open("x")

    @staticmethod
    def append(stream, row):
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def close(self):
        self.intent.close()
        self.results.close()

    def call(self, identity, endpoint, build, parse, input_hash):
        if self.halted:
            raise HaltSequence("Transport is halted; no further inference is allowed")
        assert self.active == 0, "Overlapping benchmark request"
        call_id = benchmark.digest(identity)
        assert call_id not in self.seen, "Attempt replay refused"
        assert not self.stop_file.exists(), "Stop marker present"
        self.seen.add(call_id)
        started = time.perf_counter()
        row = {**identity, "call_id": call_id, "at": now(), "input_sha256": input_hash,
               "calls": 0, "started_seconds": started - self.origin, "inflight_at_start": 1}
        terminal = False
        self.active = 1
        fatal = False
        intent_seconds = 0.0
        try:
            path, body = build()
            row.update(path=path, request_sha256=benchmark.digest(body))
            intent_start = time.perf_counter()
            self.append(self.intent, {**row, "planned_http_calls": 1})
            intent_seconds = time.perf_counter() - intent_start
            self.journal_seconds += intent_seconds
            row["calls"] = 1
            row["http_started_seconds"] = time.perf_counter() - self.origin
            with wall_deadline(self.timeout):
                response = self.http.post(endpoint + path, json=body,
                    headers={"Authorization": "Bearer " + self.key, "X-Request-Id": call_id},
                    timeout=self.timeout)
                row.update(http_status=response.status_code, response_sha256=hashlib.sha256(response.content).hexdigest())
                terminal = response.status_code in (200, 400, 401, 403, 404, 413, 422)
                response.raise_for_status()
                raw = response.json()
                row["usage"] = raw.get("usage")
                if raw.get("choices"):
                    choice = raw["choices"][0]
                    message = choice.get("message") or {}
                    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                    row.update(finish_reason=choice.get("finish_reason"), reasoning_chars=len(reasoning),
                               reasoning_sha256=benchmark.digest(reasoning))
                row.update(parse(raw), status="valid")
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            fatal = (not terminal or row.get("http_status") in (401, 403, 404)
                     or not isinstance(error, Exception))
            row["sequence_halted"] = fatal
            row["server_terminal_confirmed"] = terminal
        finally:
            self.active = 0
            row.update(elapsed_ms=(time.perf_counter() - started - intent_seconds) * 1000,
                       intent_journal_ms=intent_seconds * 1000,
                       ended_seconds=time.perf_counter() - self.origin)
            journal_start = time.perf_counter()
            self.append(self.results, row)
            self.journal_seconds += time.perf_counter() - journal_start
        if fatal:
            self.halted = True
            raise HaltSequence("Request terminal state uncertain or endpoint unavailable; no automatic retry")
        return row


def request_stats(rows):
    valid = [r for r in rows if r["status"] == "valid"]
    return {"attempted": len(rows), "http_calls": sum(r["calls"] for r in rows), "valid": len(valid),
            "failed": len(rows) - len(valid), "successful_latency_ms": benchmark.timing([r["elapsed_ms"] for r in valid]),
            "timeouts": sum("Timeout" in r.get("error_type", "") for r in rows),
            "all_attempt_latency_ms": benchmark.timing([r["elapsed_ms"] for r in rows]),
            "http_statuses": dict(Counter(str(r.get("http_status")) for r in rows)),
            "failure_types": dict(Counter(r["error_type"] for r in rows if r["status"] != "valid"))}


def retention_stats(labels, rows):
    gold = {r["case_id"]: set(r["positive_ids"]) for r in labels}
    valid = [r for r in rows if r["stage"] == "rerank" and r["status"] == "valid"]
    return {"valid_pools": len(valid), "all_pools": len(labels),
            "macro_positive_retention": statistics.mean(len(gold[r["case_id"]] & set(r["retained_ids"])) /
                len(gold[r["case_id"]]) for r in valid) if valid else None,
            "positive_denominator_on_valid_pools": sum(len(gold[r["case_id"]]) for r in valid),
            "pruned_positive_candidates": sum(len(gold[r["case_id"]] - set(r["retained_ids"])) for r in valid),
            "retained_candidates": sum(len(r["retained_ids"]) for r in valid),
            "scored_candidates": sum(len(r["scores"]) for r in valid)}


def score_chains(pools, labels, chains):
    gold = {r["case_id"]: set(r["positive_ids"]) for r in labels}
    ids = [p["case_id"] for p in pools]
    assert len(chains) == len(pools) * 2
    by_key = {(r["arm"], r["case_id"]): r for r in chains}
    assert len(by_key) == len(chains)
    arms = sorted({r["arm"] for r in chains})
    assert len(arms) == 2
    report = {"arms": {}, "paired": {}}
    values = {}
    for arm in arms:
        rows = [by_key[(arm, cid)] for cid in ids]
        metrics = [set_metrics(gold[r["case_id"]], set(r["selected_ids"]), r["status"] == "valid") for r in rows]
        values[arm] = metrics
        report["arms"][arm] = {"pools": len(rows), "valid_pools": sum(r["status"] == "valid" for r in rows),
            "quality": {k: benchmark.bootstrap_mean([v[k] for v in metrics]) for k in ("precision", "recall", "f1")},
            "successful_chain_latency_ms": benchmark.timing([r["wall_ms"] for r in rows if r["status"] == "valid"]),
            "all_chain_latency_ms": benchmark.timing([r["wall_ms"] for r in rows])}
    direct, pipeline = arms
    paired = [(by_key[(direct, cid)], by_key[(pipeline, cid)]) for cid in ids
              if by_key[(direct, cid)]["status"] == by_key[(pipeline, cid)]["status"] == "valid"]
    report["paired"] = {"direct": direct, "pipeline": pipeline, "quality_delta": {
        k: benchmark.bootstrap_mean([b[k] - a[k] for a, b in zip(values[direct], values[pipeline])])
        for k in ("precision", "recall", "f1")}, "timed_successful_pairs": len(paired),
        "time_saved_ms": benchmark.bootstrap_mean([a["wall_ms"] - b["wall_ms"] for a,b in paired]) if paired else None,
        "ratio_of_summed_paired_times": sum(a["wall_ms"] for a,b in paired) / sum(b["wall_ms"] for a,b in paired) if paired else None}
    for name, condition in (("candidates_le29", lambda n: n <= TOP_N), ("candidates_gt29", lambda n: n > TOP_N)):
        subset = [p for p in pools if condition(len(p["candidates"]))]
        if subset and len(subset) != len(pools):
            chosen = {p["case_id"] for p in subset}
            report[name] = score_chains(subset, [r for r in labels if r["case_id"] in chosen],
                                       [r for r in chains if r["case_id"] in chosen])
    return report


def audit_model(folder, dataset, model):
    """Reconstruct all request contracts from the frozen data without inference."""
    from launch_reviewed import MODELS
    family, _, served = MODELS[model]
    direct, pipeline = ("B2", "B4") if family == "llm" else ("B1", "B3")
    rows = benchmark.read_jsonl(folder / "requests.jsonl")
    intents = benchmark.read_jsonl(folder / "intents.jsonl")
    chains = benchmark.read_jsonl(folder / "chains.jsonl")
    identity_fields = ("model", "phase", "arm", "stage", "case_id", "fragment_id")
    assert len({r["call_id"] for r in rows}) == len(rows) == len(intents)
    assert [r["call_id"] for r in rows] == [r["call_id"] for r in intents]
    assert all(a["ended_seconds"] <= b["started_seconds"] for a,b in zip(rows, rows[1:]))
    by_id = {r["call_id"]: r for r in rows}
    pools = {phase: {p["case_id"]: p for p in dataset[phase]} for phase in ("engineering", "validation", "formal")}
    expected = set()

    def key(phase, arm, stage, case_id, fragment_id=None):
        identity = dict(model=model, phase=phase, arm=arm, stage=stage, case_id=case_id, fragment_id=fragment_id)
        return benchmark.digest(identity)

    for phase in ("engineering", "validation"):
        expected.update(key(phase, phase, "refine", p["case_id"], c["fragment_id"])
                        for p in dataset[phase] for c in p["candidates"])
    for p in dataset["formal"]:
        expected.update(key("formal", direct, "refine", p["case_id"], c["fragment_id"]) for c in p["candidates"])
        rerank_id = key("formal", pipeline, "rerank", p["case_id"])
        expected.add(rerank_id)
        rr = by_id[rerank_id]
        if rr["status"] == "valid":
            scores = {r["fragment_id"]: r["score"] for r in rr["scores"]}
            assert set(scores) == {c["fragment_id"] for c in p["candidates"]}
            ordered = sorted(p["candidates"], key=lambda c: -scores[c["fragment_id"]])
            assert rr["retained_ids"] == [c["fragment_id"] for c in ordered[:TOP_N]]
            expected.update(key("formal", pipeline, "refine", p["case_id"], f) for f in rr["retained_ids"])
    assert set(by_id) == expected, "Missing, unexpected or reused request identities"
    for row in rows:
        identity = {k: row[k] for k in identity_fields}
        assert benchmark.digest(identity) == row["call_id"]
        assert row["calls"] == row["inflight_at_start"] == 1
        p = pools[row["phase"]][row["case_id"]]
        if row["stage"] == "rerank":
            state = p
            path, body = "/rerank", {"model": "Qwen3-Reranker-8B", "query": p["query"],
                                      "documents": [c["content"] for c in p["candidates"]]}
        else:
            candidate = next(c for c in p["candidates"] if c["fragment_id"] == row["fragment_id"])
            state = pair_state(p, candidate)
            path, body = refine_wire(family, served, state)
        assert row["input_sha256"] == benchmark.digest(state)
        assert row["path"] == path and row["request_sha256"] == benchmark.digest(body)
    consumed = [rid for chain in chains for rid in chain["request_ids"]]
    formal_ids = {r["call_id"] for r in rows if r["phase"] == "formal"}
    assert len(consumed) == len(set(consumed)) and set(consumed) == formal_ids
    assert len(chains) == len(dataset["formal"]) * 2
    rule = json.loads((folder / "threshold.json").read_text())
    validation = [r for r in rows if r["phase"] == "validation"]
    assert rule == calibrate(dataset["validation"], dataset["validation_labels"], validation, family != "llm")
    for chain in chains:
        p = pools["formal"][chain["case_id"]]
        assert chain["input_sha256"] == benchmark.digest(p)
        assert chain["threshold_sha256"] == sha(folder / "threshold.json")
        members = [by_id[rid] for rid in chain["request_ids"]]
        valid = all(r["status"] == "valid" for r in members)
        assert (chain["status"] == "valid") == valid
        predicted = selected([r for r in members if r["stage"] == "refine"], rule)
        assert chain["selected_ids"] == (sorted(predicted) if valid else [])
    saved = json.loads((folder / "summary.json").read_text())
    recomputed = score_chains(dataset["formal"], dataset["formal_labels"], chains)
    assert {k:v for k,v in saved.items() if k not in ("requests", "reranker_retention")} == recomputed
    assert saved["reranker_retention"] == retention_stats(dataset["formal_labels"], [r for r in rows if r["phase"] == "formal"])
    for name, value in saved["requests"].items():
        arm, stage = name.split(":")
        assert value == request_stats([r for r in rows if r["phase"] == "formal" and r["arm"] == arm and r["stage"] == stage])
    return {"at": now(), "request_records": len(rows), "formal_requests": len(formal_ids),
            "chains": len(chains), "unique_calls": True, "request_hashes_reconstructed": True,
            "input_hashes_reconstructed": True, "nonoverlapping_intervals": True,
            "threshold_recomputed": True, "metrics_recomputed": True,
            "artifact_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file()}}


def readiness(http, base, key, startup_timeout, container=None):
    import httpx
    deadline = time.monotonic() + startup_timeout
    while True:
        try:
            response = http.get(base + "/health", headers={"Authorization": "Bearer " + key}, timeout=5)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Startup deadline exceeded")
        if container:
            state = subprocess.check_output(["docker", "inspect", container, "--format", "{{.State.Status}}"], text=True).strip()
            assert state == "running", "Evaluation runtime stopped during startup"
        time.sleep(3)
    codes = [http.get(base + "/health", headers={} if credential is None else
             {"Authorization": "Bearer " + credential}, timeout=5).status_code
             for credential in (None, "invalid-evaluation-key")]
    assert codes == [401, 401], "Authentication acceptance failed"
    return {"ready": 200, "unauthorized": codes}


def container_identity():
    template = ('{"id":{{json .Id}},"image_id":{{json .Image}},"pid":{{json .State.Pid}},'
                '"started_at":{{json .State.StartedAt}},"running":{{json .State.Running}},'
                '"owned":{{json (index .Config.Labels "evaluation.decision.acceptance")}}}')
    result = json.loads(subprocess.check_output(["docker", "inspect", "decision-acceptance-model", "--format", template], text=True))
    assert result["owned"] == "true" and result["running"]
    return result


def execute_model(args, model, folder, http, key, dataset, reference_top, status):
    from launch_reviewed import MODELS
    family, _, served = MODELS[model]
    direct, pipeline = ("B2", "B4") if family == "llm" else ("B1", "B3")
    transport = SerialHTTP(http, key, folder, args.timeout, args.output / "STOP")
    formal_rows, chains = [], []

    def refine(pool, candidate, phase, arm):
        state = pair_state(pool, candidate)
        return transport.call({"model": model, "phase": phase, "arm": arm, "stage": "refine",
            "case_id": pool["case_id"], "fragment_id": candidate["fragment_id"]}, args.base_url,
            lambda: refine_wire(family, served, state), lambda raw: refine_result(family, served, state, raw),
            benchmark.digest(state))

    try:
        save(folder / "auth.json", readiness(http, args.base_url, key, args.startup_timeout, "decision-acceptance-model"))
        identity = container_identity()
        save(folder / "container.json", identity)
        engineering = []
        for pool in dataset["engineering"]:
            status("engineering", model=model, case_id=pool["case_id"])
            for candidate in pool["candidates"]:
                row = refine(pool, candidate, "engineering", "engineering")
                engineering.append(row)
                assert row["status"] == "valid", "Complete-input engineering acceptance failed"
        save(folder / "engineering.json", request_stats(engineering))
        validation = []
        for pool in dataset["validation"]:
            status("calibrating", model=model, case_id=pool["case_id"])
            for candidate in pool["candidates"]:
                validation.append(refine(pool, candidate, "validation", "validation"))
        rule = calibrate(dataset["validation"], dataset["validation_labels"], validation, family != "llm")
        save(folder / "threshold.json", rule)
        assert container_identity() == identity, "Refiner changed between acceptance and formal inference"
        chain_stream = (folder / "chains.jsonl").open("x")
        try:
            for index, pool in enumerate(dataset["formal"]):
                status("formal", model=model, case_id=pool["case_id"], completed_pools=index)
                assert not (args.output / "STOP").exists(), "Run stop marker present"
                arms = (direct, pipeline) if index % 2 == 0 else (pipeline, direct)
                for arm in arms:
                    start = time.perf_counter()
                    journal_before = transport.journal_seconds
                    candidates = pool["candidates"]
                    rr = None
                    if arm == pipeline:
                        rr = transport.call({"model": model, "phase": "formal", "arm": arm, "stage": "rerank",
                            "case_id": pool["case_id"], "fragment_id": None}, args.reranker_url,
                            lambda: ("/rerank", {"model": "Qwen3-Reranker-8B", "query": pool["query"],
                                     "documents": [c["content"] for c in pool["candidates"]]}),
                            lambda raw: rerank_result(pool, raw), benchmark.digest(pool))
                        formal_rows.append(rr)
                        if rr["status"] == "valid":
                            membership = set(rr["retained_ids"])
                            previous = reference_top.get(pool["case_id"])
                            assert previous is None or set(previous) == membership, "Top29 membership changed across refiners"
                            reference_top[pool["case_id"]] = rr["retained_ids"]
                            candidates = [c for c in candidates if c["fragment_id"] in membership]
                        else:
                            candidates = []
                    rows = [refine(pool, candidate, "formal", arm) for candidate in candidates]
                    formal_rows.extend(rows)
                    valid = all(r["status"] == "valid" for r in rows) and (rr is None or rr["status"] == "valid")
                    selection = selected(rows, rule)
                    chain = {"model": model, "arm": arm, "case_id": pool["case_id"],
                        "input_sha256": benchmark.digest(pool), "status": "valid" if valid else "failed",
                        "selected_ids": sorted(selection) if valid else [],
                        "partial_selected_ids": sorted(selection) if not valid else [],
                        "retained_ids": rr.get("retained_ids") if rr else [c["fragment_id"] for c in candidates],
                        "original_revised_candidates": len(pool["candidates"]),
                        "planned_refine": len(candidates), "attempted_refine": len(rows),
                        "pruned": len(pool["candidates"]) - len(candidates) if rr and rr["status"] == "valid" else 0,
                        "upstream_blocked": rr is not None and rr["status"] != "valid",
                        "upstream_blocked_refine_count": min(TOP_N, len(pool["candidates"])) if rr and rr["status"] != "valid" else 0,
                        "request_ids": ([rr["call_id"]] if rr else []) + [r["call_id"] for r in rows],
                        "threshold_sha256": sha(folder / "threshold.json"),
                        "request_journal_ms": (transport.journal_seconds - journal_before) * 1000,
                        "wall_ms": (time.perf_counter() - start) * 1000}
                    SerialHTTP.append(chain_stream, chain)
                    chains.append(chain)
                assert container_identity() == identity, "Refiner identity changed during formal execution"
        finally:
            chain_stream.close()
        report = score_chains(dataset["formal"], dataset["formal_labels"], chains)
        report["requests"] = {arm + ":" + stage: request_stats([r for r in formal_rows if r["arm"] == arm and r["stage"] == stage])
                              for arm, stage in ((direct, "refine"), (pipeline, "rerank"), (pipeline, "refine"))}
        report["reranker_retention"] = retention_stats(dataset["formal_labels"], formal_rows)
        save(folder / "summary.json", report)
    finally:
        transport.close()


def run(args):
    import httpx
    os.umask(0o077)
    sys.path.insert(0, str(args.service_source.resolve(strict=True)))
    from launch_reviewed import MODELS, launch, read_key
    args.base_url = benchmark.local_endpoint(args.base_url)
    args.reranker_url = benchmark.local_endpoint(args.reranker_url)
    assert args.base_url != args.reranker_url
    assert args.models == list(ROSTER), "This run requires the entire frozen 15-refiner roster"
    assert args.timeout == 300 and args.startup_timeout == 2400
    dp = json.loads((args.dataset / "protocol.json").read_text())
    assert dp["candidates"] == 3188 and dp["pools"] == 200 and dp["excluded_pairs"] == 6
    check_artifacts(dp["preserved_source_sha256"])
    for name, value in dp["files"].items():
        assert sha(args.dataset / name) == value
    dataset = {name: benchmark.read_jsonl(args.dataset / filename) for name, filename in (
        ("engineering", "engineering.inputs.private.jsonl"), ("validation", "validation.inputs.private.jsonl"),
        ("validation_labels", "validation.labels.private.jsonl"), ("formal", "formal.inputs.private.jsonl"),
        ("formal_labels", "formal.labels.private.jsonl"))}
    validate_pools(dataset["formal"], dataset["formal_labels"])
    validate_pools(dataset["validation"], dataset["validation_labels"])
    preserved = json.loads(args.preservation.read_text())
    check_artifacts(preserved)
    sources = {str(p.resolve()): sha(p) for p in (Path(__file__), Path(benchmark.__file__),
        Path(__file__).with_name("decision_capacity_preflight.py"), *args.service_source.glob("*.py"),
        args.service_source / "nginx.native.conf.template", args.service_source / "qwen3_reranker.jinja")}
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "protocol.json", {"version": "passage-four-arm-n1-v1", "at": now(),
            "models": args.models, "dataset_protocol_sha256": sha(args.dataset / "protocol.json"),
            "source_sha256": sources, "preserved_artifact_sha256": preserved, "top_n": TOP_N,
            "http_concurrency": 1, "pool_concurrency": 1, "timeout": 300, "startup_timeout": 2400,
            "max_sequences": 32, "max_batched_tokens": 32768, "laya_http_admission": 32,
            "questions": QUESTIONS, "llm_mode": "capacity-accepted nonthinking strict JSON, temperature 0, max_tokens 64",
            "order": "frozen original pool order; direct/pipeline alternate first by pool index, identical across refiners",
            "reranker": json.loads(args.reranker_evidence.read_text()),
            "request_timer": "construct/hash/HTTP/parse through validation; measured intent journal time subtracted and result write excluded",
            "chain_timer": "real online wall time, including all stage HTTP, network and per-request durable journals",
            "retries": 0, "fallback": False, "truncation": False, "automatic_resume": False,
            "formal_exclusions_sha256": sha(args.dataset / "exclusions.json"),
            "limitations": ["length-filtered public comparison, not original 3194-pair workload", "extra shared proxy GPU and host resources",
                            "business embedding traffic is outside benchmark n1; not GPU isolation", "20-pool threshold calibration"]})
        state = {"pid": os.getpid(), "completed": [], "models": args.models}

        def status(stage, **values):
            state.update(stage=stage, at=now(), **values)
            benchmark.write_json(args.output / "status.json", state)
            print(json.dumps(state), flush=True)

        status("starting")
        try:
            key = read_key(args.env_file)
            with httpx.Client(transport=httpx.HTTPTransport(retries=0), trust_env=False,
                              follow_redirects=False, timeout=300) as http:
                save(args.output / "reranker-auth.json", readiness(http, args.reranker_url, key, 30))
                reference_top = {}
                for model in args.models:
                    assert not (args.output / "STOP").exists(), "Run stop marker present"
                    check_artifacts(sources)
                    check_artifacts(preserved)
                    folder = args.output / model
                    folder.mkdir()
                    status("loading", model=model, case_id=None, completed_pools=0)
                    launch(SimpleNamespace(root=args.root, model=model, env_file=args.env_file,
                        native_port=args.port - 1, public_port=args.port, seqs=32, tokens=32768,
                        enforce_eager=False, laya_max_concurrent=32 if model == "laya-multilingual" else None))
                    save(folder / "deployment.json", json.loads((args.root / "acceptance-20261015/evidence" /
                         (model + ".deployment.json")).read_text()))
                    execute_model(args, model, folder, http, key, dataset, reference_top, status)
                    status("auditing", model=model)
                    save(folder / "audit.json", audit_model(folder, dataset, model))
                    check_artifacts(preserved)
                    state["completed"].append(model)
                    status("model_complete")
                save(args.output / "top29-reference.json", reference_top)
            check_artifacts(sources)
            check_artifacts(preserved)
            save(args.output / "preservation-final.json", {"unchanged": True, "artifacts": len(preserved)})
            status("complete", model=None, case_id=None)
        except BaseException as error:
            status("failed", error_type=type(error).__name__)
            raise


def register(subparsers):
    parser = subparsers.add_parser("prepare-passage", help="Apply an explicit owner-approved fragment mask; no inference")
    for name in ("formal-dataset", "engineering-dataset", "exclusions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.set_defaults(function=prepare)
    parser = subparsers.add_parser("local-passage", help="Frozen four-arm online passage filtering at global n1")
    for name in ("root", "service-source", "dataset", "env-file", "output", "lock", "preservation", "reranker-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(ROSTER))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reranker-url", required=True)
    parser.add_argument("--port", type=int, default=30889)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=2400)
    parser.set_defaults(function=run)
