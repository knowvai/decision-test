"""Native noul validation only; reuse frozen passage transport and native launcher."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import decision_passage_benchmark as p

b = p.benchmark
MODELS = p.ROSTER[:-2]
QUESTIONS = {name: {"type": "noul", "instructions": q["instructions"]} for name, q in p.QUESTIONS.items()}


def payload(served, state):
    return {"model": served, "state": state, "questions": QUESTIONS}


def jobs(pools):
    return [(pool, candidate) for pool in pools for candidate in pool["candidates"]]


def quality(pools, labels, rows):
    by_key = {(r["case_id"], r["fragment_id"]): r for r in rows}
    assert len(by_key) == len(rows)
    gold = {r["case_id"]: set(r["positive_ids"]) for r in labels}
    cases, counts = [], Counter()
    for pool in pools:
        cid = pool["case_id"]
        records = [by_key.get((cid, c["fragment_id"])) for c in pool["candidates"]]
        valid = all(r is not None and r["status"] == "valid" for r in records)
        selected = {r["fragment_id"] for r in records if r and r["status"] == "valid" and r["prediction"] == "relevant"}
        cases.append({"case_id": cid, "valid": valid, **p.set_metrics(gold[cid], selected, valid)})
        counts["empty_pools"] += int(not selected)
        for candidate, row in zip(pool["candidates"], records):
            if row is None or row["status"] != "valid":
                counts["unscored"] += 1
            else:
                positive, chosen = candidate["fragment_id"] in gold[cid], row["prediction"] == "relevant"
                counts["tp" if positive and chosen else "fn" if positive else "fp" if chosen else "tn"] += 1
    denominator = sum(len(pool["candidates"]) for pool in pools)
    return {"pools": len(pools), "valid_pools": sum(r["valid"] for r in cases), "candidates": denominator,
            "counts": dict(counts), "candidate_accuracy": (counts["tp"] + counts["tn"]) / denominator,
            "quality": {k: b.bootstrap_mean([r[k] for r in cases]) for k in ("precision", "recall", "f1")},
            "cases": cases}


def audit_model(folder, model, pools, labels, baseline_dir):
    from deployment_probe import native_question_request, native_noul_result
    from launch_reviewed import MODELS as deployment_models
    family, _, served = deployment_models[model]
    rows = b.read_jsonl(folder / "requests.jsonl")
    intents = b.read_jsonl(folder / "intents.jsonl")
    expected = jobs(pools)
    assert len(rows) == len(intents) == len(expected)
    assert len({r["call_id"] for r in rows}) == len(rows)
    baseline = [r for r in b.read_jsonl(baseline_dir / model / "requests.jsonl") if r["phase"] == "validation"]
    old = {(r["case_id"], r["fragment_id"]): r for r in baseline}
    assert len(old) == len(baseline) == len(expected)
    last_end = 0
    for row, intent, (pool, candidate) in zip(rows, intents, expected):
        identity = {"model": model, "phase": "validation", "arm": "noul", "stage": "refine",
                    "case_id": pool["case_id"], "fragment_id": candidate["fragment_id"]}
        state = p.pair_state(pool, candidate)
        path, body = native_question_request(family, payload(served, state))
        assert all(row[k] == v for k, v in identity.items())
        assert row["call_id"] == intent["call_id"] == b.digest(identity)
        assert row["request_sha256"] == intent["request_sha256"] == b.digest(body)
        assert row["input_sha256"] == intent["input_sha256"] == b.digest(state)
        assert row["path"] == path and row["status"] == "valid" and row["calls"] == 1
        assert row["inflight_at_start"] == 1 and row["started_seconds"] >= last_end
        assert row["started_seconds"] <= row["http_started_seconds"] <= row["ended_seconds"]
        last_end = row["ended_seconds"]
        original = old[(pool["case_id"], candidate["fragment_id"])]
        assert original["status"] == "valid" and original["input_sha256"] == row["input_sha256"]
        decoded = native_noul_result(family, payload(served, state), row["native_response"])
        assert all(row[k] == decoded[k] for k in ("prediction", "probabilities", "readout", "noul", "native_choice"))
    new_quality, old_quality = quality(pools, labels, rows), quality(pools, labels, baseline)
    delta = {k: b.bootstrap_mean([a[k] - z[k] for a, z in zip(new_quality["cases"], old_quality["cases"])])
             for k in ("precision", "recall", "f1")}
    return {"model": model, "passed": True, "noul": new_quality, "choice": old_quality,
            "paired_delta": delta, "flips": sum(r["prediction"] != old[(r["case_id"], r["fragment_id"])]["prediction"] for r in rows),
            "readouts": dict(Counter(r["readout"] for r in rows)), "requests": p.request_stats(rows),
            "historical_choice_requests": p.request_stats(baseline),
            "input_and_request_hashes_verified": True, "native_readout_recomputed": True,
            "unique_calls_and_nonoverlap": True,
            "evidence_sha256": {f.name: p.sha(f) for f in folder.iterdir() if f.is_file()}}


def load_dataset(args):
    protocol = json.loads((args.dataset / "protocol.json").read_text())
    assert p.sha(args.dataset / "protocol.json") == "b4b03b04cac155e27c38c984b24d2ac31b71d89f2b8c8d05d35fa27f54b78a40"
    for name, expected in protocol["files"].items():
        assert p.sha(args.dataset / name) == expected
    pools = b.read_jsonl(args.dataset / "validation.inputs.private.jsonl")
    labels = b.read_jsonl(args.dataset / "validation.labels.private.jsonl")
    p.validate_pools(pools, labels)
    assert len(pools) == 20 and len(jobs(pools)) == 333
    assert sum(len(r["positive_ids"]) for r in labels) == 116
    return pools, labels


def run(args):
    import httpx
    from deployment_probe import native_question_request, native_noul_result
    from launch_reviewed import MODELS as deployment_models, launch, read_key
    os.umask(0o077)
    pools, labels = load_dataset(args)
    preserved = json.loads(args.preservation.read_text())
    p.check_artifacts(preserved)
    sources = {str(f.resolve()): p.sha(f) for f in (
        Path(__file__), Path(p.__file__), Path(b.__file__), Path(p.__file__).with_name("decision_capacity_preflight.py"),
        *args.service_source.glob("*.py"), args.service_source / "nginx.native.conf.template")}
    support = json.loads(args.support.read_text())
    assert set(support["models"]) == set(MODELS)
    p.check_artifacts(support["source_sha256"])
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(exist_ok=False)
        p.save(args.output / "protocol.json", {"at": p.now(), "models": MODELS, "questions": QUESTIONS,
            "source_sha256": sources, "support_sha256": p.sha(args.support),
            "preservation_sha256": p.sha(args.preservation), "dataset_protocol_sha256": p.sha(args.dataset / "protocol.json"),
            "http_concurrency": 1, "timeout": 300, "startup_timeout": 2400, "retries": 0,
            "planned_validation_pairs_per_model": 333, "extra_synthetic_inference": 0,
            "readout": "native category/boolean, otherwise true iff P(true)>P(false); false on tie; no calibration",
            "choice_baseline": str(args.baseline), "formal_inference": False})
        state = {"pid": os.getpid(), "completed": []}

        def status(stage, **extra):
            state.update(stage=stage, at=p.now(), **extra)
            b.write_json(args.output / "status.json", state)
            print(json.dumps(state), flush=True)

        try:
            key = read_key(args.env_file)
            with httpx.Client(transport=httpx.HTTPTransport(retries=0), trust_env=False, follow_redirects=False, timeout=300) as http:
                for model in MODELS:
                    assert not (args.output / "STOP").exists()
                    p.check_artifacts(preserved)
                    p.check_artifacts(sources)
                    p.check_artifacts(support["source_sha256"])
                    folder = args.output / model
                    folder.mkdir()
                    status("loading", model=model, completed_pairs=0)
                    launch(SimpleNamespace(root=args.runtime, model=model, env_file=args.env_file, native_port=30888,
                        public_port=30889, seqs=32, tokens=32768, enforce_eager=False,
                        laya_max_concurrent=32 if model == "laya-multilingual" else None))
                    deployed = json.loads((args.runtime / "acceptance-20261015/evidence" / (model + ".deployment.json")).read_text())
                    p.save(folder / "deployment.json", deployed)
                    original = json.loads((args.baseline / model / "deployment.json").read_text())
                    for field in ("model", "served_model", "family", "image_id", "tp", "cpu_budget", "host_memory_limit_gib", "command", "scheduling", "env"):
                        assert deployed[field] == original[field], "Deployment mismatch: " + field
                    p.save(folder / "auth.json", p.readiness(http, "http://127.0.0.1:30889", key, 2400, "decision-acceptance-model"))
                    identity = p.container_identity()
                    p.save(folder / "container.json", identity)
                    family, _, served = deployment_models[model]
                    transport = p.SerialHTTP(http, key, folder, 300, args.output / "STOP")
                    try:
                        count = 0
                        for pool in pools:
                            status("validation", completed_pairs=count, case_id=pool["case_id"])
                            for candidate in pool["candidates"]:
                                state_input = p.pair_state(pool, candidate)
                                body = payload(served, state_input)
                                row = transport.call({"model": model, "phase": "validation", "arm": "noul", "stage": "refine",
                                    "case_id": pool["case_id"], "fragment_id": candidate["fragment_id"]}, "http://127.0.0.1:30889",
                                    lambda: native_question_request(family, body),
                                    lambda raw: native_noul_result(family, body, raw), b.digest(state_input))
                                assert row["status"] == "valid", "Native noul response invalid; no automatic retry"
                                count += 1
                            assert p.container_identity() == identity
                    finally:
                        transport.close()
                    p.save(folder / "audit.json", audit_model(folder, model, pools, labels, args.baseline))
                    state["completed"].append(model)
                    status("model_complete", completed_pairs=count, case_id=None)
            p.check_artifacts(preserved)
            p.check_artifacts(sources)
            p.save(args.output / "preservation-final.json", {"unchanged": True, "files": len(preserved)})
            status("complete", model=None)
        except BaseException as error:
            status("failed", error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("service-source", "dataset", "baseline", "runtime", "output", "env-file", "lock", "preservation", "support"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.service_source.resolve(strict=True)))
    run(args)
