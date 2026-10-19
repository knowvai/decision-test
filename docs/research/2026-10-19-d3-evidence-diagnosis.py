"""Offline diagnosis of frozen pilot records; no inference or label rewriting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def ranking_metrics(truth, scores):
    positive = [s for y, s in zip(truth, scores, strict=True) if y]
    negative = [s for y, s in zip(truth, scores, strict=True) if not y]
    auc = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))
    groups = defaultdict(list)
    for y, s in zip(truth, scores, strict=True):
        groups[s].append(y)
    seen = found = 0
    ap = 0.0
    # Group equal scores before measuring precision; ties share one threshold.
    for s in sorted(groups, reverse=True):
        seen += len(groups[s])
        new = sum(groups[s])
        found += new
        ap += new / len(positive) * found / seen
    return auc, ap


def main(args):
    def offline(event, _args):
        if event in {"socket.connect", "subprocess.Popen", "os.system"}:
            raise RuntimeError("Diagnosis is offline")

    sys.addaudithook(offline)
    source = Path(__file__).with_name("2026-10-19-d3-tuning.py")
    spec = importlib.util.spec_from_file_location("d3_tuning", source)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    from deployment_probe import native_choice_result

    data = args.inputs / "dataset-v2"
    audit = json.loads(args.audit.read_text())
    assert audit["passed"] and t.sha(data / "protocol.json") == audit["protocol_sha256"]
    protocol = t.read_data(data)
    labels = {r["case_id"]: r for r in t.read_rows(data / "evidence-holdout.labels.private.jsonl")}
    jobs = t.read_rows(data / "evidence-holdout.private.jsonl")
    keys = [(j["case_id"], j["fragment_id"]) for j in jobs]
    assert len(keys) == len(set(keys)) == 643
    sources = {str(data / name): value for name, value in protocol["files"].items()}
    sources[str(args.audit)] = t.sha(args.audit)

    # Rejoin the original public grade/query files, not a binary label derivative.
    for name in ("queries.dev.tsv", "qrels.dev.tsv"):
        path = args.original / name
        assert t.sha(path) == t.evidence.SOURCE_HASHES[name]
        sources[str(path)] = t.sha(path)
    queries = {r["qid"]: r["text"] for r in t.evidence.read_tsv(args.original / "queries.dev.tsv")}
    qrels = {(r["qid"], r["pid"]): int(r["rel"])
             for r in t.evidence.read_tsv(args.original / "qrels.dev.tsv")}
    wanted = {j["fragment_id"].split(":")[1] for j in jobs}
    collection = args.original / "collection.verified.tsv"
    assert t.sha(collection) == t.evidence.SOURCE_HASHES["collection.tsv"]
    sources[str(collection)] = t.sha(collection)
    texts = {r["pid"]: r["text"] for r in t.evidence.read_tsv(collection) if r["pid"] in wanted}
    truth, grades = [], []
    for job in jobs:
        cid, fid = job["case_id"], job["fragment_id"]
        grade = qrels[cid.split(":")[1], fid.split(":")[1]]
        assert labels[cid]["grades"][fid] == grade
        assert job["state"] == {"query": queries[cid.split(":")[1]], "document": texts[fid.split(":")[1]]}
        assert job["input_sha256"] == t.benchmark.digest(job["state"])
        assert job["gold"] == ("relevant" if grade >= 2 else "irrelevant")
        truth.append(grade >= 2)
        grades.append(grade)

    predictions, result = {}, {}
    for model in t.MODELS:
        path = args.inputs / "quality" / model / "evidence-holdout-baseline.jsonl"
        assert t.sha(path) == audit["artifact_sha256"][f"{model}/quality/{path.name}"]
        sources[str(path)] = t.sha(path)
        rows = t.read_rows(path)
        assert [(r["case_id"], r["fragment_id"]) for r in rows] == keys
        assert all(r["status"] == "valid" and r["http_status"] == 200 and r["calls"] == 1 for r in rows)
        for job, row in zip(jobs, rows, strict=True):
            _, body = t.wire(model, job, t.variants()["evidence"]["baseline"])
            body["options"] = {"return_meta": True}
            assert row["input_sha256"] == job["input_sha256"]
            assert row["request_sha256"] == t.benchmark.digest(body)
            choice, probability = native_choice_result("worker", body, row["response"])
            assert (choice, probability) == (row["prediction"], row["probabilities"])
            assert row["prediction"] == max(row["probabilities"], key=row["probabilities"].get)
        pred = [r["prediction"] == "relevant" for r in rows]
        predictions[model] = pred
        probabilities = [r["probabilities"]["relevant"] for r in rows]
        auc, ap = ranking_metrics(truth, probabilities)
        counts = Counter("tp" if y and p else "fn" if y else "fp" if p else "tn"
                         for y, p in zip(truth, pred, strict=True))
        result[model] = {
            "counts": dict(counts), "accuracy": (counts["tp"] + counts["tn"]) / len(rows),
            "roc_auc": auc, "average_precision": ap,
            "grades": {str(g): {"n": grades.count(g), "kept": sum(p for grade, p in zip(grades, pred) if grade == g)} for g in range(4)},
            "high_confidence_fp_ge_0_9": sum(not y and p >= .9 for y, p in zip(truth, probabilities)),
            "high_confidence_fn_le_0_1": sum(y and p <= .1 for y, p in zip(truth, probabilities)),
        }

    strata = defaultdict(list)
    mini, flash = (predictions[m] for m in t.MODELS)
    for i, (y, a, b) in enumerate(zip(truth, mini, flash, strict=True)):
        kind = "disagree" if a != b else "tp" if y and a else "fn" if y else "fp" if a else "tn"
        strata[kind].append(i)
    seed = "evidence-diagnosis-v1"
    rank = lambda i: t.benchmark.digest([bytes.fromhex("6b6e6f77762d65766964656e63652d646961676e6f7369732d7631").decode("utf-8"), *keys[i]])
    selected = []
    for kind, n in {"fp": 8, "fn": 8, "disagree": 4, "tp": 2, "tn": 2}.items():
        selected.extend((i, kind) for i in sorted(strata[kind], key=rank)[:n])
    selected.sort(key=lambda v: t.benchmark.digest([bytes.fromhex("6b6e6f77762d65766964656e63652d646961676e6f7369732d7631").decode("utf-8"), "blind", *keys[v[0]]]))
    blind, answer_key = [], []
    for index, (i, kind) in enumerate(selected, 1):
        rid = f"R{index:02d}"
        job = jobs[i]
        blind.append({"review_id": rid, "state": job["state"]})
        answer_key.append({"review_id": rid, "case_id": keys[i][0], "fragment_id": keys[i][1],
                           "input_sha256": job["input_sha256"], "stratum": kind, "grade": grades[i],
                           "predictions": {m: predictions[m][i] for m in t.MODELS}})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("blind.private.jsonl", blind), ("key.jsonl", answer_key)):
        with (args.output / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "kind": "offline_diagnosis_not_new_benchmark", "inference_calls": 0,
        "source_sha256": sources, "script_sha256": t.sha(Path(__file__)),
        "verified_pairs": len(jobs), "verified_pools": len(labels),
        "original_grade_query_fulltext_matches": len(jobs), "grades": dict(Counter(grades)),
        "all_reject_accuracy": 1 - sum(truth) / len(truth), "positive_prevalence": sum(truth) / len(truth),
        "models": result, "joint_outcomes": {k: len(v) for k, v in strata.items()},
        "sample_seed": seed, "sample_counts": dict(Counter(k for _, k in selected)),
        "blind_sha256": t.sha(args.output / "blind.private.jsonl"), "key_sha256": t.sha(args.output / "key.jsonl"),
        "limits": ["Error-enriched sample; cannot estimate population label-error rate.",
                   "Agent review is not independent human adjudication.",
                   "No GPU or official-reference numeric parity test in this diagnosis.",
                   "No threshold selection or old score replacement."],
    }
    with (args.output / "statistics.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    assert all(t.sha(Path(p)) == value for p, value in sources.items())
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
