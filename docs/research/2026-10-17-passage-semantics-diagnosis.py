"""Read-only result projection and offline label/score diagnosis; no inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def snapshot(args):
    # Only these immutable result fields leave the evaluation host; no inputs/responses.
    source = '''import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
out={"models":{},"sha256":{}}
for folder in sorted(p.iterdir()):
 if not folder.is_dir() or not (folder/"requests.jsonl").exists():continue
 keys=("phase","arm","stage","case_id","fragment_id","status","prediction","probabilities")
 before={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ("requests.jsonl","threshold.json","chains.jsonl")}
 records=[json.loads(l) for l in (folder/"requests.jsonl").read_text().splitlines()]
 assert len(records)==7080 and all(r["status"]=="valid" for r in records)
 chains=[json.loads(l) for l in (folder/"chains.jsonl").read_text().splitlines()]
 out["models"][folder.name]={"rule":json.loads((folder/"threshold.json").read_text()),
  "records":[{k:r.get(k) for k in keys} for r in records if r["stage"]=="refine" and (r["phase"]=="validation" or (r["phase"]=="formal" and r["arm"] in ("B1","B2")))],
  "chains":[{k:r[k] for k in ("case_id","arm","selected_ids")} for r in chains if r["arm"] in ("B1","B2")]}
 assert before=={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in before}
 out["sha256"][folder.name]=before
assert len(out["models"])==15
print(json.dumps(out))
'''
    raw = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
         "python3 - " + shlex.quote(args.remote_run)], input=source.encode(), timeout=180)
    result = json.loads(raw)
    with args.output.open("x") as f:
        json.dump(result, f)


def analyze(args):
    def forbid_network(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Offline analysis must not connect to any service")

    sys.addaudithook(forbid_network)
    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    data = args.inputs / "dataset-six-mask-v1"
    projection = read(args.inputs / "projection.json")
    source_path = args.inputs / "t2reranking-dev.parquet"
    assert sha(source_path) == "ceaf1ba1a4777dbb7aecad86fb8aa9cae7a678af6d711d5529c238e89bb6adff"
    assert sha(args.inputs / "corpus.parquet") == "d4a207a9277eb1cf31f68ab6a7236cbf3e8fbf9493e0e2b577a34660929fcd17"
    assert sha(data / "protocol.json") == "b4b03b04cac155e27c38c984b24d2ac31b71d89f2b8c8d05d35fa27f54b78a40"
    for name, expected in read(data / "protocol.json")["files"].items():
        assert sha(data / name) == expected
    source = pq.read_table(source_path).to_pylist()
    query_ids = defaultdict(list)
    for line in (args.inputs / "queries.dev.tsv").read_text().splitlines()[1:]:
        qid, query = line.split("\t", 1)
        query_ids[query].append(qid)
    qrels = defaultdict(dict)
    for line in (args.inputs / "qrels.dev.tsv").read_text().splitlines()[1:]:
        qid, _, pid, grade = line.split()
        assert pid not in qrels[qid]
        qrels[qid][pid] = int(grade)
    corpus = defaultdict(set)
    for row in pq.read_table(args.inputs / "corpus.parquet").to_pylist():
        corpus[hashlib.sha256(row["text"].encode()).hexdigest()].add(row["id"])
    count_matches, exact, mismatches = Counter(), Counter(), []
    for index, row in enumerate(source):
        qids = query_ids[row["query"]]
        assert len(qids) == 1
        rel = qrels[qids[0]]
        grades = Counter(rel.values())
        for cutoff in (1, 2, 3):
            if len(row["positive"]) == sum(v for k, v in grades.items() if k >= cutoff) and len(row["negative"]) == sum(v for k, v in grades.items() if k < cutoff):
                count_matches[str(cutoff)] += 1
        if len(row["positive"]) != sum(v for k, v in grades.items() if k >= 1) or len(row["negative"]) != grades[0]:
            mismatches.append({"source_row": index, "qrels_grades": dict(grades),
                               "positive": len(row["positive"]), "negative": len(row["negative"])})
        for label in ("positive", "negative"):
            for text in row[label]:
                ids = corpus[hashlib.sha256(text.encode()).hexdigest()]
                values = {rel[pid] for pid in ids if pid in rel}
                key = str(next(iter(values))) if len(values) == 1 else "unmatched" if not values else "ambiguous"
                exact[label + ":" + key] += 1

    report = {"kind": "offline_diagnosis_not_new_evaluation", "no_inference": True,
              "label_provenance": {"source_pools": len(source), "count_matches_by_positive_grade_cutoff": dict(count_matches),
                                   "count_mismatches_at_cutoff1": mismatches, "exact_text_and_query_grade_matches": dict(exact),
                                   "caveat": "Counts alone do not prove per-passage grade conversion; exact matches use a partial retrieval corpus."},
              "files_sha256": {name: sha(args.inputs / name) for name in
                               ("projection.json", "t2reranking-dev.parquet", "corpus.parquet", "queries.dev.tsv", "qrels.dev.tsv")},
              "result_sha256": projection["sha256"], "splits": {}, "models": {}}
    pools, labels = {}, {}
    for phase in ("validation", "formal"):
        pools[phase] = rows(data / (phase + ".inputs.private.jsonl"))
        labels[phase] = {r["case_id"]: set(r["positive_ids"]) for r in rows(data / (phase + ".labels.private.jsonl"))}
        missing = Counter()
        for pool in pools[phase]:
            original = source[int(pool["case_id"].split(":")[1])]
            assert pool["query"] == original["query"]
            positive = {hashlib.sha256(s.encode()).hexdigest() for s in original["positive"]}
            negative = {hashlib.sha256(s.encode()).hexdigest() for s in original["negative"]}
            supplied = {c["fragment_id"] for c in pool["candidates"]}
            assert supplied <= positive | negative
            assert labels[phase][pool["case_id"]] == positive & supplied
            assert all(c["fragment_id"] == hashlib.sha256(c["content"].encode()).hexdigest() for c in pool["candidates"])
            missing["positive"] += len(positive - supplied)
            missing["negative"] += len(negative - supplied)
        assert sum(missing.values()) == (6 if phase == "formal" else 0)
        report["splits"][phase] = {"pools": len(pools[phase]), "candidates": sum(len(p["candidates"]) for p in pools[phase]),
                                   "positive": sum(len(g) for g in labels[phase].values()),
                                   "excluded_vs_source": dict(missing), "exact_local_labels_match_source": True}

    def metrics(phase, chosen):
        per_pool, total = [], Counter()
        for pool in pools[phase]:
            ids = [c["fragment_id"] for c in pool["candidates"]]
            gold = labels[phase][pool["case_id"]]
            pred = chosen[pool["case_id"]]
            truth = [i in gold for i in ids]
            guesses = [i in pred for i in ids]
            precision, recall, f1, _ = precision_recall_fscore_support(truth, guesses, average="binary", zero_division=0)
            per_pool.append([precision, recall, f1])
            total.update(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred), tn=len(set(ids) - (gold | pred)), empty_pools=int(not pred))
        out = dict(zip(("macro_precision", "macro_recall", "macro_f1"), np.mean(per_pool, axis=0).tolist()))
        out.update(total)
        out.update(selected_per_pool=(total["tp"] + total["fp"]) / len(pools[phase]),
                   negative_accept_rate=total["fp"] / (total["fp"] + total["tn"]))
        return out

    for phase in pools:
        report["splits"][phase]["select_all"] = metrics(phase, {p["case_id"]: {c["fragment_id"] for c in p["candidates"]} for p in pools[phase]})
    for model, value in projection["models"].items():
        model_out = {"frozen_rule": value["rule"], "phases": {}}
        for phase in pools:
            group = defaultdict(list)
            for r in value["records"]:
                if r["phase"] == phase:
                    group[r["case_id"]].append(r)
            for pool in pools[phase]:
                assert len(group[pool["case_id"]]) == len(pool["candidates"])
                assert {r["fragment_id"] for r in group[pool["case_id"]]} == {c["fragment_id"] for c in pool["candidates"]}
            continuous = value["rule"]["kind"] == "threshold"

            def select(threshold=None):
                return {case: {r["fragment_id"] for r in rs if (r["probabilities"]["relevant"] >= threshold if continuous else r["prediction"] == "relevant")} for case, rs in group.items()}

            chosen = select(value["rule"].get("threshold"))
            result = metrics(phase, chosen)
            if phase == "formal":
                assert chosen == {c["case_id"]: set(c["selected_ids"]) for c in value["chains"]}
            if continuous:
                y, scores, aucs, aps = [], [], [], []
                for case, rs in group.items():
                    yy = [r["fragment_id"] in labels[phase][case] for r in rs]
                    ss = [r["probabilities"]["relevant"] for r in rs]
                    y.extend(yy)
                    scores.extend(ss)
                    aucs.append(roc_auc_score(yy, ss))
                    aps.append(average_precision_score(yy, ss))
                result["score_discrimination"] = {"pooled_roc_auc": roc_auc_score(y, scores), "macro_pool_roc_auc": float(np.mean(aucs)),
                                                  "macro_pool_average_precision": float(np.mean(aps)),
                                                  "score_quantiles_p10_p50_p90": {label: np.quantile([s for yy, s in zip(y, scores) if yy == positive], [.1, .5, .9]).tolist() for label, positive in (("positive", True), ("negative", False))}}
                if phase == "validation":
                    result["diagnostic_threshold_0_5_only_validation_not_deployed"] = metrics(phase, select(.5))
            model_out["phases"][phase] = result
        report["models"][model] = model_out
    completed_audit = read(args.audit)
    for model, result in report["models"].items():
        audited = completed_audit["models"][model]
        assert result["frozen_rule"] == audited["threshold"]
        for name, value in report["result_sha256"][model].items():
            assert value == audited["audit"]["artifact_sha256"][name]
        arm = next(k for k in audited["all_pools"]["arms"] if k in ("B1", "B2"))
        for metric in ("precision", "recall", "f1"):
            assert abs(result["phases"]["formal"]["macro_" + metric] - audited["all_pools"]["arms"][arm]["quality"][metric]["point"]) < 1e-12
    report["verification"] = {"prior_audit_sha256": sha(args.audit), "result_hashes_match_prior_audit": True,
                              "all_frozen_rules_and_formal_metrics_match_prior_audit": True,
                              "analysis_network_connections_forbidden": True, "script_sha256": sha(__file__),
                              "numpy_version": np.__version__, "pyarrow_version": __import__("pyarrow").__version__,
                              "sklearn_version": __import__("sklearn").__version__}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"models": len(report["models"]), "source_pools": len(source), "local_labels_exact": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--host", required=True)
    snap.add_argument("--remote-run", required=True)
    snap.add_argument("--output", type=Path, required=True)
    offline = sub.add_parser("analyze")
    offline.add_argument("--inputs", type=Path, required=True)
    offline.add_argument("--audit", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot(args) if args.mode == "snapshot" else analyze(args)
