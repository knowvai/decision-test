"""Verify published files and independently rescore both main experiments offline."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def jsonl(raw):
    assert not raw or raw.endswith(b"\n"), "Incomplete JSONL record"
    return [json.loads(line) for line in raw.split(b"\n") if line]


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def verify(root):
    manifest = load_json(root / "manifests/publication.json")
    by_archive = defaultdict(dict)
    for row in manifest["files"]:
        if "archive" in row:
            by_archive[row["archive"]][row["path"]] = row
        elif "destination" in row:
            raw = (root / row["destination"]).read_bytes()
            if row["gzip"]:
                raw = gzip.decompress(raw)
            assert sha(raw) == row["published_sha256"], row["destination"]
    formal, summaries, label_rows = {}, {}, None
    evidence_requests, evidence_chains, evidence_audits, evidence_labels = {}, {}, {}, {}
    coverage = None
    secret_hits = []
    secret = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|Bearer [A-Za-z0-9_-]{32,})")
    for arc in manifest["archives"]:
        path = root / arc["path"]
        assert sha(path.read_bytes()) == arc["sha256"], path
        expected = dict(by_archive[arc["path"]])
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                assert member.isfile() and not Path(member.name).is_absolute()
                assert ".." not in Path(member.name).parts
                entry = expected.pop(member.name)
                raw = tar.extractfile(member).read()
                assert len(raw) == entry["published_bytes"] and sha(raw) == entry["published_sha256"], member.name
                if secret.search(raw):
                    secret_hits.append(arc["path"] + ":" + member.name)
                source, name = arc["extract_under"], member.name
                parts = Path(name).parts
                if source == "classification-20261015":
                    if name == "dataset/labels.private.jsonl":
                        label_rows = [r for r in jsonl(raw) if r["task"] == "classification"]
                    if len(parts) == 3 and parts[0].startswith("run-"):
                        key = (parts[0], parts[1])
                        if parts[-1] == "responses.private.jsonl":
                            formal[key] = jsonl(raw)
                        if parts[-1] == "summary.json":
                            summaries[key] = json.loads(raw)
                elif source == "evidence-20261018":
                    if name == "run-001/common-coverage.json":
                        coverage = json.loads(raw)
                    if name.startswith("dataset-v1/") and name.endswith(".labels.private.jsonl"):
                        evidence_labels[parts[-1].split(".")[0]] = jsonl(raw)
                    if len(parts) == 3 and parts[0] == "run-001":
                        if parts[-1] == "requests.jsonl":
                            evidence_requests[parts[1]] = jsonl(raw)
                        elif parts[-1] == "chains.jsonl":
                            evidence_chains[parts[1]] = jsonl(raw)
                        elif parts[-1] == "audit.json":
                            evidence_audits[parts[1]] = json.loads(raw)
        assert not expected, arc["path"]
    for path in root.rglob("*"):
        if path.is_file() and not {".git", ".venv", "__pycache__"}.intersection(path.parts) and path.suffix not in {".gz", ".png"}:
            if secret.search(path.read_bytes()):
                secret_hits.append(str(path.relative_to(root)))
    assert not secret_hits, {"potential_secrets_paths_only": secret_hits}
    assert label_rows and len(label_rows) == 2974
    gold = {r["case_id"]: r["gold"] for r in label_rows}
    domains = sorted(set(gold.values()))
    assert len(domains) == 18 and len(formal) == len(summaries) == 16
    classification = []
    for key, rows in sorted(formal.items()):
        assert len(rows) == 2974 and len({r["case_id"] for r in rows}) == 2974
        assert set(gold) == {r["case_id"] for r in rows}
        assert all(r["status"] == "valid" for r in rows)
        for row in rows:
            if "calls" in row:
                assert len(row["calls"]) == 1
            else:
                assert len(row["request_sha256"]) == 64 and row["native_response"]
        correct = sum(r["prediction"] == gold[r["case_id"]] for r in rows)
        f1s = []
        for domain in domains:
            tp = sum(r["prediction"] == domain == gold[r["case_id"]] for r in rows)
            predicted = sum(r["prediction"] == domain for r in rows)
            actual = sum(v == domain for v in gold.values())
            f1s.append(2 * tp / (predicted + actual))
        summary = summaries[key]
        close(correct / len(rows), summary["accuracy"])
        close(statistics.mean(f1s), summary["macro_f1"])
        close(statistics.mean(r["elapsed_ms"] for r in rows), summary["successful_task_latency"]["mean_ms"])
        close(percentile([r["elapsed_ms"] for r in rows], .99), summary["successful_task_latency"]["p99_ms"])
        classification.append({"run": key[0], "model": key[1], "records": len(rows),
                               "correct": correct, "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]})
    assert len(evidence_requests) == len(evidence_chains) == len(evidence_audits) == 15
    intervals, call_ids, stage_counts = [], set(), Counter()
    request_total = chain_total = condition_total = 0
    expected_request_phases = Counter(engineering=8652, validation=15148, formal=145740, no_evidence=26040)
    request_phases = Counter()
    evidence = []
    published = load_json(root / "docs/research/2026-10-19-answer-evidence-results.json.gz")
    for model, rows in sorted(evidence_requests.items()):
        audit = evidence_audits[model]
        assert audit["passed"] and audit["calls"] == len(rows)
        for row in rows:
            assert row["status"] == "valid" and row["http_status"] == 200 and row["calls"] == 1
            identity = (model, row["call_id"])
            assert identity not in call_ids
            call_ids.add(identity)
            intervals.append((row["started_seconds"], row["ended_seconds"]))
            request_phases[row["phase"]] += 1
            stage_counts[(model, row["phase"], row["arm"], row["stage"])] += 1
        request_total += len(rows)
        chains = evidence_chains[model]
        assert all(r["valid"] is True for r in chains)
        assert len({(r["phase"], r["arm"], r["case_id"]) for r in chains}) == len(chains)
        chain_total += len(chains)
        formal_chains = defaultdict(list)
        selected_by_requests = defaultdict(set)
        for row in rows:
            if row["phase"] == "formal" and row["stage"] == "refine" and row["prediction"] == "relevant":
                selected_by_requests[(row["arm"], row["case_id"])].add(row["fragment_id"])
        for chain in chains:
            if chain["phase"] == "formal":
                formal_chains[chain["arm"]].append(chain)
        for arm, selected in sorted(formal_chains.items()):
            condition_total += 1
            assert len(selected) == 197
            excluded = {r["case_id"] for r in coverage["excluded_pools"] if r["phase"] == "formal"}
            labels = {r["case_id"]: r for r in evidence_labels["formal"] if r["case_id"] not in excluded}
            assert {r["case_id"] for r in selected} == set(labels)
            counts, quality = Counter(), []
            for chain in selected:
                label = labels[chain["case_id"]]
                positives, kept = set(label["positive_ids"]), set(chain["selected_ids"])
                assert kept == selected_by_requests[(arm, chain["case_id"])]
                candidates = set(label["grades"])
                assert len(kept) == len(chain["selected_ids"]) and kept <= candidates
                tp, fp, fn = len(kept & positives), len(kept - positives), len(positives - kept)
                counts.update(tp=tp, fp=fp, fn=fn, tn=len(candidates) - tp - fp - fn)
                if positives:
                    quality.append((tp / len(kept) if kept else 0, tp / len(positives),
                                    2 * tp / (2 * tp + fp + fn)))
            assert len(quality) == 183 and sum(counts.values()) == 3054
            result = published["models"][model]["phases"]["formal"][arm]
            assert dict(counts) == {k: result["quality"]["counts"].get(k, 0) for k in ("tp", "fp", "fn", "tn")}
            assert result["quality"]["counts"]["planned"] == 3054
            for i, metric in enumerate(("precision", "recall", "f1")):
                close(statistics.mean(r[i] for r in quality), result["quality"]["positive_pool_macro"][metric]["point"])
            close(statistics.mean(r["wall_ms"] for r in selected), result["pool_wall_ms"]["mean_ms"])
            close(percentile([r["wall_ms"] for r in selected], .99), result["pool_wall_ms"]["p99_ms"])
            refine = [r["elapsed_ms"] for r in rows if r["phase"] == "formal" and r["arm"] == arm and r["stage"] == "refine"]
            close(statistics.mean(refine), result["stages"]["refine"]["all_attempt_latency_ms"]["mean_ms"])
            close(percentile(refine, .99), result["stages"]["refine"]["all_attempt_latency_ms"]["p99_ms"])
            assert stage_counts[(model, "formal", arm, "refine")] == (3054 if arm.endswith(".direct") else 1954)
            assert stage_counts[(model, "formal", arm, "rerank")] == (0 if arm.endswith(".direct") else 197)
            evidence.append({"model": model, "condition": arm, "questions": 197, "fragments": 3054,
                             "counts": dict(counts), "f1": result["quality"]["positive_pool_macro"]["f1"]["point"]})
    assert request_total == 195580 and chain_total == 14952 and condition_total == 56
    assert request_phases == expected_request_phases
    ordered = sorted(intervals)
    assert all(end <= following for (_, end), (following, _) in zip(ordered, ordered[1:])), "Overlapping requests"
    return {"passed": True, "network_calls": 0, "model_inference_calls": 0,
            "publication_files": len(manifest["files"]), "archives": len(manifest["archives"]),
            "classification_records": sum(r["records"] for r in classification), "classification": classification,
            "evidence_requests": request_total, "evidence_chains": chain_total,
            "evidence_request_phases": dict(request_phases), "evidence_conditions": evidence,
            "global_serial_intervals_checked": len(ordered),
            "scope": "archive integrity, primary record coverage and quality/wall point estimates; does not rerun bootstrap or host audit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"classification", "evidence_conditions"}}))
