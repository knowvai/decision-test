#!/usr/bin/env python3
"""Recalculate the published E2E tables without network access or model calls.

This checks arithmetic and record consistency, not the semantic correctness of
the AI review or the original servers' hardware/configuration.
"""

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = (
    "router", "decomposition", "retrieval", "selection", "sufficiency",
    "generation_other", "http_transport",
)


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (
        actual, expected
    )


def relative(baseline, with_d3):
    return (with_d3 / baseline - 1) * 100


def main():
    data = read_json("results.json")
    summary = read_json("summary.json")
    records = data["records"]
    questions = {q["id"]: q for q in data["questions"]}
    passages = {p["id"]: p for p in data["passages"]}
    deployments = {d["id"]: d for d in data["deployments"]}
    assert len(questions) == 30 and len(records) == 240
    assert set(deployments) == set(summary) == {"local", "aliyun"}
    assert len({r["id"] for r in records}) == len(records)
    assert Counter(
        (r["deployment"], r["mode"], r["round"], r["question_id"])
        for r in records
    ) == Counter(
        (d, m, n, q) for d in deployments
        for m in ("baseline", "with_d3") for n in (1, 2) for q in questions
    )

    for p in passages.values():
        assert digest(p["content"]) == p["content_sha256"]
        assert p["id"] == "p_" + digest(
            p["source_document"] + "\0" + p["content"]
        )[:20]

    for r in records:
        assert digest(r["answer"]) == r["answer_sha256"]
        assert r["http_status"] == 200 and r["application_cache_hit"] is False
        assert all(ref["passage_id"] in passages for ref in r["references"])
        assert all(ref["review_verdict"] in ("useful", "irrelevant")
                   for ref in r["references"])
        assessment = r["assessment"]
        strict = (
            assessment["core_verdict"] == "correct"
            and assessment["grounded"]
            and not assessment["answer_issues"]
            and all(ref["review_verdict"] == "useful" for ref in r["references"])
        )
        assert strict == assessment["strict_pass"]
        calls = r["recorded_model_calls"]
        config = deployments[r["deployment"]]
        assert [c["index"] for c in calls] == list(range(1, len(calls) + 1))
        for c in calls:
            assert c["duration_ms"] >= 0
            if c["backend"] == "d3":
                assert r["mode"] == "with_d3"
                assert c["model"] == config["decision_model"]
                assert c["stage"] in ("router", "selection")
                assert c["decision_questions"] == (
                    1 if c["stage"] == "selection" else 2
                )
            else:
                assert c["backend"] == "main_llm"
                assert c["model"] == config["main_model"]
        assert r["observed_failed_model_calls"] == sum(
            c["success"] is False for c in calls
        )
        expected_fallback = any(c["fallback"] for c in calls) or (
            r["mode"] == "with_d3" and any(
                c["backend"] == "main_llm" and c["stage"] == "smart_router"
                for c in calls
            )
        )
        assert r["has_fallback"] == expected_fallback

        nodes = r["node_timings"]
        node_seconds = {
            node: sum(n["duration_ms"] for n in nodes if n["node"] == node) / 1000
            for node in ("smart_router", "decompose", "retrieve", "select")
        }
        sufficiency = sum(
            c["duration_ms"] for c in calls if c["phase"] == "final_sufficiency"
        ) / 1000
        calculated = {
            "router": node_seconds["smart_router"],
            "decomposition": node_seconds["decompose"],
            "retrieval": node_seconds["retrieve"],
            "selection": node_seconds["select"] - sufficiency,
            "sufficiency": sufficiency,
            "generation_other": r["server_ms"] / 1000 - sum(node_seconds.values()),
            "http_transport": (r["e2e_ms"] - r["server_ms"]) / 1000,
        }
        assert set(r["stage_seconds"]) == set(STAGES)
        for stage in STAGES:
            close(r["stage_seconds"][stage], calculated[stage])
        close(sum(calculated.values()), r["e2e_ms"] / 1000)
        assert r["retrieval_passes"] == sum(n["node"] == "retrieve" for n in nodes)

    for deployment in deployments:
        subset = [r for r in records if r["deployment"] == deployment]
        assert sorted(r["sequence_within_deployment"] for r in subset) == list(
            range(1, 121)
        )
        for question in questions:
            first_modes = []
            for round_number in (1, 2):
                pair = [r for r in subset if r["question_id"] == question
                        and r["round"] == round_number]
                first_modes.append(min(
                    pair, key=lambda r: r["sequence_within_deployment"]
                )["mode"])
            assert first_modes[0] != first_modes[1]

        means = {}
        for mode in ("baseline", "with_d3"):
            rows = [r for r in subset if r["mode"] == mode]
            s = summary[deployment]["modes"][mode]
            assert len(rows) == s["requests"] == 60
            means[mode] = statistics.mean(r["e2e_ms"] / 1000 for r in rows)
            close(means[mode], s["mean_e2e_s"])
            close(statistics.median(r["e2e_ms"] / 1000 for r in rows),
                  s["median_e2e_s"])
            for key, count in {
                "fallback_requests": sum(r["has_fallback"] for r in rows),
                "observed_failed_model_requests": sum(
                    r["observed_failed_model_calls"] > 0 for r in rows
                ),
                "core_expected": sum(
                    r["assessment"]["core_verdict"] == "correct" for r in rows
                ),
                "strict_pass": sum(r["assessment"]["strict_pass"] for r in rows),
                "requests_with_irrelevant_references": sum(any(
                    ref["review_verdict"] == "irrelevant" for ref in r["references"]
                ) for r in rows),
            }.items():
                assert count == s[key], (deployment, mode, key, count, s[key])

        close(relative(means["baseline"], means["with_d3"]),
              summary[deployment]["relative_mean_change_pct"])
        for stage in STAGES:
            means_by_mode = {
                mode: statistics.mean(r["stage_seconds"][stage] for r in subset
                                      if r["mode"] == mode)
                for mode in ("baseline", "with_d3")
            }
            s = summary[deployment]["stages"][stage]
            for mode, mean in means_by_mode.items():
                close(mean, s[mode + "_mean_s"])
            close(relative(means_by_mode["baseline"], means_by_mode["with_d3"]),
                  s["relative_change_pct"])
        for round_number in (1, 2):
            round_means = {
                mode: statistics.mean(r["e2e_ms"] / 1000 for r in subset
                                      if r["mode"] == mode
                                      and r["round"] == round_number)
                for mode in ("baseline", "with_d3")
            }
            s = summary[deployment]["rounds"][str(round_number)]
            close(round_means["baseline"], s["baseline_mean_s"])
            close(round_means["with_d3"], s["with_d3_mean_s"])
            close(relative(round_means["baseline"], round_means["with_d3"]),
                  s["relative_change_pct"])

    csv_rows = list(csv.DictReader(
        (ROOT / "requests.csv").open(encoding="utf-8-sig", newline="")
    ))
    assert len(csv_rows) == len(records)
    csv_by_id = {row["id"]: row for row in csv_rows}
    assert set(csv_by_id) == {r["id"] for r in records}
    for r in records:
        row = csv_by_id[r["id"]]
        close(float(row["e2e_s"]), r["e2e_ms"] / 1000)
        for stage in STAGES:
            close(float(row[stage]), r["stage_seconds"][stage])
        assert int(row["recorded_model_calls"]) == len(r["recorded_model_calls"])
        assert row["strict_pass"] == str(r["assessment"]["strict_pass"])

    for name, expected in read_json("sha256.json").items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    public_data = (ROOT / "results.json").read_text(encoding="utf-8")
    for pattern in (
        r"/Users/", r"/data/sdb/", r"/var/lib/",
        r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+(?:\.\d+){1,2}\b",
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        r"(?i)authorization|bearer\s|api_key|BEGIN .*LICENSE",
    ):
        assert not re.search(pattern, public_data), pattern

    for deployment, s in summary.items():
        print(
            f"{deployment}: {s['modes']['baseline']['mean_e2e_s']:.3f}s -> "
            f"{s['modes']['with_d3']['mean_e2e_s']:.3f}s "
            f"({s['relative_mean_change_pct']:+.1f}%), n=60 per mode"
        )
    print(f"PASS: {len(records)} requests, {len(questions)} questions, "
          f"{sum(len(r['recorded_model_calls']) for r in records)} recorded calls; "
          "hashes, timings, review-label counts and CSV agree.")


if __name__ == "__main__":
    main()
