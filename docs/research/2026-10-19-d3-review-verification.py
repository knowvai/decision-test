"""Verify blinded review joins and prompt rendering, without loading a model."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main(args):
    def offline(event, _args):
        if event in {"socket.connect", "subprocess.Popen", "os.system"}:
            raise RuntimeError("Verification is offline")

    sys.addaudithook(offline)
    sys.path.insert(0, str(args.reference / "deps"))
    from jinja2 import Environment

    repo = Path(__file__).resolve().parents[2]
    research = repo / "docs/research"
    review_path = research / "2026-10-19-d3-evidence-blind-review.json"
    reviews = read(review_path)
    statistics = read(args.diagnosis / "statistics.json")
    assert reviews["blind_sha256"] == statistics["blind_sha256"] == sha(args.diagnosis / "blind.private.jsonl")
    assert statistics["key_sha256"] == sha(args.diagnosis / "key.jsonl")
    key = {r["review_id"]: r for r in rows(args.diagnosis / "key.jsonl")}
    assert set(key) == {r["review_id"] for r in reviews["reviews"]}
    assert len(reviews["reviews"]) == len(key) == 24
    joined, counts = [], Counter()
    for row in reviews["reviews"]:
        source = key[row["review_id"]]
        actual = "yes" if source["grade"] >= 2 else "no"
        outcome = "ambiguous" if row["judgment"] == "ambiguous" else "agrees" if row["judgment"] == actual else "disagrees"
        counts[outcome] += 1
        joined.append({**source, **row, "relation_to_original_label": outcome})

    lock = read(repo / "deploy/docker/decision-eval/d3-model-lock.json")["models"][0]["files_sha256"]
    for name in ("d3_format.py", "chat_template.jinja"):
        assert sha(args.reference / name) == lock[name]
    old = module(args.reference / "d3_format.py", "official_d3_format")
    assert sha(args.reference / "d3_format_current.py") == "3badfbc70561bb3f5b8ec16a11d2c455b96ddb32caace0a8a54ca102eae1f3fa"
    current = module(args.reference / "d3_format_current.py", "current_d3_format")
    worker_path = args.original / "vllm_srun/families/decision3/prompt.py"
    tree = ast.parse(worker_path.read_text())
    # Only pure formatting functions run; package imports/tokenization are not used.
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    worker = {}
    exec(compile(tree, str(worker_path), "exec"), worker)
    t = module(research / "2026-10-19-d3-tuning.py", "d3_tuning")
    template = Environment(extensions=["jinja2.ext.loopcontrols"]).from_string((args.reference / "chat_template.jinja").read_text())
    states = [j["state"] for j in t.read_rows(args.inputs / "dataset-v2/evidence-holdout.private.jsonl")]
    checked = 0
    for state in states:
        for mode in ("choice", "noul"):
            question = t.evidence.questions(mode)["relevance"]
            official_user = old.user_prompt(state, question, ["A", "B"])
            worker_user = worker["user_prompt"](state, question["instructions"],
                                                worker["option_texts"](mode, question.get("criteria")), ["A", "B"])
            assert official_user == worker_user == current.user_prompt(state, question, ["A", "B"])
            rendered = template.render(messages=old.messages(state, question, ["A", "B"]),
                                       add_generation_prompt=True, enable_thinking=False, tools=None)
            assert rendered == worker["render"](worker_user)
            checked += 1
    diagnosis = module(research / "2026-10-19-d3-evidence-diagnosis.py", "diagnosis")
    for truth, scores, expected in (
        ([True, False], [.9, .1], (1, 1)),
        ([True, False], [.1, .9], (0, .5)),
        ([True, False], [.5, .5], (.5, .5)),
        ([False, True, True], [.9, .8, .7], (0, 7/12)),
    ):
        actual = diagnosis.ranking_metrics(truth, scores)
        assert all(abs(a-b) < 1e-12 for a, b in zip(actual, expected))
    assert statistics["script_sha256"] == sha(research / "2026-10-19-d3-evidence-diagnosis.py")
    output = {
        "inference_calls": 0, "human_reviewers": 0,
        "review_counts": dict(counts), "cases": joined,
        "prompt_comparisons": checked, "prompt_matches": checked,
        "prompt_test_limit": "Text rendering only, not token IDs, weights, attention masks or GPU outputs.",
        "ranking_metric_unit_cases": 4,
        "source_sha256": {str(p): sha(p) for p in (
            review_path, args.diagnosis / "statistics.json", args.diagnosis / "key.jsonl",
            worker_path, args.reference / "d3_format.py", args.reference / "d3_format_current.py",
            args.reference / "chat_template.jinja", Path(__file__))},
        "limitations": statistics["limits"],
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"review_counts": dict(counts), "prompt_matches": checked, "metric_tests": 4,
                      "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("inputs", "original", "reference", "diagnosis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
