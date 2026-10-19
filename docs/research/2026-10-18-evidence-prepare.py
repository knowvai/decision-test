"""Freeze new evidence-only sources and preserve predecessors; no inference."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import subprocess
import xml.etree.ElementTree as ET


def main(args):
    source = args.root / ("source-" + args.revision)
    sys.path.insert(0, str(source / "evaluation/runner/scripts"))
    sys.path.insert(0, str(source / "deploy/docker/decision-eval"))
    import decision_evidence_benchmark as e
    from launch_reviewed import MODELS, RUNTIME, VLLM, JEV_VLLM, SGLANG
    data, protocol = e.load_dataset(args.root / "dataset-v1")
    old = args.previous / "source-r1"
    unchanged = {}
    allowed = {"decision_capacity_preflight.py": {"llm_body"},
               "decision_passage_benchmark.py": {"refine_wire", "refine_result", "rerank_result"}}
    for name in ("public_decision_benchmark.py", "decision_capacity_preflight.py",
                 "decision_passage_benchmark.py", "decision_noul_validation.py"):
        rel = Path("evaluation/runner/scripts") / name
        old_tree = ast.parse((old / rel).read_text())
        new_tree = ast.parse((source / rel).read_text())
        functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        before, after = functions(old_tree), functions(new_tree)
        changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        assert changed == allowed.get(name, set()), (name, changed)
        unchanged[name] = {"old_sha256": e.sha(old / rel), "new_sha256": e.sha(source / rel),
                           "changed_functions": sorted(changed), "unchanged_functions_or_classes": len(before) - len(changed)}
    preparation = args.root / ("preparation-" + args.revision)
    preserved = json.loads((args.previous / "preparation-r1/preservation.json").read_text())
    e.p.check_artifacts(preserved)
    for name in ("source-r1", "preparation-r1", "run-001"):
        for path in (args.previous / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                preserved[str(path)] = e.sha(path)
    for path in args.previous.iterdir():
        if path.is_file():
            preserved[str(path)] = e.sha(path)
    previous_revisions = [prefix + "-r" + str(i) for i in range(1, int(args.revision.removeprefix("r")))
                          for prefix in ("source", "preparation")]
    for name in previous_revisions:
        for path in (args.root / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                preserved[str(path)] = e.sha(path)
    e.p.save(preparation / "preservation.json", preserved)
    tests = ET.parse(preparation / "tests-final.xml").getroot()
    suites = list(tests.iter("testsuite"))
    assert suites and all(int(s.get("failures", 0)) == 0 and int(s.get("errors", 0)) == 0 for s in suites)
    tested = sum(int(s.get("tests", 0)) for s in suites)
    assert tested >= 144
    provenance = json.loads((preparation / "data-provenance.json").read_text())
    assert provenance["passed"] and provenance["source_ids_grades_and_full_text_rejoined"]
    assert provenance["dataset_protocol_sha256"] == e.sha(args.root / "dataset-v1/protocol.json")
    support = json.loads((args.previous / "preparation-r1/support.json").read_text())
    e.p.check_artifacts(support["source_sha256"])
    assert set(support["models"]) == set(e.p.ROSTER[:-2])
    e.p.save(preparation / "noul-support.json", support)
    data, excluded = e.load_coverage(preparation / "coverage.json", data,
        e.sha(args.root / "dataset-v1/protocol.json"))
    reranker = json.loads((preparation / "reranker-frozen.json").read_text())
    assert 0 <= time.time() - reranker["verified_unix_seconds"] < 3600
    assert all(c["running"] for c in reranker["containers"].values())
    coverage = json.loads((preparation / "coverage.json").read_text())
    for model in e.p.ROSTER:
        family, relative, _ = MODELS[model]
        deployment = json.loads((args.baseline / model / "deployment.json").read_text())
        tag = SGLANG if family == "sglang" else JEV_VLLM if family == "jev" else VLLM if family in ("neohorse", "llm") else RUNTIME
        assert subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip() == deployment["image_id"]
        expected_image = deployment["head_image_id"] if family == "neohorse" else deployment["image_id"]
        for mode in e.modes(model):
            check = coverage["checks"][model + "." + mode]
            assert check["image_id"] == expected_image
            for path, digest in check["encoding_source_sha256"].items():
                if path.startswith("/model/"):
                    actual = (args.previous / "runtime-r1" / relative / path.removeprefix("/model/")).resolve(strict=True)
                    assert e.sha(actual) == digest
    rank = coverage["checks"]["reranker"]
    assert rank["image_id"] == reranker["containers"]["reranker"]["image"]
    assert rank["encoding_source_sha256"]["/model/tokenizer.json"] == reranker["tokenizer_sha256"]
    assert rank["encoding_source_sha256"]["/model/tokenizer_config.json"] == reranker["tokenizer_config_sha256"]
    assert rank["encoding_source_sha256"]["/source/deploy/docker/decision-eval/qwen3_reranker.jinja"] == reranker["template_sha256"]
    runtime = args.root / ("runtime-" + args.revision)
    runtime.mkdir()
    for name in ("models", "staging"):
        (runtime / name).symlink_to((args.previous / "runtime-r1" / name).resolve(strict=True), target_is_directory=True)
    sources = {str(path.resolve()): e.sha(path) for path in source.rglob("*")
               if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts}
    pairs = {phase: sum(len(p["candidates"]) for p in pools) for phase, (pools, _) in data.items()}
    result = {"source_sha256": sources, "dataset_protocol_sha256": e.sha(args.root / "dataset-v1/protocol.json"),
        "unchanged_functions": unchanged, "preserved_files": len(preserved),
        "preserved_bytes": sum(Path(path).stat().st_size for path in preserved),
        "preservation_sha256": e.sha(preparation / "preservation.json"),
        "tests_passed": tested, "tests_xml_sha256": e.sha(preparation / "tests-final.xml"),
        "data_provenance_sha256": e.sha(preparation / "data-provenance.json"),
        "noul_support_sha256": e.sha(preparation / "noul-support.json"),
        "coverage_sha256": e.sha(preparation / "coverage.json"),
        "reranker_evidence_sha256": e.sha(preparation / "reranker-frozen.json"),
        "planned_pairs_per_condition": pairs, "planned_calls": e.planned_calls(data),
        "planned_calls_total": sum(e.planned_calls(data).values()),
        "excluded_whole_pools": len(excluded), "top_n": e.TOP_N,
        "models": list(e.p.ROSTER), "conditions": e.CONDITIONS}
    e.p.save(preparation / "frozen.json", result)
    for path in sources:
        Path(path).chmod(0o444)
    print(json.dumps({k: result[k] for k in ("preserved_files", "preserved_bytes", "tests_passed", "planned_calls_total", "conditions")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--revision", default="r2")
    main(parser.parse_args())
