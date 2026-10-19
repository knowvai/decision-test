"""Freeze a new validation workspace; no service or inference commands."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main(args):
    source = args.root / "source-r1"
    sys.path.insert(0, str(source / "deploy/docker/decision-eval"))
    sys.path.insert(0, str(source / "evaluation/runner/scripts"))
    from launch_reviewed import MODELS
    from decision_noul_validation import MODELS as roster
    original_source = args.old / "source-r1"
    unchanged = {}
    for name in ("public_decision_benchmark.py", "decision_capacity_preflight.py", "decision_passage_benchmark.py"):
        rel = "evaluation/runner/scripts/" + name
        assert sha(source / rel) == sha(original_source / rel)
        unchanged[rel] = sha(source / rel)
    public = ast.parse((source / "evaluation/runner/scripts/public_decision_benchmark.py").read_text())
    original = ast.parse((original_source / "evaluation/runner/scripts/public_decision_benchmark.py").read_text())
    functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert functions(public) == functions(original)
    preparation = args.root / "preparation-r1"
    preparation.mkdir()
    preserved = json.loads((args.old / "preparation-r1/preservation.json").read_text())
    assert all(sha(Path(path)) == digest for path, digest in preserved.items())
    for folder in ("source-r1", "preparation-r1", "dataset-six-mask-v1", "run-001", "choice-rescore-20261017"):
        for path in (args.old / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                preserved[str(path)] = sha(path)
    preserved[str(args.old / "runner-r1.log")] = sha(args.old / "runner-r1.log")
    save(preparation / "preservation.json", preserved)
    support = {"models": {}, "source_sha256": {}, "scope": "static native support; each live response must pass contract acceptance"}
    for model in roster:
        family, relative, served = MODELS[model]
        model_path = (args.old / "runtime-r1" / relative).resolve(strict=True)
        readme = model_path / "README.md"
        assert "noul" in readme.read_text().lower(), model
        evidence = [readme]
        evidence += [p for p in model_path.rglob("*.py") if "noul" in p.read_text().lower()]
        support["models"][model] = {"family": family, "served_model": served, "model_path": str(model_path),
            "native_noul_declared": True, "evidence_files": [str(p) for p in evidence]}
        support["source_sha256"].update({str(p): sha(p) for p in evidence})
    save(preparation / "support.json", support)
    runtime = args.root / "runtime-r1"
    runtime.mkdir()
    for name in ("models", "staging"):
        (runtime / name).symlink_to((args.old / "runtime-r1" / name).resolve(strict=True), target_is_directory=True)
    sources = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    save(preparation / "frozen.json", {"sources": sources, "unchanged_original_sources": unchanged,
        "unchanged_public_functions": len(functions(public)), "preserved_files": len(preserved),
        "preserved_bytes": sum(Path(p).stat().st_size for p in preserved), "models": list(roster),
        "support_sha256": sha(preparation / "support.json"), "preservation_sha256": sha(preparation / "preservation.json")})
    for name in sources:
        (source / name).chmod(0o444)
    print(json.dumps({"prepared": str(args.root), "models": len(roster), "source_files": len(sources), "preserved_files": len(preserved)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    main(parser.parse_args())
