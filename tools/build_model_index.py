"""Index existing deployment identities and model pins without loading weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tomllib


REPOS = {
    "d3-edge": "vllm-sr/d3-edge", "d3-mini": "vllm-sr/d3-mini", "d3-flash": "vllm-sr/d3-flash",
    "laya-multilingual": "convaiinnovations/laya-multilingual", "clef-flash": "Cloudflare/clef-flash",
    "jeva-4b": "LIQIIIII/Jeva-4B", "neohorse-jev-4b": "TokenRhythm/NeoHorse-Jev-4B",
    "jev-27b": "autotrust/JEV-27B", "pplx-decider-v1.1-27b": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen3-30b-a3b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
    **{"vela-2.0-" + n.lower(): "vllm-sr/Vela-2.0-" + n for n in ("0.3B", "0.8B", "4B", "9B")},
}


def main(args):
    read = lambda p: json.loads(p.read_text())
    service = args.repository / "deploy/docker/decision-eval"
    pins = {}
    for path in service.glob("*model-lock.json"):
        for row in read(path)["models"]:
            pins[row["directory"]] = {"revision": row["official_revision"],
                "files_sha256": row["files_sha256"], "pin_source": str(path.relative_to(args.repository))}
    for row in read(service / "qwen-reranker-lock.json")["models"]:
        pins[row["directory"]] = {"revision": row["official_revision"], "files_sha256": row["files_sha256"],
                                   "pin_source": "deploy/docker/decision-eval/qwen-reranker-lock.json"}
    runtime = args.evidence / "preflight-20261015/vela-source"
    table = ast.parse((runtime / "vllm_srun/registry/tables/vela2.py").read_text())
    for call in ast.walk(table):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_vela":
            values = [ast.literal_eval(arg) for arg in call.args]
            pins["vela-2.0-" + values[0].lower()] = {"revision": values[1], "files_sha256": values[7],
                "pin_source": "preflight-20261015/vela-source/vllm_srun/registry/tables/vela2.py"}
    path = args.evidence / "classification-20261015/qwen38-supplement-20261016/model-manifest.json"
    for row in read(path)["models"]:
        pins[row["directory"]] = {"revision": row["hf_revision"], "file_manifest": row["files"],
                                   "pin_source": str(path.relative_to(args.evidence))}
    models = []
    for model, repo in sorted(REPOS.items()):
        deployment = args.evidence / "evidence-20261018/run-001" / model / "deployment.json"
        observed = read(deployment) if deployment.exists() else None
        pin = pins.get(model, {"revision": None, "pin_source": None,
            "note": "No immutable upstream revision found in the collected acquisition records; do not substitute latest and claim exact reproduction."})
        models.append({"model": model, "upstream": "https://huggingface.co/" + repo, **pin,
                       "evidence_deployment": {k: observed[k] for k in ("family", "image_id", "tp", "cpu_budget", "scheduling") if k in observed} if observed else None})
    images = []
    for directory in ("framework-sglang-fetch-v1", "framework-qwen-fetch-v1"):
        row = read(args.evidence / directory / "download.private.json")
        images.append({k: row[k] for k in ("reference", "alias", "archive_sha256")})
    result = {"models": models, "base_images": images,
        "worker_version": tomllib.loads((runtime / "pyproject.toml").read_text())["project"]["version"],
        "notes": ["Pins are copied from the recorded acquisition locks or the installed runtime registry, not queried from current upstream main.",
                  "Weight files were not redistributed or rehashed during publication; original per-file manifests are archived where available.",
                  "Reranker resource settings and container identity are in the separate proxy deployment archive."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"models": len(models), "with_revision": sum(m["revision"] is not None for m in models)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
