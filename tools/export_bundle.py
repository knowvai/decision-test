"""Build a public evidence bundle from read-only, already-collected artifacts.

This is an offline publisher, not an evaluation or deployment entry point.
"""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


SCRIPT_NAMES = (
    "public_decision_benchmark.py", "decision_capacity_preflight.py",
    "decision_passage_benchmark.py", "decision_evidence_benchmark.py",
    "decision_noul_validation.py", "decision_rag_worker.py", "replay_decision_rag.py",
)
DOC_PREFIXES = (
    "2026-10-15-decision-", "2026-10-16-decision-", "2026-10-16-qwen38-",
    "2026-10-17-passage-", "2026-10-17-qwen3-capacity-", "2026-10-17-vela-",
    "2026-10-18-answer-evidence-", "2026-10-18-evidence-",
    "2026-10-19-answer-evidence-", "2026-10-19-clef-runtime-",
    "2026-10-19-d3-", "2026-10-19-evidence-", "2026-10-19-decision-evaluation-",
)
PLAN_NAMES = (
    "2026-10-14-decision-benchmark-plan.md", "2026-10-18-answer-evidence-benchmark.md",
    "2026-10-19-clef-runtime-parameters.md", "2026-10-19-d3-runtime-parameters.md",
    "2026-10-19-d3-tuning-pilot.md", "2026-10-19-evidence-root-cause.md",
)
TEXT_KEYS = {"utterance", "query", "content", "text", "passage", "document"}


class NamespaceRules:
    """Rename presentation metadata while preserving fixed hash/random inputs."""

    def __init__(self, config=None):
        config = config or {}
        self.replacements = config.get("replacements", {})
        self.seeds = config.get("seed_aliases", {})

    def text(self, value):
        for source, target in sorted({**self.replacements, **self.seeds}.items(),
                                     key=lambda item: len(item[0]), reverse=True):
            value = value.replace(source, target)
        return value

    def content(self, raw, path):
        text = raw.decode("utf-8")
        if str(path).endswith(".py") and any(seed in text for seed in self.seeds):
            tree = ast.parse(text)
            bindings = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str) and node.value.value in self.seeds:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                bindings[target.id] = node.value.value
            offsets, position = [], 0
            for line in raw.splitlines(keepends=True):
                offsets.append(position)
                position += len(line)
            edits = {}
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
                if name not in {"digest", "sha256", "Random"}:
                    continue
                for node in ast.walk(call):
                    seed = bindings.get(node.id) if isinstance(node, ast.Name) else None
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in self.seeds:
                        seed = node.value
                    if seed is None:
                        continue
                    start = offsets[node.lineno - 1] + node.col_offset
                    end = offsets[node.end_lineno - 1] + node.end_col_offset
                    edits[(start, end)] = f'bytes.fromhex("{seed.encode().hex()}").decode("utf-8")'.encode()
            for (start, end), replacement in sorted(edits.items(), reverse=True):
                raw = raw[:start] + replacement + raw[end:]
            text = raw.decode("utf-8")
        return self.text(text).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def withheld(path):
    parts, name = path.parts, path.name.lower()
    if any(p in {".git", "__pycache__", ".pytest_cache", "venv", ".venv"} for p in parts):
        return "cache_or_environment"
    if (name == ".env" or name.endswith((".env", ".pem", ".key"))
            or name in {".nginx.conf", "credentials", "id_rsa", "id_ed25519"}):
        return "private_runtime_configuration"
    if name.endswith((".whl", ".safetensors", ".bin", ".pt", ".pyc")):
        return "binary_dependency_or_model"
    if name.endswith(".log"):
        return "runtime_log_retained_at_origin_not_public"
    if name.endswith(".jsonl") and ("input" in name or "pools" in name
            or "dataset" in parts and "labels" not in name):
        return "dataset_text_download_separately"
    return None


def skeleton(value):
    if isinstance(value, list):
        return [skeleton(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {k: ({"__source_text_sha256__": sha(v.encode())}
                if k in TEXT_KEYS and isinstance(v, str) else skeleton(v))
            for k, v in value.items()}


class Publisher:
    def __init__(self, output, substitutions, namespace=None):
        self.output = output
        self.substitutions = substitutions
        self.namespace = namespace or NamespaceRules()
        self.files = []
        self.archives = []
        self.recipes = []

    def public_bytes(self, data, path=""):
        data = self.namespace.content(data, path)
        text = data.decode("utf-8")
        for private, public in self.substitutions.items():
            text = text.replace(private, public)
        name = "2026-10-19-answer-evidence-results.json"
        text = text.replace(f'href="{name}"', f'href="{name}.gz"')
        text = text.replace(f']({name})', f']({name}.gz)')
        return text.encode()

    def record(self, source, rel, raw, published=None, **extra):
        row = {"source": source, "path": self.namespace.text(str(rel)), "source_bytes": len(raw),
               "source_sha256": sha(raw), **extra}
        if source == "preflight-20261015" and str(rel) == "vela-source/LICENSE":
            row["provenance"] = {
                "kind": "publication_supplement",
                "url": "https://raw.githubusercontent.com/vllm-project/semantic-router/main/LICENSE",
                "note": "Added from the official project during publication; not a historical execution file. The recorded SHA256 identifies the collected license text.",
            }
        if published is not None:
            row.update(published_bytes=len(published), published_sha256=sha(published),
                       publication="unchanged" if raw == published else "metadata_redacted")
        self.files.append(row)

    def copy(self, path, rel, source="workspace"):
        raw = path.read_bytes()
        data = self.public_bytes(raw, rel)
        compressed = len(data) > 8 * 1024 * 1024
        destination = self.namespace.text(str(rel)) + (".gz" if compressed else "")
        save(self.output / destination, gzip.compress(data, mtime=0) if compressed else data)
        self.record(source, rel, raw, data, destination=destination, gzip=compressed)

    def archive(self, base, group, paths, source):
        buffer = io.BytesIO()
        pending = []
        archive = f"artifacts/{source}/{group}.tar.gz"
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(paths):
                rel = path.relative_to(base)
                if path.is_symlink():
                    continue
                raw = path.read_bytes()
                reason = withheld(rel)
                if reason == "dataset_text_download_separately" and source in {
                    "qwen3-diagnostic-20261017", "vela-diagnostic-20261017"
                }:
                    reason = None  # These diagnostics use generated synthetic inputs.
                if reason == "dataset_text_download_separately" and rel.name == "preflight.inputs.private.jsonl":
                    rows = [json.loads(line) for line in raw.split(b"\n") if line]
                    if rows and all(row["case_id"].startswith("synthetic:") for row in rows):
                        reason = None
                if reason:
                    self.record(source, rel, raw, publication="withheld", reason=reason)
                    if reason == "dataset_text_download_separately":
                        self.recipe(path, source, rel, raw)
                    continue
                try:
                    data = self.public_bytes(raw, rel)
                except UnicodeDecodeError:
                    self.record(source, rel, raw, publication="withheld", reason="unreviewed_binary")
                    continue
                info = tarfile.TarInfo(self.namespace.text(str(rel)))
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
                self.record(source, rel, raw, data, archive=archive)
                pending.append(str(rel))
        if pending:
            data = gzip.compress(buffer.getvalue(), mtime=0)
            assert len(data) < 95 * 1024 * 1024, (archive, len(data))
            save(self.output / archive, data)
            self.archives.append({"path": archive, "bytes": len(data), "sha256": sha(data),
                                  "files": len(pending), "extract_under": source})

    def recipe(self, path, source, rel, raw):
        rows = [json.loads(line) for line in raw.split(b"\n") if line]
        transformed = [skeleton(row) for row in rows]
        # Keep the original serialization contract, not just equivalent JSON.
        serialization = None
        for ascii_only in (False, True):
            for sort_keys in (False, True):
                candidate = "".join(json.dumps(row, ensure_ascii=ascii_only, sort_keys=sort_keys) + "\n" for row in rows).encode()
                if candidate == raw:
                    serialization = {"ensure_ascii": ascii_only, "sort_keys": sort_keys}
                    break
            if serialization is not None:
                break
        if transformed == rows or serialization is None:
            return
        name = f"datasets/recipes/{source}/{rel}.recipe.json.gz"
        value = {"source": source, "path": str(rel), "sha256": sha(raw),
                 "bytes": len(raw), "serialization": serialization, "records": transformed}
        save(self.output / name, gzip.compress(json_bytes(value), mtime=0))
        self.recipes.append({"recipe": name, "source": source, "path": str(rel),
                             "source_sha256": sha(raw), "records": len(rows)})


def main(args):
    substitutions = json.loads(args.redactions.read_text()) if args.redactions else {}
    namespace = NamespaceRules(json.loads(args.namespace_rules.read_text()) if args.namespace_rules else {})
    pub = Publisher(args.output, substitutions, namespace)
    selected = []
    for name in SCRIPT_NAMES:
        selected.append(args.runner_root / "scripts" / name)
    selected.extend(p.relative_to(args.workspace) for p in
                    (args.workspace / "deploy/docker/decision-eval").iterdir() if p.is_file())
    names = {"test_" + name for name in SCRIPT_NAMES[:5]}
    selected.extend(args.runner_root / "tests" / name for name in sorted(names))
    for p in (args.workspace / "deploy/docker/tests").glob("test_*.py"):
        if p.name.startswith(("test_decision_", "test_d3_", "test_clef_", "test_evidence_",
                              "test_deployment_", "test_reviewed_", "test_sglang_", "test_vllm_",
                              "test_jeva_", "test_neohorse_", "test_laya_", "test_qwen_reranker")):
            selected.append(p.relative_to(args.workspace))
    for p in (args.workspace / "docs/research").iterdir():
        if p.is_file() and p.name.startswith(DOC_PREFIXES):
            selected.append(p.relative_to(args.workspace))
    selected.extend(Path("docs/plans") / name for name in PLAN_NAMES)
    for rel in sorted(set(selected)):
        pub.copy(args.workspace / rel, rel)
    scorer = args.runner_root / "src/evaluation/retrieval_eval.py"
    pub.copy(args.evidence / "evidence-20261018/source-r3" / scorer, scorer, "frozen-evidence-r3")
    for root in sorted(args.evidence.iterdir()):
        if not root.is_dir():
            continue
        groups = defaultdict(list)
        for p in root.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            parts = p.relative_to(root).parts
            group = parts[0] if len(parts) > 1 else "root"
            if group.startswith("run-") and len(parts) > 2:
                group += "--" + parts[1]
            groups[group].append(p)
        for group, files in sorted(groups.items()):
            pub.archive(root, group, files, root.name)
    pub.archive(args.proxy, "deployment", [p for p in args.proxy.rglob("*") if p.is_file()],
                "passage-reranker-20261017")
    save(args.output / "manifests/publication.json", json_bytes({
        "schema": 1, "purpose": "publication provenance; not a rerun or revised experiment audit",
        "files": pub.files, "archives": pub.archives, "dataset_recipes": pub.recipes,
        "substitutions": "host addresses and local home paths only; private values not published",
        "counts": dict(Counter(row["publication"] for row in pub.files)),
    }))
    print(json.dumps({"files": len(pub.files), "archives": len(pub.archives),
                      "recipes": len(pub.recipes), "counts": dict(Counter(r["publication"] for r in pub.files))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "evidence", "proxy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--redactions", type=Path)
    parser.add_argument("--namespace-rules", type=Path, help="Optional external publication namespace rules")
    parser.add_argument("--runner-root", type=Path, default=Path("evaluation/runner"))
    main(parser.parse_args())
