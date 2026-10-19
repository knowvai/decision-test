"""Isolated Clef scheduling experiment using the existing measurement/launcher."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("d3_runtime", Path(__file__).with_name("2026-10-19-d3-runtime-run.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
p, t, launcher = r.p, r.t, r.launcher
MODEL = "clef-flash"
IMAGE = "sha256:a682ef469e9900ec254bb6d98e9c6052d0e0c727dcb005b4ea1a229b77424fae"
LEVELS = (1, 8, 16, 32, 64)
PROFILES = {
    "baseline": dict(seqs=32, tokens=32768),
    "seq8": dict(seqs=8, tokens=32768),
    "tokens8192": dict(seqs=32, tokens=8192),
    "seq64": dict(seqs=64, tokens=32768),
    "tokens65536": dict(seqs=32, tokens=65536),
    "wide": dict(seqs=64, tokens=65536),
    "baseline-end": dict(seqs=32, tokens=32768),
}
CALLS, BLOCKS = 19712, 196
RESOLVED_KEYS = {
    "max_running_requests", "max_prefill_tokens", "mem_fraction_static", "context_length",
    "is_embedding", "disable_radix_cache", "chunked_prefill_size", "disable_cuda_graph",
    "cuda_graph_config", "enable_metrics", "dtype", "tp_size", "pp_size", "served_model_name",
    "model_path", "version", "max_total_num_tokens", "max_req_input_len", "max_req_len",
}


def blocks(jobs):
    ordered = sorted(jobs, key=lambda row: row["chars"])
    warm = [ordered[i * (len(ordered) - 1) // 31] for i in range(32)]
    for repeat in (1, 2):
        for n in (1, 64):
            yield f"warm-n{n}-r{repeat}", warm, n, "warmup", repeat
    for repeat, order in ((1, LEVELS), (2, tuple(reversed(LEVELS)))):
        for n in order:
            yield f"n{n}-r{repeat}", jobs if repeat == 1 else list(reversed(jobs)), n, "performance", repeat


def safe_settings(value):
    if isinstance(value, dict):
        return {k: v if k in RESOLVED_KEYS else safe_settings(v) for k, v in value.items()
                if k in RESOLVED_KEYS or isinstance(v, (dict, list)) and safe_settings(v)}
    if isinstance(value, list):
        return [item for v in value if isinstance(v, (dict, list)) and (item := safe_settings(v))]
    return None


ENCODE = r'''
import hashlib, inspect, json, sys
from pathlib import Path
from transformers import AutoTokenizer
from sglang.srt.entrypoints.systemone.joint_schema import encode_joint_schema, SystemOneRequest
from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
payload = json.load(sys.stdin)
tok = AutoTokenizer.from_pretrained('/model', local_files_only=True)
limit = max_joint_prompt_tokens(context_len=8192, num_reserved_tokens=0,
    max_req_input_len=8186, max_prefill_tokens=8192)
rows = []
for task, items in payload['tasks'].items():
 for job in items:
  req = SystemOneRequest(model='clef-flash', state=job['state'], questions=payload['questions'][task])
  ids, _ = encode_joint_schema(tok, req, max_length=1000000, image_token_counts=[])
  supported = len(ids) <= limit
  if supported:
   actual, _ = encode_joint_schema(tok, req, max_length=limit, image_token_counts=[])
   assert ids == actual
  rows.append(dict(task=task, case_id=job['case_id'], fragment_id=job['fragment_id'],
      input_sha256=job['input_sha256'], tokens=len(ids), supported=supported))
sources = [Path(inspect.getfile(encode_joint_schema)), Path(inspect.getfile(max_joint_prompt_tokens)),
    Path('/model/tokenizer.json'), Path('/model/config.json'), Path('/model/joint_head_config.json')]
print(json.dumps(dict(limit=limit, rows=rows, inference_calls=0,
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})))
'''


def model_inventory(base):
    return {str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted((base / "models" / MODEL).iterdir()) if path.is_file()}


def check_original():
    original = r.check_original()
    assert p.command(["docker", "image", "inspect", launcher.SGLANG, "--format", "{{.Id}}"]) == IMAGE
    return original


def prepare(args):
    os.umask(0o077)
    protocol = t.read_data(args.data)
    assert t.sha(args.data / "protocol.json") == r.DATA_SHA
    assert not (args.root / "preparation.json").exists()
    original = check_original()
    tasks = {task: r.workload(args.data, protocol, task) for task in protocol["variants"]}
    payload = {"tasks": tasks, "questions": {task: protocol["variants"][task]["baseline"] for task in tasks}}
    cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp",
           "--cpus", "2", "--memory", "8g", "--cap-drop", "ALL", "--env", "PYTHONDONTWRITEBYTECODE=1",
           "--env", "HF_HUB_OFFLINE=1", "--env", "TOKENIZERS_PARALLELISM=false",
           "--env", "HOME=/tmp", "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
           "--env", "SGLANG_CACHE_DIR=/tmp/sglang",
           "--env", "LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64",
           "-v", f"{args.base / 'models' / MODEL}:/model:ro", "--entrypoint", "python3", IMAGE, "-B", "-c", ENCODE]
    result = subprocess.run(cmd, input=json.dumps(payload), text=True, capture_output=True, timeout=600)
    if result.returncode:
        t.save(args.root / "preflight-failure.json", {"returncode": result.returncode,
               "stderr_sha256": t.benchmark.digest(result.stderr), "inference_calls": 0})
        raise RuntimeError("CPU encoder failed; no inference or service mutation")
    pre = json.loads(result.stdout)
    t.save(args.root / "preflight.json", pre)
    assert len(pre["rows"]) == 256 and all(row["supported"] for row in pre["rows"])
    prior = args.base / "d3-runtime-20261019"
    assert json.loads((prior / "status.json").read_text())["stage"] == "complete"
    preserved = json.loads((prior / "preservation.json").read_text())
    for path in prior.rglob("*"):
        if path.is_file() and not {"runtime", "__pycache__", ".pytest_cache"}.intersection(path.relative_to(prior).parts):
            preserved[str(path)] = t.sha(path)
    t.capacity.check_artifacts(preserved)
    t.save(args.root / "preservation.json", preserved)
    sources = {str(path): t.sha(path) for path in t.REPO.rglob("*") if path.is_file()
               and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)}
    t.save(args.root / "preparation.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "original": original,
        "source_sha256": sources, "dataset_sha256": r.DATA_SHA, "image": IMAGE,
        "profiles": PROFILES, "levels": LEVELS, "preflight_sha256": t.sha(args.root / "preflight.json"),
        "model_inventory": model_inventory(args.base), "planned_http_calls": CALLS, "planned_blocks": BLOCKS})
    print(json.dumps({"prepared": True, "preserved": len(preserved), "sources": len(sources),
                      "inputs": 256, "max_tokens": max(row["tokens"] for row in pre["rows"]), "calls": CALLS}), flush=True)


async def performance(args, key, folder, protocol):
    limits = p.httpx.Limits(max_connections=64, max_keepalive_connections=64)
    tokens = {(row["case_id"], row["fragment_id"]): row["tokens"]
              for row in json.loads((args.root / "preflight.json").read_text())["rows"]}
    async with p.httpx.AsyncClient(transport=p.httpx.AsyncHTTPTransport(retries=0, limits=limits),
                                  trust_env=False, follow_redirects=False, timeout=300) as client:
        session = p.Session(args, client, key, MODEL, measure=r.fail_closed_measure,
                            native_family="sglang", expected_tokens=tokens)
        response = await client.get(args.base_url + "/get_server_info", headers=session.headers)
        response.raise_for_status()
        settings = safe_settings(response.json())
        assert settings, "Resolved server settings not observable"
        t.save(folder / "resolved-settings.json", settings)
        response = await client.get(args.base_url + "/v1/models", headers=session.headers)
        response.raise_for_status()
        t.save(folder / "model-cards.json", response.json())
        for task in protocol["variants"]:
            jobs = r.workload(args.data, protocol, task)
            for name, selected, n, phase, repeat in blocks(jobs):
                await session.block(folder, task + "-" + name, selected, protocol["variants"][task]["baseline"], n,
                                    phase=phase, task=task, repeat=repeat, variant="baseline", threshold=None)


def launch_and_measure(args, profile, folder, protocol, key):
    runtime = folder / "runtime"
    runtime.mkdir()
    (runtime / "models").symlink_to(args.base / "models", target_is_directory=True)
    p.status(args.root, "loading", model=MODEL, profile=profile)
    began = time.monotonic()
    launcher.launch(SimpleNamespace(root=runtime, model=MODEL, env_file=args.env_file, native_port=30888,
                    public_port=30889, enforce_eager=False, sglang_metrics=True, **PROFILES[profile]))
    t.save(folder / "deployment.json", json.loads((runtime / "acceptance-20261015/evidence" /
                                                  (MODEL + ".deployment.json")).read_text()))
    with (folder / "gpu.csv").open("x") as output:
        sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                                    "--format=csv,noheader,nounits", "--loop=1"], stdout=output, stderr=subprocess.DEVNULL)
        try:
            asyncio.run(p.ready(args.base_url, key))
            t.save(folder / "startup.json", {"loading_seconds": time.monotonic() - began,
                   "ready_at_utc": datetime.now(timezone.utc).isoformat(), "health_inference": False})
            initial = p.identity("decision-acceptance-model")
            t.save(folder / "identity.json", initial)
            asyncio.run(performance(args, key, folder, protocol))
            terminal = p.identity("decision-acceptance-model")
            assert initial == terminal and terminal["running"] and not terminal["oom"] and terminal["restart_count"] == 0
            t.save(folder / "terminal-identity.json", terminal)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()


def run(args):
    os.umask(0o077)
    protocol = t.read_data(args.data)
    prep = json.loads((args.root / "preparation.json").read_text())
    assert prep["profiles"] == PROFILES and prep["levels"] == list(LEVELS)
    assert t.sha(args.data / "protocol.json") == prep["dataset_sha256"] == r.DATA_SHA
    assert not (args.root / "run-protocol.json").exists(), "No restart or resume"
    assert model_inventory(args.base) == prep["model_inventory"]
    assert t.sha(args.root / "preflight.json") == prep["preflight_sha256"]
    preserved = json.loads((args.root / "preservation.json").read_text())
    t.capacity.check_artifacts(prep["source_sha256"])
    t.capacity.check_artifacts(preserved)
    lock = (args.base / "classification-20261015/classification.lock").open("r+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = check_original()
    assert original == prep["original"], "Service identity changed since preparation"
    t.save(args.root / "run-protocol.json", {"original": original, "preparation_sha256": t.sha(args.root / "preparation.json"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "profile_order": list(PROFILES),
        "retries": 0, "fail_closed": True})
    key = launcher.read_key(args.env_file)
    try:
        for row in original:
            p.command(["docker", "stop", "--time", "30", row["id"]])
            p.command(["docker", "rename", row["id"], row["name"] + "-held-clefruntime"])
        for profile in PROFILES:
            folder = args.root / MODEL / profile
            folder.mkdir(parents=True)
            launch_and_measure(args, profile, folder, protocol, key)
        assert model_inventory(args.base) == prep["model_inventory"]
        t.capacity.check_artifacts(preserved)
        t.capacity.check_artifacts(prep["source_sha256"])
        for name in r.ORIGINAL_IDS:
            launcher.remove_owned(name)
        for row in reversed(original):
            assert p.identity(row["name"] + "-held-clefruntime")["id"] == row["id"]
            p.command(["docker", "rename", row["id"], row["name"]])
            p.command(["docker", "start", row["id"]])
        p.status(args.root, "restoring-original")
        asyncio.run(p.ready(args.base_url, key))
        restored = [p.identity(row["name"]) for row in original]
        assert all(all(a[k] == b[k] for k in ("id", "name", "image", "command")) and b["running"] and not b["oom"]
                   for a, b in zip(original, restored))
        t.save(args.root / "restoration.json", {"before": original, "after": restored, "preserved": len(preserved)})
        p.status(args.root, "complete", restored_original=True)
    except BaseException as error:
        p.status(args.root, "failed", error_type=type(error).__name__, automatic_restoration=False)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30889")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (prepare if args.action == "prepare" else run)(args)
