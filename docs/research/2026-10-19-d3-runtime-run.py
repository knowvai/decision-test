"""Isolated native D3 scheduling experiment, reusing the audited pilot adapters."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("pilot_run", Path(__file__).with_name("2026-10-19-d3-tuning-run.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
t, launcher = p.t, p.launcher
IMAGE = "sha256:f3ccfb750e184c66e3aee485d64b7ed19cfc0ce83d73650b8b92b3463a3a66b4"
DATA_SHA = "3a223a53f3d2b0a96046ab26d308b157c6a0fd8f652073acc9a12ae2a09faf17"
LEVELS = (1, 8, 16, 32)
BASE = dict(worker_threads=2, seqs=32, tokens=32768, batch_window_ms=2)
PROFILES = {
    "baseline": BASE,
    "window0": {**BASE, "batch_window_ms": 0},
    "window5": {**BASE, "batch_window_ms": 5},
    "window10": {**BASE, "batch_window_ms": 10},
    "seq8": {**BASE, "seqs": 8},
    "tokens8192": {**BASE, "tokens": 8192},
    "tokens65536": {**BASE, "tokens": 65536},
    "auto": {**BASE, "worker_auto_budget": True},
    "baseline-end": BASE,
}
ORIGINAL_IDS = {
    "decision-acceptance-model": "d18a58afabef1dfb624378ed7a6365c603ff39a8320dcb9e12678fa261d7b3e5",
    "decision-acceptance-proxy": "56b73142e97098100daad87383967c4e709ddebdf99ac020a71aef80cfe6d514",
}


def profile_order(model):
    alternatives = list(PROFILES)[1:-1]
    return ["baseline", *(alternatives if model == "d3-mini" else reversed(alternatives)), "baseline-end"]


def workload(data, protocol, task):
    keys = {(r["case_id"], r["fragment_id"]) for r in protocol["performance"][task]}
    rows = [r for r in t.read_rows(data / f"{task}-dev.private.jsonl")
            if (r["case_id"], r["fragment_id"]) in keys]
    assert len(rows) == len(keys) == 128
    return rows


def blocks(jobs):
    ordered = sorted(jobs, key=lambda r: r["chars"])
    warm = [ordered[i * (len(ordered) - 1) // 31] for i in range(32)]
    for repeat in (1, 2):
        for n in (1, 32):
            yield f"warm-n{n}-r{repeat}", warm, n, "warmup", repeat
    for repeat, order in ((1, LEVELS), (2, tuple(reversed(LEVELS)))):
        for n in order:
            yield f"n{n}-r{repeat}", jobs if repeat == 1 else list(reversed(jobs)), n, "performance", repeat


class NotAttemptedAfterFailure(Exception):
    capacity_metadata = {"calls": 0}


async def fail_closed_measure(jobs, infer, concurrency, timeout):
    failed = False

    async def guarded(job):
        nonlocal failed
        if failed:
            raise NotAttemptedAfterFailure()
        try:
            return await infer(job)
        except BaseException:
            failed = True
            raise

    rows, meta = await t.capacity.measure(jobs, guarded, concurrency, timeout)
    # The shared legacy summary counts these as unsuccessful; the independent audit
    # separates attempted failures from this explicit zero-call ledger.
    for row in rows:
        if row.get("error_type") == "NotAttemptedAfterFailure":
            row["status"] = "not_attempted"
    return rows, meta


def check_original():
    original = [p.identity(name) for name in ("decision-acceptance-proxy", "decision-acceptance-model")]
    assert all(r["id"] == ORIGINAL_IDS[r["name"]] and r["running"] and not r["oom"] for r in original)
    processes = {int(x) for x in p.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).splitlines()}
    owned = {int(x) for x in p.command(["docker", "top", "decision-acceptance-model", "-eo", "pid"]).splitlines()[1:]}
    assert processes <= owned, "Foreign GPU process; no mutation"
    assert p.command(["docker", "image", "inspect", launcher.RUNTIME, "--format", "{{.Id}}"] ) == IMAGE
    return original


def prepare(args):
    protocol = t.read_data(args.data)
    assert t.sha(args.data / "protocol.json") == DATA_SHA
    assert not (args.root / "preparation.json").exists()
    previous = args.base / "d3-tuning-20261019"
    assert json.loads((previous / "status.json").read_text())["stage"] == "complete"
    preflights = {}
    for model in t.MODELS:
        prepath = previous / f"preflight-{model}.json"
        pre = json.loads(prepath.read_text())
        assert pre["inference_calls"] == 0 and all(r["supported"] for r in pre["rows"])
        checked = []
        for task in protocol["variants"]:
            jobs = workload(args.data, protocol, task)
            native = {(r["case_id"], r["fragment_id"]): r for r in pre["rows"]
                      if r["task"] == task and r["split"] == "dev" and r["variant"] == "baseline"}
            for job in jobs:
                row = native[job["case_id"], job["fragment_id"]]
                assert row["input_sha256"] == job["input_sha256"] and row["tokens"] <= 8192
                checked.append(row)
        preflights[model] = {"source": str(prepath), "sha256": t.sha(prepath), "rows": checked}
    preserved = json.loads((previous / "preservation.json").read_text())
    for path in previous.rglob("*"):
        relative = path.relative_to(previous)
        if path.is_file() and not {"runtime", "__pycache__", ".pytest_cache"}.intersection(relative.parts):
            preserved[str(path)] = t.sha(path)
    t.capacity.check_artifacts(preserved)
    t.save(args.root / "preservation.json", preserved)
    sources = {str(path): t.sha(path) for path in t.REPO.rglob("*")
               if path.is_file() and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)}
    t.save(args.root / "preparation.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "original": check_original(),
        "source_sha256": sources, "dataset_sha256": DATA_SHA, "image": IMAGE,
        "profiles": PROFILES, "levels": LEVELS, "preflights": preflights,
        "planned_http_calls": 41472, "planned_blocks": 432,
        "startup_calibration": "internal native GPU work, not HTTP workload calls"})
    print(json.dumps({"prepared": True, "preserved": len(preserved), "sources": len(sources), "calls": 41472}), flush=True)


async def performance(args, model, key, folder, protocol):
    limits = p.httpx.Limits(max_connections=32, max_keepalive_connections=32)
    async with p.httpx.AsyncClient(transport=p.httpx.AsyncHTTPTransport(retries=0, limits=limits),
                                  trust_env=False, follow_redirects=False, timeout=300) as client:
        session = p.Session(args, client, key, model, measure=fail_closed_measure)
        cards = await client.get(args.base_url + "/v1/models", headers=session.headers)
        cards.raise_for_status()
        t.save(folder / "model-cards.json", cards.json())
        for task in protocol["variants"]:
            jobs = workload(args.data, protocol, task)
            for name, selected, n, phase, repeat in blocks(jobs):
                await session.block(folder, task + "-" + name, selected, protocol["variants"][task]["baseline"], n,
                                    phase=phase, task=task, repeat=repeat, variant="baseline", threshold=None)


def launch_and_measure(args, model, profile, folder, protocol, key):
    runtime = folder / "runtime"
    runtime.mkdir()
    (runtime / "models").symlink_to(args.base / "models", target_is_directory=True)
    p.status(args.root, "loading", model=model, profile=profile)
    launcher.launch(SimpleNamespace(root=runtime, model=model, env_file=args.env_file, native_port=30888,
        public_port=30889, enforce_eager=False, **PROFILES[profile]))
    t.save(folder / "deployment.json", json.loads((runtime / "acceptance-20261015/evidence" /
                                                  (model + ".deployment.json")).read_text()))
    with (folder / "gpu.csv").open("x") as output:
        sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                                    "--format=csv,noheader,nounits", "--loop=1"], stdout=output, stderr=subprocess.DEVNULL)
        try:
            asyncio.run(p.ready(args.base_url, key))
            t.save(folder / "identity.json", p.identity("decision-acceptance-model"))
            logs = subprocess.check_output(["docker", "logs", "decision-acceptance-model"], text=True, stderr=subprocess.STDOUT)
            resolved = [line for line in logs.splitlines() if re.search(r"serving .*max_num_seqs \d+, max_num_batched_tokens \d+ from", line)]
            assert len(resolved) == 1, "Resolved native limits not observable"
            t.save(folder / "resolved-limits.json", {"serving": resolved, "auto_requested": profile == "auto"})
            asyncio.run(performance(args, model, key, folder, protocol))
            terminal = p.identity("decision-acceptance-model")
            assert terminal["running"] and not terminal["oom"] and terminal["restart_count"] == 0
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
    assert t.sha(args.data / "protocol.json") == prep["dataset_sha256"] == DATA_SHA
    assert not (args.root / "run-protocol.json").exists(), "No restart or resume"
    preserved = json.loads((args.root / "preservation.json").read_text())
    t.capacity.check_artifacts(prep["source_sha256"])
    t.capacity.check_artifacts(preserved)
    lock = (args.base / "classification-20261015/classification.lock").open("r+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = check_original()
    assert original == prep["original"], "Service identity changed since preparation"
    t.save(args.root / "run-protocol.json", {"original": original, "preparation_sha256": t.sha(args.root / "preparation.json"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "profile_order": {m: profile_order(m) for m in t.MODELS},
        "retries": 0, "fail_closed": True})
    key = launcher.read_key(args.env_file)
    try:
        for row in original:
            p.command(["docker", "stop", "--time", "30", row["id"]])
            p.command(["docker", "rename", row["id"], row["name"] + "-held-d3runtime"])
        for model in t.MODELS:
            for profile in profile_order(model):
                folder = args.root / model / profile
                folder.mkdir(parents=True)
                launch_and_measure(args, model, profile, folder, protocol, key)
        t.capacity.check_artifacts(preserved)
        t.capacity.check_artifacts(prep["source_sha256"])
        for name in ORIGINAL_IDS:
            launcher.remove_owned(name)
        for row in reversed(original):
            assert p.identity(row["name"] + "-held-d3runtime")["id"] == row["id"]
            p.command(["docker", "rename", row["id"], row["name"]])
            p.command(["docker", "start", row["id"]])
        p.status(args.root, "restoring-original")
        asyncio.run(p.ready(args.base_url, key))
        restored = [p.identity(row["name"]) for row in original]
        assert all(a["id"] == b["id"] and a["image"] == b["image"] and a["command"] == b["command"]
                   and b["running"] and not b["oom"] for a, b in zip(original, restored))
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
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30889")
    args = parser.parse_args()
    os.umask(0o077)
    (prepare if args.action == "prepare" else run)(args)
