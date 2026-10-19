"""Bounded D3 quality/throughput pilot on the isolated evaluation host."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("d3_tuning", Path(__file__).with_name("2026-10-19-d3-tuning.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
import httpx
import launch_reviewed as launcher
from deployment_probe import native_choice_request, native_choice_result


def command(parts):
    return subprocess.check_output(list(map(str, parts)), text=True).strip()


def identity(name):
    obj = json.loads(command(["docker", "inspect", name]))[0]
    assert obj["Config"]["Labels"].get(launcher.LABEL) == "true"
    return {"id": obj["Id"], "name": name, "image": obj["Image"], "running": obj["State"]["Running"],
            "pid": obj["State"]["Pid"], "started_at": obj["State"]["StartedAt"],
            "command": [obj["Path"], *obj["Args"]], "restart_count": obj["RestartCount"],
            "oom": obj["State"]["OOMKilled"]}


def status(root, stage, **fields):
    value = {"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    t.benchmark.write_json(root / "status.json", value)
    print(json.dumps(value), flush=True)


PREFLIGHT = r'''
import hashlib, inspect, json
from pathlib import Path
from vllm_srun.families.decision3 import prompt
root, data = Path('/model'), Path('/data')
tok, _ = prompt.load_tokenizer(root)
prompt.check_chat_template(root)
decision = json.loads((root/'decision_config.json').read_text())
config = json.loads((root/'config.json').read_text())
limit = decision.get('max_length') or config['text_config']['max_position_embeddings']
protocol = json.loads((data/'protocol.json').read_text())
rows = []
for task, variants in protocol['variants'].items():
 for split in ('dev', 'holdout'):
  with (data/f'{task}-{split}.private.jsonl').open() as stream:
   jobs = [json.loads(line) for line in stream]
  for variant, questions in variants.items():
   q = next(iter(questions.values()))
   options = prompt.option_texts('choice', q['criteria'])
   for job in jobs:
    rendered = prompt.render(prompt.user_prompt(job['state'], q['instructions'], options, decision['codes'][:len(options)]))
    n = len(tok.encode(rendered, add_special_tokens=False).ids)
    rows.append({'task':task, 'split':split, 'variant':variant, 'case_id':job['case_id'],
                 'fragment_id':job['fragment_id'], 'tokens':n, 'supported':n<=limit,
                 'input_sha256':job['input_sha256']})
print(json.dumps({'limit':limit, 'rows':rows, 'inference_calls':0,
 'prompt_source_sha256':hashlib.sha256(Path(inspect.getfile(prompt)).read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256((root/'MODEL_MANIFEST.json').read_bytes()).hexdigest()}))
'''


def preflight(args):
    t.read_data(args.data)
    args.root.mkdir(parents=True, exist_ok=True)
    for model in t.MODELS:
        result = json.loads(command(["docker", "run", "--rm", "--network", "none", "--read-only", "--user", "0", "--cap-drop", "ALL",
            "--tmpfs", "/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python",
            "-v", f"{args.base / 'models' / model}:/model:ro", "-v", f"{args.data}:/data:ro",
            launcher.RUNTIME, "-B", "-c", PREFLIGHT]))
        t.save(args.root / f"preflight-{model}.json", result)
        assert all(r["supported"] for r in result["rows"]), "Native coverage failure; no inference started"
        assert max(r["tokens"] for r in result["rows"]) <= 32768
        print(json.dumps({"model": model, "encoded": len(result["rows"]),
                          "max_tokens": max(r["tokens"] for r in result["rows"]), "limit": result["limit"]}), flush=True)


async def ready(base_url, key):
    deadline = time.monotonic() + 2400
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            assert identity("decision-acceptance-model")["running"], "Model stopped during loading"
            try:
                response = await client.get(base_url + "/health", headers={"Authorization": "Bearer " + key})
                if response.status_code == 200:
                    for credential in (None, "incorrect-evaluation-key", key.swapcase()):
                        if credential == key:
                            continue
                        result = await client.get(base_url + "/health", headers={} if credential is None else
                                                  {"Authorization": "Bearer " + credential})
                        assert result.status_code == 401
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    raise TimeoutError("Model load deadline exceeded")


def cgroup_stats():
    pid = identity("decision-acceptance-model")["pid"]
    groups = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    unified = next((line.split(":", 2)[2] for line in groups if line.startswith("0::")), None)
    if unified is not None:
        root = Path("/sys/fs/cgroup") / unified.lstrip("/")
        return {"quota": (root / "cpu.max").read_text(), "stats": (root / "cpu.stat").read_text()}
    cpu = next(line.split(":", 2)[2] for line in groups if "cpu" in line.split(":")[1].split(","))
    root = Path("/sys/fs/cgroup/cpu,cpuacct") / cpu.lstrip("/")
    return {"quota": (root / "cpu.cfs_quota_us").read_text(), "stats": (root / "cpu.stat").read_text(),
            "usage_ns": int((root / "cpuacct.usage").read_text())}


class Session:
    def __init__(self, args, client, key, model, measure=None, native_family="worker", expected_tokens=None):
        self.args, self.client, self.model = args, client, model
        self.headers = {"Authorization": "Bearer " + key}
        self.measure = measure or t.capacity.measure
        assert native_family in ("worker", "sglang")
        self.native_family, self.expected_tokens = native_family, expected_tokens

    def request(self, job, question):
        path, body = native_choice_request(self.native_family, {
            "model": self.model, "state": job["state"], "questions": question})
        if self.native_family == "worker":
            body["options"] = {"return_meta": True}
        return path, body

    def decode(self, body, raw, job):
        prediction, probabilities = native_choice_result(self.native_family, body, raw)
        assert raw.get("model", self.model) == self.model
        if self.native_family == "worker":
            assert raw["meta"]["device"].startswith("cuda"), "GPU execution required"
        if self.expected_tokens is not None:
            assert raw["usage"]["input_tokens"] == self.expected_tokens[job["case_id"], job["fragment_id"]]
            assert raw["usage"]["output_tokens"] == 0
        return prediction, probabilities

    async def block(self, folder, name, jobs, question, n, **info):
        assert not (folder / (name + ".json")).exists()
        assert not (self.args.root / "STOP").exists()
        status(self.args.root, "measuring", model=self.model, block=name, directory=str(folder.relative_to(self.args.root)),
               concurrency=n, planned=len(jobs))
        response = await self.client.get(self.args.base_url + "/metrics", headers=self.headers)
        response.raise_for_status()
        before, cpu_before = response.text, cgroup_stats()
        began_utc = datetime.now(timezone.utc).isoformat()

        async def infer(job):
            path, body = self.request(job, question)
            meta = {"request_sha256": t.benchmark.digest(body), "calls": 1,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
            try:
                result = await self.client.post(self.args.base_url + path, json=body, headers=self.headers)
                meta.update(http_status=result.status_code,
                            response_sha256=hashlib.sha256(result.content).hexdigest())
                result.raise_for_status()
                raw = result.json()
                prediction, probabilities = self.decode(body, raw, job)
                return {**meta, "prediction": prediction, "probabilities": probabilities,
                        "response": {k: raw[k] for k in ("answers", "usage", "execution", "meta") if k in raw}}
            except Exception as error:
                error.capacity_metadata = meta
                raise

        rows, block = await self.measure(jobs, infer, n, 300)
        t.save_rows(folder / (name + ".jsonl"), rows)
        cpu_after = cgroup_stats()
        response = await self.client.get(self.args.base_url + "/metrics", headers=self.headers)
        response.raise_for_status()
        result = {**t.capacity.block_summary(rows, block), **info, "model": self.model,
                  "question": question, "started_at_utc": began_utc, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                  "metrics_before": before, "metrics_after": response.text,
                  "cpu_before": cpu_before, "cpu_after": cpu_after,
                  "p99_ms": t.benchmark.percentile([r["elapsed_ms"] for r in rows], .99)}
        t.save(folder / (name + ".json"), result)
        assert all(r["status"] == "valid" for r in rows), "Block failed; no further inference"
        return rows


async def quality(args, model, key, folder, protocol):
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0, limits=limits),
                                trust_env=False, follow_redirects=False, timeout=300) as client:
        session = Session(args, client, key, model)
        for task, choices in protocol["variants"].items():
            jobs = t.read_rows(args.data / f"{task}-dev.private.jsonl")
            for repeat in (1, 2):
                await session.block(folder, f"{task}-warm-r{repeat}", jobs[:16], choices["baseline"], 1,
                                    phase="warmup", task=task)
            # Same development items for all fixed candidates, reverse model order reduces one order bias.
            order = list(choices) if model == "d3-mini" else list(reversed(choices))
            by_variant = {}
            for variant in order:
                by_variant[variant] = await session.block(folder, f"{task}-dev-{variant}", jobs, choices[variant], 1,
                                                          phase="dev", task=task, variant=variant)
            by_variant = {v: by_variant[v] for v in choices}
            selected = t.selection(args.data, task, by_variant)
            t.save(folder / f"{task}-selection.json", selected)
            held = t.read_rows(args.data / f"{task}-holdout.private.jsonl")
            for variant in dict.fromkeys(("baseline", selected["variant"])):
                await session.block(folder, f"{task}-holdout-{variant}", held, choices[variant], 1,
                                    phase="holdout", task=task, variant=variant)


async def performance(args, model, key, folder, protocol, quality_folder):
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0, limits=limits),
                                trust_env=False, follow_redirects=False, timeout=300) as client:
        session = Session(args, client, key, model)
        for task in protocol["variants"]:
            selected = json.loads((quality_folder / f"{task}-selection.json").read_text())
            question = protocol["variants"][task][selected["variant"]]
            keys = {(r["case_id"], r["fragment_id"]) for r in protocol["performance"][task]}
            jobs = [r for r in t.read_rows(args.data / f"{task}-dev.private.jsonl")
                    if (r["case_id"], r["fragment_id"]) in keys]
            assert len(jobs) == 128
            ordered = sorted(jobs, key=lambda r: r["chars"])
            warmup = [ordered[i * (len(ordered)-1)//15] for i in range(16)]
            for n in t.LEVELS:
                for repeat in (1, 2):
                    await session.block(folder, f"{task}-warm-n{n}-r{repeat}", warmup, question, n,
                                        phase="warmup", task=task, threshold=selected["threshold"])
            for repeat, order in ((1, t.LEVELS), (2, tuple(reversed(t.LEVELS)))):
                for n in order:
                    await session.block(folder, f"{task}-n{n}-r{repeat}", jobs, question, n,
                                        phase="performance", task=task, repeat=repeat, threshold=selected["threshold"])


def launch(args, model, profile, folder):
    runtime = folder / "runtime"
    runtime.mkdir()
    (runtime / "models").symlink_to(args.base / "models", target_is_directory=True)
    launcher.launch(SimpleNamespace(root=runtime, model=model, env_file=args.env_file,
        native_port=30888, public_port=30889, enforce_eager=False, **t.PROFILES[profile]))
    t.save(folder / "deployment.json", json.loads((runtime / "acceptance-20261015/evidence" /
                                                   (model + ".deployment.json")).read_text()))
    status(args.root, "loading", model=model, profile=profile)
    asyncio.run(ready(args.base_url, launcher.read_key(args.env_file)))
    t.save(folder / "identity.json", identity("decision-acceptance-model"))


def run(args):
    os.umask(0o077)
    protocol = t.read_data(args.data)
    assert not (args.root / "run-protocol.json").exists(), "No automatic repeat or resume"
    for model in t.MODELS:
        pre = json.loads((args.root / f"preflight-{model}.json").read_text())
        assert pre["inference_calls"] == 0 and all(r["supported"] for r in pre["rows"])
        keys = {(r["case_id"], r["fragment_id"]) for rows in protocol["performance"].values() for r in rows}
        assert max(r["tokens"] for r in pre["rows"] if (r["case_id"], r["fragment_id"]) in keys) <= 8192
    lock = (args.base / "classification-20261015/classification.lock").open("r+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = [identity(name) for name in ("decision-acceptance-proxy", "decision-acceptance-model")]
    assert all(r["running"] and not r["oom"] for r in original)
    processes = {int(x.strip()) for x in command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).splitlines()}
    owned = {int(line.strip()) for line in command(["docker", "top", "decision-acceptance-model", "-eo", "pid"]).splitlines()[1:]}
    assert processes <= owned, "Foreign GPU process; refuse mutation"
    preserved = json.loads((args.base / "evidence-20261018/preparation-r3/preservation.json").read_text())
    for name in ("source-r3", "dataset-v1", "preparation-r3", "run-001", "offline-audit-20261019-online-variation-v1"):
        for path in (args.base / "evidence-20261018" / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                preserved[str(path)] = t.sha(path)
    t.capacity.check_artifacts(preserved)
    t.save(args.root / "preservation.json", preserved)
    sources = {str(p): t.sha(p) for p in t.REPO.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    t.save(args.root / "run-protocol.json", {"dataset_sha256": t.sha(args.data / "protocol.json"),
        "original_service": original, "source_sha256": sources, "models": t.MODELS,
        "profiles": t.PROFILES, "levels": t.LEVELS, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent pilot; no old benchmark rerun", "retries": 0})
    key = launcher.read_key(args.env_file)
    for row in original:
        command(["docker", "stop", "--time", "30", row["id"]])
        command(["docker", "rename", row["id"], row["name"] + "-held-d3pilot"])
    try:
        for model in t.MODELS:
            model_root = args.root / model
            model_root.mkdir()
            qfolder = model_root / "quality"
            qfolder.mkdir()
            launch(args, model, "baseline", qfolder)
            asyncio.run(quality(args, model, key, qfolder, protocol))
            # Reverse the two nonbaseline profiles between models; baseline remains the common anchor.
            profiles = ("baseline", "thread1", "compact") if model == "d3-mini" else ("baseline", "compact", "thread1")
            for profile in profiles:
                folder = model_root / profile
                folder.mkdir()
                launch(args, model, profile, folder)
                asyncio.run(performance(args, model, key, folder, protocol, qfolder))
                t.save(folder / "terminal-identity.json", identity("decision-acceptance-model"))
        t.capacity.check_artifacts(preserved)
        t.capacity.check_artifacts(sources)
        for name in ("decision-acceptance-proxy", "decision-acceptance-model"):
            launcher.remove_owned(name)
        for row in reversed(original):
            assert identity(row["name"] + "-held-d3pilot")["id"] == row["id"]
            command(["docker", "rename", row["id"], row["name"]])
            command(["docker", "start", row["id"]])
        asyncio.run(ready(args.base_url, key))
        restored = [identity(row["name"]) for row in original]
        assert all(a["id"] == b["id"] and a["image"] == b["image"] and a["command"] == b["command"]
                   and b["running"] for a,b in zip(original, restored))
        t.save(args.root / "restoration.json", {"before": original, "after": restored,
                                                "preserved_files": len(preserved), "checked": True})
        status(args.root, "complete", restored_original=True)
    except Exception as error:
        status(args.root, "failed", error_type=type(error).__name__,
               restoration="not automatic on failure; original containers retained")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--base-url", default="http://127.0.0.1:30889")
    args = p.parse_args()
    (preflight if args.action == "preflight" else run)(args)
