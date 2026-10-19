"""Exclusive one-shot orchestration of network-disabled reference diagnostics."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request


spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("2026-10-19-evidence-reference-check.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def command(parts):
    return subprocess.check_output(list(map(str, parts)), text=True).strip()


def identity(name):
    obj = json.loads(command(["docker", "inspect", name]))[0]
    assert obj["Config"]["Labels"].get("evaluation.decision.acceptance") == "true"
    return {"name": obj["Name"].lstrip("/"), "id": obj["Id"], "image": obj["Image"],
            "running": obj["State"]["Running"], "pid": obj["State"]["Pid"],
            "started_at": obj["State"]["StartedAt"], "oom": obj["State"]["OOMKilled"],
            "command": [obj["Path"], *obj["Args"]], "restart_count": obj["RestartCount"]}


def main(args):
    lock = (args.base / "classification-20261015/classification.lock").open("r+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (args.root / "startup.json").exists(), "No automatic resume"
    original = [identity(n) for n in ("decision-acceptance-proxy", "decision-acceptance-model")]
    assert all(o["running"] and not o["oom"] for o in original)
    gpu = {int(s) for s in command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).splitlines()}
    owned = {int(s) for s in command(["docker", "top", "decision-acceptance-model", "-eo", "pid"]).splitlines()[1:]}
    assert gpu <= owned, "Unknown GPU process"
    image = command(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"])
    assert image == "sha256:f3ccfb750e184c66e3aee485d64b7ed19cfc0ce83d73650b8b92b3463a3a66b4"
    preserved = r.read(args.base / "clef-runtime-20261019/preservation.json")
    for name in ("d3-tuning-20261019", "d3-runtime-20261019", "clef-runtime-20261019",
                 "evidence-reference-20261019", "evidence-reference-20261019-r2"):
        for path in (args.base / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                preserved[str(path)] = r.sha(path)
    assert all(r.sha(p) == h for p, h in preserved.items())
    r.save(args.root / "preservation.json", preserved)
    source = {str(p): r.sha(p) for p in (args.root / "source").rglob("*") if p.is_file()}
    r.save(args.root / "startup.json", {"original": original, "image": image, "source": source,
                                        "preserved_files": len(preserved), "planned": 1590})
    for obj in original:
        command(["docker", "stop", "--time", "30", obj["id"]])
    try:
        for model in ("d3-mini", "d3-flash"):
            print(json.dumps({"stage": "reference_start", "model": model}), flush=True)
            name = args.root.name + "-" + model
            cmd = ["docker", "run", "--name", name, "--gpus", "device=0", "--network", "none",
                   "--user", "0",
                   "--label", "evaluation.decision.acceptance=true", "--label", "evaluation.decision.diagnostic=evidence-reference-20261019",
                   "--read-only", "--cap-drop", "ALL", "--cpus", "8", "--memory", "96g", "--shm-size", "2g",
                   "--tmpfs", "/tmp:rw,size=8g", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                   "-e", "HF_HOME=/tmp/hf", "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-e", "OMP_NUM_THREADS=2", "-v", f"{args.base / 'models' / model}:/model:ro",
                   "-v", f"{args.base / 'd3-tuning-20261019'}:/prior:ro", "-v", f"{args.root / 'source'}:/source:ro",
                   "-v", f"{args.root / 'results'}:/output", "--entrypoint", "python", image, "-B",
                   "/source/2026-10-19-evidence-reference-check.py", "infer", "--model", model,
                   "--data", "/prior/dataset", "--old", f"/prior/{model}/quality/evidence-holdout-baseline.jsonl",
                   "--package", "/model", "--lock", "/source/d3-model-lock.json", "--output", f"/output/{model}"]
            with (args.root / (model + ".log")).open("x") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            state = identity(name)
            r.save(args.root / (model + ".terminal.json"), {"container": state, "exit_code": completed.returncode})
            assert completed.returncode == 0 and not state["running"] and not state["oom"]
            assert r.read(args.root / "results" / model / "complete.json")["valid"] == 795
        assert all(r.sha(p) == h for p, h in preserved.items())
        assert all(r.sha(p) == h for p, h in source.items())
        for obj in reversed(original):
            current = identity(obj["id"])
            assert current["image"] == obj["image"] and current["command"] == obj["command"]
            command(["docker", "start", obj["id"]])
        deadline = time.monotonic() + 2400
        while True:
            assert all(identity(o["id"])["running"] for o in original)
            try:
                with urllib.request.urlopen("http://127.0.0.1:30888/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline, "Restore health deadline"
            time.sleep(3)
        r.save(args.root / "complete.json", {"valid": 1590, "original": original,
               "restored": [identity(o["id"]) for o in original], "history_unchanged": True,
               "native_health": 200})
        print(json.dumps({"stage": "complete_restored", "valid": 1590}), flush=True)
    except BaseException as exc:
        r.save(args.root / "failed.json", {"error_type": type(exc).__name__,
               "original": [identity(o["id"]) for o in original], "automatic_resume": False})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--image", default="decision-eval-runtime:readiness-20261015")
    main(p.parse_args())
