"""Launch one isolated, reviewed evaluation runtime; never changes business services."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess


RUNTIME = "decision-eval-runtime:readiness-20261015"
VLLM = "decision-eval-vllm:v0.31.0-pinned"
JEV_VLLM = "decision-eval-vllm:jev-verified-lora-r2-20261015"
SGLANG = "decision-eval-sglang:clef-strict-length-v1"
LABEL = "evaluation.decision.acceptance"
MODELS = {
    **{name: ("worker", "models/" + name, name) for name in ("d3-edge", "d3-mini", "d3-flash")},
    **{"vela-2.0-" + size: ("worker", "staging/vela-2.0-" + size, "Vela-2.0-" + size.upper())
       for size in ("0.3b", "0.8b", "4b", "9b")},
    "jeva-4b": ("sdk", "staging/jeva-4b", "Jeva-4B"),
    "intern-4b": ("sdk", "models/intern-4b", "Intern-Decision-4B"),
    "laya-multilingual": ("laya", "models/laya-multilingual", "laya-multilingual"),
    "neohorse-jev-4b": ("neohorse", "staging/neohorse-jev-4b", "NeoHorse-Jev-4B"),
    "clef-flash": ("sglang", "models/clef-flash", "clef-flash"),
    "pplx-decider-v1.1-27b": ("sglang", "staging/pplx-decider-v1.1-27b", "pplx-decider-v1.1-27b"),
    "jev-27b": ("jev", "staging/jev-27b", "autotrust/JEV-27B"),
    "qwen3-reranker-8b": ("reranker", "models/qwen3-reranker-8b", "Qwen3-Reranker-8B"),
    "qwen3-30b-a3b-instruct-2507": ("llm", "staging/qwen3-30b-a3b-instruct-2507", "Qwen3-30B-A3B-Instruct-2507"),
    "qwen3.8-27b": ("llm", "staging/qwen3.8-27b", "Qwen3.8-27B"),
}
CLASSIFICATION_PROFILES = {
    "qwen3.8-27b-thinking": {
        "base_model": "qwen3.8-27b", "enable_thinking": True,
        "reasoning_effort": "xhigh", "reasoning_parser": "qwen3", "max_tokens": 4096,
    },
}
MODELS.update({name: MODELS[profile["base_model"]] for name, profile in CLASSIFICATION_PROFILES.items()})


def docker(*args):
    return subprocess.check_output(["docker", *map(str, args)], text=True).strip()


def read_key(path):
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "DECISION_EVAL_API_KEY":
            value = value.strip().strip("\"'")
            if re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value):
                return value
    raise ValueError("Missing or invalid independent evaluation key")


def remove_owned(name):
    result = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if result.returncode:
        return
    container = json.loads(result.stdout)[0]
    if container["Config"].get("Labels", {}).get(LABEL) != "true":
        raise RuntimeError("Refusing to replace an unowned container")
    if container["State"]["Running"]:
        docker("stop", "--time", 30, name)
    docker("rm", name)


def write_proxy_config(source, work, key, native_port, public_port):
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", key):
        raise ValueError("Invalid evaluation key")
    conf = (source / "nginx.native.conf.template").read_text()
    for name, value in {"DECISION_EVAL_API_KEY": key, "EVAL_PUBLIC_PORT": public_port,
                        "EVAL_NATIVE_PORT": native_port}.items():
        conf = conf.replace("${" + name + "}", str(value))
    path = work / ".nginx.conf"
    path.write_text(conf)
    path.chmod(0o600)
    return path


def resource_settings(args, family):
    cohost = getattr(args, "cohost_container_id", None)
    overrides = [getattr(args, name, None) for name in
                 ("gpu_memory_utilization", "memory_gib", "cpus")]
    if (cohost is not None or any(value is not None for value in overrides)) and family != "reranker":
        raise ValueError("Explicit shared-host resources are only supported for the reranker")
    if cohost is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", cohost) or any(value is None for value in overrides):
            raise ValueError("Cohost requires an exact container ID and all three resource limits")
    gpu, memory, cpus = [default if value is None else value
                         for default, value in zip((0.90, 96, 8), overrides)]
    if (type(gpu) not in (int, float) or not math.isfinite(gpu) or not 0 < gpu < 1
            or type(memory) is not int or memory < 1
            or type(cpus) not in (int, float) or not math.isfinite(cpus) or cpus <= 0):
        raise ValueError("Invalid resource limits")
    return {"cohost": cohost, "gpu": "0.90" if overrides[0] is None else str(gpu),
            "memory_gib": memory, "cpus": cpus}


def cohost_identity(container_id):
    template = ('{"id":{{json .Id}},"image_id":{{json .Image}},'
                '"running":{{json .State.Running}},"pid":{{json .State.Pid}},'
                '"started_at":{{json .State.StartedAt}},'
                '"owned":{{json (index .Config.Labels "evaluation.decision.acceptance")}}}')
    record = json.loads(docker("inspect", container_id, "--format", template))
    if (record["id"] != container_id or record["running"] is not True
            or record["pid"] <= 0 or record["owned"] == "true"):
        raise RuntimeError("Cohost must be the exact running non-evaluation container")
    return record


def check_gpu_processes(cohost=None):
    before = cohost_identity(cohost) if cohost else None
    allowed = set()
    if cohost:
        rows = docker("top", cohost, "-eo", "pid").splitlines()
        if not rows or rows[0].strip() != "PID":
            raise RuntimeError("Unable to establish cohost GPU process ownership")
        allowed = {int(row.strip()) for row in rows[1:]}
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    pids = {int(row.strip()) for row in active.splitlines()} if active else set()
    if pids - allowed:
        raise RuntimeError("GPU has unapproved compute processes; no business service will be stopped")
    if cohost and cohost_identity(cohost) != before:
        raise RuntimeError("Cohost identity changed during GPU inspection")
    return {**before, "gpu_pids": sorted(pids)} if before else None


def launch(args):
    os.umask(0o077)
    root = args.root.resolve(strict=True)
    work = root / "acceptance-20261015"
    work.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parent
    family, relative_path, model_name = MODELS[args.model]
    resources = resource_settings(args, family)
    worker_threads = getattr(args, "worker_threads", None)
    batch_window = getattr(args, "batch_window_ms", None)
    worker_auto_budget = getattr(args, "worker_auto_budget", False)
    sglang_metrics = getattr(args, "sglang_metrics", False)
    if type(sglang_metrics) is not bool or (sglang_metrics and family != "sglang"):
        raise ValueError("SGLang metrics require the SGLang family")
    if type(worker_auto_budget) is not bool or (worker_auto_budget and family != "worker"):
        raise ValueError("Automatic batching budgets require the native worker family")
    if worker_threads is not None and (family != "worker" or type(worker_threads) is not int
                                       or not 1 <= worker_threads <= 8):
        raise ValueError("Worker threads require the worker family and an integer in 1..8")
    if batch_window is not None and (family != "worker" or type(batch_window) not in (int, float)
                                     or not math.isfinite(batch_window) or not 0 <= batch_window <= 20):
        raise ValueError("Batch window requires the worker family and finite milliseconds in 0..20")
    profile = CLASSIFICATION_PROFILES.get(args.model)
    laya_admission = getattr(args, "laya_max_concurrent", None)
    if laya_admission is not None and (family != "laya" or type(laya_admission) is not int
                                       or not 1 <= laya_admission <= 256):
        raise ValueError("Laya admission override requires the Laya profile and an integer in 1..256")
    if args.enforce_eager and family not in {"jev", "llm", "reranker"}:
        raise ValueError("Eager diagnostic override is only supported for these vLLM paths")
    model = (root / relative_path).resolve(strict=True)
    key = read_key(args.env_file)
    port = args.native_port
    if resources["cohost"]:
        check_gpu_processes(resources["cohost"])
    for name in ("decision-acceptance-proxy", "decision-acceptance-head", "decision-acceptance-model"):
        remove_owned(name)
    for used_port in (port, port - 1, args.public_port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", used_port))
    cohost = check_gpu_processes(resources["cohost"])
    cache = work / "runtime" / args.model
    cache.mkdir(parents=True, exist_ok=True)
    env = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/runtime/hf",
        "HOME": "/runtime", "OMP_NUM_THREADS": "2", "TRITON_CACHE_DIR": "/runtime/triton",
        "XDG_CACHE_HOME": "/runtime/cache", "TORCHINDUCTOR_CACHE_DIR": "/runtime/inductor",
        "SGLANG_CACHE_DIR": "/runtime/sglang", "SGLANG_JIT_CACHE_DIR": "/runtime/sglang/jit",
        "VLLM_CACHE_ROOT": "/runtime/vllm", "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ENABLE_CUDA_COMPATIBILITY": "1", "DECISION_EVAL_API_KEY": key,
        "MODEL_NAME": model_name, "MODEL_PATH": "/model", "GPU_MEMORY_FRACTION": resources["gpu"],
        "MAX_INPUT_TOKENS": "8192", "MAX_INFLIGHT_REQUESTS": "1",
        "SYSTEMONE_QUEUE_SINGLE_REQUESTS": "1", "SYSTEMONE_QUEUE_SIZE": "256",
        "SYSTEMONE_BATCH_SIZE": "1", "SYSTEMONE_TIMEOUT_SECONDS": "300",
    }
    image, tp = RUNTIME, 1
    command = []
    base_vllm = ["--model", "/model", "--served-model-name", model_name, "--dtype", "bfloat16",
                 "--host", "127.0.0.1", "--port", str(port), "--max-model-len", "8192",
                 "--max-num-seqs", str(args.seqs), "--max-num-batched-tokens", str(args.tokens),
                 "--gpu-memory-utilization", resources["gpu"], "--no-enable-prefix-caching",
                 "--safetensors-load-strategy", "prefetch", "--safetensors-prefetch-num-threads", "2"]
    # This pinned JEV/LoRA deployment returned NaN with compiled execution.
    # Its accepted profile keeps native vLLM batching but uses eager execution.
    if args.enforce_eager or family == "jev":
        base_vllm.append("--enforce-eager")
    if family == "worker":
        if worker_threads is not None:
            env["OMP_NUM_THREADS"] = str(worker_threads)
        command = ["vllm-srun", "serve", "/model", "--device", "cuda:0", "--served-model-name", model_name,
                   "--host", "127.0.0.1", "--port", str(port), "--dtype", "auto", "--offline",
                   "--threads", str(worker_threads if worker_threads is not None else 2), "--max-queue", "256",
                   "--result-cache-entries", "0", "--no-enable-shared-context", "--load-attempts", "1"]
        if not worker_auto_budget:
            command += ["--max-num-seqs", str(args.seqs), "--max-num-batched-tokens", str(args.tokens)]
        if batch_window is not None:
            command += ["--batch-window-ms", str(batch_window)]
    elif family == "sdk":
        env["MODEL_BACKEND"] = "jeva" if args.model == "jeva-4b" else "intern"
        command = ["python", "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
                   "--workers", "1", "--no-access-log", "--limit-concurrency", "512"]
    elif family == "laya":
        env.update(LAYA_DEVICE="cuda", LAYA_MODELS=model_name,
                   LAYA_EXTRA_MODELS=json.dumps({model_name: "/model"}), LAYA_DEFAULT_MODEL=model_name,
                   LAYA_API_KEY=key, LAYA_MAX_TOKEN_BUDGET="8192", LAYA_THREADS="2")
        if laya_admission is not None:
            env["LAYA_MAX_CONCURRENT"] = str(laya_admission)
        command = ["python", "-m", "uvicorn", "laya_http:create_app", "--factory",
                   "--host", "127.0.0.1", "--port", str(port), "--no-access-log"]
    elif family == "neohorse":
        image = VLLM
        command = ["python3", "/model/infer/vllm/launch.py", "--bundle", "/model", "--port", str(port - 1)]
    elif family == "sglang":
        image = SGLANG
        tp = 2 if args.model.startswith("pplx-") else 1
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        command = ["python3", "-m", "sglang.launch_server", "--model-path", "/model",
                   "--served-model-name", model_name, "--dtype", "bfloat16", "--tp-size", str(tp),
                   "--context-length", "8192", "--max-running-requests", str(args.seqs),
                   "--max-prefill-tokens", str(args.tokens), "--mem-fraction-static", "0.85",
                   "--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads", "2",
                   "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
        if sglang_metrics:
            command.append("--enable-metrics")
    else:
        image = VLLM
        tp = 1 if family == "reranker" else 2
        env["VLLM_API_KEY"] = key
        if family == "jev":
            image = JEV_VLLM
            command = ["python3", "/model/serve_decide.py", *base_vllm,
                       "--tensor-parallel-size", "2", "--enable-lora", "--max-lora-rank", "32",
                       "--lora-modules", "jev-decision=/model/adapter_vllm", "--logprobs-mode", "processed_logprobs"]
            # This appliance corrupts some pinned host-to-device LoRA transfers.
            # Only adapter loading changes; model execution remains on the GPUs.
            env.update(JEV_DECIDE_THINKING="off", JEV_DECIDE_STRATEGY="single",
                       VLLM_LORA_DISABLE_PIN_MEMORY="1", VLLM_LORA_VERIFY_TRANSFER="1")
        else:
            command = ["python3", "-m", "vllm.entrypoints.openai.api_server", *base_vllm,
                       "--tensor-parallel-size", str(tp)]
            if (profile or {}).get("base_model", args.model) == "qwen3.8-27b":
                template = ({key: profile[key] for key in ("enable_thinking", "reasoning_effort")}
                            if profile else {"enable_thinking": False})
                command += ["--language-model-only", "--default-chat-template-kwargs",
                            json.dumps(template)]
                if profile:
                    command += ["--reasoning-parser", profile["reasoning_parser"]]
            if family == "reranker":
                command += ["--runner", "pooling", "--hf-overrides", json.dumps({
                    "architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True}), "--chat-template", "/service/qwen3_reranker.jinja"]
    env_path = work / ".runtime.env"
    env_path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    image_id = docker("image", "inspect", image, "--format", "{{.Id}}")
    runtime_user = docker("image", "inspect", image, "--format", "{{.Config.User}}")
    if runtime_user == "10001:10001":
        os.chown(cache, 10001, 10001)
    common = ["--network", "host", "--label", LABEL + "=true",
              "--memory", str(resources["memory_gib"]) + "g", "--shm-size", "8g",
              "--read-only", "--tmpfs", "/tmp:size=4g,mode=1777", "--cap-drop", "ALL",
              "--security-opt", "no-new-privileges:true", "--env-file", env_path,
              "-v", str(model) + ":/model:ro", "-v", str(cache) + ":/runtime",
              "-v", str(source) + ":/service:ro"]
    docker("run", "-d", "--name", "decision-acceptance-model", "--gpus", "all" if tp == 2 else "device=0",
           "--cpus", "6" if family == "neohorse" else str(resources["cpus"]),
           *common, "--entrypoint", command[0], image_id, *command[1:])
    if family == "neohorse":
        docker("run", "-d", "--name", "decision-acceptance-head", "--cpus", "2", *common,
               "-e", "MODEL_BACKEND=neohorse", "-e", f"NEOHORSE_POOLING_URL=http://127.0.0.1:{port - 1}",
               "-e", "SYSTEMONE_BATCH_SIZE=8", "--entrypoint", "python", RUNTIME,
               "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
               "--no-access-log", "--limit-concurrency", "512")
    config_path = write_proxy_config(source, work, key, port, args.public_port)
    docker("run", "-d", "--name", "decision-acceptance-proxy", "--network", "host",
           "--label", LABEL + "=true", "--read-only", "--tmpfs", "/tmp",
           "-v", str(config_path) + ":/etc/nginx/nginx.conf:ro", "--entrypoint", "nginx", "nginx:1.27-alpine",
           "-g", "daemon off;")
    scheduling = {"max_sequences": args.seqs, "max_batched_tokens": args.tokens}
    if worker_auto_budget:
        scheduling = {"max_sequences": None, "max_batched_tokens": None,
                      "source": "native device defaults and startup calibration; inspect resolved runtime limits"}
    if family == "worker" and batch_window is not None:
        scheduling["batch_window_ms"] = batch_window
    if family == "neohorse":
        scheduling = {"max_sequences": 128, "max_batched_tokens": 32768,
                      "source": "author launch.py", "adapter_max_records": 8}
    elif family == "sdk":
        scheduling = {"max_sequences": 1, "max_batched_tokens": None,
                      "cross_request_batching": False, "queue_size": 256}
    elif family == "laya":
        scheduling = {"max_sequences": None, "max_batched_tokens": None,
                      "source": "native batch request batch_size and token budget"}
        if laya_admission is not None:
            scheduling.update(http_admission_limit=laya_admission,
                              single_request_executor_workers=1, cross_request_batching=False)
    record = {"model": args.model, "served_model": model_name, "family": family, "image_id": image_id,
              "at": datetime.now(timezone.utc).isoformat(),
              "tp": tp, "cpu_budget": resources["cpus"], "command": command, "scheduling": scheduling,
              "host_memory_limit_gib": resources["memory_gib"],
              "env": {k:v for k,v in env.items() if "KEY" not in k}, "native_port": port, "public_port": args.public_port}
    if profile:
        record["classification_profile"] = profile
    if cohost:
        record["cohost"] = cohost
        record["gpu_isolation"] = "shared; engine memory budget is not a hardware partition"
    if family == "neohorse":
        record["head_image_id"] = docker("image", "inspect", RUNTIME, "--format", "{{.Id}}")
    record["source_sha256"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in ("launch_reviewed.py", "server.py", "neohorse_backend.py", "jeva_backend.py",
                                            "laya_http.py", "deployment_probe.py", "nginx.native.conf.template")}
    if family == "jev":
        record["source_sha256"]["patch_vllm_lora.py"] = hashlib.sha256(
            (source / "patch_vllm_lora.py").read_bytes()).hexdigest()
    if family == "reranker":
        record["source_sha256"]["qwen3_reranker.jinja"] = hashlib.sha256(
            (source / "qwen3_reranker.jinja").read_bytes()).hexdigest()
    evidence = work / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / (args.model + ".deployment.json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"model": args.model, "family": family, "image_id": image_id, "tp": tp}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--native-port", type=int, default=30888)
    parser.add_argument("--public-port", type=int, default=30889)
    parser.add_argument("--seqs", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--worker-threads", type=int, help="Opt-in native worker CPU threads (1..8)")
    parser.add_argument("--batch-window-ms", type=float, help="Opt-in native worker batch collection delay (0..20 ms)")
    parser.add_argument("--worker-auto-budget", action="store_true",
                        help="Native worker only: omit seq/token flags, overriding --seqs/--tokens for device calibration")
    parser.add_argument("--sglang-metrics", action="store_true", help="Opt-in SGLang Prometheus instrumentation")
    parser.add_argument("--laya-max-concurrent", type=int,
                        help="Explicit Laya HTTP admission override; does not enable GPU batching")
    parser.add_argument("--enforce-eager", action="store_true", help="Explicit vLLM numerical diagnostic; recorded separately")
    parser.add_argument("--cohost-container-id", help="Reranker only: exact approved running container ID; never stopped")
    parser.add_argument("--gpu-memory-utilization", type=float, help="Reranker engine VRAM budget, not hardware isolation")
    parser.add_argument("--memory-gib", type=int, help="Reranker container host memory limit")
    parser.add_argument("--cpus", type=float, help="Reranker container CPU limit")
    launch(parser.parse_args())
