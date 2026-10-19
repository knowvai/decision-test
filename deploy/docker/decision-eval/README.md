# Decision Evaluation Services

## Reviewed Native Deployments (2026-10-15)

The current 15-refiner scope (13 Decision models and two LLM baselines) is defined in
[the benchmark plan](../../../docs/plans/2026-10-14-decision-benchmark-plan.md),
not the retained legacy Compose files or archived launch examples. `launch_reviewed.py` starts one model
at a time using the installed official worker/engine and a protected HTTP proxy.
It only replaces containers carrying its evaluation ownership label, refuses
occupied ports or unapproved GPU compute processes, and never changes business services.
The retained preflight/legacy evaluation containers must be stopped explicitly
before rotating through the reviewed paths.

For the owner-approved two-host passage test, only the reranker may explicitly
share a GPU with one existing container. Pass its full 64-character ID through
`--cohost-container-id`, together with `--gpu-memory-utilization`, `--memory-gib`
and `--cpus`. The launcher verifies the running identity and rejects every GPU
PID outside that container; it never stops the cohost. All defaults and the
exclusive-GPU guard remain unchanged when these flags are absent. The owner-selected
proxy profile is 0.80 engine GPU-memory budget, 32 GiB container RAM and 4 CPUs,
with seq16/tokens16384. GPU-memory utilization is not a hardware partition or a
hard VRAM quota, and CPU limits are not CPU affinity. Disclose cohost contention,
monitor resources and include inter-host transport in measured chain time.
Keep source and evidence in a new deployment root; do not overwrite old runs.

```sh
python3 launch_reviewed.py d3-mini --root "$EVAL_ROOT" --env-file "$EVAL_ENV_FILE"
python3 deployment_probe.py d3-mini --root "$EVAL_ROOT" --env-file "$EVAL_ENV_FILE"
```

Use the existing private evaluation env file; do not generate a new credential.
The public default is HTTP port 30889. Native engines bind only loopback 30888;
NeoHorse's backbone uses loopback 30887 behind its CPU-head adapter. The proxy
authenticates every route, including health, with the existing bearer key.
Generated secret files are owner-only and stay outside this repository. No TLS
is added. Changing the selected model changes the native API served at this
endpoint: SystemOne for worker/SDK models, `/v1/decide` for JEV, `/rerank` for
Qwen Reranker, and OpenAI chat completions for the ordinary Qwen LLM.

The readiness repair extends the already accepted runtime without fetching
packages. Confirm the base image ID against the execution evidence first:

```sh
docker build --pull=false --network none -f Dockerfile.acceptance \
  --build-arg BASE_IMAGE=decision-eval-runtime:acceptance-20261015 \
  -t decision-eval-runtime:readiness-20261015 .
```

JEV uses a separate derivative of the pinned vLLM image. Its hash-checked patch
adds opt-in LoRA loading and transfer verification, without changing the forward pass:

```sh
docker build --pull=false --network none -f Dockerfile.vllm-lora \
  -t decision-eval-vllm:jev-verified-lora-r2-20261015 .
```

The launcher records the resolved image ID, commands and non-secret settings.
It disables decision result caches and vLLM prefix caching. SGLang's specialized
noncausal/joint-head path owns its required cache and chunking restrictions.
Models use the same eight-CPU budget; NeoHorse splits it between vLLM and CPUHead.
Single-card and TP=2 runs remain separate resource groups.
JEV retains native vLLM eager execution, but a subsequent cold start also
failed in eager mode. Copy-only diagnostics reproduced corrupt transfers from
some pinned CPU LoRA buffers on this appliance. The JEV profile now explicitly
sets `VLLM_LORA_DISABLE_PIN_MEMORY=1` for actual and dummy adapter loading,
plus `VLLM_LORA_VERIFY_TRANSFER=1`. Disabling pinning alone passed isolated
copy tests but failed real warmup in run-004, so that derivative is not accepted.
The second flag takes independent unpinned CPU snapshots and verifies exact
GPU transfer before native per-layer installation. A mismatch fails explicitly;
there is no retry or numeric replacement. GPU inference, BF16, TP=2, weights
and decision math remain unchanged. Both flags default off and are not applied to Qwen,
NeoHorse or SGLang; it is not CPU inference or automatic fallback. Retire it
only after transfer and cold-start checks pass on a corrected runtime/driver.
The exact underlying driver/allocator defect remains unproven. The launcher
supports `--enforce-eager` for diagnostic runs of the other vLLM paths too.
The first verified-transfer image failed startup because embedding layers do
not expose `.device`; the r2 image uses the native manager's target device.
Failed attempts and current cold-start acceptance are recorded in the
[classification execution record](../../../docs/research/2026-10-15-decision-classification-execution.md).
Native checkpoint prefetch is enabled for vLLM/SGLang launches; it affects cold
loading only and must not be counted as a measured inference speed improvement.

For SDK backends, `SYSTEMONE_QUEUE_SINGLE_REQUESTS=1` opts into the existing
bounded queue at batch size one. Overflow returns 429 and expiration returns
504; serial execution explicitly reports no native batch. Defaults remain
unchanged for historical deployments. Jeva does not gain cross-request GPU batching merely by enabling this queue.
Intern remains in the launcher registry for historical reproducibility only and
is excluded from the current benchmark. NeoHorse preserves the author's encoder
and CPUHead and batches encoded records through vLLM pooling, without replacing
its model head. Laya uses the official HTTP/batch application with a supported
input hook reusing the existing state/head/option truncation guard.

NeoHorse's authenticated health route also checks native vLLM readiness with a
bounded health GET. It returns 503 while pooling is unavailable, so the existing
runner waits instead of sending warmup to an unloaded backbone. Other native
inference paths and the successful health payload are unchanged.

`deployment_probe.py` is a small synthetic transport/runtime check, not a second
benchmark runner. The legacy smoke script assumes older endpoint/length contracts;
this probe exercises the current native APIs, 18 options, single-versus-pool
outputs and the 49-request acceptance boundary. Probe timings include first-use
compilation and are not comparable performance results. The probe's default startup allowance is 2400
seconds for cold large-model loading, configurable with `--startup-timeout`.
Inference requests still have a separate 300-second deadline. The public-data runner
and the plan's limited engineering-pool tuning remain authoritative for formal
quality and latency measurements. Preserve failed probe attempts as well as the
latest result; no hidden retries, ranking or threshold selection occur here.


## Current Evaluation Entry Points

- Scope: [current plan](../../../docs/plans/2026-10-14-decision-benchmark-plan.md).
- Runtime evidence: [deployment remediation](../../../docs/research/2026-10-15-decision-deployment-remediation.md).
- Scenario A: [classification execution](../../../docs/research/2026-10-15-decision-classification-execution.md), through `public_decision_benchmark.py local-classification`.
- Scenario B is not running yet. Native batching, isolated threshold calibration and the ordinary Qwen score path still require the checks in the plan.

Classification uses 13 Decision models and one ordinary local Qwen LLM, one
request in flight, with no load test. Qwen3-Reranker-8B only enters scenario B.
Intern, D3-27B and Clef-27B are excluded. The launcher registry is not the roster.

The frozen classification queue is already active. Do not run the launch/probe
examples against its ports or GPUs, rebuild its images, replace its snapshots,
or move its data while it is running. The original queue's STOP exit records
an intentional roster change, not a model failure.

## Retained Reproduction Files

| Files | Role |
| --- | --- |
| `launch_reviewed.py`, `deployment_probe.py`, `nginx.native.conf.template` | Current launcher, native mapping and authenticated proxy |
| `server.py`, `jeva_backend.py`, `neohorse_backend.py`, `laya_http.py` | Current thin adapters and reused input/queue safeguards |
| `Dockerfile.acceptance`, `Dockerfile.preflight`, `Dockerfile.sglang`, `patch_sglang_length.py`, `qwen3_reranker.jinja` | Build chain, strict-length guard and official template |
| Other `Dockerfile*`, `compose*.yml`, `nginx.conf`, `decision_reranker.py`, `qwen_reranker.py`, `smoke_test.py` | Retained reproduction/test dependencies; not the current launch entry |
| Lock JSON files, `download_models.py`, `initialize_runtime.py` | Acquisition/provenance support; no new downloads implied |

Build contexts, imports and historical tests still reference these files.
Do not use `docker compose up` here as a shortcut to the reviewed rotation.
[Historical examples](../../../docs/archive/decision-eval/pre-2026-10-15/deployments/2026-10-15-deployment-examples.md)
and [old results](../../../docs/archive/decision-eval/pre-2026-10-15/README.md)
are isolated from this runbook; their protocols and timings are not current.

HTTP without TLS is user-selected. Use public/synthetic inputs and a trusted
network or SSH tunnel, not this setup as a production security recommendation.
Credentials and private evidence remain outside the repository. Cleanup does
not remove weights, images, caches or run artifacts.
