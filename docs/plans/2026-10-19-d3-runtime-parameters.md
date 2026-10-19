# D3 Runtime Parameter Experiment

## Authority And Frozen Flow

Owner requested actual D3-mini / D3-flash runtime tuning verification after the
completed quality/throughput pilot. This is an isolated evaluation experiment,
not a rerun of historical benchmarks or a production rollout. Preserve all old
evidence, weights, precision, native readout, input text, and authentication.

Frozen flow: prior pilot's hash-checked performance input IDs -> original baseline
prompt / native choice -> authenticated HTTP -> native vllm-srun 0.2.0 scheduler
and Decision3 engine -> existing native decoder -> immutable records -> offline
audit. Reuse Session, wire, measurement, metrics, and resource helpers. Only add
an opt-in launcher switch omitting seq/token flags for native automatic budgets;
all historical defaults stay unchanged. The Session accepts an optional measurement
adapter so this experiment can fail closed without changing the completed pilot.

## Source Findings

Pinned runtime image:
`sha256:f3ccfb750e184c66e3aee485d64b7ed19cfc0ce83d73650b8b92b3463a3a66b4`.
Native `max-num-seqs` caps forward rows, token budget caps forward cost, and batch
window controls collection delay. `threads` controls CPU threads. The SDK's
MAX_INFLIGHT_REQUESTS/SYSTEMONE_BATCH_SIZE variables do not control this native path.
Native auto mode resolves device defaults then runs internal synthetic startup
calibration; report it separately from HTTP workload calls. Decision3 also adapts
its coalesced budget toward a 300 ms forward target. That internal budget is not
a supported CLI control and its live value is not externally measured here.
The 4096-token graph threshold selects eager execution for larger shapes; it is
not a request length limit. No engine, kernel, or scheduler patch is authorized.

## Frozen Matrix

Same A100 GPU 0, 8 CPU quota, 96 GiB host memory limit, threads=2, unchanged image,
offline model, result cache off, shared context off, full input, no HTTP retries.

| Profile | max-num-seqs | max-num-batched-tokens | window ms |
| --- | ---: | ---: | ---: |
| baseline | 32 | 32768 | 2 |
| window0 | 32 | 32768 | 0 |
| window5 | 32 | 32768 | 5 |
| window10 | 32 | 32768 | 10 |
| seq8 | 8 | 32768 | 2 |
| tokens8192 | 32 | 8192 | 2 |
| tokens65536 | 32 | 65536 | 2 |
| auto | native device default | native calibration | 2 |
| baseline-end | 32 | 32768 | 2 |

Baseline anchors each model; reverse the seven alternatives for flash. Each
profile gets a fresh runtime directory/cold process. Each task uses the prior
frozen 128-input performance subset but the original baseline prompt, not the
previous pilot's selected concise prompt or evidence threshold. This is reused
runtime-regression data, not fresh unseen quality evaluation or full-pool recall.
All input encodings must fit the smallest explicit 8192 budget before mutation.

Warmup: 32 length-spread inputs at n1 and n32, twice each per task. Performance:
all 128 inputs at n1/8/16/32, two rounds in opposite concurrency orders; reverse
input order in the second round. Record every warmup and measured call. Total
planned 41472 HTTP calls: 4608 warmup, 36864 measured; 432 blocks, 18 deployments.
No best-run filtering. Baseline-end checks drift, but each nonbaseline deployment
is only one replicate; this screening cannot establish long-run capacity or SLA.

## Measurement And Decision Rules

Record effective valid RPS using whole-block wall, per-request mean/P95/P99,
failures/timeouts/not-attempted, in-flight intervals, request/input hashes,
prediction/probabilities, Prometheus deltas, cgroup CPU and 1-second GPU samples.
Save startup-resolved limits and model cards before measured calls. Startup,
calibration and warmup are excluded from measured RPS, disclosed separately.
Check exact native decisions against baseline and n1; report probability drift
separately. Report sample accuracy/P/R/F1 without calling repeated inputs new data.

Call a profile only a promising candidate when both measured rounds improve
same-concurrency RPS by >=5%, pooled P95 is <=110% of baseline, no request failure,
no decision regression, and baseline-end RPS drift is <=10%. Otherwise report
no verified gain or a throughput/latency tradeoff. No automatic follow-on or
production selection. No quality prompt/threshold tuning in this runtime round.

## Safety And Audit

New root `/data/sdb/decision-model-eval/d3-runtime-20261019`; no resume/retry.
Nonblocking shared classification.lock plus exact expected original container IDs,
images/commands and GPU ownership checks before stopping anything. Retain original
acceptance model/proxy containers under new names; only replace this experiment's
containers. Restore original IDs/names/images/commands after normal completion;
their startup time/PID will change. Never touch proxy-host business embedding,
reranker or proxy; independently verify their identities before and after.

Load/JIT budget 2400 seconds, single request 300 seconds. First observed inference
failure prevents new requests; already in-flight calls drain only to their existing
deadline. Keep not-attempted separate. On failure preserve containers/evidence and
do not automatically restart, resume, restore or send probes. A client timeout
does not prove the GPU has stopped. Save hashes of all old evidence and the new
frozen source. Independent offline audit uses no inference/network/subprocess and
writes new files; verify external service terminal state separately.
