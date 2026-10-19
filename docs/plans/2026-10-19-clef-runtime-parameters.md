# Clef-Flash Runtime Parameter Experiment

## Authority And Baseline

Owner explicitly requested Clef-Flash parameter/concurrency testing. This is a
new isolated experiment, not a restart of old runs or a production change.
Reuse the reviewed launcher, native SystemOne adapter, D3 pilot Session, fixed
performance inputs and closed-loop measurement. Only add opt-in SGLang metrics
and a family-specific request/response adapter to the existing Session. The
frozen historical sources and results remain untouched.

Frozen slice: 128 fixed development inputs per task -> original classification
or answer-evidence choice prompt -> authenticated HTTP proxy -> SGLang native
noncausal joint schema head -> native choice/probabilities -> immutable request
records -> independent offline audit. No prompt, threshold, weights, precision,
context, cache or head changes. This is not unseen quality evaluation, complete
pool evaluation, or a comparison against the previous D3 RPS table.

Pinned image: sha256:a682ef469e9900ec254bb6d98e9c6052d0e0c727dcb005b4ea1a229b77424fae.
One A100-PCIE-40GB GPU0, TP1/BF16, 8 CPU quota, 96 GiB RAM, OMP threads2,
8192 context, mem-fraction-static0.85. Every profile enables identical metrics.
Baseline matches the prior evidence deployment's scheduling32/tokens32768.
The specialized joint head controls cache/chunking; do not override its safeguards.

## Frozen Matrix

| Profile | max-running-requests | max-prefill-tokens |
| --- | ---: | ---: |
| baseline | 32 | 32768 |
| seq8 | 8 | 32768 |
| tokens8192 | 32 | 8192 |
| seq64 | 64 | 32768 |
| tokens65536 | 32 | 65536 |
| wide | 64 | 65536 |
| baseline-end | 32 | 32768 |

Client concurrency1/8/16/32/64 is distinct from scheduler limits. Single-factor
candidates plus the explicitly combined wide profile; baseline repeated last.
Before mutation, native CPU encoding must prove all 256 full inputs fit the
smallest budget and native8186 prompt limit. Reuse dataset protocol SHA256
3a223a53f3d2b0a96046ab26d308b157c6a0fd8f652073acc9a12ae2a09faf17.
Do not remove, truncate or replace unsupported samples; stop preparation instead.

Each fresh deployment/task warms 32 length-spread inputs at n1/n64 twice.
Measurement uses all128 inputs at all5 concurrency levels twice; reverse both
input and concurrency order in round2. Fixed total:196 blocks,19712 HTTP calls,
1792 warmup and17920 measured. Startup internal checks are separate from these
HTTP calls. No result-based extra warmup, retries or selective fastest-round use.

## Measurements And Decision Rule

Effective RPS = valid calls / complete block wall; per-call mean/P95/P99 includes
HTTP/authentication, queue and native execution, excludes waiting for a client
worker slot. Save request/input hashes, native answers, token counts, response
hashes, concurrency intervals, metrics, cgroup CPU, sampled GPU resources and
startup-resolved settings. Keep all failures, timeouts and not-attempted separate.
Compare decisions by input ID and repeat against same-profile n1 and same-n
baseline. Report probability drift separately; do not mistake drift for a broken
protocol or claim runtime tuning improves semantic quality.

A promising candidate needs >=5% RPS improvement in both rounds at the same n,
pooled P95 <=110% of baseline, zero failed calls/decision changes, and <=10%
start/end baseline RPS drift. Otherwise report a tradeoff or unconfirmed gain.
One cold deployment/candidate and short closed-loop blocks cannot certify
production capacity/SLA. No automatic follow-on run or production rollout.

## Safety And Preservation

Use a new clef-runtime-20261019 root and nonblocking classification.lock. Verify
exact original acceptance container identities, command/image and exclusive GPU
ownership before stopping only those owned evaluation containers. Retain them
under held names; on normal completion restore original IDs/names/images/commands
(PID/start time necessarily change). Never change the protected proxy-host
embedding/reranker/proxy. Check their original identities before/after externally.
Freeze all new sources and preserve prior evidence by hash before inference.

No restart/resume. Load deadline2400s; individual HTTP deadline300s. First observed
failure closes admission; already-in-flight calls drain to their existing deadline.
On failure preserve services/evidence, with no automatic repair, restart, retry,
restore or probe inference. A transport timeout is not proof GPU work has stopped.
Independent audit writes new outputs and forbids network/subprocesses. Verify
systemd exit, lock release and service restoration separately.
