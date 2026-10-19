# Answer Evidence Root-Cause Verification

Owner request: establish why fragment-filtering quality is low, rather than stop
at a list of possible causes. Baseline repository revision is
`37070e98d32ff92ef153095a42b8fdf746bd1c0e` plus separately hashed untracked sources.
This is diagnosis only, not a production fix or a new model leaderboard.

## Frozen Flow And Scope

Original T2Ranking full query/passage and grade -> frozen pilot state and evidence
question -> native encoding -> backbone/decision head -> named choice/probability
-> fixed L2/L3 scoring. Existing joins and prompt-text checks passed, but official
GPU numerical equivalence and semantic label validity remain unproven.

Reuse the pilot's exact 643 holdout pairs, original baseline question, historical
worker responses, fixed model packages and existing dependency image. This is a
new diagnostic replay, explicitly not a new independent generalization sample.
The official AutoModel path must be called in-process because comparing another
HTTP wrapper around the same worker would not test the suspected boundary.

Before GPU work: verify package hashes against the model lock, CPU-import the
official implementation, freeze script/question/input/old-response hashes, take
the existing nonblocking evaluation lock, verify only owned acceptance containers
occupy the GPU, and preserve historical evidence. Stop only the original owned
model/proxy for the experiment; retain their IDs, image, command and files. Run
new uniquely named network-disabled containers on one A100 with 8 CPUs/96GiB.
No credentials are needed inside the reference containers. Protected proxy-host
embedding/reranker/proxy are outside mutation.

## Predeclared Contrasts

1. D3-mini and D3-flash official fixed-package AutoModel, BF16 backbone/FP32
   readout, noncausal attention as configured, one question/forward, all 643 pairs.
   Compare full token IDs between worker tokenizer and official tokenizer, actual
   backbone input IDs, usage, named predictions and probability differences.
   Historical worker results are a quality comparator, not concurrent timing.
2. On 128 pairs chosen by a fixed ID hash independent of grade/output, reverse
   only the two criteria entries. Compare named decisions and probabilities to
   the new official baseline, not to a changed label. Do not tune on results.
3. Twenty-four frozen synthetic Chinese controls: six fictional situations with
   complete answer, partial answer, topic-only and wrong-object passages. They
   diagnose basic instruction/option mapping and do not estimate business quality.

Planned model evaluations: (643 + 128 + 24) * 2 = 1590. No warmup predictions,
retry, fallback, threshold selection, truncation, network inference, new weights,
model-source patch, or automatic resume. Calls are sequential. A failed call
stops the sequence with attempted and remaining counts; no favorable exclusions.
Loading budget 2400s; per-call alarm 300s; cold timing is diagnostic only.

On normal completion verify container exit, complete output counts, immutable
sources and historical hashes; restore original acceptance container IDs and
check readiness without inference. On failure retain diagnostic logs and stopped
original containers; do not assume interrupted GPU work is finished or silently
resume. Record actual failure/restoration state before further action.

## Offline Diagnosis And Reporting

Independently recompute original and reference confusion matrices, agreement,
probability drift, original-grade errors, high-confidence disagreement and
reverse-order effects. Also reuse existing development prompt contrasts; no new
prompt search is part of this run. Keep proxy-label findings separate from
semantic correctness. Any model-assisted review is not human adjudication and
cannot establish a population label-error rate. Do not infer Clef numerical
equivalence from D3 results. Preserve all original labels and scores.

Report proven divergent stages, quantified changes, rejected explanations and
remaining limits. This experiment may establish or exclude a substantial D3
runtime discrepancy; it cannot by itself prove business ground truth or explain
all models' errors. No new production tuning or recurring automation is included.

## Pre-Inference Preparation Correction

First container exited before Python could read the diagnostic script: the image's
default nonroot user could not access the mode-0700 private source mount. No model
was loaded and no input or inference intent was created. Preserve that unit,
container and source snapshot. Confirm PID0/no GPU process, restore the original
owned services, then use a new `-r2` directory/unit/container identity. Run the
new network-disabled, read-only diagnostic container as uid0, matching the prior
CPU import preflight; keep capabilities dropped and private permissions intact.
Validate mount access without GPU before starting r2. This changes preparation
only, not model numerics, cases, questions or the 1590-call protocol. It is not
an inference retry or continuation of a partly measured dataset.

## Optional Context Limit Correction (r3)

The second attempt loaded D3-mini, then failed before its first inference intent:
the diagnostic compared a token count to the official runtime's `max_length=None`.
Official `d3_runtime.py` treats None as no explicit package limit. Both intent and
request files are empty; attempted/valid are zero. The diagnostic container exited
with PID0, the GPU was empty and the shared lock released. Original model/proxy
IDs, images and commands were preserved and restored; native health200 and proxy
unauthenticated health401 were checked before the owner authorized continuing.

Preserve both failed directories/units/containers. For the new r3 preparation only,
make the context limit check conditional on a non-None value. Keep full text,
token-ID equality and actual-backbone-input checks mandatory, with no truncation
or new context override. Regression tests cover absent/finite/exceeded limits,
empty input and changed/truncated encodings. Freeze a new source snapshot under
`evidence-reference-20261019-r3`; do not edit either failed run's source or outputs.
The data, 1590-call contrasts, inference settings and failure policy are unchanged.
