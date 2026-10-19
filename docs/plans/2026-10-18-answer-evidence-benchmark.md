# Answer Evidence Selection Benchmark

## Authorization And Baseline

The owner clarified that the product task is answer-evidence selection and on
2026-10-18 authorized rethinking the method and rerunning the comparison. This is
an independent evaluation, not a correction to historical measurements or a
production change. Baseline repository HEAD is
`37070e98d32ff92ef153095a42b8fdf746bd1c0e`, with existing uncommitted evaluation
files. Preserve all old classification, capacity, passage, noul and diagnosis
artifacts and frozen sources. No production Agent behavior changes.

Before any new inference, the owner accepted the coverage-aware approach and
requested reranker comparisons with TopN fixed at 10. This revision supersedes
the initial direct-only 28-condition design, not the historical Top29 results.

First divergent stage: the previous C-MTEB binary labels include grade-1
relevance and the prompt asks for retrieval relevance, whereas the owner needs
information that can contribute to an answer. The old quality scores therefore
do not answer the product question. Runtime/protocol evidence remains valid.

New flow: pinned original queries + complete collection + graded qrels ->
deterministic independent query splits -> frozen native complete-input coverage
-> paired direct / online reranker Top10 native binary decisions -> immutable
requests and measured pool walls -> independent offline audit.
The existing serial transport, readiness checks, launcher, native mapping,
nonthinking LLM parser, timing and bootstrap helpers remain authoritative.

## Labels And Data

- Use original `THUIR/T2Ranking` revision
  `2a369a430a70979223f1b9a41b1919774d46b432`, not guessed grades from C-MTEB lists.
- Positive = original grade 2 or 3; negative = grade 0 or 1. Preserve original
  grades and IDs in private labels. This is a public proxy for evidence utility,
  not human verification of factual truth, source reliability or generated answers.
- The original [T2Ranking paper](https://arxiv.org/abs/2304.03679), annotation
  section, defines Level-2 as relevant and partly satisfying the query's information
  needs, and Level-3 as satisfying those needs and precisely containing the answer.
  Level-1 is relevant without meeting the information need. Thus L2/L3 is a
  source-grounded operational definition of partial/full answer-bearing content,
  not an assumption that every passage is factually verified, independently
  sufficient, trustworthy or free of annotation error. The paper itself uses
  L2/L3 as positives for retrieval; our binary set-selection metrics are still
  a separate task from its official ranking scores. Rechecked against the paper
  before inference on 2026-10-18.
- Verify full source SHA256 and exact qid/pid joins. No fuzzy text mapping,
  synthetic labels, treating unjudged passages as negatives or model relabeling.
- Exclude normalized queries appearing anywhere in the previous C-MTEB source,
  not just the 232 recently used pools. Deduplicate normalized query identities.
- Freeze seed `answer-evidence-v1` before model results. Hash-order the
  remaining queries into 12 engineering, 20 validation and 200 main-test pools.
  Keep every judged candidate in each selected pool and the complete text.
- Independently select 40 additional genuinely no-evidence pools from the
  remaining queries as a diagnostic set. Do not merge this oversampled set into
  representative main-test aggregates. Main-test pools can also have no evidence.
- Freeze exact counts, source hashes, split IDs and input/label hashes before any
  inference. No automatic truncation, result-based exclusions or replacements.
  An unsupported complete input is an explicit coverage failure, not a silent crop.
- Freeze native tokenization/complete-input checks for all 28 refiner modes and
  the deployed reranker before inference. Include template, decision/output-token
  reservations and actual effective limits. The existing PPLX-only check is not
  enough to establish common coverage. Top10 does not solve overlength inputs:
  the reranker must read every complete candidate before it can rank them.
- Keep the original full dataset and denominators in coverage reporting. Main
  paired quality and timing comparisons use the same common COMPLETE POOLS:
  every original candidate must be fully supported by all compared modes and
  the reranker. Unsupported pools stay in the original coverage ledger, with no
  replacement or dropped candidate disguised as a correct negative. This avoids
  turning an evidence-containing pool into a no-evidence pool by deleting positives.
  Report the unsupported candidates, affected whole pools and original grades.
- Report quality on common covered pools separately from original-denominator
  coverage-adjusted end-to-end results. An unsupported pool is not a model's
  valid empty selection; give it no successful-pool/true-negative credit. Label
  any zero-credit coverage-adjusted quality explicitly, not as native model quality.
- Do not send inputs already proven unsupported merely to obtain an HTTP error.
  Persist a preflight failure reason and zero HTTP attempts instead. Do not give
  each model a different favorable subset, shorten passages, expand only one
  model's context or remove a failure after observing outputs. Per-model native
  coverage may additionally be reported from the frozen offline checks, but it
  must not replace the common-pool comparison or imply successful inference.
- Do not tune prompts, thresholds or models on validation or formal outputs in
  this run. Validation reports remain separate from the new main test.

## Conditions

Compare the existing 13 Decision models in BOTH native choice and native noul,
plus the two existing nonthinking LLMs: 15 refiner deployments, 28 refiner modes,
each under two arms, giving 56 conditions. Use the
unchanged roster from `decision_passage_benchmark.py`, no Intern or thinking
profiles. Modes share deployment. Balance mode and arm order by a frozen
four-position schedule across pools (two positions for each LLM), without phase
overlap. Do not systematically run the direct arm first.

| Arm | Decision models | Nonthinking LLMs |
| --- | --- | --- |
| Direct | Full pool -> choice or noul | Full pool -> strict JSON binary choice |
| Top10 | Full pool -> online reranker -> at most 10 -> same choice or noul | Full pool -> online reranker -> at most 10 -> same strict JSON binary choice |

- Reuse the retained Qwen3-Reranker-8B native vLLM pooling deployment unchanged.
  Each covered pool/mode receives its own measured online reranker request over
  ALL original candidates; return all scores, then take `min(10, candidate_count)`.
  No cached ranking or reuse of another condition's timings as online latency.
- Equal scores use the frozen original candidate order. Refine retained members
  in original order. Save all scores, selected membership and input/request hashes.
- The refiner sees only the same complete query/passage pair and evidence rubric,
  without a rank, score, selected-set hint or altered budget. It may reject every
  member. A Top10 cutoff is a ranking budget, NOT a decision confidence threshold
  and not a requirement to retain ten final answers.
- Final chain scoring uses the original full pool labels. A positive removed by
  Top10 is a false negative of the chain; never score only the retained candidates.
  Also report the reranker's pre-refiner evidence retention as a recall ceiling.
- No calibrated retention threshold, forced Top1 fallback, TopN sweep, prompt
  tuning or replay. Engineering, validation, formal and no-evidence diagnostic
  phases remain separate; their results cannot be used to change Top10 mid-run.

All models receive the same evidence rubric: does this passage provide concrete
information that partially or fully meets this question's information need?
Partial answers, applicable conditions and useful counterevidence can qualify;
topic overlap alone does not. Do not require a complete standalone answer or
agreement with a potentially false premise. Treat passage text as data, not an
instruction. Retain binary schema IDs `relevant`/`irrelevant` for compatibility,
but explicitly define them as evidence yes/no in this protocol.

Choice uses the returned native category. Noul uses the returned native category
or boolean, otherwise fixed binary argmax (false on a tie), never label-fitted
thresholds. LLM uses strict JSON, temperature 0, nonthinking, 64 output tokens.

## Measurement And Safety

- Global HTTP concurrency 1, single complete pair per refinement request; no
  overlapping modes, pools, phases or deployments. No automatic inference retries.
- Existing deployment images, weights and resource budgets; startup allowance
  2400 seconds and per-HTTP hard wall limit 300 seconds. Hold the existing shared
  evaluation flock across deployment and execution. New exclusive output root.
- Only replace owned `evaluation.decision.acceptance=true` evaluation containers on
  the test host. Use the retained proxy reranker, without replacing/restarting it,
  its proxy or business embedding. Freeze and recheck their identities/settings.
- Any invalid response stops this sequence; timeout/5xx/ambiguous transport
  failures do not imply GPU work ended. No automatic repair, restart or resume.
- Preserve the first failing request and its affected chain before stopping.
  Downstream stages and later conditions not issued after that stop are
  `not_attempted_after_stop`, not executed failures or valid negative decisions.
  A new failure never causes the common-coverage subset to be recomputed. Report
  incomplete coverage and avoid ranking conditions with incomparable exposure.
- Engineering is separate from scored phases. No additional synthetic inference.
  Record cold engineering latency separately; validation/formal timing follows it.
- Actual pool wall time includes request construction, network/auth, parsing and
  request journals; excludes pool-row persistence and between-pool identity checks.
  Report measured walls, not sums of request latency. Request timing separately
  exposes journal overhead; no n=1-to-concurrency or equal-hardware claims.
- Top10 wall time includes the actual reranker round trip, selection and remaining
  refinement calls. Compare paired successful pools, reporting difference and
  ratio with query-level intervals. Do not substitute stage sums for pool wall.
  Disclose the additional proxy-host resources and cohosted embedding traffic;
  this is not equal-total-resource acceleration or production capacity.

## Reporting

### Failure And Omission Ledger

Keep a structured record for every unsupported/failed/skipped case, including
run/model/native mode/arm/stage, case/fragment IDs, original grade, input/request
hash, reason code, HTTP attempt count, HTTP status when available, elapsed time,
and the originating upstream failure ID where applicable. For length failures,
include actual native tokens, effective limit and reserved output/schema budget.
Use sanitized errors only; no input text, reasoning, credentials or full environment.
Compare the two arms on identical frozen full pools and keep these causes separate:

| Event | Category and accounting |
| --- | --- |
| Native input cannot fit before execution | `preflight_input_unsupported`; coverage failure, zero HTTP attempts, original labels retained |
| A pool is outside the all-condition common intersection | `outside_common_pool_coverage`; cite the actual unsupported input/condition, do not imply this particular model failed |
| HTTP/transport/timeout failure | `http_error`, `transport_error`, or `timeout`; attempted failure, no fabricated binary prediction; sequence stops |
| Missing/invalid structured model output or incomplete input | `invalid_output` or `incomplete_input`; attempted protocol failure, sequence stops |
| Work not issued after a stopped stage/run | `not_attempted_after_stop`; link the first cause, no fabricated request or elapsed time |
| Valid ranking removes an L2/L3 candidate | `reranker_pruned_positive`; chain false negative, NOT an execution failure |
| Valid native binary decision rejects an L2/L3 candidate | `refiner_rejected_positive`; semantic false negative, NOT an execution failure |
| Valid native decision accepts an L0/L1 candidate | `refiner_accepted_negative`; semantic false positive |

The primary quality table is semantic quality on the predeclared common covered
pools; coverage and execution success have their own tables and original planned
denominators. Never quietly fold unavailable predictions into ordinary TP/FP/FN/TN.
If additionally presenting zero-credit end-to-end failure penalties, identify them
as coverage-adjusted metrics and retain separate semantic confusion counts.
Observed causes describe the failing stage, not an unproven explanation such as
"bad prompt", "model lacks ability" or "label error".

### Quality And Timing

- Main evidence-containing pools: macro Precision/Recall/F1 with paired query
  bootstrap 95% intervals, missed-all-evidence rate and complete-set exact match.
- No-evidence pools: correct-empty rate, false-accept rate and false-accepted
  candidates; show natural main-test and oversampled diagnostics separately.
- All candidates: TP/FP/FN/TN, failures/missing coverage, Accuracy, kept count and
  text-character reduction (not token savings), grade-2/3 retention and grade-0/1
  rejection. Failures remain in planned denominators; no scoring only successes.
- Compare all-keep and all-reject baselines. Empty prediction on a positive pool
  has P/R/F1 zero; empty prediction on a negative-only pool is a correct rejection,
  evaluated separately rather than a fabricated positive-class F1.
- Derive a reranker-Top10-only baseline from each measured ranking, without extra
  inference. Report how much refinement improves precision and loses recall after
  that selection. Separate reranker-dropped positives from refiner-rejected ones.
- Report original candidate-count `<=10` and `>10` slices. The smaller slice has
  no pruning and therefore no expected call-count savings. Also report pools with
  more than ten positives, for which Top10 imposes unavoidable evidence loss.
- Compare each model's choice/noul pair, each refiner's direct/Top10 pair, and
  Decision versus LLM on identical covered pools. Differences need paired query
  bootstrap intervals; disclose exploratory multiple comparisons. Do not choose
  a "best mode" on formal results and claim independent confirmatory superiority.
- Report request and measured-pool mean/P95/P99, valid/attempted/planned counts,
  failed/timeouts and all-attempt elapsed time. Pair mode differences on the same
  queries; do not declare tiny point differences significant or a universal winner.
- Audit source/deployment identity, exact grade joins, split isolation, every
  call/input/request hash, no retries, global nonoverlap and preservation. Keep
  raw inputs and reasoning out of console logs and public reports.
- This experiment does not generate final RAG answers, so it cannot establish
  answer correctness, production throughput or the best attainable prompt.
- New labels, queries and prompt differ from the old relevance experiment. A new
  score cannot establish that prompt revision caused an improvement; that would
  require a separately authorized same-data prompt ablation. Public graded labels
  remain an imperfect proxy for answer evidence, not a universal product verdict.

## Pre-Inference Top10 Arithmetic

On the unchanged original 200 main-test pools (before common-coverage filtering),
there are 3105 candidates, 935 positives and 18 pools with at most ten candidates.
Top10 would cause 1984 refinement calls plus 200 reranker calls per refiner mode,
versus 3105 direct refinement calls: 36.10% fewer refinement calls, NOT a measured
latency saving. There are 11 pools with more than ten positives. Even perfect
ordering retains at most 885 of 935 positives (micro-recall ceiling 94.65%;
positive-pool macro-recall ceiling 98.49%), before any refiner mistakes.
These are offline budget bounds, not model results. Recompute them for the frozen
common-covered subset and disclose the changed denominator without replacing it.

## Implementation Scope

Extend the existing new evidence-only runner and tests with paired arms, coverage
accounting and Top10 chain audits; reuse the established passage reranker parsing,
stable selection and serial transport through a narrowly parameterized TopN,
preserving historical Top29 defaults and snapshots. Keep optional question
parameters on the existing passage/LLM adapters with unchanged defaults; preparation/audit artifacts; this
plan, execution record and current evaluation tasks. Reuse all other mechanisms.
No change to historical remote snapshots, production routes, model weights,
classification logic, capacity measurements or service infrastructure.

The previous 142-test evidence covers the initial direct-only implementation,
not this new 56-condition protocol. Extend and verify runner, preflight, preparation
and offline reporting together before freeze/start; never launch the old
direct-only entry and describe it as the newly authorized Top10 comparison.
