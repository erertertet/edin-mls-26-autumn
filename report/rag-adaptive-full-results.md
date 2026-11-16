# Full RAG comparison: predicted depth and distance filtering

The full run completed on fuji3 on 2026-11-15. It produced **2,335 answers and
2,335 primary judgements**, plus 467 planning calls and 300 additional audit
judgements. All **5,437 API completions** have reconciled OpenRouter billing
records. Total API cost was **$2.748036519**, including the audit and excluding
earlier pilots. No primary tasks failed. This is experimental evidence and
report material, not the final coursework PDF.

## Dataset and the dev/test distinction

The course bundle supplies 24,000 corpus documents and 475 audio queries:
350 practice and 125 dev. It supplies no separate official test split. The corpus
is shared across splits; query audio and relevance judgements are split.

At the user's request, dev was used for tuning. To obtain an independent final
test, the prior experiment manifests and smoke-query history were checked before
tuning. Of the practice queries, 26 had already appeared in experiments. A
researcher-defined test partition was locked from the other **324 previously
unused practice queries**. It must not be described as an official supplied test.

| Group | Queries | Role | Answers across five policies |
| --- | ---: | --- | ---: |
| Supplied dev | 125 | Tune distance parameters; report in-sample evaluation separately | 625 |
| Previously unused practice | 324 | Locked held-out test, no parameter tuning | 1,620 |
| Previously seen practice, excluding calibration | 18 | Exploratory diagnostics only | 90 |
| Original practice calibration | 8 | Excluded from the full run | 0 |

Thus all 467 eligible queries were evaluated; none of the seen practice results
are pooled into the final test results. Models were not trained or fine-tuned.
Only retrieval-policy thresholds were fitted.

## Methods and tuning

All arms use the same supplied enc_a query encoder, exact cosine retriever,
corpus and context prompt, custom CUDA ASR, Mistral Small 4 answer generator
(`mistralai/mistral-small-2603`) and Gemini 2.5 Flash judge
(`google/gemini-2.5-flash`). The query-only depth predictor also uses Mistral
Small 4. Its prompt, output range 1–20 and raw predictions were not tuned.

The five policies are fixed top-5, fixed top-20, model-predicted k, top-20 filtered
by an absolute distance cutoff, and top-20 filtered by a relative score gap.
Threshold policies retain at least the best hit, even if no hit meets the cutoff,
and at most 20 records. With normalised vectors, cosine distance is 1 minus
cosine similarity. This modifies evidence selection, not the ranking algorithm.

The selection objective was **maximum macro retrieval F1 on dev qrels**; ties
favoured fewer mean documents. The grid comprised 97 absolute similarities from
0.75 to 0.99 and 81 relative gaps from 0 to 0.08. Parameters were frozen at
21:59:04 UTC, before any test answer was evaluated:

- Absolute similarity **>= 0.8925**, equivalent to cosine distance **<= 0.1075**.
- Relative similarity **>= best similarity - 0.009**.

The dev macro F1 values at selection were 0.2659 and 0.2949 respectively.
No test labels or answer-judge scores were used to choose parameters. Precision
and F1 treat unjudged documents as non-relevant, so incomplete qrels are a
limitation of this tuning objective.

Generation temperature was 0.1; planner and judge temperatures were zero.
Reasoning was requested disabled. The uniform output caps were 2,048 tokens for
answers, 64 for planning and 768 for judging. One test top-20 answer hit the
answer cap; all other answers terminated without reaching it. Primary API role
counts were exactly 2,335 generator, 2,335 judge and 467 planner completions.

## Timing and parallel execution

Eight API workers overlapped independent planning, generation and judging calls.
Each thread used its own API client. ASR used one persistent worker on one A6000
and each query was transcribed once; transcripts and CPU query vectors were
shared across policies. Policy order was randomised per query. GPU memory was
released after inference, before billing reconciliation.

The actual warm audio-to-answer measurement starts before ASR and ends when the
answer returns; it includes worker queueing and excludes loading and judging.
It is available for every test query because all test transcripts were newly
computed. Prepared dev transcripts do not have an equivalent live measurement.
The table below uses these **actual measured test E2E times**.

`service_e2e_ms` is additionally saved: it sums measured ASR, encoding, planning,
retrieval/context construction and generation times, excluding queueing. That
quantity is reconstructed, not an independent wall-clock measurement. The
parallel serving conditions differ from the sequential pilot; do not attribute
cross-run timing differences solely to the policies.

The main parallel process took about 22.9 minutes including billing
reconciliation, following separate dev transcript preparation. The repeat-judge
audit followed the main run. Timings describe this host/service configuration,
not ELM hosting or isolated production latency.

## Held-out results: 324 queries

The answer score is the supplied automated rubric's normalised 0–1 score, **not
accuracy**. Inference costs include planning where applicable but exclude the
experimental judge; they are dollars per 1,000 queries.

| Policy | Mean docs | Recall | Precision | Retrieval F1 | Answer score | Actual warm E2E | Inference USD/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5 | 5.00 | 0.424 | 0.177 | 0.229 | 0.538 | 5.81 s | $0.378 |
| Fixed top-20 | 20.00 | 0.578 | 0.064 | 0.110 | 0.545 | 6.57 s | $1.020 |
| Model-predicted k | 2.81 | 0.350 | 0.282 | 0.280 | 0.589 | 5.93 s | $0.248 |
| Absolute distance cutoff | 2.35 | 0.324 | 0.379 | 0.305 | 0.634 | 5.06 s | $0.227 |
| Relative score-gap cutoff | 2.55 | 0.306 | 0.392 | 0.303 | 0.594 | 5.26 s | $0.242 |

Judge dimensions (each 0–4) are retained separately:

| Policy | Correctness | Grounding | Completeness |
| --- | ---: | ---: | ---: |
| Fixed top-5 | 2.160 | 2.127 | 2.167 |
| Fixed top-20 | 2.204 | 2.139 | 2.198 |
| Model-predicted k | 2.361 | 2.336 | 2.373 |
| Absolute cutoff | 2.543 | 2.525 | 2.537 |
| Relative gap | 2.383 | 2.364 | 2.380 |

Paired differences versus top-5, with 5,000-resample query bootstrap intervals:

| Policy | Answer-score difference, 95% interval | Actual E2E difference, 95% interval |
| --- | --- | --- |
| Fixed top-20 | +0.007 [-0.031, +0.046] | +0.760 s [+0.626, +0.898] |
| Model-predicted k | +0.051 [+0.012, +0.092] | +0.116 s [-0.001, +0.234] |
| Absolute cutoff | +0.096 [+0.049, +0.143] | -0.747 s [-0.897, -0.593] |
| Relative gap | +0.056 [+0.015, +0.097] | -0.556 s [-0.682, -0.422] |

These are unadjusted exploratory intervals; they do not account for multiple
comparisons, generation randomness, or systematic judge error.

### Research question 1: model-chosen document count

The planner averaged **0.524 s** and cost **$0.0138 per 1,000 test queries**.
Generation took about 0.391 s less than top-5, so planning mostly consumed the
time saved. The net actual E2E increase was 0.116 s (about 2%); its interval
slightly overlaps zero. Net inference cost fell about **34%**, because shorter
contexts saved more generation cost than the additional planning call incurred.

The judge score increased by 0.051, while recall fell by 0.074. This is a useful
quality/cost/latency trade-off, not an unconditional win. Unlike the small pilot,
the larger independent sample favours predicted depth on the automated score;
the pilot alone was not representative enough for a firm conclusion.

The model sees only the question. It cannot know the corpus's record granularity
or where relevant hits occur. This run also does not establish that prediction
beats a cheap fixed k=2 or k=3: those controls were only run on four earlier
queries and are not a full held-out comparison. Match small-k budgets in a future
control before attributing the benefit specifically to adaptive prediction.

### Research question 2: top-k plus distance filtering

The absolute cutoff had the highest automated score on this test set. Relative
to top-5 it used about 53% fewer documents, cost **40% less**, and reduced mean
actual warm E2E by **13%**. Its score increased by 0.096. However, recall fell
from 0.424 to 0.324, and **135/324** queries required the top-1 fallback because
no candidate met the global threshold.

The relative gap used about 49% fewer documents, reduced inference cost by
**36%** and actual E2E by **10%**, and increased mean judge score by 0.056.
It had lower recall than the absolute cutoff and no empty-cutoff fallbacks.

Fixed top-20 improved recall substantially but did not show a clear answer-score
gain over top-5. It cost approximately 2.7 times as much for inference and added
0.76 s of mean E2E latency. The results are consistent with low-relevance extra
context adding distraction or unsupported detail, but they do not isolate that
mechanism from answer length, redundancy, abstention or judge behaviour.

The distance-policy parameters differ from the original eight-query calibration
and are now selected from all dev queries. Do not attribute differences from the
pilot solely to tuning: evaluation samples, serving concurrency and output caps
also changed.

## Judge stability audit

Before any main test answers completed, 30 test query IDs were randomly locked
with seed 20261118. Each of their 150 saved answers received two additional
judgements with the same model, rubric and settings; answers were not regenerated.
This added 300 calls and cost **$0.20250625**.

**144/150 answers (96%)** received identical normalised scores across all three
judgements. Six answers showed some disagreement. This suggests limited judge
sampling instability on this subset, not that the judge is factually reliable.
The 30-query subset's policy ranking differs from the full 324-query ranking;
it should not be described as independently reproducing that ranking.

The earlier manual inspection also showed that the evaluator can reward a
qualified insufficient-evidence answer with a high score despite zero qrels
recall. Therefore higher automated scores alone do not establish more successfully
answered questions. Human checking and an explicit answer/abstention breakdown
would strengthen the interpretation. Public qrels may omit useful documents.

## Costs, validation and reproducibility

| Component | API cost |
| --- | ---: |
| Main 2,335 answers, 2,335 judgements and 467 predictions | $2.545530269 |
| Additional 300 audit judgements | $0.202506250 |
| Total full run and audit | **$2.748036519** |

This is actual inference usage, not a token-price-only estimate. All 5,437
generation billing records were recovered and matched returned usage costs.
Hardware/electricity and any credit-purchase fees are excluded.

Validation confirmed all 467 query IDs, all 2,335 unique query-policy pairs,
paired transcript identity, vector dimensions/normalisation, valid document IDs,
recomputed retrieval metrics, normalised judge scores, nonnegative timings and
all 300 unique audit judgements. Existing adapter tests passed, and a simulated
eight-thread test checked that in-flight budget reservations prevent excess
API calls without making real requests.

The code is in:

- [`adaptive_retrieval_batch.py`](../rag/extensions/adaptive_retrieval_batch.py)
- [`adaptive_judge_audit.py`](../rag/extensions/adaptive_judge_audit.py)
- [`prepare_adaptive_dev.py`](../rag/extensions/prepare_adaptive_dev.py)
- [Extension documentation](../rag/extensions/README.md)

Evidence is retained on fuji3 at `~/ragexp/full-adaptive-20261115T215241Z/`
and copied locally to `rag/results/full-adaptive-20261115T215241Z/` (ignored
generated files). It includes partition and audit manifests, full tuning grid,
frozen parameters, configuration, source hashes, package versions, transcripts,
query vectors, candidate answers, all judgements, API usage/billing records,
validation, latency analysis and per-phase summaries.

The main `summary.json` covers primary evaluation; `judge-audit-summary.json`
covers repeat judging; `experiment-overview.json` gives the combined cost.
Do not pool dev or previously seen query results into the held-out test table.

Mistral Small 4 matches one named ELM option. OpenRouter model names do not prove
identical checkpoint revisions, quantisation or serving configuration; these
costs and latencies must not be represented as ELM hosting measurements. The
earlier Qwen3-235B pilot is a separate generator configuration.

AI assistance was used for experimental implementation, orchestration,
validation, analysis and these notes. Include this in the final report's
AI-use declaration.
