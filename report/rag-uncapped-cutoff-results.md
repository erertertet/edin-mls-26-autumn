# Uncapped cutoff follow-up: completed results

Completed on fuji3 on 16 November 2026. This supersedes the unlaunched top-50/five-policy preparation. Earlier sufficiency/subset-selection studies remain stopped.

Research question: **at matched labelled-evidence recall, do absolute similarity and top-score-relative gap cutoffs improve precision or token efficiency over fixed top-k, and do these differences improve generated answers, API cost or latency?**

## Controlled design

- Same exact cosine ranking of all 24,000 corpus documents; no artificial count cap. Top-1 fallback for empty selections; stable corpus-row tie ordering.
- Same dev objective for all methods: maximum mean per-query qrels F1, ties favour fewer documents. Fixed k searches 1–24,000; cutoffs use predeclared grids.
- Frozen on 125 dev queries: **k=2, absolute similarity >=0.8925, relative gap <=0.009**.
- Same 324 previously inspected evaluation queries. This is a follow-up, not a fresh confirmatory holdout.
- Mistral Small 4 generates all 972 fresh answers before Gemini 2.5 Flash judges any. Judge receives transcript, answer and gold evidence, without policy identity or selection metadata.
- Twelve API workers; shuffled query/policy order. Identical contexts receive independent calls. No model or prompt changes between methods.
- All contexts fit without truncation: maximum 8,040 full input tokens. Frozen absolute/relative settings retrieve at most 32/18 documents despite being uncapped.

## Retrieval results

| Method | Mean documents | Macro recall | Macro precision | Micro precision | Full input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed top-2 | 2.00 | 32.09% | 31.33% | 31.33% | 532 |
| Absolute cutoff | 2.40 | 32.39% | 37.89% | 28.41% | 657 |
| Relative gap | 2.55 | 30.60% | 39.21% | 23.34% | 667 |

At each cutoff's macro recall, compare against a query-independent mixture of adjacent fixed-k settings:

| Cutoff | Macro-precision gain | Retrieved-text token increase | Pooled precision difference |
| --- | ---: | ---: | ---: |
| Absolute | +6.87 pp [3.02, 10.97] | 566.6 vs 456.5: **+24.1%** | −2.46 pp [−7.27, 3.05] |
| Relative | +6.13 pp [2.73, 10.06] | 575.9 vs 407.3: **+41.4%** | −8.94 pp [−11.99, −5.48] |

Brackets are paired 5,000-resample query-bootstrap intervals; the fixed mixture is recomputed within each sample. They are exploratory and unadjusted for multiple comparisons. Frontier token cost sums separately tokenized document texts, excluding prompt scaffolding. The full input tokens in the first table come from actual provider usage and include scaffolding.

The supported claim is **higher macro precision, worse retrieved-text token efficiency at the selected matched-recall points**. It is not a universal precision gain: pooled precision is worse for relative and uncertain for absolute. The uncapped curves and the prior capped curves are different experiments; do not carry the earlier high-budget crossover claim into this study.

## Final blind LLM judgement

| Method | Correctness /4 | Grounding /4 | Completeness /4 | Score /1 | Downstream seconds | Generation $/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed top-2 | 2.38 | 2.37 | 2.39 | 0.594 | 2.12 | 0.232 |
| Absolute cutoff | 2.57 | 2.57 | 2.57 | 0.642 | 1.92 | 0.231 |
| Relative gap | 2.40 | 2.39 | 2.40 | 0.599 | 2.16 | 0.255 |

Absolute's paired score gain versus top-2 is +0.0476 [0.0046, 0.0926]. Relative's is +0.0044 [−0.0352, 0.0437], providing no clear gain. Absolute also exceeds relative by 0.0432 [0.0033, 0.0831]. These are rubric scores, not accuracy; the rubric can reward qualified refusal when gold evidence is inadequate.

Absolute uses more input tokens but produces fewer output tokens (229 vs 254 for top-2), so total generation charges are approximately equal. Its downstream service time is 0.200 seconds lower [−0.315, −0.086]. Relative costs about 9.7% more in generation charges without a clear score or latency improvement. Do not conflate input-token inefficiency with total dollar cost.

Latency includes local selection/context construction plus generation API service under concurrency. It excludes cached ASR/embedding/ranking and worker queue wait: **not a fresh audio-to-answer wall-clock measurement**. Quality evaluation nevertheless judges the final generated answers using the cached ASR transcripts.

## Completion, spending and evidence

- 972 unique answers and 972 valid final judgements; zero failed tasks and zero length-limited generator answers.
- One length-limited judge response required a replacement: 972 generation plus 973 judge calls.
- All **1,945** charges reconciled: generation **$0.232637334**, judging **$0.6125919**, total **$0.845229234**.
- All 972 predicted full prompt counts match actual provider input usage exactly.
- Generation completion is earlier than judge start; the request log's first 972 calls are generator calls.
- Complete local evidence: `rag/results/cutoff-uncapped-20261116/`. Remote equivalent: `/home/zcm/ragexp/cutoff-uncapped-20261116/`.
- Full ranking cache: `/home/zcm/ragexp/uncapped-rankings-20261116/`; no ASR, encoder or cross-encoder inference was rerun.
- Figures: `report/draft/figures/uncapped-frontiers.png`, `final-answer-evaluation.png`; revised report: `report/draft/report.pdf`.

Source scripts are `prepare_uncapped_rankings.py`, `cutoff_followup.py`, `uncapped_frontiers.py` under `rag/extensions`, and `report/plot_uncapped_report.py`. Configuration and source hashes are frozen with the run. The old prepared top-50 run has zero API calls and should not be resumed for this design.
