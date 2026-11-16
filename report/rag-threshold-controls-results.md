# Rounded similarity-threshold controls

Completed 16 November 2026 on fuji3. This supersedes the five-row answer table with a fresh nine-setting batch: 324 identical evaluation queries per setting, 2,916 answers, then 2,916 policy-blind judgements. No answers or timings are pooled from earlier runs. Absolute numbers are minimum cosine similarities (tau); relative numbers are maximum differences from the top score (delta). All settings use the full 24,000-document ranking, with top-1 fallback and no document-count cap.

Absolute 0.8925 and relative 0.009 retain their earlier dev-F1 selection. Additional round controls are tau=0.88/0.87 and delta=0.02/0.03. These were fixed before generation, motivated by previously inspected evaluation curves; they are exploratory, not a clean held-out test or exact recall-target fits.

| Method | Correctness /4 | Grounding /4 | Completeness /4 | Score /1 | Full input tokens | Downstream s | Generation $/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5 | 2.15 | 2.11 | 2.15 | 0.534 | 1,227 | 2.85 | 0.361 |
| Fixed top-10 | 2.24 | 2.19 | 2.22 | 0.554 | 2,391 | 3.32 | 0.547 |
| Fixed top-20 | 2.27 | 2.19 | 2.26 | 0.560 | 4,730 | 3.82 | 0.929 |
| Absolute 0.8925 | **2.56** | **2.56** | **2.56** | **0.639** | **657** | **1.99** | **0.207** |
| Absolute 0.88 | 2.51 | 2.48 | 2.51 | 0.625 | 1,612 | 2.60 | 0.375 |
| Absolute 0.87 | 2.35 | 2.29 | 2.35 | 0.583 | 3,586 | 3.26 | 0.738 |
| Relative 0.009 | 2.36 | 2.36 | 2.36 | 0.590 | 667 | 2.11 | 0.225 |
| Relative 0.02 | 2.26 | 2.25 | 2.26 | 0.564 | 2,048 | 2.89 | 0.468 |
| Relative 0.03 | 2.22 | 2.17 | 2.22 | 0.551 | 5,437 | 3.63 | 1.038 |

Bold marks the best observed unrounded value, not significance. Generator: Mistral Small 4; judge: Gemini 2.5 Flash. Latency covers selection/context construction plus generator API service under 12 workers, excluding cached ASR, embedding/ranking and queue wait. Costs include generation input/output; judge costs are separate experiment overhead.

## What the additional controls show

| Setting | Macro recall % | Macro precision % | Mean / max documents | Extra text tokens vs matched-recall fixed-k mixture |
| --- | ---: | ---: | ---: | ---: |
| Absolute 0.88 | 39.58 | 29.12 | 6.15 / 82 | 83.5% |
| Absolute 0.87 | 46.24 | 19.34 | 14.10 / 180 | 98.7% |
| Relative 0.02 | 42.81 | 32.25 | 8.28 / 113 | 61.1% |
| Relative 0.03 | 52.87 | 22.86 | 22.54 / 280 | 70.9% |

At matched recall, the four new settings gain 6.74, 6.23, 15.31 and 14.42 macro-precision percentage points over query-independent mixtures of adjacent fixed depths. Their additional retrieved-text-token costs are above. These are point estimates; do not confuse additive text tokens with complete billed prompt counts.

Absolute 0.88 has higher recall than absolute 0.8925 but a small, uncertain score difference: -0.014, paired 95% interval [-0.055, 0.026]. Absolute 0.87 falls by 0.057 [-0.099, -0.013] relative to the dev-selected cutoff. Relative 0.02/0.03 have lower mean scores than 0.009, but their paired intervals include zero.

Relative 0.02 and fixed top-5 have similar recalls (42.81% and 42.40%). The relative rule has much higher macro precision but uses 2,048 versus 1,227 full input tokens. Its score gain of 0.030 [-0.007, 0.067] is uncertain, while generation costs $0.468 versus $0.361 per 1,000. Thus higher macro precision does not establish answer quality or token efficiency. All intervals are 5,000-resample paired query bootstraps, exploratory and unadjusted for multiple comparisons; systematic judge/label error and repeated-generation variance are omitted.

## Validation and accounting

All 2,916 prompt hashes and token counts match preflight. Maximum prompt: 71,670 tokens; no input truncation. One absolute-0.87 answer reaches the 2,048-token output cap and is retained. There are 2,921 judge calls (five extra attempts), with no manual score or JSON-format repair. All 5,837 distinct request IDs reconcile to provider billing: generation $1.583303721, judging $1.974546010, total **$3.557849731**.

The stopped recall-target proposal logged 52 calls and $0.032235654 before interruption; in-flight charges may be absent from that log. Its partial answers are excluded. Earlier five-row, top-2, sufficiency and subset runs remain separate historical evidence.

Evidence: `rag/results/cutoff-threshold-controls-20261116/` (Git-ignored), including `summary.json`, `final-validation.json`, raw responses, complete billing and `rounded-matched-recall.json`. Source: `rag/extensions/cutoff_threshold_controls.py`; preflight: `rag/extensions/preflight_threshold_controls.py`. Report protocol: `report/draft/threshold-controls-protocol.md`. Curves retain x=macro precision, y=macro recall; point labels use actual thresholds.
