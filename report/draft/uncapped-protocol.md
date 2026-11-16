# Focused uncapped-cutoff experiment

Research question: at matched labelled-evidence recall, do absolute similarity and top-score-relative gap selection improve precision or retrieved-text token efficiency compared with fixed top-k, and do the differences translate into answer quality, API cost or downstream latency?

The expected precision/token trade-off is a hypothesis, not a required outcome. Macro precision and pooled micro precision must both be reported.

## Three methods

1. Fixed top-k, with one global k selected on dev.
2. Absolute similarity: retain scores at least tau.
3. Relative gap: retain scores within delta of the best score for that query.

All methods use the same complete cosine ranking of all 24,000 documents. No top-50 or other artificial document-count cap applies. Empty absolute selections fall back to top-1; ties use corpus row order. Both cutoff rules select a prefix (a subset of a sufficiently large top-K candidate pool), not arbitrary noncontiguous documents.

## Tuning and evaluation

Use all 125 dev queries for maximum macro qrels F1, breaking ties towards fewer retrieved documents. Fixed k searches 1 through 24,000; absolute similarity uses the predeclared 97-point 0.75–0.99 grid and relative gap the 81-point 0–0.08 grid. The frozen values are k=2, tau=0.8925, delta=0.009. No answer score participates in tuning.

Evaluate all three frozen policies on the same 324 previously inspected practice queries. This is a follow-up on a reused evaluation set, not a new confirmatory holdout. Generate 972 fresh answers with Mistral Small 4, then judge all 972 saved answers with Gemini 2.5 Flash. Twelve workers, shuffled query/policy order, common prompts/decoding, independent calls even for identical contexts, and no judging until generation is complete. The judge receives the transcript, answer and qrels-derived gold evidence; policy names, scores and selections are hidden.

Maximum selected context is 8,040 estimated full prompt tokens; every context fits without truncation. At the frozen settings, absolute selection retrieves at most 32 documents and relative selection at most 18, although neither is capped.

## Measurement boundaries

- Retrieval: macro recall, macro precision, micro precision and micro recall; matched-recall fixed-k mixtures; paired query bootstrap intervals.
- Frontier token cost: sum of individually tokenized retrieved document texts, excluding prompt scaffolding. This is a reproducible additive evidence-cost metric.
- Answer-stage input cost: actual full prompt input tokens from provider usage, including scaffolding.
- Answer quality: correctness, grounding and completeness separately (0–4), plus their normalized sum (0–1). These are model rubric scores, not accuracy or human ground truth.
- Downstream latency: local selection/context construction plus generator API service, under 12-way concurrency. Cached ASR, embedding and corpus ranking are excluded. Do not call this fresh audio-to-answer wall-clock latency.
- Costs: generation charged as deployment cost; judge charged as experiment overhead. Report and reconcile both.

New run cap is $8, well within the checked $36.31 remaining account balance. The earlier unlaunched top-50/five-policy preparation has zero API calls and is superseded. Previously stopped sufficiency/subset experiments remain stopped.

## Report scope

Main report: this question, three methods, metrics, retrieval frontiers, final answer-evaluation table and interpretation. Keep a brief ASR implementation/validation summary and integrated kernel ablation table; move detailed CUDA mechanisms, tuning and sanitizer evidence to the appendix. Keep earlier planner/sufficiency/reranker studies in the evidence archive rather than the main comparison.
