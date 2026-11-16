# Rounded cutoff controls

The final nine rows compare fixed top-5, top-10 and top-20; the original dev-F1 cutoffs (absolute tau=0.8925 and relative delta=0.009); and four additional, round thresholds:

| Table name | Rule | Threshold |
| --- | --- | --- |
| Absolute 0.88 | cosine similarity >= tau | 0.88 |
| Absolute 0.87 | cosine similarity >= tau | 0.87 |
| Relative 0.02 | best cosine score minus candidate cosine score <= delta | 0.02 |
| Relative 0.03 | best cosine score minus candidate cosine score <= delta | 0.03 |

These four values were specified before any answers in this run. They are exploratory controls motivated by previously inspected evaluation curves, not dev-tuned parameters, exact recall matches or choices from answer scores. The earlier proposal to choose thresholds matching 40%/50% recall was stopped when the user requested rounded, parameter-based settings. Its partial answers are excluded.

All nine settings receive independent fresh generation for all 324 evaluation queries in one shuffled 12-worker batch, followed only then by policy-blind judging. Existing answers from earlier studies are not pooled into this batch. The fixed controls and dev-F1 thresholds are retained unchanged. All methods share cached ASR transcripts, embeddings and full 24,000-document rankings. No count cap or silent prompt truncation is permitted. Full prompt preflight passes: maximum 71,670 input tokens, plus 2,048 maximum output, within the advertised 262,144-token context window.

Generator: Mistral Small 4 (`mistralai/mistral-small-2603`), temperature 0.1, output cap 2,048. Judge: Gemini 2.5 Flash, temperature zero, cap 768; identical rubric and blinded payload to the preceding study. Reasoning requested disabled. All 2,916 answers must finish before judging starts. Preserve request IDs, usage, raw responses, retries, final judgements and reconciled billing. Budget guard: $10; projected generation plus judging approximately $4 at launch. A guard is not an account-level provider limit.

The graph uses macro precision on x and macro recall on y. Squares label absolute tau values; triangles label relative delta values. Diamonds preserve dev-selected references. Table names identify the actual parameter, not a desired recall level; report measured recall separately. Bold marks the best observed unrounded value per column, not statistical significance.

Latency measures selection/context construction plus generation service under concurrency, excluding cached upstream work and queue wait. Input tokens are complete provider-reported prompt counts, distinct from additive retrieved-text tokens on the frontier. Generation charges exclude judging; report total experiment cost separately. Paired query bootstrap intervals are exploratory, not adjusted for multiple comparisons or repeated-generation variability.

Run: `/home/zcm/ragexp/cutoff-threshold-controls-20261116`.
Runner: `rag/extensions/cutoff_threshold_controls.py`.
Preflight: `rag/extensions/preflight_threshold_controls.py`.
The earlier five-row study is archived under `cutoff-pipeline-20261116`; its protocol and results remain historical records.
