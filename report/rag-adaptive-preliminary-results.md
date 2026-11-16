# Preliminary RAG experiments: predicted depth and distance filtering

These are exploratory results, not final coursework claims. Both requested
ideas were implemented and run on fuji3 on 2026-11-15. The total OpenRouter
charge was **$0.095398512**, reconciled from generation billing records.

## Design and controls

The main experiment used eight calibration queries and twelve disjoint
evaluation queries sampled from practice with seed 20261116. Dev was untouched.
The sampling lists, prompts and settings are in `manifest.json` in the evidence
directory. There were five policies and sixty complete answers/evaluations.

All arms used Mistral Small 4 (`mistralai/mistral-small-2603`) for generation and
Gemini 2.5 Flash (`google/gemini-2.5-flash`) as judge. The predicted-depth arm
used the same Mistral model to select k from 1 to 20 using only the transcript,
before retrieval. It had no corpus, score or qrels access. Invalid predictions
would fall back to 5; none did. Generation temperature was 0.1; planner and
judge temperature were 0. Reasoning was requested disabled consistently.
The main output limits were 1024 tokens for answers, 64 for planning and 768 for
judging. Five answer truncations received a separate sensitivity check below.

The thresholds were set without qrels or answer scores: the absolute similarity
threshold was the median rank-5 score on the eight calibration queries, and
the relative threshold was the median first-to-fifth score difference. The
frozen values were similarity >= **0.876169**, or top similarity minus
**0.018487**, respectively. Both policies retrieve a top-20 candidate set,
retain the qualifying prefix and fall back to the best hit if none qualifies.
Cosine distance is 1 minus similarity for these normalised embeddings; the
absolute distance radius was therefore approximately 0.123831.

Each policy's end-to-end time was measured from the input audio through custom
CUDA ASR, CPU query encoding, optional planning, retrieval/filtering, context
building and answer generation. Model loading and judging were excluded. ASR
was rerun for each policy, and paired transcripts were required to match.
Policy order was randomised per query. One A6000 was exposed by GPU UUID; other
users' processes were left untouched. Threshold calibration preceded evaluation.

## Main results (12 paired queries)

The answer score is the supplied judge's normalised rubric score, not accuracy.
Inference costs include generation and the planner where applicable, and exclude
the experimental judge. Dollar values in the table are per 1,000 queries.

| Policy | Mean documents | Mean recall | Mean answer score | Mean warm E2E | Inference USD/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5 | 5.00 | 0.396 | 0.542 | 5.37 s | $0.328 |
| Fixed top-20 | 20.00 | 0.528 | 0.549 | 6.32 s | $0.965 |
| Model-predicted depth | 3.17 | 0.271 | 0.458 | 5.64 s | $0.256 |
| Top-20 + absolute distance cutoff | 4.58 | 0.271 | 0.493 | 5.21 s | $0.304 |
| Top-20 + relative score-gap cutoff | 12.50 | 0.500 | 0.542 | 6.21 s | $0.822 |

The main experiment cost $0.072761671 across 132 API completions: sixty answers,
sixty judgements and twelve planning calls. Each has billing metadata.

### Model-predicted depth

The chosen depths were `3,1,3,3,3,3,5,5,3,3,4,2`. The planner averaged 0.602 s
and cost $0.000260175 in total, about $0.0217 per 1,000 queries. Shorter contexts
saved more generation cost than the planner added: net inference cost was
approximately 22% lower than top-5. Mean E2E nevertheless increased by 0.274 s
(about 5%) because the planning call outweighed the mean generation-time saving.

The answer-score difference against top-5 was -0.083. Its paired bootstrap
95% interval was [-0.292, 0.083]; the E2E-difference interval was
[-0.164 s, 0.712 s]. These do not establish a reliable improvement or slowdown.
There was one higher score, nine ties and two lower scores relative to top-5.
Recall fell by 0.125 on average.

A plausible limitation is that question complexity alone does not reveal where
relevant documents occur in this particular corpus. A score-aware or
evidence-aware policy would test a different hypothesis; it is not part of this
query-only experiment.

### Distance filtering

The absolute cutoff selected
`7,1,1,1,11,1,7,13,1,9,1,2` documents. Four of twelve queries needed the explicit
top-1 fallback. This shows poor transfer of a single global cutoff across query
score distributions on this sample. It had lower recall and lower average judge
score than top-5, with only a small, uncertain latency reduction. Despite fewer
documents on average, its mean prompt was slightly longer than top-5 (5,127 vs
4,905 characters): document count is not a token-cost budget.

The relative cutoff selected
`5,1,20,15,2,17,20,12,11,20,17,10` documents. Compared with top-20 it used 37.5%
fewer documents on average and about 15% less inference spend, while mean recall
was 0.500 vs 0.528 and the primary answer score was 0.542 vs 0.549. E2E was
almost unchanged. This is a candidate trade-off, not demonstrated superiority.
Against top-5, its score-difference bootstrap interval was [-0.125, 0.125].

## Follow-up controls and limitations

A small follow-up used the first four primary evaluation queries (not selected
by their scores) and fresh, randomly ordered runs of fixed k=2, k=3, k=5 and
predicted depth. It cost $0.012089229 across 36 completions.

| Policy | Mean documents | Mean score | Mean warm E2E | Inference USD/1,000 |
| --- | ---: | ---: | ---: | ---: |
| Fixed top-2 | 2.0 | 0.625 | 3.56 s | $0.140 |
| Fixed top-3 | 3.0 | 0.479 | 3.76 s | $0.187 |
| Fixed top-5 | 5.0 | 0.729 | 4.01 s | $0.290 |
| Model-predicted depth | 2.5 | 0.500 | 4.36 s | $0.192 |

This does not establish a benefit over a cheap fixed small-k policy. Four queries
are insufficient for robust ranking; this is a post-primary diagnostic, not a
second independent test set.

Five primary answers hit the 1024-token output cap: four relative-gap answers
and one top-20 answer. All five were regenerated with a 2048-token cap and
judged again, costing $0.010547612. None remained length-limited. One relative-gap
answer changed from 0 to 1; the top-20 answer changed from 0.417 to 0.250; three
others stayed at 0. Replacing only these rows would give relative-gap mean 0.625
and top-20 mean 0.535. This is **post-hoc sensitivity**, confounded by regeneration
and judge variability; it is not a corrected primary comparison. Keep the
original results. A final study should use an adequate uniform cap and inspect
termination reasons for every arm.

Manual inspection found a useful ambiguity example: "stock delivery" retrieved
retail shipping material, causing some short-context answers to discuss goods
rather than financial shares. Larger contexts sometimes enabled a qualified
answer or abstention. Another case retrieved all qrels-relevant evidence but
added financial claims from extra documents and received a poor grounding score.
These are examples for diagnosis, not authoritative financial advice.

Some answers scored 1 despite recall 0 because the judge rewarded an appropriate
statement of insufficient evidence. Thus the combined judge score cannot by
itself establish successful question answering. A final study should audit
representative answers, distinguish abstention from supported answers, and
check repeated-judge stability. Retrieval qrels may not enumerate every useful
document; neither a zero qrels recall nor an automated score is definitive alone.

All bootstrap intervals are exploratory and account only for sampled query
variation, not judge uncertainty, model randomness or multiple comparisons.
There is no statistically supported quality win here. Threshold choice,
sample size, query ambiguity and response length all merit further testing.

## Matching the supplied ELM models

| ELM name | Corresponding OpenRouter ID / status |
| --- | --- |
| EuroLLM | Course default `utter-project/EuroLLM-22B-Instruct-2512`; no EuroLLM entry was listed in the OpenRouter catalogue checked for this experiment |
| Llama 3.3 | `meta-llama/llama-3.3-70b-instruct` for the 70B Instruct variant; verify ELM's exact deployment ID for an exact checkpoint claim |
| Mistral Small 4 | `mistralai/mistral-small-2603`, used for all primary/control generation and planning |
| Qwen 3.5 397B | `qwen/qwen3.5-397b-a17b`; earlier Qwen3-235B experiments do not match it |

The key control is matching the generator, prompt and decoding settings between
fixed-k baseline and policy variants. This pilot does not require running all
four generator families. The planner intentionally uses the generator model.
The independent judge stays fixed across policies and need not be an ELM-listed
model. Matching the model name is not proof of matching checkpoint revision,
quantisation or serving configuration. OpenRouter latency and API charges are
not ELM latency or its free-to-student price.

Primary model catalogue pages:
[Mistral Small 4](https://openrouter.ai/mistralai/mistral-small-2603),
[Llama 3.3 70B Instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct),
[Qwen 3.5 397B A17B](https://openrouter.ai/qwen/qwen3.5-397b-a17b).

## Reproduction and evidence

The main runner is
[`adaptive_retrieval_pilot.py`](../rag/extensions/adaptive_retrieval_pilot.py);
see the [extension guide](../rag/extensions/README.md) for its design and command.
Existing RAG adapter tests passed before execution. Source interfaces and the
private API configuration were not modified.

Evidence is copied locally under
`rag/results/adaptive-preliminary-20261115T213017Z/` (ignored generated files) and
retained at `~/ragexp/adaptive-preliminary-20261115T213017Z/` on fuji3. It includes
the primary run, `small-k-control/`, `length-sensitivity/`, all request/billing
records, prompts/settings, candidates and scores, and a runner source copy.

To reproduce the extra small-k control after a primary run:

```bash
python rag/extensions/adaptive_retrieval_pilot.py \
  --manifest-input /path/to/primary/manifest.json \
  --out /path/to/new-control-directory --queries 4 \
  --arms fixed2 fixed3 fixed5 llm_depth --budget 0.5
```

AI assistance was used for experimental implementation, execution, analysis and
these notes. Include this work in the final report's AI-use declaration.
