# Retrieval frontier extended to top-50

Completed an offline CPU sweep on the same 324 queries used in the earlier
retrieval-only analysis. Cached ASR transcripts and enc_a query embeddings were
reused, with exact cosine search over the same 24,000-record corpus. The maximum
candidate pool increases from 20 to 50. **No inference/API calls were made; the
separate evidence-sufficiency experiment remains paused.**

## Policies and measurement

Fixed-k points are **5,10,15,20,25,30,35,40,45,50**, with top-1 retained only as
the shared fallback anchor. Lines connect these sampled settings, not every
possible fixed k. Internal prefix arrays include all lengths because cutoff
policies and oracle optimisation can select any prefix length.

- Absolute cutoff: retain candidates with cosine similarity >= tau.
- Relative cutoff: retain candidates with top similarity minus candidate
  similarity <= delta.
- Both policies use the top-50 pool and return top-1 if nothing qualifies.
- Every distinct threshold selection state is evaluated: **15,720 absolute**
  and **15,765 relative** settings, including fallback anchors.

X is mean full generator input tokens, using the same prompt template, pinned
Mistral tokenizer and 17-token chat overhead as before. Counts above k=20 are
offline estimates using that previously API-validated format; no new API-based
token validation was performed. X excludes selector, output-generation and
evaluation costs. Y is macro **qrels document recall**, not answer sufficiency.

The two oracle curves allow query-specific k or query-specific similarity
thresholds, using relevance labels in hindsight. They retain ranking prefixes
and may randomise at a mean-budget boundary. Both curves coincide in these
data; the two adjacent score ties do not affect the optimum. All global cutoff
sweeps are descriptive test-set analyses, not newly validated operating points.

## Fixed-k results

| k | Mean input tokens | Document recall |
| ---: | ---: | ---: |
| 5 | 1,227.0 | 42.40% |
| 10 | 2,391.2 | 48.46% |
| 15 | 3,571.4 | 54.06% |
| 20 | 4,730.3 | 57.75% |
| 25 | 5,912.2 | 60.68% |
| 30 | 7,073.5 | 62.36% |
| 35 | 8,223.6 | 64.05% |
| 40 | 9,370.5 | 65.83% |
| 45 | 10,586.1 | 67.81% |
| 50 | 11,715.8 | 68.42% |

Going from top-20 to top-50 adds **10.67 percentage points** of recall at
**2.48 times** the input-token cost. The last step, top-45 to top-50, adds only
0.61 percentage points in this sample. These are descriptive trade-offs, not
claims about generated-answer benefit.

At the tested fixed-k budgets, the best shared absolute-cutoff envelope remains
below fixed-k except the shared endpoints. Relative cutoffs approach fixed-k at
larger budgets. At the top-40 token budget, the relative-cutoff random-mixture
envelope reaches 66.33% versus fixed top-40's 65.83%, a small **0.50 pp**
descriptive advantage. No uncertainty interval or independent confirmation is
claimed for this test-inspected comparison. It does not establish a general
adaptive-policy win. The plotted cutoff lines show actual threshold states;
the envelope comparisons in summary.json additionally permit random mixtures.

## Expanded oracle bound

The candidate-pool recall ceiling rises to **68.42%**. The query-specific oracle
reaches it with **2,825.6 tokens and 11.769 documents per query on average**,
versus 11,715.8 tokens for fixed top-50: **75.9% fewer input tokens**. This
saturation point is a deterministic per-query allocation. Intermediate oracle
line segments may require budget-boundary randomisation.

At the fixed top-5 token budget the expanded oracle reaches 59.48% recall;
at the top-10 budget it reaches 67.11%. These bounds assume perfect knowledge
of relevance and free selection. The remaining 31.58% macro recall deficit is
outside the top-50 candidate pool and cannot be recovered by a stopping rule
restricted to this pool.

## Previously frozen policies and validation

The earlier dev-selected cutoffs are retained, without retuning:

| Policy | Parameter | Mean docs | Mean input tokens | Recall |
| --- | ---: | ---: | ---: | ---: |
| Absolute | tau=0.8925 | 2.401 | 656.6 | 32.39% |
| Relative | delta=0.009 | 2.552 | 666.5 | 30.60% |

The absolute policy now selects more than 20 documents for two queries but
recovers no additional qrels hits. The relative policy selects at most 20 for
every query at its existing setting. These diamonds are historical operating
points, not settings optimised for the expanded pool.

Validation confirmed unchanged cached top-20 IDs, scores, recall and precision,
with no tie-order restoration needed. Re-tokenising all 324 top-20 prompts
gave zero token-count differences. Threshold states were independently checked
by direct score masking. Three expanded oracle budget points agreed with an
independent general linear-program solver to within 1e-8 recall.

## Files and reproduction

Local and remote artifacts are under
`rag/results/full-adaptive-20261115T215241Z/retrieval-frontier-top50/` locally
and the corresponding run directory on fuji3:

- `retrieval-frontier-top50.png` / `.svg`: full range.
- `retrieval-frontier-top50-zoom.png` / `.svg`: up to 5,100 input tokens.
- `sweep.csv`, `summary.json`, `oracle-curves.csv`: complete numerical results.
- `queries-top50.jsonl`, `query-prefix-metrics.npz`, `oracle-allocation.npz`:
  per-query rankings, prefix costs/metrics and oracle allocations.

```bash
python rag/extensions/retrieval_frontier_top50.py \
  --run ~/ragexp/full-adaptive-20261115T215241Z \
  --out ~/ragexp/full-adaptive-20261115T215241Z/NEW_TOP50_DIRECTORY
```

The runner uses the oracle helper in `retrieval_oracle_frontier.py` and does not
modify supplied coursework interfaces. AI assistance was used for experimental
implementation, validation, plotting and these notes.
