# Oracle stopping bounds for the existing embedding ranking

This offline analysis uses all 324 queries from the earlier retrieval-only
experiment, not the 101 completed queries from the paused sufficiency experiment.
It makes **zero new API calls**. The inference process remains paused.

## What each oracle is allowed to do

Every policy uses the same cached embedding-ranked top-20 list and must return
a prefix of at least one record. Nothing is reranked, skipped within a prefix,
or retrieved from outside that pool. Recall is macro qrels document recall,
with relevant documents outside the pool still included in the denominator.
The cost is the full generator input-token count previously validated against
2,335 API requests, excluding decision, output-generation and evaluation costs.

- **Query-specific top-k oracle:** knows the relevance labels and chooses a
  different k for each query to maximise mean recall under a mean token budget.
- **Query-specific similarity oracle:** knows the labels and chooses a different
  similarity threshold for each query, with the same top-1 fallback. Distinct
  sorted scores permit exactly the same prefixes as top-k. A tied score cannot
  be split by a threshold; the one adjacent score tie in these data does not
  change the optimal frontier. The two computed oracle curves coincide exactly.
- **Best shared top-k:** one k for all queries, or a query-independent random
  mixture of k settings. All k=1..20 are considered.
- **Best shared similarity cutoff:** one threshold for all queries, or a
  query-independent random mixture of thresholds. All 6,140 distinct selection
  states are considered, rather than the earlier coarse threshold grid.

The shared-rule curves are hindsight comparison envelopes, not settings tuned
on test for deployment. The per-query oracles have more information and more
flexibility than either shared rule. Labelling one query-specific rule and one
shared rule as interchangeable 'oracles' would obscure that distinction.

## Computation and checks

For query q and prefix k, let C(q,k) be its input tokens and R(q,k) its qrels
recall. The oracle maximises mean R subject to mean C <= B. For a precise
continuous bound, each query may randomise between prefix lengths.

We form each query's upper concave cost/recall hull and merge the hull segments
in descending marginal recall per token. This solves the separable fractional
allocation problem. Three budget points were independently checked using a
general linear-program solver; recall agreed within 1e-8.

Every vertex of the per-query curve is a feasible deterministic allocation.
Between adjacent vertices, interpolation randomises one query between two
prefixes. Taking the preceding deterministic vertex loses at most 1/324 macro
recall, or 0.309 percentage points. The continuous line is not asserted to be
the exact deterministic integer-budget optimum.

Additional checks verified sorted scores, increasing prefix token counts,
nondecreasing recall, legal threshold-induced selections and matching full-pool
endpoints. Threshold states were independently spot-checked by direct masking.

## Results

| Policy budget | Mean input tokens | Observed recall | Per-query oracle recall | Gap |
| --- | ---: | ---: | ---: | ---: |
| Fixed top-2 | 531.9 | 32.09% | 46.06% | 13.97 pp |
| Fixed top-3 | 772.9 | 37.72% | 52.61% | 14.89 pp |
| Fixed top-5 | 1,227.0 | 42.40% | 57.61% | 15.20 pp |
| Fixed top-10 | 2,391.2 | 48.46% | 57.75% | 9.29 pp |
| Existing absolute cutoff | 644.4 | 32.39% | 49.73% | 17.35 pp |
| Cached query-only LLM depth | 724.7 | 35.04% | 51.66% | 16.62 pp |

The top-20 pool's recall ceiling is **57.75%**, not 100%: a stopping policy
cannot recover relevant documents absent from its candidate list. Fixed top-20
uses 4,730.3 input tokens per query to reach this ceiling. The oracle reaches
the same recall at **1,262.75 tokens and 5.071 documents on average**, a **73.3%
token reduction**. This saturation point is deterministic, not dependent on
fractional randomisation.

This demonstrates substantial headroom for allocating different retrieval
depths across queries. It does not establish that an LLM or another deployable
controller can identify the right allocation cheaply. At the fixed top-5 token
budget, even the best shared similarity-cutoff envelope reaches only 38.12%
recall, versus 42.40% for fixed top-5. Better global threshold tuning alone does
not close the observed oracle gap at this budget.

These conclusions concern **document recall**, not whether the retrieved set
contains the minimum facts needed to answer. A sufficiency oracle would require
separate reliable sufficiency labels for candidate contexts; qrels cannot supply
that automatically. The ongoing sufficiency experiment remains paused and is
not silently combined with this complete 324-query analysis.

## Artifacts and reproduction

Local artifacts are in
`rag/results/full-adaptive-20261115T215241Z/oracle-frontier/`:

- `retrieval-oracle-frontier.png` / `.svg`: full token range.
- `retrieval-oracle-zoom.png` / `.svg`: lower-token region.
- `summary.json`, `comparisons.csv`, `curves.csv`: results and validation.
- `oracle-allocation.npz`: per-query saturation stopping depths.

```bash
python rag/extensions/retrieval_oracle_frontier.py \
  --run ~/ragexp/full-adaptive-20261115T215241Z \
  --out ~/ragexp/full-adaptive-20261115T215241Z/NEW_ORACLE_DIRECTORY
```

This is a CPU-only analysis on fuji3 and needs NumPy, SciPy and Matplotlib.
AI assistance was used for implementation, validation, plotting and these notes.
